// BSD 3-Clause License
//
// Copyright (c) 2025, Jianhui Dai
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu_cpp_print.h>
#include <webgpu/webgpu_cpp.h>

#include <iostream>
#include <algorithm>

#include "base_compute_runner.h"
#include "wgpu_context.h"

// WGSL shader code for a simple compute operation: output = 2 * input + 1
static const std::string kCode = R"(
enable f16;
enable subgroups;
const workgroup_size_x: u32 = 64;
const workgroup_size_y: u32 = 1;
const workgroup_size_z: u32 = 1;
@group(0) @binding(0) var<storage, read> q: array<vec4<f16>>; // shape: [seq_length, num_head, head_size / vec_length]
@group(0) @binding(1) var<storage, read> present_key: array<vec4<f16>>; // shape: [num_kv_head, seq_length, head_size / vec_length]
@group(0) @binding(2) var<storage, read> present_value: array<vec4<f16>>;  // shape: [num_kv_head, seq_length, head_size / vec_length]
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f16>>;
struct Uniforms {
  new_sequence_length: u32, // 1369. prompt length
  total_sequence_length: u32, // 1369. ditto
  present_sequence_length: u32, // 2048
  past_sequence_length: u32, // 0
  is_gqa: u32, // 1
  n_reps: u32, // 3
  alpha: f32, // 0.088388
  num_seq_tile: u32 // 22
};
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

alias q_value_t = vec4<f16>;
alias q_element_t = f16;

const qkv_head_size: u32 = 128;
const num_heads: u32 =24;

  // For max performance max_k_step should be the same as sg_size, however we might run out of registers
  // for qk_1, qk_2 .. qk_(sg_size). So we cap it at max_k_step (16).
  const max_k_step: u32 = 16u;
  const vec_factor: u32 = 4u;
  const qkv_head_size_vec: u32 = qkv_head_size / vec_factor; // 32
  const min_value : q_element_t = q_element_t(-65504.0);

  // Default SHM usage limit is 16KB in Dawn.
  var<workgroup> k_tile : array<array<q_value_t, qkv_head_size_vec>, max_k_step>; // 8 * 32 * 16 = 4KB. // shape[max_k_step, head_size / vec_length] (16, 32)
  var<workgroup> v_tile : array<array<q_value_t, qkv_head_size_vec>, max_k_step>; // 8 * 32 * 16 = 4KB. // shape[max_k_step, head_size / vec_length] (16, 32)

  // Private memory per lane.
  var<private> q_tile : array<q_value_t, qkv_head_size_vec>; // 8 * 32 = 256B register
  var<private> o_tile : array<q_value_t, qkv_head_size_vec>; // 8 * 32 = 256B register
  fn loadq(q_idx_global : u32, head_idx: u32) // NOTE: each thread load one head of one token into registers!
  {
      // Stored as float16[batch_size,sequence_length,3072] the inputs as per onnx MHA
      // This is the layout if TransferBSDToBNSH has not been run.
      let offset = q_idx_global * (qkv_head_size_vec) * num_heads + qkv_head_size_vec * head_idx;
      // Stored as BNSH - which is what webgpu uses after TransferBSDToBNSH has been run.
      //let offset = head_idx * uniforms.new_sequence_length * qkv_head_size_vec + q_idx_global * qkv_head_size_vec;
      for (var idx:u32 = 0; idx < qkv_head_size_vec; idx++)
      {
          q_tile[idx] = q[idx+offset];
      }
  }
  fn loadk(k_start : u32, head_idx: u32, local_idx: u32, k_step: u32) // NOTE: each block will load one head of a k_step(16) of tokens into SRAM., so each thread in the block just load a head of them, because k_steps equals to warp_size.
  {
      // Stored as float16[batch_size,num_heads,present_sequence_length,96]
      let offset = head_idx * uniforms.present_sequence_length * qkv_head_size_vec + k_start * qkv_head_size_vec;
      for (var idx:u32 = local_idx; idx < qkv_head_size_vec*k_step; idx+=workgroup_size_x)
      {
          let slot = u32(idx/qkv_head_size_vec); // range(0, 16), the inner token id of the block
          let val = select(q_value_t(0), present_key[offset+idx], k_start + slot < uniforms.total_sequence_length);
          k_tile[slot][idx%qkv_head_size_vec] = val;
      }
  }
  fn loadv(v_start : u32, head_idx: u32, local_idx: u32, k_step: u32) // ditto as loadk()
  {
      // Stored as float16[batch_size,num_heads,present_sequence_length,96]
      let offset = head_idx * uniforms.present_sequence_length * qkv_head_size_vec + v_start * qkv_head_size_vec;
      for (var idx:u32 = local_idx; idx < qkv_head_size_vec*k_step; idx+=workgroup_size_x)
      {
          let slot = u32(idx/qkv_head_size_vec);
          let val  = select(q_value_t(0), present_value[offset+idx], v_start + slot < uniforms.total_sequence_length);
          v_tile[slot][idx%qkv_head_size_vec] = val;
      }
  }
  fn writeo(o_idx_global: u32, head_idx: u32)
  {
      // Stored as float16[batch_size,sequence_length,3072]
      let offset = o_idx_global * num_heads * qkv_head_size_vec + head_idx * qkv_head_size_vec;
      for (var idx:u32 = 0; idx < qkv_head_size_vec; idx ++)
      {
          output[offset+idx] = o_tile[idx];
      }
  }

      fn loadAttentionBias(q_idx_global : u32, k_idx_global : u32, head_idx: u32) -> vec4<q_element_t>
      {
        return vec4<q_element_t>(0);
      }
    @compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>,
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_index) local_idx : u32,
        @builtin(local_invocation_id) local_id : vec3<u32>,
        @builtin(subgroup_invocation_id) sg_id : u32,
        @builtin(subgroup_size) sg_size : u32) {
  let global_idx = global_id.x;
  let workgroup_idx = workgroup_id.x;

  let head_idx = u32(workgroup_idx / uniforms.num_seq_tile);
  let capped_sg_id = min(sg_id, max_k_step); // TODO(Weqnin): why the min work? I guess it may bring regresiion if the sg_id will be greater than 16, like on NVIDIA GPU. it semms Intel GPU just use 16 threads in a warp.
  let capped_sg_size = min(sg_size, max_k_step); // 16

  // Load Q
  let q_idx_global = (workgroup_idx % uniforms.num_seq_tile) * workgroup_size_x + local_idx;
  let valid_q = q_idx_global < uniforms.new_sequence_length;
  if (valid_q)
  {
    loadq(q_idx_global, head_idx);
  }

  var previous_max : q_element_t = min_value;
  var previous_denom : q_element_t = 0;

  for(var k_start = 0u; k_start < uniforms.total_sequence_length; k_start+=capped_sg_size)
  {
    workgroupBarrier();
    loadk(k_start, head_idx / uniforms.n_reps, local_idx, capped_sg_size);
    loadv(k_start, head_idx / uniforms.n_reps, local_idx, capped_sg_size);
    workgroupBarrier();

    // Compute QKt
    var qk_1:vec4<q_element_t>;
    var qk_2:vec4<q_element_t>;
    var qk_3:vec4<q_element_t>;
    var qk_4:vec4<q_element_t>;
    if (sg_size > 8)
    {
      for (var i:u32 = 0u; i < qkv_head_size_vec; i++)
      {
        // TODO(wenqin): ditto as above, it's hard code for a warp with 16 threads.
        // TODO(Wenqin): try not use shuffle but local register to see whether perf gain on Intel paltform.
        var k_local = k_tile[capped_sg_id][i];
        var q_own = q_tile[i];
        qk_1[0] += dot(q_own, subgroupShuffle(k_local, 0));
        qk_1[1] += dot(q_own, subgroupShuffle(k_local, 1));
        qk_1[2] += dot(q_own, subgroupShuffle(k_local, 2));
        qk_1[3] += dot(q_own, subgroupShuffle(k_local, 3));
        qk_2[0] += dot(q_own, subgroupShuffle(k_local, 4));
        qk_2[1] += dot(q_own, subgroupShuffle(k_local, 5));
        qk_2[2] += dot(q_own, subgroupShuffle(k_local, 6));
        qk_2[3] += dot(q_own, subgroupShuffle(k_local, 7));
        qk_3[0] += dot(q_own, subgroupShuffle(k_local, 8));
        qk_3[1] += dot(q_own, subgroupShuffle(k_local, 9));
        qk_3[2] += dot(q_own, subgroupShuffle(k_local, 10));
        qk_3[3] += dot(q_own, subgroupShuffle(k_local, 11));
        qk_4[0] += dot(q_own, subgroupShuffle(k_local, 12));
        qk_4[1] += dot(q_own, subgroupShuffle(k_local, 13));
        qk_4[2] += dot(q_own, subgroupShuffle(k_local, 14));
        qk_4[3] += dot(q_own, subgroupShuffle(k_local, 15));
      }
    }
    else
    {
      for (var i:u32 = 0u; i < qkv_head_size_vec; i++)
      {
        var k_local = k_tile[capped_sg_id][i];
        var q_own = q_tile[i];
        qk_1[0] += dot(q_own, subgroupShuffle(k_local, 0));
        qk_1[1] += dot(q_own, subgroupShuffle(k_local, 1));
        qk_1[2] += dot(q_own, subgroupShuffle(k_local, 2));
        qk_1[3] += dot(q_own, subgroupShuffle(k_local, 3));
        qk_2[0] += dot(q_own, subgroupShuffle(k_local, 4));
        qk_2[1] += dot(q_own, subgroupShuffle(k_local, 5));
        qk_2[2] += dot(q_own, subgroupShuffle(k_local, 6));
        qk_2[3] += dot(q_own, subgroupShuffle(k_local, 7));
      }
    }

    qk_1 = qk_1 * q_element_t(uniforms.alpha) + loadAttentionBias(q_idx_global, k_start, head_idx);
    qk_2 = qk_2 * q_element_t(uniforms.alpha) + loadAttentionBias(q_idx_global, k_start+4, head_idx);
    if (sg_size > 8)
    {
      qk_3 = qk_3 * q_element_t(uniforms.alpha) + loadAttentionBias(q_idx_global, k_start+8, head_idx);
      qk_4 = qk_4 * q_element_t(uniforms.alpha) + loadAttentionBias(q_idx_global, k_start+12, head_idx);
    }

    let seq_causal_length = select(uniforms.total_sequence_length, uniforms.past_sequence_length + q_idx_global + 1, uniforms.is_gqa > 0); // q_idx_global
    // Neuter qk values where K is out of bounds.
    qk_1[0] = select(min_value, qk_1[0], k_start+0 < seq_causal_length);
    qk_1[1] = select(min_value, qk_1[1], k_start+1 < seq_causal_length);
    qk_1[2] = select(min_value, qk_1[2], k_start+2 < seq_causal_length);
    qk_1[3] = select(min_value, qk_1[3], k_start+3 < seq_causal_length);
    qk_2[0] = select(min_value, qk_2[0], k_start+4 < seq_causal_length);
    qk_2[1] = select(min_value, qk_2[1], k_start+5 < seq_causal_length);
    qk_2[2] = select(min_value, qk_2[2], k_start+6 < seq_causal_length);
    qk_2[3] = select(min_value, qk_2[3], k_start+7 < seq_causal_length);
    if (sg_size > 8)
    {
      qk_3[0] = select(min_value, qk_3[0], k_start+8 < seq_causal_length);
      qk_3[1] = select(min_value, qk_3[1], k_start+9 < seq_causal_length);
      qk_3[2] = select(min_value, qk_3[2], k_start+10 < seq_causal_length);
      qk_3[3] = select(min_value, qk_3[3], k_start+11 < seq_causal_length);
      qk_4[0] = select(min_value, qk_4[0], k_start+12 < seq_causal_length);
      qk_4[1] = select(min_value, qk_4[1], k_start+13 < seq_causal_length);
      qk_4[2] = select(min_value, qk_4[2], k_start+14 < seq_causal_length);
      qk_4[3] = select(min_value, qk_4[3], k_start+15 < seq_causal_length);
    }

    //
    // Compute SoftMax as per Flash Attention technique.
    //
    // Crux of Flash Attention is here, that allows for partial softmax computation,
    // direct update of output and merging with previous results.
    // https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
    // Where b is the block size of the tile. Xi is storing QKtranspose for the ith tile.
    // mi_local is the max of Xi. Note: _ in this notation means what follows is a
    // subscript. max_j=1:b (Xi[j]) is the max of Xi[j] for j=1 to b.
    //
    // for i = 1, #tiles do
    //  Xi = Q[k,:] Kt[:, (i-1) b : i b]
    //  mi_local= max_j=1:b (Xi[j])
    //  Mi = max(M_(i-1), mi_local)
    //  d'_i = d'_(i-1) * e^(M_(i-1)-M_i) + Î£_j=1:b e^(Xi[j]-Mi)
    //  o'_i = o'_(i-1) * d'_(i-1) * e^(M_(i-1)-M_i) / d'_i + Î£_j=1:b (e^(Xi[j]-Mi) / d'_i) V[j + (i - 1)b,:]
    // end
    //
    // In the code below:
    // dleft is the first term of d'_i expression above : d'_(i-1) * e^(M_(i-1)-M_i).
    // sum is the second term of the same expression    : Î£_j=1:b e^(Xi[j]-Mi)
    // o_ratio is the part of the first term of o'_i expression above : d'_(i-1) * e^(M_(i-1)-M_i) / d'_i
    //
    var local_max_temp = max(qk_1, qk_2);
    if (sg_size > 8)
    {
      local_max_temp = max(local_max_temp, qk_3);
      local_max_temp = max(local_max_temp, qk_4);
    }
    let local_max = max(max(local_max_temp.x, local_max_temp.y),max(local_max_temp.z, local_max_temp.w));
    let new_max = max(previous_max, local_max);
    qk_1 = q_value_t(exp(vec4<f32>(qk_1) - f32(new_max))); // get exp of QKt
    qk_2 = q_value_t(exp(vec4<f32>(qk_2) - f32(new_max)));
    if (sg_size > 8) {
      qk_3 = q_value_t(exp(vec4<f32>(qk_3) - f32(new_max)));
      qk_4 = q_value_t(exp(vec4<f32>(qk_4) - f32(new_max)));
    }
    let sum_vec = qk_1 + qk_2 + qk_3 + qk_4;
    let sum = sum_vec.x + sum_vec.y + sum_vec.z + sum_vec.w;

    // Compute lhs term of update di prime and the compute di prime.
    let dleft = previous_denom * exp(previous_max-new_max); // update the previous row sum.
    var d = dleft + sum; // the new row sum.
    d = select(d,q_element_t(0.0000001),d==0);
    qk_1 = qk_1 / d; // get the soft max (attention score) of QKt
    qk_2 = qk_2 / d;
    if (sg_size > 8) {
      qk_3 = qk_3 / d;
      qk_4 = qk_4 / d;
    }
    previous_max = new_max; // maintain the row max
    previous_denom = d; // maintain the row sum
    let o_ratio = dleft / d;

    if (sg_size > 8) {
      for (var i:u32 = 0; i < qkv_head_size_vec; i++)
      {
          var val = v_tile[capped_sg_id][i];
          // qk_1[0] is a scaler, it was used for scale the attention weight for a vec (4 scalers) of value.
          var sum = subgroupShuffle(val, 0) * qk_1[0];
          sum += subgroupShuffle(val, 1) * qk_1[1];
          sum += subgroupShuffle(val, 2) * qk_1[2];
          sum += subgroupShuffle(val, 3) * qk_1[3];
          sum += subgroupShuffle(val, 4) * qk_2[0];
          sum += subgroupShuffle(val, 5) * qk_2[1];
          sum += subgroupShuffle(val, 6) * qk_2[2];
          sum += subgroupShuffle(val, 7) * qk_2[3];
          sum += subgroupShuffle(val, 8) * qk_3[0];
          sum += subgroupShuffle(val, 9) * qk_3[1];
          sum += subgroupShuffle(val, 10) * qk_3[2];
          sum += subgroupShuffle(val, 11) * qk_3[3];
          sum += subgroupShuffle(val, 12) * qk_4[0];
          sum += subgroupShuffle(val, 13) * qk_4[1];
          sum += subgroupShuffle(val, 14) * qk_4[2];
          sum += subgroupShuffle(val, 15) * qk_4[3];
          o_tile[i] = o_tile[i] * o_ratio + sum;
      }
    }
    else
    {
      for (var i:u32 = 0; i < qkv_head_size_vec; i++)
      {
          var val = select(vec4<q_element_t>(0), v_tile[capped_sg_id][i], k_start + capped_sg_id < seq_causal_length);
          var sum = subgroupShuffle(val, 0) * qk_1[0];
          sum += subgroupShuffle(val, 1) * qk_1[1];
          sum += subgroupShuffle(val, 2) * qk_1[2];
          sum += subgroupShuffle(val, 3) * qk_1[3];
          sum += subgroupShuffle(val, 4) * qk_2[0];
          sum += subgroupShuffle(val, 5) * qk_2[1];
          sum += subgroupShuffle(val, 6) * qk_2[2];
          sum += subgroupShuffle(val, 7) * qk_2[3];
          o_tile[i] = o_tile[i] * o_ratio + sum;
      }
    }
  }

  if (valid_q) {
    writeo(q_idx_global, head_idx);
  }

}
)";

// Convert float32 to float16
uint16_t float32ToFloat16(float value) {
    uint32_t f32 = *reinterpret_cast<uint32_t*>(&value);
    uint32_t sign = (f32 >> 31) & 0x1;
    uint32_t exponent = (f32 >> 23) & 0xFF;
    uint32_t fraction = f32 & 0x7FFFFF;

    if (exponent == 0xFF) { // Handle NaN or Infinity
        return (sign << 15) | 0x7C00 | (fraction ? 0x200 : 0);
    }

    int16_t halfExponent = static_cast<int16_t>(exponent - 127 + 15);
    if (halfExponent <= 0) { // Subnormal or zero
        return static_cast<uint16_t>(sign << 15);
    } else if (halfExponent >= 0x1F) { // Overflow to infinity
        return static_cast<uint16_t>((sign << 15) | 0x7C00);
    }

    uint16_t halfFraction = static_cast<uint16_t>(fraction >> 13);
    return static_cast<uint16_t>((sign << 15) | (halfExponent << 10) | halfFraction);
}

struct Uniforms {
  uint32_t new_sequence_length; // 1369. prompt length
  uint32_t total_sequence_length; // 1369. ditto
  uint32_t present_sequence_length; // 2048
  uint32_t past_sequence_length; // 0
  uint32_t is_gqa; // 1
  uint32_t n_reps; // 3
  float_t alpha; // 0.088388
  uint32_t num_seq_tile; // 22
};

int main() {
  constexpr uint32_t seq_length = 4096;
  constexpr uint32_t num_head = 24;
  constexpr uint32_t num_kv_head = 8;
  constexpr uint32_t num_q_per_kv = num_head / num_kv_head;
  constexpr uint32_t head_size = 128;
  constexpr uint32_t vec_size = 4;
  constexpr uint32_t head_vec_size = head_size / vec_size;
  constexpr uint32_t q_scalar_cnt = seq_length * num_head * head_size;
  constexpr uint32_t kv_scalar_cnt = seq_length * num_kv_head * head_size;
  constexpr uint32_t q_seq_tile_size = 64;
  constexpr uint32_t num_q_seq_tile = (seq_length + q_seq_tile_size - 1) / q_seq_tile_size;

  uint32_t grid = num_head * num_q_seq_tile;


  // Initialize Dawn's function pointers.
  dawnProcSetProcs(&dawn::native::GetProcs());

  // Create WGPUContext.
  std::unique_ptr<WGPUContext> wgpu_context = std::make_unique<WGPUContext>();
  wgpu_context->initialize();

  // Create BaseComputeRunner, passing the WGPUContext.
  std::unique_ptr<BaseComputeRunner> compute_runner =
      std::make_unique<BaseComputeRunner>(wgpu_context.get());
  compute_runner->initialize();

  // Create input/output buffer on the GPU.
  // Buffer creation sequence must align with shader binding indices.
  wgpu::BufferUsage usage = wgpu::BufferUsage::Storage |
                            wgpu::BufferUsage::CopySrc |
                            wgpu::BufferUsage::CopyDst;
  wgpu::BufferUsage uni_usage = wgpu::BufferUsage::Uniform |
                                wgpu::BufferUsage::CopySrc |
                                wgpu::BufferUsage::CopyDst;
  wgpu::Buffer query_buffer =
      compute_runner->add_buffer(q_scalar_cnt * sizeof(Float16), usage,
                                 wgpu::BufferBindingType::ReadOnlyStorage);
  wgpu::Buffer key_buffer =
      compute_runner->add_buffer(kv_scalar_cnt * sizeof(Float16), usage,
                                 wgpu::BufferBindingType::ReadOnlyStorage);
  wgpu::Buffer value_buffer =
      compute_runner->add_buffer(kv_scalar_cnt * sizeof(Float16), usage,
                                 wgpu::BufferBindingType::ReadOnlyStorage);
  wgpu::Buffer output_buffer = compute_runner->add_buffer(
      q_scalar_cnt * sizeof(Float16), usage, wgpu::BufferBindingType::Storage);

  wgpu::Buffer uniforms_buffer = compute_runner->add_buffer(
      sizeof(struct Uniforms), uni_usage, wgpu::BufferBindingType::Uniform);

  struct Uniforms uni({seq_length, seq_length, 2048, 0, 1, num_q_per_kv, 0.088388, num_q_seq_tile});
  

  // Set the shader code and the entry point function.
  compute_runner->set_shader(kCode, "main");
  // Set the dispatch size for the compute shader.
  compute_runner->set_dispatch(grid, 1, 1);
  // Initialize the compute pipeline.
  compute_runner->initialize_pipeline();

  // Create host-side data for the input buffer.
  std::vector<Float16> query_data(q_scalar_cnt);
  for (size_t i = 0; i < query_data.size(); ++i) {
    query_data[i] = float32ToFloat16(i);
  }
  std::vector<Float16> key_data(kv_scalar_cnt);
  std::vector<Float16> value_data(kv_scalar_cnt);
  for (size_t i = 0; i < key_data.size(); ++i) {
    key_data[i] = float32ToFloat16(i);
    value_data[i] = float32ToFloat16(i);
  }
  // Write the input data to the input buffer on the GPU.
  compute_runner->write_buffer(query_buffer, query_data.data(),
                               query_data.size() * sizeof(query_data[0]));
  compute_runner->write_buffer(key_buffer, key_data.data(),
                               key_data.size() * sizeof(key_data[0]));
  compute_runner->write_buffer(value_buffer, value_data.data(),
                               value_data.size() * sizeof(value_data[0]));
  compute_runner->write_buffer(uniforms_buffer, &uni, sizeof(struct Uniforms));

  // Run the compute shader.
  // auto times_ns = compute_runner->run();

  // Print the contents of the input buffer.
  // std::cout << "======\n";
  // std::cout << "Input Buffer:" << std::endl;
  // compute_runner->log_buffer<float>(query_buffer);

  // Print the contents of the output buffer.
  // std::cout << "======\n";
  // std::cout << "Output Buffer:" << std::endl;
  // compute_runner->log_buffer<float>(output_buffer);

  // Print the latency.
  // std::cout << "======\n";
  // std::cout << "Latency (nanoseconds):\n";
  // std::cout << "  CPU Time: " << times_ns.first << " ns\n";
  // std::cout << "  GPU Time: " << times_ns.second << " ns\n";

  for(int i = 0; i < 10; i ++) {
    // warm up
    compute_runner->run();
  }

  std::vector<double> sorted_time;

  double cpu_time = 0;
  double gpu_time = 0;

  int round = 100;

  for(int i = 0; i < round; i ++) {
    auto times_ns = compute_runner->run();

    cpu_time += times_ns.first;
    gpu_time += times_ns.second;
    sorted_time.push_back(times_ns.second);
  }

  std::cout << "  CPU Time: " << cpu_time / round << " ns\n";
  std::cout << "  GPU Time: " << gpu_time / round << " ns\n";
 
  size_t total = sorted_time.size();
  size_t n50 = static_cast<size_t>(total * 0.5);
  size_t n60 = static_cast<size_t>(total * 0.6);
  size_t n70 = static_cast<size_t>(total * 0.7);
  size_t n80 = static_cast<size_t>(total * 0.8);
  size_t n90 = static_cast<size_t>(total * 0.9);
  size_t n95 = static_cast<size_t>(total * 0.95);
  size_t n99 = static_cast<size_t>(total * 0.99);
  size_t n999 = static_cast<size_t>(total * 0.999);
 
  std::sort(sorted_time.begin(), sorted_time.end());
 
  std::cout << std::left << std::setprecision(3) << std::fixed;
  std::cout << "Totla samples: " << sorted_time.size() << "\n";
  std::cout << "Min Latency: " << sorted_time[0] << " us\n";
  std::cout << "Max Latency: " << sorted_time[total - 1] << " us\n";
  std::cout << "P50 Latency: " << sorted_time[n50] << " us\n";
  std::cout << "P60 Latency: " << sorted_time[n60] << " us\n";
  std::cout << "P70 Latency: " << sorted_time[n70] << " us\n";
  std::cout << "P80 Latency: " << sorted_time[n80] << " us\n";
  std::cout << "P90 Latency: " << sorted_time[n90] << " us\n";
  std::cout << "P95 Latency: " << sorted_time[n95] << " us\n";
  std::cout << "P99 Latency: " << sorted_time[n99] << " us\n";
  std::cout << "P999 Latency: " << sorted_time[n999] << " us" << std::endl;

  return 0;
}
