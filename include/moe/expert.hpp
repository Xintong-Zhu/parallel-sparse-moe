#pragma once

#include "moe/data.hpp"

#include <cstddef>
#include <vector>

namespace moe {

// Shared production FP32 Expert kernel for serial, DP, and EP implementations.
void expert_forward_cpu(const ExpertWeights& weights, const MoEConfig& config,
                        const float* token, float* output,
                        std::vector<float>& hidden_scratch);

// Contiguous [count][d] input and output. Reuses each weight row across the
// batch while preserving each Token's scalar accumulation order.
void expert_forward_cpu_batch(const ExpertWeights& weights, const MoEConfig& config,
                              const float* tokens, std::size_t count, float* outputs,
                              std::vector<float>& hidden_scratch);

// Independent FP64 accumulation path for naive_ref validation.
void expert_forward_reference(const ExpertWeights& weights, const MoEConfig& config,
                              const float* token, float* output,
                              std::vector<double>& hidden_scratch);

}  // namespace moe
