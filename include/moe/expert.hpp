#pragma once

#include "moe/data.hpp"

#include <vector>

namespace moe {

// Shared production FP32 Expert kernel for serial, DP, and EP implementations.
void expert_forward_cpu(const ExpertWeights& weights, const MoEConfig& config,
                        const float* token, float* output,
                        std::vector<float>& hidden_scratch);

// Independent FP64 accumulation path for naive_ref validation.
void expert_forward_reference(const ExpertWeights& weights, const MoEConfig& config,
                              const float* token, float* output,
                              std::vector<double>& hidden_scratch);

}  // namespace moe
