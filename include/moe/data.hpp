#pragma once

#include "moe/config.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace moe {

struct ExpertWeights {
  std::vector<float> w1;  // [d][h], row-major
  std::vector<float> w2;  // [h][d], row-major
};

struct Model {
  std::vector<float> router_weights;  // [d][E], row-major
  std::vector<ExpertWeights> experts;
};

struct RoutingDecision {
  std::array<std::uint32_t, 2> expert_ids{};
  std::array<float, 2> gate_weights{};
};

// Every value is a function of its seed and global index, never MPI rank.
Model generate_model(const MoEConfig& config);
std::vector<float> generate_tokens(const MoEConfig& config,
                                   std::uint64_t first_token,
                                   std::uint64_t token_count);

// Trace mode creates exactly 2N assignments, with no repeated Expert in a Token.
std::vector<RoutingDecision> generate_controlled_trace(const MoEConfig& config);
std::vector<std::uint64_t> routing_histogram(
    const std::vector<RoutingDecision>& routes, std::uint32_t expert_count);

}  // namespace moe
