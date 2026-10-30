#pragma once

#include "moe/data.hpp"

#include <vector>

namespace moe {

struct ForwardResult {
  std::vector<float> output;  // [N][d]
  std::vector<RoutingDecision> routes;
};

ForwardResult forward_serial(const MoEConfig& config, const Model& model,
                             const std::vector<float>& tokens,
                             const std::vector<RoutingDecision>* trace_routes);

ForwardResult forward_naive_reference(
    const MoEConfig& config, const Model& model, const std::vector<float>& tokens,
    const std::vector<RoutingDecision>* trace_routes);

}  // namespace moe
