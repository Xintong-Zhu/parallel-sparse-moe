#pragma once

#include "moe/data.hpp"

namespace moe {

// Production FP32 router. Ties select the smaller Expert ID.
RoutingDecision route_token(const Model& model, const MoEConfig& config,
                            const float* token);

// Independent FP64 accumulation path used by naive_ref validation.
RoutingDecision route_token_reference(const Model& model, const MoEConfig& config,
                                      const float* token);

}  // namespace moe
