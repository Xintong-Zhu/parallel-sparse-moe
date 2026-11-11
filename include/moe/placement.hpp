#pragma once

#include <cstdint>
#include <vector>

namespace moe {

std::vector<int> round_robin_placement(std::uint32_t experts, int world_size);
// Longest-processing-time-first: descending predicted assignments, then Expert ID.
// Equal current rank loads are resolved by rank ID.
std::vector<int> load_aware_placement(const std::vector<std::uint64_t>& predicted,
                                      int world_size);
std::vector<std::uint64_t> rank_assignment_loads(
    const std::vector<std::uint64_t>& expert_loads,
    const std::vector<int>& owners, int world_size);

}  // namespace moe
