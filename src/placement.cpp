#include "moe/placement.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace moe {

std::vector<int> round_robin_placement(std::uint32_t experts, int world_size) {
  if (experts == 0 || world_size <= 0) {
    throw std::invalid_argument("placement requires Experts and positive MPI size");
  }
  std::vector<int> owners(experts);
  for (std::uint32_t expert = 0; expert < experts; ++expert) {
    owners[expert] = static_cast<int>(expert % static_cast<std::uint32_t>(world_size));
  }
  return owners;
}

std::vector<std::uint64_t> rank_assignment_loads(
    const std::vector<std::uint64_t>& expert_loads,
    const std::vector<int>& owners, int world_size) {
  if (world_size <= 0 || expert_loads.empty() || expert_loads.size() != owners.size()) {
    throw std::invalid_argument("invalid Expert placement dimensions");
  }
  std::vector<std::uint64_t> loads(static_cast<std::size_t>(world_size), 0);
  for (std::size_t expert = 0; expert < owners.size(); ++expert) {
    if (owners[expert] < 0 || owners[expert] >= world_size) {
      throw std::invalid_argument("Expert owner is outside the MPI world");
    }
    loads[static_cast<std::size_t>(owners[expert])] += expert_loads[expert];
  }
  return loads;
}

std::vector<int> load_aware_placement(const std::vector<std::uint64_t>& predicted,
                                      int world_size) {
  if (world_size <= 0 || predicted.empty()) {
    throw std::invalid_argument("load-aware placement requires predicted Expert loads");
  }
  std::vector<std::size_t> order(predicted.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    return predicted[left] == predicted[right] ? left < right
                                               : predicted[left] > predicted[right];
  });
  std::vector<int> owners(predicted.size(), -1);
  std::vector<std::uint64_t> loads(static_cast<std::size_t>(world_size), 0);
  for (const std::size_t expert : order) {
    const auto smallest = std::min_element(loads.begin(), loads.end());
    const int rank = static_cast<int>(smallest - loads.begin());
    owners[expert] = rank;
    *smallest += predicted[expert];
  }
  return owners;
}

}  // namespace moe
