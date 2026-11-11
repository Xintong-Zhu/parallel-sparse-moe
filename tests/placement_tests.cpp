#include "moe/data.hpp"
#include "moe/placement.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  const std::vector<std::uint64_t> predicted{40, 15, 15, 15, 15};
  const auto rr = moe::round_robin_placement(5, 2);
  const auto la = moe::load_aware_placement(predicted, 2);
  if (rr != std::vector<int>{0, 1, 0, 1, 0} ||
      la != std::vector<int>{0, 1, 1, 1, 0} ||
      moe::rank_assignment_loads(predicted, la, 2) !=
          std::vector<std::uint64_t>{55, 45}) {
    std::cerr << "placement test: deterministic LPT assignment failed\n";
    return EXIT_FAILURE;
  }
  try {
    moe::rank_assignment_loads(predicted, {0, 1, 2, 0, 0}, 2);
    std::cerr << "placement test: invalid owner accepted\n";
    return EXIT_FAILURE;
  } catch (const std::invalid_argument&) {
  }
  moe::MoEConfig config;
  config.dimension = 4;
  config.hidden_dimension = 8;
  config.experts = 5;
  const auto full = moe::generate_model(config);
  for (int rank = 0; rank < 2; ++rank) {
    const auto part = moe::generate_partitioned_model(config, rank, 2, la);
    if (part.router_weights != full.router_weights) return EXIT_FAILURE;
    for (std::size_t expert = 0; expert < la.size(); ++expert) {
      const bool owned = la[expert] == rank;
      if (owned ? (part.experts[expert].w1 != full.experts[expert].w1 ||
                   part.experts[expert].w2 != full.experts[expert].w2)
                : (!part.experts[expert].w1.empty() || !part.experts[expert].w2.empty())) {
        std::cerr << "placement test: partitioned weights differ\n";
        return EXIT_FAILURE;
      }
    }
  }
  std::cout << "placement tests: PASS\n";
  return EXIT_SUCCESS;
}
