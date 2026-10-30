#include "moe/config.hpp"
#include "moe/protocol.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void test_contiguous_partition() {
  const std::vector<moe::TokenRange> expected{{0, 2}, {2, 2}, {4, 2}, {6, 1}};
  for (int rank = 0; rank < 4; ++rank) {
    const auto observed = moe::contiguous_token_range(7, rank, 4);
    require(observed.begin == expected[rank].begin && observed.count == expected[rank].count,
            "N=7, P=4 must assign remainders to the first ranks");
  }
  require(moe::contiguous_token_range(2, 3, 4).count == 0,
          "N<P must permit an empty local token range");
}

void test_wire_header_round_trip() {
  const moe::AssignmentHeader expected{123456789ULL, 7U, 15U, 1U};
  const auto wire = moe::pack_header(expected);
  const auto observed = moe::unpack_header(wire);
  require(observed.global_token_id == expected.global_token_id &&
              observed.source_rank == expected.source_rank &&
              observed.expert_id == expected.expert_id && observed.slot_id == expected.slot_id,
          "wire header must round-trip exactly");
}

void test_assignment_validation() {
  moe::MoEConfig config;
  config.tokens = 7;
  config.dimension = 4;
  config.experts = 4;
  config.top_k = 2;

  moe::AssignmentBuffer valid;
  valid.headers.push_back(moe::pack_header({6, 3, 2, 1}));
  valid.token_values.assign(4, 1.0F);
  std::string error;
  require(moe::valid_assignment_buffer(valid, config, 4, &error),
          "well-formed assignment buffer must validate");

  valid.token_values.pop_back();
  require(!moe::valid_assignment_buffer(valid, config, 4, &error),
          "assignment buffer must contain exactly d floats per header");
}

void test_config_contract() {
  const auto valid = moe::parse_command_line(
      {"--router-mode", "trace", "--skew", "moderate", "--hot-layout", "B",
       "--placement", "la", "--communication", "p2p", "--backend", "cuda", "--topk", "2"});
  require(valid.ok, "valid trace configuration must parse");
  require(valid.config.router_mode == moe::RouterMode::kTrace &&
              valid.config.skew == moe::Skew::kModerate &&
              valid.config.placement == moe::Placement::kLoadAware &&
              valid.config.expert_backend == moe::ExpertBackend::kCuda,
          "parsed trace options must retain their values");

  const auto invalid =
      moe::parse_command_line({"--router-mode", "router", "--skew", "extreme"});
  require(!invalid.ok, "router mode must reject a controlled skew option");
}

}  // namespace

int main() {
  test_contiguous_partition();
  test_wire_header_round_trip();
  test_assignment_validation();
  test_config_contract();
  std::cout << "protocol tests: PASS\n";
  return EXIT_SUCCESS;
}
