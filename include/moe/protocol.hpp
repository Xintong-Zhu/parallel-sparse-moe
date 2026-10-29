#pragma once

#include "moe/config.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace moe {

struct TokenRange {
  std::uint64_t begin = 0;
  std::uint64_t count = 0;
};

// Logical metadata used by the application. It is never sent as a raw C++
// struct, so compiler padding cannot change the MPI wire format.
struct AssignmentHeader {
  std::uint64_t global_token_id = 0;
  std::uint32_t source_rank = 0;
  std::uint32_t expert_id = 0;
  std::uint8_t slot_id = 0;
};

// A fixed 16-byte representation for MPI_BYTE exchanges. word[1] stores,
// low-to-high: source_rank (32 bits), expert_id (16 bits), slot_id (8 bits),
// and eight reserved zero bits.
struct WireAssignmentHeader {
  std::array<std::uint64_t, 2> word{};
};

static_assert(sizeof(WireAssignmentHeader) == 16,
              "The MPI assignment wire header must be 16 bytes.");
static_assert(std::is_trivially_copyable<WireAssignmentHeader>::value,
              "Wire headers must be safe to exchange as MPI_BYTE.");

struct AssignmentBuffer {
  std::vector<WireAssignmentHeader> headers;
  std::vector<float> token_values;  // headers.size() contiguous vectors of d floats.
};

TokenRange contiguous_token_range(std::uint64_t global_tokens, int rank,
                                   int world_size);
int round_robin_expert_owner(std::uint32_t expert_id, int world_size);

WireAssignmentHeader pack_header(const AssignmentHeader& header);
AssignmentHeader unpack_header(const WireAssignmentHeader& wire);

bool valid_assignment_header(const AssignmentHeader& header,
                             const MoEConfig& config, int world_size,
                             std::string* error = nullptr);
bool valid_assignment_buffer(const AssignmentBuffer& buffer,
                             const MoEConfig& config, int world_size,
                             std::string* error = nullptr);

}  // namespace moe
