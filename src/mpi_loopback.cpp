#include "moe/config.hpp"
#include "moe/data.hpp"
#include "moe/protocol.hpp"
#include "moe/router.hpp"

#include <mpi.h>

#include <climits>
#include <cstddef>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using BuffersByRank = std::vector<moe::AssignmentBuffer>;

int checked_int(std::size_t value, const char* description) {
  if (value > static_cast<std::size_t>(INT_MAX)) {
    throw std::overflow_error(std::string(description) + " exceeds MPI int count limit");
  }
  return static_cast<int>(value);
}

moe::AssignmentBuffer flatten(const BuffersByRank& by_rank, std::vector<int>& counts,
                              std::vector<int>& displacements, std::uint32_t dimension) {
  moe::AssignmentBuffer flat;
  counts.resize(by_rank.size());
  displacements.resize(by_rank.size());
  std::size_t header_offset = 0;
  for (std::size_t rank = 0; rank < by_rank.size(); ++rank) {
    const auto& buffer = by_rank[rank];
    if (buffer.token_values.size() != buffer.headers.size() * dimension) {
      throw std::logic_error("cannot flatten malformed assignment buffer");
    }
    counts[rank] = checked_int(buffer.headers.size(), "assignment count");
    displacements[rank] = checked_int(header_offset, "assignment displacement");
    header_offset += buffer.headers.size();
    flat.headers.insert(flat.headers.end(), buffer.headers.begin(), buffer.headers.end());
    flat.token_values.insert(flat.token_values.end(), buffer.token_values.begin(),
                             buffer.token_values.end());
  }
  return flat;
}

std::vector<int> scaled_counts(const std::vector<int>& counts, std::uint32_t scale,
                               const char* description) {
  std::vector<int> scaled(counts.size());
  for (std::size_t i = 0; i < counts.size(); ++i) {
    const auto value = static_cast<std::size_t>(counts[i]) * scale;
    scaled[i] = checked_int(value, description);
  }
  return scaled;
}

moe::AssignmentBuffer exchange_alltoallv(const BuffersByRank& by_rank,
                                         std::uint32_t dimension,
                                         MPI_Comm communicator) {
  int world_size = 0;
  MPI_Comm_size(communicator, &world_size);
  std::vector<int> send_counts;
  std::vector<int> send_displacements;
  const auto send = flatten(by_rank, send_counts, send_displacements, dimension);
  std::vector<int> receive_counts(static_cast<std::size_t>(world_size));
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, receive_counts.data(), 1, MPI_INT,
               communicator);

  std::vector<int> receive_displacements(static_cast<std::size_t>(world_size));
  std::size_t received_assignments = 0;
  for (int rank = 0; rank < world_size; ++rank) {
    receive_displacements[rank] = checked_int(received_assignments, "receive displacement");
    received_assignments += static_cast<std::size_t>(receive_counts[rank]);
  }

  moe::AssignmentBuffer received;
  received.headers.resize(received_assignments);
  received.token_values.resize(received_assignments * dimension);
  const auto send_header_counts = scaled_counts(send_counts, 2U, "header word count");
  const auto send_header_displacements =
      scaled_counts(send_displacements, 2U, "header word displacement");
  const auto receive_header_counts = scaled_counts(receive_counts, 2U, "header word count");
  const auto receive_header_displacements =
      scaled_counts(receive_displacements, 2U, "header word displacement");
  const auto send_value_counts = scaled_counts(send_counts, dimension, "float count");
  const auto send_value_displacements =
      scaled_counts(send_displacements, dimension, "float displacement");
  const auto receive_value_counts = scaled_counts(receive_counts, dimension, "float count");
  const auto receive_value_displacements =
      scaled_counts(receive_displacements, dimension, "float displacement");

  MPI_Alltoallv(send.headers.data(), send_header_counts.data(),
                send_header_displacements.data(), MPI_UINT64_T, received.headers.data(),
                receive_header_counts.data(), receive_header_displacements.data(), MPI_UINT64_T,
                communicator);
  MPI_Alltoallv(send.token_values.data(), send_value_counts.data(),
                send_value_displacements.data(), MPI_FLOAT, received.token_values.data(),
                receive_value_counts.data(), receive_value_displacements.data(), MPI_FLOAT,
                communicator);
  return received;
}

BuffersByRank repartition_by_destination(const moe::AssignmentBuffer& source,
                                         const moe::MoEConfig& config, int world_size,
                                         bool return_to_source) {
  BuffersByRank result(static_cast<std::size_t>(world_size));
  for (std::size_t assignment = 0; assignment < source.headers.size(); ++assignment) {
    const auto header = moe::unpack_header(source.headers[assignment]);
    const int destination = return_to_source
                                ? static_cast<int>(header.source_rank)
                                : moe::round_robin_expert_owner(header.expert_id, world_size);
    auto& destination_buffer = result[static_cast<std::size_t>(destination)];
    destination_buffer.headers.push_back(source.headers[assignment]);
    const auto begin = source.token_values.begin() +
                       static_cast<std::ptrdiff_t>(assignment * config.dimension);
    destination_buffer.token_values.insert(destination_buffer.token_values.end(), begin,
                                           begin + config.dimension);
  }
  return result;
}

int run(int argc, char** argv) {
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<std::string> arguments;
  for (int i = 1; i < argc; ++i) {
    arguments.emplace_back(argv[i]);
  }
  const auto parsed = moe::parse_command_line(arguments);
  int parse_ok = parsed.ok && !parsed.help_requested && provided >= MPI_THREAD_FUNNELED;
  MPI_Allreduce(MPI_IN_PLACE, &parse_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!parse_ok) {
    if (rank == 0) {
      if (parsed.help_requested) {
        std::cout << moe::command_line_help();
      } else if (!parsed.ok) {
        std::cerr << "Error: " << parsed.error << '\n';
      } else {
        std::cerr << "MPI implementation does not provide MPI_THREAD_FUNNELED\n";
      }
    }
    MPI_Finalize();
    return 2;
  }
  const auto& config = parsed.config;
  try {
    const auto local_range = moe::contiguous_token_range(config.tokens, rank, world_size);
    const auto local_tokens = moe::generate_tokens(config, local_range.begin, local_range.count);
    const auto model = moe::generate_model(config);
    const auto trace = config.router_mode == moe::RouterMode::kTrace
                           ? moe::generate_controlled_trace(config)
                           : std::vector<moe::RoutingDecision>{};

    BuffersByRank dispatch(static_cast<std::size_t>(world_size));
    for (std::uint64_t local = 0; local < local_range.count; ++local) {
      const auto global_token = local_range.begin + local;
      const float* token = local_tokens.data() + static_cast<std::size_t>(local) * config.dimension;
      const auto route = config.router_mode == moe::RouterMode::kTrace
                             ? trace[static_cast<std::size_t>(global_token)]
                             : moe::route_token(model, config, token);
      for (std::uint8_t slot = 0; slot < config.top_k; ++slot) {
        const moe::AssignmentHeader header{global_token, static_cast<std::uint32_t>(rank),
                                           route.expert_ids[slot], slot};
        const int destination = moe::round_robin_expert_owner(header.expert_id, world_size);
        auto& buffer = dispatch[static_cast<std::size_t>(destination)];
        buffer.headers.push_back(moe::pack_header(header));
        buffer.token_values.insert(buffer.token_values.end(), token, token + config.dimension);
      }
    }

    const auto received = exchange_alltoallv(dispatch, config.dimension, MPI_COMM_WORLD);
    std::string error;
    int local_ok = moe::valid_assignment_buffer(received, config, world_size, &error) ? 1 : 0;
    for (const auto& wire : received.headers) {
      const auto header = moe::unpack_header(wire);
      local_ok &= moe::round_robin_expert_owner(header.expert_id, world_size) == rank;
    }

    // Do not use malformed headers to choose a return destination.  This path
    // is only expected for a protocol failure, but it keeps validation from
    // turning into an out-of-bounds access before the collective result is
    // reported.
    int all_receives_valid = 0;
    MPI_Allreduce(&local_ok, &all_receives_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_receives_valid) {
      throw std::runtime_error(error.empty() ? "invalid received assignment" : error);
    }

    const auto returned = exchange_alltoallv(
        repartition_by_destination(received, config, world_size, true), config.dimension,
        MPI_COMM_WORLD);
    std::vector<unsigned char> seen(static_cast<std::size_t>(local_range.count) * config.top_k,
                                    0U);
    if (returned.headers.size() != static_cast<std::size_t>(local_range.count) * config.top_k) {
      local_ok = 0;
    }
    for (std::size_t assignment = 0; assignment < returned.headers.size(); ++assignment) {
      const auto header = moe::unpack_header(returned.headers[assignment]);
      if (header.source_rank != static_cast<std::uint32_t>(rank) ||
          header.global_token_id < local_range.begin ||
          header.global_token_id >= local_range.begin + local_range.count ||
          header.slot_id >= config.top_k) {
        local_ok = 0;
        continue;
      }
      const auto local_token = header.global_token_id - local_range.begin;
      const auto seen_index = static_cast<std::size_t>(local_token) * config.top_k + header.slot_id;
      if (seen[seen_index] != 0U) {
        local_ok = 0;
        continue;
      }
      seen[seen_index] = 1U;
      const float* expected = local_tokens.data() + local_token * config.dimension;
      const float* actual = returned.token_values.data() + assignment * config.dimension;
      if (std::memcmp(expected, actual, static_cast<std::size_t>(config.dimension) * sizeof(float)) != 0) {
        local_ok = 0;
      }
    }
    for (const auto value : seen) {
      local_ok &= value == 1U;
    }

    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "MPI loopback: " << (global_ok ? "PASS" : "FAIL") << " (P=" << world_size
                << ", N=" << config.tokens << ")\n";
    }
    MPI_Finalize();
    return global_ok ? 0 : 1;
  } catch (const std::exception& exception) {
    std::cerr << "rank " << rank << " error: " << exception.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  return run(argc, argv);
}
