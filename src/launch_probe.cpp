#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

struct Options {
  int expected_threads = -1;
  bool require_distinct_hosts = false;
};

const char* thread_level_name(int level) {
  switch (level) {
    case MPI_THREAD_SINGLE:
      return "MPI_THREAD_SINGLE";
    case MPI_THREAD_FUNNELED:
      return "MPI_THREAD_FUNNELED";
    case MPI_THREAD_SERIALIZED:
      return "MPI_THREAD_SERIALIZED";
    case MPI_THREAD_MULTIPLE:
      return "MPI_THREAD_MULTIPLE";
    default:
      return "unknown";
  }
}

bool parse_options(int argc, char** argv, Options& options, std::string& error) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--require-distinct-hosts") {
      options.require_distinct_hosts = true;
    } else if (argument == "--expect-threads" && i + 1 < argc) {
      try {
        options.expected_threads = std::stoi(argv[++i]);
      } catch (const std::exception&) {
        error = "--expect-threads requires a positive integer";
        return false;
      }
      if (options.expected_threads <= 0) {
        error = "--expect-threads requires a positive integer";
        return false;
      }
    } else {
      error = "unknown or incomplete option: " + argument;
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  Options options;
  std::string option_error;
  const bool valid_options = parse_options(argc, argv, options, option_error);
  int valid_options_int = valid_options ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &valid_options_int, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  if (!valid_options_int) {
    if (rank == 0) {
      std::cerr << "Usage: moe_launch_probe [--expect-threads N] "
                   "[--require-distinct-hosts]\n";
      if (!option_error.empty()) {
        std::cerr << "Error: " << option_error << '\n';
      }
    }
    MPI_Finalize();
    return 2;
  }

  if (provided < MPI_THREAD_FUNNELED) {
    if (rank == 0) {
      std::cerr << "MPI implementation provided " << thread_level_name(provided)
                << "; this project requires at least MPI_THREAD_FUNNELED.\n";
    }
    MPI_Finalize();
    return 3;
  }

  std::array<char, MPI_MAX_PROCESSOR_NAME> hostname{};
  int hostname_length = 0;
  MPI_Get_processor_name(hostname.data(), &hostname_length);
  hostname[std::min(hostname_length, MPI_MAX_PROCESSOR_NAME - 1)] = '\0';

  int observed_threads = 0;
#pragma omp parallel
  {
#pragma omp single
    observed_threads = omp_get_num_threads();
  }

  const int successor = (rank + 1) % world_size;
  const int predecessor = (rank - 1 + world_size) % world_size;
  const int sent_value = rank;
  int received_value = -1;
  MPI_Sendrecv(&sent_value, 1, MPI_INT, successor, 101, &received_value, 1,
               MPI_INT, predecessor, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  int local_ok = (received_value == predecessor) ? 1 : 0;

  std::vector<int> all_threads;
  std::vector<int> all_ring_ok;
  std::vector<char> all_hostnames;
  if (rank == 0) {
    all_threads.resize(world_size);
    all_ring_ok.resize(world_size);
    all_hostnames.resize(static_cast<std::size_t>(world_size) *
                         MPI_MAX_PROCESSOR_NAME);
  }

  MPI_Gather(&observed_threads, 1, MPI_INT, all_threads.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);
  MPI_Gather(&local_ok, 1, MPI_INT, all_ring_ok.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);
  MPI_Gather(hostname.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
             all_hostnames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0,
             MPI_COMM_WORLD);

  int passed = 1;
  if (rank == 0) {
    std::set<std::string> hosts;
    std::cout << "MPI/OpenMP launch probe\n"
              << "  MPI ranks: " << world_size << '\n'
              << "  MPI thread support: " << thread_level_name(provided) << '\n'
              << "  OMP_NUM_THREADS: "
              << (std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS")
                                                   : "UNSET")
              << "\n";
    for (int current_rank = 0; current_rank < world_size; ++current_rank) {
      const char* current_host =
          all_hostnames.data() + current_rank * MPI_MAX_PROCESSOR_NAME;
      hosts.emplace(current_host);
      std::cout << "  rank " << current_rank << ": host=" << current_host
                << ", OpenMP threads=" << all_threads[current_rank]
                << ", ring=" << (all_ring_ok[current_rank] ? "ok" : "failed")
                << '\n';
      if (!all_ring_ok[current_rank]) {
        passed = 0;
      }
      if (options.expected_threads > 0 &&
          all_threads[current_rank] != options.expected_threads) {
        passed = 0;
      }
    }
    if (options.require_distinct_hosts && hosts.size() < 2) {
      std::cerr << "Expected ranks on at least two hosts, but observed "
                << hosts.size() << ".\n";
      passed = 0;
    }
    if (passed) {
      std::cout << "RESULT: PASS\n";
    } else {
      std::cerr << "RESULT: FAIL\n";
    }
  }

  MPI_Bcast(&passed, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return passed ? 0 : 4;
}
