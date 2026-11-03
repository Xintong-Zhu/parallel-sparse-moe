cmake_minimum_required(VERSION 3.20)

project(parallel_sparse_moe LANGUAGES CXX)

include(CTest)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(moe_core STATIC
  src/config.cpp
  src/data.cpp
  src/expert.cpp
  src/protocol.cpp
  src/router.cpp
  src/serial.cpp
)
target_compile_features(moe_core PUBLIC cxx_std_17)
target_include_directories(moe_core PUBLIC include)

add_executable(moe_launch_probe src/launch_probe.cpp)
target_compile_features(moe_launch_probe PRIVATE cxx_std_17)
target_link_libraries(moe_launch_probe PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(moe_contract_check src/contract_check.cpp)
target_link_libraries(moe_contract_check PRIVATE moe_core)

add_executable(moe_serial src/serial_main.cpp)
target_link_libraries(moe_serial PRIVATE moe_core)

add_executable(moe_mpi_loopback src/mpi_loopback.cpp)
target_link_libraries(moe_mpi_loopback PRIVATE moe_core MPI::MPI_CXX)

add_executable(moe_parallel_check src/parallel_check.cpp)
target_link_libraries(moe_parallel_check PRIVATE moe_core MPI::MPI_CXX)

if(BUILD_TESTING)
  add_executable(moe_protocol_tests tests/protocol_tests.cpp)
  target_link_libraries(moe_protocol_tests PRIVATE moe_core)
  add_test(NAME moe_protocol_tests COMMAND moe_protocol_tests)

  add_executable(moe_serial_tests tests/serial_tests.cpp)
  target_link_libraries(moe_serial_tests PRIVATE moe_core)
  add_test(NAME moe_serial_tests COMMAND moe_serial_tests)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(moe_core PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
  target_compile_options(moe_launch_probe PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(moe_contract_check PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(moe_serial PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(moe_mpi_loopback PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(moe_parallel_check PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
  if(BUILD_TESTING)
    target_compile_options(moe_protocol_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(moe_serial_tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()
