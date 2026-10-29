cmake_minimum_required(VERSION 3.20)

project(parallel_sparse_moe LANGUAGES CXX)

include(CTest)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(moe_core STATIC
  src/config.cpp
  src/protocol.cpp
)
target_compile_features(moe_core PUBLIC cxx_std_17)
target_include_directories(moe_core PUBLIC include)

add_executable(moe_launch_probe src/launch_probe.cpp)
target_compile_features(moe_launch_probe PRIVATE cxx_std_17)
target_link_libraries(moe_launch_probe PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(moe_contract_check src/contract_check.cpp)
target_link_libraries(moe_contract_check PRIVATE moe_core)

if(BUILD_TESTING)
  add_executable(moe_protocol_tests tests/protocol_tests.cpp)
  target_link_libraries(moe_protocol_tests PRIVATE moe_core)
  add_test(NAME moe_protocol_tests COMMAND moe_protocol_tests)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(moe_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(moe_launch_probe PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(moe_contract_check PRIVATE -Wall -Wextra -Wpedantic)
  if(BUILD_TESTING)
    target_compile_options(moe_protocol_tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()
