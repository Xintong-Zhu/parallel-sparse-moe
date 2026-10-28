cmake_minimum_required(VERSION 3.20)

project(parallel_sparse_moe LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(moe_launch_probe src/launch_probe.cpp)
target_compile_features(moe_launch_probe PRIVATE cxx_std_17)
target_link_libraries(moe_launch_probe PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(moe_launch_probe PRIVATE -Wall -Wextra -Wpedantic)
endif()
