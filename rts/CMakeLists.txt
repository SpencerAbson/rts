cmake_minimum_required(VERSION 3.20.0)
project(rts)

# --------------------------------------------------------------
# Library sources
# --------------------------------------------------------------
add_library(rts STATIC
	src/network.cpp
	src/tensor.cpp
	src/threads.cpp
	src/buffers.cpp
	src/signal.cpp
	src/layers/layer.cpp
	src/layers/linear_lif.cpp
	src/layers/rec_linear_lif.cpp
	src/layers/full_rec_linear_lif.cpp
)

target_include_directories(rts PUBLIC include)
target_compile_options(rts PUBLIC -Wall -Wpedantic -O2 -std=c++20 -mcpu=cortex-a76)
