cmake_minimum_required(VERSION 3.20.0)
project(dvs_gesture_bench)

# --------------------------------------------------------------
# Definitions
# --------------------------------------------------------------
add_definitions(-DPROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
		-DRTS_PERF_DIR="${CMAKE_CURRENT_SOURCE_DIR}/perf"
		-DRTS_EN_PROFILE_NETWORK=1
		-DRTS_EN_DEBUG_PRINT=1
		-DRTS_EN_CHECKING_ASSERT=1)

# --------------------------------------------------------------
# Libraries
# --------------------------------------------------------------
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../rts rts_build)

add_executable(bench main.cpp)
target_link_libraries(bench PUBLIC rts)
