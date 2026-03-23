# --------------------------------------------------------------
# Definitions
# --------------------------------------------------------------
add_definitions(-DPROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

# --------------------------------------------------------------
# Linear Leaky-integrate-and-fire
# --------------------------------------------------------------
add_executable(lif_layer_f32 linear_lif/test_layer_f32.cpp)
target_include_directories(lif_layer_f32 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lif_layer_f32 PRIVATE rts)
add_test(NAME linear_lif_layer_f32 COMMAND lif_layer_f32)

add_executable(lif_layer_f16 linear_lif/test_layer_f16.cpp)
target_include_directories(lif_layer_f16 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lif_layer_f16 PRIVATE rts)
add_test(NAME linear_lif_layer_f16 COMMAND lif_layer_f16)
