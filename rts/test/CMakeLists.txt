# --------------------------------------------------------------
# Definitions
# --------------------------------------------------------------
add_definitions(-DPROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

# --------------------------------------------------------------
# Linear leaky-integrate-and-fire
# --------------------------------------------------------------
add_executable(lif_layer_f32 linear_lif/test_layer_f32.cpp)
target_include_directories(lif_layer_f32 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lif_layer_f32 PRIVATE rts)
add_test(NAME linear_lif_layer_f32 COMMAND lif_layer_f32)

add_executable(lif_layer_f16 linear_lif/test_layer_f16.cpp)
target_include_directories(lif_layer_f16 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lif_layer_f16 PRIVATE rts)
add_test(NAME linear_lif_layer_f16 COMMAND lif_layer_f16)

add_executable(lif_net_f32_n_1 linear_lif/test_network_f32_n_1.cpp)
target_include_directories(lif_net_f32_n_1 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lif_net_f32_n_1 PRIVATE rts)
add_test(NAME linear_lif_net_f32_n_1 COMMAND lif_net_f32_n_1)

add_executable(lif_net_f32_n_2 linear_lif/test_network_f32_n_2.cpp)
target_include_directories(lif_net_f32_n_2 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lif_net_f32_n_2 PRIVATE rts)
add_test(NAME linear_lif_net_f32_n_2 COMMAND lif_net_f32_n_2)

add_executable(lif_net_f32_n_5 linear_lif/test_network_f32_n_5.cpp)
target_include_directories(lif_net_f32_n_5 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lif_net_f32_n_5 PRIVATE rts)
add_test(NAME linear_lif_net_f32_n_5 COMMAND lif_net_f32_n_5)

add_executable(lif_net_f16_n_1 linear_lif/test_network_f16_n_1.cpp)
target_include_directories(lif_net_f16_n_1 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lif_net_f16_n_1 PRIVATE rts)
add_test(NAME linear_lif_net_f16_n_1 COMMAND lif_net_f16_n_1)

add_executable(lif_net_f16_n_2 linear_lif/test_network_f16_n_2.cpp)
target_include_directories(lif_net_f16_n_2 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lif_net_f16_n_2 PRIVATE rts)
add_test(NAME linear_lif_net_f16_n_2 COMMAND lif_net_f16_n_2)

add_executable(lif_net_f16_n_5 linear_lif/test_network_f16_n_5.cpp)
target_include_directories(lif_net_f16_n_5 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lif_net_f16_n_5 PRIVATE rts)
add_test(NAME linear_lif_net_f16_n_5 COMMAND lif_net_f16_n_5)

# --------------------------------------------------------------
# One-to-one recurrent linear Leaky-integrate-and-fire
# --------------------------------------------------------------
add_executable(rec_lif_layer_f32 rec_linear_lif/test_layer_f32.cpp)
target_include_directories(rec_lif_layer_f32 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rec_lif_layer_f32 PRIVATE rts)
add_test(NAME rec_linear_lif_layer_f32 COMMAND rec_lif_layer_f32)

add_executable(rec_lif_layer_f16 rec_linear_lif/test_layer_f16.cpp)
target_include_directories(rec_lif_layer_f16 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rec_lif_layer_f16 PRIVATE rts)
add_test(NAME rec_linear_lif_layer_f16 COMMAND rec_lif_layer_f16)

add_executable(rec_lif_net_f32_n_1 rec_linear_lif/test_network_f32_n_1.cpp)
target_include_directories(rec_lif_net_f32_n_1 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rec_lif_net_f32_n_1 PRIVATE rts)
add_test(NAME rec_linear_lif_net_f32_n_1 COMMAND rec_lif_net_f32_n_1)

add_executable(rec_lif_net_f32_n_2 rec_linear_lif/test_network_f32_n_2.cpp)
target_include_directories(rec_lif_net_f32_n_2 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rec_lif_net_f32_n_2 PRIVATE rts)
add_test(NAME rec_linear_lif_net_f32_n_2 COMMAND rec_lif_net_f32_n_2)

add_executable(rec_lif_net_f32_n_4 rec_linear_lif/test_network_f32_n_4.cpp)
target_include_directories(rec_lif_net_f32_n_4 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rec_lif_net_f32_n_4 PRIVATE rts)
add_test(NAME rec_linear_lif_net_f32_n_4 COMMAND rec_lif_net_f32_n_4)

add_executable(rec_lif_net_f16_n_1 rec_linear_lif/test_network_f16_n_1.cpp)
target_include_directories(rec_lif_net_f16_n_1 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rec_lif_net_f16_n_1 PRIVATE rts)
add_test(NAME rec_linear_lif_net_f16_n_1 COMMAND rec_lif_net_f16_n_1)

add_executable(rec_lif_net_f16_n_2 rec_linear_lif/test_network_f16_n_2.cpp)
target_include_directories(rec_lif_net_f16_n_2 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rec_lif_net_f16_n_2 PRIVATE rts)
add_test(NAME rec_linear_lif_net_f16_n_2 COMMAND rec_lif_net_f16_n_2)

add_executable(rec_lif_net_f16_n_4 rec_linear_lif/test_network_f16_n_4.cpp)
target_include_directories(rec_lif_net_f16_n_4 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rec_lif_net_f16_n_4 PRIVATE rts)
add_test(NAME rec_linear_lif_net_f16_n_4 COMMAND rec_lif_net_f16_n_4)
