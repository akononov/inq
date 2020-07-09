include_directories(.)
include_directories(../external_libs)
include_directories(../src)

link_libraries(inq)

add_executable(electron_gas_non_interacting electron_gas_non_interacting.${SRC_EXT}) 
add_executable(electron_gas_lda_pz electron_gas_lda_pz.${SRC_EXT})
add_executable(hydrogen_local hydrogen_local.${SRC_EXT})
add_executable(hydrogen_non_local hydrogen_non_local.${SRC_EXT})
add_executable(neon neon.${SRC_EXT})
add_executable(kripton kripton.${SRC_EXT})
add_executable(nitrogen nitrogen.${SRC_EXT})
add_executable(silicon silicon.${SRC_EXT})
add_executable(h2o_ground_state h2o_ground_state.${SRC_EXT})
add_executable(h2o_real_time h2o_real_time.${SRC_EXT})