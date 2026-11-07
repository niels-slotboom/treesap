set(SOURCES_EXAMPLE_01_WAVE_EQN
    main.cpp
)

set(HEADERS_EXAMPLE_01_WAVE_EQN
)


set_source_files_properties(${SOURCES_EXAMPLE_01_WAVE_EQN} PROPERTIES LANGUAGE CUDA)

add_executable(example-01-wave-eqn ${SOURCES_EXAMPLE_01_WAVE_EQN} ${HEADERS_EXAMPLE_01_WAVE_EQN})

target_include_directories(example-01-wave-eqn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(example-01-wave-eqn PRIVATE amrex treesap)