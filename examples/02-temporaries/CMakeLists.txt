set(SOURCES_EXAMPLE_02_TEMPORARIES
    main.cpp
)

set(HEADERS_EXAMPLE_02_TEMPORARIES
    NonLinearRHS.hpp
)


set_source_files_properties(${SOURCES_EXAMPLE_02_TEMPORARIES} PROPERTIES LANGUAGE CUDA)

add_executable(example-02-temporaries ${SOURCES_EXAMPLE_02_TEMPORARIES} ${HEADERS_EXAMPLE_02_TEMPORARIES})

target_include_directories(example-02-temporaries PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(example-02-temporaries PRIVATE amrex treesap)