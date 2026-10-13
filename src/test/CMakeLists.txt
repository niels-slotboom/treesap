set(SOURCES_TEST
    main.cpp
)

set(HEADERS_TEST
)


set_source_files_properties(${SOURCES_TEST} PROPERTIES LANGUAGE CUDA)

add_executable(test ${SOURCES_TEST} ${HEADERS_TEST})

target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(test PRIVATE amrex treesap)