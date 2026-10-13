add_library(treesap INTERFACE)
add_library(treesap::treesap ALIAS treesap)

target_include_directories(treesap INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(treesap INTERFACE AMReX::amrex)

target_compile_features(treesap INTERFACE cxx_std_20)