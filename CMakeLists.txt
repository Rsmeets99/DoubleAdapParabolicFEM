cmake_minimum_required(VERSION 3.16)
project(AdapParabolicFEM VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------
# C++ standard
# ---------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------
# Production build options
# ---------------------------
# This publication snapshot builds the Snellius production runner only.

# ---------------------------
# Dependencies
# ---------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(Dependencies)

# ---------------------------
# Header-only library (INTERFACE)
# ---------------------------
add_library(adap_parabolic_fem INTERFACE)
add_library(adap_parabolic_fem::adap_parabolic_fem ALIAS adap_parabolic_fem)

set(
    ADAPPARABOLICFEM_COMPILE_OPTIONS
    ""
    CACHE STRING
    "Semicolon-separated extra compile options applied to adap_parabolic_fem consumers.")
set(
    ADAPPARABOLICFEM_LINK_OPTIONS
    ""
    CACHE STRING
    "Semicolon-separated extra link options applied to adap_parabolic_fem consumers.")

target_include_directories(
    adap_parabolic_fem
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(adap_parabolic_fem INTERFACE cxx_std_20)
target_compile_definitions(
    adap_parabolic_fem
    INTERFACE
        ADAPPARABOLICFEM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        ADAPPARABOLICFEM_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(adap_parabolic_fem INTERFACE Eigen3::Eigen)

if(ADAPPARABOLICFEM_COMPILE_OPTIONS)
    target_compile_options(adap_parabolic_fem INTERFACE ${ADAPPARABOLICFEM_COMPILE_OPTIONS})
endif()

if(ADAPPARABOLICFEM_LINK_OPTIONS)
    target_link_options(adap_parabolic_fem INTERFACE ${ADAPPARABOLICFEM_LINK_OPTIONS})
endif()

if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(adap_parabolic_fem INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(adap_parabolic_fem INTERFACE ADAPPARABOLICFEM_HAS_OPENMP=1)
else()
    target_compile_definitions(adap_parabolic_fem INTERFACE ADAPPARABOLICFEM_HAS_OPENMP=0)
endif()

if(TARGET MKL::MKL)
    target_link_libraries(adap_parabolic_fem INTERFACE MKL::MKL)
    target_compile_definitions(
        adap_parabolic_fem
        INTERFACE
            ADAPPARABOLICFEM_HAVE_MKL_PARDISO=1)
else()
    target_compile_definitions(
        adap_parabolic_fem
        INTERFACE
            ADAPPARABOLICFEM_HAVE_MKL_PARDISO=0)
endif()

# ---------------------------
# Algorithm runners
# ---------------------------
add_executable(
    run_adaptive_algorithm
    algorithm/runners/run_adaptive_algorithm.cpp
    algorithm/runners/runner_case_p1_dim1.cpp
    algorithm/runners/runner_case_p1_dim2.cpp
    algorithm/runners/runner_case_p2_dim1.cpp
    algorithm/runners/runner_case_p2_dim2.cpp
    algorithm/runners/runner_case_p3_dim1.cpp
    algorithm/runners/runner_case_p3_dim2.cpp
    algorithm/runners/runner_case_p4_dim1.cpp
    algorithm/runners/runner_case_p4_dim2.cpp)
target_link_libraries(run_adaptive_algorithm PRIVATE adap_parabolic_fem)
set_target_properties(run_adaptive_algorithm PROPERTIES FOLDER "algorithm")
