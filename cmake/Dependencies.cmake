include(FetchContent)

option(ADAPPARABOLICFEM_FETCH_EIGEN
       "Download Eigen if no compatible system installation is found."
       ON)
option(ADAPPARABOLICFEM_USE_OPENMP
       "Link OpenMP when it is available."
       ON)
option(ENABLE_MKL_PARDISO
       "Enable Intel oneMKL Pardiso support through Eigen's PARDISO wrappers."
       OFF)

# ---------------------------
# OpenMP
# ---------------------------
if(ADAPPARABOLICFEM_USE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "Using OpenMP: ${OpenMP_CXX_VERSION}")
    else()
        message(STATUS "OpenMP not found, continuing without it.")
    endif()
endif()

# ---------------------------
# Intel oneMKL / Pardiso (optional)
# ---------------------------
if(ENABLE_MKL_PARDISO)
    message(STATUS "MKL Pardiso support requested (ENABLE_MKL_PARDISO=ON).")

    set(_adapparabolicfem_user_mkl_root "")
    if(DEFINED MKL_ROOT AND NOT MKL_ROOT)
        unset(MKL_ROOT CACHE)
    endif()

    if(DEFINED MKL_ROOT AND MKL_ROOT)
        set(_adapparabolicfem_user_mkl_root "${MKL_ROOT}")
    endif()

    if(NOT DEFINED MKL_LINK)
        set(MKL_LINK "dynamic" CACHE STRING
            "oneMKL link model used by MKLConfig.cmake.")
    endif()

    if(NOT DEFINED MKL_INTERFACE)
        # The Eigen backend currently uses 32-bit sparse indices.
        set(MKL_INTERFACE "lp64" CACHE STRING
            "oneMKL integer interface used by MKLConfig.cmake.")
    endif()

    if(NOT DEFINED MKL_THREADING)
        if(OpenMP_CXX_FOUND)
            set(_adapparabolicfem_default_mkl_threading "gnu_thread")
        else()
            set(_adapparabolicfem_default_mkl_threading "sequential")
        endif()

        set(MKL_THREADING "${_adapparabolicfem_default_mkl_threading}" CACHE STRING
            "oneMKL threading layer used by MKLConfig.cmake.")
        unset(_adapparabolicfem_default_mkl_threading)
    endif()

    set_property(CACHE MKL_LINK PROPERTY STRINGS dynamic static sdl)
    set_property(CACHE MKL_INTERFACE PROPERTY STRINGS lp64 ilp64)
    set_property(
        CACHE MKL_THREADING PROPERTY STRINGS sequential gnu_thread intel_thread tbb_thread)

    if(_adapparabolicfem_user_mkl_root)
        if(EXISTS "${_adapparabolicfem_user_mkl_root}/MKLConfig.cmake")
            get_filename_component(
                _adapparabolicfem_normalized_mkl_root
                "${_adapparabolicfem_user_mkl_root}/../../../"
                ABSOLUTE)
        else()
            set(
                _adapparabolicfem_normalized_mkl_root
                "${_adapparabolicfem_user_mkl_root}")
        endif()

        # MKLConfig.cmake reuses MKL_ROOT internally, so normalize it before discovery.
        set(MKL_ROOT "${_adapparabolicfem_normalized_mkl_root}")
    endif()

    # Try standard CMake package discovery first, then explicit user/common WSL hints.
    find_package(MKL CONFIG QUIET)

    if(NOT TARGET MKL::MKL)
        set(_adapparabolicfem_mkl_hints "")

        if(_adapparabolicfem_user_mkl_root)
            list(APPEND _adapparabolicfem_mkl_hints
                "${_adapparabolicfem_normalized_mkl_root}"
                "${_adapparabolicfem_normalized_mkl_root}/lib/cmake/mkl")
        endif()

        list(APPEND _adapparabolicfem_mkl_hints
            "/opt/intel/oneapi/mkl/latest"
            "/opt/intel/oneapi/mkl/latest/lib/cmake/mkl")

        file(GLOB _adapparabolicfem_mkl_versioned_roots
            LIST_DIRECTORIES true
            "/opt/intel/oneapi/mkl/[0-9]*")
        foreach(_adapparabolicfem_mkl_root IN LISTS _adapparabolicfem_mkl_versioned_roots)
            list(APPEND _adapparabolicfem_mkl_hints
                "${_adapparabolicfem_mkl_root}"
                "${_adapparabolicfem_mkl_root}/lib/cmake/mkl")
        endforeach()

        list(REMOVE_DUPLICATES _adapparabolicfem_mkl_hints)
        find_package(MKL CONFIG QUIET HINTS ${_adapparabolicfem_mkl_hints})

        unset(_adapparabolicfem_mkl_hints)
        unset(_adapparabolicfem_normalized_mkl_root)
        unset(_adapparabolicfem_mkl_root)
        unset(_adapparabolicfem_mkl_versioned_roots)
    endif()

    if(TARGET MKL::MKL)
        message(
            STATUS
            "Using Intel oneMKL for Pardiso support (MKL_DIR=${MKL_DIR}, "
            "MKL_INTERFACE=${MKL_INTERFACE}, MKL_THREADING=${MKL_THREADING}, MKL_LINK=${MKL_LINK}).")
    else()
        message(FATAL_ERROR
            "ENABLE_MKL_PARDISO=ON, but Intel oneMKL was not found.\n"
            "CMake first tried standard package discovery and then common WSL paths such as\n"
            "  /opt/intel/oneapi/mkl/latest\n"
            "Please point CMake to the directory that contains MKLConfig.cmake via either:\n"
            "  -DMKL_ROOT=/opt/intel/oneapi/mkl/latest\n"
            "or:\n"
            "  -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/mkl/latest\n"
            "You can also point CMAKE_PREFIX_PATH directly at the config directory, for example:\n"
            "  -DCMAKE_PREFIX_PATH=/opt/intel/oneapi/mkl/latest/lib/cmake/mkl")
    endif()

    unset(_adapparabolicfem_user_mkl_root)
else()
    message(STATUS "MKL Pardiso support disabled (ENABLE_MKL_PARDISO=OFF).")
endif()

# ---------------------------
# Eigen (hybrid mode: system first, FetchContent fallback)
# ---------------------------
find_package(Eigen3 3.4 QUIET NO_MODULE)

if(TARGET Eigen3::Eigen)
    message(STATUS "Using system Eigen via find_package(Eigen3).")
else()
    find_path(EIGEN3_INCLUDE_DIR
        NAMES Eigen/Core
        PATHS
            /usr/include
            /usr/local/include
            /opt/homebrew/include
            /opt/local/include
        PATH_SUFFIXES eigen3
    )

    if(EIGEN3_INCLUDE_DIR)
        message(STATUS "Using Eigen headers from: ${EIGEN3_INCLUDE_DIR}")

        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    elseif(ADAPPARABOLICFEM_FETCH_EIGEN)
        message(STATUS "System Eigen not found, downloading Eigen 3.4.0...")

        FetchContent_Declare(
            eigen
            URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
        )

        FetchContent_GetProperties(eigen)
        if(NOT eigen_POPULATED)
            FetchContent_Populate(eigen)
        endif()

        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
        )
        message(STATUS "Using downloaded Eigen: ${eigen_SOURCE_DIR}")
    else()
        message(FATAL_ERROR
            "Eigen3 was not found.\n"
            "Install a system Eigen package (for example libeigen3-dev), or\n"
            "reconfigure with ADAPPARABOLICFEM_FETCH_EIGEN=ON to allow a download.")
    endif()
endif()
