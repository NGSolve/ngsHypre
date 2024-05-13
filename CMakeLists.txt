cmake_minimum_required(VERSION 3.16)

###############################################################################
# Edit here
###############################################################################

# make sure the module_name matches the first argument of PYBIND11_MODULE
set(module_name ngs_hypre)
set(source_files src/hypre_precond.cpp src/ngsHypre.cpp)

###############################################################################
# STOP editing here
###############################################################################

# Don't change this
project(${module_name})
include(ngsolve_addon.cmake)

###############################################################################
# Fetch and link hypre

include(FetchContent)
FetchContent_Declare(hypre
  GIT_REPOSITORY https://github.com/hypre-space/hypre.git
  GIT_TAG master
)

FetchContent_GetProperties(hypre)
if(NOT hypre_POPULATED)
  message(STATUS "Fetching hypre ...")
  FetchContent_Populate(hypre)
  # Set EXCLUDE_FROM_ALL to avoid installing hypre files
  add_subdirectory(${hypre_SOURCE_DIR}/src ${hypre_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

target_link_libraries(${module_name} PRIVATE HYPRE)
set_target_properties(HYPRE PROPERTIES POSITION_INDEPENDENT_CODE ON)
