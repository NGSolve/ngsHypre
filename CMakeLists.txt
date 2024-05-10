cmake_minimum_required(VERSION 3.16)

###############################################################################
# Edit here
###############################################################################

# make sure the module_name matches the first argument of PYBIND11_MODULE
set(module_name ngsHypre)
set(source_files src/hypre_precond.cpp src/ngsHypre.cpp)

include(ExternalProject)
ExternalProject_Add(hypre GIT_REPOSITORY https://github.com/hypre-space/hypre.git
                         GIT_TAG master
                         SOURCE_SUBDIR src
                         )

###############################################################################
# STOP editing here
###############################################################################

# Don't change this
project(${module_name})
include(ngsolve_addon.cmake)
