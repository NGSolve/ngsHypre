cmake_minimum_required(VERSION 3.16)

###############################################################################
# Edit here
###############################################################################

add_subdirectory(hypre/src)

# make sure the module_name matches the first argument of PYBIND11_MODULE
set(module_name ngsHypre)
set(source_files src/hypre_precond.cpp src/ngsHypre.cpp)

###############################################################################
# STOP editing here
###############################################################################

# Don't change this
project(${module_name})
include(ngsolve_addon.cmake)
