cmake_minimum_required(VERSION 3.16)

# change this to your desired addon name
# make sure, that the name matches exactly the argument of PYBIND11_MODULE()
set(addon_name ngs_hypre)

project(${addon_name})
include(ngsolve_addon.cmake)

# change the source file arguments to your source files
add_ngsolve_addon(${addon_name} src/hypre_precond.cpp src/ngsHypre.cpp)

# install the compiled python module and __init__.py ( don't change this )
install(TARGETS ${addon_name} DESTINATION ${addon_name})
install(FILES src/__init__.py DESTINATION ${addon_name})

# install additional python files/demos/examples
install(FILES demos/example1.py DESTINATION ${addon_name}/demos)

# generate stub files for autocomplete in IDEs
# this must be done at the very end (such that the stubgen generation happens after the python modules are installed)
install(CODE ${stubgen_generation_code})
install(DIRECTORY ${stubgen_directory} DESTINATION ${addon_name})

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

target_link_libraries(${addon_name} PRIVATE HYPRE)
set_target_properties(HYPRE PROPERTIES POSITION_INDEPENDENT_CODE ON)
