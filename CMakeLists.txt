# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(GraphSparsity)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Option to enable timing (if needed in the future)
option(ENABLE_TIMING "Enable timing functionality" ON)
if(ENABLE_TIMING)
    add_definitions(-DENABLE_TIMING)
endif()

# Include directories
include_directories(include)

# Find MPI
find_package(MPI REQUIRED)
if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
endif()

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add source files
set(SOURCES
    src/main.cpp
    src/utils.cpp
    src/mpi_specific.cpp
    src/four_hit.cpp
)

add_executable(run ${SOURCES})

# Link MPI libraries
target_link_libraries(run ${MPI_LIBRARIES})

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(run OpenMP::OpenMP_CXX)
endif()

# Set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

