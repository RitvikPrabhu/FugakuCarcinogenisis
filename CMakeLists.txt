# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(run)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
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

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add source files
set(SOURCES
    src/main.cpp
    src/readFile.cpp
    src/fourHit.cpp
)

add_executable(run ${SOURCES})

# Link MPI libraries
target_link_libraries(run PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX)


