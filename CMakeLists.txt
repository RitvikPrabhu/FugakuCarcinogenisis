# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(run)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(NUMHITS "4" CACHE STRING "Number of hits in the pattern (e.g. 4, 5, 6...)")
set(CHUNK_SIZE "102400" CACHE STRING "Chunk size used in code")

add_definitions(-DNUMHITS=${NUMHITS})
add_definitions(-DCHUNK_SIZE=${CHUNK_SIZE})

# Option to enable timing (if needed in the future)
option(ENABLE_PROFILE "Enable profiling" ON)
if(ENABLE_PROFILE)
  add_definitions(-DENABLE_PROFILE)
endif()

option(USE_CPP_SET "Use C++ Set implementation" OFF)
if(USE_CPP_SET)
    add_definitions(-DUSE_CPP_SET)
endif()

option(USE_SVE_BIT "Use C++ Set implementation" OFF)
if(USE_SVE_BIT)
    add_definitions(-DUSE_SVE_BIT)
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


