cmake_minimum_required(VERSION 3.23)
project(SSAToggle)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(src)

add_executable(SSAToggle
        src/SSAToggle.cpp
        src/SSAToggle.h)
target_link_libraries(SSAToggle PRIVATE "${OpenMP_CXX_FLAGS}")
target_compile_options(SSAToggle PRIVATE "${OpenMP_CXX_FLAGS}")