cmake_minimum_required(VERSION 3.15)
project(Lectures)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -S")

#add_executable(Lectures main.cpp Matrix.cpp Matrix.h)
#add_executable(Lectures Matrix.cpp Matrix.h lecture/main_daxpy.cpp lecture/main_daxpy.h main.cpp)

add_executable(Lectures Matrix.cpp Matrix.h test/test.cpp test/test-main.cpp CSRMatrix.cpp CSRMatrix.h utilities.h)