cmake_minimum_required(VERSION 3.15)
project(Assignment)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -S")

#use Open BLAS library for optimizations
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -lopenblas -lpthread")

#enable_language(Fortran)
#set(FC = gfortran)

set(LDFLAGS "-L/usr/local/opt/openblas/lib")
set(CPPFLAGS "-I/usr/local/opt/openblas/include")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS} ${LDFLAGS} ${CPPFLAGS}" )

add_executable(Assignment Matrix.cpp Matrix.h test/test.cpp  CSRMatrix.cpp CSRMatrix.h utilities.h main.cpp)