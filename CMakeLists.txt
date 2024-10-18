# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(CoordinateTransform LANGUAGES Fortran)

set(CMAKE_Fortran_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/compilers/bin/nvfortran")

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cuda -gpu=cc89 -Minfo=accel")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.9/compilers/include")


# set(SOURCES
#     main.f90
#     grid_module.f90
#     transform_kernel.cuf
# )

set(SOURCES
    test_code.f90
)

set_source_files_properties(transform_kernel.cuf PROPERTIES LANGUAGE Fortran)

add_executable(coordinate_transform ${SOURCES})
