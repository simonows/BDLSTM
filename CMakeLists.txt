cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(BDLSTM LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_subdirectory(src)

