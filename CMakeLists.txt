cmake_minimum_required(VERSION 3.24)

project(
    DeepTorch
    VERSION 0.1.0
    DESCRIPTION "Modern C++ Deep Learning Framework"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(nova STATIC)

target_sources(nova PRIVATE)

target_include_directories(
    nova
    PUBLIC
    include
)