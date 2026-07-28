cmake_minimum_required(VERSION 3.24)

project(
    DeepTorch
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE NOVA_SOURCES
    src/*.cpp
)

add_library(nova STATIC
    ${NOVA_SOURCES}
)

target_include_directories(nova
    PUBLIC
        include
)

# Add this
find_package(GTest CONFIG REQUIRED)

enable_testing()

add_subdirectory(tests)