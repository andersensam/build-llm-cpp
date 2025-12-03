cmake_minimum_required(VERSION 3.10.0)

project(build-llm-cpp CXX)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -Wextra -Wpedantic -g -fsanitize=address,undefined")

include_directories(../src/include)

add_library(Log ../src/Log.cpp)
add_library(Tokenizer ../src/Tokenizer.cpp)

target_link_libraries(Tokenizer Log)
 
add_executable(build-llm-cpp
    ../src/main.cpp)

target_link_libraries(build-llm-cpp Log)
target_link_libraries(build-llm-cpp Tokenizer)
