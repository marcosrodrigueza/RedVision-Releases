cmake_minimum_required(VERSION 2.8)

project(Red-Vision)

## Compile as C++11
add_compile_options(-std=c++11)


set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/")


find_package(OpenCV 4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable (${PROJECT_NAME} main.cpp)

target_link_libraries (${PROJECT_NAME} ${OpenCV_LIBRARIES})