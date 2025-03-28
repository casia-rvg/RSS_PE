cmake_minimum_required(VERSION 3.10)
project(RSS_PE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
    -O3 \
    -march=native \
    -funroll-loops \
    -ffast-math \
    -funsafe-math-optimizations \
    -msse2 \
    -DNDEBUG")

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(PCL 1.7 REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
)

add_executable(rss_pe
        main.cpp
        src/PlaneExtractor.cpp
        src/ConfigManager.cpp
        src/triangle.cpp
)

target_link_libraries(rss_pe
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
)
