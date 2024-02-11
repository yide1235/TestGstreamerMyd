cmake_minimum_required(VERSION 3.10)

project(TestGstreamer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories for CUDA and GStreamer
include_directories(
    ${CUDA_INCLUDE_DIRS}
    /usr/include/gstreamer-1.0
    /usr/include/glib-2.0
    /usr/lib/x86_64-linux-gnu/glib-2.0/include
)

# Set CUDA sources and C++ sources
set(CUDA_SOURCES TestGstreamer.cu)
set(CPP_SOURCES GstreamerReader.cpp GstreamerWriter.cpp)

# Specify target executable and source files
cuda_add_executable(${PROJECT_NAME} ${CUDA_SOURCES} ${CPP_SOURCES})

# Find PkgConfig for finding GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
    ${GSTREAMER_LIBRARIES}
    gstreamer-1.0
    gobject-2.0
    glib-2.0
)

# RPATH settings for finding shared libraries at runtime
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "/usr/local/lib;/usr/lib/x86_64-linux-gnu")
set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
