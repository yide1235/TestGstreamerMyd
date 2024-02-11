cmake_minimum_required(VERSION 3.10)

project(TestGstreamer)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)


find_package(CUDA REQUIRED)


find_package(OpenCV REQUIRED)


find_package(PkgConfig)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)


include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    /usr/include/glib-2.0
    /usr/lib/x86_64-linux-gnu/glib-2.0/include
)


set(CUDA_SOURCES TestGstreamer.cu)
set(CPP_SOURCES GstreamerReaderRAW.cpp VideoWriterRaw.cpp)


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=sm_75)


cuda_add_library(cuda_lib ${CUDA_SOURCES})


add_executable(${PROJECT_NAME} ${CPP_SOURCES})


target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    cuda_lib
    gstreamer-1.0
    gobject-2.0
    glib-2.0
)


set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "/usr/local/lib;/usr/lib/x86_64-linux-gnu")
set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)