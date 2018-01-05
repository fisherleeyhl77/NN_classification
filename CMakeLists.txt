cmake_minimum_required(VERSION 2.8)
project( NNClassification )

# set compiler
set( CMAKE_CXX_COMPILER "g++")
add_compile_options(-std=c++11)

# Set the directory of binaries
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 

find_package( OpenCV 2.4.9 REQUIRED )

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SOURCES "src/*.cpp")
add_executable( nn_classification ${SOURCES} )
target_link_libraries( nn_classification ${OpenCV_LIBS} )
