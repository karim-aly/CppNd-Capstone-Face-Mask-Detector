cmake_minimum_required(VERSION 3.10)
project(face_mask_detect)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow/lib)

set(CMAKE_CXX_STANDARD 17)

add_executable(face_mask_detect main.cpp ./src/Model.cpp ./src/Tensor.cpp ./src/FaceMaskDetector.cpp ./src/BoundingBoxesWindow.cpp)
find_package( OpenCV REQUIRED )
target_include_directories(face_mask_detect PRIVATE ./include $ENV{HOME}/libtensorflow/include)
target_link_libraries (face_mask_detect "${TENSORFLOW_LIB}" ${OpenCV_LIBS})
