cmake_minimum_required(VERSION 3.4)
project(gendata)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -I/usr/include/ImageMagick -I/usr/local/include -Iinclude")

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})
if(NOT GLUT_FOUND)
    message(ERROR "GLUT not found!")
endif(NOT GLUT_FOUND)

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR "OpenGL not found!")
endif(NOT OPENGL_FOUND)

set(SOURCE_FILES main.cxx SceneLoader.hxx Utils.hxx ImageGenerator.hxx ModelImage.hxx)
add_executable(gendata ${SOURCE_FILES})
target_link_libraries(gendata ${OPENGL_LIBRARIES} ${GLUT_LIBRARY}
        "-lOpenThreads -losg -losgUtil -losgText -losgDB -losgGA -losgViewer -lMagick++ -lMagickCore -ltinyxml2 -ljsoncpp  -L/usr/local/lib/x86_64-linux-gnu -Llib")