cmake_minimum_required(VERSION 3.4)
project(model_render)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

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

set(SOURCE_FILES main.cpp SnapImageDrawCallback.hxx SceneHandler.hxx Utils.hxx)
add_executable(model_render ${SOURCE_FILES})
target_link_libraries(model_render ${OPENGL_LIBRARIES} ${GLUT_LIBRARY}
        "-lOpenThreads -losg -losgUtil -losgText -losgDB -losgGA -losgViewer")