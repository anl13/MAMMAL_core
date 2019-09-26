cmake_minimum_required(VERSION 3.8 FATAL_ERROR) 

set(CMAKE_CXX_FLAGS 
    "-O2 -g -std=c++11 "
)

include_directories("/usr/include/eigen3/") 


include_directories(/usr/local/include)
FILE(GLOB MY_OpenCV_LIBS /usr/local/lib/libopencv_*.so.3.4)

# glogs and gflags
include_directories(
    ${GFLAGS_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIR}
)
FILE(GLOB My_GFLAGS_LIBS /usr/local/lib/libgflags.so*) 
FILE(GLOB My_GLOG_LIBS /usr/local/lib/libglog.so*)

# ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# json 
find_package(PkgConfig REQUIRED) 
pkg_check_modules(JSONCPP jsoncpp) 
link_libraries(${JSONCPP_LIBRARIES})

# GL 
set(OpenGL_GL_PREFERENCE LEGACY) 
find_package(OpenGL) 
find_package(GLUT) 
find_package(glfw3 REQUIRED) 
include_directories(${OPENGL_INCLUDE_DIR})

# Boost 
find_package(Boost) 
include_directories(${Boost_INCLUDE_DIR})
FILE(GLOB MY_Boost_LIBS /usr/local/lib/libboost*.so.1.64.0)

## link libraries
link_directories(/usr/lib/x86_64-linux-gnu)
link_libraries(
    ${MY_OpenCV_LIBS}
    ${CERES_LIBRARIES}
    /usr/lib/x86_64-linux-gnu/libpthread.so
    libgomp.so.1
    ${My_GFLAGS_LIBS}
    ${My_GLOG_LIBS}
    ${MY_Boost_LIBS}
)

## exe: bundle adjustment 
include_directories(bundle/)

add_executable(calib
    bundle/main.cpp 
    bundle/BASolver.cpp 
    bundle/calibration.cpp 
    associate/camera.cpp 
    associate/geometry.cpp 
    associate/math_utils.cpp 
    associate/image_utils.cpp 
)
target_compile_features(calib PUBLIC cxx_std_11)
target_link_libraries(calib)

## exe: renderer 
include_directories(render/) 
add_executable(render 
    render/test.cpp 
    render/renderer.cpp 
    render/render_object.cpp 
    render/shader.cpp 
    render/camviewer.cpp 
)
target_compile_definitions(render PUBLIC cxx_std_11) 
target_link_libraries(render
    ${OPENGL_gl_LIBRARY}
    ${GLUT_LIBRARIES}
    ${OPENGL_LIBRARY}
    glfw
    glad
)

## add lib: mce
FILE(GLOB MCE_FILES          associate/mce/*.h associate/mce/*.cpp) 
# add_definitions(-DRETURN_CLIQUES_ONE_BY_ONE)
add_library(mce
${MCE_FILES}
)

## exe: association
include_directories(associate)
add_executable(associate
    associate/main.cpp 
    associate/camera.cpp 
    associate/framedata.cpp
    associate/math_utils.cpp 
    associate/image_utils.cpp
    associate/geometry.cpp
    associate/clusterclique.cpp 
)
target_link_libraries(associate
mce
)


## exe: sync 
include_directories(sync)
add_executable(sync 
sync/main.cpp 
sync/extract_utils.cpp 
)
target_link_libraries(sync) 