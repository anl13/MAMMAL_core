cmake_minimum_required(VERSION 3.8 FATAL_ERROR) 

set(CMAKE_CXX_FLAGS 
    "-O2 -g -std=c++11 "
)

include_directories("/usr/include/eigen3/") 

find_package(Threads REQUIRED)

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

)

## add lib: mce
FILE(GLOB MCE_FILES          associate/mce/*.h associate/mce/*.cpp) 
# add_definitions(-DRETURN_CLIQUES_ONE_BY_ONE)
add_library(mce
${MCE_FILES}
)

FILE(GLOB ASSOC_FILES        associate/*.h associate/*.cpp) 
FILE(GLOB RENDER_FILES       render/*.h render/*.cpp) 
FILE(GLOB RENDER_DEPS        render/deps/*.cpp render/deps/*.c) 
FILE(GLOB BUNDLE_FILES       bundle/*.cpp) 
FILE(GLOB SMAL_FILES         smal/*.cpp) 

## exe: bundle adjustment 
include_directories(bundle/)
add_executable(calib
    exec/calib_bundle.cpp 
    ${ASSOC_FILES}
    ${BUNDLE_FILES}
)
target_compile_features(calib PUBLIC cxx_std_11)
target_link_libraries(calib
${CMAKE_THREAD_LIBS_INIT}
${MY_OpenCV_LIBS}
${CERES_LIBRARIES}
# /usr/lib/x86_64-linux-gnu/libpthread.so
libgomp.so.1
${My_GFLAGS_LIBS}
${My_GLOG_LIBS}
${MY_Boost_LIBS}
mce
)

## exe: renderer :: test 
include_directories(render/) 
include_directories(render/deps/)
add_executable(test_render
    exec/test_render.cpp 
    ${RENDER_DEPS}
)
target_link_libraries(test_render
    ${OPENGL_gl_LIBRARY}
    ${GLUT_LIBRARIES}
    ${OPENGL_LIBRARY}
    glfw
    glad
    ${CMAKE_THREAD_LIBS_INIT}
)

## exe: renderer :: main func 
add_executable(render
    exec/render.cpp 
    ${RENDER_FILES}
    ${ASSOC_FILES}
    ${SMAL_FILES}
)
target_compile_definitions(render PUBLIC cxx_std_11) 
target_link_libraries(render
    ${CMAKE_THREAD_LIBS_INIT}
    ${MY_OpenCV_LIBS}
    ${CERES_LIBRARIES}
    # /usr/lib/x86_64-linux-gnu/libpthread.so
    libgomp.so.1
    ${My_GFLAGS_LIBS}
    ${My_GLOG_LIBS}
    ${MY_Boost_LIBS}
    ${OPENGL_gl_LIBRARY}
    ${GLUT_LIBRARIES}
    ${OPENGL_LIBRARY}
    glfw
    glad
    mce
)



## exe: association
include_directories(associate)
add_executable(associate
    exec/associate.cpp 
    ${ASSOC_FILES}
)
target_link_libraries(associate
mce
${CMAKE_THREAD_LIBS_INIT}
${MY_OpenCV_LIBS}
${CERES_LIBRARIES}
# /usr/lib/x86_64-linux-gnu/libpthread.so
libgomp.so.1
${My_GFLAGS_LIBS}
${My_GLOG_LIBS}
${MY_Boost_LIBS}
)


add_executable(test_json
exec/test.cpp
)