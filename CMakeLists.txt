cmake_minimum_required(VERSION 3.8 FATAL_ERROR) 
project(PigCapture C CXX)
set(CMAKE_CXX_FLAGS 
    "-O2 -g -std=c++11 "
)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/)

include_directories("/usr/include/eigen3/") 

find_package(Threads REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(/usr/local/include/opencv4)

# glogs and gflags
find_package(glog)
include_directories(
    ${GFLAGS_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIR}
)

# ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# json 
find_package(PkgConfig REQUIRED) 
pkg_check_modules(JSONCPP jsoncpp) 
include_directories(/usr/include/jsoncpp/)
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
FILE(GLOB MY_Boost_LIBS /usr/local/lib/libboost*.so.1.72.0)

## link libraries
link_directories(/usr/lib/x86_64-linux-gnu)
link_directories(/usr/local/lib)

## add lib: mce
FILE(GLOB MCE_FILES          associate/mce/*.h associate/mce/*.cpp) 
# add_definitions(-DRETURN_CLIQUES_ONE_BY_ONE)
add_library(mce
${MCE_FILES}
)

FILE(GLOB ASSOC_FILES        associate/*.h associate/*.cpp) 
FILE(GLOB RENDER_FILES       render/*.h render/*.cpp render/*.c) 
FILE(GLOB RENDER_DEPS        render/deps/*.cpp render/deps/*.c) 
FILE(GLOB BUNDLE_FILES       bundle/*.cpp) 
FILE(GLOB SMAL_FILES         smal/*.cpp) 
FILE(GLOB UTIL_FILES         utils/*.cpp) 

## exe: bundle adjustment 
include_directories(bundle/)
add_executable(calib
    exec/calib_bundle.cpp 
    ${BUNDLE_FILES}
    ${UTIL_FILES}
)
target_compile_features(calib PUBLIC cxx_std_11)
target_link_libraries(calib
${CMAKE_THREAD_LIBS_INIT}
${OpenCV_LIBS}
${CERES_LIBRARIES}
libgomp.so.1
${GFLAGS_LIBS}
${GLOG_LIBRARY}
${MY_Boost_LIBS}
mce
)

## exe: renderer :: main func 
add_executable(render
    exec/render.cpp 
    ${RENDER_FILES}
    ${ASSOC_FILES}
    ${SMAL_FILES}
    ${UTIL_FILES}
)
target_compile_definitions(render PUBLIC cxx_std_11) 
target_link_libraries(render
    ${CMAKE_THREAD_LIBS_INIT}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    libgomp.so.1
    ${GFLAGS_LIBS}
    ${GLOG_LIBRARY}
    ${MY_Boost_LIBS}
    ${OPENGL_gl_LIBRARY}
    ${GLUT_LIBRARIES}
    ${OPENGL_LIBRARY}
    glfw
    mce
)

## exe: association
include_directories(associate)
add_executable(associate
    exec/associate.cpp 
    ${ASSOC_FILES}
    ${UTIL_FILES}
    ${SMAL_FILES}
)
target_link_libraries(associate
mce
${CMAKE_THREAD_LIBS_INIT}
${OpenCV_LIBS}
${CERES_LIBRARIES}
libgomp.so.1
${GFLAGS_LIBS}
${GLOG_LIBRARY}
${MY_Boost_LIBS}
)

## exe: smal fitting test 
add_executable(smal
exec/smal_fitting.cpp
${UTIL_FILES}
${SMAL_FILES}
${RENDER_FILES}
${ASSOC_FILES}
)
target_link_libraries(smal 
${CERES_LIBRARIES}
${MY_Boost_LIBS}
${GFLAGS_LIBS}
${GLOG_LIBRARY}
${OpenCV_LIBS}
${OPENGL_gl_LIBRARY}
${GLUT_LIBRARIES}
${OPENGL_LIBRARY}
glfw
mce
)

## exe: smal fitting test 
add_executable(custom
exec/custom_pig.cpp
${UTIL_FILES}
smal/pigmodel.cpp 
${RENDER_FILES}
)
target_link_libraries(custom 
${CERES_LIBRARIES}
${MY_Boost_LIBS}
${GFLAGS_LIBS}
${GLOG_LIBRARY}
${OpenCV_LIBS}
${OPENGL_gl_LIBRARY}
${GLUT_LIBRARIES}
${OPENGL_LIBRARY}
glfw
)