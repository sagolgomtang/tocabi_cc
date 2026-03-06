cmake_minimum_required(VERSION 2.8.3)
project(tocabi_cc)

add_compile_options(-std=c++17 -Ofast)

find_package(catkin REQUIRED
roscpp
roslib
urdf
tocabi_lib
tocabi_msgs
eigen_conversions
)

find_package(casadi QUIET)
find_package(yaml-cpp REQUIRED)

catkin_package(
  INCLUDE_DIRS include /usr/local/include/onnxruntime
  LIBRARIES ${PROJECT_NAME} onnxruntime
  CATKIN_DEPENDS roslib urdf tocabi_lib tocabi_msgs
# DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

if(casadi_FOUND)
  add_definitions(-DTOCABI_CC_USE_CASADI)
  include_directories(${casadi_INCLUDE_DIRS})
endif()

add_library(${PROJECT_NAME}
  src/cc.cpp
)
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Link with catkin and ONNX Runtime libraries
target_include_directories(${PROJECT_NAME} PUBLIC 
  /usr/local/include/onnxruntime
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${catkin_LIBRARIES}
  onnxruntime
  yaml-cpp
)

if(casadi_FOUND)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${casadi_LIBRARIES})
endif()

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)
