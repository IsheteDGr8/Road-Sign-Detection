cmake_minimum_required(VERSION 3.15)
project(SignDetector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Only dependency: OpenCV 4. Change this path if your install differs.
if(NOT OpenCV_DIR)
    set(OpenCV_DIR "C:/opencv/build")
endif()

# Locate OpenCV installation
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Grab all .cpp files in the src directory automatically
file(GLOB SOURCES "src/*.cpp")

# Build the executable
add_executable(SignDetector ${SOURCES})

# Link OpenCV to the executable
target_link_libraries(SignDetector ${OpenCV_LIBS})