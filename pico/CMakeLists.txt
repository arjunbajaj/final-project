# |-------------------------------------|
# |       [CM3070] FINAL PROJECT        |
# |    CLASSIFYING SATELLITE IMAGERY    |
# |    BSc CS - UNIVERSITY OF LONDON    |
# |             ARJUN BAJAJ             |
# |-------------------------------------|

cmake_minimum_required(VERSION 3.12)

# Automatic Detection of Pico SDK Path
include(pico_sdk_import.cmake)

# The project's name is `diode`
project(diode C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

# Initialize the Pico SDK
pico_sdk_init()

# Add the code and model files.
add_executable(diode
  ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/eurosat.h
)

# Add the TFLite library so that includes are possible
add_subdirectory("lib/pico-tflmicro" EXCLUDE_FROM_ALL)

# Link the Pico Standard Library, Pico Time, and TFLite libraries.
target_link_libraries(diode pico_time pico_stdlib pico-tflmicro)

# Enable UART over USB, and disable UART over GPIO Pins
pico_enable_stdio_usb(diode 1)
pico_enable_stdio_uart(diode 0)

# Create a UF2 file which can be flashed onto the Pico easily
pico_add_extra_outputs(diode)
