#    Copyright (c) 2025 Au-Zone Technologies
#
#    Permission is hereby granted, free of charge, to any person obtaining a
#    copy of this software and associated documentation files (the "Software"),
#    to deal in the Software without restriction, including without limitation
#    the rights to use, copy, modify, merge, publish, distribute, sublicense,
#    and/or sell copies of the Software, and to permit persons to whom the
#    Software is furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in
#    all copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#    DEALINGS IN THE SOFTWARE.

# CameraAdaptor Test - Integration test for RGBA→RGB conversion with DMA-BUF

# Option to enable G2D support (requires libg2d)
option(ENABLE_G2D "Enable G2D hardware acceleration for image resize" ON)

# Find required packages
find_package(PkgConfig)
if(ENABLE_G2D)
    # G2D is typically not in pkg-config, check for header
    find_path(G2D_INCLUDE_DIR g2d.h
        PATHS /usr/include /opt/yocto-sdk-imx8mp-frdm-6.12.49-2.2.0/sysroots/armv8a-poky-linux/usr/include
    )
    find_library(G2D_LIBRARY g2d
        PATHS /usr/lib /opt/yocto-sdk-imx8mp-frdm-6.12.49-2.2.0/sysroots/armv8a-poky-linux/usr/lib
    )
    if(G2D_INCLUDE_DIR AND G2D_LIBRARY)
        message(STATUS "G2D found: ${G2D_LIBRARY}")
        set(G2D_FOUND TRUE)
    else()
        message(WARNING "G2D not found, using software resize fallback")
        set(G2D_FOUND FALSE)
    endif()
endif()

# Source files
set(CAMERA_ADAPTOR_TEST_SOURCES
    camera_adaptor_test.cc
    image_loader.cc
)

if(G2D_FOUND)
    list(APPEND CAMERA_ADAPTOR_TEST_SOURCES g2d_helper.cc)
endif()

# Create executable
add_executable(camera_adaptor_test ${CAMERA_ADAPTOR_TEST_SOURCES})

# Include directories
target_include_directories(camera_adaptor_test PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/camera_adaptor
)

if(G2D_FOUND)
    target_include_directories(camera_adaptor_test PRIVATE ${G2D_INCLUDE_DIR})
    target_compile_definitions(camera_adaptor_test PRIVATE HAVE_G2D=1)
    target_link_libraries(camera_adaptor_test ${G2D_LIBRARY})
endif()

# Link libraries
target_link_libraries(camera_adaptor_test
    ${VX_DELEGATE_TARGET}
    TensorFlow::tensorflow-lite
)

# Copy test data to build directory
file(GLOB TEST_DATA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/test_data/*")
if(TEST_DATA_FILES)
    file(COPY ${TEST_DATA_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/test_data/)
endif()
