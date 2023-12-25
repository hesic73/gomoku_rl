cmake_minimum_required(VERSION 3.5)
project(gomoku-gui)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

find_package(Qt5 REQUIRED COMPONENTS Widgets)


# https://github.com/pytorch/pytorch/issues/12449
set(Torch_DIR "D:\\24_spring\\libtorch\\share\\cmake\\Torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(PROJECT_SOURCES
        main.cpp
        mainwindow.hpp
        board.hpp
        core.hpp
        model.hpp
        buttonpanel.hpp
        labeledcombobox.hpp
)

add_executable(gomoku-gui
            ${PROJECT_SOURCES}
        )

target_link_libraries(gomoku-gui PRIVATE Qt${QT_VERSION_MAJOR}::Widgets "${TORCH_LIBRARIES}")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set_property(TARGET gomoku-gui PROPERTY WIN32_EXECUTABLE true)
endif()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET gomoku-gui
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:gomoku-gui>)
endif (MSVC)

