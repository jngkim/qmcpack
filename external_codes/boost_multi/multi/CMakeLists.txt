cmake_minimum_required(VERSION 3.11) # for reference: Ubuntu 20.04 ships cmake 3.16.3, Fedora 31 ships cmake 3.17.4, (Gitlab CI) debian:stable-backports ships cmake 3.13.4, debian:testing ships 3.18.4
message("CMake version: ${CMAKE_VERSION}")
message("current build directory: ${CMAKE_CURRENT_BINARY_DIR}")

project(multi VERSION 0.76.0
	DESCRIPTION "A header only C++ library that provides multidimensional array access to contiguous or regularly contiguous memory (or ranges)."
	HOMEPAGE_URL "https://gitlab.com/correaa/boost-multi"
	LANGUAGES CXX)

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED True)
#set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(
	${PROJECT_NAME}
	INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
	          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_14)

find_program(MEMORYCHECK_COMMAND valgrind)
set(VALGRIND_COMMAND_OPTIONS    "-q --tool=memcheck --leak-check=yes --num-callers=51 --trace-children=yes --leak-check=full --track-origins=yes --gen-suppressions=all")
set(MEMORYCHECK_COMMAND_OPTIONS "-q --tool=memcheck --leak-check=yes --num-callers=52 --trace-children=yes --leak-check=full --track-origins=yes --gen-suppressions=all")  # must go before `include(CTest)`
set(MEMORYCHECK_SUPPRESSIONS_FILE "${PROJECT_SOURCE_DIR}/.valgrind-suppressions") # must go before `include(CTest)`

list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")  # before enable_testing()

include(CTest)

enable_testing()

add_subdirectory(test)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY SameMajorVersion)

#configure_package_config_file(
#  "${PROJECT_SOURCE_DIR}/cmake/multiConfig.cmake.in"
#  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
#  INSTALL_DESTINATION
#  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE boost::multi::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

#install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
#              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
#        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

#install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/SI DESTINATION include)
install(FILES ${PROJECT_SOURCE_DIR}/array_ref.hpp ${PROJECT_SOURCE_DIR}/array.hpp DESTINATION include/multi)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/detail DESTINATION include/multi)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/adaptors DESTINATION include/multi)

add_subdirectory(adaptors/blas)
add_subdirectory(adaptors/fftw)
add_subdirectory(adaptors/cuda)

#to install this project:
#cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$HOME
#cmake --build . --config Release --target test --target install -- -j $(nproc)

# to use this project do
#project("Your project")
#find_package(boost-multi CONFIG REQUIRED)
#add_executable(${PROJECT_NAME} src/your_main.cpp)
#target_link_libraries(${PROJECT_NAME} boost-multi::boost-multi)



