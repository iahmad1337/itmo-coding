cmake_minimum_required(VERSION 3.14)
project(my_project)

set(CMAKE_CXX_STANDARD 20)

set(USE_CPPUTILS ON)
set(USE_GTEST ON)
set(USE_FMT OFF)
set(USE_RANGEV3 ON)
set(USE_RE2 OFF)
set(USE_JSON OFF)
set(USE_SPDLOG OFF)
set(USE_ARGPARSE OFF)
set(USE_ABSL ON)

include(cmake/ahmad1337_deps.cmake)

################################################################################
#                                   Targets                                    #
################################################################################

include(cmake/ahmad1337_utils.cmake)

add_basic_executable(NAME a SRCS a.cc)

link_to_all(
    TARGETS a
    DEPS cpputils::cpputils range-v3::range-v3 absl::strings absl::str_format
    absl::log GTest::GTest
)
