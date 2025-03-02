cmake_minimum_required(VERSION 3.21)
project(static-keys-userspace LANGUAGES CXX)
# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
include_directories(include)
# set(REAL_FLAGS "-pipe -Wall -Wextra -Werror=multichar -Wno-deprecated -Wno-unused-parameter -pthread -fPIC -Woverloaded-virtual -Wnon-virtual-dtor -Werror -Wno-unkn

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-inline-functions -O1 -fsanitize=address")

add_executable(a src/a.cc)

###########
#  Tests  #
###########


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(test-a test/test-a.cc)
target_include_directories(test-a PRIVATE src)
target_link_libraries(test-a GTest::gtest_main GTest::gmock)

set_target_properties(test-a
    PROPERTIES
    CXX_STANDARD 20
)


###################
#  Miscellaneous  #
###################

function(emit_disasm target)
    set(output ${CMAKE_SOURCE_DIR}/${target}.asm )
    add_custom_command(
        TARGET ${target} POST_BUILD
        BYPRODUCTS ${output}
        # remove -S to get rid of source lines
        COMMAND objdump --disassemble -S -Mintel ${CMAKE_BINARY_DIR}/${target} | c++filt >${output}
        COMMENT "Generating ${output}..."
    )
endfunction()
emit_disasm(a)
