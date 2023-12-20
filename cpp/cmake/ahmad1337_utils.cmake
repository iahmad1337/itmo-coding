include(CMakeParseArguments)

function(add_basic_executable)
    cmake_parse_arguments(
        PARSED_ARGS # prefix of output variables
        "" # list of names of the boolean arguments (only defined ones will be true)
        "NAME" # list of names of mono-valued arguments
        "SRCS" # list of names of multi-valued arguments (output variables are lists)
        ${ARGN} # arguments of the function to parse, here we take the all original ones
    )

    set(name "${PARSED_ARGS_NAME}")

    add_executable("${name}" ${PARSED_ARGS_SRCS})
    target_compile_options("${name}" PRIVATE -Wall -Wextra -Wshadow=compatible-local -Wno-sign-compare -pedantic)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      message(STATUS "Enabling libc++ for ${name}...")
      target_compile_options("${name}" PUBLIC -stdlib=libc++)
      target_link_options("${name}" PUBLIC -stdlib=libc++)
    endif()

    if (USE_SANITIZERS)
      message(STATUS "Enabling sanitizers for ${name}...")
      # NOTE: leak sanitizer doesn't work on MacOS
      if (CMAKE_SYSTEM MATCHES "Darwin")
          set(COMPILE_OPTS PUBLIC -fsanitize=address,undefined -fno-sanitize-recover=all)
          set(LINK_OPTS PUBLIC -fsanitize=address,undefined)
      else()
          set(COMPILE_OPTS PUBLIC -fsanitize=address,undefined,leak -fno-sanitize-recover=all)
          set(LINK_OPTS PUBLIC -fsanitize=address,undefined,leak)
      endif()
      target_compile_options("${name}" PUBLIC ${COMPILE_OPTS})
      target_link_options("${name}" PUBLIC ${LINK_OPTS})
    endif()

    if (CMAKE_BUILD_TYPE MATCHES "Debug")
      message(STATUS "Enabling debugging compile options for ${name}...")
      set(DEBUG_COMPILE_OPTS PUBLIC -g -fno-omit-frame-pointer)
      # NOTE: -DGLIBCXX_DEBUG doesn't work with range-v3
      # set(DEBUG_COMPILE_OPTS PUBLIC -g -fno-omit-frame-pointer -D_GLIBCXX_DEBUG)
      # NOTE: target_compile_options appends the options to the target, i.e. it
      # doesn't rewrite existing options:
      # https://cmake.org/cmake/help/latest/command/target_compile_options.html
      target_compile_options("${name}" PUBLIC ${DEBUG_COMPILE_OPTS})
    endif()
endfunction()

function(link_to_all)
    cmake_parse_arguments(
        PARSED_ARGS # prefix of output variables
        "" # list of names of the boolean arguments (only defined ones will be true)
        "" # list of names of mono-valued arguments
        "TARGETS;DEPS" # list of names of multi-valued arguments (output variables are lists)
        ${ARGN} # arguments of the function to parse, here we take the all original ones
    )
    message(STATUS "Linking libs ${PARSED_ARGS_DEPS} to targets ${PARSED_ARGS_TARGETS}")
    foreach(target ${PARSED_ARGS_TARGETS})
        target_link_libraries("${target}" ${PARSED_ARGS_DEPS})
    endforeach()
endfunction()
