# I don't know if setting these policies is okay but I did it anyway to get rid
# of warnings
# https://cmake.org/cmake/help/latest/policy/CMP0077.html
cmake_policy(SET CMP0077 NEW)

# I found out that FetchContent_MakeAvailable does add_subdirectory by default
# (which I did manually for all of the libs anyway) so I will roll with it until
# I find any problems
include(FetchContent)

macro(AddUrlLib vcpkg_name url)
    if (USE_VCPKG)
        message("Using vcpkg version of ${vcpkg_name}")
        find_package(${vcpkg_name} REQUIRED)
    else()
        FetchContent_Declare(
            ${vcpkg_name}
            URL ${url}
            )
        message("Fetching ${vcpkg_name} from url...")
        FetchContent_MakeAvailable(${vcpkg_name})
        message("Done")
    endif()
endmacro()

if (USE_CPPUTILS)
    FetchContent_Declare(
        cpputils
        URL https://github.com/iahmad1337/cpputils/archive/refs/heads/main.zip
        )
    message("Fetching cpputils from url...")
    FetchContent_MakeAvailable(cpputils)
    message("Done")
endif()

if (USE_GTEST)
    AddUrlLib(
        gtest
        https://github.com/google/googletest/archive/refs/heads/main.zip
        )
endif()

if(USE_FMT)
    AddUrlLib(
        fmt
        https://github.com/fmtlib/fmt/archive/refs/heads/master.zip
        )
endif()

if(USE_RANGEV3)
    AddUrlLib(
        range-v3
        https://github.com/ericniebler/range-v3/archive/refs/heads/master.zip
        )
endif()

if(USE_RE2)
    AddUrlLib(
        re2
        https://github.com/google/re2/archive/refs/heads/main.zip
        )
endif()

if(USE_JSON)
    AddUrlLib(
        nlohmann-json
        https://github.com/nlohmann/json/archive/refs/heads/master.zip
        )
endif()

if(USE_SPDLOG)
    AddUrlLib(
        spdlog
        https://github.com/gabime/spdlog/archive/refs/heads/v1.x.zip
        )
endif()

if(USE_ARGPARSE)
    AddUrlLib(
        argparse
        https://github.com/p-ranav/argparse/archive/refs/heads/master.zip
    )
endif()

if (USE_ABSL)
    set(ABSL_PROPAGATE_CXX_STD ON)
    AddUrlLib(
        absl
        https://github.com/abseil/abseil-cpp/archive/refs/heads/master.zip
        )
endif()
