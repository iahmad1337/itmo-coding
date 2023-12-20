#!/usr/bin/env sh

if [ ! -f ./CMakeLists.txt ]
then
    echo "This script should be launched from the project root"
    exit 1
fi

################################################################################
#                            Determining build type                            #
################################################################################

additional_opts="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
build_dir=""
if [ "$1" = "debug" ]
then
    additional_opts="$additional_opts -DCMAKE_BUILD_TYPE=Debug -DUSE_SANITIZERS=ON"
    build_dir="./debug"
elif [ "$1" = "release" ]
then
    additional_opts="$additional_opts -DCMAKE_BUILD_TYPE=Release"
    build_dir="./release"
else
    echo "Please pass the build type (debug or release)"
    exit 1
fi

################################################################################
#               Determining mechanism for fetching dependencies                #
################################################################################

if [ "$#" -lt 2 ]
then
    while true
    do
        read -p "Are you sure you want to continue without providing vcpkg root? (y/n)" answer
        case $answer in
            [Yy]* )
                echo "Proceeding with installation from urls"
                break;;
            [Nn]* )
                echo "Please relaunch the script and pass vcpkg root as the second argument. Aborting"
                exit 0
                break;;
            * ) echo "Please answer with 'y' or 'n'.";;
        esac
    done
fi

additional_opts="$additional_opts -DUSE_VCPKG=ON -DCMAKE_TOOLCHAIN_FILE=$2/scripts/buildsystems/vcpkg.cmake"

# NOTE: the second variable is not quoted because I want it to be split
cmake -S . -B "$build_dir" $additional_opts
cp "$build_dir/compile_commands.json" .
