#!/usr/bin/env bash

set -e

err() {
    echo "$1"
    exit 1
}

if [ ! -d build ]; then
    mkdir build
    pushd build
    cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cp compile_commands.json ..
    popd
fi

test_bin() {
    echo "Running $1..."
    ./$1 && echo "**** $1: SUCCESS ****" || err "**** $1: FAILURE ****"
}

(
    cd build
    cmake --build .

    test_bin test-a
) && echo SUCCESS || echo FAILURE

