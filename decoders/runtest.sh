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

test_a() {
    test_bin test-a  || err "Unit tests on A failed"

    for input_file in ../test/a-input*; do
        cp "$input_file" ./input.txt
        echo "testing on $input_file"
        test_bin a
        echo "output.txt:"
        cat output.txt
    done || err "E2E test failed on A"
}

(
    cd build
    cmake --build . || err "Build failed!"

    test_a
) && echo SUCCESS || echo FAILURE

