#!/bin/bash
set -e

# see: https://developer.apple.com/metal/cpp/
# set the url to the correct version for your OS
# retrieve your macOS version with `sw_vers -productVersion`
METAL_CPP_URL="https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
DEST_DIR="metal-cpp"

if [ -d "$DEST_DIR" ]; then
    echo "metal-cpp already exists in $DEST_DIR. Skipping download."
else
    curl -L -o metal-cpp.zip "$METAL_CPP_URL"
    unzip -q metal-cpp.zip
    rm metal-cpp.zip
fi

echo "compiling: MSL -> LLVM IR"
xcrun -sdk macosx metal -S -emit-llvm shader.metal -o shader.ll

echo "compiling: LLVM IR -> AIR"
xcrun -sdk macosx metal -c shader.ll -o shader.air

echo "compiling: AIR -> metallib"
xcrun -sdk macosx metallib shader.air -o shader.metallib

echo "compiling: harness.cpp"
# also need to link against Metal, Foundation, and QuartzCore frameworks.
clang++ -std=c++17 -I./metal-cpp -framework Metal -framework Foundation -framework QuartzCore harness.cpp -o harness

echo "running harness"
./harness
rm harness shader.air shader.metallib shader.ll
rm -rf "$DEST_DIR"
