#!/bin/bash
set -e

echo "compiling: MSL -> LLVM IR"
xcrun -sdk macosx metal -S -emit-llvm shader.metal -o shader.ll

echo "compiling: LLVM IR -> AIR"
xcrun -sdk macosx metal -c shader.ll -o shader.air

echo "compiling: AIR -> metallib"
xcrun -sdk macosx metallib shader.air -o shader.metallib

# objective-c++ isn't great but still easier to run than metal-cpp
echo "running harness"
clang++ -std=c++17 -framework Metal -framework Foundation harness.mm -o harness
./harness
rm harness shader.air shader.metallib shader.ll
