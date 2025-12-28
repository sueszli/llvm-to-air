#!/bin/bash
set -e

echo "forging AIR..."
python3 air_forge.py input.ll > test.ll
xcrun -sdk macosx metal -c test.ll -o test.air
xcrun -sdk macosx metallib test.air -o test.metallib
echo "compiling harness..."
clang++ -std=c++17 -framework Metal -framework Foundation verify.mm -o verify
echo "running verification..."
./verify
rm test.ll test.air test.metallib verify
