#!/bin/bash
set -e
echo "Forging AIR..."
python3 air_forge.py tests/input.ll > tests/test.ll
xcrun -sdk macosx metal -c tests/test.ll -o tests/test.air
xcrun -sdk macosx metallib tests/test.air -o tests/test.metallib
echo "Compiling harness..."
clang++ -std=c++17 -framework Metal -framework Foundation tests/verify.mm -o tests/verify
echo "Running verification..."
./tests/verify
