#!/bin/bash
set -e

# uv run demo.py > input.ll  ----> somehow breaks assert statement
cat custom.ll > input.ll

python3 air_forge.py input.ll > test.ll

xcrun -sdk macosx metal -c test.ll -o test.air

xcrun -sdk macosx metallib test.air -o test.metallib

clang++ -std=c++17 -framework Metal -framework Foundation verify.mm -o verify

./verify
rm -rf input.ll test.ll test.air test.metallib verify
