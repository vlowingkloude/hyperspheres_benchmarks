mkdir build

cd build

cmake -DCMAKE_BUILD_TYPE=Release -G Ninja -S .. -B .

cmake --build . --target example
