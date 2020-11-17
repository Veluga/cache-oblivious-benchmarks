#!/bin/bash

cd /var/tmp

sudo apt update -y
sudo apt upgrade -y
sudo apt install g++ make sendmail mutt -y

git clone https://github.com/google/benchmark.git
git clone https://github.com/google/googletest.git benchmark/googletest
cd benchmark
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release ../
cmake --build "build" --config Release
sudo cmake --build "build" --config Release --target install

git clone https://github.com/Veluga/cache-oblivious-benchmarks
cd cache-oblivious-benchmarks
cmake -Bbin .
cmake --build .
# ./rm_benchmark --benchmark_format=csv --benchmark_out=benchmarks_machine_$1.csv
touch benchmarks_machine_$1.csv

echo "Benchmarking on machine $1 has concluded." | mutt -s "Benchmark Results - $1" -a benchmarks_machine_$1.csv -- $2
