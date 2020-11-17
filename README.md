# Motivation

While cache-oblivious algorithms are based on a solid theoretical foundation, their concrete performance in comparison to RAM model algorithms is underexplored. This project provides benchmarks for three cache-oblivious algorithms based on the work by [Frigo et al.](https://archive.is/W40rU).

# Compilation

⚠️  Compilation of unit tests requires [Catch2](https://github.com/catchorg/Catch2) ⚠️
⚠️  Compilation of benchmarks requires [Google Benchmark](https://github.com/google/benchmark/) ⚠️

```shell
cmake -Bbin
cmake --build bin
./bin/rm_benchmarks
```




