#include <benchmark/benchmark.h>
#include <cstdint>
#include <memory>

#include "ra/matrix_transpose.hpp"
#include "ra/matrix_multiply.hpp"
#include "ra/fft.hpp"

using namespace ra::cache;

/* Matrix Transposition */

static void BM_naive_transpose(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
		auto a = random_matrix<std::int32_t>(state.range(0), state.range(1));
		auto b = std::make_unique<std::int32_t[]>(state.range(0) * state.range(1));
    state.ResumeTiming();
		naive_matrix_transpose<std::int32_t>(a.get(), state.range(0), state.range(1), b.get());
  }
}

static void BM_transpose(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
		auto a = random_matrix<std::int32_t>(state.range(0), state.range(1));
		auto b = std::make_unique<std::int32_t[]>(state.range(0) * state.range(1));
    state.ResumeTiming();
		matrix_transpose<std::int32_t>(a.get(), state.range(0), state.range(1), b.get());
  }
}

template <class T> void BM_naive_transpose_types(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
		auto a = random_matrix<T>(state.range(0), state.range(1));
		auto b = std::make_unique<T[]>(state.range(0) * state.range(1));
    state.ResumeTiming();
		naive_matrix_transpose<T>(a.get(), state.range(0), state.range(1), b.get());
  }
}

template <class T> void BM_transpose_types(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
		auto a = random_matrix<T>(state.range(0), state.range(1));
		auto b = std::make_unique<T[]>(state.range(0) * state.range(1));
    state.ResumeTiming();
		matrix_transpose<T>(a.get(), state.range(0), state.range(1), b.get());
  }
}

/* Matrix Multiplication */

static void BM_naive_multiply(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
		auto a = random_matrix<std::int32_t>(state.range(0), state.range(1));
		auto b = random_matrix<std::int32_t>(state.range(1), state.range(2));
		auto c = random_matrix<std::int32_t>(state.range(0), state.range(2));
    state.ResumeTiming();
		naive_matrix_multiply<std::int32_t>(a.get(), b.get(), state.range(0), state.range(1), state.range(2), c.get());
  }
}

static void BM_multiply(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
		auto a = random_matrix<std::int32_t>(state.range(0), state.range(1));
		auto b = random_matrix<std::int32_t>(state.range(1), state.range(2));
		auto c = random_matrix<std::int32_t>(state.range(0), state.range(2));
    state.ResumeTiming();
		matrix_multiply<std::int32_t>(a.get(), b.get(), state.range(0), state.range(1), state.range(2), c.get());
  }
}

template <class T> void BM_naive_multiply_types(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
		auto a = random_matrix<T>(state.range(0), state.range(1));
		auto b = random_matrix<T>(state.range(1), state.range(2));
		auto c = random_matrix<T>(state.range(0), state.range(2));
    state.ResumeTiming();
		naive_matrix_multiply<T>(a.get(), b.get(), state.range(0), state.range(1), state.range(2), c.get());
  }
}

template <class T> void BM_multiply_types(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
		auto a = random_matrix<T>(state.range(0), state.range(1));
		auto b = random_matrix<T>(state.range(1), state.range(2));
		auto c = random_matrix<T>(state.range(0), state.range(2));
    state.ResumeTiming();
		matrix_multiply<T>(a.get(), b.get(), state.range(0), state.range(1), state.range(2), c.get());
  }
}

/* Fast Fourier Transform */

static void BM_naive_fft(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
		auto x = generate_random_vector<std::complex<std::int32_t>>(state.range(0));
    state.ResumeTiming();
		dit_fft<std::complex<std::int32_t>>(x.get(), state.range(0));
  }
}

static void BM_fft(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
		auto x = generate_random_vector<std::complex<std::int32_t>>(state.range(0));
    state.ResumeTiming();
		forward_fft<std::complex<std::int32_t>>(x.get(), state.range(0));
  }
}

template <class T> void BM_naive_fft_types(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
		auto x = generate_random_vector<std::complex<T>>(state.range(0));
    state.ResumeTiming();
		dit_fft<std::complex<T>>(x.get(), state.range(0));
  }
}

template <class T> void BM_fft_types(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
		auto x = generate_random_vector<std::complex<T>>(state.range(0));
    state.ResumeTiming();
		forward_fft<std::complex<T>>(x.get(), state.range(0));
  }
}

/* Matrix Transposition */

// Naive transposition, varying sizes

BENCHMARK(BM_naive_transpose)
	->Args({5, 5})
	->Args({5, 10})
	->Args({10, 5})
	->Args({10, 10})
	->Args({50, 10})
	->Args({10, 50})
	->Args({50, 50})
	->Args({50, 100})
	->Args({100, 50})
	->Args({100, 100})
	->Args({100, 500})
	->Args({500, 100})
	->Args({500, 500})
	->Args({500, 1000})
	->Args({1000, 500})
	->Args({1000, 1000})
	->Args({1000, 5000})
	->Args({5000, 1000})
	->Args({5000, 5000})
	->Args({5000, 10000})
	->Args({10000, 5000})
	->Args({10000, 10000});

// Cache-oblivious transposition, varying sizes

BENCHMARK(BM_transpose)
	->Args({5, 5})
	->Args({5, 10})
	->Args({10, 5})
	->Args({10, 10})
	->Args({50, 10})
	->Args({10, 50})
	->Args({50, 50})
	->Args({50, 100})
	->Args({100, 50})
	->Args({100, 100})
	->Args({100, 500})
	->Args({500, 100})
	->Args({500, 500})
	->Args({500, 1000})
	->Args({1000, 500})
	->Args({1000, 1000})
	->Args({1000, 5000})
	->Args({5000, 1000})
	->Args({5000, 5000})
	->Args({5000, 10000})
	->Args({10000, 5000})
	->Args({10000, 10000});

// Naive transposition, varying types

BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int8_t)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int8_t)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int8_t)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int16_t)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int16_t)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int16_t)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int32_t)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int32_t)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int32_t)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int64_t)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int64_t)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int64_t)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int8_t>)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int8_t>)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int8_t>)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int16_t>)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int16_t>)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int16_t>)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int32_t>)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int32_t>)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int32_t>)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int64_t>)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int64_t>)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int64_t>)->Args({1024, 512});

// Cache-oblivious transposition, varying types

BENCHMARK_TEMPLATE(BM_transpose_types, std::int8_t)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_transpose_types, std::int8_t)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_transpose_types, std::int8_t)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_transpose_types, std::int16_t)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_transpose_types, std::int16_t)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_transpose_types, std::int16_t)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_transpose_types, std::int32_t)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_transpose_types, std::int32_t)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_transpose_types, std::int32_t)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_transpose_types, std::int64_t)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_transpose_types, std::int64_t)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_transpose_types, std::int64_t)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int8_t>)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int8_t>)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int8_t>)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int16_t>)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int16_t>)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int16_t>)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int32_t>)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int32_t>)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int32_t>)->Args({1024, 512});

BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int64_t>)->Args({512, 512});
BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int64_t>)->Args({512, 1024});
BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int64_t>)->Args({1024, 512});

/* Matrix Multiplication */

// Naive multiplication, varying sizes

BENCHMARK(BM_naive_multiply)
	->Args({8, 8, 8})
	->Args({16, 4, 16})
	->Args({4, 16, 4})
	->Args({16, 16, 16})
	->Args({32, 8, 32})
	->Args({8, 32, 8})
	->Args({32, 32, 32})
	->Args({64, 16, 64})
	->Args({16, 64, 16})
	->Args({64, 64, 64})
	->Args({128, 32, 128})
	->Args({32, 128, 32})
	->Args({128, 128, 128})
	->Args({256, 64, 256})
	->Args({64, 256, 64})
	->Args({256, 256, 256})
	->Args({512, 128, 512})
	->Args({128, 512, 128})
	->Args({512, 512, 512})
	->Args({1024, 256, 1024})
	->Args({256, 1024, 256})
	->Args({1024, 1024, 1024})
	->Args({2048, 512, 2048})
	->Args({512, 2048, 512})
	->Args({2048, 2048, 2048})
	->Args({4096, 1024, 4096})
	->Args({1024, 4096, 1024});

// Cache-oblivious multiplication, varying sizes

BENCHMARK(BM_multiply)
	->Args({8, 8, 8})
	->Args({16, 4, 16})
	->Args({4, 16, 4})
	->Args({16, 16, 16})
	->Args({32, 8, 32})
	->Args({8, 32, 8})
	->Args({32, 32, 32})
	->Args({64, 16, 64})
	->Args({16, 64, 16})
	->Args({64, 64, 64})
	->Args({128, 32, 128})
	->Args({32, 128, 32})
	->Args({128, 128, 128})
	->Args({256, 64, 256})
	->Args({64, 256, 64})
	->Args({256, 256, 256})
	->Args({512, 128, 512})
	->Args({128, 512, 128})
	->Args({512, 512, 512})
	->Args({1024, 256, 1024})
	->Args({256, 1024, 256})
	->Args({1024, 1024, 1024})
	->Args({2048, 512, 2048})
	->Args({512, 2048, 512})
	->Args({2048, 2048, 2048})
	->Args({4096, 1024, 4096})
	->Args({1024, 4096, 1024});

// Naive multiplication, varying types

BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int8_t)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int8_t)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int8_t)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int16_t)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int16_t)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int16_t)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int32_t)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int32_t)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int32_t)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int64_t)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int64_t)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int64_t)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int8_t>)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int8_t>)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int8_t>)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int16_t>)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int16_t>)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int16_t>)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int32_t>)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int32_t>)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int32_t>)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int64_t>)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int64_t>)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int64_t>)->Args({64, 1024, 64});

// Cache-oblivious multiplication, varying types

BENCHMARK_TEMPLATE(BM_multiply_types, std::int8_t)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_multiply_types, std::int8_t)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_multiply_types, std::int8_t)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_multiply_types, std::int16_t)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_multiply_types, std::int16_t)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_multiply_types, std::int16_t)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_multiply_types, std::int32_t)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_multiply_types, std::int32_t)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_multiply_types, std::int32_t)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_multiply_types, std::int64_t)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_multiply_types, std::int64_t)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_multiply_types, std::int64_t)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int8_t>)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int8_t>)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int8_t>)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int16_t>)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int16_t>)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int16_t>)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int32_t>)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int32_t>)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int32_t>)->Args({64, 1024, 64});

BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int64_t>)->Args({256, 256, 256});
BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int64_t>)->Args({1024, 64, 1024});
BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int64_t>)->Args({64, 1024, 64});

/* Fast Fourier Transform */

// Naive FFT, varying sizes

BENCHMARK(BM_naive_fft)
	->Args({8})
	->Args({8 << 3})
	->Args({8 << 6})
	->Args({8 << 9})
	->Args({8 << 12})
	->Args({8 << 15})
	->Args({8 << 18});

// Cache-oblivious FFT, varying types

BENCHMARK(BM_fft)
	->Args({8})
	->Args({8 << 3})
	->Args({8 << 6})
	->Args({8 << 9})
	->Args({8 << 12})
	->Args({8 << 15})
	->Args({8 << 18});

// Naive FFT, varying types

BENCHMARK_TEMPLATE(BM_naive_fft_types, std::int8_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_naive_fft_types, std::int16_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_naive_fft_types, std::int32_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_naive_fft_types, std::int64_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_naive_fft_types, long double)->Args({4096});

// Cache-oblivious, varying types

BENCHMARK_TEMPLATE(BM_fft_types, std::int8_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_fft_types, std::int16_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_fft_types, std::int32_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_fft_types, std::int64_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_fft_types, long double)->Args({4096});

BENCHMARK_MAIN();
