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

//BENCHMARK(BM_naive_transpose)
//	->Args({5, 5})
//	->Args({25, 10})
//	->Args({25, 100})
//	->Args({250, 100})
//	->Args({250, 1000})
//	->Args({2500, 1000})
//	->Args({2500, 10000})
//	->Args({25000, 10000});
//
//BENCHMARK(BM_transpose)
//	->Args({5, 5})
//	->Args({25, 10})
//	->Args({25, 100})
//	->Args({250, 100})
//	->Args({250, 1000})
//	->Args({2500, 1000})
//	->Args({2500, 10000})
//	->Args({25000, 10000});
//
//BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int8_t)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int16_t)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int32_t)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::int64_t)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int8_t>)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int16_t>)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int32_t>)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_naive_transpose_types, std::complex<std::int64_t>)->Args({512, 512});
//
//BENCHMARK_TEMPLATE(BM_transpose_types, std::int8_t)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_transpose_types, std::int16_t)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_transpose_types, std::int32_t)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_transpose_types, std::int64_t)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int8_t>)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int16_t>)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int32_t>)->Args({512, 512});
//BENCHMARK_TEMPLATE(BM_transpose_types, std::complex<std::int64_t>)->Args({512, 512});

/* Matrix Multiplication */

//BENCHMARK(BM_naive_multiply)
//	->Args({3, 3, 3})
//	->Args({9, 9, 9})
//	->Args({27, 27, 27})
//	->Args({81, 81, 81})
//	->Args({243, 243, 243})
//	->Args({729, 729, 729})
//	->Args({2187, 2187, 2187});
//
//BENCHMARK(BM_multiply)
//	->Args({3, 3, 3})
//	->Args({9, 9, 9})
//	->Args({27, 27, 27})
//	->Args({81, 81, 81})
//	->Args({243, 243, 243})
//	->Args({729, 729, 729})
//	->Args({2187, 2187, 2187});
//
//BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int8_t)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int16_t)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int32_t)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::int64_t)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int8_t>)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int16_t>)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int32_t>)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_naive_multiply_types, std::complex<std::int64_t>)->Args({243, 243, 243});
//
//BENCHMARK_TEMPLATE(BM_multiply_types, std::int8_t)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_multiply_types, std::int16_t)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_multiply_types, std::int32_t)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_multiply_types, std::int64_t)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int8_t>)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int16_t>)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int32_t>)->Args({243, 243, 243});
//BENCHMARK_TEMPLATE(BM_multiply_types, std::complex<std::int64_t>)->Args({243, 243, 243});

/* Fast Fourier Transform */

BENCHMARK(BM_naive_fft)
	->Args({8})
	->Args({8 << 3})
	->Args({8 << 6})
	->Args({8 << 9})
	->Args({8 << 12})
	->Args({8 << 15})
	->Args({8 << 18});

BENCHMARK(BM_fft)
	->Args({8})
	->Args({8 << 3})
	->Args({8 << 6})
	->Args({8 << 9})
	->Args({8 << 12})
	->Args({8 << 15})
	->Args({8 << 18});

BENCHMARK_TEMPLATE(BM_naive_fft_types, std::int8_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_naive_fft_types, std::int16_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_naive_fft_types, std::int32_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_naive_fft_types, std::int64_t)->Args({4096});

BENCHMARK_TEMPLATE(BM_fft_types, std::int8_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_fft_types, std::int16_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_fft_types, std::int32_t)->Args({4096});
BENCHMARK_TEMPLATE(BM_fft_types, std::int64_t)->Args({4096});

BENCHMARK_MAIN();
