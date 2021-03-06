#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <complex>
#include <memory>
#include <random>

#include "ra/fft.hpp"

using namespace ra::cache;

template <class T> 
void check_vector_equal(T *a, T *b, std::size_t n) {
	for (std::size_t i = 0; i < n; ++i) {
		REQUIRE(a[i].real() == Approx(b[i].real()).margin(1e-12));
		REQUIRE(a[i].imag() == Approx(b[i].imag()).margin(1e-12));
	}
};

template <class T>
std::unique_ptr<T[]> copy_vector(const T* v, std::size_t n) {
	auto copy = std::make_unique<T[]>(n);
	for (int i = 0; i < n; ++i) {
		copy.get()[i] = v[i];
	}
	return copy;
}

TEST_CASE("Naive FFT.") {
  SECTION("Two elements.") {
		auto x = generate_random_vector<std::complex<double>>(2);
		std::complex<double> expected[2] = {
			x.get()[0] + x.get()[1],
			x.get()[0] - x.get()[1]
		};
		
		naive_fft<std::complex<double>>(x.get(), 2);
		check_vector_equal(x.get(), expected, 2);
  }

  SECTION("Four elements.") {
		auto x = generate_random_vector<std::complex<double>>(4);
		std::complex<double> expected[4] = {
			x.get()[0] + x.get()[1] + x.get()[2] + x.get()[3],
			x.get()[0] - x.get()[2] + std::complex<double>(0, 1) * (-x.get()[1] + x.get()[3]),
			x.get()[0] - x.get()[1] + x.get()[2] - x.get()[3],
			x.get()[0] - x.get()[2] + std::complex<double>(0, 1) * (x.get()[1] - x.get()[3])
		};

		naive_fft<std::complex<double>>(x.get(), 4);
		check_vector_equal(x.get(), expected, 4);
  }
}

TEST_CASE("Cache oblivious FFT.") {
  SECTION("Two elements.") {
		auto x = generate_random_vector<std::complex<double>>(2);
		auto expected = copy_vector(x.get(), 2);
		
		forward_fft<std::complex<double>>(x.get(), 2);
		naive_fft<std::complex<double>>(expected.get(), 2);
		check_vector_equal(x.get(), expected.get(), 2);
  }
	
  SECTION("Four elements.") {
		auto x = generate_random_vector<std::complex<double>>(4);
		auto expected = copy_vector(x.get(), 4);
		
		forward_fft<std::complex<double>>(x.get(), 4);
		naive_fft<std::complex<double>>(expected.get(), 4);
		check_vector_equal(x.get(), expected.get(), 4);
  }

  SECTION("Eight elements.") {
		auto x = generate_random_vector<std::complex<double>>(8);
		auto expected = copy_vector(x.get(), 8);
		
		forward_fft<std::complex<double>>(x.get(), 8);
		naive_fft<std::complex<double>>(expected.get(), 8);
		check_vector_equal(x.get(), expected.get(), 8);
  }

  SECTION("128 elements.") {
		auto x = generate_random_vector<std::complex<double>>(128);
		auto expected = copy_vector(x.get(), 128);
		
		forward_fft<std::complex<double>>(x.get(), 128);
		naive_fft<std::complex<double>>(expected.get(), 128);
		check_vector_equal(x.get(), expected.get(), 128);
  }

  SECTION("4096 elements.") {
		auto x = generate_random_vector<std::complex<double>>(4096);
		auto expected = copy_vector(x.get(), 4096);
		
		forward_fft<std::complex<double>>(x.get(), 4096);
		naive_fft<std::complex<double>>(expected.get(), 4096);
		check_vector_equal(x.get(), expected.get(), 4096);
  }
}
