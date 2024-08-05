// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/sp.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

// FIXME: what is grb.hpp? add it to cmake or remove this dependency
// #include <grb/grb.hpp>

#include <concepts>

namespace sp = dr::sp;

template <typename T, typename I> auto local_gemv(dr::sp::__detail::coo_matrix<T, I> &&a) {
  std::vector<T> b(a.shape()[1], 1);
  std::vector<T> c(a.shape()[0], 0);

  for (auto &&[index, v] : a) {
    auto &&[i, k] = index;
    c[i] += v * b[k];
  }

  return c;
}

template <typename T, typename U> bool is_equal(T &&x, U &&y) { return x == y; }

template <std::floating_point T>
bool is_equal(T a, T b, T epsilon = 128 * std::numeric_limits<T>::epsilon()) {
  if (a == b) {
    return true;
  }

  auto abs_th = std::numeric_limits<T>::min();

  auto diff = std::abs(a - b);

  auto norm =
      std::min((std::abs(a) + std::abs(b)), std::numeric_limits<T>::max());
  return diff < std::max(abs_th, epsilon * norm);
}

template <rng::forward_range A, rng::forward_range B>
bool is_equal(A &&a, B &&b) {
  for (auto &&[x, y] : rng::views::zip(a, b)) {
    if (!is_equal(x, y, 1.)) {
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {

  if (argc != 2) {
    fmt::print("usage: ./gemv_benchmark [number of devices to use]\n");
    return 1;
  }

  auto all_devices = sp::get_numa_devices(sycl::default_selector_v);
  std::vector<sycl::device> devices;
  int device_num = std::min(all_devices.size(), std::stoul(argv[1]));
  for (int i = 0; i < device_num; i++) {
    devices.push_back(all_devices[i]);
  }
  sp::init(devices);

  fmt::print("Number of devices {}\n", devices.size());

  using T = double;
  using I = long;

  std::size_t m = 100000;
  std::size_t k = 100000;

  fmt::print("Generating matrix\n");
    auto begin1 = std::chrono::high_resolution_clock::now();
  auto csr = dr::sp::generate_random_csr<T, I>({m, k}, 0.01, 42);
    auto end1 = std::chrono::high_resolution_clock::now();
    double duration1 = std::chrono::duration<double>(end1 - begin1).count();
 fmt::print("genertion time:{}...\n", duration1);

  sp::duplicated_vector<T> b_duplicated(k);

  fmt::print("Initializing distributed data structures...\n");
     begin1 = std::chrono::high_resolution_clock::now();
  auto a = dr::sp::sparse_matrix<T, I>(csr, dr::sp::row_cyclic());
    end1 = std::chrono::high_resolution_clock::now();
     duration1 = std::chrono::duration<double>(end1 - begin1).count();
 fmt::print("distribution time:{}...\n", duration1);

  dr::sp::distributed_vector<T, dr::sp::device_allocator<T>> b(k);
  dr::sp::distributed_vector<T, dr::sp::device_allocator<T>> c(m);

  dr::sp::for_each(dr::sp::par_unseq, b, [](auto &&v) { v = 1; });
  dr::sp::for_each(dr::sp::par_unseq, c, [](auto &&v) { v = 0; });

  std::size_t n_iterations = 100;

  std::vector<double> durations;

  // GEMV
  fmt::print("Verification:\n");

  fmt::print("Computing GEMV...\n");
  dr::sp::gemv(c, a, b, b_duplicated);
  fmt::print("Copying...\n");
  std::vector<T> l(c.size());
  dr::sp::copy(c.begin(), c.end(), l.begin());
  fmt::print("Verifying...\n");
  
  std::vector<T> c_local(c.size());
  for (int i = 0; i < c.size(); i++) {
    c_local[i] = 0;
  }

  for (auto &&[index, val]: csr) {
    auto [j, i] = index;
    // fmt::print("value: {} {} {}\n", j , i, val);
    c_local[j] += val * b[i];
  }

  // for (auto &&[index, val]: a) {
  //   auto [j, i] = index;
  //   fmt::print("value: {} {} {}\n", j , i, val);
  // }
  // auto c_local = local_gemv(std::move(coo_a));
  for (std::size_t i = 0; i < l.size(); i++) {
    if (!is_equal(l[i], c_local[i])) {
      fmt::print("{} {} != {}\n", i, l[i], c_local[i]);
    }
  }
  assert(is_equal(c_local, l));

  fmt::print("Benchmarking...\n");
  durations.reserve(n_iterations);
  for (std::size_t i = 0; i < n_iterations; i++) {
    auto begin = std::chrono::high_resolution_clock::now();
    dr::sp::gemv(c, a, b, b_duplicated);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    durations.push_back(duration);
  }

  fmt::print("Durations: {}\n", durations | rng::views::transform([](auto &&x) {
                                  return x * 1000;
                                }));

  std::sort(durations.begin(), durations.end());

  double median_duration = durations[durations.size() / 2];

  std::cout << "GEMV Row: " << median_duration * 1000 << " ms" << std::endl;

  std::size_t n_bytes = sizeof(T) * a.size() +
                        sizeof(I) * (a.size() + a.shape()[0] + 1) // size of A
                        + sizeof(T) * b.size()                    // size of B
                        + sizeof(T) * c.size();                   // size of C
  double n_gbytes = n_bytes * 1e-9;
  fmt::print("{} GB/s\n", n_gbytes / median_duration);

  durations.clear();

  fmt::print("Finalize...\n");

  sp::finalize();
  return 0;
}
