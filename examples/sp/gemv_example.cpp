// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/sp.hpp>

namespace sp = dr::sp;

int main(int argc, char **argv) {
  auto devices = sp::get_numa_devices(sycl::gpu_selector_v);
  sp::init(devices);

  if (argc != 2) {
    std::cout << "usage: ./sparse_test [matrix market file]\n";
    return 1;
  }

  std::string fname(argv[1]);

  for (auto &device : devices) {
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
              << "\n";
  }

  using T = float;
  using I = long;

  std::cout << "Reading in matrix file " << fname << "\n";
  auto a = dr::sp::mmread<T, I>(fname);

  sp::distributed_vector<T, sp::device_allocator<T>> b(a.shape()[1]);

  sp::duplicated_vector<T> b_duplicated(a.shape()[1]);

  sp::for_each(sp::par_unseq, sp::enumerate(b), [](auto &&tuple) {
    auto &&[idx, value] = tuple;
    value = 1;
  });

  sp::distributed_vector<T, sp::device_allocator<T>> c(a.shape()[0]);

  sp::for_each(sp::par_unseq, c, [](auto &&v) { v = 0; });

  printf("a tiles: %lu x %lu\n", a.grid_shape()[0], a.grid_shape()[1]);

  sp::print_range(b, "b");

  sp::print_matrix(a, "a");

  sp::gemv(c, a, b, b_duplicated);

  sp::print_range(c, "c");

  auto iter = a.end();
  while (iter > a.begin()) {
    iter--;
    auto [index, val] = *iter;
    auto [m, n] = index;

    std::cout << m << " " << n << "\n";
  }

  return 0;
}
