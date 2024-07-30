// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>
#include <dr/sp.hpp>
#include <fmt/core.h>

int main(int argc, char **argv) {
  auto devices = dr::sp::get_numa_devices(sycl::default_selector_v);
  dr::sp::init(devices);

  if (argc != 2) {
    fmt::print("usage: ./sparse_test [matrix market file]\n");
    return 1;
  }

  std::string fname(argv[1]);

  using T = float;
  using I = int;

  fmt::print("Reading in matrix file {}\n", fname);
  auto a = dr::sp::mmread<T, I>(fname);
  
  dr::sp::print_matrix(a, "a");

  return 0;
}
