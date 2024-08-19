// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>
#include <dr/sp.hpp>
#include <fmt/core.h>

int main(int argc, char **argv) {
  auto devices = dr::sp::get_numa_devices(sycl::default_selector_v);
  fmt::print("Number of found devices {}\n", devices.size());
  dr::sp::init(devices);
  
  std::size_t n = 32;

  dr::sp::distributed_vector<int> v(n);
  return 0;
}