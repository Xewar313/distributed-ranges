// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"

TEST(SparseMatrix, Gemv) {
  std::size_t m = 100;
  std::size_t k = 100;

  dr::sp::sparse_matrix<float> a(
      {m, k}, 0.1f,
      dr::sp::block_cyclic({dr::sp::tile::div, dr::sp::tile::div},
                           {dr::sp::nprocs(), 1}));

  dr::sp::distributed_vector<float> b(k, 1.f);
  dr::sp::distributed_vector<float> c(m, 0.f);

  dr::sp::gemv(c, a, b);

  std::vector<float> c_local(m);

  dr::sp::copy(c.begin(), c.end(), c_local.begin());

  std::vector<float> c_ref(m, 0.f);

  for (auto &&[index, v] : a) {
    auto &&[i, k] = index;

    c_ref[i] += v;
  }

  EXPECT_TRUE(fp_equal(c_ref, c_local))
      << fmt::format("Reference:\n  {}\nActual:\n  {}\n", c_ref, c_local);
}


TEST(SparseMatrix, EmptyGemv) {
  std::size_t m = 100;
  std::size_t k = 100;

  dr::sp::_detail::coo_matrix<float, int> base;
  auto csr = __detail::convert_to_csr(base, {m,k}, base.size(), std::allocator<T>{});
  dr::sp::sparse_matrix<float> a(
      csr,
      dr::sp::rows_cyclic());

  dr::sp::distributed_vector<float> b(k, 1.f);
  dr::sp::distributed_vector<float> c(m, 0.f);

  dr::sp::gemv(c, a, b);

  std::vector<float> c_local(m);

  dr::sp::copy(c.begin(), c.end(), c_local.begin());

  std::vector<float> c_ref(m, 0.f);

  EXPECT_TRUE(fp_equal(c_ref, c_local))
      << fmt::format("Reference:\n  {}\nActual:\n  {}\n", c_ref, c_local);
}

TEST(SparseMatrix, ZeroVector) {
  std::size_t m = 100;
  std::size_t k = 100;

  dr::sp::_detail::coo_matrix<float, int> base;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      base.push_back({i, j}, i + j);
    }
  }
  auto csr = __detail::convert_to_csr(base, {m,k}, base.size(), std::allocator<T>{});
  dr::sp::sparse_matrix<float> a(
      csr,
      dr::sp::rows_cyclic());

  dr::sp::distributed_vector<float> b(k, 0.f);
  dr::sp::distributed_vector<float> c(m, 0.f);

  dr::sp::gemv(c, a, b);

  std::vector<float> c_local(m);

  dr::sp::copy(c.begin(), c.end(), c_local.begin());

  std::vector<float> c_ref(m, 0.f);

  EXPECT_TRUE(fp_equal(c_ref, c_local))
      << fmt::format("Reference:\n  {}\nActual:\n  {}\n", c_ref, c_local);
}