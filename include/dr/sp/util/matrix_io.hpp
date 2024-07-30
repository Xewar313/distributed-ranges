// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <dr/sp/util/coo_matrix.hpp>
#include <dr/sp/views/csr_matrix_view.hpp>

namespace dr::sp {

namespace __detail {


/// Read in the Matrix Market file at location `file_path` and a return
/// a coo_matrix data structure with its contents.
template <typename T, typename I = std::size_t>
inline coo_matrix<T, I> mmread(std::string file_path, bool one_indexed = true) {
  using size_type = std::size_t;

  std::ifstream f;

  f.open(file_path.c_str());

  if (!f.is_open()) {
    // TODO better choice of exception.
    throw std::runtime_error("mmread: cannot open " + file_path);
  }

  std::string buf;

  // Make sure the file is matrix market matrix, coordinate, and check whether
  // it is symmetric. If the matrix is symmetric, non-diagonal elements will
  // be inserted in both (i, j) and (j, i).  Error out if skew-symmetric or
  // Hermitian.
  std::getline(f, buf);
  std::istringstream ss(buf);
  std::string item;
  ss >> item;
  if (item != "%%MatrixMarket") {
    throw std::runtime_error(file_path +
                             " could not be parsed as a Matrix Market file.");
  }
  ss >> item;
  if (item != "matrix") {
    throw std::runtime_error(file_path +
                             " could not be parsed as a Matrix Market file.");
  }
  ss >> item;
  if (item != "coordinate") {
    throw std::runtime_error(file_path +
                             " could not be parsed as a Matrix Market file.");
  }
  bool pattern;
  ss >> item;
  if (item == "pattern") {
    pattern = true;
  } else {
    pattern = false;
  }
  // TODO: do something with real vs. integer vs. pattern?
  ss >> item;
  bool symmetric;
  if (item == "general") {
    symmetric = false;
  } else if (item == "symmetric") {
    symmetric = true;
  } else {
    throw std::runtime_error(file_path + " has an unsupported matrix type");
  }

  bool outOfComments = false;
  while (!outOfComments) {
    std::getline(f, buf);

    if (buf[0] != '%') {
      outOfComments = true;
    }
  }

  I m, n, nnz;
  // std::istringstream ss(buf);
  ss.clear();
  ss.str(buf);
  ss >> m >> n >> nnz;

  // NOTE for symmetric matrices: `nnz` holds the number of stored values in
  // the matrix market file, while `matrix.nnz_` will hold the total number of
  // stored values (including "mirrored" symmetric values).
  coo_matrix<T, I> matrix({m, n});

  size_type c = 0;
  while (std::getline(f, buf)) {
    I i, j;
    T v;
    std::istringstream ss(buf);
    if (!pattern) {
      ss >> i >> j >> v;
    } else {
      ss >> i >> j;
      v = T(1);
    }
    if (one_indexed) {
      i--;
      j--;
    }

    if (i >= m || j >= n) {
      throw std::runtime_error(
          "read_MatrixMarket: file has nonzero out of bounds.");
    }

    matrix.push_back({{i, j}, v});

    if (symmetric && i != j) {
      matrix.push_back({{j, i}, v});
    }

    c++;
    if (c > nnz) {
      throw std::runtime_error("read_MatrixMarket: error reading Matrix Market "
                               "file, file has more nonzeros than reported.");
    }
  }

  f.close();

  return matrix;
}

} // namespace __detail

template <typename T, typename I>
auto create_distributed(dr::sp::csr_matrix_view<T, I> local_mat,
                        const matrix_partition &partition) {
  return dr::sp::sparse_matrix<T, I>(local_mat, partition);
}

template <typename T, typename I = std::size_t>
auto mmread(std::string file_path, const matrix_partition &partition,
            bool one_indexed = true) {
  auto m = __detail::mmread<T, I>(file_path, one_indexed);
  auto shape = m.shape();
  auto nnz = m.size();

  auto local_mat = __detail::convert_to_csr(m, shape, nnz, std::allocator<T>{});

  auto a = create_distributed(local_mat, partition);

  __detail::destroy_csr_matrix_view(local_mat, std::allocator<T>{});

  return a;
}

template <typename T, typename I = std::size_t>
auto mmread(std::string file_path, bool one_indexed = true) {
  return mmread<T, I>(
      file_path,
      dr::sp::row_cyclic(),
      one_indexed);
}

} // namespace dr::sp
