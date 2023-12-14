// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "fmt/core.h"
#include "mpi.h"
#include "oneapi/mkl/dfti.hpp"

#include "dr/mhp.hpp"

MPI_Comm comm;
int comm_rank;
int comm_size;

// Adapted examples/sycl/dft/source/dp_complex_3d.cpp
void init_matrix(auto &mat) {

  constexpr double TWOPI = 6.2831853071795864769;
  constexpr int H1 = -1, H2 = -2, H3 = -3;
  int N1 = mat.extent(2), N2 = mat.extent(1), N3 = mat.extent(0);

  auto moda = [](int K, int L, int M) {
    return (double)(((long long)K * L) % M);
  };

  auto init = [=](auto index, auto v) {
    auto phase = TWOPI * (moda(static_cast<int>(index[2]), H1, N1) / N1 +
                          moda(static_cast<int>(index[1]), H2, N2) / N2 +
                          moda(static_cast<int>(index[0]), H3, N3) / N3);
    std::get<0>(v) = {std::cos(phase) / (N3 * N2 * N1),
                      std::sin(phase) / (N3 * N2 * N1)};
  };

  dr::mhp::for_each(init, mat);
}

template <typename T> void transpose_matrix(auto &i_mat, auto &o_mat) {
  dr::mhp::transpose(i_mat, o_mat);
}

template <typename T> struct dft_precision {
  static const oneapi::mkl::dft::precision value =
      oneapi::mkl::dft::precision::SINGLE;
};

template <> struct dft_precision<double> {
  static const oneapi::mkl::dft::precision value =
      oneapi::mkl::dft::precision::DOUBLE;
};

template <typename T> class distributed_fft {
  using fft_plan_t =
      oneapi::mkl::dft::descriptor<dft_precision<T>::value,
                                   oneapi::mkl::dft::domain::COMPLEX>;
  fft_plan_t *fft_1d_plan;
  fft_plan_t *fft_2d_plan;

public:
  explicit distributed_fft(sycl::queue &q, auto &i_slab, auto &o_slab) {
    int i_x = i_slab.extent(0);
    int i_y = i_slab.extent(1);
    int i_z = i_slab.extent(2);
    fft_2d_plan = new fft_plan_t({i_y, i_z});
    fft_2d_plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                           i_x);
    fft_2d_plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                           i_y * i_z);
    fft_2d_plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                           i_y * i_z);
    fft_2d_plan->set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                           (1.0 / (i_x * i_y * i_z)));
    fft_2d_plan->commit(q);

    auto o_x = o_slab.extent(0);
    auto o_y = o_slab.extent(1);
    auto o_z = o_slab.extent(2);
    fft_1d_plan = new fft_plan_t(o_z);
    fft_1d_plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                           o_x * o_y);
    fft_1d_plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, o_z);
    fft_1d_plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, o_z);
    fft_1d_plan->commit(q);
  }

  void transpose_matrix(auto &i_mat, auto &o_mat) {}

  void compute_forward(auto &i_mat, auto &i_slab, auto &o_mat, auto &o_slab) {
    oneapi::mkl::dft::compute_forward(*fft_2d_plan, i_slab.data_handle())
        .wait();

    transpose_matrix(i_mat, o_mat);

    oneapi::mkl::dft::compute_forward(*fft_1d_plan, o_slab.data_handle())
        .wait();
  }

  void compute_backward(auto &i_mat, auto &i_slab, auto &o_mat, auto &o_slab) {
    oneapi::mkl::dft::compute_backward(*fft_1d_plan, i_slab.data_handle())
        .wait();

    transpose_matrix(i_mat, o_mat);

    oneapi::mkl::dft::compute_backward(*fft_2d_plan, o_slab.data_handle())
        .wait();
  }
};

int do_fft(std::size_t nreps, std::size_t x, std::size_t y, std::size_t z) {
  sycl::queue q = dr::mhp::select_queue();
  dr::mhp::init(q);

  using real_t = double;
  using value_t = std::complex<real_t>;
  using mat = dr::mhp::distributed_mdarray<value_t, 3>;
  std::array<std::size_t, 3> i_shape({x, y, z});
  std::array<std::size_t, 3> o_shape({y, z, x});
  mat i_mat(i_shape);
  mat o_mat(o_shape);

  // The distribution creates one tile per rank
  auto i_slabs = dr::mhp::local_mdspans(i_mat);
  auto i_slab = *i_slabs.begin();

  auto o_slabs = dr::mhp::local_mdspans(o_mat);
  auto o_slab = *o_slabs.begin();

  distributed_fft<real_t> fft3d(q, i_slab, o_slab);

  fmt::print("Initializing...\n");
  init_matrix(i_mat);

  if (nreps == 0) { // debug
    mat t_mat(i_shape);
    fft3d.compute_forward(i_mat, i_slab, o_mat, o_slab);
    fft3d.compute_backward(o_mat, o_slab, i_mat, i_slab);

    auto sub_view = dr::mhp::views::zip(i_mat, t_mat) |
                    dr::mhp::views::transform([](auto &&e) {
                      auto [a, b] = e;
                      return value_t(a) - value_t(b);
                    });
    auto diff_sum = dr::mhp::reduce(sub_view, value_t{});
    fmt::print("i_mat:\n{}o_mat:\n{}", i_mat, o_mat);
    fmt::print("Difference {} {} \n", diff_sum.real(), diff_sum.imag());
  }

  for (int iter = 0; iter < nreps; ++iter) {
    fft3d.compute_forward(i_mat, i_slab, o_mat, o_slab);
    fft3d.compute_backward(o_mat, o_slab, i_mat, i_slab);
  }

  dr::mhp::finalize();
  return 0;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  cxxopts::Options options_spec(argv[0], "fft3d");
  // clang-format off
  options_spec.add_options()
    ("n", "Number of repetitions", cxxopts::value<std::size_t>()->default_value("1"))
    ("l,log", "enable logging")
    ("h,help", "Print help");
  // clang-format on

  cxxopts::ParseResult options;
  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  std::unique_ptr<std::ofstream> logfile;
  if (options.count("log")) {
    logfile.reset(new std::ofstream(fmt::format("dr.{}.log", comm_rank)));
    dr::drlog.set_file(*logfile);
  }

  std::size_t x = 8;
  std::size_t y = 8;
  std::size_t z = 8;
  std::size_t nreps = options["n"].as<std::size_t>();

  do_fft(nreps, x, y, z);

  MPI_Finalize();
  return 0;
}
