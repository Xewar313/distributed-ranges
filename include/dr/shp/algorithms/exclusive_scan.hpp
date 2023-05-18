// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <sycl/sycl.hpp>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include <oneapi/dpl/async>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/shp/algorithms/execution_policy.hpp>
#include <dr/shp/allocators.hpp>
#include <dr/shp/detail.hpp>
#include <dr/shp/init.hpp>
#include <dr/shp/vector.hpp>
#include <dr/shp/views/views.hpp>

namespace dr::shp {

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename U, typename BinaryOp>
void exclusive_scan_impl_(ExecutionPolicy &&policy, R &&r, O &&o, U init,
                          BinaryOp &&binary_op) {
  using T = rng::range_value_t<O>;

  static_assert(
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  auto zipped_view = dr::shp::views::zip(r, o);
  auto zipped_segments = zipped_view.zipped_segments();

  if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                               device_policy>) {

    U *d_inits = sycl::malloc_device<U>(rng::size(zipped_segments),
                                        shp::devices()[0], shp::context());

    std::vector<sycl::event> events;

    std::size_t segment_id = 0;
    for (auto &&segs : zipped_segments) {
      auto &&[in_segment, out_segment] = segs;

      auto last_element = rng::prev(rng::end(__detail::local(in_segment)));
      auto dest = d_inits + segment_id;

      auto &&q = __detail::queue(dr::ranges::rank(in_segment));

      auto e = q.single_task([=] { *dest = *last_element; });
      events.push_back(e);
      segment_id++;
    }

    __detail::wait(events);
    events.clear();

    std::vector<U> inits(rng::size(zipped_segments));

    shp::copy(d_inits, d_inits + inits.size(), inits.data() + 1);

    sycl::free(d_inits, shp::context());

    inits[0] = init;

    auto root = dr::shp::devices()[0];
    dr::shp::device_allocator<T> allocator(dr::shp::context(), root);
    dr::shp::vector<T, dr::shp::device_allocator<T>> partial_sums(
        std::size_t(zipped_segments.size()), allocator);

    segment_id = 0;
    for (auto &&segs : zipped_segments) {
      auto &&[in_segment, out_segment] = segs;

      auto &&q = __detail::queue(dr::ranges::rank(in_segment));
      auto &&local_policy = __detail::dpl_policy(dr::ranges::rank(in_segment));

      auto dist = rng::distance(in_segment);
      assert(dist > 0);

      auto first = rng::begin(in_segment);
      auto last = rng::end(in_segment);
      auto d_first = rng::begin(out_segment);

      auto init = inits[segment_id];

      auto event = oneapi::dpl::experimental::exclusive_scan_async(
          local_policy, dr::__detail::direct_iterator(first),
          dr::__detail::direct_iterator(last),
          dr::__detail::direct_iterator(d_first), init, binary_op);

      auto dst_iter = dr::ranges::local(partial_sums).data() + segment_id;

      auto src_iter = dr::ranges::local(out_segment).data();
      rng::advance(src_iter, dist - 1);

      auto e = q.submit([&](auto &&h) {
        h.depends_on(event);
        h.single_task([=]() {
          rng::range_value_t<O> value = *src_iter;
          *dst_iter = value;
        });
      });

      events.push_back(e);

      segment_id++;
    }

    __detail::wait(events);
    events.clear();

    auto &&local_policy = __detail::dpl_policy(0);

    auto first = dr::ranges::local(partial_sums).data();
    auto last = first + partial_sums.size();

    oneapi::dpl::experimental::inclusive_scan_async(local_policy, first, last,
                                                    first, binary_op)
        .wait();

    std::size_t idx = 0;
    for (auto &&segs : zipped_segments) {
      auto &&[in_segment, out_segment] = segs;

      auto &&local_policy = __detail::dpl_policy(dr::ranges::rank(out_segment));

      if (idx > 0) {
        T sum = partial_sums[idx - 1];

        auto first = rng::begin(out_segment);
        auto last = rng::end(out_segment);

        sycl::event e = oneapi::dpl::experimental::for_each_async(
            local_policy, dr::__detail::direct_iterator(first),
            dr::__detail::direct_iterator(last),
            [=](auto &&x) { x = binary_op(x, sum); });

        events.push_back(e);
      }
      idx++;
    }

    __detail::wait(events);

  } else {
    assert(false);
  }
}

// Ranges versions

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename T, typename BinaryOp>
void exclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o, T init,
                    BinaryOp &&binary_op) {
  exclusive_scan_impl_(std::forward<ExecutionPolicy>(policy),
                       std::forward<R>(r), std::forward<O>(o), init,
                       std::forward<BinaryOp>(binary_op));
}

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename T>
void exclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o, T init) {
  exclusive_scan_impl_(std::forward<ExecutionPolicy>(policy),
                       std::forward<R>(r), std::forward<O>(o), init,
                       std::plus<>{});
}

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename T, typename BinaryOp>
void exclusive_scan(R &&r, O &&o, T init, BinaryOp &&binary_op) {
  exclusive_scan_impl_(dr::shp::par_unseq, std::forward<R>(r),
                       std::forward<O>(o), init,
                       std::forward<BinaryOp>(binary_op));
}

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename T>
void exclusive_scan(R &&r, O &&o, T init) {
  exclusive_scan_impl_(dr::shp::par_unseq, std::forward<R>(r),
                       std::forward<O>(o), init, std::plus<>{});
}

// Iterator versions

template <typename ExecutionPolicy, dr::distributed_iterator Iter,
          dr::distributed_iterator OutputIter, typename T, typename BinaryOp>
void exclusive_scan(ExecutionPolicy &&policy, Iter first, Iter last,
                    OutputIter d_first, T init, BinaryOp &&binary_op) {
  auto dist = rng::distance(first, last);
  auto d_last = d_first;
  rng::advance(d_last, dist);
  exclusive_scan_impl_(
      std::forward<ExecutionPolicy>(policy), rng::subrange(first, last),
      rng::subrange(d_first, d_last), init, std::forward<BinaryOp>(binary_op));
}

template <typename ExecutionPolicy, dr::distributed_iterator Iter,
          dr::distributed_iterator OutputIter, typename T>
void exclusive_scan(ExecutionPolicy &&policy, Iter first, Iter last,
                    OutputIter d_first, T init) {
  exclusive_scan(std::forward<ExecutionPolicy>(policy), first, last, d_first,
                 init, std::plus<>{});
}

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
          typename T, typename BinaryOp>
void exclusive_scan(Iter first, Iter last, OutputIter d_first, T init,
                    BinaryOp &&binary_op) {
  exclusive_scan(dr::shp::par_unseq, first, last, d_first, init,
                 std::forward<BinaryOp>(binary_op));
}

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
          typename T>
void exclusive_scan(Iter first, Iter last, OutputIter d_first, T init) {
  exclusive_scan(dr::shp::par_unseq, first, last, d_first, init);
}

} // namespace dr::shp