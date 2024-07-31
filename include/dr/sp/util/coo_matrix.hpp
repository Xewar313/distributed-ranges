// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/sp/containers/matrix_entry.hpp>
#include <map>
#include <memory>

namespace dr::sp {

namespace __detail {

template <typename T, typename I, typename Iter> class coo_matrix_accessor {
public:
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using scalar_type = T;
  using scalar_reference = T &;

  using value_type = dr::sp::matrix_entry<T, I>;

  using reference = dr::sp::matrix_ref<T, I>;

  using iterator_category = std::bidirectional_iterator_tag;

  using iterator_accessor = coo_matrix_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  using key_type = dr::index<>;

  constexpr coo_matrix_accessor() noexcept = default;
  constexpr ~coo_matrix_accessor() noexcept = default;
  constexpr coo_matrix_accessor(const coo_matrix_accessor &) noexcept = default;
  constexpr coo_matrix_accessor &
  operator=(const coo_matrix_accessor &) noexcept = default;

  constexpr coo_matrix_accessor(Iter iter) noexcept : iter_(iter) {}

  auto &operator++() {
    iter_++;
    return *this;
  }
  auto operator++(int) {
    auto old = *this;
    iter_++;
    return old;
  }
  auto &operator--() {
    iter_--;
    return *this;
  }
  auto operator--(int) {
    auto old = *this;
    iter_--;
    return old;
  }
  constexpr bool operator==(const iterator_accessor &other) const noexcept {
    return iter_ == other.iter_;
  }

  constexpr bool operator<(const iterator_accessor &other) const noexcept {
    return iter_ < other.iter_;
  }

  constexpr reference operator*() const noexcept {
    auto [index, val] = *iter_;
    return reference(index, val);
  }

private:
  Iter iter_;
};

template <typename T, typename I, typename L>
using coo_matrix_iterator = dr::iterator_adaptor<coo_matrix_accessor<T, I, L>>;

template <typename T, typename I, typename Allocator = std::allocator<T>>
class coo_matrix {
public:
  using value_type = dr::sp::matrix_entry<T, I>;
  using scalar_type = T;
  using index_type = I;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using allocator_type = Allocator;

  using key_type = dr::index<I>;
  using map_type = T;

  using backend_allocator_type = typename std::allocator_traits<
      allocator_type>::template rebind_alloc<std::pair<const key_type, T>>;
  using backend_type =
      std::map<key_type, T, std::less<key_type>, backend_allocator_type>;

  using iterator = coo_matrix_iterator<T, I, typename backend_type::iterator>;
  using const_iterator = coo_matrix_iterator<T, I, typename backend_type::iterator>;

  using reference = dr::sp::matrix_ref<T, I>;
  using const_reference = dr::sp::matrix_ref<std::add_const_t<T>, I>;

  using scalar_reference = T &;

  coo_matrix(dr::index<I> shape) : shape_(shape) {}

  dr::index<I> shape() const noexcept { return shape_; }

  size_type size() const noexcept { return tuples_.size(); }

  iterator begin() noexcept { return iterator(tuples_.begin()); }

  const_iterator begin() const noexcept { return iterator(tuples_.begin()); }

  iterator end() noexcept { return iterator(tuples_.end()); }

  const_iterator end() const noexcept { return iterator(tuples_.end()); }

  // template <typename InputIt> void insert(InputIt first, InputIt last) {
  //   for (auto iter = first; iter != last; ++iter) {
  //     insert(*iter);
  //   }
  // }

  template <typename InputIt> void push_back(InputIt first, InputIt last) {
    for (auto iter = first; iter != last; ++iter) {
      push_back(*iter);
    }
  }

  void push_back(const value_type &value) {
    auto [index, val] = value;
    tuples_[index] = val;
  }

  // template <typename InputIt> void assign_tuples(InputIt first, InputIt last)
  // {
  //   tuples_.assign(first, last);
  // }

  // std::pair<iterator, bool> insert(value_type &&value) {
  //   auto &&[insert_index, insert_value] = value;
  //   for (auto iter = begin(); iter != end(); ++iter) {
  //     auto &&[index, v] = *iter;
  //     if (index == insert_index) {
  //       return {iter, false};
  //     }
  //   }
  //   tuples_.push_back(value);
  //   return {--tuples_.end(), true};
  // }

  // std::pair<iterator, bool> insert(const value_type &value) {
  //   auto &&[insert_index, insert_value] = value;
  //   for (auto iter = begin(); iter != end(); ++iter) {
  //     auto &&[index, v] = *iter;
  //     if (index == insert_index) {
  //       return {iter, false};
  //     }
  //   }
  //   tuples_.push_back(value);
  //   return {--tuples_.end(), true};
  // }

  // template <class M>
  // std::pair<iterator, bool> insert_or_assign(key_type k, M &&obj) {
  //   for (auto iter = begin(); iter != end(); ++iter) {
  //     auto &&[index, v] = *iter;
  //     if (index == k) {
  //       v = std::forward<M>(obj);
  //       return {iter, false};
  //     }
  //   }
  //   tuples_.push_back({k, std::forward<M>(obj)});
  //   return {--tuples_.end(), true};
  // }

  // iterator find(key_type key) noexcept {
  //   return std::find_if(begin(), end(), [&](auto &&v) {
  //     auto &&[i, v_] = v;
  //     return i == key;
  //   });
  // }

  // const_iterator find(key_type key) const noexcept {
  //   return std::find_if(begin(), end(), [&](auto &&v) {
  //     auto &&[i, v_] = v;
  //     return i == key;
  //   });
  // }

  // void reshape(dr::index<I> shape) {
  //   bool all_inside = true;
  //   for (auto &&[index, v] : *this) {
  //     auto &&[i, j] = index;
  //     if (!(i < shape[0] && j < shape[1])) {
  //       all_inside = false;
  //       break;
  //     }
  //   }

  //   if (all_inside) {
  //     shape_ = shape;
  //     return;
  //   } else {
  //     coo_matrix<T, I> new_tuples(shape);
  //     for (auto &&[index, v] : *this) {
  //       auto &&[i, j] = index;
  //       if (i < shape[0] && j < shape[1]) {
  //         new_tuples.insert({index, v});
  //       }
  //     }
  //     shape_ = shape;
  //     assign_tuples(new_tuples.begin(), new_tuples.end());
  //   }
  // }

  coo_matrix() = default;
  ~coo_matrix() = default;
  coo_matrix(const coo_matrix &) = default;
  coo_matrix(coo_matrix &&) = default;
  coo_matrix &operator=(const coo_matrix &) = default;
  coo_matrix &operator=(coo_matrix &&) = default;

  std::size_t nbytes() const noexcept {
    return tuples_.size() * sizeof(value_type);
  }

private:
  dr::index<I> shape_;
  backend_type tuples_;
};

} // namespace __detail

} // namespace dr::sp
