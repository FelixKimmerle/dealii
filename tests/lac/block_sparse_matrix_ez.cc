// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2009 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#include <deal.II/lac/block_sparse_matrix_ez.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <fstream>
#include <iomanip>

using namespace dealii;

namespace
{
  using size_type = types::global_dof_index;

  template <typename Number>
  void
  reinit_ez(SparseMatrixEZ<Number> &m,
            const size_type         n_rows,
            const size_type         n_cols,
            const unsigned int      max_entries_per_row)
  {
    m.reinit(n_rows, n_cols, max_entries_per_row);
  }

  template <typename Number>
  void
  setup_blocks(BlockSparseMatrixEZ<Number>  &M,
               const std::vector<size_type> &row_block_sizes,
               const std::vector<size_type> &col_block_sizes,
               const unsigned int            max_entries_per_row)
  {
    const unsigned int n_br = row_block_sizes.size();
    const unsigned int n_bc = col_block_sizes.size();

    M.reinit(n_br, n_bc);

    for (unsigned int br = 0; br < n_br; ++br)
      for (unsigned int bc = 0; bc < n_bc; ++bc)
        reinit_ez(M.block(br, bc),
                  row_block_sizes[br],
                  col_block_sizes[bc],
                  max_entries_per_row);

    M.collect_sizes();
  }

  // Fill a sparse pattern with at most two entries per row:
  // For row i, put entries at c1 and c2:
  //   c1 = i % n_cols
  //   c2 = (i+1) % n_cols   (if n_cols > 1)
  // with values:
  //   a(i,c1) = 1 + i
  //   a(i,c2) = 10 + i
  template <typename Number>
  void
  fill_two_entries_per_row(BlockSparseMatrixEZ<Number> &M)
  {
    const size_type n_rows = M.m();
    const size_type n_cols = M.n();

    for (size_type i = 0; i < n_rows; ++i)
      {
        const size_type c1 = i % n_cols;
        const Number    v1 = Number(1 + i);
        M.set(i, c1, v1);

        if (n_cols > 1)
          {
            const size_type c2 = (i + 1) % n_cols;
            const Number    v2 = Number(10 + i);
            if (c2 != c1)
              M.set(i, c2, v2);
            else
              M.add(i, c2, v2);
          }
      }
  }

  template <typename Number>
  Vector<Number>
  reference_vmult(const BlockSparseMatrixEZ<Number> &M, const Vector<Number> &x)
  {
    const size_type n_rows = M.m();
    const size_type n_cols = M.n();

    AssertThrow(x.size() == n_cols, ExcInternalError());

    Vector<Number> y(n_rows);
    for (size_type i = 0; i < n_rows; ++i)
      {
        Number          sum = Number();
        const size_type c1  = i % n_cols;
        const Number    v1  = Number(1 + i);
        sum += v1 * x(c1);

        if (n_cols > 1)
          {
            const size_type c2 = (i + 1) % n_cols;
            const Number    v2 = Number(10 + i);
            sum += v2 * x(c2);
          }

        y(i) = sum;
      }
    return y;
  }

  template <typename Number>
  Vector<Number>
  reference_Tvmult(const BlockSparseMatrixEZ<Number> &M,
                   const Vector<Number>              &x)
  {
    const size_type n_rows = M.m();
    const size_type n_cols = M.n();

    AssertThrow(x.size() == n_rows, ExcInternalError());

    Vector<Number> y(n_cols);
    y = 0;

    for (size_type i = 0; i < n_rows; ++i)
      {
        const size_type c1 = i % n_cols;
        const Number    v1 = Number(1 + i);
        y(c1) += v1 * x(i);

        if (n_cols > 1)
          {
            const size_type c2 = (i + 1) % n_cols;
            const Number    v2 = Number(10 + i);
            y(c2) += v2 * x(i);
          }
      }
    return y;
  }

  template <typename VectorType>
  void
  fill_linear(VectorType &v)
  {
    for (size_type i = 0; i < v.size(); ++i)
      v(i) = static_cast<double>(i + 1);
  }

  template <typename VectorType>
  void
  print_vector(const std::string &label, const VectorType &v)
  {
    deallog << label;
    for (size_type i = 0; i < v.size(); ++i)
      deallog << ' ' << v(i);
    deallog << std::endl;
  }

  void
  test_get_set_add_and_scaling()
  {
    deallog.push("get/set/add/scalar");

    BlockSparseMatrixEZ<double> M;

    // 2x2 blocks; total size 5x5
    setup_blocks<double>(M,
                         /*row blocks*/ {2, 3},
                         /*col blocks*/ {3, 2},
                         /*max entries/row*/ 4);

    fill_two_entries_per_row(M);

    // Basic "get" checks (via operator()(i,j)):
    deallog << "M(0,0) " << M(0, 0) << std::endl;
    deallog << "M(0,1) " << M(0, 1) << std::endl;
    deallog << "M(3,3) " << M(3, 3) << std::endl;
    deallog << "M(3,4) " << M(3, 4) << std::endl;

    // Entry add:
    M.add(3, 4, 5.0); // modifies an entry that is present in our pattern
    deallog << "after add M(3,4) " << M(3, 4) << std::endl;

    // Scalar mult/div:
    M *= 2.0;
    deallog << "after *=2 M(3,4) " << M(3, 4) << std::endl;

    M /= 4.0; // net scale = 1/2
    deallog << "after /=4 M(3,4) " << M(3, 4) << std::endl;

    // Also check that scaling affects vmult consistently:
    BlockVector<double> x, y;
    x.reinit({3, 2}); // col blocks
    y.reinit({2, 3}); // row blocks
    fill_linear(x);

    M.vmult(y, x);
    print_vector("vmult_scaled", y);

    deallog.pop();
  }

  void
  test_vmult_and_Tvmult_block_block()
  {
    deallog.push("vmult/Tvmult block-block");

    BlockSparseMatrixEZ<double> M;
    setup_blocks<double>(M, {2, 3}, {3, 2}, 4);
    fill_two_entries_per_row(M);

    BlockVector<double> x, y;
    x.reinit({3, 2}); // columns
    y.reinit({2, 3}); // rows
    fill_linear(x);

    M.vmult(y, x);

    Vector<double> x_flat(M.n());
    for (size_type i = 0; i < x_flat.size(); ++i)
      x_flat(i) = x(i);

    const Vector<double> y_ref = reference_vmult(M, x_flat);

    for (size_type i = 0; i < M.m(); ++i)
      AssertThrow(y(i) == y_ref(i), ExcInternalError());

    print_vector("vmult", y);

    // Tvmult (block-block)
    BlockVector<double> xt, yt;
    xt.reinit({2, 3}); // rows
    yt.reinit({3, 2}); // cols
    fill_linear(xt);

    M.Tvmult(yt, xt);

    Vector<double> xt_flat(M.m());
    for (size_type i = 0; i < xt_flat.size(); ++i)
      xt_flat(i) = xt(i);

    const Vector<double> yt_ref = reference_Tvmult(M, xt_flat);

    for (size_type j = 0; j < M.n(); ++j)
      AssertThrow(yt(j) == yt_ref(j), ExcInternalError());

    print_vector("Tvmult", yt);

    deallog.pop();
  }

  void
  test_vmult_block_nonblock_and_Tvmult_nonblock_block()
  {
    deallog.push("vmult block-Vector + Tvmult Vector-block");

    // 2x1 blocks => one block column
    BlockSparseMatrixEZ<double> M;
    setup_blocks<double>(M, {2, 3}, {5}, 4);
    fill_two_entries_per_row(M);

    Vector<double> x(M.n());
    fill_linear(x);

    BlockVector<double> y;
    y.reinit({2, 3});

    M.vmult(y, x);

    const Vector<double> y_ref = reference_vmult(M, x);
    for (size_type i = 0; i < M.m(); ++i)
      AssertThrow(y(i) == y_ref(i), ExcInternalError());

    print_vector("vmult", y);

    // Tvmult(Vector&, BlockVector&) valid because: one block column
    BlockVector<double> xt;
    xt.reinit({2, 3});
    fill_linear(xt);

    Vector<double> yt(M.n());
    M.Tvmult(yt, xt);

    Vector<double> xt_flat(M.m());
    for (size_type i = 0; i < xt_flat.size(); ++i)
      xt_flat(i) = xt(i);

    const Vector<double> yt_ref = reference_Tvmult(M, xt_flat);
    for (size_type j = 0; j < M.n(); ++j)
      AssertThrow(yt(j) == yt_ref(j), ExcInternalError());

    print_vector("Tvmult", yt);

    deallog.pop();
  }

  void
  test_vmult_nonblock_block_and_Tvmult_block_nonblock()
  {
    deallog.push("vmult Vector-block + Tvmult block-Vector");

    // 1x2 blocks => one block row
    BlockSparseMatrixEZ<double> M;
    setup_blocks<double>(M, {5}, {2, 3}, 4);
    fill_two_entries_per_row(M);

    BlockVector<double> x;
    x.reinit({2, 3});
    fill_linear(x);

    Vector<double> y(M.m());
    M.vmult(y, x);

    Vector<double> x_flat(M.n());
    for (size_type i = 0; i < x_flat.size(); ++i)
      x_flat(i) = x(i);

    const Vector<double> y_ref = reference_vmult(M, x_flat);
    for (size_type i = 0; i < M.m(); ++i)
      AssertThrow(y(i) == y_ref(i), ExcInternalError());

    print_vector("vmult", y);

    // Tvmult(BlockVector&, Vector&) valid because: one block row
    Vector<double> xt(M.m());
    fill_linear(xt);

    BlockVector<double> yt;
    yt.reinit({2, 3});
    M.Tvmult(yt, xt);

    const Vector<double> yt_ref = reference_Tvmult(M, xt);
    for (size_type j = 0; j < M.n(); ++j)
      AssertThrow(yt(j) == yt_ref(j), ExcInternalError());

    print_vector("Tvmult", yt);

    deallog.pop();
  }

  void
  test_vmult_nonblock_nonblock_and_Tvmult_nonblock_nonblock()
  {
    deallog.push("vmult/Tvmult Vector-Vector");

    // 1x1 blocks => single block
    BlockSparseMatrixEZ<double> M;
    setup_blocks<double>(M, {5}, {5}, 4);
    fill_two_entries_per_row(M);

    Vector<double> x(M.n()), y(M.m());
    fill_linear(x);

    M.vmult(y, x);

    const Vector<double> y_ref = reference_vmult(M, x);
    for (size_type i = 0; i < M.m(); ++i)
      AssertThrow(y(i) == y_ref(i), ExcInternalError());

    print_vector("vmult", y);

    Vector<double> xt(M.m()), yt(M.n());
    fill_linear(xt);

    M.Tvmult(yt, xt);

    const Vector<double> yt_ref = reference_Tvmult(M, xt);
    for (size_type j = 0; j < M.n(); ++j)
      AssertThrow(yt(j) == yt_ref(j), ExcInternalError());

    print_vector("Tvmult", yt);

    deallog.pop();
  }

} // namespace



void
test()
{
  std::ofstream logfile("output");
  deallog.attach(logfile);
  deallog << std::fixed << std::setprecision(2);

  test_get_set_add_and_scaling();
  test_vmult_and_Tvmult_block_block();
  test_vmult_block_nonblock_and_Tvmult_nonblock_block();
  test_vmult_nonblock_block_and_Tvmult_block_nonblock();
  test_vmult_nonblock_nonblock_and_Tvmult_nonblock_nonblock();
}



int
main()
{
  try
    {
      test();
    }
  catch (const std::exception &e)
    {
      std::cerr << "\n\n"
                << "----------------------------------------------------\n"
                << "Exception on processing: " << e.what() << "\n"
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 2;
    }
  catch (...)
    {
      std::cerr << "\n\n"
                << "----------------------------------------------------\n"
                << "Unknown exception!\n"
                << "Aborting!\n"
                << "----------------------------------------------------\n";
      return 3;
    }

  return 0;
}
