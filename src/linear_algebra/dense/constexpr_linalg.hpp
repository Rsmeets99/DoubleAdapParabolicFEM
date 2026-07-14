#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

// ============================================================================
// Small fixed-size linear algebra utilities for FEM / polynomial basis code.
//
// QR factorization uses Householder reflections with column pivoting:
//
//     A * P = Q * R
//
// where perm[j] is the original column index currently occupying column j
// in the permuted matrix A*P.
//
// For solving A x = b:
//   1. Factor A*P = Q*R
//   2. Solve R z = Q^T b
//   3. Recover x by x[perm[j]] = z[j]
//
// Everything is written to be constexpr-friendly in C++20.
// ============================================================================

namespace cdla // constexpr dense linear algebra
{

// ---------------------------------
// Basic fixed-size vector/matrix
// ---------------------------------
template<std::size_t N>
using Vec = std::array<double, N>;

template<std::size_t R, std::size_t C>
using Mat = std::array<std::array<double, C>, R>;

// ---------------------------------
// Scalar helpers
// ---------------------------------
constexpr double cabs(double x)
{
    return (x < 0.0) ? -x : x;
}

constexpr double cmax(double a, double b)
{
    return (a > b) ? a : b;
}

constexpr double cmin(double a, double b)
{
    return (a < b) ? a : b;
}

// High-accuracy constexpr sqrt using Newton-Raphson
constexpr double constexpr_sqrt(double x, double guess = -1.0, int iter = 0)
{
    if (x < 0.0) return -1.0;
    if (x == 0.0) return 0.0;

    if (guess < 0.0)
        guess = (x >= 1.0) ? x : 1.0;

    const double next = 0.5 * (guess + x / guess);

    if (iter >= 50 || next == guess || cabs(next - guess) <= 1e-15 * next)
        return next;

    return constexpr_sqrt(x, next, iter + 1);
}

constexpr double my_sqrt(double x)
{
    if (std::is_constant_evaluated())
        return constexpr_sqrt(x);
    else
        return std::sqrt(x);
}

// ---------------------------------
// Matrix constructors
// ---------------------------------
template<std::size_t N>
constexpr Mat<N,N> identity_matrix()
{
    Mat<N,N> I{};
    for (std::size_t i = 0; i < N; ++i)
        I[i][i] = 1.0;
    return I;
}

// ---------------------------------
// Vector helpers
// ---------------------------------
template<std::size_t N>
constexpr double vec_dot(const Vec<N>& a, const Vec<N>& b)
{
    double s = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        s += a[i] * b[i];
    return s;
}

template<std::size_t N>
constexpr Vec<N> vec_add(const Vec<N>& a, const Vec<N>& b)
{
    Vec<N> r{};
    for (std::size_t i = 0; i < N; ++i)
        r[i] = a[i] + b[i];
    return r;
}

template<std::size_t N>
constexpr Vec<N> vec_sub(const Vec<N>& a, const Vec<N>& b)
{
    Vec<N> r{};
    for (std::size_t i = 0; i < N; ++i)
        r[i] = a[i] - b[i];
    return r;
}

template<std::size_t N>
constexpr Vec<N> vec_scale(const Vec<N>& v, double s)
{
    Vec<N> r{};
    for (std::size_t i = 0; i < N; ++i)
        r[i] = s * v[i];
    return r;
}

template<std::size_t N>
constexpr double vec_norm_sq(const Vec<N>& v)
{
    return vec_dot(v, v);
}

template<std::size_t N>
constexpr double vec_norm(const Vec<N>& v)
{
    return my_sqrt(vec_norm_sq(v));
}

template<std::size_t N>
constexpr double vec_inf_norm(const Vec<N>& v)
{
    double m = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        m = cmax(m, cabs(v[i]));
    return m;
}

// ---------------------------------
// Matrix helpers
// ---------------------------------
template<std::size_t N>
constexpr Vec<N> matvec_mul(const Mat<N,N>& A, const Vec<N>& x)
{
    Vec<N> y{};
    for (std::size_t i = 0; i < N; ++i)
    {
        double sum = 0.0;
        for (std::size_t j = 0; j < N; ++j)
            sum += A[i][j] * x[j];
        y[i] = sum;
    }
    return y;
}

template<std::size_t R, std::size_t K, std::size_t C>
constexpr Mat<R,C> matmul(const Mat<R,K>& A, const Mat<K,C>& B)
{
    Mat<R,C> Cmat{};
    for (std::size_t i = 0; i < R; ++i)
    {
        for (std::size_t j = 0; j < C; ++j)
        {
            double s = 0.0;
            for (std::size_t k = 0; k < K; ++k)
                s += A[i][k] * B[k][j];
            Cmat[i][j] = s;
        }
    }
    return Cmat;
}

template<std::size_t R, std::size_t C>
constexpr Mat<C,R> transpose(const Mat<R,C>& A)
{
    Mat<C,R> T{};
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            T[j][i] = A[i][j];
    return T;
}

template<std::size_t R, std::size_t C>
constexpr Mat<R,C> mat_add(const Mat<R,C>& A, const Mat<R,C>& B)
{
    Mat<R,C> M{};
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            M[i][j] = A[i][j] + B[i][j];
    return M;
}

template<std::size_t R, std::size_t C>
constexpr Mat<R,C> mat_sub(const Mat<R,C>& A, const Mat<R,C>& B)
{
    Mat<R,C> M{};
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            M[i][j] = A[i][j] - B[i][j];
    return M;
}

template<std::size_t R, std::size_t C>
constexpr double mat_frobenius_norm_sq(const Mat<R,C>& A)
{
    double s = 0.0;
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j)
            s += A[i][j] * A[i][j];
    return s;
}

template<std::size_t R, std::size_t C>
constexpr double mat_frobenius_norm(const Mat<R,C>& A)
{
    return my_sqrt(mat_frobenius_norm_sq(A));
}

template<std::size_t N>
constexpr Mat<N,N> permute_columns(const Mat<N,N>& A,
                                   const std::array<std::size_t, N>& perm)
{
    Mat<N,N> AP{};
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            AP[i][j] = A[i][perm[j]];
    return AP;
}

template<std::size_t N>
constexpr Vec<N> unpermute_solution(const Vec<N>& z,
                                    const std::array<std::size_t, N>& perm)
{
    Vec<N> x{};
    for (std::size_t j = 0; j < N; ++j)
        x[perm[j]] = z[j];
    return x;
}

template<std::size_t N>
constexpr double residual_norm(const Mat<N,N>& A,
                               const Vec<N>& x,
                               const Vec<N>& b)
{
    return vec_norm(vec_sub(matvec_mul(A, x), b));
}

// ---------------------------------
// Pivoted QR factorization
// ---------------------------------
template<std::size_t N>
struct QRFactorization
{
    Mat<N,N> A{};                              // upper triangle = R, lower = Householder vectors
    Vec<N> tau{};                              // Householder scalars
    std::array<std::size_t, N> perm{};         // column permutation
};

template<std::size_t N>
constexpr QRFactorization<N> qr_factorize(Mat<N,N> A)
{
    QRFactorization<N> qr{};
    qr.A = A;

    for (std::size_t j = 0; j < N; ++j)
        qr.perm[j] = j;

    for (std::size_t k = 0; k < N; ++k)
    {
        // Column pivoting: choose column with largest trailing 2-norm.
        std::size_t pivot_col = k;
        double pivot_norm_sq = -1.0;

        for (std::size_t j = k; j < N; ++j)
        {
            double nrm_sq = 0.0;
            for (std::size_t i = k; i < N; ++i)
                nrm_sq += qr.A[i][j] * qr.A[i][j];

            if (nrm_sq > pivot_norm_sq)
            {
                pivot_norm_sq = nrm_sq;
                pivot_col = j;
            }
        }

        if (pivot_col != k)
        {
            for (std::size_t i = 0; i < N; ++i)
            {
                const double tmp = qr.A[i][k];
                qr.A[i][k] = qr.A[i][pivot_col];
                qr.A[i][pivot_col] = tmp;
            }

            const std::size_t tmp_p = qr.perm[k];
            qr.perm[k] = qr.perm[pivot_col];
            qr.perm[pivot_col] = tmp_p;
        }

        double norm_sq = 0.0;
        for (std::size_t i = k; i < N; ++i)
            norm_sq += qr.A[i][k] * qr.A[i][k];

        const double norm = my_sqrt(norm_sq);

        if (norm == 0.0)
        {
            qr.tau[k] = 0.0;
            continue;
        }

        const double akk   = qr.A[k][k];
        const double alpha = (akk >= 0.0) ? -norm : norm;
        const double denom = akk - alpha;

        for (std::size_t i = k + 1; i < N; ++i)
            qr.A[i][k] /= denom;

        qr.tau[k] = (alpha - akk) / alpha;
        qr.A[k][k] = alpha;

        for (std::size_t j = k + 1; j < N; ++j)
        {
            double dot = qr.A[k][j];
            for (std::size_t i = k + 1; i < N; ++i)
                dot += qr.A[i][k] * qr.A[i][j];

            dot *= qr.tau[k];

            qr.A[k][j] -= dot;
            for (std::size_t i = k + 1; i < N; ++i)
                qr.A[i][j] -= dot * qr.A[i][k];
        }
    }

    return qr;
}

// Apply Q^T to b in place.
template<std::size_t N>
constexpr void qr_apply_qt(const QRFactorization<N>& qr,
                           Vec<N>& b)
{
    for (std::size_t k = 0; k < N; ++k)
    {
        double dot = b[k];
        for (std::size_t i = k + 1; i < N; ++i)
            dot += qr.A[i][k] * b[i];

        dot *= qr.tau[k];

        b[k] -= dot;
        for (std::size_t i = k + 1; i < N; ++i)
            b[i] -= dot * qr.A[i][k];
    }
}

// Apply Q to b in place.
template<std::size_t N>
constexpr void qr_apply_q(const QRFactorization<N>& qr,
                          Vec<N>& b)
{
    for (std::size_t ii = N; ii-- > 0; )
    {
        const std::size_t k = ii;

        double dot = b[k];
        for (std::size_t i = k + 1; i < N; ++i)
            dot += qr.A[i][k] * b[i];

        dot *= qr.tau[k];

        b[k] -= dot;
        for (std::size_t i = k + 1; i < N; ++i)
            b[i] -= dot * qr.A[i][k];
    }
}

template<std::size_t N>
constexpr Mat<N,N> qr_form_q(const QRFactorization<N>& qr)
{
    Mat<N,N> Q{};
    for (std::size_t col = 0; col < N; ++col)
    {
        Vec<N> e{};
        e[col] = 1.0;
        qr_apply_q(qr, e);
        for (std::size_t i = 0; i < N; ++i)
            Q[i][col] = e[i];
    }
    return Q;
}

template<std::size_t N>
constexpr Mat<N,N> qr_form_r(const QRFactorization<N>& qr)
{
    Mat<N,N> R{};
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i; j < N; ++j)
            R[i][j] = qr.A[i][j];
    return R;
}

template<std::size_t N>
constexpr double qr_reconstruction_error(const QRFactorization<N>& qr,
                                         const Mat<N,N>& A_orig)
{
    const Mat<N,N> Q  = qr_form_q(qr);
    const Mat<N,N> R  = qr_form_r(qr);
    const Mat<N,N> AP = permute_columns(A_orig, qr.perm);
    return mat_frobenius_norm(mat_sub(matmul(Q, R), AP));
}

// ---------------------------------
// Checked triangular solve
// ---------------------------------
template<std::size_t N>
constexpr bool try_back_substitute(const QRFactorization<N>& qr,
                                   const Vec<N>& y,
                                   Vec<N>& x,
                                   double rel_tol = 1e-12)
{
    x = {};

    double max_diag = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        max_diag = cmax(max_diag, cabs(qr.A[i][i]));

    const double thresh = (max_diag > 0.0) ? rel_tol * max_diag : rel_tol;

    for (std::size_t ii = N; ii-- > 0; )
    {
        const std::size_t i = ii;
        const double rii = qr.A[i][i];

        if (cabs(rii) <= thresh)
            return false;

        double s = y[i];
        for (std::size_t j = i + 1; j < N; ++j)
            s -= qr.A[i][j] * x[j];

        x[i] = s / rii;
    }

    return true;
}

template<std::size_t N>
constexpr Vec<N> back_substitute(const QRFactorization<N>& qr,
                                 const Vec<N>& y,
                                 double rel_tol = 1e-12)
{
    Vec<N> x{};
    (void)try_back_substitute(qr, y, x, rel_tol);
    return x;
}

// ---------------------------------
// Checked pivoted QR solve
// ---------------------------------
template<std::size_t N>
constexpr bool try_qr_solve(const QRFactorization<N>& qr,
                            const Vec<N>& b,
                            Vec<N>& x,
                            double rel_tol = 1e-12)
{
    Vec<N> y = b;
    qr_apply_qt(qr, y);

    Vec<N> z{};
    if (!try_back_substitute(qr, y, z, rel_tol))
    {
        x = {};
        return false;
    }

    x = unpermute_solution(z, qr.perm);
    return true;
}

template<std::size_t N>
constexpr Vec<N> qr_solve(const QRFactorization<N>& qr,
                          const Vec<N>& b,
                          double rel_tol = 1e-12)
{
    Vec<N> x{};
    (void)try_qr_solve(qr, b, x, rel_tol);
    return x;
}

// ---------------------------------
// Inverse via repeated solves
// ---------------------------------
template<std::size_t N>
constexpr Mat<N,N> qr_inverse(const QRFactorization<N>& qr,
                              double rel_tol = 1e-12)
{
    Mat<N,N> inv{};

    for (std::size_t col = 0; col < N; ++col)
    {
        Vec<N> e{};
        e[col] = 1.0;

        const Vec<N> x = qr_solve(qr, e, rel_tol);

        for (std::size_t i = 0; i < N; ++i)
            inv[i][col] = x[i];
    }

    return inv;
}

template<std::size_t N>
constexpr bool try_qr_inverse(
    const QRFactorization<N>& qr,
    Mat<N,N>& inv,
    double rel_tol = 1e-12)
{
    inv = {};

    for (std::size_t col = 0; col < N; ++col)
    {
        Vec<N> e{};
        e[col] = 1.0;

        Vec<N> x{};
        if (!try_qr_solve(qr, e, x, rel_tol))
        {
            inv = {};
            return false;
        }

        for (std::size_t i = 0; i < N; ++i)
            inv[i][col] = x[i];
    }

    return true;
}

} // namespace cdla
