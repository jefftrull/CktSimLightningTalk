// Functions for manipulating circuit values via MNA in Eigen
// Author: Jeff Trull <edaskel@att.net>

/*
Copyright (c) 2014 Jeffrey E. Trull

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <Eigen/Dense>

// "stamp" functions for adding components to a circuit
template<int sz>
void stamp_r(Eigen::Matrix<double, sz, sz>& G,
             Eigen::Matrix<double, sz, sz> const&,
             int node1, int node2, double r) {
    // You can think of this as KCL at the two nodes the resistor connects
    G(node1, node1) += 1.0/r;
    G(node1, node2) -= 1.0/r;
    G(node2, node2) += 1.0/r;
    G(node2, node1) -= 1.0/r;
}

// ground lumped variant
template<int sz>
void stamp_r(Eigen::Matrix<double, sz, sz>& G,
             Eigen::Matrix<double, sz, sz> const&,
             int node, double r) {
    G(node, node) += 1.0/r;
}

// voltage source inputs and inductors get this treatment:
template<int sz>
void stamp_i(Eigen::Matrix<double, sz, sz>& G,
             Eigen::Matrix<double, sz, sz> const&,
             int node1, int node2, int istate) {
    G(node1, istate) =  1;
    G(istate, node1) = -1;
    G(node2, istate) = -1;
    G(istate, node2) =  1;
}

template<int sz>
void stamp_i(Eigen::Matrix<double, sz, sz>& G,
             Eigen::Matrix<double, sz, sz> const&,
             int node, int istate) {
    G(node, istate) =  1;
    G(istate, node) = -1;
}

template<int sz>
void stamp_c(Eigen::Matrix<double, sz, sz> const&,
             Eigen::Matrix<double, sz, sz>& C,
             int node1, int node2, double c) {
    C(node1, node1) += c;
    C(node1, node2) -= c;
    C(node2, node2) += c;
    C(node2, node1) -= c;
}

template<int sz>
void stamp_c(Eigen::Matrix<double, sz, sz> const&,
             Eigen::Matrix<double, sz, sz>& C,
             int node, double c) {
    C(node, node) += c;  // assumes other terminal is ground
}

template<int sz>
void stamp_l(Eigen::Matrix<double, sz, sz>& G,
             Eigen::Matrix<double, sz, sz>& C,
             int node1, int node2, int istate, double l) {
    C(istate, istate) += l;
    // For inductors we have an extra state that remembers its current
    stamp_i(G, C, node1, node2, istate);
}

template<int sz>
void stamp_l(Eigen::Matrix<double, sz, sz>& G,
             Eigen::Matrix<double, sz, sz>& C,
             int node, int istate, double l) {
    C(istate, istate)  +=  l;
    stamp_i(G, C, node, istate);
}

template<class M>
bool isSingular(const M& m) {
   // A singular matrix has at least one zero eigenvalue -
   // in theory, at least... but due to machine precision we can have "nearly singular"
   // matrices that misbehave.  Comparing rank instead is safer because it uses thresholds
   // for near-zero values.

   assert(m.rows() == m.cols());   // singularity has no meaning for a non-square matrix
   return (m.fullPivLu().rank() != m.rows());

}

// Calculate moments of given system in MNA form
template<typename Float, int nrows, int ncols>
using MatrixVector = std::vector<Eigen::Matrix<Float, nrows, ncols>,
                                 Eigen::aligned_allocator<Eigen::Matrix<Float, nrows, ncols> > >;


template<int icount, int ocount, int scount, typename Float = double>
MatrixVector<Float, ocount, icount>
moments(Eigen::Matrix<Float, scount, scount> const & G,
        Eigen::Matrix<Float, scount, scount> const & C,
        Eigen::Matrix<Float, scount, icount> const & B,
        Eigen::Matrix<Float, scount, ocount> const & L,
        Eigen::Matrix<Float, ocount, icount> const & E,
        size_t count) {
    using namespace Eigen;

    MatrixVector<Float, ocount, icount> result;

    auto G_QR = G.fullPivHouseholderQr();
    Matrix<Float, scount, scount> A = -G_QR.solve(C);
    Matrix<Float, scount, icount> R = G_QR.solve(B);

    result.push_back(L.transpose() * R + E);   // incorporate feedthrough into first moment
    Matrix<Float, scount, scount> AtotheI = A;
    for (size_t i = 1; i < count; ++i) {
        result.push_back(L.transpose() * AtotheI * R);
        AtotheI = A * AtotheI;
    }

    return result;
}

// Implementation of Natarajan regularization
// Each iteration of this process can produce an input derivative term that we subsequently
// absorb into the state variable once the process is complete.  This means potentially
// a series of input derivative coefficients (B's).  We hide that from the users by delegating here:

template<int icount, int ocount, int scount, typename Float = double>
std::tuple<Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>,   // G result
           Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>,   // C result
           Eigen::Matrix<Float, Eigen::Dynamic, icount>,    // B result
           Eigen::Matrix<Float, Eigen::Dynamic, ocount>,    // D result
           Eigen::Matrix<Float, ocount, icount> >    // E result (feedthrough)
regularize(Eigen::Matrix<Float, scount, scount> const & G,
           Eigen::Matrix<Float, scount, scount> const & C,
           MatrixVector<Float, scount, icount> const & B, // in decreasing order of derived-ness
           Eigen::Matrix<Float, scount, ocount> const & D) {

    // Implements the algorithm in [Natarajan]
    // Circuits, Devices and Systems, IEE Proceedings G, June 1991

    using namespace Eigen;
    typedef Matrix<Float, Dynamic, Dynamic> MatrixD;

    // Step 1: put C into "Row Echelon" form by performing LU factorization
    auto lu = C.fullPivLu();
    auto k = lu.rank();
    if (k == C.rows()) {
        // C is already non-singular
        Matrix<Float, ocount, icount>   E = Matrix<Float, ocount, icount>::Zero();
        return std::make_tuple(G, C, B.back(), D, E);
    }

    MatrixD U = lu.matrixLU().template triangularView<Upper>();
    MatrixD L = lu.matrixLU().template triangularView<UnitLower>();

    // Step 2: "perform the same elementary operations on G and B"
    // given that C = P.inverse() * L * U * Q.inverse()
    // (from source it seems that permutationP/Q is inverse)
    // then to get the new G we reverse those operations:
    auto Cprime = U;   // note we may have small non-zero values in bottom rows, but they will be ignored
    auto P = lu.permutationP();
    auto Q = lu.permutationQ();

    assert(!isSingular(L));
    MatrixD Gprime = L.fullPivLu().solve(P * G * Q);                   // rows and columns
    MatrixVector<Float, scount, icount> Bprime;
    std::transform(B.begin(), B.end(), std::back_inserter(Bprime),
                   [L, P](Matrix<Float, scount, icount> const& b) -> Matrix<Float, scount, icount> {
                       return L.fullPivLu().solve(P * b);              // rows only
                   });

    // The D input is like L in PRIMA but this algorithm uses the transpose
    Matrix<Float, ocount, scount> Dprime = D.transpose() * Q;          // columns only

    // Step 3: "Convert [G21 G22] matrix into row echelon form starting from the last row"
    MatrixD Cnew, Gnew, Dnew;
    MatrixVector<Float, scount, icount> Bnew;

    if (Cprime.rows() == (k+1)) {
        // if G22 is only a single row, there is no point attempting to decompose it
        Cnew = Cprime; Gnew = Gprime; Bnew = Bprime; Dnew = Dprime;
    } else {
        // decompose the bottom rows
        // Upon close review of the first example in the paper, the author is not only
        // converting from the last row, but also *from the last column*, i.e., he
        // performs a standard gaussian elimination on the matrix rotated 180 degrees

        // Plan of attack: reverse G2, perform LU decomposition, reverse result, reverse permutations
        MatrixD G2R = Gprime.bottomRows(C.rows() - k).reverse();

        auto G2R_LU = G2R.fullPivLu();
        MatrixD G2R_U = (G2R_LU.matrixLU().template triangularView<Upper>());

        // Since the order of the rows is irrelevant, I'll perform the decomposition, then
        // combine reversing the rows with the row reordering the LU decomposition produces

        MatrixD exchange_columns = G2R_LU.permutationQ();
        Gnew = Gprime * exchange_columns.reverse();

        // insert already-permuted rows that came from LU, but in reverse order
        Gnew.block(k, 0, Gprime.rows() - k, Gprime.cols()) = G2R_U.reverse();

        // Step 4: "Carry out the same row operations in the B matrix"
        // Note: not necessary to do it for C, because all coefficients are zero in those rows

        // 4.1 reverse the rows in B2
        typedef PermutationMatrix<Dynamic, Dynamic, std::size_t> PermutationD;
        PermutationD reverse_rows;                // order of rows is completely reversed
        reverse_rows.setIdentity(G2R.rows());     // start with null permutation
        for (std::size_t i = 0; i < (G2R.rows() / 2); ++i) {
            reverse_rows.applyTranspositionOnTheRight(i, (G2R.rows()-1) - i);
        }

        // 4.2 extract and apply L operation from reversed G2
        MatrixD G2R_L = G2R_LU.matrixLU().leftCols(G2R.rows()).template triangularView<UnitLower>();
        std::transform(Bprime.begin(), Bprime.end(), std::back_inserter(Bnew),
                       [reverse_rows, k, G2R_L, G2R_LU]
                       (Matrix<Float, scount, icount> const& bp) {
                           MatrixD B2R = reverse_rows * bp.bottomRows(bp.rows() - k);
                           Matrix<Float, scount, icount> bn = bp;
                           bn.block(k, 0, bn.rows() - k, bn.cols()) =
                               reverse_rows.transpose() * G2R_L.fullPivLu().solve(G2R_LU.permutationP() * B2R);
                           return bn;
                       });

        // Step 5: "Interchange the columns in the G, C, and D matrices... such that G22 is non-singular"
        // Since we have done a full pivot factorization of G2 I assume G22 is already non-singular,
        // so the only thing left to do is reorder the C and D matrices according to the G2 factorization
        Cnew = Cprime * exchange_columns.reverse();
        Dnew = Dprime * exchange_columns.reverse();
    }

    // Step 6: compute reduced matrices using equations given in paper
    MatrixD G11 = Gnew.topLeftCorner(k, k);
    MatrixD G12 = Gnew.topRightCorner(k, Gnew.rows() - k);
    MatrixD G21 = Gnew.bottomLeftCorner(Gnew.rows() - k, k);
    MatrixD G22 = Gnew.bottomRightCorner(Gnew.rows() - k, Gnew.rows() - k);
    MatrixD C11 = Cnew.topLeftCorner(k, k);
    MatrixD C12 = Cnew.topRightCorner(k, Cnew.rows() - k);
    MatrixD D01 = Dnew.leftCols(k);
    MatrixD D02 = Dnew.rightCols(Dnew.cols() - k);

    assert(!isSingular(G22));
    auto    G22_LU = G22.fullPivLu();

    MatrixD Gfinal  = G11 - G12 * G22_LU.solve(G21);
    MatrixD Cfinal  = C11 - C12 * G22_LU.solve(G21);
    Matrix<Float, ocount, Dynamic> Dfinal
                    = D01 - D02 * G22_LU.solve(G21);

    Matrix<Float, Dynamic, icount> B02 = Bnew.back().bottomRows(Bnew.back().rows() - k);
    Matrix<Float, ocount, icount> E1
                    =       D02 * G22_LU.solve(B02);

    // reduce the entire series of B's to the new size
    // Performing the same substitution as in Natarajan beginning with eqn [5]
    // but with additional input derivatives present.  Adding B11/B12 multiplying a first
    // derivative of Ws demonstrates that each additional input derivative term contributes:
    // Bn1 - G12 * G22^-1 * Bn2  to its own term, and
    //     - C12 * G22^-1 * Bn2  to the derivative n+1 coefficient,
    // once reduced.
    MatrixVector<Float, Dynamic, icount> Btrans;
    // n+1's first (equation 9d)
    std::transform(Bnew.begin(), Bnew.end(), std::back_inserter(Btrans),
                   [k, G12, C12, G22_LU](Matrix<Float, scount, icount> const& Bn) {
                       Matrix<Float, Dynamic, icount> Bn2 = Bn.bottomRows(Bn.rows() - k);
                       return -C12 * G22_LU.solve(Bn2);
                   });
    Btrans.push_back(Matrix<Float, Dynamic, icount>::Zero(k, icount));  // contribution from n-1 is 0 (nonexistent)

    // n's next, shifted by one (equation 9c)
    std::transform(Bnew.begin(), Bnew.end(), Btrans.begin()+1, Btrans.begin()+1,
                   [k, G12, G22_LU](Matrix<Float, scount, icount> const& Bn,
                                         Matrix<Float, Dynamic, icount> const& Bnm1_contribution)
                   -> Matrix<Float, Dynamic, icount> {  // without explicitly declared return type Eigen
                                                        // will keep references to these locals:
                       Matrix<Float, Dynamic, icount> Bn1 = Bn.topRows(k);
                       Matrix<Float, Dynamic, icount> Bn2 = Bn.bottomRows(Bn.rows() - k);

                       return Bn1 - G12 * G22_LU.solve(Bn2) + Bnm1_contribution;
                   });

    // If Cfinal is singular, we need to repeat this analysis on the new matrices
    if (isSingular(Cfinal)) {
        Matrix<Float, Dynamic, ocount> Dtrans = Dfinal.transpose();   // no implicit conversion on fn tmpl args
        auto recursive_result = regularize<icount, ocount, Dynamic>(Gfinal, Cfinal, Btrans, Dtrans);
        return std::make_tuple(std::get<0>(recursive_result),  // G
                               std::get<1>(recursive_result),  // C
                               std::get<2>(recursive_result),  // B
                               std::get<3>(recursive_result),  // D
                               std::get<4>(recursive_result) + E1);  // combine E
    }

    // We've found a non-singular Cfinal and a set of B's
    // We need to apply a transformation suggested by Chen (TCAD July 2012) to eliminate
    // all input derivative terms.  Chen gives only the simplest case, for B0 * Ws + B1 * Ws' :
    // Br = B0 - Gr * Cr^-1 * B1
    // based on a variable substitution of:
    // Xnew = X - Cr^-1 * B1 * Ws
    // and mentions the rest should be done "recursively".  I believe the general case is:
    // Br = B0 - Gr * Cr^-1 * (B1 - Gr * Cr^-1 *(B2 - ... ))
    Matrix<Float, Dynamic, icount> Bfinal = Matrix<Float, Dynamic, icount>::Zero(k, icount);
    Bfinal = std::accumulate(
        // starting with the first (most derived) coefficient, compute above expression for Br:
        Btrans.begin(), Btrans.end(), Bfinal,
        [Gfinal, Cfinal](Matrix<Float, Dynamic, icount> const& acc,
                         Matrix<Float, Dynamic, icount> const& B) {
            return B - Gfinal * Cfinal.fullPivHouseholderQr().solve(acc);
        });

    // The variable substitution for the 2nd derivative case is:
    // Xnew = X - Cr^-1 * (B2 * Ws' - (Gr * Cr^-1 * B2 - B1) * Ws)
    // Making this substitution in the output equation Y = D * X + E * Ws gives
    // Y = D * Xnew + D * Cr^-1 * (B1 - Gr * Cr^-1 * B2) * Ws + Cr^-1 * B2 * Ws'
    // however, if the Ws' term is nonzero the system is ill-formed:
    if (Btrans.size() >= 3) {
        Matrix<Float, Dynamic, icount> CinvB = Cfinal.fullPivLu().solve(*(Btrans.rbegin()+2));
        assert(CinvB.isZero());
    }

    // now I can calculate the new value for E, which can only be:
    // E = E1 + D * Cr^-1 * B1
    // because, thanks to the assertion, all other terms must be 0
    Matrix<Float, ocount, icount> Efinal = E1 + Dfinal * Cfinal.fullPivHouseholderQr().solve(*(Btrans.rbegin()+1));

    return std::make_tuple(Gfinal, Cfinal, Bfinal,
                           Dfinal.transpose(),  // for PRIMA compatibility
                           Efinal);
}

// user-facing function (only one "B" parameter)
template<int icount, int ocount, int scount, typename Float = double>
std::tuple<Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>,   // G result
           Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>,   // C result
           Eigen::Matrix<Float, Eigen::Dynamic, icount>,    // B result
           Eigen::Matrix<Float, Eigen::Dynamic, ocount>,    // D result
           Eigen::Matrix<Float, ocount, icount> >    // E result (feedthrough)
regularize(Eigen::Matrix<Float, scount, scount> const & G,
           Eigen::Matrix<Float, scount, scount> const & C,
           Eigen::Matrix<Float, scount, icount> const & B,
           Eigen::Matrix<Float, scount, ocount> const & D) {
    return regularize(G, C, MatrixVector<Float, scount, icount>(1, B), D);
}
