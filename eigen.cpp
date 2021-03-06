// A numerically integrated implementation of the circuit described in
// "Rolling Your Own Circuit Simulator with Eigen and Boost.ODEInt"
// using Eigen to store and manipulate circuit values
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

#include <iostream>
#include <array>
#include <boost/numeric/odeint.hpp>
#include <Eigen/Dense>

#include "mna.hpp"

typedef std::array<double, 2> state_t;   // 0 = V_out, 1 = I_L

struct circuit {
    circuit(double r, double l, double c) {

        using namespace Eigen;

        typedef Matrix<double, 4, 4> matrix4_t;
        matrix4_t G = matrix4_t::Zero();
        matrix4_t C = matrix4_t::Zero();

        // state assignment: 0, 1 = node voltages; 2 = I_L, 3 = I_in
        stamp_r(G, C, 0, 1, r);
        stamp_c(G, C, 1, c);
        stamp_l(G, C, 1, 2, l);
        stamp_i(G, C, 0, 3);           // V_in, I_in

        // input application vector - connects single input to appropriate equation
        typedef Matrix<double, 4, 1> vector4_t;
        vector4_t B;
        B << 0, 0, 0, -1;

        // output extraction vector - connects state element to output
        Matrix<double, 1, 4> D;
        D << 0, 1, 0, 0;         // state element idx 1 = V_out

        // Now we have C*dX/dt = -G*X + B*u, and the output is = D * X

        // regularize so C is non-singular
        // First perform Gaussian elimination with full pivoting
        // This will put non-zero elements in the upper left, generally
        auto lu_fact = C.fullPivLu();
        matrix4_t Cprime = lu_fact.matrixLU().template triangularView<Upper>();
        
        // repeat process on G matrix and u vector using the factored components
        // LU produces a factorization P^-1 * L * U * Q^-1
        matrix4_t L = lu_fact.matrixLU().template triangularView<UnitLower>();
        matrix4_t P = lu_fact.permutationP();
        matrix4_t Q = lu_fact.permutationQ();
        matrix4_t Gprime = L.fullPivLu().solve(P * G * Q);   // permute rows and columns
        vector4_t Bprime = L.fullPivLu().solve(P * B);       // rows only
        Matrix<double, 1, 4> Dprime = D * Q;

        // Use the "rank" (# independent rows/columns) to determine where to split
        int sz = lu_fact.rank();
        int remainder = 4 - sz;

        // Produce smaller (and hopefully non-singular) matrices for calculation
        MatrixXd Cnew = Cprime.topLeftCorner(sz, sz);   // the rest is zero
        
        // break up Gprime into chunks based on the size of the nonzero portion of Cprime
        MatrixXd G11  = Gprime.topLeftCorner(sz, sz);
        MatrixXd G12  = Gprime.topRightCorner(sz, remainder);
        MatrixXd G21  = Gprime.bottomLeftCorner(remainder, sz);
        MatrixXd G22  = Gprime.bottomRightCorner(remainder, remainder);
        
        // solve for the (sz) state variables we will retain
        auto G22LU = G22.fullPivLu();
        MatrixXd Gnew = G11 - G12 * G22LU.solve(G21);

        // adjust input and output connections
        Matrix<double, Dynamic, 1> Bnew;
        Bnew = Bprime.topRows(sz) - G12 * G22LU.solve(Bprime.bottomRows(remainder));
        Matrix<double, 1, Dynamic> Dnew;
        Dnew = Dprime.leftCols(sz) - Dprime.rightCols(remainder) * G22LU.solve(G21);

        // verify the new C is non-singular
        assert(Cnew.rows() == Cnew.fullPivLu().rank());

        // factor Cnew out of our equation by multiplying both sides by Cnew^-1
        // new equation will be dX/dt = - Cnew^-1 * Gnew * X + Cnew^-1 * Bnew * u
        drift_term_ = - Cnew.fullPivLu().solve(Gnew);
        input_term_ =   Cnew.fullPivLu().solve(Bnew);

        // Vout may have been moved in the reduction process, so we must supply a map
        s2o_        = Dnew;
    }

    double state2output(state_t const& x) const {
        Eigen::Map<const Eigen::Matrix<double, 2, 1> > xvec(x.data());

        return (s2o_ * xvec)(0, 0);
    }

    void operator()(state_t const& x, state_t& dxdt, double t) {
        using namespace Eigen;

        Map<const Matrix<double, 2, 1> > xvec(x.data());
        Map<Matrix<double, 2, 1> > result(dxdt.data());
        
        result = drift_term_ * xvec + input_term_ ;   // input is always 1V
    }

private:
    Eigen::MatrixXd drift_term_;            // connects current state to development over time
    Eigen::MatrixXd input_term_;            // connects input to development
    Eigen::Matrix<double, 1, Eigen::Dynamic> s2o_; // transforms state to output
};

int main() {
    using namespace boost::numeric::odeint;
    circuit ckt(100.0, 20e-6, 20e-9);
    state_t x{0.0, 0.0};                    // initial conditions

    integrate( ckt, x, 0.0, 10e-6, 0.1e-6,  // time range and increment
               [ckt](state_t const& x, double t) {
                   std::cout << t << " " << ckt.state2output(x) << std::endl;
               });
}
