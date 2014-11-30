// An RLC bandpass filter using Boost.odeint for simulation and Eigen for circuit manipulation
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
#include <cmath>

#include <boost/numeric/odeint.hpp>
#include <boost/math/constants/constants.hpp>
#include <Eigen/Dense>

#include "mna.hpp"

typedef std::array<double, 2> state_t;   // 0 = V_out, 1 = I_L

struct circuit {
    circuit(double r, double l, double c) : circuit(r, l, c, 0, true) {}

    circuit(double r, double l, double c,
            double freq,        // cycles per second for sine wave
            bool   step=false   // step response instead
        ) : freq_(boost::math::constants::two_pi<double>() * freq),  // radian/s conversion
        step_(step) {

        using namespace Eigen;

        typedef Matrix<double, 5, 5> matrix5_t;
        matrix5_t G = matrix5_t::Zero();
        matrix5_t C = matrix5_t::Zero();

        // state assignment: 0, 1, 2 = node voltages; 3 = I_L, 4 = I_in
        stamp_r(G, C, 2, r);
        stamp_c(G, C, 1, 2, c);
        stamp_l(G, C, 0, 1, 3, l);     // n1, n2, I_L
        stamp_i(G, C, 0, 4);           // V_in, I_in

        // input application vector - connects single input to appropriate equation
        typedef Matrix<double, 5, 1> vector5_t;
        vector5_t B;
        B << 0, 0, 0, 0, -1;

        // output extraction vector - connects state element to output
        Matrix<double, 1, 5> D;
        D << 0, 0, 1, 0, 0;         // state element idx 2 = V_out

        Matrix<double, 1, 1> E = Matrix<double, 1, 1>::Zero();
        Matrix<double, 5, 1> DT = D.transpose();
        auto momentvec = moments(G, C, B, DT, E, 4);

        // Now we have C*dX/dt = -G*X + B*u, and the output is = D * X

        // we could not use Su regularization here because after LU factorization Cprime was
        // not square (UL elements were 2x3 because one row was -1 times another row)
        Matrix<double, Dynamic, Dynamic> Gnew, Cnew;
        Matrix<double, Dynamic, 1> Bnew, Dnew;
        Matrix<double, 1, 1> Enew;
        std::tie(Gnew, Cnew, Bnew, Dnew, Enew) = regularize(G, C, B, DT);

        Matrix<double, Dynamic, 1> DNT = Dnew.transpose();
        momentvec = moments(Gnew, Cnew, Bnew, DNT, E, 4);

        // verify the new C is non-singular
        assert(Cnew.rows() == Cnew.fullPivLu().rank());
        // and there is no feedthrough term
        assert(Enew.isZero());

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
        
        Matrix<double, 1, 1> input;
        if (step_) {
            input << 1.0;
        } else {
            // calculate current value of sine wave input
            input << std::sin(freq_ * t);
        }

        result = drift_term_ * xvec + input_term_ * input;

    }

private:
    Eigen::MatrixXd drift_term_;            // connects current state to development over time
    Eigen::MatrixXd input_term_;            // connects input to development
    Eigen::Matrix<double, 1, Eigen::Dynamic> s2o_; // transforms state to output
    double freq_;                           // remembers frequency for calculating input
    bool   step_;                           // indicates using step function instead of sine wave
};

int main() {
    using namespace boost::numeric::odeint;
    const double freq = 50e3;
    circuit ckt(100.0, 20e-6, 20e-9, freq);
    state_t x{0.0, 0.0};                    // initial conditions

    integrate( ckt, x, 0.0, 10/freq, 0.1e-6,  // time range and increment
               [ckt,freq](state_t const& x, double t) {
                   double inp = std::sin(freq*boost::math::constants::two_pi<double>()*t);
                   std::cout << t << " " << inp << " " << ckt.state2output(x) << std::endl;
               });
}
