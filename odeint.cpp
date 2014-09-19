// A numerically integrated implementation of the circuit described in
// "Rolling Your Own Circuit Simulator with Eigen and Boost.ODEInt"
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

typedef std::array<double, 2> state_t;   // 0 = V_out, 1 = I_L

struct circuit {
    circuit(double r, double l, double c) : r_(r), l_(l), c_(c) {}

    void operator()(state_t const& x, state_t& dxdt, double t) {
        // calculate state derivatives from current state
        dxdt[0] = ((1 - x[0]) / r_ - x[1]) / c_;  // KCL at V_out node
        dxdt[1] = x[0] / l_;                      // from V_out = L * dI_L/dt
    }
private:
    double r_, l_, c_;
};

int main() {
    using namespace boost::numeric::odeint;
    circuit ckt(100.0, 20e-6, 20e-9);
    state_t x{0.0, 0.0};                    // initial conditions
    integrate( ckt, x, 0.0, 10e-6, 0.1e-6,  // time range and increment
               [](state_t const& x, double t) {
                   std::cout << t << " " << x[0] << std::endl;
               });
}
