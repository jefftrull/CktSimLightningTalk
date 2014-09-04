// A numerically integrated implementation of the circuit described in
// "Rolling Your Own Circuit Simulator with Eigen and Boost.ODEInt"
// Author: Jeff Trull <edaskel@att.net>
// MIT licensed, boilerplate TBD

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
