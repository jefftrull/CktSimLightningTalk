// A directly calculated implementation of the circuit described in
// "Rolling Your Own Circuit Simulator with Eigen and Boost.ODEInt"
// Author: Jeff Trull <edaskel@att.net>
// Boost licensed, boilerplate TBD

#include <cmath>
#include <iostream>

struct circuit {
    circuit(double r, double l, double c) : r_(r), c_(c) {
        lambda_ = -1.0 / (2.0 * r * c);
        mu_     = sqrt((1.0 / (l * c)) - lambda_ * lambda_);  // reverse subtraction for overdamped case
    }
    double operator()(double t) {
        return (1.0 / (mu_ * r_ * c_)) * exp(lambda_ * t) * sin(mu_ * t);
    }
private:
    double r_, c_, lambda_, mu_;
};

int main() {
    circuit ckt(100.0, 20e-6, 20e-9);
    for (double t = 0; t < 10e-6; t += 0.1e-6) {
        std::cout << t << " " << ckt(t) << std::endl;
    }
}
