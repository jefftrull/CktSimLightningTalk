// A directly calculated implementation of the circuit described in
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
