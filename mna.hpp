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

// voltage source inputs and inductors get this treatment:
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
             int node, double c) {
    C(node, node) += c;  // assumes other terminal is ground
}

template<int sz>
void stamp_l(Eigen::Matrix<double, sz, sz>& G,
             Eigen::Matrix<double, sz, sz>& C,
             int node, int istate, double l) {
    C(istate, istate)  +=  l;
    // For inductors we have an extra state that remembers its current
    stamp_i(G, C, node, istate);
}

