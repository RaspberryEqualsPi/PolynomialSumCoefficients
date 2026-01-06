#include <iostream>
#include "LinearAlgae.h"


using namespace LinearAlgae;
void findSumFormula(int p) { // sum of k^p
	Matrix fitAugmented(p + 2, p + 3);
	// need to fit polynomial a_1n^(p+1) + a_2n^p +...+ a_n
	int fn = 0; // f(n) prev
	for (int i = 1; i <= p + 2; i++) {
		fn += pow(i, p); // += i^p
		int val = 0;
		for (int j = 0; j <= p + 1; j++) {
			fitAugmented.setEntry(i, j + 1, pow(i, j));
		}
		fitAugmented.setEntry(i, p + 3, fn);
	}
	//printMatrix(fitAugmented);
	RREF(fitAugmented); // RREF to solve system
	// We know that each coefficient is rational, furthermore it has p = 1. We invert then print the fraction inverted.
	for (int i = p + 2; i >= 2; i--) {
		std::cout << "(1/" << (1 / fitAugmented.getEntry(i, p + 3)) << ")n^" << i - 1 << " + ";
	}
	std::cout << "(1/" << (1 / fitAugmented.getEntry(1, p + 3)) << ")"; // constant term
}
int main() {
	findSumFormula(3);
	return 0;
}