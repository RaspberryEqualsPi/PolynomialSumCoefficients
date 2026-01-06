#include <vector>
#include <iostream>
#include <math.h>
#include "LinearAlgae.h"

namespace LinearAlgae {
	void Row::setEntry(int j, float val) { // sets the jth entry to val. j is 1-indexed
		content[j - 1] = val;
	}
	float Row::getEntry(int j) {
		return content[j - 1];
	}
	int Row::getSize() {
		return content.size();
	}
	std::vector<float>* Row::getRaw() { // returns raw content
		return &content;
	}
	Row Row::operator+(Row& obj)
	{
		Row res(content);
		std::vector<float>* resRaw = res.getRaw();
		std::vector<float>* objRaw = obj.getRaw();
		for (int j = 1; j <= content.size(); j++) { // maintain convention for j
			(*resRaw)[j - 1] += (*objRaw)[j - 1]; // since it's a pointer, this will directly modify the content of res
		}
		return res;
	}
	Row& Row::operator+=(Row& obj)
	{
		std::vector<float>* objRaw = obj.getRaw();
		for (int j = 1; j <= content.size(); j++) { // maintain convention for j
			content[j - 1] += (*objRaw)[j - 1];
		}
		return *this;
	}
	Row Row::operator-(Row& obj)
	{
		Row res(content);
		std::vector<float>* resRaw = res.getRaw();
		std::vector<float>* objRaw = obj.getRaw();
		for (int j = 1; j <= content.size(); j++) { // maintain convention for j
			(*resRaw)[j - 1] -= (*objRaw)[j - 1]; // since it's a pointer, this will directly modify the content of res
		}
		return res;
	}
	Row& Row::operator-=(Row& obj)
	{
		std::vector<float>* objRaw = obj.getRaw();
		for (int j = 1; j <= content.size(); j++) { // maintain convention for j
			content[j - 1] -= (*objRaw)[j - 1];
		}
		return *this;
	}
	Row Row::operator*(float const& obj) // multiply by scalar
	{
		Row res(content);
		std::vector<float>* resRaw = res.getRaw();
		for (int j = 1; j <= content.size(); j++) { // maintain convention for j
			(*resRaw)[j - 1] *= obj; // since it's a pointer, this will directly modify the content of res
		}
		return res;
	}
	Row& Row::operator*=(float const& obj) // multiply by scalar
	{
		for (int j = 1; j <= content.size(); j++) { // maintain convention for j
			content[j - 1] *= obj;
		}
		return *this;
	}
	Row Row::operator/(float const& obj) // divide by scalar
	{
		Row res(content);
		std::vector<float>* resRaw = res.getRaw();
		for (int j = 1; j <= content.size(); j++) { // maintain convention for j
			(*resRaw)[j - 1] /= obj; // since it's a pointer, this will directly modify the content of res
		}
		return res;
	}
	Row& Row::operator/=(float const& obj) // divide by scalar
	{
		for (int j = 1; j <= content.size(); j++) { // maintain convention for j
			content[j - 1] /= obj;
		}
		return *this;
	}
	Row::Row(std::vector<float> arr) {
		content = arr;
	}
	Row::Row(int len) {
		content = std::vector<float>(len, 0);
	}
	Row::Row() {
		// dead row, this kind of row should not be used
	}

	// Elementary Operations

	ElemOp Matrix::swap(int a, int b) { // R_a <--> R_b
		// this operation is O(1)
		ElemOp swapOp;
		swapOp.operation = ElemOps::swap;
		swapOp.a = a;
		swapOp.b = b;

		Row temp = content[a - 1];
		content[a - 1] = content[b - 1];
		content[b - 1] = temp;

		return swapOp;
	}

	ElemOp Matrix::replace(float a, int b, float c, int d) { // a*R_b + c*R_d --> R_d
		// this operation is O(n)
		ElemOp replaceOp;
		replaceOp.operation = ElemOps::replace;
		replaceOp.a = a;
		replaceOp.b = b;
		replaceOp.c = c;
		replaceOp.d = d;

		Row res1 = content[b - 1] * a;
		Row res2 = content[d - 1] * c;
		content[d - 1] = res1 + res2;

		return replaceOp;
	}

	ElemOp Matrix::scale(float a, int b) { // a*R_b --> R_b
		// this operation is O(n)
		ElemOp scaleOp;
		scaleOp.operation = ElemOps::scale;
		scaleOp.a = a;
		scaleOp.b = b;

		content[b - 1] *= a;

		return scaleOp;
	}

	void Matrix::applyOperation(ElemOp op) {
		switch (op.operation) {
		case ElemOps::swap:
			swap(op.a, op.b);
			break;
		case ElemOps::scale:
			scale(op.a, op.b);
			break;
		case ElemOps::replace:
			replace(op.a, op.b, op.c, op.d);
			break;
		default:
			break;
		}
	}

	// Everything else
	void Matrix::setEntry(int i, int j, float val) { // sets the jth entry to val. j is 1-indexed
		content[i - 1].setEntry(j, val);
	}
	float Matrix::getEntry(int i, int j) {
		return content[i - 1].getEntry(j);
	}
	int Matrix::getM() { // m in m x n dimension
		return content.size();
	}
	int Matrix::getN() { // n in m x n dimension
		return content[0].getSize(); // each row should be the same size, so this should be representative
	}
	std::vector<Row>* Matrix::getRaw() { // returns raw content
		return &content;
	}
	Matrix Matrix::operator+(Matrix& obj)
	{
		Matrix res(content);
		std::vector<Row>* resRaw = res.getRaw();
		std::vector<Row>* objRaw = obj.getRaw();
		for (int i = 1; i <= content.size(); i++) { // maintain convention for i
			(*resRaw)[i - 1] += (*objRaw)[i - 1]; // since it's a pointer, this will directly modify the content of res
		}
		return res;
	}
	Matrix& Matrix::operator+=(Matrix& obj)
	{
		std::vector<Row>* objRaw = obj.getRaw();
		for (int i = 1; i <= content.size(); i++) { // maintain convention for i
			content[i - 1] += (*objRaw)[i - 1];
		}
		return *this;
	}
	Matrix Matrix::operator-(Matrix& obj)
	{
		Matrix res(content);
		std::vector<Row>* resRaw = res.getRaw();
		std::vector<Row>* objRaw = obj.getRaw();
		for (int i = 1; i <= content.size(); i++) { // maintain convention for i
			(*resRaw)[i - 1] -= (*objRaw)[i - 1]; // since it's a pointer, this will directly modify the content of res
		}
		return res;
	}
	Matrix& Matrix::operator-=(Matrix& obj)
	{
		std::vector<Row>* objRaw = obj.getRaw();
		for (int i = 1; i <= content.size(); i++) { // maintain convention for i
			content[i - 1] -= (*objRaw)[i - 1];
		}
		return *this;
	}
	Matrix Matrix::operator*(float const& obj) // scalar multiplication
	{
		Matrix res(content);
		std::vector<Row>* resRaw = res.getRaw();
		for (int i = 1; i <= content.size(); i++) { // maintain convention for i
			(*resRaw)[i - 1] *= obj; // since it's a pointer, this will directly modify the content of res
		}
		return res;
	}
	Matrix Matrix::operator*(Matrix& obj) { // O(m_A*n_B*n_A) algorithm to take the matrix product AB
		if (getN() != obj.getM()) // in multiplying matrices, the dimensions must have n_A = m_B
			throw std::invalid_argument("inner dimensions must match to multiply");
		Matrix res(getM(), obj.getN()); // the resultant matrix will be m x n under the previously stated set of dimensions
		for (int i = 1; i <= getM(); i++) { // go down the rows of this matrix
			for (int j = 1; j <= obj.getN(); j++) { // traverse the columns of the second matrix
				float entry = 0;
				for (int z = 1; z <= getN(); z++) {
					entry += getEntry(i, z) * obj.getEntry(z, j); // taking dot product of the row and column
				}
				res.setEntry(i, j, entry);
			}
		}
		return res;
	}
	Matrix& Matrix::operator*=(float const& obj) // scalar multiplication
	{
		for (int i = 1; i <= content.size(); i++) { // maintain convention for i
			content[i - 1] *= obj;
		}
		return *this;
	}
	Matrix Matrix::operator/(float const& obj) // scalar division
	{
		Matrix res(content);
		std::vector<Row>* resRaw = res.getRaw();
		for (int i = 1; i <= content.size(); i++) { // maintain convention for i
			(*resRaw)[i - 1] /= obj; // since it's a pointer, this will directly modify the content of res
		}
		return res;
	}
	Matrix& Matrix::operator/=(float const& obj) // scalar division
	{
		for (int i = 1; i <= content.size(); i++) { // maintain convention for i
			content[i - 1] /= obj;
		}
		return *this;
	}
	Matrix::Matrix(std::vector<Row> arr) {
		content = arr;
	}
	Matrix::Matrix(int len) { // forms square matrix
		for (int i = 0; i < len; i++) {
			content.push_back(Row(len));
		}
	}
	Matrix::Matrix(int m, int n) { // forms an m x n zero matrix
		for (int i = 0; i < m; i++) {
			content.push_back(Row(n));
		}
	}

	// Utility functions

	void printRow(Row row) {
		std::vector<float>* raw = row.getRaw();
		for (int i = 0; i < raw->size(); i++) {
			std::cout << raw->at(i) << " ";
		}
	}

	void printMatrix(Matrix matrix) {
		std::vector<Row>* raw = matrix.getRaw();
		for (int i = 0; i < raw->size(); i++) {
			printRow(raw->at(i));
			std::cout << "\n";
		}
	}

	void printOperation(ElemOp op) {
		switch (op.operation) {
		case ElemOps::swap:
			std::cout << "R_" << op.a << " <--> " << "R_" << op.b;
			break;
		case ElemOps::scale:
			std::cout << "(" << op.a << ")R_" << op.b << " --> " << "R_" << op.b;
			break;
		case ElemOps::replace:
			std::cout << "(" << op.a << ")R_" << op.b << " + " << "(" << op.c << ")R_" << op.d << " --> " << "R_" << op.d;
			break;
		default:
			break;
		}
	}

	ElemOp inverseOperation(ElemOp op) { // find the inverse of an elementary operation
		ElemOp res;
		res.operation = op.operation;
		switch (op.operation) {
		case ElemOps::swap:
			res = op; // inverse of a swap operation is the same
			break;
		case ElemOps::replace: // aR_b + cR_d --> R_d, so to invert it we'd have (-a/c)R_b + (1/c)R_d --> R_d
			res.a = (-1 * op.a / op.c);
			res.b = op.b;
			res.c = 1 / op.c;
			res.d = op.d;
			break;
		case ElemOps::scale: // aR_b --> R_b, so to invert it we'd have (1/a)R_b --> R_b
			res.a = 1 / op.a;
			res.b = op.b;
			break;
		default:
			break;
		}
		return res;
	}

	Matrix createIdentityMatrix(int n) {
		Matrix I(n); // n x n zero matrix
		for (int i = 1; i <= n; i++) {
			I.setEntry(i, i, 1);
		}
		return I;
	}

	std::vector<ElemOp> REF(Matrix& matrix) { // O(m^2*n) algorithm to turn m x n matrices into REF, and returns the elementary operations done
		std::vector<ElemOp> operationList;
		for (int pivot = 1; pivot <= matrix.getM() - 1; pivot++) { // pivot = pivot_i and pivot_j
			int pivotRow = -1;
			for (int i = pivot; i <= matrix.getM(); i++) { // find a row without a pivot of 0, ignore old reduced rows, call it our pivot row
				if (matrix.getEntry(i, pivot) != 0) {
					pivotRow = i;
					break;
				}
			}
			if (pivotRow == -1) {
				throw std::invalid_argument("matrix not invertable"); // zero column means singular matrix as det = 0
			}
			else if (pivotRow != pivot) {
				operationList.push_back(matrix.swap(pivotRow, pivot)); // R_pivotRow <--> R_pivot; we put it in the proper row to conform with row-echelon form
			}
			float pivotNum = (float)matrix.getEntry(pivot, pivot);
			if (pivotNum == 0)
				throw std::invalid_argument("matrix not invertable"); // division by zero implies non-invertable
			operationList.push_back(matrix.scale(1 / pivotNum, pivot)); // (1/a_pivotpivot)R_pivot --> R_pivot; make its pivot equal to 1

			for (int i = pivot + 1; i <= matrix.getM(); i++) { // make the pivot of the other rows equal 0
				operationList.push_back(matrix.replace(-1 * matrix.getEntry(i, pivot), pivot, 1, i)); // (-a_ipivot)R_pivot + R_i --> R_i; step that makes its pivot 0
			}
		}
		float last = (float)matrix.getEntry(matrix.getM(), matrix.getM());
		if (last == 0)
			throw std::invalid_argument("matrix not invertable"); // division by zero implies non-invertable
		operationList.push_back(matrix.scale(1 / last, matrix.getM())); // (1/a_mm)R_m --> R_m; finally, reduce the last row to have its entry be one

		return operationList;
	}

	std::vector<ElemOp> RREF(Matrix& matrix) { // O(m^2*n) algorithm to turn m x n matrices into RREF, and returns the elementary operations done
		std::vector<ElemOp> operationList = REF(matrix); // REFing it first makes the process very simple
		for (int i = matrix.getM() - 1; i >= 1; i--) { // traverse from the back; this loop is O(m^2*n)
			for (int j = i + 1; j <= matrix.getM(); j++) { // to get rid of non-zero entries beyond the pivot within the coefficient section
				operationList.push_back(matrix.replace(-1 * matrix.getEntry(i, j), j, 1, i)); // (-a_ij)R_j + R_i --> R_i; step that makes its pivot 0
			}
		}
		return operationList;
	}


	Matrix findInverse(Matrix mat) { // O(m^3) algorithm to find inverse of m x m matrix
		Matrix matrix = mat; // copy
		if (matrix.getM() != matrix.getN())
			throw std::invalid_argument("matrix must be square to invert"); // cannot invert non-square matrices
		std::vector<ElemOp> operationList = RREF(matrix); // so we can get the operations necessary to reduce to the identity matrix
		// given that (E_n)...(E_4)(E_3)(E_2)(E_1)A = I, where E are the elementary matrices of the operations, A^-1 will be the product of the elementary matrices beginning at the back
		// for this reason, we begin with the last operation and work our way to the first
		Matrix inverse = createIdentityMatrix(matrix.getM());
		for (int i = 0; i < operationList.size(); i++) {
			inverse.applyOperation(operationList[i]);
		}
		return inverse;
	}

	float findDeterminant(Matrix mat) { // O(m^3) algorithm to find inverse of m x m matrix
		Matrix matrix = mat;
		if (matrix.getM() != matrix.getN())
			throw std::invalid_argument("matrix must be square to find determinant"); // cannot find determinant of non-square matrices
		std::vector<ElemOp> operationList;
		try {
			operationList = REF(matrix);
		}
		catch (std::invalid_argument e) {
			return 0; // this would happen if non-invertable -> det = 0
		}
		float coefficient = 1;
		for (int i = 0; i < operationList.size(); i++) { // account for changes to the determinant through the REF process
			ElemOp op = operationList[i];
			switch (op.operation) {
			case ElemOps::replace:
				break; // replace operation doesn't change determinant
			case ElemOps::scale:
				coefficient /= op.a; // scale operation scales the determinant 1:1, dividing reverses this
				break;
			case ElemOps::swap:
				coefficient *= -1; // swapping multiplies the determinant by -1
				break;
			default:
				break;
			}
		}
		float determinant = 1;
		for (int i = 1; i <= matrix.getM(); i++) {
			determinant *= matrix.getEntry(i, i); // determinant of a triangular matrix is the product of its diagonal entries; find the determinant of REF'd matrix
		}
		return determinant * coefficient; // the coefficient will scale the determinant of the REF matrix back to the original determinant
	}

	std::pair<Matrix, Matrix> findLUFactorization(Matrix mat) { // O(m^3) algorithm to find LU factorization of m x m matrices ONLY; returns {L, U}
		if (mat.getM() != mat.getN())
			throw std::invalid_argument("findLUFactorization only accepts square matrices");
		Matrix U = mat;
		std::vector<ElemOp> operationList = REF(U);
		// we know that, if we REF, we have the elementary matrices that correspond to each operation implying the following:
		// E_n...(E_4)(E_3)(E_2)(E_1)A = REF(A). If REF(A) = U, and A = LU, then L = the product of the inverses of the elementary matrices
		// this product can be represented by applying the operations in reverse order (n..., 4, 3, 2, 1 order) to I_n
		Matrix L = createIdentityMatrix(mat.getN()); // I_n
		for (int i = operationList.size() - 1; i >= 0; i--) {
			L.applyOperation(inverseOperation(operationList[i]));
		}
		return std::pair<Matrix, Matrix>(L, U);
	}
}