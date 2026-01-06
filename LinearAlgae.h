#pragma once
#include <vector>

namespace LinearAlgae {
	class Row {
	public:
		void setEntry(int j, float val);
		float getEntry(int j);
		int getSize();
		std::vector<float>* getRaw();
		Row operator+(Row& obj);
		Row& operator+=(Row& obj);
		Row operator-(Row& obj);
		Row& operator-=(Row& obj);
		Row operator*(float const& obj);
		Row& operator*=(float const& obj);
		Row operator/(float const& obj);
		Row& operator/=(float const& obj);
		Row(std::vector<float> arr);
		Row(int len);
		Row();
	private:
		std::vector<float> content;
	};

	enum ElemOps {
		swap,
		replace,
		scale
	};

	struct ElemOp {
		ElemOps operation;
		float a;
		float b;
		float c;
		float d;
	};

	class Matrix {
	public:
		// Elementary Operations

		ElemOp swap(int a, int b);
		ElemOp replace(float a, int b, float c, int d);
		ElemOp scale(float a, int b);
		void applyOperation(ElemOp op);

		// Everything else
		void setEntry(int i, int j, float val);
		float getEntry(int i, int j);
		int getM();
		int getN();
		std::vector<Row>* getRaw();
		Matrix operator+(Matrix& obj);
		Matrix& operator+=(Matrix& obj);
		Matrix operator-(Matrix& obj);
		Matrix& operator-=(Matrix& obj);
		Matrix operator*(float const& obj);
		Matrix operator*(Matrix& obj);
		Matrix& operator*=(float const& obj);
		Matrix operator/(float const& obj);
		Matrix& operator/=(float const& obj);
		Matrix(std::vector<Row> arr);
		Matrix(int m, int n);
		Matrix(int len);
	private:
		std::vector<Row> content;
	};

	// Utility functions

	void printRow(Row row);

	void printMatrix(Matrix matrix);

	void printOperation(ElemOp op);

	ElemOp inverseOperation(ElemOp op);

	Matrix createIdentityMatrix(int n);

	std::vector<ElemOp> REF(Matrix& matrix);

	std::vector<ElemOp> RREF(Matrix& matrix);

	Matrix findInverse(Matrix mat);

	float findDeterminant(Matrix mat);

	std::pair<Matrix, Matrix> findLUFactorization(Matrix mat);
}