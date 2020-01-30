#pragma once
#include "Matrix.h"

template <class T>
class Solver
{
public:
	static Matrix<T>* solveJacobi(Matrix<T>* LHS, Matrix<T>* b, double tolerance, int max_iterations, T initial_guess[]);
	static Matrix<T>* solveGaussSeidel(Matrix<T>* LHS, Matrix<T>* b, double tolerance, int max_iterations, T initial_guess[]);
	static Matrix<T>* solveLU(Matrix<T>* LHS, Matrix<T>* b);
	static Matrix<T>* conjugateGradient(Matrix<T>* LHS, Matrix<T>* b, double TOL, int max_iterations);
};