#pragma once
#include "Matrix.h"

template <class T>
class Solver
{
public:

    // Actual Solvers
	static Matrix<T>* solveJacobi(Matrix<T>* LHS, Matrix<T>* b, double tolerance, int max_iterations, T initial_guess[]);
	static Matrix<T>* solveGaussSeidel(Matrix<T>* LHS, Matrix<T>* b, double tolerance, int max_iterations, T initial_guess[]);
	static Matrix<T>* solveLU(Matrix<T>* LHS, Matrix<T>* b);
	static Matrix<T>* conjugateGradient(Matrix<T>* LHS, Matrix<T>* b, double TOL, int max_iterations);
    static Matrix<T>* solveGaussian(Matrix<T>* LHS, Matrix<T>* b);

private:
    // Helper functions
    static void luDecomposition(Matrix<T>* LHS, Matrix<T>* upper_tri, Matrix<T>* lower_tri);
    static void luDecompositionPivot(Matrix<T>* LHS, Matrix<T>* upper_tri, Matrix<T>* lower_tri, Matrix<T>* permutation);
    static void upperTriangular(Matrix<T>* LHS, Matrix<T>* b);
    static Matrix<T>* backSubstitution(Matrix<T>* LHS, Matrix<T>* b);
    static Matrix<T>* forwardSubstitution(Matrix<T>* LHS, Matrix<T>* b);
};