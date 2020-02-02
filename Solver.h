#ifndef SOLVER_H
#define SOLVER_H

#include "Matrix.h"

template <class T>
class Solver
{
public:
    // Dense Matrice Solvers
	static Matrix<T>* solveJacobi(Matrix<T>* LHS, Matrix<T>* b, double tolerance = TOL, int max_iterations = maxIt, T initial_guess[] = nullptr, bool sortMatrix = false);
    static Matrix<T>* solveGaussSeidel(Matrix<T>* LHS, Matrix<T>* b, double tolerance = TOL, int max_iterations = maxIt, T initial_guess[] = nullptr, bool sortMatrix = false);
    static Matrix<T>* solveLU(Matrix<T>* LHS, Matrix<T>* b);
    static Matrix<T>* solveConjugateGradient(Matrix<T>* LHS, Matrix<T>* b, double epsilon = TOL, int max_iterations = maxIt, T initial_guess[] = nullptr);
    static Matrix<T>* solveGaussian(Matrix<T>* LHS, Matrix<T>* b);

    // Sparse Matrices
    static Matrix<T>* solveConjugateGradient(CSRMatrix<T>* LHS, Matrix<T>* b, double epsilon = TOL, int max_iterations = maxIt, T initial_guess[] = nullptr);
    static Matrix<T>* solveJacobi(CSRMatrix<T>* LHS, Matrix<T>* b, double tolerance = TOL, int max_iterations = maxIt, bool sortMatrix = false);
    static Matrix<T>* solveGaussSeidel(CSRMatrix<T>* LHS, Matrix<T>* b, double tolerance = TOL, int max_iterations = maxIt, bool sortMatrix = false);

    // Can be passed individually to functions or changed here to affect all solvers
    inline static double TOL = 0.001;
    inline static int maxIt = 1000;

private:
    // Helper functions
    static void luDecomposition(Matrix<T>* LHS, Matrix<T>* upper_tri, Matrix<T>* lower_tri);
    static void luDecompositionPivot(Matrix<T>* LHS, Matrix<T>* upper_tri, Matrix<T>* lower_tri, Matrix<T>* permutation);
    static void upperTriangular(Matrix<T>* LHS, Matrix<T>* b);
    static Matrix<T>* backSubstitution(Matrix<T>* LHS, Matrix<T>* b);
    static Matrix<T>* forwardSubstitution(Matrix<T>* LHS, Matrix<T>* b);
    static void incompleteCholesky(Matrix<T>* matrix);
    static bool check_finish(CSRMatrix<T>* LHS, Matrix<T>* mat_b, Matrix<T>* output, double tolerance);
};

#endif