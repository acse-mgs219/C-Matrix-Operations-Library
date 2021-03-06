#define CATCH_CONFIG_RUNNER
#include "test/catch.hpp"
#include <iostream>
#include <ctime>

#include "Matrix.h"
#include "Matrix.cpp"
#include "CSRMatrix.h"
#include "CSRMatrix.cpp"
#include "Solver.h"
#include "Solver.cpp"
#include "utilities.h"

int main()
{
    // We can change the default max iterations and tolerance values like so
    Solver<double>::maxIt = 500;

    std::cout << "Loading large matrices from file, please wait...\n\n";

    // A was previously generated with our makeRandomSparseSPD function
    // It is SPD (required for CG), Diagonally Dominany (required for Gauss-Seidel and Jacobi) and has a lot of 0s (adequate for sparse use)
    auto A = new Matrix<double>(3000, 3000, (std::string) "sampleMatrices/3000LHS.txt");

    // x was generated with our makeRandom function
    // It has random uniformly distributed values between 0 and 100
    auto x = new Matrix<double>(3000, 1, (std::string) "sampleMatrices/3000Sol.txt");

    // b was generated by doing A * x = b, so that we know solving (A,b) should return x
    auto b = new Matrix<double>(3000, 1, (std::string) "sampleMatrices/3000RHS.txt");

    // Space to hold our solution variable
    auto sol = new Matrix<double>(3000, 1, true);

    std::cout << "Running dense solvers:\n";

    std::cout << "\nLU Solver:\n";
    sol = Solver<double>::solveLU(A, b);
    std::cout << "Has really converged: " << hasConverged(sol->values, x->values, x->rows, 0.001);
    std::cout << "\nGaussian Elimination:\n";
    sol = Solver<double>::solveGaussian(A, b);
    std::cout << "Has really converged: " << hasConverged(sol->values, x->values, x->rows, 0.001);
    std::cout << "\nJacobi:\n";
    sol = Solver<double>::solveJacobi(A, b);
    std::cout << "Has really converged: " << hasConverged(sol->values, x->values, x->rows, 0.001);
    std::cout << "\nGauss-Seidel:\n";
    sol = Solver<double>::solveGaussSeidel(A, b);
    std::cout << "Has really converged: " << hasConverged(sol->values, x->values, x->rows, 0.001);
    std::cout << "\nConjugate Gradient:\n";
    sol = Solver<double>::solveConjugateGradient(A, b);
    std::cout << "Has really converged: " << hasConverged(sol->values, x->values, x->rows, 0.001) << std::endl;

    std::cout << "\nNow running Sparse solvers on the same system:\n";
    // Store A sparsely instead, useful since it has so many 0s
    auto sparseA = new CSRMatrix(A);

    std::cout << "\nJacobi Sparse:\n";
    sol = Solver<double>::solveJacobi(sparseA, b);
    std::cout << "Has really converged: " << hasConverged(sol->values, x->values, x->rows, 0.001);
    std::cout << "\nGauss-Seidel Sparse:\n";
    sol = Solver<double>::solveGaussSeidel(sparseA, b);
    std::cout << "Has really converged: " << hasConverged(sol->values, x->values, x->rows, 0.001);
    std::cout << "\nConjugate Gradient Sparse:\n";
    sol = Solver<double>::solveConjugateGradient(sparseA, b);
    std::cout << "Has really converged: " << hasConverged(sol->values, x->values, x->rows, 0.001) << std::endl;

}
