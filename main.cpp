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

int main()
{
    // seed our random generator
    srand(time(0));

    // run tests
    int result = Catch::Session().run();


//    Solver<double>::maxIt = 500;
//    auto A = new Matrix<double>(3000, 3000, (std::string) "3000LHS.txt");
//    A->printMatrix();
//    auto x = new Matrix<double>(3000, 1, (std::string) "3000Sol.txt");
//    x->printMatrix();
//    auto b = new Matrix<double>(3000, 1, (std::string) "3000RHS.txt");
//    b->printMatrix();
//
//    auto sol = new Matrix<double>(3000, 1, true);
//
//    std::cout << "\nJacobi:\n";
//    sol = Solver<double>::solveJacobi(A, b);
//    sol->printMatrix();
//
//    std::cout << "\nGauss-Seidel:\n";
//    sol = Solver<double>::solveGaussSeidel(A, b);
//    sol->printMatrix();
//
//    auto sparseA = new CSRMatrix(A);
//
//    std::cout << "\nJacobi Sparse:\n";
//    sol = Solver<double>::solveJacobi(sparseA, b);
//    sol->printMatrix();

//    std::cout << "\nGauss-Seidel Sparse:\n";
//    sol = Solver<double>::solveGaussSeidel(sparseA, b);
//    sol->printMatrix();

    return result;
}
