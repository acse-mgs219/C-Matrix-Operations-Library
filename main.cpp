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

    std::cout << "Reading Matrix A from text file:\n";
    auto A = new Matrix<double>(10, 10, (std::string) "smallMatrix.txt");
    A->printMatrix();
    std::cout << "Reading matrix b from text files:\n";
    auto b = new Matrix<double>(10, 1, (std::string) "smallMatrixB.txt");
    b->printMatrix();
    std::cout << "Solving Ax = b for x, using Jacobi: ";
    auto sol = Solver<double>::solveJacobi(A, b);
    sol->printMatrix();
    std::cout << "Solving the same matrix using Gauss-Seidel: ";
    auto sol2 = Solver<double>::solveGaussSeidel(A, b);
    sol2->printMatrix();

    delete A;
    delete b;
    delete sol;
    delete sol2;

    std::cout << "Press any key to continue . . . ";
    std::cin.get();
    std::cout << "Moving onto large matrices, we will generate a 400x400 random matrix guaranteed to be SPD\n";
    A = new Matrix<double>(400, 400, true);
    A->makeRandomSPD();
    A->printMatrix();
    A->writeMatrix("SPDMatrixA.txt");
    std::cout << "We will also generate an x to use:\n";
    auto realSol = new Matrix<double>(400, 1, true);
    realSol->makeRandom();
    realSol->printMatrix();
    realSol->writeMatrix("SPDMatrixSol.txt");
    std::cout << "By doing A * x = b, we will generate b and then Solve(A,b) should return our original x.\n";
    b = A->matMatMult(*realSol);
    b->writeMatrix("SPDMatrixb.txt");
    std::cout << "Since A is SPD, we can solve it with Conjugate Gradient. The solution will be:\n";
    Solver<double>::maxIt = 10000;
    sol = Solver<double>::conjugateGradient(A, b);
    sol->printMatrix();

    delete sol;

    std::cout << "Press any key to continue . . . ";    
    std::cin.get();
    std::cout << "We can also represent A as sparse and use our Sparse Conjugate Gradient on it.\n";
    CSRMatrix<double>* sparseA = new CSRMatrix<double>(A);
    sol = Solver<double>::conjugateGradient(sparseA, b);
    std::cout << "We get the same solution:\n";
    sol->printMatrix();

    delete A;
    delete b;
    delete sol;

    std::cout << "Press any key to continue to timing tests, starting with Jacobi\n";
    std::cin.get();
    std::cout << "\n\nTiming tests for Jacobi as n grows:\n";
    for (int i = 10; i <= 1000; i *= 10)
    {
        std::cout << "When running " << i << "x" << i << " system:\n";
        A = new Matrix<double>(i, i, true);
        A->makeRandomDD();
        b = new Matrix<double>(i, 1, true);
        b->makeRandom();
        sol = Solver<double>::solveJacobi(A, b);
        sparseA = new CSRMatrix<double>(A);
        sol2 = Solver<double>::solveJacobi(sparseA, b);
        sol->printMatrix();
        sol2->printMatrix();
        delete A;
        delete b;
        std::cout << "\n";
    }

    std::cout << "Press any key to continue to Gauss-Seidel tests\n";
    std::cin.get();
    std::cout << "\n\nTiming tests for Gauss-Seidel as n grows:\n";
    for (int i = 10; i <= 1000; i *= 10)
    {
        std::cout << "When running " << i << "x" << i << " system:\n";
        A = new Matrix<double>(i, i, true);
        A->makeRandomDD();
        b = new Matrix<double>(i, 1, true);
        b->makeRandom();
        sol = Solver<double>::solveGaussSeidel(A, b);
        sparseA = new CSRMatrix<double>(A);
        sol2 = Solver<double>::solveGaussSeidel(sparseA, b);
        sol->printMatrix();
        sol2->printMatrix();
        delete A;
        delete b;
        std::cout << "\n";
    }

    std::cout << "Press any key to continue to Gaussian tests\n";
    std::cin.get();
    std::cout << "\n\nTiming tests for Gaussian as n grows:\n";
    for (int i = 10; i <= 1000; i *= 10)
    {
        std::cout << "When running " << i << "x" << i << " system:\n";
        A = new Matrix<double>(i, i, true);
        A->makeRandomDD();
        b = new Matrix<double>(i, 1, true);
        b->makeRandom();
        Solver<double>::solveGaussian(A, b);
        delete A;
        delete b;
        std::cout << "\n";
    }

    std::cout << "Press any key to continue to LU Decomp tests\n";
    std::cin.get();
    std::cout << "\n\nTiming tests for LU Decomp as n grows:\n";
    for (int i = 10; i <= 1000; i *= 10)
    {
        std::cout << "When running " << i << "x" << i << " system:\n";
        A = new Matrix<double>(i, i, true);
        A->makeRandomDD();
        b = new Matrix<double>(i, 1, true);
        b->makeRandom();
        Solver<double>::solveLU(A, b);
        delete A;
        delete b;
        std::cout << "\n";
    }

    std::cout << "Press any key to continue to Conjugate Gradient tests\n";
    std::cin.get();
    std::cout << "\n\nTiming tests for CG as n grows:\n";
    for (int i = 10; i <= 1000; i *= 10)
    {
        std::cout << "When running " << i << "x" << i << " system:\n";
        A = new Matrix<double>(i, i, true);
        A->makeRandomSPD();
        b = new Matrix<double>(i, 1, true);
        b->makeRandom();
        Solver<double>::conjugateGradient(A, b);
        delete A;
        delete b;
        std::cout << "\n";
    }

    // run tests
    int result = Catch::Session().run();

    return result;
}
