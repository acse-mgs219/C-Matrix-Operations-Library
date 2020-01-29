#include "catch.hpp"
#include "../Matrix.h"
#include "../Matrix.cpp"
#include <stdexcept>
#include "../utilities.h"
#include "../CSRMatrix.h"
#include "../CSRMatrix.cpp"

#define TOL 0.0001

//#define RUN_ALL_TESTS

TEST_CASE("jacobi iteration")
{
    bool test_result = true;
    int rows = 4;
    int cols = 4;
    auto* A = new Matrix<double>(rows, cols, true);
    // create rhs vector
    auto* b = new Matrix<double>(cols, 1, true);
    double  A_values[16] = { 92, 8, -4, 1.5, 4, 33, 1, 8, 19, 18, 70, 6, 1, 2, 3, 44 };
    double b_values[4] = { 7, 3, 5, 2 };
    A->setMatrix(16, A_values);
    b->setMatrix(4, b_values);
    double initial_guess[4] = { 1, 1, 1, 1 };
    auto solution = A->solveJacobi(b, TOL, 1000, initial_guess);
    //double correct_values[2] = { 5.11111, -3.22222 };
    /*for (int i = 0; i < 2; i++)
    {
        if (!fEqual(solution->values[i], correct_values[i], TOL))
        {
            test_result = false;
            break;
        }
    }*/
    solution->printMatrix();
    delete A;
    delete b;
    delete solution;
    REQUIRE(test_result);
}

TEST_CASE("sparse matrix; mat-mat mult; small matrix")
{
    bool test_result = true;

    int rows = 4;
    int cols = 4;
    int nnzs = 4;

    // non-zero values of our sparse matrices
    double values[4] = {5, 8, 3, 6};
    double right_values[5] = {3, 6, 3, 8, 5};

    int iA[5] = {0, 0, 2, 3, 4};
    int jA[4] = {0, 1, 2, 1};

    int iA_right[5] = {0, 1, 3, 4, 5};
    int jA_right[5] = {0, 0, 1, 2, 1};

    int correct_values[5] = {63, 24, 24, 36, 18};
    int correct_col_index[5] = {0, 1, 2, 0, 1};

    // create sparse matrices - smart pointers automatically exit after out of scope
    std::unique_ptr< CSRMatrix<double> >  A(new CSRMatrix<double>(rows, cols, nnzs, true));
    std::unique_ptr< CSRMatrix<double> > B(new CSRMatrix<double>(rows, cols, 5, true));

    A->setMatrix(values, iA, jA);
    B->setMatrix(right_values, iA_right, jA_right);

    // generate the output matrix
    auto output = A->matMatMult(*B);

    // check values are correct
    for (int i=0; i<output->nnzs; i++)
    {
        if (!fEqual(output->values[i], correct_values[i], TOL))
        {
            test_result = false;
            break;
        }

        if (output->col_index[i] != correct_col_index[i]) {
            test_result = false;
            break;
        }
    }

    output->printMatrix();

    delete output;

    REQUIRE(test_result);
}

TEST_CASE("sparse matrix mat-mat mult, massive matrix size")
{
    int rows = 400;
    int cols = 400;
    int nnzs = 4;

    double values[4] = { 5, 8, 3, 6 }; // i = 398, k = 399
    //    double right_values[4] = {5, 8, 3, 6}; // k = 399, j = 2
    double right_values[5] = { 3, 6, 3, 8, 5 }; //ij = ik + kj

    int iA[401] = { 0 };
    iA[399] = 2;
    iA[400] = 4;
    int jA[4] = { 0, 399, 2, 100 }; // col 0 row 399 col 1 row 399 col 2 row 400 col 100 row 400

    //    int iA_right[5] = {0, 0, 2, 3, 4};
    //    int jA_right[4] = {0, 1, 2, 1};

    int iA_right[401] = { 0 };
    iA_right[398] = 1;
    iA_right[399] = 3;
    iA_right[400] = 5;
    int jA_right[5] = { 0, 0, 1, 2, 100 }; // col 0 row 398 col 0 row 399 col 1 row 399 col 2 row 400 col 100 row 400

    // create sparse matrix
    auto A = new CSRMatrix<double>(rows, cols, nnzs, true);
    auto B = new CSRMatrix<double>(rows, cols, 5, true);

    A->setMatrix(values, iA, jA);
    B->setMatrix(right_values, iA_right, jA_right);

    //A->printMatrix();
    //B->printMatrix();

    //    B->printMatrix();
    auto output = A->matMatMult(*B);

    output->printMatrix();

    delete A;
    delete B;
    delete output;
    REQUIRE(true);
}

#if defined(RUN_ALL_TESTS)
TEST_CASE("set all values of the matrix")
{
    // create new matrix
    bool test_result = true;

    int rows = 2;
    int cols = 2;

    // create matrix - preallocate memory
    auto *matrix = new Matrix<int>(rows, cols, true);

    // set values in the matrix
    int *values_ptr = new int[rows*cols];

    // set values
    for (int i=0; i<rows*cols; i++) {
        values_ptr[i] = i;
    };

    matrix->setMatrix(rows*cols, values_ptr);

    // test values have been set
    for (int i=0; i<rows*cols; i++) {
        if (!fEqual(matrix->values[i], i, TOL))
        {
            test_result = false;
            break;
        }
    };

    delete matrix;
    delete[] values_ptr;

    REQUIRE(test_result);
}

TEST_CASE("matrix multiplication; square matrix")
{
    // flag to check if we pass or fail the test
    bool test_result = true;

    int rows = 2;
    int cols = 2;

    // the values we want to set in our matrices
    double matrix_values[4] = {1, 2, 3, 4};

    // the correct output of multiply A * A
    double correct_values[4] = {7, 10, 15, 22};

    // create a matrix and set all the values to 10
    auto *matrix = new Matrix<double>(rows, cols, true);
    auto *right_matrix = new Matrix<double>(rows, cols, true);

    matrix->setMatrix(rows*cols, matrix_values);
    right_matrix->setMatrix(rows*cols, matrix_values);

    // create output matrix to hold the results
    auto *output_matrix =  matrix->matMatMult(*right_matrix);

    // check that the values in the result match the correct values
    for (int i=0; i<rows*cols; i++)
    {
        if (!fEqual(output_matrix->values[i], correct_values[i], TOL))
        {
            test_result = false;
            break;
        }
    }

    // clean up memory
    delete matrix;
    delete right_matrix;
    delete output_matrix;

    REQUIRE(test_result);
}

TEST_CASE("test matrix row swap without right hand side vector")
{
    // flag to check if we pass or fail the test
    bool test_result = true;

    int rows = 2;
    int cols = 2;

    // create a blank square matrix
    auto *A = new Matrix<double>(rows, cols, true);

    // create rhs vector
    auto *b = new Matrix<double>(cols, 1, true);

    // define the values we want to fill the matrix with
    double A_values[4] = {2, 3, 1, -4};

    // use the set matrix function to fill the values of the array in 1 go
    A->setMatrix(4, A_values);

    // this should swap rows i and j
    A->swapRowsMatrix(0, 1);

    // construct the test values - what we expect after swap
    double A_correct_values[4] = {1, -4, 2, 3};

    // check A values are correct
    for (int i=0; i<A->rows*A->cols; i++)
    {
        // check values of A
        if (!fEqual(A->values[i], A_correct_values[i], TOL))
        {
            test_result = false;
            break;
        }
    }

    delete A;
    delete b;

    REQUIRE(test_result);
}


TEST_CASE("testing the gaussian elimination function")
{
    // flag to check if we pass or fail the test
    bool test_result = true;

    int rows = 2;
    int cols = 2;

    // create a blank square matrix
    auto *A = new Matrix<double>(rows, cols, true);

    // create rhs vector
    auto *b = new Matrix<double>(cols, 1, true);

    // define the values we want to fill the matrix with
    double A_values[4] = {2, 3, 1, -4};
    double b_values[2] {7, 3};

    // use the set matrix function to fill the values of the array in 1 go
    A->setMatrix(4, A_values);
    b->setMatrix(2, b_values);

    SECTION("test the row swap function") {
        // row indexes we want to swap
        int i = 0;
        int j = 1;

        // this should swap rows i and j
        A->swapRows(b, i, j);

        // construct the test values - what we expect after swap
        double A_correct_values[4] = {1, -4, 2, 3};
        double b_correct_values[2] = {3, 7};

        // check A values are correct
        for (int i=0; i<A->rows*A->cols; i++)
        {
            // check values of A
            if (!fEqual(A->values[i], A_correct_values[i], TOL))
            {
                test_result = false;
                break;
            }

            // check values of b
            if (i < b->rows*b->cols && !fEqual(b->values[i], b_correct_values[i], TOL))
            {
                test_result = false;
                break;
            }
        }

        REQUIRE(test_result);
    }

    SECTION("test the upper triangular function")
    {
        // construct the upper triangle matrix for A
        A->upperTriangular(b);

        // construct the test values
        double A_correct_values[4] = {2, 3, 0, -5.5};
        double b_correct_values[2] = {7, -0.5};

        // check A values are correct
        for (int i=0; i<rows*cols; i++)
        {
            if (!fEqual(A->values[i], A_correct_values[i], TOL))
            {
                test_result = false;
                break;
            }
        }

        REQUIRE(test_result);

        // check b values are correct
        for (int i=0; i<rows; i++)
        {
            if (!fEqual(b->values[i], b_correct_values[i], TOL))
            {
                test_result = false;
                break;
            }
        }

        REQUIRE(test_result);
    }

    SECTION("Gaussian elimination solver")
    {
        //    clock_t start = clock();
        // create vector to hold our matrix
        auto *solution = A->solveGaussian(b);
        //    clock_t end = clock();
//            std::cout << "\ntime spent " << (end - start) / (double) (CLOCKS_PER_SEC) * 1000.0 << std::endl;

        double correct_solution[2] = {3.36364, 0.0909091};

        // check the values are expected
        for (int i=0; i<solution->rows; i++)
        {
            if (!fEqual(solution->values[i], correct_solution[i], TOL))
            {
                test_result = false;
                break;
            }
        }

        delete solution;

        REQUIRE(test_result);
    }

    // clean up memory
    delete A;
    delete b;
    REQUIRE(test_result);
}

TEST_CASE("gauss-seidel solver")
{
    bool test_result = true;

    int rows = 2;
    int cols = 2;

    auto* A = new Matrix<double>(rows, cols, true);

    // create rhs vector
    auto* b = new Matrix<double>(cols, 1, true);

    double A_values[4] = { 2, 1, 5, 7 };
    double b_values[2]{ 7, 3 };
    /* Representing the system:
       [ 2  1 ]   [ x ]  = [ 7 ]
       [ 5  7 ]   [ y ]  = [ 3 ]*/

    A->setMatrix(4, A_values);
    b->setMatrix(2, b_values);

    double initial_guess[2] = { 1, 1 };

    auto solution = A->solveGaussSeidel(b, TOL, 1000, initial_guess);

    double correct_values[2] = { 5.11111, -3.22222 };

    for (int i = 0; i < 2; i++)
    {
        if (!fEqual(solution->values[i], correct_values[i], TOL))
        {
            test_result = false;
            break;
        }
    }

    delete A;
    delete b;
    delete solution;

    REQUIRE(test_result);
}

TEST_CASE("lu decomposition")
{
    bool test_result = true;

    // LU Decomp only works on square matrices
    int rows = 4;
    int cols = 4;

    // Create Matrix to apply LU decomp to
    auto *A = new Matrix<double>(rows, cols, true);

    // create rhs vector
    auto *b = new Matrix<double>(cols, 1, true);

    double A_values[16] = {1, 0, 3, 7, 2, 1, 0, 4, 5, 4, 1, -2, 4, 1, 6, 2};
    double b_values[4] {1, 2, -3, 2};
    /* Create the system:
       [ 1 0 3 7 ] [ x1 ] = [ 1 ]
       [ 2 1 0 4 ] [ x2 ] = [ 2 ]
       [ 5 4 1 -2] [ x3 ] = [-3 ]
       [ 4 1 6 2 ] [ x4 ] = [ 2 ]*/


    double correct_values[4] = {2.6375, -3.7750000000000004, -0.8375, 0.125};

    A->setMatrix(16, A_values);
    b->setMatrix(4, b_values);

    auto solution = A->solveLU(b);

    for (int i=0; i<4; i++)
    {
        if (!fEqual(solution->values[i], correct_values[i], TOL))
        {
            test_result = false;
            break;
        }
    }

    delete A;
    delete b;
    delete solution;

    REQUIRE(test_result);
}

TEST_CASE("lu decomposition test - partial pivoting")
{
    bool test_result = true;

    int rows = 4;
    int cols = 4;

    auto *matrix = new Matrix<double>(rows, cols, true);
    auto *upper_tri = new Matrix<double>(rows, cols, true);
    auto *lower_tri = new Matrix<double>(rows, cols, true);
    auto *permutation = new Matrix<double>(rows, cols, true);

    double values[16] = {5, 7, 5, 9, 5, 14, 7, 10, 20, 77, 41, 48, 25, 91, 55, 67};

    // these are the correct values that we expect from the upper and lower triangle decomposition
    double correct_upper_tri[16] = {25, 91, 55, 67, 0, -11.2, -6, -4.4, 0, 0, -5.25, -7.25, 0, 0, 0, 0.666667};
    double correct_lower_tri[16] = {1, 0, 0, 0, 0.2, 1, 0, 0, 0.8, -0.375, 1, 0, 0.2, 0.375, 0.3333, 1};

    // fill the values of the matrix
    matrix->setMatrix(16, values);

    // decompose A into upper and lower triangle
    matrix->luDecompositionPivot(upper_tri, lower_tri, permutation);

    for (int i = 0; i < rows * cols; i++) {
        // test upper triangle values
        if (!fEqual(upper_tri->values[i], correct_upper_tri[i], TOL))
        {
            test_result = false;
            break;
        }

        // test lower triangle values
        if (!fEqual(lower_tri->values[i], correct_lower_tri[i], TOL)) {
            test_result = false;
            break;
        }
    }

    delete matrix;
    delete upper_tri;
    delete lower_tri;
    REQUIRE(test_result);
}

TEST_CASE("jacobi iteration")
{
    bool test_result = true;

    int rows = 2;
    int cols = 2;

    auto *A = new Matrix<double>(rows, cols, true);

    // create rhs vector
    auto *b = new Matrix<double>(cols, 1, true);

    double A_values[4] = {2, 1, 5, 7};
    double b_values[2] {7, 3};

    A->setMatrix(4, A_values);
    b->setMatrix(2, b_values);

    double initial_guess[2] = {1, 1};

    auto solution = A->solveJacobi(b, TOL, 1000, initial_guess);

    double correct_values[2] = {5.11111, -3.22222};

    for (int i=0; i<2; i++)
    {
        if (!fEqual(solution->values[i], correct_values[i], TOL))
        {
            test_result = false;
            break;
        }
    }

    delete A;
    delete b;
    delete solution;

    REQUIRE(test_result);
}



TEST_CASE("lu decomposition test - no partial pivoting")
{
    bool test_result = true;

    int rows = 4;
    int cols = 4;

    auto *matrix = new Matrix<double>(rows, cols, true);
    auto *upper_tri = new Matrix<double>(rows, cols, true);
    auto *lower_tri = new Matrix<double>(rows, cols, true);

    double values[16] = {5, 7, 5, 9, 5, 14, 7, 10, 20, 77, 41, 48, 25, 91, 55, 67};

    // these are the correct values that we expect from the upper and lower triangle decomposition
    double correct_upper_tri[16] = {5, 7, 5, 9, 0, 7, 2, 1, 0, 0, 7, 5, 0, 0, 0, 4};
    double correct_lower_tri[16] = {1, 0, 0, 0, 1, 1, 0, 0, 4, 7, 1, 0, 5, 8, 2, 1};

    // fill the values of the matrix
    matrix->setMatrix(16, values);

    // decompose A into upper and lower triangle
    matrix->luDecomposition(upper_tri, lower_tri);

    for(int i=0; i<rows*cols; i++)
    {
        // test upper triangle values
        if (!fEqual(upper_tri->values[i], correct_upper_tri[i], TOL))
        {
            test_result = false;
            break;
        }

        // test lower triangle values
        if (!fEqual(lower_tri->values[i], correct_lower_tri[i], TOL))
        {
            test_result = false;
            break;
        }
    }

    delete matrix;
    delete upper_tri;
    delete lower_tri;
    REQUIRE(test_result);
}

TEST_CASE("Conjugate Gradient Method")
{
    bool test_result = true;

    double A_values[9] = {2, -1, 0, -1, 3, -1, 0, -1, 2};
    double b_values[3] = {1, 8, -5};

    auto A = new Matrix<double>(3, 3, true);
    auto b = new Matrix<double>(3, 1, true);

    A->setMatrix(9, A_values);
    b->setMatrix(3, b_values);

//    A->printValues();
//    b->printValues();

    auto result = A->conjugateGradient(b, TOL, 1000);

    result->printValues();


    delete A;
    delete b;
    delete result;

    REQUIRE(test_result);
}


#endif