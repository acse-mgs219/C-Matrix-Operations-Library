#pragma once

#ifndef MATRIX_H
#define MATRIX_H

template<class T>
class Matrix {
public:
    // dimensions of the matrix
    int rows = -1;
    int cols = -1;

    // pointer to where values are stored
    T *values = nullptr;

    // constructor
    Matrix(int rows, int cols, bool preallocate);
    Matrix(int rows, int cols, T *values_ptr);

    // destructor
    virtual ~Matrix();

    // set value of element at a certain position to the value
    void setValue(int row_index, int col_index, T value);

    // set all the values of the matrix
    virtual void setMatrix(int length, T *values_ptr);

    // get the value of an element at a certain position
    T getValue(int row_index, int col_index);

    /////////// Matrix Operations Methods /////
    // matrix multiplication
    Matrix<T> *matMatMult(Matrix<T>& mat_right);

    void transpose();

    // calculate inner product
    T innerVectorProduct(Matrix<T>& mat_right);

    // print values of the matrix
    void printValues();
    virtual void printMatrix();

    /////////// Solvers /////
    Matrix<T> *conjugateGradient(Matrix<T> *b, double TOL, int max_iterations);

    // jacobi iterative solver
    Matrix<T> *solveJacobi(Matrix<T> *b, double tolerance, int max_iterations, T initial_guess[]);

    // gauss seidel iterative solver
    Matrix<T> *solveGaussSeidel(Matrix<T> *b, double tolerance, int max_iterations, T initial_guess[]);

    // function that implements gaussian elimination
    Matrix<T> *solveGaussian(Matrix<T> *b);

    // uses direct LU decomposition to solve the matrix system
    Matrix<T> *solveLU(Matrix<T> *b);

    /// HELPER FUNCTIONS - SHOULD EVENTUALLY BE MADE PRIVATE
    // lu decomposition function
    void luDecomposition(Matrix<T> *upper_tri, Matrix<T> *lower_tri);

    void luDecompositionPivot(Matrix<T> *upper_tri, Matrix<T> *lower_tri, Matrix<T> *permutation);

    // changes the matrix to upper triangular
    void upperTriangular(Matrix<T> *b);

    // helper function that back substitutes solution
    Matrix<T> *backSubstitution(Matrix<T> *b);

    // helper function that forward substitutes solution
    Matrix<T> *forwardSubstitution(Matrix<T> *b);

    // swap rows
    void swapRows(Matrix<T> *b, int i, int j);
    void swapRowsMatrix(int i, int j);

protected:
    bool preallocated = false;

private:
    int size_of_values = -1;
};

#endif //LECTURES_MATRIX_H
