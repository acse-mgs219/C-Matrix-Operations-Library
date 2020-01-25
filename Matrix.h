#pragma once

#ifndef MATRIX_H
#define MATRIX_H

#include <stdexcept>
#include <iostream>

template<class T>
class Matrix {
public:
    // dimensions of the matrix
    int rows = -1;
    int cols = -1;

    // pointer to where values are stored
    T *values = nullptr;

    // tells us whether matrix is row major
    bool is_row_major;

    // constructor
    Matrix(int rows, int cols, bool preallocate, bool is_row_major = true);
    Matrix(int rows, int cols, T *values_ptr, bool is_row_major = true);

    // destructor
    virtual ~Matrix();

    // set value of element at a certain position to the value
    void setValue(int row_index, int col_index, T value);

    // set all the values of the matrix
    virtual void setMatrix(int length, T *values_ptr);

    // get the value of an element at a certain position
    T getValue(int row_index, int col_index);

    void transpose();

    // print values of the matrix
    void printValues();
    virtual void printMatrix();

    /////////// Matrix Operations Methods /////

    // matrix multiplication
    void matMatMult(Matrix<T>& mat_right, Matrix<T>& output);

    // jacobi iterative solver
    Matrix<T> *solveJacobi(Matrix<T> *b, double tolerance, int max_iterations, T initial_guess[]);

    // gauss seidel iterative solver
    Matrix<T> *solveGaussSeidel(Matrix<T> *b, double tolerance, int max_iterations, T initial_guess[]);

    // function that implements gaussian elimination
    Matrix<T> *solveGaussian(Matrix<T> *b);

    Matrix<T> *solveLU(Matrix<T> *b);

    void upperTriangular(Matrix<T> *b);

    Matrix<T> *backSubstitution(Matrix<T> *b);

    Matrix<T> *forwardSubstitution(Matrix<T> *b);

    // lu decomposition function
    void luDecomposition(Matrix<T> *upper_tri, Matrix<T> *lower_tri);
    void luDecompositionPivot(Matrix<T> *upper_tri, Matrix<T> *lower_tri, Matrix<T> *permutation);

    // swap rows
    void swapRows(Matrix<T> *b, int i, int j);
    void swapRowsMatrix(int i, int j);

protected:
    bool preallocated = false;

private:
    int size_of_values = -1;
};

#endif //LECTURES_MATRIX_H
