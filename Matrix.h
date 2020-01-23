#pragma once

#ifndef LECTURES_MATRIX_H
#define LECTURES_MATRIX_H

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
    bool is_row_major = true;

    // constructor
    Matrix(int rows, int cols, bool preallocate, bool is_row_major);
    Matrix(int rows, int cols, T *values_ptr, bool is_row_major);

    // destructor
    virtual ~Matrix();

    // set value of element at a certain position to the value
    void setValue(int row_index, int col_index, T value);

    // set all the values of the matrix
    void setMatrix(int length, T *values_ptr);

    // get the value of an element at a certain position
    void getValue(int row_index, int col_index);

    // print values of the matrix
    void printValues();
    virtual void printMatrix();

    /////////// Matrix Operations Methods /////
    // matrix multiplication
    void matMatMul(Matrix<T>& mat_right, Matrix<T>& output);

    // function that implements gaussian elimination
    Matrix<T> *gaussianElimination(Matrix<T> *b);

    // upper triangular elimination
    void upperTriangular(Matrix<T> *b);

    // back substitution
    Matrix<T> *backSubstitution(Matrix<T> *b);

    // lu decomposition function
    void luDecomposition(Matrix<T> *upper_tri, Matrix<T> *lower_tri);
    void luDecomposition_pp(Matrix<T> *upper_tri, Matrix<T> *lower_tri);

    // swap rows
    void swapRows(Matrix<T> *b, int i, int j);
    void swapRowsMatrix(int i, int j);

protected:
    bool preallocated = false;

private:
    int size_of_values = -1;
};

#endif //LECTURES_MATRIX_H
