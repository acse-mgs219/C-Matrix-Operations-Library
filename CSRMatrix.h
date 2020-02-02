#ifndef CSRMATRIX_H
#define CSRMATRIX_H

#include "Matrix.h"

template <class T>
class CSRMatrix : public Matrix<T>
{
public:

    // constructor where we want to preallocate ourselves
    CSRMatrix(int rows, int cols, int nnzs, bool preallocate);

    // constructor where we already have allocated memory outside
    CSRMatrix(int rows, int cols, int nnzs, T* values_ptr, int* row_position, int* col_index);

    // constructor where we already have allocated memory outside
    CSRMatrix(Matrix<T>* dense);

    // destructor
    ~CSRMatrix();

    // Print out the values in our matrix
    virtual void printMatrix();

    void printNonZeroValues();

    void setMatrix(T* values_ptr, int iA[], int jA[]);

    // Perform some operations with our matrix
    CSRMatrix<T>* matMatMult(CSRMatrix<T>& mat_right);
    CSRMatrix<T>* matMatMultSymbolic(CSRMatrix<T>& mat_right, std::vector< std::pair< std::pair<int, int>, T> >& result);
    void matMatMultNumeric(CSRMatrix<T>* symbolic_res, std::vector< std::pair< std::pair<int, int>, T> >& result);

    // Perform some operations with our matrix
    Matrix<T>* matVecMult(Matrix<T>& vector);

    void sort_mat(Matrix<T>* rhs);
    void find_unique_p(std::vector<bool> check_list, std::vector<int>& unique_list);

    // Explicitly using the C++11 nullptr here
    int* row_position = nullptr;
    int* col_index = nullptr;

    // How many non-zero entries we have in the matrix
    int nnzs = -1;

    // Private variables - there is no need for other classes
    // to know about these variables
private:
};


#endif