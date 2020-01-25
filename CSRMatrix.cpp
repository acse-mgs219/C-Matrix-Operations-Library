#include "CSRMatrix.h"
#include <iostream>
#include "CSRMatrix.h"

// Constructor - using an initialisation list here
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, bool preallocate, bool is_row_major):
Matrix<T>(rows, cols, false, is_row_major), nnzs(nnzs)
{
    // If we don't pass false in the initialisation list base constructor, it would allocate values to be of size
    // rows * cols in our base matrix class
    // So then we need to set it to the real value we had passed in
    this->preallocated = preallocate;

    // If we want to handle memory ourselves
    if (this->preallocated)
    {
        // Must remember to delete this in the destructor
        this->values = new T[this->nnzs];
        this->row_position = new int[this->rows+1];
        this->col_index = new int[this->nnzs];

        // set all the values to 0
        for (int i=0; i<this->nnzs; i++)
        {
            this->values[i] = 0;
        }
    }
}

// Constructor - now just setting the value of our T pointer
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, T *values_ptr, int *row_position, int *col_index, bool is_row_major):
Matrix<T>(rows, cols, values_ptr, is_row_major), nnzs(nnzs), row_position(row_position), col_index(col_index)
{}

// destructor
template <class T>
CSRMatrix<T>::~CSRMatrix()
{
    // Delete the values array
    if (this->preallocated){
        delete[] this->row_position;
        delete[] this->col_index;
    }
    // The super destructor is called after we finish here
    // This will delete this->values if preallocated is true
}

// Explicitly print out the values in values array as if they are a matrix
template <class T>
void CSRMatrix<T>::printMatrix()
{
    std::cout << "Printing matrix" << std::endl;
    std::cout << "Values: ";

    for (int j = 0; j< this->nnzs; j++)
    {
        std::cout << this->values[j] << " ";
    }
    std::cout << std::endl;
    std::cout << "row_position: ";

    for (int j = 0; j< this->nnzs+1; j++)
    {
        std::cout << this->row_position[j] << " ";
    }

    std::cout << std::endl;
    std::cout << "col_index: ";

    for (int j = 0; j<this->nnzs; j++)
    {
        std::cout << this->col_index[j] << " ";
    }
    std::cout << std::endl;
}

// Do a matrix-vector product
// output = this * input
template<class T>
void CSRMatrix<T>::matVecMult(T *input, T *output)
{
    if (input == nullptr || output == nullptr)
    {
        std::cerr << "Input or output haven't been created" << std::endl;
        return;
    }

    // Set the output to zero
    for (int i = 0; i < this->rows; i++)
    {
        output[i] = 0.0;
    }

    int val_counter = 0;

    // Loop over each row
    for (int i = 0; i<this->rows; i++)
    {
        // Loop over all the entries in this col
        for (int val_index = this->row_position[i]; val_index < this->row_position[i+1]; val_index++)
        {
            // This is an example of indirect addressing
            // Can make it harder for the compiler to vectorize!
            output[i] += this->values[val_index] * input[this->col_index[val_index]];

        }
    }
}


// Do matrix matrix multiplication
// output = this * mat_right
template <class T>
void CSRMatrix<T>::matMatMult(CSRMatrix<T>& mat_right, CSRMatrix<T>& output)
{
    // Check our dimensions match
    if (this->cols != mat_right.rows)
    {
        std::cerr << "Input dimensions for matrices don't match" << std::endl;
        return;
    }

    // Check if our output matrix has had space allocated to it
    if (output.values != nullptr)
    {
        // Check our dimensions match
        if (this->rows != output.rows || this->cols != output.cols)
        {
            std::cerr << "Input dimensions for matrices don't match" << std::endl;
            return;
        }
    }
        // The output hasn't been preallocated, so we are going to do that
    else
    {
        std::cerr << "OUTPUT HASN'T BEEN ALLOCATED" << std::endl;

    }

    // Set the output to zero
    for (int i = 0; i < this->rows; i++)
    {
        output.values[i] = 0;
    }

    // Loop over each row
    for (int i = 0; i<this->rows; i++)
    {
        // Loop over all the entries in this row
        for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
        {
            std::cout << i << " " << this->col_index[val_index] << " " << this->values[val_index] << std::endl;


            for (int c=0; c<mat_right.cols; c++)
            {
                if (c == i)
                {
                    std::cout << c << std::endl;
                }
            }


        }
        std::cout << std::endl;
    }

    // HOW DO WE SET THE SPARSITY OF OUR OUTPUT MATRIX HERE??
}

template<class T>
void CSRMatrix<T>::printNonZeroValues()
{

    if (this->values == nullptr || this->row_position == nullptr || this->col_index == nullptr)
    {
        throw std::invalid_argument("matrix has not been set");
    }

    std::cout << "Printing non-zero values of the sparse matrix" << std::endl;

    for (int i=0; i<this->nnzs; i++)
    {
        std::cout << this->values[i] << " ";
    }

    std::cout << std::endl;
    std::cout << "row position: " << std::endl;

    for (int i=0; i<this->nnzs+1; i++)
    {
        std::cout << this->row_position[i] << " ";
    };

    std::cout << std::endl;
    std::cout << "column position: " << std::endl;

    for (int i=0; i<this->nnzs; i++)
    {
        std::cout << this->col_index[i] << " ";
    };

    std::cout << std::endl;

}

template<class T>
void CSRMatrix<T>::setMatrix(T *values_ptr, int iA[], int jA[])
{
    iA[0];
    for (int i=0; i<this->nnzs+1; i++)
    {
        if (i<this->nnzs)
        {
            this->values[i] = values_ptr[i];
            this->col_index[i] = jA[i];
        }

        this->row_position[i] = iA[i];
    };
}