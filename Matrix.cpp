#include "Matrix.h"
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cmath>

// constructor - creates a matrix initialized to 0
template <class T>
Matrix<T>::Matrix(int nrows, int ncols, bool preallocate) :
rows(nrows), cols(ncols), size_of_values(nrows * ncols), preallocated(preallocate)
{
    if (this->preallocated)
    {
        this->values = new T[this->size_of_values];
        // initialize values to 0
        for (int i=0; i<size_of_values; i++)
        {
            this->values[i] = 0;
        }
    }
}

template <class T>
Matrix<T>::Matrix(int nrows, int ncols, std::string fileName) :
rows(nrows), cols(ncols), size_of_values(nrows* ncols), preallocated(true)
{
    std::ifstream myfile;
    myfile.open(fileName);
    this->values = new T[this->size_of_values];
    // initialize values to 0
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            myfile >> this->values[i * cols + j];
        }
    } 
}


// constructor for not preallocated
template <class T>
Matrix<T>::Matrix(int nrows, int ncols, T *values_ptr) :
rows(nrows), cols(ncols), size_of_values(nrows * ncols), values(values_ptr)
{}

// destructor
template <class T>
Matrix<T>::~Matrix()
{
    // if preallocated delete
    if (this->preallocated)
    {
        delete[] this->values;
    }
}

// sets an element of the matrix to a designated value
template <class T>
void Matrix<T>::setValue(int row_index, int col_index, T value)
{
    this->values[row_index * this->cols + col_index] = value;
}

template <class T>
void Matrix<T>::setMatrix(int length, T *values_ptr)
{
    // trying to set the matrix with wrong size inputs
    if (length != this->size_of_values)
    {
        throw std::invalid_argument("input has wrong number of elements");
    }

    // set the values of the array (just overwrite as we don't want dangling pointers)
    for (int i=0; i<length; i++)
    {
        this->values[i] = values_ptr[i];
    }
}

// print values (not matrix form)
template <class T>
void Matrix<T>::printValues()
{
    std::cout << "Printing values in memory order:\n";
    for (int i=0; i<this->size_of_values; i++)
    {
        std::cout << this->values[i] << " ";
    }
    std::cout << std::endl;
};

// print matrix in matrix form
template <class T>
void Matrix<T>::printMatrix()
{
    std::cout << "Printing in Matrix form:" << std::endl;
    for (int i=0; i<this->rows; i++)
    {
        for (int j=0; j<this->cols; j++)
        {
            // we have explicitly assumed row-major ordering here
            std::cout << this->values[i * this->cols + j] << " ";
        }
        std::cout << "\n";
    }
}

// assumes user has already created mat_right and output matrices
template <class T>
Matrix<T> *Matrix<T>::matMatMult(Matrix& mat_right)
{
    // check dimensions make sense return without doing any multiplication
    if (this->cols != mat_right.rows)
    {
        throw std::invalid_argument("input dimensions don't match");
    }

   // create an output matrix that will hold our values
   auto output = new Matrix(this->rows, mat_right.cols, this->preallocated);

    // set output values to 0 beforehand
    for (int i = 0; i < output->size_of_values; i++)
    {
        output->values[i] = 0;
    }

    // matrix multiplication is O(n^3) - need to be careful about performance - need to be contiguous for fast performance
    for (int i=0; i<this->rows; i++)
    {
        for (int k=0; k<this->cols; k++)
        {
            for (int j=0; j<mat_right.cols; j++)
            {
                output->values[i * output->cols + j] += this->values[i * this->cols + k] * mat_right.values[k * mat_right.cols + j];
            }
        }
    }

    return output;
}

template<class T>
void Matrix<T>::swapRows(Matrix<T> *b, int i, int j)
{
    // no swap required
    if (i == j) {
        return;
    }

    // create copy of the first row (both A and b)
    T *iA = new T[this->cols];
    T *ib = new T[b->cols];

    for (int k=0; k<this->cols; k++)
    {
        iA[k] = this->values[i * this->cols + k];

        // also copy b
        if (k < b->cols)
        {
            ib[k] = b->values[i * b->cols + k];
        }
    }

    // swap the rows
    for (int k=0; k<this->cols; k++)
    {
        // copy row j of A into row i of A
        this->values[i * this->cols + k] = this->values[j *this->cols + k];

        // copy row 1 into row 2
        this->values[j * this->cols + k] = iA[k];

        if (k < b->cols)
        {
            // row j into row i of b
            b->values[i * b->cols + k] = b->values[j * b->cols + k];

            // row i into row j
            b->values[j * b->cols + k] = ib[k];

        }
    }

    // clean memory
    delete[] iA;
    delete[] ib;
}

template<class T>
void Matrix<T>::swapRowsMatrix(int i, int j)
{
    // no swap required
    if (i == j) {
        return;
    }

//    // create copy of the first row
    T *iA = new T[this->cols];

    for (int k=0; k<this->cols; k++)
    {
        iA[k] = this->values[i * this->cols + k];
    }

    // swap the rows
    for (int k=0; k<this->cols; k++)
    {
         //copy row j of A into row i of A
        this->values[i * this->cols + k] = this->values[j * this->cols + k];

         //copy row 1 into row 2
        this->values[j * this->cols + k] = iA[k];
    }

    // clean memory
    delete[] iA;
}

template<class T>
void Matrix<T>::transpose()
{
    // create a new values array to hold the data
    T *new_values_ptr = new T[this->size_of_values];

    for (int i=0; i<this->rows; i++)
    {
        for (int j=0; j<this->cols; j++)
        {
            new_values_ptr[i * this->cols + j] = this->values[j * this->cols + i];
        }
    }

    delete[] this->values;

    int temp = this->rows;
    this->rows = this->cols;
    this->cols = temp;
    this->values = new_values_ptr;
}

template<class T>
T Matrix<T>::getValue(int row_index, int col_index)
{
    if (row_index >= this->rows || col_index >= this->cols)
    {
        throw std::invalid_argument("wrong index values");
    }

    return this->values[row_index * this->cols + col_index];
}

template<class T>
T Matrix<T>::innerVectorProduct(Matrix<T> &mat_right)
{
    // ensure function is called on a vector
    if (this->cols != 1 || mat_right.cols != 1)
    {
        throw std::invalid_argument("both inputs should be vectors");
    }

    // check dimensions make sense
    if (this->size_of_values != mat_right.size_of_values)
    {
        throw std::invalid_argument("The number of values must match");
    }

    T result = 0;

    // calculate inner product
    for (int i=0; i<this->size_of_values; i++)
    {
        //std::cout << this->values[i] << " " << mat_right.values[i] << std::endl;
        result += this->values[i] * mat_right.values[i];
    }

    // return result
    return result;
}

template <class T>
void Matrix<T>::find_unique(std::vector<bool> check_list, std::vector<int>& unique_list)
{
    int row_index = 0;
    int count = 0;
    for (int i = 0; i < this->cols; i++)
    {
        if (check_list[i] == false) continue;

        for (int j = 0; j < this->rows; j++)
        {
            if (this->values[i + j * this->cols] != 0)
            {
                row_index = j;
                count++;
            }
        }
        if (count >= 2)
        {
            unique_list[i] = -1;
        }
        else
        {
            unique_list[i] = row_index;
        }
        count = 0;
    }
    //    for (int i = 0; i < this->cols; i++)
    //    {
    //        std::cout << unique_list[i] << " ";
    //    }
}

template <class T>
void Matrix<T>::sort_mat(Matrix<T>* rhs)
{
    auto* temp_mat = new Matrix<double>(this->rows, this->cols, true);
    auto* temp_rhs = new double[this->rows];

    std::vector <bool> check_list(this->cols, true);
    //    check_list[1] = false;
    //    check_list[0] = false;
    //    check_list[2] = 0;
    //    check_list[3] = false;
    while (!(std::none_of(check_list.begin(), check_list.end(), [](bool v) { return v; })))
    {
        //        std::cout<<"some are still inside";
        std::vector <int> unique_list(this->cols, -1);

        //update unique_list with hanchao's function
        this->find_unique(check_list, unique_list);
        //if column j has a unique entry on row i (equals to "unique_list[j]")
        //then in temp_mat, set row j equals to (row i in original matrix)
        //so that in temp_mat, the entry on [i,j] is the unique one;
        //set this column j as false in the while loop to be excluded
        for (int j = 0; j < this->cols; j++)
        {
            if (unique_list[j] != -1)
            {
                for (int col = 0; col < this->cols; col++)
                {
                    temp_mat->values[j * this->cols + col] = this->values[unique_list[j] * this->cols + col];
                    this->values[unique_list[j] * this->cols + col] = 0;
                }
                temp_rhs[j] = rhs->values[unique_list[j]];
                check_list[j] = false;
            }
        }

        //next, fill the 1st available column with max value,
        //and remove it from check_list;
        //remember to delete

        for (int j = 0; j < this->cols; j++)
        {
            //            std::cout<<"unique value: "<<check_list[j]<<std::endl;
            if (check_list[j])
            {
                //                std::cout<<"random assignment in progress: "<< j<<std::endl;
                int index_row(-1);
                int max_value(0);
                //now we fill temp_mat row j with value in row???? let's find out
                for (int row = j; row < this->rows;row++)
                {
                    if (abs(this->values[row * this->rows + j]) > abs(max_value))
                    {
                        index_row = row;
                        max_value = this->values[row * this->rows + j];
                    }
                }
                // now index_row takes the index of row???
                // fill and exclude
                if (index_row != -1) {
                    for (int kk = 0; kk < this->cols;kk++)
                    {
                        temp_mat->values[j * this->cols + kk] = this->values[index_row * this->cols + kk];
                        this->values[index_row * this->cols + kk] = 0;
                    }
                    temp_rhs[j] = rhs->values[index_row];
                    check_list[j] = false;
                }

                if (index_row == -1)
                {
                    std::cout << std::endl << "index cannot be found here: " << j << std::endl;

                }
                //                std::cout<<"random assignment finished: "<< j<<std::endl;
                j = this->cols;
            }
        }

    }

    for (int i = 0; i < this->size_of_values; i++)
    {
        this->values[i] = temp_mat->values[i];
    }

    for (int i = 0; i < this->rows; i++)
    {
        rhs->values[i] = temp_rhs[i];
    }

    delete temp_mat;
    delete temp_rhs;

}

template <class T>
int Matrix<T>::size()
{
    return this->size_of_values;
}

template<class T>
Matrix<T> *Matrix<T>::matVectMult(Matrix<T> &b)
{
    if (b.cols != 1)
    {
        throw std::invalid_argument("argument must be a column vector (number of columns = 1)");
    }
    if (this->cols != b.rows)
    {
        throw std::invalid_argument("A and b dimensions do not match");
    }

    // create output vector
    auto output = new Matrix<T>(b.rows, b.cols, true);

    // Loop over each row of A
    for (int i = 0; i < this->rows; i++)
    {
        // go over the column and
        for (int j=0; j < this->cols; j++)
        {
            output->values[i] += this->values[i * this->cols + j] * b.values[j];
        }
    }

    return output;

}
