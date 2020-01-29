#include "Matrix.h"
#include <memory>
#include <stdexcept>
#include <iostream>
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

// convert A into upper triangular form - with partial pivoting
template <class T>
void Matrix<T>::upperTriangular(Matrix<T> *b)
{
    // check if A is square
    if (this->rows != this->cols)
    {
        throw std::invalid_argument("A should be a square matrix!");
    }

    // check that the dimensions of A and b are compatible
    if (this->rows != b->size_of_values)
    {
        throw std::invalid_argument("A and b dimensions dont match");
    }

    // scaling factor
    double s = -1;
    int kmax = -1;

    // loop over each pivot row except the last one
    for (int k=0; k<this->rows-1; k++)
    {
        // initialize with current pivot row
        kmax = k;

        // find pivot column to avoid zeros on diagonal
        for (int i=k+1; i<this->rows; i++)
        {
            if (fabs(this->values[kmax*this->cols + k]) < fabs(this->values[i*this->cols + k]))
            {
                kmax = i;
            }
        }

        this->swapRows(b, kmax, k);

        // loop over each row below the pivot
        for (int i=k+1; i<this->rows; i++)
        {
            // calculate scaling value for this row
            s = this->values[i * this->cols + k] / this->values[k * this->cols + k];

            // start looping from k and update the row
            for (int j=k; j<this->rows; j++)
            {
                this->values[i*this->cols + j] -= s * this->values[k*this->cols + j];
            }

            // update corresponding entry of b
            b->values[i] -= s * b->values[k];
        }
    }
}

template <class T>
Matrix<T> *Matrix<T>::backSubstitution(Matrix<T> *b)
{
    // check if A is square
    if (this->rows != this->cols)
    {
        throw std::invalid_argument("A should be a square matrix!");
    }

    // check that the dimensions of A and b are compatible
    if (this->rows != b->size_of_values)
    {
        throw std::invalid_argument("A and b dimensions don't match");
    }

    // create an empty vector
    Matrix<T> *solution = new Matrix<T>(b->rows, b->cols, true);

    double s;

    // iterate over system backwards
    for (int k=b->size_of_values-1; k>=0; k--)
    {
        s = 0;

        for (int j=k+1; j<b->size_of_values; j++)
        {
            // assumes row major order
            s += this->values[k * this->cols + j] * solution->values[j];
        }

        solution->values[k] = (b->values[k] - s) / this->values[k * this->cols + k];
    }

    return solution;
}

template<class T>
Matrix<T> *Matrix<T>::forwardSubstitution(Matrix<T> *b)
{
    // check if A is square
    if (this->rows != this->cols)
    {
        throw std::invalid_argument("A should be a square matrix!");
    }

    // check that the dimensions of A and b are compatible
    if (this->rows != b->size_of_values)
    {
        throw std::invalid_argument("A and b dimensions don't match");
    }

    // create an empty vector
    Matrix<T> *solution = new Matrix<T>(b->rows, b->cols, true);

    double s;

    // iterate over system
    for (int k=0; k<b->size_of_values; k++)
    {
        s = 0;

        for (int j=0; j<k; j++)
        {
            // assumes row major order
            s += this->values[k * this->cols + j] * solution->values[j];
        }

        solution->values[k] = (b->values[k] - s) / this->values[k * this->cols + k];
    }

    return solution;
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

// function that implements gaussian elimination
template<class T>
Matrix<T> *Matrix<T>::solveGaussian(Matrix<T> *b)
{
    // transform matrices to upper triangular
    this->upperTriangular(b);

    // generate solution
    auto *solution = this->backSubstitution(b);

    return solution;
}



// implementation of the LU decomposition function
template<class T>
void Matrix<T>::luDecomposition(Matrix<T> *upper_tri, Matrix<T> *lower_tri)
{
    // make sure the matrix is square
    if (this->cols != this->rows)
    {
        throw std::invalid_argument("input has wrong number dimensions");
    }

    double s = -1;

    // copy the values of A into upper triangular matrix
    for (int i=0; i<this->size_of_values; i++)
    {
        upper_tri->values[i] = this->values[i];
    }

    // loop over each pivot row
    for (int k=0; k<this->rows-1; k++)
    {
        // loop over each equation below the pivot
       for (int i=k+1; i<this->rows; i++)
       {
           // assumes row major order
           s = upper_tri->values[i * this->rows + k] / upper_tri->values[k * upper_tri->rows + k];

           for (int j=k; j<this->rows; j++)
           {
               upper_tri->values[i * this->rows + j] -= s * upper_tri->values[k * upper_tri->rows + j];
           }

           lower_tri->values[i * this->rows + k] = s;
       }
    }

    // add zeroes to the diagonal
    for (int i=0; i<this->rows; i++)
    {
        lower_tri->values[i * lower_tri->rows + i] = 1;
    }
}


template<class T>
void Matrix<T>::luDecompositionPivot(Matrix<T> *upper_tri, Matrix<T> *lower_tri, Matrix<T> *permutation)
{
    // make sure the matrix is square
    if (this->cols != this->rows)
    {
        throw std::invalid_argument("input has wrong number dimensions");
    }

    int max_index = -1;
    int max_val = -1;
    double s = -1;

    // copy the values of A into upper triangular matrix
    for (int i=0; i<upper_tri->size_of_values; i++)
    {
        upper_tri->values[i] = this->values[i];
    }

    // make permuation matrix an idenity matrix
    for (int i=0; i<permutation->rows; i++)
    {
        permutation->values[i * permutation->cols + i] = 1;
    }

    // loop over each pivot row
    for (int k=0; k<upper_tri->rows-1; k++)
    {
        max_val = -1;
        max_index = k;

        // find the index of the largest value in the column
        for (int z=k; z<upper_tri->rows; z++)
        {
            if (fabs(upper_tri->values[z * upper_tri->cols + k]) > max_val)
            {
                max_val = fabs(upper_tri->values[z * upper_tri->cols + k]);
                max_index = z;
            }
        }

        max_index;

        upper_tri->swapRowsMatrix(k, max_index);
        lower_tri->swapRowsMatrix(k, max_index);
        permutation->swapRowsMatrix(k, max_index);

        // loop over each equation below the pivot
        for (int i=k+1; i<upper_tri->rows; i++)
        {
            // assumes row major order
            s = upper_tri->values[i * upper_tri->cols + k] / upper_tri->values[k * upper_tri->cols + k];

            for (int j=k; j<upper_tri->cols; j++)
            {
                upper_tri->values[i * upper_tri->cols + j] -= s * upper_tri->values[k * upper_tri->cols + j];
            }

            lower_tri->values[i * lower_tri->rows + k] = s;
        }
    }

    // add zeroes to the diagonal
    for (int i=0; i<upper_tri->rows; i++)
    {
        lower_tri->values[i * lower_tri->rows + i] = 1;
    }

    permutation->transpose();
}



template<class T>
Matrix<T> *Matrix<T>::solveJacobi(Matrix<T> *b, double tolerance, int max_iterations, T initial_guess[])
{
    // create some space to hold the solution to the iteration
    auto x_var = new Matrix<T>(b->rows, b->cols, true);
    auto x_var_prev = new Matrix<T>(b->rows, b->cols, true); // is b->cols always 1?

    x_var_prev->setMatrix(b->size_of_values, initial_guess); // should check that sizes are correct

    auto estimated_rhs = this->matMatMult(*x_var_prev);

    // initialize residual which will be used to determine ending position
    double residual = tolerance * 2;
    double resid_sum; // not actually necessary
    double *sum = new double[this->cols];
    int iteration = 0;

    while (residual > tolerance && iteration < max_iterations)
    {
        for (int i=0; i<this->rows; i++) // should be this->rows?
        {
            sum[i] = 0;

            for (int j=0; j<this->cols; j++) // should be this->cols?
            {
                if (i != j)
                {
                    sum[i] += this->values[i * this->cols + j] * x_var_prev->values[j];
                }
            }
        }

        for (int i = 0; i < this->rows; i++) // should be this->rows?
        {
            x_var->values[i] = 1 / this->values[i * this->rows + i] * (b->values[i] - sum[i]);
            x_var_prev->values[i] = x_var->values[i];
        }

        resid_sum = 0;

        // check residual
        for (int i=0; i<b->size_of_values; i++)
        {
            resid_sum += fabs(estimated_rhs->values[i] - b->values[i]);
        }

        residual = resid_sum / b->size_of_values;
        ++iteration;
    }

    // clean memory
    delete x_var_prev;
    delete estimated_rhs;
    delete[] sum;

    return x_var;
}

template<class T>
Matrix<T> *Matrix<T>::solveGaussSeidel(Matrix<T> *b, double tolerance, int max_iterations, T *initial_guess) {

    // create some space to hold the solution to the iteration
    auto x_var = new Matrix<T>(b->rows, b->cols, true);

    x_var->setMatrix(b->rows, initial_guess);

    auto estimated_rhs = this->matMatMult(*x_var);

    // initialize residual which will be used to determine ending position
    double residual = tolerance * 2;
    double resid_sum;
    double sum;
    int iteration = 0;

    while (residual > tolerance && iteration < max_iterations)
    {
        for (int i=0; i<b->size_of_values; i++)
        {
            sum = 0;

            for (int j=0; j<b->size_of_values; j++)
            {
                if (i != j)
                {
                    sum += this->values[i * this->cols + j] * x_var->values[j];
                }
            }

            x_var->values[i] = 1 / this->values[i * this->cols + i] * (b->values[i] - sum);
        }

        resid_sum = 0;

        // check residual
        for (int i=0; i<b->size_of_values; i++)
        {
            resid_sum += fabs(estimated_rhs->values[i] - b->values[i]);
        }

        residual = resid_sum / b->size_of_values;
        ++iteration;
    }

    // clean memory
    delete estimated_rhs;

    return x_var;
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

    this->values = new_values_ptr;
}

template<class T>
Matrix<T> *Matrix<T>::solveLU(Matrix<T> *b) {

    auto upper_tri = new Matrix<T>(this->rows, this->cols, true);
    auto lower_tri = new Matrix<T>(this->rows, this->cols, true);
    auto permutation = new Matrix<T>(this->rows, this->cols, true);

    this->luDecompositionPivot(upper_tri, lower_tri, permutation);

    permutation->transpose();

    auto p_inv_b = permutation->matMatMult(*b);

    auto y_values = lower_tri->forwardSubstitution(p_inv_b);


    auto *solution = upper_tri->backSubstitution(y_values);

    delete upper_tri;
    delete lower_tri;
    delete permutation;
    delete p_inv_b;
    delete y_values;

    return solution;
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

// solve Ax = b;
template<class T>
Matrix<T> *Matrix<T>::conjugateGradient(Matrix<T> *b, double epsilon, int max_iterations)
{
    int k = 0;
    T beta = 1;
    double alpha = 1;
    T delta_old = 1;

    // intialize to x to 0
    auto x = new Matrix<T>(b->rows, b->cols, true);

    // workout Ax
    auto Ax = this->matMatMult(*x);

    // r = b - Ax
    auto r = new Matrix<T>(b->rows, b->cols, true);
    for (int i=0; i<r->size_of_values; i++)
    {
        r->values[i] = b->values[i] - Ax->values[i];
    }

    auto p = new Matrix<T>(r->rows, r->cols, true);
    auto w = new Matrix<T>(r->rows, r->cols, true);

    double delta = r->innerVectorProduct(*r);

    while (k < max_iterations && (sqrt(delta) > epsilon*sqrt(b->innerVectorProduct(*b))))
    {
        if (k==1)
        {
            for (int i=0; i<p->size_of_values; i++)
            {
                p->values[i] = r->values[i];
            }

        } else {
            beta = delta / delta_old;

            // p = r + beta * p
            for (int i=0; i<p->size_of_values; i++)
            {
                p->values[i] = r->values[i] + beta*p->values[i];
            }
        }

        auto Ap = this->matMatMult(*p);

        for (int i=0; i<w->size_of_values; i++)
        {
            w->values[i] = Ap->values[i];
        }

        alpha = delta / p->innerVectorProduct(*w);

        for (int i=0; i<x->size_of_values; i++)
        {
            x->values[i] = x->values[i] + alpha*p->values[i];
        }

        for (int i=0; i<r->size_of_values; i++)
        {
            r->values[i] = r->values[i] - alpha*w->values[i];
        }

        delta_old = delta;
        delta = r->innerVectorProduct(*r);

        delete Ap;
        k++;
    }

    delete Ax;
    delete r;
    delete p;

    return x;
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
        result += this->values[i] * mat_right.values[i];
    }

    // return result
    return result;
}