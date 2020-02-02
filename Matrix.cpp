#include "Matrix.h"
#include <memory>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cmath>
//#include "cblas.h"

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

    if (!myfile)
        throw std::invalid_argument(fileName + " not found. Please check you have access to it");

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

template<class T>
Matrix<T> *Matrix<T>::matVectMult(Matrix<T> &b)
{
    if (b.cols != 1)
    {
        throw std::invalid_argument("b must be a column vector (number of columns == 1)");
    }

    if (this->cols != b.rows)
    {
        throw std::invalid_argument("A and b dimensions do not match");
    }

    // create output vector
    auto output = new Matrix<T>(b.rows, b.cols, true);

    #if defined(USE_BLAS)
    /*============================
    *  call to BLAS level 1 routine; determines dot product of row and column quickly.
    *  row-major order should take advantage of caching for fast memory access.
    ============================ */
    for (int i = 0; i < this->rows; i++)
    {
        output->values[i] = cblas_ddot(b.rows, (double *) (this->values + i * this->cols), 1, (double *) b.values, 1);
    }

    #else
    // Loop over each row of A
    for (int i = 0; i < this->rows; i++)
    {
        // go over the column and multiply element wise.
        for (int j=0; j < this->cols; j++)
        {
            output->values[i] += this->values[i * this->cols + j] * b.values[j];
        }
    }
    #endif

    return output;
}


// sets an element of the matrix to a designated value
template <class T>
void Matrix<T>::setValue(int row_index, int col_index, T value)
{
    this->values[row_index * this->cols + col_index] = value;
}

template <class T>
void Matrix<T>::makeRandom()
{
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < this->cols; j++)
        {
            this->values[i * this->cols + j] = rand() % 100;
        }
    }
}

template <class T>
void Matrix<T>::makeRandomSPD()
{
    auto tr = new Matrix<T>(this->rows, this->cols, true);
    // Makes this into a lower triangular matrix, with b its transpose
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            this->values[i * this->cols + j] = rand() % 100 + 1; // make sure no value on the diagonal is 0
            if (i == j)
                this->values[i * this->cols + j] += 100 * this->cols; // make sure eigen value is positive
            tr->values[j * this->cols + i] = this->values[i * this->cols + j];
        }

        for (int j = i+1; j < this->cols; j++)
        {
            this->values[i * this->cols + j] = 0;
            tr->values[j * this->cols + i] = 0;
        }
    }

    // L * L' is always SPD if all values on L's diagonal are strictly positive
    auto c = this->matMatMult(*tr);
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < this->cols; j++)
        {
            this->values[i * this->cols + j] = c->values[i * this->cols + j];
        }
    }
    delete tr;
    delete c;
}

template <class T>
void Matrix<T>::makeRandomSparseSPD()
{
    auto tr = new Matrix<T>(this->rows, this->cols, true);
    // Makes this into a lower triangular matrix, with b its transpose
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            this->values[i * this->cols + j] = rand() % 100 + 1; // make sure no value on the diagonal is 0
            if (i == j)
                this->values[i * this->cols + j] += 100 * this->cols; // make sure eigen value is positive
            tr->values[j * this->cols + i] = this->values[i * this->cols + j];
        }

        for (int j = i + 1; j < this->cols; j++)
        {
            this->values[i * this->cols + j] = 0;
            tr->values[j * this->cols + i] = 0;
        }
    }

    // L * L' is always SPD if all values on L's diagonal are strictly positive
    auto c = this->matMatMult(*tr);
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < this->cols; j++)
        {
            if (rand() % 100 > 70 || i == j)
                this->values[i * this->cols + j] = c->values[i * this->cols + j];
            else
                this->values[i * this->cols + j] = 0;
        }
    }
    delete tr;
    delete c;
}

template <class T>
void Matrix<T>::makeRandomDD()
{
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < this->cols; j++)
        {
            this->values[i * this->cols + j] = rand() % 100;
            if (i == j)
                this->values[i * this->cols + j] += 100 * this->cols; // max number in any cell is 100, there are this->col other cells
        }
    }
}

template <class T>
void Matrix<T>::setMatrix(int length, T *values_ptr)
{
    // trying to set the matrix with wrong size inputs
    if (length != this->size_of_values)
    {
        throw std::invalid_argument("input has wrong number of elements");
    }

    #if defined(USE_BLAS)
    /*====================================
     * use low level BLAS routine to quickly copy over desired values to where matrix/vector.
    ==================================== */
    cblas_dcopy(length, (double *) values_ptr, 1, (double *) this->values, 1);

    #else
    /*====================================
     *  naive implementation uses for loops to copy the elements over.
     *  // set the values of the array (just overwrite as we don't want dangling pointers)
     *==================================== */
    for (int i=0; i<length; i++)
    {
        this->values[i] = values_ptr[i];
    }
    #endif
}

template <class T>
void Matrix<T>::writeMatrix(std::string fileName)
{
    std::ofstream file;
    file.open(fileName);
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < this->cols; j++)
        {
            file << this->values[i * this->cols + j] << " ";
        }
        file << "\n";
    }
    file.close();
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
    std::cout << "\nPrinting in Matrix form:" << std::endl;
    
    // if the matrix is too big, only display the corner values 
    int rowStart, rowEnd, colStart, colEnd, dotRows = 0, dotCols = 0;
    if (this->rows > 20)
    {
        rowStart = 1; // display first 2 rows
        rowEnd = this->rows - 3; // and last 2
    }
    else
    {
        rowStart = this->rows; // if less than 20, just display all rows
        rowEnd = this->rows; // this does not matter
    }

    if (this->cols > 20)
    {
        colStart = 1; // display first 2 cols 
        colEnd = this->cols - 3; // and last 2
    }
    else
    {
        colStart = this->cols; // if less than 20, just display all cols
        colEnd = this->cols; // this does not matter
    }

    for (int i=0; i<this->rows; i++)
    {
        if (i > rowStart&& i < rowEnd)
        {
            if (dotRows < 3)
            {
                dotRows++;
                std::cout << ".\n";
            }
            else
                i = rowEnd; // no need to keep looping
            continue;
        }
        dotCols = 0; // display 3 dots on each row
        for (int j=0; j<this->cols; j++)
        {
            // we have explicitly assumed row-major ordering here
            if (j < colStart || j > colEnd)
                std::cout << this->values[i * this->cols + j] << " ";
            else if (dotCols < 3)
            {
                std::cout << " . ";
                dotCols++;
            }
            else
                j = colEnd; // no need to keep looking
        }
        std::cout << "\n";
    }

    std::cout << "\n";

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

    #if defined(USE_BLAS)
    /*======================================
     * Using the BLAS daxpy routine, we can take advantage of of the row by row access and contiguous memory.
     ======================================*/
    for (int i=0; i<this->rows; i++)
    {
        for (int k=0; k<this->cols; k++)
        {
            cblas_daxpy(output->cols, (double) this->values[i * this->cols + k], (double *) (mat_right.values + k * mat_right.cols), 1, (output->values + i * mat_right.cols), 1);
        }
    }
    #else
    /*======================================
     *  matrix multiplication is O(n^3).
     *  Although this is loop ordering takes advantage of caching, it
     *  does not take advantage of BLAS routines (for row by row access).
     *======================================*/
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
    #endif

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

    #if defined(USE_BLAS)
    // first copy into iA
    cblas_dcopy(this->cols, (double *) (this->values + i * this->cols), 1, (double *) iA, 1);

    // now copy into ib
    cblas_dcopy(b->cols, (double *) (b->values + i * b->cols), 1, (double *) ib, 1);

    // copy row j of A into row i of A
    cblas_dcopy(this->cols, (double *) (this->values + j * this->cols), 1, (double *) (this->values + i * this->cols), 1);

    // copy row 1 into row 2
    cblas_dcopy(this->cols, (double *) iA, 1, (double *) (this->values + j * this->cols), 1);

    // row j into row i of b
    cblas_dcopy(b->cols, (double *) (b->values + j * b->cols), 1, (double *) (b->values + i * b->cols), 1);

    // copy row i into row j
    cblas_dcopy(b->cols, (double *) ib, 1, (double *) (b->values + j * b->cols), 1);

    #else
    /*=================================
     * implementation without BLAS calls
     =================================*/
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
         //copy row j of A into row i of A
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
    #endif

    // clean memory
    delete[] iA;
    delete[] ib;
}

template<class T>
void Matrix<T>::swapRowsMatrix(int i, int j)
{
    // no swap required
    if (i == j)
        return;

    // create copy of the first row
    T *iA = new T[this->cols];

    #if defined(USE_BLAS)
    // copy row i into iA
    cblas_dcopy(this->cols, (double *) (this->values + i * this->cols), 1, (double *) iA, 1);

    //copy row j of A into row i of A
    cblas_dcopy(this->cols, (double *) (this->values + j * this->cols), 1, (double *) (this->values + i * this->cols), 1);

    //copy row 1 into row 2
    cblas_dcopy(this->cols, (double *) iA, 1, (double *) (this->values + j * this->cols), 1);

    #else
    /*===================================
    * Implementation without BLAS calls
    ===================================*/
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
    #endif

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
}

template <class T>
void Matrix<T>::sort_mat(Matrix<T>* rhs)
{
    auto* temp_mat = new Matrix<double>(this->rows, this->cols, true);
    auto* temp_rhs = new double[this->rows];

    std::vector <bool> check_list(this->cols, true);

    while (!(std::none_of(check_list.begin(), check_list.end(), [](bool v) { return v; })))
    {
        std::vector <int> unique_list(this->cols, -1);

        //update unique_list with find_unique function
        this->find_unique(check_list, unique_list);
        //if column j has a unique entry on row i (equals to "unique_list[j]"), then in temp_mat, set row j equals to (row i in original matrix), so that in temp_mat, the entry on [i,j] is the unique one, and then set this column j as false in the while loop to be excluded
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

        //Then fill the first available column with max value, and remove it from check_list;
        for (int j = 0; j < this->cols; j++)
        {

            // only need to deal with columns whose status is true.
            if (check_list[j])
            {

                int index_row(-1);
                int max_value(0);
                // loop over columns
                for (int row = j; row < this->rows;row++)
                {
                    // find the max value of column's elements
                    if (abs(this->values[row * this->rows + j]) > abs(max_value))
                    {
                        index_row = row;
                        max_value = this->values[row * this->rows + j];
                    }
                }

                // fill and exclude
                if (index_row != -1)
                {
                    for (int kk = 0; kk < this->cols; kk++)
                    {
                        temp_mat->values[j * this->cols + kk] = this->values[index_row * this->cols + kk];
                        this->values[index_row * this->cols + kk] = 0;
                    }
                    temp_rhs[j] = rhs->values[index_row];
                    check_list[j] = false;
                }
                j = this->cols;
            }
        }
    }

    // generate the new output matrix
    for (int i = 0; i < this->size_of_values; i++)
    {
        this->values[i] = temp_mat->values[i];
    }

    for (int i = 0; i < this->rows; i++)
    {
        rhs->values[i] = temp_rhs[i];
    }

    delete temp_mat;
    delete[] temp_rhs;

}

template <class T>
int Matrix<T>::size()
{
    return this->size_of_values;
}