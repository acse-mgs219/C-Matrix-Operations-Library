#pragma once

template <class T>
class Matrix
{
public:

   // constructor where we want to preallocate ourselves
   Matrix(int rows, int cols, bool preallocate);
   // constructor where we already have allocated memory outside
   Matrix(int rows, int cols, T *values_ptr);
   // destructor
   virtual ~Matrix();

   // Print out the values in our matrix
   void printValues();
    virtual void printMatrix();

   // Perform some operations with our matrix
   void matMatMult(Matrix<T>& mat_right, Matrix<T>& output);

   // Explicitly using the C++11 nullptr here
   T *values = nullptr;
   int rows = -1;
   int cols = -1;

    // LU function;

    void LU_solve(T* rhs, T *result);
    void LU_decomposition(T *L, T *R);
    void backward_substitution(T * rhs, T * result);
    void forward_substitution(T * rhs, T * result);
    void sort_mat(Matrix<T> *rhs);
    void find_unique(std::vector<bool> check_list, std::vector<int> &unique_list);

// We want our subclass to know about this
protected:
   bool preallocated = false;
    // LU function


// Private variables - there is no need for other classes
// to know about these variables
private:
   int size_of_values = -1;
};


