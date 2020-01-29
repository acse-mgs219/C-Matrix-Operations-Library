//
//  Matrix.cpp
//  Matrix_solver_assignment
//
//  Created by Darren Shan on 2020/1/28.
//  Copyright © 2020 Darren Shan. All rights reserved.
//

#include <iostream>
#include "Matrix.hpp"
#include <vector>
#include <cmath>

// Constructor - using an initialisation list here
template <class T>
Matrix<T>::Matrix(int rows, int cols, bool preallocate): rows(rows), cols(cols), size_of_values(rows * cols), preallocated(preallocate)
{
   // If we want to handle memory ourselves
   if (this->preallocated)
   {
      // Must remember to delete this in the destructor
      this->values = new T[size_of_values];
   }
}

// Constructor - now just setting the value of our T pointer
template <class T>
Matrix<T>::Matrix(int rows, int cols, T *values_ptr): rows(rows), cols(cols), size_of_values(rows * cols), values(values_ptr)
{}

// destructor
template <class T>
Matrix<T>::~Matrix()
{
   // Delete the values array
   if (this->preallocated){
      delete[] this->values;
   }
}

// Just print out the values in our values array
template <class T>
void Matrix<T>::printValues()
{
   std::cout << "Printing values" << std::endl;
    for (int i = 0; i< this->size_of_values; i++)
   {
      std::cout << this->values[i] << " ";
   }
   std::cout << std::endl;
}

// Explicitly print out the values in values array as if they are a matrix
template <class T>
void Matrix<T>::printMatrix()
{
   std::cout << "Printing matrix" << std::endl;
   for (int j = 0; j< this->rows; j++)
   {
      std::cout << std::endl;
      for (int i = 0; i< this->cols; i++)
      {
         // We have explicitly used a row-major ordering here
         std::cout << this->values[i + j * this->cols] << " ";
      }
   }
   std::cout << std::endl;
}

// Do matrix matrix multiplication
// output = this * mat_right
template <class T>
void Matrix<T>::matMatMult(Matrix<T>& mat_right, Matrix<T>& output)
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
      output.values = new T[this->rows * mat_right.cols];
      // Don't forget to set preallocate to true now it is protected
      output.preallocated = true;
   }

   // Set values to zero before hand
   for (int i = 0; i < output.size_of_values; i++)
   {
      output.values[i] = 0;
   }

   // Now we can do our matrix-matrix multiplication
   // CHANGE THIS FOR LOOP ORDERING AROUND
   // AND CHECK THE TIME SPENT
   // Does the ordering matter for performance. Why??
   for(int i = 0; i < this->rows; i++)
   {
      for(int k = 0; k < this->cols; k++)
      {
         for(int j = 0; j < mat_right.cols; j++)
         {
               output.values[i * output.cols + j] += this->values[i * this->cols + k] * mat_right.values[k * mat_right.cols + j];
         }
      }
   }
}
/*
 def LU_decomposition(A):
 # construct upper triangular matrix contains Gaussian elimination result
 # we won't change A in-place but create a local copy
 # if we don't do this then A will be over-written by the U we
 # compute and return
 A = A.copy()
 m, n = A.shape
 assert(m == n)
 # For simplicity we set up a matrix to store L, but note the comment above
 # that if we had memory concerns we would reuse zeroed entries in A.
 # We don't initialise this to the identity now, as this won't be correct
 # when we use partial pivoting a little later.
 L = np.zeros((n,n))
 # Loop over each pivot row - don't need to consider the final row as a pivot
 for k in range(n-1):
     # Loop over each equation below the pivot row - now we do need to consider the last row
     for i in range(k+1, n):
         # Define the scaling factor outside the innermost
         # loop otherwise its value gets changed.
         s = (A[i, k] / A[k, k])
         for j in range(k, n):
             A[i, j] = A[i, j] - s*A[k, j]
         # store the scaling factors which make up the lower tri matrix
         L[i, k] = s
 # remember to add in the ones on the main diagonal to L
 L += np.eye(m)
 # A now is the upper triangular matrix U
 return L, A
 */
template <class T>
void Matrix<T>::LU_decomposition(T *lmat, T *umat)
{
    if (this->cols != this->rows) std::cout<<"rows != cols, cannot be decomposed with LU method";
    for (int i=0; i<this->size_of_values; i++)
    {
        lmat[i]=0;
        umat[i]=this->values[i];
    }
    
    for (int k =0; k<this->cols-1; k++)
    {
        for (int i =k+1; i<this->cols; i++)
        {
            T s = (this->values[i*this->cols+k]/this->values[k*this->cols+k]);
            
            for (int j =k; j<this->cols;j++)
            {
                umat[i*this->cols + j] = this->values[i*this->cols + j]- s * this->values[k*this->cols + j];
            }
            lmat[i*this->cols + k] = s;
            s = 0;
        }
    }
    for (int i =0;i<this->rows; i++)
    {
        lmat[i*this->cols+i] += 1;
    }
}

/*
 def backward_substitution(A, b):
 n = np.size(b)
 x = np.zeros(n)
 for k in range(n-1, -1, -1):
     s = 0.
     for j in range(k+1, n):
         s = s + A[k, j]*x[j]
     x[k] = (b[k] - s)/A[k, k]
 return x
 */
template <class T>
void Matrix<T>::backward_substitution(T * rhs, T * result)
{
    for (int i = 0; i<this->rows; i++){
        result[i] = 0;
    }
    double s;
    for (int k = this->rows-1; k>-1; k--){
        s = 0.0;
        for (int j = k+1; j<this->rows; j++)
        {
            s += this->values[k*this->cols +j]*result[j];
        }
        result[k] = (rhs[k]-s)/this->values[k*this->cols+k];
    }
}

/*
 def forward_substitution(A, b):
 n = np.size(b)
 x = np.zeros(n)
 for k in range(n):
     s = 0.
     for j in range(k):
         s = s + A[k, j]*x[j]
     x[k] = (b[k] - s)/A[k, k]
 
 return x
 */
template <class T>
void Matrix<T>::forward_substitution(T * rhs, T * result)
{
    for (int i = 0; i<this->rows; i++){
        result[i] = 0;
    }
    
    double s;
    for (int k = 0; k<this->rows; k++){
        s = 0.0;
        for (int j = 0; j<k; j++)
        {
            s += this->values[k*this->cols +j]*result[j];
        }
        result[k] = (rhs[k]-s)/this->values[k*this->cols+k];
    }
}

/*
 def LU_solve(A, b):
 """ An LU solve function that makes use of our
 non partial pivoting versions of LU_decomposition
 followed by forward_substitution and
 backward_substitution
 """
 L, U = LU_decomposition(A)
 y = forward_substitution(L, b)
 x = backward_substitution(U, y)
 return x
 */
template <class T>
void Matrix<T>::LU_solve(T* rhs, T* result)
{
    //LU_decomposition(T & lmat, T & umat)
    auto *lmat = new T[this->size_of_values];
    auto *umat = new T[this->size_of_values];
    
    auto *for_result = new T[this->rows];

    this->LU_decomposition(lmat, umat);

    auto *lmatrix = new Matrix<T>(this->rows, this->cols, lmat);
    auto *umatrix = new Matrix<T>(this->rows, this->cols, lmat);
    lmatrix->forward_substitution(rhs,for_result);
    umatrix->backward_substitution(for_result, result);

    delete[] lmat;
    delete[] umat;
    delete[] for_result;
    delete lmatrix;
    delete umatrix;

}

template <class T>
void Matrix<T>::find_unique(std::vector<bool> check_list, std::vector<int> &unique_list)
{
    int row_index = 0;
    int count = 0;
    for (int i = 0; i < this->cols; i++)
    {
        if (check_list[i] == false) continue;
        
        for (int j = 0; j < this->rows; j++)
        {
            if (this->values[i + j*this->cols] != 0)
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
void Matrix<T>::sort_mat(){
    auto *temp_mat = new Matrix<double>(this->rows, this->cols, true);
    
    std::vector <bool> check_list (this->cols,true);
//    check_list[1] = false;
//    check_list[0] = false;
//    check_list[2] = 0;
//    check_list[3] = false;
    while (!(std::none_of(check_list.begin(), check_list.end(), [](bool v) { return v; })))
    {
//        std::cout<<"some are still inside";
        std::vector <int> unique_list (this->cols,-1);
        
        //update unique_list with hanchao's function
        this->find_unique(check_list, unique_list);
        //if column j has a unique entry on row i (equals to "unique_list[j]")
        //then in temp_mat, set row j equals to (row i in original matrix)
        //so that in temp_mat, the entry on [i,j] is the unique one;
        //set this column j as false in the while loop to be excluded
        for (int j=0; j<this->cols; j++)
        {
            if (unique_list[j]!=-1)
            {
                for (int col =0; col<this->cols; col++)
                {
                    temp_mat->values[j*this->cols+col]= this->values[unique_list[j]*this->cols+col];
                    this->values[unique_list[j]*this->cols+col] =0;
                }
                check_list[j]=false;
            }
        }
        
        //next, fill the 1st available column with max value,
        //and remove it from check_list;
        //remember to delete
        
        
        for (int j=0; j<this->cols; j++)
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
//                    std::cout<<"assignment in progress with max: "<< max_value<<std::endl;
//                    std::cout<<"assignment in progress with value: "<< this->values[row*this->rows+j]<<std::endl;
                    if (abs(this->values[row*this->rows+j])>abs(max_value))
                    {
//                        std::cout<<"assigning: "<< j<<std::endl;
                        index_row = row;
                        max_value = this->values[row*this->rows+j];
                    }
                }
                // now index_row takes the index of row???
                // fill and exclude
                if (index_row != -1){
                    for (int kk = 0; kk < this->cols;kk++)
                    {
                        temp_mat->values[j*this->cols+kk]= this->values[index_row*this->cols+kk];
                        this->values[index_row*this->cols+kk] =0;
                    }
                    check_list[j]=false;
                }
                
                if (index_row ==-1)
                {
                    std::cout<<std::endl<<"index cannot be found here: "<< j<<std::endl;
                    
                }
//                std::cout<<"random assignment finished: "<< j<<std::endl;
            }
            break;
        }
        
    }
    
    for (int i=0; i<this->size_of_values; i++)
    {
        this->values[i] = temp_mat->values[i];
    }
    delete temp_mat;
    
}
