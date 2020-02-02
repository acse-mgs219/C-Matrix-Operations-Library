# Readme



## Installation Instructions

**Requirements:**

-  In order to run the optimized versions of the algorithms with BLAS calls, you must have a BLAS library installed. The software used development team used Open BLAS on a Mac to develop the software. If you do not have BLAS, you can remove the compiler flag "USE_BLAS" to still use the library.

**Building the Project**

* By far the easiest way to build the project is to build using the CMake file included in the root directory. The file that is provided is written for a Mac OS, but can easily be adapted for any other operating system. You can also open the folder using Visual Studio and simply build and run the main.

**Data Files**

* The project makes extensive use of txt files that contain different matrix structures and vectors that combined produce linear systems of the form **Ax = b**. If using Visual Studio, the generated exe will always be able to read the files. If compiling manually, please make sure to generate the exe in the same director as main.cpp so that it can read the files.


### Performance test results

* We have run several performance tests, testing the performance of all the solvers for different size matrices, with and without BLAS and with and without Compiler Optimizations. The raw results of those tests were written to files that were saved under PerformanceTestResults.

### Matrix Operations

For all of the matrix functions, a vector is defined as a **m x 1** column matrix.

#### Dense matrices

```c++
void matMatMult(Matrix<T>& mat_right, Matrix<T>& output);
```

This function computes matrix product between A and B.

```c++
void matVectMult(Matrix<T> &b)
```

This function computes matrix-vector product between A and v.

```c++
void makeRandomDD()
```

This function generates a random diagonally donminant matrix.

```c++
void setMatrix(int length, T *values_ptr)
```

This function sets the matrix by using the input pointer array.

```c++
void writeMatrix(std::string fileName)
```

This function writes the contents of the matrix to a text file.

```c++
void printValues()
```

This function print all the values of the function in 1D array format.

```c++
void printMatrix()
```

This function print all the values of the function in matrix format. if the matrix is too large (more than 20 cols or rows) it prints only the first and last few entries to not flood the output.

```c++
void swapRows(Matrix<T> *b, int i, int j)
```

This function swaps two certain rows of the input matrix.

```c++
void swapRowsMatrix(int i, int j)
```

This function swaps two certain rows of the input matrix.

```c++
void transpose()
```

This function transposes a matrix in place.

```c++
void getValue(int row_index, int col_index)
```

This function returns A[i][j] for a matrix A

```c++
void innerVectorProduct(Matrix<T> &mat_right)
```

This function computes the inner vector product of matrices.

#### Sparse matrices

```c++
void printMatrix()
```

This function print the matrix in sparse fotmat, including three arrays: values, column index and row position.

```c++
Matrix<T>* matVecMult(Matrix<T>& b)
```

This function compute matix-vector product in sparse format.

```c++
CSRMatrix<T>* matMatMult(CSRMatrix<T>& mat_right)
```

This function compute matix-matrix product in sparse format.

```c++
CSRMatrix<T>* matMatMultSymbolic(CSRMatrix<T>& mat_right, std::vector< std::pair< std::pair<int, int>, T> >& result)
```

This function compute matix-matrix symbolic product in sparse format. It finds the locations of non-zeros.

```c++
void matMatMultNumeric(CSRMatrix<T>* symbolic_res, std::vector< std::pair< std::pair<int, int>, T> >& result)
```

This function compute matix-matrix numerical product in sparse format. It takes the output of matMatMultSymbolic and finds the value of non-zeros

```c++
void setMatrix(T* values_ptr, int iA[], int jA[])
```

This function sets the matrix by using three input pointer array: values, column index and row position.

### Solvers

#### Dense matrices

```c++
Matrix<T>* solveJacobi(Matrix<T>* LHS, Matrix<T>* b, double tolerance, int max_iterations, T initial_guess[])
```

This is an implementation of a Jacobi iterative solver on a dense format matrix.

Takes in a matrix LHS, a right hand side vector b, a tolerance which determines convergence criteria, and a maximum number of iterations. The user can provide a initial guess to start the solver.

It returns the solution of x of matrix class.

```c++
Matrix<T>* solveGaussSeidel(Matrix<T>* LHS, Matrix<T>* b, double tolerance, int max_iterations, T* initial_guess)
```

This is an implementation of a Gauss Seidel iterative solver on a dense format matrix.

Takes in a matrix LHS, a right hand side vector b, a tolerance which determines convergence criteria, and a maximum number of iterations. The user can provide a initial guess to start the solver.

It returns the solution of x of matrix class.

```C++
Matrix<T>* solveGaussian(Matrix<T>* LHS, Matrix<T>* b)
```

This is an implementation of a Gaussian elimination direct solver on a dense format matrix.

Takes in a matrix LHS, a right hand side vector b. It is a direct method which can be used for every linear system.

It returns the solution of x of matrix class.

```C++
Matrix<T>* solveLU(Matrix<T>* LHS, Matrix<T>* b)
```

This is an implementation of a LU direct solver on a dense format matrix.

Takes in a matrix LHS, a right hand side vector b. It is a direct method which can be used for every linear system.

It returns the solution of x of matrix class.

```C++
Matrix<T>* solveConjugateGradient(Matrix<T>* LHS, Matrix<T>* b, double epsilon, int max_iterations, T initial_guess[])
```

This function implements the conjugate gradient algorithm for a dense matrix. It returns the estimated solution vector.

```c++
Matrix<T>* solveConjugateGradient(Matrix<T>* LHS, Matrix<T>* b, double epsilon, int max_iterations, T initial_guess[])
```

This implements the conjugate gradient algorithm over a CSR format matrix. It returns a vector.

```c++
void incompleteCholesky(Matrix<T> *matrix)
```

This is a function that preconditions the matrices for iterative solvers in order to increase the rate of convergence. The function implements the incomplete Cholesky factorization which generates a lower triangular matrix.

#### Sparse matrices

```c++
Matrix<T>* solveJacobi(CSRMatrix<T>* LHS, Matrix<T>* mat_b)
```

This is an implementation of a Jacobi iterative solver on a CSR format matrix.

Takes in a matrix LHS, a right hand side vector b. The tolerance which determines convergence criteria can be defined in ` bool check_finish(T* mat_b, T* output)`.

It returns the solution of x of matrix class.

```c++
Matrix<T>* solveGaussSeidel(CSRMatrix<T>* LHS, Matrix<T>* mat_b)
```

This is an implementation of a Gauss Seidel iterative solver on a CSR format matrix.

Takes in a matrix LHS, a right hand side vector b. The tolerance which determines convergence criteria can be defined in ` bool check_finish(T* mat_b, T* output)`.

It returns the solution of x of matrix class.

```c++
Matrix<T>* conjugateGradient(CSRMatrix<T>* LHS, Matrix<T>* b, double epsilon, int max_iterations, T initial_guess[])
```





### Function under development

```c++
sort_mat(Matrix<T>* rhs)
```

This function changes the input matrix by using row elementary operations and get an output matrix whose diagonal elements are all non-zero (We can always do that since the input matrices are positive definite). 

It is included in several solvers (Jacobi, Gauss Seidel). Users need to set the parameter `bool sortmatrix` as `true` or `false` when they call the solvers. `true` means the input matrix will be adjusted and `false` means no adjustment.

The users should also be aware of one defect of this function that it may change a diagonally dominant matrix to a non-diagonally dominant one. 









