#include "Matrix.h"
#include "Solver.h"

template <class T>
Matrix<T>* Solver<T>::solveJacobi(Matrix<T>* LHS, Matrix<T>* b, double tolerance, int max_iterations, T initial_guess[])
{
    // create some space to hold the solution to the iteration
    auto x_var = new Matrix<T>(b->rows, b->cols, true);

    // create a vector to hold the previous values of the x values we iterate over
    std::unique_ptr< Matrix<T> > x_var_prev(new Matrix<T>(b->rows, b->cols, true));

    // set the initial x values to our initial guess
    x_var_prev->setMatrix(b->size(), initial_guess);

    // create a vector to hold the estimated right hand side - smart pointer as only used inside this scope
    std::unique_ptr< Matrix<T> > estimated_rhs(LHS->matMatMult(*x_var_prev));

    // initialize residual which will be used to determine ending position
    double residual = tolerance * 2;

    // create a variable to hold the sum of row into columns
    std::unique_ptr<double[]> sum(new double[LHS->cols]);

    // iteration counter to count how many iterations we have completed. Used to keep iterations below max_iterations
    int iteration = 0;

    // iterate until we hit the convergence criteria or max iterations
    while (residual > tolerance&& iteration < max_iterations)
    {
        // loop over each row of the left hand side, each column of b, and calculate the sum
        for (int i = 0; i < LHS->rows; i++)
        {
            sum[i] = 0;
            for (int j = 0; j < LHS->cols; j++)
            {
                if (i != j)
                {
                    sum[i] += LHS->values[i * LHS->cols + j] * x_var_prev->values[j];
                }
            }
        }

        // update the values of the current x and the previous x
        for (int i = 0; i < LHS->rows; i++)
        {
            x_var->values[i] = 1 / LHS->values[i * LHS->rows + i] * (b->values[i] - sum[i]);
            x_var_prev->values[i] = x_var->values[i];
        }

        // reset the residual so we can calculate for the current iterative step only
        residual = 0;

        // check residual against the tolerance
        for (int i = 0; i < b->size(); i++)
        {
            residual += pow(fabs(estimated_rhs->values[i] - b->values[i]), 2);
        }

        // calculate the RMSE norm of the residuals
        residual = sqrt(residual / b->size());
        ++iteration;
    }

    return x_var;
}

template<class T>
Matrix<T>* Solver<T>::solveGaussSeidel(Matrix<T>* LHS, Matrix<T>* b, double tolerance, int max_iterations, T* initial_guess) {

    // create some space to hold the solution to the iteration
    auto x_var = new Matrix<T>(b->rows, b->cols, true);

    // set the first x value to the initial guess
    x_var->setMatrix(b->rows, initial_guess);

    // create a vector to hold the estimated RHS for each iteration
    std::unique_ptr< Matrix<T> > estimated_rhs(LHS->matMatMult(*x_var));

    // initialize residual which will be used to determine ending position
    double residual = tolerance * 2;
    double sum;
    int iteration = 0;

    // iterate until we hit the convergence criteria or max_iterations
    while (residual > tolerance&& iteration < max_iterations)
    {
        // loop over each row in the LHS matrix and multiply into the column of b
        for (int i = 0; i < LHS->rows; i++)
        {
            sum = 0;

            for (int j = 0; j < b->size(); j++)
            {
                // because we are essentially rearranging the equation, only calculate the sum if not the same variable
                if (i != j)
                {
                    sum += LHS->values[i * LHS->cols + j] * x_var->values[j];
                }
            }

            // update the x variable values
            x_var->values[i] = 1 / LHS->values[i * LHS->cols + i] * (b->values[i] - sum);
        }

        // reset the residual to 0
        residual = 0;

        // check residual to see if we have hit the convergence criteria
        for (int i = 0; i < b->size(); i++)
        {
            residual += pow(fabs(estimated_rhs->values[i] - b->values[i]), 2);
        }

        // calculate RMSE to check for convergence condition
        residual = sqrt(residual / b->size());
        ++iteration;
    }

    return x_var;
}

// function that implements gaussian elimination
template<class T>
Matrix<T>* Solver<T>::solveGaussian(Matrix<T>* LHS, Matrix<T>* b)
{
    // transform matrices to upper triangular
    Solver<T>::upperTriangular(LHS, b);

    // generate solution
    auto *solution = Solver<T>::backSubstitution(LHS, b);

    return solution;
}

template<class T>
Matrix<T>* Solver<T>::solveLU(Matrix<T>* LHS, Matrix<T>* b) {

    // create space to hold the upper triangular, lower triangular and permutation
    auto upper_tri = new Matrix<T>(LHS->rows, LHS->cols, true);
    auto lower_tri = new Matrix<T>(LHS->rows, LHS->cols, true);
    auto permutation = new Matrix<T>(LHS->rows, LHS->cols, true);

    // construct LU decomposition of the LHS matrix - this gives us the permutation
    Solver<T>::luDecompositionPivot(LHS, upper_tri, lower_tri, permutation);

    // transpose the permutation matrix
    permutation->transpose();

    // multiply the transpose of the permutation matrix by b
    auto p_inv_b = permutation->matMatMult(*b);

    // calculate the y values using forward substitution
    auto y_values = Solver<T>::forwardSubstitution(lower_tri, p_inv_b);

    // calculate the solution using back substitution and the y values we calculated earlier
    auto *solution = Solver<T>::backSubstitution(upper_tri, y_values);

    // clean memory
    delete upper_tri;
    delete lower_tri;
    delete permutation;
    delete p_inv_b;
    delete y_values;

    return solution;
}

// solve Ax = b;
template<class T>
Matrix<T>* Solver<T>::conjugateGradient(Matrix<T>* LHS, Matrix<T>* b, double epsilon, int max_iterations, T initial_guess[])
{
    double residual = 2 * epsilon;

    // intialize x values to initial guess
    auto x_prev = new Matrix<T>(b->rows, b->cols, true); // memory cleared
    x_prev->setMatrix(b->rows, initial_guess);

    // 1. set the residual to  r_0 = b - Ax_0 initially
    std::unique_ptr< Matrix<T> > Ax(LHS->matVectMult(*x_prev));
    auto r_prev = new Matrix<T>(b->rows, b->cols, true); // memory cleared
    for (int i = 0; i < r_prev->size(); i++)
    {
        r_prev->values[i] = b->values[i] - Ax->values[i];
    }

    // create the preconditioner - copy the values of LHS in and then do an incomplete Cholesky factorization
    auto M = new Matrix<T>(LHS->rows, LHS->cols, true); // memory cleared
    for(int i=0; i<M->size(); i++)
    {
        M->values[i] = LHS->values[i];
    };

    // use an incomplete Cholesky factorization
    Solver<T>::incompleteCholesky(M);
    M->printMatrix();

    // 2. Solve M z_0 = r_0
    auto z_prev = Solver<T>::forwardSubstitution(M, r_prev); // memory cleared
    z_prev->printMatrix();

    // 3. set p_0 = z_0
    auto p = new Matrix<T>(z_prev->rows, z_prev->cols, true); // memory cleared
    for (int i = 0; i < z_prev->size(); i++)
    {
        p->values[i] = z_prev->values[i];
    }

    // 4. w =  A p_1
    auto w = LHS->matVectMult(*p); // memory cleared

    // 5. alpha = r_0.T z_0 /  (p_0.T w)
    double alpha = r_prev->innerVectorProduct(*z_prev) /  p->innerVectorProduct(*w);

    // 6. x_1 = x_0 + alpha * p_1
    auto x = new Matrix<T>(b->rows, b->cols, true);

    // 7. r_1 = r_0 - alpha * w
    auto r = new Matrix<T>(b->rows, b->cols, true); //  memory cleared

    for (int i=0; i<x->size(); i++)
    {
        x->values[i] = x_prev->values[i] + alpha  * p->values[i];
        r->values[i] = r_prev->values[i] - alpha * w->values[i];
    }

    // set iteration to 1
    int iteration = 1;
    double beta;

    while (residual > epsilon && iteration < max_iterations)
    {
        // 8. solve M z_k = r_k
        auto z = Solver<T>::forwardSubstitution(M, r); // memory cleared inside loop
        //z_prev->printMatrix();

        beta = r->innerVectorProduct(*z) / r_prev->innerVectorProduct(*z_prev);
        //std::cout << r->innerVectorProduct(*z) << " ";

        for (int i=0; i<p->size(); i++)
        {
            p->values[i] = z->values[i] + beta * p->values[i];
        }

        delete w;

        w = LHS->matVectMult(*p);

        alpha = r->innerVectorProduct(*z) / p->innerVectorProduct(*w);
        //std::cout << r->innerVectorProduct(*z) << " ";

        for (int i=0; i<x->size(); i++)
        {
            x_prev->values[i] = x->values[i];
            x->values[i] = x_prev->values[i] + alpha  * p->values[i];
            r_prev->values[i] = r->values[i];
            r->values[i] = r_prev->values[i] - alpha * w->values[i];
            z_prev->values[i] = z->values[i];
        }
        delete z;
        ++iteration;
    }

    delete x_prev;
    delete r_prev;
    delete M;
    delete z_prev;
    delete p;
    delete w;
    delete r;

    return x;
}

template<class T>
void Solver<T>::luDecomposition(Matrix<T>* LHS, Matrix<T>* upper_tri, Matrix<T>* lower_tri)
{
    // make sure the matrix is square
    if (LHS->cols != LHS->rows)
    {
        throw std::invalid_argument("input has wrong number dimensions");
    }

    // intialize the scaling factor s to -1
    double s = -1;

    // copy the values of A into the upper triangular matrix
    for (int i = 0; i < LHS->size(); i++)
    {
        upper_tri->values[i] = LHS->values[i];
    }

    // loop over each pivot row
    for (int k = 0; k < LHS->rows - 1; k++)
    {
        // loop over each equation below the pivot
        for (int i = k + 1; i < LHS->rows; i++)
        {
            // assumes row major order
            s = upper_tri->values[i * LHS->rows + k] / upper_tri->values[k * upper_tri->rows + k];

            for (int j = k; j < LHS->rows; j++)
            {
                upper_tri->values[i * LHS->rows + j] -= s * upper_tri->values[k * upper_tri->rows + j];
            }

            lower_tri->values[i * LHS->rows + k] = s;
        }
    }

    // add zeroes to the diagonal
    for (int i = 0; i < LHS->rows; i++)
    {
        lower_tri->values[i * lower_tri->rows + i] = 1;
    }
}


template<class T>
void Solver<T>::luDecompositionPivot(Matrix<T>* LHS, Matrix<T>* upper_tri, Matrix<T>* lower_tri, Matrix<T>* permutation)
{
    // make sure the matrix is square
    if (LHS->cols != LHS->rows)
    {
        throw std::invalid_argument("input has wrong number dimensions");
    }

    // variables to keep track of the index of the maximum value and the maximum value itself
    int max_index = -1;
    int max_val = -1;

    // initialize the scaling factor to -1
    double s = -1;

    // copy the values of A into upper triangular matrix
    for (int i = 0; i < upper_tri->size(); i++)
    {
        upper_tri->values[i] = LHS->values[i];
    }

    // make permuation matrix an idenity matrix
    for (int i = 0; i < permutation->rows; i++)
    {
        permutation->values[i * permutation->cols + i] = 1;
    }

    // loop over each pivot row
    for (int k = 0; k < upper_tri->rows - 1; k++)
    {
        max_val = -1;
        max_index = k;

        // find the index of the largest value in the column
        for (int z = k; z < upper_tri->rows; z++)
        {
            if (fabs(upper_tri->values[z * upper_tri->cols + k]) > max_val)
            {
                max_val = fabs(upper_tri->values[z * upper_tri->cols + k]);
                max_index = z;
            }
        }

        // generate the upper triangular, lower triangular and permutation matrix decomposition
        upper_tri->swapRowsMatrix(k, max_index);
        lower_tri->swapRowsMatrix(k, max_index);
        permutation->swapRowsMatrix(k, max_index);

        // loop over each equation below the pivot
        for (int i = k + 1; i < upper_tri->rows; i++)
        {
            // assumes row major order
            s = upper_tri->values[i * upper_tri->cols + k] / upper_tri->values[k * upper_tri->cols + k];

            // update the upper tri values using the scaling factor
            for (int j = k; j < upper_tri->cols; j++)
            {
                upper_tri->values[i * upper_tri->cols + j] -= s * upper_tri->values[k * upper_tri->cols + j];
            }

            // update the lower tri values
            lower_tri->values[i * lower_tri->rows + k] = s;
        }
    }

    // add zeroes to the diagonal
    for (int i = 0; i < upper_tri->rows; i++)
    {
        lower_tri->values[i * lower_tri->rows + i] = 1;
    }

    // return the transpose of the permutation matrix
    permutation->transpose();
}

template <class T>
void Solver<T>::upperTriangular(Matrix<T>* LHS, Matrix<T>* b)
{
    // check if A is square
    if (LHS->rows != LHS->cols)
    {
        throw std::invalid_argument("A should be a square matrix!");
    }

    // check that the dimensions of A and b are compatible
    if (LHS->rows != b->size())
    {
        throw std::invalid_argument("A and b dimensions dont match");
    }

    // s is the scaling factor to adjust a row. kmax keeeps track of the index of the maximum value; need for pivoting
    double s = -1;
    int kmax = -1;

    // loop over each pivot row except the last one
    for (int k = 0; k < LHS->rows - 1; k++)
    {
        // initialize with current pivot row
        kmax = k;

        // find pivot column to avoid zeros on diagonal
        for (int i = k + 1; i < LHS->rows; i++)
        {
            if (fabs(LHS->values[kmax * LHS->cols + k]) < fabs(LHS->values[i * LHS->cols + k]))
            {
                kmax = i;
            }
        }

        // swap the rows if we have found a bigger value in the column below the pivot
        LHS->swapRows(b, kmax, k);

        // loop over each row below the pivot
        for (int i = k + 1; i < LHS->rows; i++)
        {
            // calculate scaling value for LHS row
            s = LHS->values[i * LHS->cols + k] / LHS->values[k * LHS->cols + k];

            // start looping from k and update the row
            for (int j = k; j < LHS->rows; j++)
            {
                LHS->values[i * LHS->cols + j] -= s * LHS->values[k * LHS->cols + j];
            }

            // update corresponding entry of b
            b->values[i] -= s * b->values[k];
        }
    }
}

template <class T>
Matrix<T>* Solver<T>::backSubstitution(Matrix<T>* LHS, Matrix<T>* b)
{
    // check if A is square
    if (LHS->rows != LHS->cols)
    {
        throw std::invalid_argument("A should be a square matrix!");
    }

    // check that the dimensions of A and b are compatible
    if (LHS->rows != b->size())
    {
        throw std::invalid_argument("A and b dimensions don't match");
    }

    // create an empty vector
    Matrix<T>* solution = new Matrix<T>(b->rows, b->cols, true);

    double s;

    // iterate over system backwards
    for (int k = b->size() - 1; k >= 0; k--)
    {
        s = 0;

        for (int j = k + 1; j < b->size(); j++)
        {
            // assumes row major order
            s += LHS->values[k * LHS->cols + j] * solution->values[j];
        }

        // adjust the values in the solution vector
        solution->values[k] = (b->values[k] - s) / LHS->values[k * LHS->cols + k];
    }

    return solution;
}

template<class T>
Matrix<T>* Solver<T>::forwardSubstitution(Matrix<T>* LHS, Matrix<T>* b)
{
    // check if A is square
    if (LHS->rows != LHS->cols)
    {
        throw std::invalid_argument("A should be a square matrix!");
    }

    // check that the dimensions of A and b are compatible
    if (LHS->rows != b->size())
    {
        throw std::invalid_argument("A and b dimensions don't match");
    }

    // create an empty vector
    Matrix<T>* solution = new Matrix<T>(b->rows, b->cols, true);

    double s;

    // iterate over system
    for (int k = 0; k < b->size(); k++)
    {
        s = 0;

        for (int j = 0; j < k; j++)
        {
            // assumes row major order
            s = s + LHS->values[k * LHS->cols + j] * solution->values[j];
        }

        // adjust the values in the solution vector
        solution->values[k] = (b->values[k] - s) / LHS->values[k * LHS->cols + k];
    }

    return solution;
}

template<class T>
void Solver<T>::incompleteCholesky(Matrix<T> *matrix)
{
    T* matrixCop = new T[matrix->size()];
    for (int k=0; k<matrix->rows; k++)
    {
        // a_kk = sqrt(a_kk)
        matrixCop[k * matrix->cols + k] = sqrt(matrix->values[k * matrix->cols + k]);

        for (int i=k+1; i<matrix->rows; i++)
        {
            if (matrix->values[i * matrix->cols + k] != 0)
            {
                matrixCop[i * matrix->cols + k] =  matrix->values[i * matrix->cols + k] / matrix->values[k * matrix->cols + k];
            }
        }

        for (int j=k+1; j<matrix->rows; j++)
        {
            for (int i=j; i<matrix->cols; i++)
            {
                if (matrix->values[i * matrix->cols + j] != 0)
                    matrixCop[i * matrix->cols + j] = matrix->values[i * matrix->cols + j] - matrix->values[i * matrix->cols + k]*matrix->values[j * matrix->cols + k];
            }
        }
    }

    for (int i=0; i<matrix->rows; i++)
    {
        for (int j=i+1; j<matrix->cols; j++)
        {
            matrixCop[i * matrix->cols + j] = 0;
        }
    }

    matrix->values = matrixCop;
}


template<class T>
Matrix<T>* Solver<T>::conjugateGradient(CSRMatrix<T>* LHS, Matrix<T>& b, double epsilon, int max_iterations)
{
    int k = 0;
    T beta = 1;
    double alpha = 1;
    T delta_old = 1;

    // intialize to x to 0
    auto x = new Matrix<T>(b.rows, b.cols, true);

    // workout Ax
    auto Ax = LHS->matVecMult(*x);

    // r = b - Ax
    auto r = new Matrix<T>(b.rows, b.cols, true);

    for (int i = 0; i < r->rows * r->cols; i++)
    {
        r->values[i] = b.values[i] - Ax->values[i];
    }

    auto p = new Matrix<T>(r->rows, r->cols, true);
    auto w = new Matrix<T>(r->rows, r->cols, true);

    double delta = r->innerVectorProduct(*r);

    while (k < max_iterations && (sqrt(delta) > epsilon* sqrt(b.innerVectorProduct(b))))
    {
        if (k == 1)
        {
            // p = r
            for (int i = 0; i < p->rows * p->cols; i++)
            {
                p->values[i] = r->values[i];
            }

        }
        else {

            beta = delta / delta_old;

            // p = r + beta * p
            for (int i = 0; i < p->rows * p->cols; i++)
            {
                p->values[i] = r->values[i] + beta * p->values[i];
            }
        }

        auto Ap = LHS->matVecMult(*p);

        // w = Ap
        for (int i = 0; i < w->rows * w->cols; i++)
        {
            w->values[i] = Ap->values[i];
        }

        alpha = delta / p->innerVectorProduct(*w);

        // x = x + alpha * p
        for (int i = 0; i < x->rows * x->cols; i++)
        {
            x->values[i] = x->values[i] + alpha * p->values[i];
        }

        // r = r - alpha * w
        for (int i = 0; i < r->rows * r->cols; i++)
        {
            r->values[i] = r->values[i] - alpha * w->values[i];
        }

        delta_old = delta;

        delta = r->innerVectorProduct(*r);

        delete Ap;
        k++;
    }

    delete Ax;
    delete r;
    delete p;
    delete w;

    return x;
}