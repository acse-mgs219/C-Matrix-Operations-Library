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

    // create a vector to hold the estimated right hand side - smart pointer as only used inside LHS scope
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

    // construct LU decomposition of the LHS matrix - LHS gives us the permutation
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
    // variable to keep track of the iterations
    int iteration = 0;

    // algorithm specific variables - initialize all of them to 1
    double beta = 1;
    double alpha = 1;
    double delta_old = 1;

    // intialize x values to 0
    auto x = new Matrix<T>(b->rows, b->cols, true);
    x->setMatrix(b->rows, initial_guess);

    // workout Ax initially
    std::unique_ptr< Matrix<T> > Ax(LHS->matMatMult(*x));

    // set the residual to  r = b - Ax initially
    std::unique_ptr< Matrix<T> > r(new Matrix<T>(b->rows, b->cols, true));

    // set r = b - Ax for this iteration
    for (int i = 0; i < r->size(); i++)
    {
        r->values[i] = b->values[i] - Ax->values[i];
    }

    // create some space for p and w matrices used in the iterations
    std::unique_ptr< Matrix<T> > p(new Matrix<T>(r->rows, r->cols, true));
    std::unique_ptr< Matrix<T> > w(new Matrix<T>(r->rows, r->cols, true));

    // calculate delta parameter which is the result of the inner product of r with itself
    double delta = r->innerVectorProduct(*r);

    // iterate until the convergence criteria is reached or we hit the max iterations
    while (iteration < max_iterations && (sqrt(delta) > epsilon* sqrt(b->innerVectorProduct(*b))))
    {
        // for the first iterations set p = r
        if (iteration == 0)
        {
            for (int i = 1; i < p->size(); i++)
            {
                p->values[i] = r->values[i];
            }

        }
            // for all the other iterations
        else {
            // update beta based on the delta values
            beta = delta / delta_old;

            // set p = r + beta * p for this iteration
            for (int i = 0; i < p->size(); i++)
            {
                p->values[i] = r->values[i] + beta * p->values[i];
            }
        }

        std::unique_ptr< Matrix<T> > Ap(LHS->matMatMult(*p));

        // set w = Ap for this iteration
        for (int i = 0; i < w->size(); i++)
        {
            w->values[i] = Ap->values[i];
        }

        // update the alpha value based on delta and the inner product of p and w
        alpha = delta / p->innerVectorProduct(*w);

        // set x = x + alpha * p for this iteration
        for (int i = 0; i < x->size(); i++)
        {
            x->values[i] = x->values[i] + alpha * p->values[i];
        }

        // set r = r - alpha * w for this iteration
        for (int i = 0; i < r->size(); i++)
        {
            r->values[i] = r->values[i] - alpha * w->values[i];
        }

        // update both delta and delta_old for this iteration
        delta_old = delta;
        delta = r->innerVectorProduct(*r);

        ++iteration;
    }

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
    for (int k=0; k<matrix->rows; k++)
    {
        // if first column in row is negative, simply take its absolute value
        if (matrix->values[k * matrix->cols + k] < 0 && k == 0)
        {
            matrix->values[k * matrix->cols + k] = fabs(matrix->values[k * matrix->cols + k]);
        }
        // otherwise sum up all the other values in the row
        else if (matrix->values[k * matrix->cols + k] < 0)
        {
            double sum = 0;
            for(int c=k; c>=0; c--)
            {
                sum += fabs(matrix->values[k * matrix->cols + c]);
            }

            matrix->values[k * matrix->cols + k] = sum;
        }
        else {

            matrix->values[k * matrix->cols + k] = sqrt(matrix->values[k * matrix->cols + k]);
        }


        for (int i=k+1; i<matrix->rows; i++)
        {
            if (matrix->values[i * matrix->cols + k] != 0)
            {
                matrix->values[i * matrix->cols + k] =  matrix->values[i * matrix->cols + k] / matrix->values[k * matrix->cols + k];
            }
        }

        for (int j=k+1; j<matrix->rows; j++)
        {
            for (int i=j; i<matrix->cols; i++)
            {
                if (matrix->values[i * matrix->cols + j] != 0)
                    matrix->values[i * matrix->cols + j] = matrix->values[i * matrix->cols + j]
                            - matrix->values[i * matrix->cols + k]*matrix->values[j * matrix->cols + k];
            }
        }
    }

    for (int i=0; i<matrix->rows; i++)
    {
        for (int j=i+1; j<matrix->cols; j++)
        {
            matrix->values[i * matrix->cols + j] = 0;
        }
    }
}


template<class T>
Matrix<T>* Solver<T>::conjugateGradient(CSRMatrix<T>* LHS, Matrix<T>* b, double epsilon, int max_iterations, T initial_guess[])
{
    // variable to keep track of the iterations
    int iteration = 0;

    // algorithm specific variables - initialize all of them to 1
    double beta = 1;
    double alpha = 1;
    double delta_old = 1;

    // intialize to x to 0
    auto x = new Matrix<T>(b->rows, b->cols, true);
    x->setMatrix(b->rows, initial_guess);

    // workout Ax initially
    std::unique_ptr< Matrix<T> > Ax(LHS->matVectMult(*x));

    // r = b - Ax
    auto r = new Matrix<T>(b->rows, b->cols, true);

    for (int i = 0; i < r->size(); i++)
    {
        r->values[i] = b->values[i] - Ax->values[i];
    }

    auto p = new Matrix<T>(r->rows, r->cols, true);
    auto w = new Matrix<T>(r->rows, r->cols, true);

    double delta = r->innerVectorProduct(*r);

    while (iteration < max_iterations && (sqrt(delta) > epsilon* sqrt(b->innerVectorProduct(*b))))
    {
        if (iteration == 0)
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
        iteration++;
    }

    delete r;
    delete p;
    delete w;

    return x;
}

template <class T>
bool Solver<T>::check_finish(CSRMatrix<T>* LHS, Matrix<T>* mat_b, Matrix<T>* output)
{
    T tol = 0.00001;
    T* cal_out = new T[LHS->rows];
    T res = 0;

    for (int i = 0; i < LHS->rows; i++)
    {
        cal_out[i] = 0;
        for (int val_index = LHS->row_position[i]; val_index < LHS->row_position[i + 1]; val_index++)
        {
            cal_out[i] += LHS->values[val_index] * output->values[LHS->col_index[val_index]];
        }
        res += abs(mat_b->values[i] - cal_out[i]);
    }

    delete[] cal_out;
    if ((res / LHS->rows) < tol)
        return true;
    else
        return false;
}


template <class T>
Matrix<T>* Solver<T>::solveGaussSeidel(CSRMatrix<T>* LHS, Matrix<T>* mat_b) {
    Matrix<T>* output = new Matrix<T>(LHS->cols, 1, true);
    for (int i = 0; i < LHS->cols; i++)
    {
        output->values[i] = 0.0;
    }

    T temp;
    T* diag_ele = new T[LHS->cols];

    int it_max = 10;

    //find diagonal elements
    for (int i = 0; i < LHS->rows; i++)
    {
        for (int val_index = LHS->row_position[i]; val_index < LHS->row_position[i + 1]; val_index++)
        {
            if (i == LHS->col_index[val_index])
            {
                diag_ele[i] = LHS->values[val_index];
            }
        }
    }


    for (int k = 0; k < it_max; k++)//count for iterations, k times maximium
    {
        for (int i = 0; i < LHS->rows; i++) {
            temp = mat_b->values[i];
            for (int val_index = LHS->row_position[i]; val_index < LHS->row_position[i + 1]; val_index++) {
                if (i == LHS->col_index[val_index]) continue;
                temp = temp - LHS->values[val_index] * output->values[LHS->col_index[val_index]];
            }
            output->values[i] = temp / diag_ele[i];
        }
        if (Solver<T>::check_finish(LHS, mat_b, output)) break;
    }

    delete[] diag_ele;
    return output;
}

template <class T>
Matrix<T>* Solver<T>::solveJacobi(CSRMatrix<T>* LHS, Matrix<T>* mat_b) {
    Matrix<T>* output = new Matrix<T>(LHS->cols, 1, true);
    T* temp = new T[LHS->rows];
    for (int i = 0; i < LHS->rows; i++)
    {
        output->values[i] = 0.0;
    }

    T* diagonal = new T[LHS->rows];

    int iteration = 10;

    //find diagonal elements
    for (int i = 0; i < LHS->rows; i++)
    {
        for (int val_index = LHS->row_position[i]; val_index < LHS->row_position[i + 1]; val_index++)
        {
            if (i == LHS->col_index[val_index])
            {
                diagonal[i] = LHS->values[val_index];
            }
        }
    }

    for (int k = 0; k < iteration; k++) //count for iterations, 10 times maximium
    {
        for (int i = 0; i < LHS->rows; i++)
        {
            temp[i] = mat_b->values[i];
            for (int val_index = LHS->row_position[i]; val_index < LHS->row_position[i + 1]; val_index++)
            {
                if (i == LHS->col_index[val_index]) continue;
                temp[i] = temp[i] - LHS->values[val_index] * output->values[LHS->col_index[val_index]];
            }
            temp[i] = temp[i] / diagonal[i];
        }
        for (int i = 0; i < LHS->rows; i++) output->values[i] = temp[i];
        if (Solver<T>::check_finish(LHS, mat_b, output)) break;
    }

    delete[] temp;

    delete[] diagonal;
    return output;
}
