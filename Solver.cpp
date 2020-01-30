#include "Matrix.h"
#include "Solver.h"

template <class T>
Matrix<T>* Solver<T>::solveJacobi(Matrix<T>* LHS, Matrix<T>* b, double tolerance, int max_iterations, T initial_guess[])
{
    //LHS->sort_mat(b);
    // create some space to hold the solution to the iteration
    auto x_var = new Matrix<T>(b->rows, b->cols, true);
    auto x_var_prev = new Matrix<T>(b->rows, b->cols, true); // is b->cols always 1?

    x_var_prev->setMatrix(b->size(), initial_guess); // should check that sizes are correct

    auto estimated_rhs = LHS->matMatMult(*x_var_prev);

    // initialize residual which will be used to determine ending position
    double residual = tolerance * 2;
    double resid_sum; // not actually necessary
    double* sum = new double[LHS->cols];
    int iteration = 0;

    while (residual > tolerance&& iteration < max_iterations)
    {
        for (int i = 0; i < LHS->rows; i++) // should be LHS->rows?
        {
            sum[i] = 0;

            for (int j = 0; j < LHS->cols; j++) // should be LHS->cols?
            {
               if (i != j)
               {
                    sum[i] += LHS->values[i * LHS->cols + j] * x_var_prev->values[j];
               }
            }
        }

       for (int i = 0; i < LHS->rows; i++) // should be LHS->rows?
       {
            x_var->values[i] = 1 / LHS->values[i * LHS->rows + i] * (b->values[i] - sum[i]);
            x_var_prev->values[i] = x_var->values[i];
       }

        resid_sum = 0;

        // check residual
        for (int i = 0; i < b->size(); i++)
        {
            resid_sum += fabs(estimated_rhs->values[i] - b->values[i]);
        }
        residual = resid_sum / b->size();
        ++iteration;
    }
    
    // clean memory
    delete x_var_prev;
    delete estimated_rhs;
    delete[] sum;

    return x_var;
}

template<class T>
Matrix<T>* Solver<T>::solveGaussSeidel(Matrix<T>* LHS, Matrix<T>* b, double tolerance, int max_iterations, T* initial_guess) {

    // create some space to hold the solution to the iteration
    auto x_var = new Matrix<T>(b->rows, b->cols, true);

    x_var->setMatrix(b->rows, initial_guess);

    auto estimated_rhs = LHS->matMatMult(*x_var);

    // initialize residual which will be used to determine ending position
    double residual = tolerance * 2;
    double resid_sum;
    double sum;
    int iteration = 0;

    while (residual > tolerance&& iteration < max_iterations)
    {
        for (int i = 0; i < b->size(); i++)
        {
            sum = 0;

            for (int j = 0; j < b->size(); j++)
            {
                if (i != j)
                {
                    sum += LHS->values[i * LHS->cols + j] * x_var->values[j];
                }
            }

            x_var->values[i] = 1 / LHS->values[i * LHS->cols + i] * (b->values[i] - sum);
        }

        resid_sum = 0;

        // check residual
        for (int i = 0; i < b->size(); i++)
        {
            resid_sum += fabs(estimated_rhs->values[i] - b->values[i]);
        }

        residual = resid_sum / b->size();
        ++iteration;
    }

    // clean memory
    delete estimated_rhs;

    return x_var;
}

// function that implements gaussian elimination
template<class T>
Matrix<T>* Solver<T>::solveGaussian(Matrix<T>* LHS, Matrix<T>* b)
{
    // transform matrices to upper triangular
    Solver<T>::upperTriangular(LHS, b);

    // generate solution
    auto* solution = Solver<T>::backSubstitution(LHS, b);

    return solution;
}

template<class T>
Matrix<T>* Solver<T>::solveLU(Matrix<T>* LHS, Matrix<T>* b) {

    auto upper_tri = new Matrix<T>(LHS->rows, LHS->cols, true);
    auto lower_tri = new Matrix<T>(LHS->rows, LHS->cols, true);
    auto permutation = new Matrix<T>(LHS->rows, LHS->cols, true);

    Solver<T>::luDecompositionPivot(LHS, upper_tri, lower_tri, permutation);

    permutation->transpose();

    auto p_inv_b = permutation->matMatMult(*b);

    auto y_values = Solver<T>::forwardSubstitution(lower_tri, p_inv_b);


    auto* solution = Solver<T>::backSubstitution(upper_tri, y_values);

    delete upper_tri;
    delete lower_tri;
    delete permutation;
    delete p_inv_b;
    delete y_values;

    return solution;
}

// solve Ax = b;
template<class T>
Matrix<T>* Solver<T>::conjugateGradient(Matrix<T>* LHS, Matrix<T>* b, double epsilon, int max_iterations)
{
    int k = 0;
    T beta = 1;
    double alpha = 1;
    T delta_old = 1;

    // intialize to x to 0
    auto x = new Matrix<T>(b->rows, b->cols, true);

    // workout Ax
    auto Ax = LHS->matMatMult(*x);

    // r = b - Ax
    auto r = new Matrix<T>(b->rows, b->cols, true);
    for (int i = 0; i < r->size(); i++)
    {
        r->values[i] = b->values[i] - Ax->values[i];
    }

    auto p = new Matrix<T>(r->rows, r->cols, true);
    auto w = new Matrix<T>(r->rows, r->cols, true);

    double delta = r->innerVectorProduct(*r);

    while (k < max_iterations && (sqrt(delta) > epsilon* sqrt(b->innerVectorProduct(*b))))
    {
        if (k == 1)
        {
            for (int i = 0; i < p->size(); i++)
            {
                p->values[i] = r->values[i];
            }

        }
        else {
            beta = delta / delta_old;

            // p = r + beta * p
            for (int i = 0; i < p->size(); i++)
            {
                p->values[i] = r->values[i] + beta * p->values[i];
            }
        }

        auto Ap = LHS->matMatMult(*p);

        for (int i = 0; i < w->size(); i++)
        {
            w->values[i] = Ap->values[i];
        }

        alpha = delta / p->innerVectorProduct(*w);

        for (int i = 0; i < x->size(); i++)
        {
            x->values[i] = x->values[i] + alpha * p->values[i];
        }

        for (int i = 0; i < r->size(); i++)
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

    double s = -1;

    // copy the values of A into upper triangular matrix
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

    int max_index = -1;
    int max_val = -1;
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

        max_index;

        upper_tri->swapRowsMatrix(k, max_index);
        lower_tri->swapRowsMatrix(k, max_index);
        permutation->swapRowsMatrix(k, max_index);

        // loop over each equation below the pivot
        for (int i = k + 1; i < upper_tri->rows; i++)
        {
            // assumes row major order
            s = upper_tri->values[i * upper_tri->cols + k] / upper_tri->values[k * upper_tri->cols + k];

            for (int j = k; j < upper_tri->cols; j++)
            {
                upper_tri->values[i * upper_tri->cols + j] -= s * upper_tri->values[k * upper_tri->cols + j];
            }

            lower_tri->values[i * lower_tri->rows + k] = s;
        }
    }

    // add zeroes to the diagonal
    for (int i = 0; i < upper_tri->rows; i++)
    {
        lower_tri->values[i * lower_tri->rows + i] = 1;
    }

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

    // scaling factor
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
            s += LHS->values[k * LHS->cols + j] * solution->values[j];
        }

        solution->values[k] = (b->values[k] - s) / LHS->values[k * LHS->cols + k];
    }

    return solution;
}