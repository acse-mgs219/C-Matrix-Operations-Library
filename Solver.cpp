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

template<class T>
Matrix<T>* Solver<T>::solveLU(Matrix<T>* LHS, Matrix<T>* b) {

    auto upper_tri = new Matrix<T>(LHS->rows, LHS->cols, true);
    auto lower_tri = new Matrix<T>(LHS->rows, LHS->cols, true);
    auto permutation = new Matrix<T>(LHS->rows, LHS->cols, true);

    LHS->luDecompositionPivot(upper_tri, lower_tri, permutation);

    permutation->transpose();

    auto p_inv_b = permutation->matMatMult(*b);

    auto y_values = lower_tri->forwardSubstitution(p_inv_b);


    auto* solution = upper_tri->backSubstitution(y_values);

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