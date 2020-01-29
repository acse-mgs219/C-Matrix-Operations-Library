#include <iostream>
#include "Matrix.hpp"
#include "Matrix.cpp"
using namespace std;

int main()
{
    double rhs_b_data[4] = {1., 2., -3., 2.};
    double mat_A[16] =
        {1., 0., 3., 7.,
        2., 1., 0., 4.,
        5., 4., 1., -2.,
        4., 1., 6., 2.};
//    double mat_A[16] =
//        {1., 2., 5., 4.,
//        0., 1., 4., 1.,
//        3., 0., 1., 6.,
//        7., 4., -2., 2.};
    
//    double mat_A[16] =
//        {0., 1., 0., 0.,
//        0., 0., 0., 1.,
//        0., 0., 1., 0.,
//        1., 0., 0., 0.};
    
//    double mat_A[16] =
//        {1., 0., 0., 7.,
//        2., 1., 0., 4.,
//        5., 4., 1., -2.,
//        4., 1., 0., 2.};

    int rows = 4;
    int cols = 4;
    //testing our matrix class
    auto *dense_mat = new Matrix<double>(rows, cols, true);//auto here automatically dedect that the class to be created is a Matrix class
    // lets fill our matrix
    for (int i = 0; i < rows * cols; i++)
    {
        dense_mat->values[i] = mat_A[i];
    }
    dense_mat -> printMatrix();

    auto *rhs_b = new Matrix<double>(4, 1, true);
    
    for (int i = 0; i < rows; i++)
    {
        rhs_b->values[i] = rhs_b_data[i];
    }
    
//
//    double* result = new double[4];
//
////    dense_mat->LU_solve(rhs_b, result);
////    cout<<"!!!!!!!! we got solution:"<<endl;
////    for (int i =0; i<4; i++)
////    {
////        cout<<result[i]<<" ";
////    }

    
    rhs_b->printMatrix();
    
    dense_mat->sort_mat(rhs_b);
    dense_mat->printMatrix();
    rhs_b->printMatrix();
    

    delete rhs_b;
    delete dense_mat;
}
