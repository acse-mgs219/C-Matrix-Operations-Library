#ifndef UTILITIES_H
#define UTILITIES_H

#include <cmath>
//#define USE_BLAS
#define PERFORMANCE_INFO

template <class T>
bool hasConverged(T approximate[], T real[], int length, double tolerance)
{
    bool test_result = true;
    double residual = 0;

    // check values are reasonably accurate
    for (int i = 0; i < length; i++)
    {
        residual += pow(fabs(approximate[i] - real[i]), 2);
    }

    residual = sqrt(residual) / length;

    return residual < tolerance*10;
}

#endif
