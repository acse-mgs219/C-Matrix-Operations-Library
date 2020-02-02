#ifndef UTILITIES_H
#define UTILITIES_H

#include <cmath>
#define USE_BLAS

// a function to compare two numbers within a given tolerance
bool fEqual(double a, double b, double tolerance)
{
    return fabs(a - b) < tolerance;
}

#endif
