#include <cmath>

#ifndef LECTURES_UTILITIES_H
#define LECTURES_UTILITIES_H

// a function to compare two numbers within a given tolerance
bool fEqual(double a, double b, double tolerance)
{
    return fabs(a - b) < tolerance;
}

#endif //LECTURES_UTILITIES_H
