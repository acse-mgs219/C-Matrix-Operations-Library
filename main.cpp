#define CATCH_CONFIG_RUNNER
#include "test/catch.hpp"
#include "cblas.h"


//double *matmatmult()
//{
//
//    //    character 	TRANSA,
//
////    character 	TRANSB,
//
////    integer 	M,
//
////    integer 	N,
//
////    integer 	K,
//
////    real 	ALPHA,
//
////    real, dimension(lda,*) 	A,
//
////    integer 	LDA,
//
////    real, dimension(ldb,*) 	B,
//
////    integer 	LDB,
//
////    real 	BETA,
//
////    real, dimension(ldc,*) 	C,
//
////    integer 	LDC
//}



int main()
{
    // run tests
    int result = Catch::Session().run();


//    int i=0;
//    double A[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};
//    double B[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};
//    double C[9] = {.5,.5,.5,.5,.5,.5,.5,.5,.5};
////    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,3,3,2,1, A, 3, B, 3,2, C,3);
//
//    std::cout << cblas_dnrm2(6, A, 1) << std::endl;


    return 0;
}
