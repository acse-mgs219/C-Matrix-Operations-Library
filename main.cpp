#define CATCH_CONFIG_RUNNER
#include "test/catch.hpp"
#include "cblas.h"
#include <time.h>


int main()
{
    clock_t tStart = clock();
    // run tests
    int result = Catch::Session().run();
    clock_t tEnd = clock();

    std::cout << "Time: " << (1000 * (tEnd - tStart))/ CLOCKS_PER_SEC << "ms.";
    return result;
}
