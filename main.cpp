#define CATCH_CONFIG_RUNNER
#include "test/catch.hpp"

int main()
{
    // run tests
    int result = Catch::Session().run();

    return result;
}
