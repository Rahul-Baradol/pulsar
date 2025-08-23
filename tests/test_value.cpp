#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cmath>
#include "external/doctest.h"
#include "value.h"

double d1 = 2.131;
double d2 = 3.14;

TEST_CASE("testing addition") {
    Value *a = new Value(d1);
    Value *b = new Value(d2);

    Value *resptr = (*a) + (*b);

    double res = resptr -> get_data();
    double expected = d1 + d2;
    
    CHECK(res == expected);
}

TEST_CASE("testing subtraction") {
    Value *a = new Value(d1);
    Value *b = new Value(d2);

    Value *resptr = (*a) - (*b);

    double res = resptr -> get_data();
    double expected = d1 - d2;
    
    CHECK(res == expected);
}

TEST_CASE("testing multiplication") {
    Value *a = new Value(d1);
    Value *b = new Value(d2);

    Value *resptr = (*a) * (*b);

    double res = resptr -> get_data();
    double expected = d1 * d2;

    CHECK(res == expected);
}

TEST_CASE("testing tanh") {
    Value *a = new Value(d1);

    Value *resptr = a -> tanh();
    double res = resptr -> get_data();
    double expected = tanh(d1);

    CHECK(res == expected);
}