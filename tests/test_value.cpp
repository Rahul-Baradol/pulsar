#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cmath>
#include "external/doctest.h"
#include "value.h"

double d1 = 2.131;
double d2 = 3.14;

double g1 = 54.32;

TEST_CASE("testing addition") {
    Value *a = new Value(d1);
    Value *b = new Value(d2);

    Value *resptr = (*a) + (*b);

    double res = resptr -> get_data();
    double expected = d1 + d2;
    
    CHECK(res == expected);
}

TEST_CASE("testing addition gradient") {
    Value *a = new Value(d1);
    Value *b = new Value(d2);
    double expected = g1;

    Value *resptr = (*a) + (*b);
    
    resptr -> add_gradient(g1);
    resptr -> update_gradients();

    CHECK(resptr -> get_gradient() == expected);

    CHECK(a -> get_gradient() == expected);
    CHECK(b -> get_gradient() == expected);
}

TEST_CASE("testing subtraction") {
    Value *a = new Value(d1);
    Value *b = new Value(d2);

    Value *resptr = (*a) - (*b);

    double res = resptr -> get_data();
    double expected = d1 - d2;
    
    CHECK(res == expected);
}

TEST_CASE("testing subtraction gradient") {
    Value *a = new Value(d1);
    Value *b = new Value(d2);

    double expected_a = g1;
    double expected_b = -1.0 * g1;

    Value *resptr = (*a) - (*b);

    resptr -> add_gradient(g1);
    resptr -> update_gradients();

    CHECK(resptr -> get_gradient() == g1);

    CHECK(a -> get_gradient() == expected_a);
    CHECK(b -> get_gradient() == expected_b);
}


TEST_CASE("testing multiplication") {
    Value *a = new Value(d1);
    Value *b = new Value(d2);

    Value *resptr = (*a) * (*b);

    double res = resptr -> get_data();
    double expected = d1 * d2;

    CHECK(res == expected);
}

TEST_CASE("testing multiplication gradient") {
    Value *a = new Value(d1);
    Value *b = new Value(d2);

    double expected_a = g1 * d2;
    double expected_b = g1 * d1;

    Value *resptr = (*a) * (*b);

    resptr -> add_gradient(g1);
    resptr -> update_gradients();

    CHECK(resptr -> get_gradient() == g1);

    CHECK(a -> get_gradient() == expected_a);
    CHECK(b -> get_gradient() == expected_b);
}

TEST_CASE("testing tanh") {
    Value *a = new Value(d1);

    Value *resptr = a -> tanh();
    double res = resptr -> get_data();
    double expected = tanh(d1);

    CHECK(res == expected);
}

TEST_CASE("testing tanh gradient") {
    Value *a = new Value(d1);
    double tan_value = std::tanh(d1);

    double expected = g1 * (1 - (tan_value * tan_value));

    Value *resptr = a -> tanh();

    resptr -> add_gradient(g1);
    resptr -> update_gradients();

    CHECK(resptr -> get_gradient() == g1);
    CHECK(a -> get_gradient() == expected);
}