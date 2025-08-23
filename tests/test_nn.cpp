#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cmath>
#include "external/doctest.h"
#include "nn.h"
#include "value.h"

TEST_CASE("testing linear case 1") {
    std::vector<Value*> input = {
        new Value(1),
        new Value(2),
        new Value(3),
        new Value(4)
    };

    std::vector<Value*> ygt = {
        new Value(3),
        new Value(5),
        new Value(7),
        new Value(9)
    };  

    NeuralNet *net = new NeuralNet(4, {4, 2, 4});

    std::vector<Value*> ypred;
    for (int i = 0; i < 250; i++) {
        ypred = net -> forward(input);
        net -> backward(ypred, ygt);
    }

    std::cout << "predictions: ";
    for (Value *pred: ypred) {
        std::cout << pred -> get_data() << ", ";
    }
    std::cout << std::endl;

    std::cout << "ground truth: ";
    for (Value *gt: ygt) {
        std::cout << gt -> get_data() << ", ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < ypred.size(); ++i) {
        CHECK(doctest::Approx(ypred[i] -> get_data()).epsilon(0.1) == ygt[i] -> get_data());
    }
}

TEST_CASE("testing linear case 2") {
    std::vector<Value*> input = {
        new Value(-2),
        new Value(0),
        new Value(2),
        new Value(4)
    };

    std::vector<Value*> ygt = {
        new Value(3),
        new Value(2),
        new Value(1),
        new Value(0)
    };  

    NeuralNet *net = new NeuralNet(4, {4, 2, 4});

    std::vector<Value*> ypred;
    for (int i = 0; i < 250; i++) {
        ypred = net -> forward(input);
        net -> backward(ypred, ygt);
    }

    std::cout << "predictions: ";
    for (Value *pred: ypred) {
        std::cout << pred -> get_data() << ", ";
    }
    std::cout << std::endl;

    std::cout << "ground truth: ";
    for (Value *gt: ygt) {
        std::cout << gt -> get_data() << ", ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < ypred.size(); ++i) {
        CHECK(doctest::Approx(ypred[i] -> get_data()).epsilon(0.1) == ygt[i] -> get_data());
    }
}

TEST_CASE("testing non linear case 1") {
    std::vector<Value*> input = {
        new Value(-2),
        new Value(-1),
        new Value(0),
        new Value(1),
        new Value(2)
    };

    std::vector<Value*> ygt = {
        new Value(9),
        new Value(4),
        new Value(1),
        new Value(0),
        new Value(1)
    };  

    NeuralNet *net = new NeuralNet(5, {4, 2, 5});

    std::vector<Value*> ypred;
    for (int i = 0; i < 250; i++) {
        ypred = net -> forward(input);
        net -> backward(ypred, ygt);
    }

    std::cout << "predictions: ";
    for (Value *pred: ypred) {
        std::cout << pred -> get_data() << ", ";
    }
    std::cout << std::endl;

    std::cout << "ground truth: ";
    for (Value *gt: ygt) {
        std::cout << gt -> get_data() << ", ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < ypred.size(); ++i) {
        CHECK(doctest::Approx(ypred[i] -> get_data()).epsilon(0.1) == ygt[i] -> get_data());
    }
}

TEST_CASE("testing non linear case 2") {
    std::vector<Value*> input = {
        new Value(-3),
        new Value(-1),
        new Value(0),
        new Value(1),
        new Value(3)
    };

    std::vector<Value*> ygt = {
        new Value(-0.995),
        new Value(-0.761),
        new Value(0.0),
        new Value(0.761),
        new Value(0.995)
    };  

    NeuralNet *net = new NeuralNet(5, {4, 2, 5});

    std::vector<Value*> ypred;
    for (int i = 0; i < 250; i++) {
        ypred = net -> forward(input);
        net -> backward(ypred, ygt);
    }

    std::cout << "predictions: ";
    for (Value *pred: ypred) {
        std::cout << pred -> get_data() << ", ";
    }
    std::cout << std::endl;

    std::cout << "ground truth: ";
    for (Value *gt: ygt) {
        std::cout << gt -> get_data() << ", ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < ypred.size(); ++i) {
        CHECK(doctest::Approx(ypred[i] -> get_data()).epsilon(0.1) == ygt[i] -> get_data());
    }
}