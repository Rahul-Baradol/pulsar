#ifndef NEURON_H
#define NEURON_H

#include "value.h"
#include <vector>
#include <random>

enum class activation_function {
    tanh,
    sigmoid,
    none,
};

class Neuron {
private:    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist{-1.0, 1.0};

    int number_of_inputs;
    std::vector<Value*> weights;
    Value *bias;

public:
    Neuron(int number_of_inputs);

    Value* forward(std::vector<Value*> &input, activation_function act_fun);

    std::vector<Value*> get_parameters();
};

#endif