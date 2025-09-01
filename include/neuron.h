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
    int parameter_count;
    std::vector<Value*> weights;
    Value *bias;
    std::vector<Value*> residual_pointers;
    activation_function act_fun;

public:
    Neuron(int neuron_index, int number_of_inputs, activation_function act_fun);

    Value* forward(std::vector<Value*> &input);

    void set_parameters(std::vector<Value*> parameters);

    std::vector<Value*> get_parameters();

    int get_parameter_count();

    void clear_residual_data();
};

#endif