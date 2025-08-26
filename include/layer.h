#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

class Layer {
private:
    int number_of_inputs;
    int number_of_neurons;
    std::vector<Neuron*> neurons;

public:
    Layer(int layer_index, int number_of_inputs, int number_of_neurons, activation_function act_fun);

    std::vector<Value*> forward(std::vector<Value*> &inputs);

    std::vector<Value*> get_parameters();

    void clear_residual_data();
};

#endif