#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

class Layer {
private:
    int number_of_inputs;
    int number_of_neurons;
    std::vector<Neuron*> neurons;

public:
    Layer(int number_of_inputs, int number_of_neurons);

    std::vector<Value*> forward(std::vector<Value*> &inputs, activation_function act_fun);

    std::vector<Value*> get_parameters();
};

#endif