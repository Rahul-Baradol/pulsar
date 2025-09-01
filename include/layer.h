#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

class Layer {
private:
    int number_of_inputs;
    int number_of_neurons;
    int parameter_count;
    std::vector<Neuron*> neurons;

public:
    Layer(int layer_index, int number_of_inputs, int number_of_neurons, activation_function act_fun);

    std::vector<Value*> forward(std::vector<Value*> &inputs);

    void set_parameters(std::vector<Value*> parameters);

    std::vector<Value*> get_parameters();

    int get_parameter_count();

    void clear_residual_data();
};

#endif