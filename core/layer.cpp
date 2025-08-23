#include "value.h"
#include "layer.h"
#include <vector>

Layer::Layer(int number_of_inputs, int number_of_neurons) {
    this -> number_of_inputs = number_of_inputs;
    this -> number_of_neurons = number_of_neurons;

    (this -> neurons).reserve(number_of_neurons);
    for (int i = 0; i < number_of_neurons; i++) {
        (this -> neurons).emplace_back(
            new Neuron(number_of_inputs)
        );
    }
}

std::vector<Value*> Layer::forward(std::vector<Value*> &input, bool enable_activation) {    
    std::vector<Value*> outs;
    for (Neuron *neuron: this -> neurons) {
        Value *res = neuron -> forward(input, enable_activation);
        outs.push_back(res);
    }
    return outs;
}   

std::vector<Value*> Layer::get_parameters() {
    std::vector<Value*> params;

    for (Neuron *neuron: this -> neurons) {
        std::vector<Value*> neuron_params = neuron -> get_parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}