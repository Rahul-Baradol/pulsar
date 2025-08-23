#include <layer.h>

Layer::Layer(int number_of_inputs, int number_of_neurons) {
    this -> number_of_inputs = number_of_inputs;
    this -> number_of_neurons = number_of_neurons;

    neurons.reserve(number_of_neurons);
    for (int i = 0; i < number_of_neurons; i++) {
        neurons.emplace_back(
            new Neuron(number_of_inputs)
        );
    }
}

std::vector<Value*> Layer::forward(std::vector<Value*> &input) {    
    std::vector<Value*> outs;
    for (Neuron *neuron: this -> neurons) {
        Value *res = neuron -> forward(input);
        outs.push_back(res);
    }
    return outs;
}   