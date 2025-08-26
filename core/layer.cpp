#include "value.h"
#include "layer.h"
#include <vector>

Layer::Layer(int layer_index, int number_of_inputs, int number_of_neurons, activation_function act_fun)
    : number_of_inputs(number_of_inputs), number_of_neurons(number_of_neurons) {

    (this -> neurons).reserve(number_of_neurons);
    for (int i = 0; i < number_of_neurons; i++) {
        (this -> neurons).emplace_back(
            new Neuron( 
                (layer_index * number_of_neurons) + i,
                number_of_inputs, 
                act_fun
            )
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

std::vector<Value*> Layer::get_parameters() {
    std::vector<Value*> params;

    for (Neuron *neuron: this -> neurons) {
        std::vector<Value*> neuron_params = neuron -> get_parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}

void Layer::clear_residual_data() {
    for (Neuron *neuron: neurons) {
        neuron -> clear_residual_data();
    }
}