#include "value.h"
#include "neuron.h"
#include <vector>

Neuron::Neuron(int number_of_inputs)
 : gen(rd()), dist(-1.0, 1.0), number_of_inputs(number_of_inputs) 
 {
    this -> bias = new Value(dist(gen));

    weights.reserve(number_of_inputs);
    for (int i = 0; i < number_of_inputs; i++) {
        weights.emplace_back(
            new Value(dist(gen))
        );
    }
}

Value* Neuron::forward(std::vector<Value*> &input, activation_function act_fun) {
    if (input.size() != number_of_inputs) {
        throw std::invalid_argument("Input size does not match number of inputs for neuron.");
    }

    Value *sum = new Value(bias -> get_data());
    for (int i = 0; i < number_of_inputs; i++) {
        Value *a = input[i];
        Value *b = this -> weights[i];
        
        Value *tmp = (*a) * (*b);

        sum = (*sum) + (*tmp);      
    }
    
    if (act_fun == activation_function::tanh) {
        sum = sum -> tanh();
    }
    
    if (act_fun == activation_function::sigmoid) {
        sum = sum -> sigmoid();
    }

    return sum;
}

std::vector<Value*> Neuron::get_parameters() {
    std::vector<Value*> params = this -> weights;
    params.push_back(this -> bias);
    return params;
}