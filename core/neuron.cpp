#include "value.h"
#include "neuron.h"
#include <vector>

Neuron::Neuron(int neuron_index, int number_of_inputs, activation_function act_fun)
 : gen(42 + neuron_index), dist(-1.0, 1.0), number_of_inputs(number_of_inputs), act_fun(act_fun) 
 {
    this -> bias = new Value(dist(gen));

    weights.reserve(number_of_inputs);
    for (int i = 0; i < number_of_inputs; i++) {
        weights.emplace_back(
            new Value(dist(gen))
        );
    }

    this -> parameter_count = number_of_inputs + 1;
}

Value* Neuron::forward(std::vector<Value*> &input) {
    if (input.size() != number_of_inputs) {
        throw std::invalid_argument("Input size does not match number of inputs for neuron.");
    }

    Value *sum = new Value(bias -> get_data());
    for (int i = 0; i < number_of_inputs; i++) {
        Value *a = input[i];
        Value *b = this -> weights[i];
        
        Value *tmp = (*a) * (*b);
        
        sum = (*sum) + (*tmp);      
        
        residual_pointers.push_back(tmp);
        residual_pointers.push_back(sum);
    }
    
    switch (this -> act_fun) {
        case activation_function::tanh:
            sum = sum -> tanh();
            break;

        case activation_function::sigmoid:
            sum = sum -> sigmoid();
            break;

        default: 
            throw std::runtime_error("Unknown activation function.");
    }
    
    residual_pointers.push_back(sum);

    return sum;
}

void Neuron::set_parameters(std::vector<Value*> parameters) {
    this -> bias = parameters.back();
    parameters.pop_back();

    this -> weights = parameters;
}

std::vector<Value*> Neuron::get_parameters() {
    std::vector<Value*> params = this -> weights;
    params.push_back(this -> bias);
    return params;
}

int Neuron::get_parameter_count() {
    return this -> parameter_count;
}

void Neuron::clear_residual_data() {
    while (!(this -> residual_pointers).empty()) {
        Value *node = residual_pointers.back();
        residual_pointers.pop_back();

        delete node;
    }
}