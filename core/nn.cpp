#include "nn.h"
#include "layer.h"
#include "value.h"
#include "loss.h"
#include <vector>

NeuralNet::NeuralNet(int number_of_inputs, std::vector<int> neurons_per_layer, loss_function loss_fn) {
    this -> learning_rate = 0.01;
    this -> number_of_inputs = number_of_inputs;
    this -> neurons_per_layer = neurons_per_layer;
    this -> loss_fn = loss_fn;

    neurons_per_layer.insert(neurons_per_layer.begin(), number_of_inputs);
    int size = neurons_per_layer.size();
    for (int i = 0; i < size-1; i++) {
        (this -> layers).emplace_back(
            new Layer(neurons_per_layer[i], neurons_per_layer[i+1])
        );
    }
}

std::vector<Value*> NeuralNet::forward(std::vector<Value*> &input) {
    int size = (this -> layers).size();
    
    std::vector<Value*> out = input;
    for (int i = 0; i < size - 1; i++) {
        out = (this -> layers[i]) -> forward(out, activation_function::tanh);
    }
    
    out = (this -> layers[size-1]) -> forward(out, activation_function::sigmoid);
    return out;
}

void NeuralNet::backward(std::vector<Value*> ypred, std::vector<Value*> ygt) {
    std::vector<Value*> parameters = this -> get_parameters();

    Value *loss;
    if (this -> loss_fn == loss_function::mse) {
        loss = Loss::mean_squared(ypred, ygt);
    }

    if (this -> loss_fn == loss_function::bce) {
        loss = Loss::binary_cross_entropy(ypred, ygt);
    }

    loss -> add_gradient(1.0);
    loss -> update_gradients();
    
    int size = parameters.size();
    
    for (Value *value: parameters) {
        double updated_value = value -> get_data() - (value -> get_gradient() * this -> learning_rate);
        value -> set_data(updated_value);
        value -> reset_gradient();
    }
} 

std::vector<Value*> NeuralNet::get_parameters() {
    std::vector<Value*> params;

    for (Layer *layer: this -> layers) {
        std::vector<Value*> layer_params = layer -> get_parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}