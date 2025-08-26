#include "nn.h"
#include "layer.h"
#include "value.h"
#include "loss.h"
#include <vector>
#include <string>

std::string act_fun_to_string(activation_function act_fun) {
    switch (act_fun) {
        case activation_function::tanh:
            return "tanh";
        
        case activation_function::sigmoid:
            return "sigmoid";
    }

    return "none";
}

NeuralNet::NeuralNet(int number_of_inputs, std::vector<int> neurons_per_layer, std::vector<activation_function> layer_funs, loss_function loss_fn) {
    this -> learning_rate = 0.01;
    this -> number_of_inputs = number_of_inputs;
    this -> neurons_per_layer = neurons_per_layer;
    this -> loss_fn = loss_fn;

    neurons_per_layer.insert(neurons_per_layer.begin(), number_of_inputs);

    int size = neurons_per_layer.size();
    int parameter_count = 0;
    
    for (int i = 0; i < size - 1; i++) {
        parameter_count += (neurons_per_layer[i] + 1) * neurons_per_layer[i+1];
    }
    
    std::cout << "Neural net of " << (size - 1) << " layers, " << parameter_count << " parameters" << std::endl;
    for (int i = 0; i < size - 1; i++) {
        std::cout << "Layer " << (i+1) << ": " << neurons_per_layer[i+1] << " neurons, " << act_fun_to_string(layer_funs[i]) << " activation" << std::endl;

        (this -> layers).emplace_back(
            new Layer(i, neurons_per_layer[i], neurons_per_layer[i+1], layer_funs[i])
        );
    }
    std::cout << std::endl;
}

std::vector<Value*> NeuralNet::forward(std::vector<Value*> &input) {
    int size = (this -> layers).size();
    
    std::vector<Value*> out = input;
    for (int i = 0; i < size; i++) {
        out = (this -> layers[i]) -> forward(out);
    }
    return out;
}

void NeuralNet::backward(std::vector<Value*> ypred, std::vector<Value*> ygt) {
    std::vector<Value*> parameters = this -> get_parameters();

    Value *loss;

    switch (this -> loss_fn) {
        case loss_function::mse:
            loss = Loss::mean_squared(ypred, ygt);
            break;
        
        case loss_function::bce:
            loss = Loss::binary_cross_entropy(ypred, ygt);
            break;

        default:
            throw std::runtime_error("invalid loss function");
    }

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

    for (Layer *layer: layers) {
        layer -> clear_residual_data();
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