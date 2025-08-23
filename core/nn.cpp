#include "nn.h"
#include "layer.h"
#include "value.h"
#include <vector>

NeuralNet::NeuralNet(int number_of_inputs, std::vector<int> neurons_per_layer) {
    this -> learning_rate = 0.01;
    this -> number_of_inputs = number_of_inputs;
    this -> neurons_per_layer = neurons_per_layer;

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
        out = (this -> layers[i]) -> forward(out, true);
    }
    
    out = (this -> layers[size-1]) -> forward(out, false);
    return out;
}

Value* NeuralNet::get_loss(std::vector<Value*> ypred, std::vector<Value*> ygt) {
    if (ypred.size() != ygt.size()) {
        throw std::invalid_argument("ypred size does not match ygt size");
    }

    Value *loss = new Value(0.0);
    
    int size = ypred.size();
    for (int i = 0; i < size; i++) {
        Value *diff = (*ypred[i]) - (*ygt[i]);
        Value *sq = (*diff) * (*diff);
        loss = (*loss) + (*sq);
    }

    return loss;
}

void NeuralNet::backward(std::vector<Value*> ypred, std::vector<Value*> ygt) {
    std::vector<Value*> parameters = this -> get_parameters();

    
    Value *loss = get_loss(ypred, ygt);
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