#include "nn.h"
#include "layer.h"
#include "value.h"
#include <vector>

NeuralNet::NeuralNet(int number_of_inputs, std::vector<int> neurons_per_layer) {
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
    std::vector<Value*> out = input;
    for (Layer *layer: this -> layers) {
        out = layer -> forward(out);
    } 
    return out;
}