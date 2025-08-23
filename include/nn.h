#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <layer.h>
#include <value.h>
#include <vector>

class NeuralNet {
private:
    int number_of_inputs;
    std::vector<int> neurons_per_layer;
    std::vector<Layer*> layers;

public: 
    NeuralNet(int number_of_inputs, std::vector<int> neurons_per_layer);

    std::vector<Value*> forward(std::vector<Value*> &input);
};      

#endif