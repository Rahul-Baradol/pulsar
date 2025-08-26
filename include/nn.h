#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "loss.h"
#include <layer.h>
#include <value.h>
#include <vector>

class NeuralNet {
private:
    double learning_rate;
    int number_of_inputs;
    std::vector<int> neurons_per_layer;
    std::vector<Layer*> layers;
    loss_function loss_fn;

public: 
    NeuralNet(int number_of_inputs, std::vector<int> neurons_per_layer, loss_function loss_fn);

    std::vector<Value*> forward(std::vector<Value*> &input);

    void backward(std::vector<Value*> ypred, std::vector<Value*> ygt);

    std::vector<Value*> get_parameters();
};      

#endif