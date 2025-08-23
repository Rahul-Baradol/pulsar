#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <layer.h>
#include <value.h>
#include <vector>

class NeuralNet {
private:
    double learning_rate;
    int number_of_inputs;
    std::vector<int> neurons_per_layer;
    std::vector<Layer*> layers;

public: 
    NeuralNet(int number_of_inputs, std::vector<int> neurons_per_layer);

    std::vector<Value*> forward(std::vector<Value*> &input);

    Value* get_loss(std::vector<Value*> ypred, std::vector<Value*> ygt);

    void backward(std::vector<Value*> ypred, std::vector<Value*> ygt);

    std::vector<Value*> get_parameters();
};      

#endif