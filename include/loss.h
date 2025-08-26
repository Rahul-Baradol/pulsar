#ifndef LOSS_H
#define LOSS_H

#include "value.h"
#include <vector>

enum class loss_function {
    mse,
    bce
};  

class Loss {
public: 
    static Value* mean_squared(std::vector<Value*> ypred, std::vector<Value*> ygt);
    static Value* binary_cross_entropy(std::vector<Value*> ypred, std::vector<Value*> ygt);
};

#endif