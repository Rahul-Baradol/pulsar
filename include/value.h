#ifndef VALUE_H
#define VALUE_H

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>

class Value {
private:
    double data;
    double gradient;
    std::string op;    
    std::vector<Value*> children;
    std::function<void()> _backward;
    
public: 
    void update_gradients();
    
    Value(double data);

    Value(double data, std::vector<Value*> children, std::string op);
    
    Value* operator+(Value &other);

    Value* operator*(Value &other);

    Value* operator-();

    Value* operator-(Value &other);

    Value* tanh();

    Value* sigmoid();

    Value* log();

    void show();

    double get_data();

    double get_gradient();

    std::string get_op();

    std::vector<Value*> get_children();

    void add_gradient(double gradient);

    void set_data(double gradient);

    void reset_gradient();
};

#endif