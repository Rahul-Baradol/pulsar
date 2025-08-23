#ifndef VALUE_H
#define VALUE_H

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>

using namespace std;

class Value {
private:
    double data;
    double gradient;
    vector<Value*> children;
    string op;    
    
public: 
    function<void()> _backward;
    
    Value(double data);

    Value(double data, vector<Value*> children, string op);
    
    Value* operator+(Value &other);

    Value* operator*(Value &other);

    Value* operator-();

    Value* operator-(Value &other);

    Value* tanh();

    void show();

    double get_data();

    double get_gradient();

    void add_gradient(double gradient);
};

#endif