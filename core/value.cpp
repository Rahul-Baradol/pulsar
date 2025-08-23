#include "value.h"
#include <cmath>

using namespace std;

Value::Value(double data) {
    this -> data = data;
    this -> gradient = 0.0;
    this -> children = vector<Value*>();
    this -> _backward = [](){};
}

Value::Value(double data, vector<Value*> children, string op) {
    this -> data = data;
    this -> gradient = 0.0;
    this -> children = children;
    this -> op = op;
}

Value* Value::operator+(Value &other) {
    Value *new_value = new Value(
        this -> data + other.get_data(),
        vector<Value*>{this, &other}, 
        "+"
    );

    auto _backward = [this, new_value, &other]() {
        this -> add_gradient(new_value -> get_gradient());
        other.add_gradient(new_value -> get_gradient());
    };

    new_value -> _backward = _backward;

    return new_value;
}       

Value* Value::operator*(Value &other) {
    Value *new_value = new Value(
        this -> data * other.get_data(),
        vector<Value*>{this, &other}, 
        "*"
    );

    auto _backward = [this, new_value, &other]() {
        this -> add_gradient(
            (new_value -> get_gradient()) * (other.get_data())
        );

        other.add_gradient(
            (new_value -> get_gradient()) * (this -> get_data())
        );
    };

    new_value -> _backward = _backward;

    return new_value;
}       

Value* Value::operator-() {
    return new Value(this -> data * -1);
}

Value* Value::operator-(Value &other) {
    return (*this) + *(-other);
}

Value* Value::tanh() {
    double tan_value = std::tanh(this -> data);

    Value *new_value = new Value(
        tan_value,
        vector<Value*>{this},
        "tanh"
    );

    auto _backward = [this, tan_value, new_value]() {
        double gradient = (1 - (tan_value * tan_value)) * (new_value -> get_gradient());
        this -> add_gradient(gradient);
    };

    this -> _backward = _backward;
    return new_value;
}

void Value::show() {
    cout << "Value={data=" << data << ", gradient=" << gradient << "}" << endl; 
}

double Value::get_data() {
    return this -> data;
}

double Value::get_gradient() {
    return this -> gradient;
}

void Value::add_gradient(double gradient) {
    this -> gradient += gradient;
}