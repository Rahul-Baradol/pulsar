#include "value.h"
#include <cmath>
#include <set>
#include <algorithm>

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
    this -> _backward = [](){};
}

std::vector<Value*> build_topo(Value *node) {
    std::set<Value*> visited;
    std::vector<Value*> topo_sorted;

    std::function<void(Value*)> topo_sort = [&visited, &topo_sorted, &topo_sort](Value *node) {
        if (visited.find(node) != visited.end()) {
            return;
        }   

        visited.insert(node);
        for (Value *next: node -> get_children()) {
            topo_sort(next);
        }

        topo_sorted.push_back(node);
    };

    topo_sort(node);

    return topo_sorted;
}

void Value::update_gradients() {
    std::vector<Value*> topo_sort = build_topo(this);
    reverse(topo_sort.begin(), topo_sort.end());

    for (Value *value: topo_sort) {
        value -> _backward();
    }
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

Value* Value::operator-(Value &other) {
    Value *new_value = new Value(
        this->data - other.get_data(),
        vector<Value*>{this, &other},
        "-"
    );

    auto _backward = [this, new_value, &other]() {
        this->add_gradient(new_value -> get_gradient());
        other.add_gradient(-1.0 * new_value -> get_gradient());
    };

    new_value->_backward = _backward;

    return new_value;
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

    new_value -> _backward = _backward;
    
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

std::string Value::get_op() {
    return this -> op;
}

std::vector<Value*> Value::get_children() {
    return this -> children;
}

void Value::add_gradient(double gradient) {
    this -> gradient += gradient;
}

void Value::set_data(double data) {
    this -> data = data;
}

void Value::reset_gradient() {
    this -> gradient = 0.0;
}