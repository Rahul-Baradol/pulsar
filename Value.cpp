#include <iostream>
#include <vector>
#include <string>
#include <functional>

using namespace std;

class Value {
private:
    double data;
    double gradient;
    vector<Value*> children;
    char op;    
public: 
    function<void()> _backward;
    
    Value(double data) {
        this -> data = data;
        this -> gradient = 0.0;
        this -> children = vector<Value*>();
    }

    Value(double data, vector<Value*> children, char op) {
        this -> data = data;
        this -> gradient = 0.0;
        this -> children = children;
        this -> op = op;
    }
    
    Value* operator+(Value &other) {
        Value *new_value = new Value(
            this -> data + other.get_data(),
            vector<Value*>{this, &other}, 
            '+'
        );

        auto _backward = [this, new_value, &other]() {
            this -> add_gradient(new_value -> get_gradient());
            other.add_gradient(new_value -> get_gradient());
        };

        new_value -> _backward = _backward;

        return new_value;
    }       

    void show() {
        cout << "Value={data=" << data << ", gradient=" << gradient << "}" << endl; 
    }

    double get_data() {
        return this -> data;
    }

    double get_gradient() {
        return this -> gradient;
    }

    void add_gradient(double gradient) {
        this -> gradient += gradient;
    }
};

int main() {
    Value *op1 = new Value(3.0);
    Value *op2 = new Value(4.2);

    Value *result = *op1 + *op2;
    
    cout << op1 -> get_gradient() << endl;
    cout << op2 -> get_gradient() << endl;

    result -> add_gradient(4.0);
    result -> _backward();

    cout << op1 -> get_gradient() << endl;
    cout << op2 -> get_gradient() << endl;

    return 0;
}