#include "neuron.h"

int main() {
    Neuron *neuron = new Neuron(4);

    std::vector<Value*> v = {
        new Value(2.3),
        new Value(3.3),
        new Value(4.3),
        new Value(5.3)
    };

    Value *result = neuron -> forward(v);
    cout << result -> get_data() << endl;

    return 0;   
}