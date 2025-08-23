#include "nn.h"

int main() {
    std::vector<Value*> v = {
        new Value(2.3),
        new Value(3.3),
        new Value(4.3),
        new Value(5.3)
    };

    NeuralNet *net = new NeuralNet(4, {5, 3, 1});
    std::vector<Value*> outs = net -> forward(v);

    for (Value *ele: outs) {    
        cout << ele -> get_data() << endl;
    }

    
}