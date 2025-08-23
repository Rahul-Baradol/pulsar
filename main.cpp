#include "nn.h"
#include "value.h"

using namespace std;

int main() {
    std::vector<Value*> input = {
        new Value(2.3),
        new Value(3.3),
        new Value(4.3),
        new Value(5.3)
    };

    std::vector<Value*> ygt = {
        new Value(0.76)
    };


    NeuralNet *net = new NeuralNet(4, {5, 3, 1});
    
    std::vector<Value*> ypred = net -> forward(input);
    Value *loss = net -> get_loss(ypred, ygt);

    cout << "ypred: " << ypred[0] -> get_data() << endl;
    cout << "loss: " << loss -> get_data() << endl << endl;

    for (int i = 0; i < 1000; i++) {
        std::vector<Value*> ypred = net -> forward(input);    
        Value *loss = net -> get_loss(ypred, ygt);
        net -> backward(ypred, ygt);
    }

    ypred = net -> forward(input);
    loss = net -> get_loss(ypred, ygt);

    cout << "ypred: " << ypred[0] -> get_data() << endl;
    cout << "loss: " << loss -> get_data() << endl << endl;
}