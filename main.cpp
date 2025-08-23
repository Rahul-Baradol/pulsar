#include "layer.h"

int main() {
    std::vector<Value*> v = {
        new Value(2.3),
        new Value(3.3),
        new Value(4.3),
        new Value(5.3)
    };

    Layer *layer = new Layer(4, 3);
    vector<Value*> outs = layer -> forward(v);

    for (Value *ele: outs) {
        cout << ele -> get_data() << endl;
    }

    return 0;   
}