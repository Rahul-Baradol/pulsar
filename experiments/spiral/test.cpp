    #include "nn.h"
#include "loss.h"
#include "checkpoint.h"
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

class Row {
public:
    float x;
    float y;
    float label;
};

vector<Row> read_csv(string csv_file) { 
    ifstream file(csv_file);
    if (!file.is_open()) {
        cerr << "Error opening the file" << endl;
        exit(1);
    }

    string line;
    vector<Row> values;
    
    while (getline(file, line)) {
        stringstream ss(line);
        string token;

        Row row;
        getline(ss, token, ',');

        try {
            stof(token);
        } catch (const invalid_argument&) {
            continue;
        }

        row.x = stof(token);

        getline(ss, token, ',');
        row.y = stof(token);

        getline(ss, token, ',');
        row.label = stof(token);

        values.push_back(row);
    }

    return values;
}

void write_csv(vector<Row> rows, string csv_file) {
    ofstream file(csv_file);
    if (!file.is_open()) {
        cerr << "Error opening the file for writing" << endl;
        exit(1);
    }   

    file << "x,y,label\n";
    for (const Row& row : rows) {
        file << row.x << "," << row.y << "," << row.label << "\n";
    }

    file.close();
}

void test(NeuralNet *net) {
    vector<Row> rows = read_csv("/home/rahulbaradol/Documents/projects/pulsar/experiments/spiral/spiral_test.csv");
    int size = rows.size();
    cout << "Size of testing dataset: " << size << " rows!" << endl;

    for (Row &row: rows) {
        vector<Value*> input = {
            new Value(row.x),
            new Value(row.y)  
        };

        vector<Value*> prediction = net -> forward(input);
        row.label = (prediction[0] -> get_data()) > 0.5;
    } 

    write_csv(rows, "spiral_test_prediction.csv");
}

int main() {
    NeuralNet *net = new NeuralNet(2, {96, 64, 1}, {
        activation_function::tanh,
        activation_function::tanh,
        activation_function::sigmoid
    }, loss_function::bce);

    checkpoint *chp = new checkpoint();
    std::vector<Value*> parameters = chp -> load_checkpoint("net.pulse");

    std::cout << "loaded parameter size: " << parameters.size() << std::endl;

    net -> set_parameters(parameters);

    test(net);
    
    return 0;
}