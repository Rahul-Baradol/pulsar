#include "nn.h"
#include "loss.h"
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>

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

void train(NeuralNet *net) {
    vector<Row> rows = read_csv("/home/rahulbaradol/Documents/projects/pulsar/experiments/spiral/spiral_train.csv");
    int size = rows.size();
    cout << "Size of training dataset: " << size << " rows!" << endl;

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(1, rows.size() - 1);

    int steps = 5000;
    for (int i = 0; i < steps; i++) {
        int sample_index = dist(gen);
        Row sample = rows[sample_index];

        vector<Value*> input = {
            new Value(sample.x),
            new Value(sample.y)  
        };

        vector<Value*> ygt = {
            new Value(sample.label)
        };

        vector<Value*> ypred = net -> forward(input);
        net -> backward(ypred, ygt);
    }
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
    NeuralNet *net = new NeuralNet(2, {64, 64, 1}, loss_function::bce);

    train(net);

    test(net);

    return 0;
}