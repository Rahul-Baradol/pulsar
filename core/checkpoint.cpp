#include "checkpoint.h"
#include "value.h"
#include "util.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

checkpoint::checkpoint() {
    this -> body = std::vector<double>(1, 1.0); 
}

void checkpoint::set_parameters(std::vector<Value*> &parameters) {
    (this -> body).push_back(parameters.size());

    for (Value *val: parameters) {
        (this -> body).push_back(val -> get_data());
        (this -> body).push_back(val -> get_gradient());
    }
}

void checkpoint::save(std::string file_name) {
    std::ofstream ofs(file_name, std::ios::binary);
    if (!ofs) {
        std::cerr << "Couldn't open/write to the file '" << file_name << "'" << std::endl;
        perror("checkpoint save failed");
    } 

    for (double ele: this -> body) {
        uint64_t trans_ele = Util::doubleToUint64(ele);
        trans_ele = Util::toLittleEndian(trans_ele);
        ofs.write(reinterpret_cast<char*>(&trans_ele), sizeof(trans_ele));
    }
}

std::vector<Value*> checkpoint::load_checkpoint(std::string file_name) {
    std::ifstream ifs(file_name, std::ios::binary);
    if (!ifs) {
        std::cerr << "Couldn't open/read from the file '" << file_name << "'" << std::endl;
        perror("checkpoint load failed");
    }

    std::size_t checkpoint_version_sizet; // ignoring this for now
    std::size_t parameter_count_sizet;

    ifs.read(reinterpret_cast<char*>(&checkpoint_version_sizet), sizeof(checkpoint_version_sizet)); 
    ifs.read(reinterpret_cast<char*>(&parameter_count_sizet), sizeof(parameter_count_sizet));

    parameter_count_sizet = Util::fromLittleEndian(parameter_count_sizet);

    double parameter_count = Util::uint64ToDouble(parameter_count_sizet);

    std::vector<Value*> parameters;

    for (int i = 0; i < parameter_count; i++) {
        std::size_t data_sizet, gradient_sizet;
        ifs.read(reinterpret_cast<char*>(&data_sizet), sizeof(data_sizet));
        ifs.read(reinterpret_cast<char*>(&gradient_sizet), sizeof(gradient_sizet));

        data_sizet = Util::fromLittleEndian(data_sizet);
        gradient_sizet = Util::fromLittleEndian(gradient_sizet);

        double data = Util::uint64ToDouble(data_sizet);
        double gradient = Util::uint64ToDouble(gradient_sizet);
    
        Value *value = new Value(data);
        value -> add_gradient(gradient);

        parameters.push_back(value);
    }

    return parameters;
}   