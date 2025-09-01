#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "value.h"
#include <vector>
#include <string>

class checkpoint {
private:
    std::vector<double> body;

public: 
    // version number | no. of params | data, grad, data, grad...
    checkpoint();

    void set_parameters(std::vector<Value*> &parameters);

    void save(std::string file_name);

    std::vector<Value*> load_checkpoint(std::string file_name);

    std::vector<Value*> load(std::string checkpoint_file_name);
};

#endif