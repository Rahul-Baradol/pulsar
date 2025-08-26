#include "loss.h"

Value* Loss::mean_squared(std::vector<Value*> ypred, std::vector<Value*> ygt) {
    if (ypred.size() != ygt.size()) {
        throw std::invalid_argument("ypred size does not match ygt size");
    }

    Value *loss = new Value(0.0);
    
    int size = ypred.size();
    for (int i = 0; i < size; i++) {
        Value *diff = (*ypred[i]) - (*ygt[i]);
        Value *sq = (*diff) * (*diff);
        loss = (*loss) + (*sq);
    }

    return loss;
}

Value* Loss::binary_cross_entropy(std::vector<Value*> ypred, std::vector<Value*> ygt) {
    Value* pred = ypred[0];  
    Value* target = ygt[0];

    // log(pred)
    Value* log_pred = pred -> log();  

    // target * log(pred)
    Value* t_log_pred = (*target) * (*log_pred);

    // (1 - target)
    Value* one_val = new Value(1.0);
    Value* one_minus_target = (*one_val) - (*target);

    // (1 - pred)
    Value* one_minus_pred = (*one_val) - (*pred);

    // log(1 - pred)
    Value* log_one_minus_pred = one_minus_pred -> log();

    // (1 - target) * log(1 - pred)
    Value* t2_log = (*one_minus_target) * (*log_one_minus_pred);

    // combine: target*log(pred) + (1 - target)*log(1 - pred)
    Value* sum = (*t_log_pred) + (*t2_log);

    // negate: -( ... )
    Value *neg_one = new Value(-1.0);
    Value* loss = (*neg_one) * (*sum);

    return loss;
}