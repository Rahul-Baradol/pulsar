#ifndef UTIL_H
#define UTIL_H

#include <cstdint>

class Util {
public:
    static uint64_t doubleToUint64(double value);

    static double uint64ToDouble(uint64_t value);

    static uint64_t toLittleEndian(uint64_t value);

    static uint64_t fromLittleEndian(uint64_t value);

};

#endif