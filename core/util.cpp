#include "util.h"

#include <cstdint>
#include <cstring>
#include <bit>

uint64_t Util::doubleToUint64(double value) {
    uint64_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

double Util::uint64ToDouble(uint64_t value) {
    double result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

uint64_t Util::toLittleEndian(uint64_t value) {
    if constexpr (std::endian::native == std::endian::big) {
        return std::byteswap(value);
    } else {
        return value;
    }
} 

uint64_t Util::fromLittleEndian(uint64_t value) {
    if constexpr (std::endian::native == std::endian::big) {
        return std::byteswap(value);
    } else {
        return value;
    }
}