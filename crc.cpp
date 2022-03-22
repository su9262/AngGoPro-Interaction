#include <stdio.h> 
#include <cstdint>


#define polynom   0x4C11DB7
#define xorValue  0x00000000
#define initValue 0xFFFFFFFF

uint32_t calcCrc32_32(const uint8_t * data, const uint32_t size){
    uint32_t crc = initValue;
    for (uint32_t i = 0; i < size; i++) {
        crc = calcCrc32Uint32(crc, data[i]);
    }
    return crc ^ xorValue;
}
uint32_t calcCrc32Uint32(uint32_t crc, uint32_t data) {
    int32_t i;
    crc = crc ^ data;
    for (i = 0; i < 32; i++) {
        if (crc & 0x80000000) {
            crc = (crc << 1) ^ polynom;
        } else {
            crc = (crc << 1);
        }
    }
    return (crc);
}

