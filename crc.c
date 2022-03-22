#include <stdio.h> 
#include <stdint.h>
#define polynom   0x4C11DB7
#define xorValue  0x00000000
#define initValue 0xFFFFFFFF


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

uint32_t calcCrc32_32(const uint8_t * data, const uint32_t size){
    uint32_t crc = initValue;
    for (uint32_t i = 0; i < size; i++) {
        crc = calcCrc32Uint32(crc, data[i]);
    }
    return crc ^ xorValue;
}

void dist_amp(const uint8_t *data, uint32_t *dist, uint32_t *amp, int32_t len){
    int32_t i,idx;
    idx = 0;
    for(i = 0; i < len; i += 4){
        dist[idx] = ((data[i + 1] & 0xff) << 8) | data[i] & 0xff;
        amp[idx] = ((data[i + 3] & 0xff) << 8) | data[i+2] & 0xff;
    }
}
