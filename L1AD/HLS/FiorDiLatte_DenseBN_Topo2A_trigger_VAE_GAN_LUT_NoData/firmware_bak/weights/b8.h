//Numpy array shape [3]
//Min -0.250000000000
//Max 0.312500000000
//Number of zeros 0

#ifndef B8_H_
#define B8_H_

#ifndef __SYNTHESIS__
bias8_t b8[3];
#else
bias8_t b8[3] = {0.1875, 0.3125, -0.2500};
#endif

#endif
