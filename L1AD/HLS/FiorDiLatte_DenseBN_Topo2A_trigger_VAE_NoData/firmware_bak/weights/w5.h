//Numpy array shape [32, 16]
//Min -0.312500000000
//Max 0.562500000000
//Number of zeros 64

#ifndef W5_H_
#define W5_H_

#ifndef __SYNTHESIS__
weight5_t w5[512];
#else
weight5_t w5[512] = {0.1875, 0.0000, 0.1250, -0.0625, 0.0000, 0.2500, 0.0625, 0.0000, -0.0625, 0.0000, 0.1250, 0.0625, 0.0000, 0.1250, 0.1250, 0.3750, 0.0625, -0.0625, -0.0625, -0.1875, 0.1875, -0.1250, 0.2500, 0.1250, 0.0625, -0.0625, -0.0625, 0.1250, 0.1250, -0.1250, 0.0625, 0.0625, 0.1250, 0.0000, -0.1250, 0.1250, 0.0000, 0.0000, 0.0625, -0.1250, 0.0000, 0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, 0.0000, 0.0000, 0.0000, 0.1250, 0.0625, -0.0625, 0.1250, -0.1250, 0.0000, -0.0625, 0.0625, 0.1250, 0.0000, -0.1250, -0.0625, -0.0625, 0.0625, 0.0625, 0.1875, -0.1250, 0.0000, 0.0625, 0.0000, 0.1250, 0.1875, 0.0625, 0.0000, 0.0625, 0.1875, -0.0625, -0.1250, 0.2500, 0.0000, 0.0000, -0.0625, 0.1875, -0.1250, 0.0625, -0.0625, -0.1250, 0.0625, 0.0000, 0.1250, 0.0000, 0.2500, -0.1875, 0.0000, 0.0625, -0.0625, 0.1250, 0.3125, 0.0000, 0.0625, 0.1250, 0.0000, 0.0000, 0.0625, -0.1250, 0.0000, -0.1875, -0.0625, 0.0625, -0.0625, -0.1250, 0.0625, 0.1875, -0.0625, 0.0625, 0.0625, 0.0000, -0.1250, -0.1250, -0.1875, 0.0000, -0.1875, 0.1250, 0.0000, 0.0000, 0.1875, 0.1875, 0.1250, 0.0000, -0.1875, 0.1250, 0.1250, -0.1250, -0.1250, -0.1875, -0.1250, 0.0625, 0.1875, 0.0625, 0.1250, 0.0000, 0.3125, 0.1250, 0.2500, -0.0625, 0.1250, -0.0625, 0.0000, 0.1250, -0.2500, 0.0000, 0.0000, 0.2500, -0.1875, -0.2500, -0.0625, 0.0625, 0.0625, 0.0000, -0.1250, 0.0000, -0.0625, 0.0625, -0.0625, -0.0625, 0.0625, -0.0625, 0.1875, 0.0000, -0.0625, 0.1250, 0.1250, 0.1250, 0.0625, 0.2500, -0.0625, -0.1250, 0.0000, -0.1250, 0.0000, -0.0625, -0.0625, 0.0000, 0.0625, 0.0625, -0.1250, -0.0625, 0.1875, -0.0625, -0.1875, -0.0625, -0.1875, -0.1250, 0.0625, -0.0625, 0.1250, -0.1875, -0.0625, 0.0625, 0.1875, -0.0625, -0.1875, 0.0000, -0.1250, -0.0625, -0.0625, 0.1875, -0.0625, -0.1875, -0.1250, 0.0000, 0.1250, 0.1250, 0.1875, -0.0625, 0.1875, 0.1875, 0.0625, 0.1875, 0.4375, 0.1875, 0.0625, 0.1250, -0.0625, 0.0625, 0.0000, 0.0000, 0.0000, -0.1250, -0.1250, 0.1250, -0.0625, 0.0000, 0.0000, 0.0625, 0.0000, 0.0000, 0.1250, 0.3125, 0.2500, 0.0625, 0.0625, -0.0625, -0.1875, 0.0625, 0.0000, -0.0625, -0.1250, -0.0625, -0.1250, 0.1250, 0.1250, 0.0000, 0.0625, 0.0000, -0.1875, 0.1250, 0.0625, 0.0625, 0.0625, -0.0625, -0.2500, 0.0625, 0.0625, -0.0625, -0.0625, 0.0000, -0.1250, 0.1250, 0.1875, 0.1875, 0.0000, -0.1875, 0.0000, 0.0625, 0.0000, 0.1250, 0.1875, 0.1250, 0.2500, -0.0625, -0.1250, 0.1875, -0.1250, -0.0625, -0.0625, 0.0000, 0.2500, 0.1250, 0.3125, 0.0625, 0.1250, -0.0625, 0.0000, 0.0625, -0.1875, -0.0625, 0.0000, 0.1250, 0.0000, 0.0625, 0.1250, 0.1875, 0.1250, 0.1875, -0.1250, -0.1250, -0.0625, -0.1250, 0.0625, 0.0625, -0.1250, -0.0625, 0.1250, 0.1250, -0.0625, -0.1250, -0.0625, -0.2500, -0.0625, 0.0000, 0.1250, 0.4375, 0.2500, 0.0625, 0.2500, -0.1250, -0.0625, 0.0625, 0.2500, -0.1250, 0.0625, 0.2500, 0.1250, 0.2500, 0.0000, 0.0625, 0.1875, 0.0000, 0.1250, 0.1875, 0.1875, 0.0625, 0.0625, 0.1250, 0.2500, 0.2500, -0.1250, 0.1250, 0.1250, 0.1250, 0.2500, 0.0625, -0.0625, 0.0625, -0.1875, 0.1250, -0.0625, 0.0000, 0.0625, 0.0000, 0.0625, 0.0625, 0.1250, -0.1250, -0.1250, -0.1250, -0.1250, 0.1875, 0.3125, -0.1250, 0.1875, 0.0000, -0.0625, 0.0625, 0.0000, 0.2500, 0.2500, -0.2500, 0.0000, 0.1875, 0.0625, -0.1875, 0.0000, -0.0625, 0.2500, 0.0625, 0.0000, 0.1875, 0.1250, 0.0625, 0.0625, 0.1250, -0.1250, -0.0625, 0.1250, 0.1875, -0.2500, 0.3750, -0.0625, 0.1250, 0.1875, -0.0625, 0.0000, 0.0000, 0.0000, 0.1875, 0.0625, 0.0000, 0.1250, -0.1875, 0.0000, 0.1250, -0.1250, 0.2500, 0.2500, 0.0625, 0.0625, 0.1875, 0.1250, 0.1875, -0.0625, 0.2500, 0.1250, 0.0000, 0.2500, -0.1250, 0.1875, 0.1875, 0.1875, -0.0625, -0.0625, 0.0625, 0.1250, -0.0625, 0.3750, 0.0625, 0.1875, 0.1250, 0.4375, 0.0625, 0.2500, 0.1875, 0.1875, 0.0000, -0.0625, -0.1875, 0.0000, -0.1875, 0.0000, 0.0000, 0.0000, -0.0625, 0.0000, 0.0000, 0.0000, 0.1250, 0.0000, 0.0625, 0.0000, 0.1250, -0.1875, 0.3750, 0.0000, 0.1250, 0.0625, 0.0000, -0.1250, 0.0625, 0.2500, 0.0625, -0.0625, -0.0625, 0.0000, 0.0000, 0.0000, -0.0625, 0.0625, 0.1875, 0.2500, 0.0625, 0.0000, 0.3125, 0.0625, 0.1250, 0.4375, 0.0000, 0.1250, 0.1250, 0.2500, 0.1250, 0.0625, 0.1250, 0.1875, 0.0000, 0.0625, -0.1250, 0.0625, 0.2500, 0.0625, -0.0625, 0.0000, 0.0625, 0.0625, 0.0625, -0.0625, 0.1250, -0.0625, 0.3125, 0.1250, 0.0000, -0.1250};
#endif

#endif