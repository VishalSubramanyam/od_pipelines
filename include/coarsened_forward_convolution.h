#ifndef COARSENED_FORWARD_CONVOLUTION_H
#define COARSENED_FORWARD_CONVOLUTION_H

__global__ void coarsened_convolution_1C32S(const int batchSize, const float *images, 
const float *filters, float *output,  int gOutputSize, int gPadZeros, int gEven, int gOutputSizeSquared,
int gInputSize, int gInputPlanes, int gFilterSize, int gNumFilters);


__global__ void coarsened_convolution_2C32S(const int batchSize, const float *images, 
const float *filters, float *output,  int gOutputSize, int gPadZeros, int gEven, int gOutputSizeSquared,
int gInputSize, int gInputPlanes, int gFilterSize, int gNumFilters);


__global__ void coarsened_convolution_4C32S(const int batchSize, const float *images, 
const float *filters, float *output,  int gOutputSize, int gPadZeros, int gEven, int gOutputSizeSquared,
int gInputSize, int gInputPlanes, int gFilterSize, int gNumFilters);


__global__ void coarsened_convolution_8C32S(const int batchSize, const float *images, 
const float *filters, float *output,  int gOutputSize, int gPadZeros, int gEven, int gOutputSizeSquared,
int gInputSize, int gInputPlanes, int gFilterSize, int gNumFilters);


__global__ void coarsened_convolution_16C32S(const int batchSize, const float *images, 
const float *filters, float *output,  int gOutputSize, int gPadZeros, int gEven, int gOutputSizeSquared,
int gInputSize, int gInputPlanes, int gFilterSize, int gNumFilters);

__global__ void coarsened_convolution_32C32S(const int batchSize, const float *images, 
const float *filters, float *output,  int gOutputSize, int gPadZeros, int gEven, int gOutputSizeSquared,
int gInputSize, int gInputPlanes, int gFilterSize, int gNumFilters);


#endif
