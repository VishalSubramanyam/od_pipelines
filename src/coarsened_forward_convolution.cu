#include "coarsened_forward_convolution.h"
#define ACTIVATION(output) (output> 0 ? output : 0)

__global__ void coarsened_convolution_1C32S(const int batchSize, const float *images, 
const float *filters, float *output,  int gOutputSize, int gPadZeros, int gEven, int gOutputSizeSquared,
int gInputSize, int gInputPlanes, int gFilterSize, int gNumFilters)
{
    __shared__ float _filterCube[600]; // < filterCubeLength
    __shared__ float _upstreamImage[512]; // < gInputSizeSquared

    int gFilterSizeSquared = (gFilterSize*gFilterSize);
    int gHalfFilterSize = (gFilterSize/2);
    int filterCubeLength =  (gInputPlanes*gFilterSizeSquared);
    int gInputSizeSquared = (gInputSize * gInputSize);

    const int globalId = blockDim.x*blockIdx.x+threadIdx.x;
    const int workgroupId = blockIdx.x;
    const int workgroupSize = 1*blockDim.x;
    const int n = workgroupId / gNumFilters;
    const int outPlane = workgroupId % gNumFilters;

    const int localId0 = (threadIdx.x/32)*32*1 +threadIdx.x%32+0*32;
    int outputRow0 = localId0/gOutputSize;
    int outputCol0 = localId0%gOutputSize;
    const int minu0 = gPadZeros ? max(-gHalfFilterSize, -outputRow0) : -gHalfFilterSize;
    const int maxu0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow0  - gEven) : gHalfFilterSize - gEven;
    const int minv0 = gPadZeros ? max(-gHalfFilterSize, -outputCol0) : - gHalfFilterSize;
    const int maxv0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol0 - gEven) : gHalfFilterSize - gEven;

    const int numUpstreamsPerThread = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;
    //const int filterCubeLength = gInputPlanes * gFilterSizeSquared;
    const int filterCubeGlobalOffset = outPlane * filterCubeLength;
    const int numPixelsPerThread = (filterCubeLength + workgroupSize - 1) / workgroupSize;
    for (int i = 0; i < numPixelsPerThread; i++) 
    {

        int thisOffset0 = localId0 + i * workgroupSize;
		if(thisOffset0 < filterCubeLength){
			_filterCube[thisOffset0] = filters[filterCubeGlobalOffset + thisOffset0];
			}

    }

    float sum0 = 0;


    for (int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++) {
        int thisUpstreamImageOffset = (n * gInputPlanes + upstreamPlane) * gInputSizeSquared;
        __syncthreads();
        for (int i = 0; i < numUpstreamsPerThread; i++) {
    
        int thisOffset0 = workgroupSize * i + localId0;
	    if (thisOffset0 < gInputSizeSquared){
		_upstreamImage[ thisOffset0 ] = images[ thisUpstreamImageOffset + thisOffset0 ];
		}
    
    }
        __syncthreads();
        int filterImageOffset = upstreamPlane * gFilterSizeSquared;
    
        for (int u = minu0; u <= maxu0; u++) {
            int inputRow = outputRow0 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv0; v <= maxv0; v++) {
                int inputCol = outputCol0 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId0 < gOutputSizeSquared) {
                   sum0 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
    }

    int resultIndex0 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId0;
    if (localId0 < gOutputSizeSquared){
	 output[resultIndex0] = sum0;
 	}

}

  





























__global__ void coarsened_convolution_2C32S(const int batchSize, const float *images, 
const float *filters, float *output,  int gOutputSize, int gPadZeros, int gEven, int gOutputSizeSquared,
int gInputSize, int gInputPlanes, int gFilterSize, int gNumFilters)
{
    __shared__ float _filterCube[100]; // < filterCubeLength
    __shared__ float _upstreamImage[2000]; // < gInputSizeSquared

    int gFilterSizeSquared = (gFilterSize*gFilterSize);
    int gHalfFilterSize = (gFilterSize/2);
    int filterCubeLength =  (gInputPlanes*gFilterSizeSquared);
    int gInputSizeSquared = (gInputSize * gInputSize);

    const int globalId = blockDim.x*blockIdx.x+threadIdx.x;
    const int workgroupId = blockIdx.x;
    const int workgroupSize = 2*blockDim.x;
    const int n = workgroupId / gNumFilters;
    const int outPlane = workgroupId % gNumFilters;

    const int localId0 = (threadIdx.x/32)*32*2 +threadIdx.x%32+0*32;
    int outputRow0 = localId0/gOutputSize;
    int outputCol0 = localId0%gOutputSize;
    const int minu0 = gPadZeros ? max(-gHalfFilterSize, -outputRow0) : -gHalfFilterSize;
    const int maxu0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow0  - gEven) : gHalfFilterSize - gEven;
    const int minv0 = gPadZeros ? max(-gHalfFilterSize, -outputCol0) : - gHalfFilterSize;
    const int maxv0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol0 - gEven) : gHalfFilterSize - gEven;

    const int localId1 = (threadIdx.x/32)*32*2 +threadIdx.x%32+1*32;
    int outputRow1 = localId1/gOutputSize;
    int outputCol1 = localId1%gOutputSize;
    const int minu1 = gPadZeros ? max(-gHalfFilterSize, -outputRow1) : -gHalfFilterSize;
    const int maxu1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow1  - gEven) : gHalfFilterSize - gEven;
    const int minv1 = gPadZeros ? max(-gHalfFilterSize, -outputCol1) : - gHalfFilterSize;
    const int maxv1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol1 - gEven) : gHalfFilterSize - gEven;

    const int numUpstreamsPerThread = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;
    //const int filterCubeLength = gInputPlanes * gFilterSizeSquared;
    const int filterCubeGlobalOffset = outPlane * filterCubeLength;
    const int numPixelsPerThread = (filterCubeLength + workgroupSize - 1) / workgroupSize;
    for (int i = 0; i < numPixelsPerThread; i++) 
    {

        int thisOffset0 = localId0 + i * workgroupSize;
		if(thisOffset0 < filterCubeLength){
			_filterCube[thisOffset0] = filters[filterCubeGlobalOffset + thisOffset0];
			}

        int thisOffset1 = localId1 + i * workgroupSize;
		if(thisOffset1 < filterCubeLength){
			_filterCube[thisOffset1] = filters[filterCubeGlobalOffset + thisOffset1];
			}

    }

    float sum0 = 0;

    float sum1 = 0;


    for (int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++) {
        int thisUpstreamImageOffset = (n * gInputPlanes + upstreamPlane) * gInputSizeSquared;
        __syncthreads();
        for (int i = 0; i < numUpstreamsPerThread; i++) {
    
        int thisOffset0 = workgroupSize * i + localId0;
	    if (thisOffset0 < gInputSizeSquared){
		_upstreamImage[ thisOffset0 ] = images[ thisUpstreamImageOffset + thisOffset0 ];
		}
    
        int thisOffset1 = workgroupSize * i + localId1;
	    if (thisOffset1 < gInputSizeSquared){
		_upstreamImage[ thisOffset1 ] = images[ thisUpstreamImageOffset + thisOffset1 ];
		}
    
    }
        __syncthreads();
        int filterImageOffset = upstreamPlane * gFilterSizeSquared;
    
        for (int u = minu0; u <= maxu0; u++) {
            int inputRow = outputRow0 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv0; v <= maxv0; v++) {
                int inputCol = outputCol0 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId0 < gOutputSizeSquared) {
                   sum0 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu1; u <= maxu1; u++) {
            int inputRow = outputRow1 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv1; v <= maxv1; v++) {
                int inputCol = outputCol1 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId1 < gOutputSizeSquared) {
                   sum1 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
    }

    int resultIndex0 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId0;
    if (localId0 < gOutputSizeSquared){
	 output[resultIndex0] = sum0;
 	}

    int resultIndex1 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId1;
    if (localId1 < gOutputSizeSquared){
	 output[resultIndex1] = sum1;
 	}

}

  





























__global__ void coarsened_convolution_4C32S(const int batchSize, const float *images, 
const float *filters, float *output,  int gOutputSize, int gPadZeros, int gEven, int gOutputSizeSquared,
int gInputSize, int gInputPlanes, int gFilterSize, int gNumFilters)
{
    __shared__ float _filterCube[100]; // < filterCubeLength
    __shared__ float _upstreamImage[2000]; // < gInputSizeSquared

    int gFilterSizeSquared = (gFilterSize*gFilterSize);
    int gHalfFilterSize = (gFilterSize/2);
    int filterCubeLength =  (gInputPlanes*gFilterSizeSquared);
    int gInputSizeSquared = (gInputSize * gInputSize);

    const int globalId = blockDim.x*blockIdx.x+threadIdx.x;
    const int workgroupId = blockIdx.x;
    const int workgroupSize = 4*blockDim.x;
    const int n = workgroupId / gNumFilters;
    const int outPlane = workgroupId % gNumFilters;

    const int localId0 = (threadIdx.x/32)*32*4 +threadIdx.x%32+0*32;
    int outputRow0 = localId0/gOutputSize;
    int outputCol0 = localId0%gOutputSize;
    const int minu0 = gPadZeros ? max(-gHalfFilterSize, -outputRow0) : -gHalfFilterSize;
    const int maxu0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow0  - gEven) : gHalfFilterSize - gEven;
    const int minv0 = gPadZeros ? max(-gHalfFilterSize, -outputCol0) : - gHalfFilterSize;
    const int maxv0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol0 - gEven) : gHalfFilterSize - gEven;

    const int localId1 = (threadIdx.x/32)*32*4 +threadIdx.x%32+1*32;
    int outputRow1 = localId1/gOutputSize;
    int outputCol1 = localId1%gOutputSize;
    const int minu1 = gPadZeros ? max(-gHalfFilterSize, -outputRow1) : -gHalfFilterSize;
    const int maxu1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow1  - gEven) : gHalfFilterSize - gEven;
    const int minv1 = gPadZeros ? max(-gHalfFilterSize, -outputCol1) : - gHalfFilterSize;
    const int maxv1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol1 - gEven) : gHalfFilterSize - gEven;

    const int localId2 = (threadIdx.x/32)*32*4 +threadIdx.x%32+2*32;
    int outputRow2 = localId2/gOutputSize;
    int outputCol2 = localId2%gOutputSize;
    const int minu2 = gPadZeros ? max(-gHalfFilterSize, -outputRow2) : -gHalfFilterSize;
    const int maxu2 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow2  - gEven) : gHalfFilterSize - gEven;
    const int minv2 = gPadZeros ? max(-gHalfFilterSize, -outputCol2) : - gHalfFilterSize;
    const int maxv2 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol2 - gEven) : gHalfFilterSize - gEven;

    const int localId3 = (threadIdx.x/32)*32*4 +threadIdx.x%32+3*32;
    int outputRow3 = localId3/gOutputSize;
    int outputCol3 = localId3%gOutputSize;
    const int minu3 = gPadZeros ? max(-gHalfFilterSize, -outputRow3) : -gHalfFilterSize;
    const int maxu3 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow3  - gEven) : gHalfFilterSize - gEven;
    const int minv3 = gPadZeros ? max(-gHalfFilterSize, -outputCol3) : - gHalfFilterSize;
    const int maxv3 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol3 - gEven) : gHalfFilterSize - gEven;

    const int numUpstreamsPerThread = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;
    //const int filterCubeLength = gInputPlanes * gFilterSizeSquared;
    const int filterCubeGlobalOffset = outPlane * filterCubeLength;
    const int numPixelsPerThread = (filterCubeLength + workgroupSize - 1) / workgroupSize;
    for (int i = 0; i < numPixelsPerThread; i++) 
    {

        int thisOffset0 = localId0 + i * workgroupSize;
		if(thisOffset0 < filterCubeLength){
			_filterCube[thisOffset0] = filters[filterCubeGlobalOffset + thisOffset0];
			}

        int thisOffset1 = localId1 + i * workgroupSize;
		if(thisOffset1 < filterCubeLength){
			_filterCube[thisOffset1] = filters[filterCubeGlobalOffset + thisOffset1];
			}

        int thisOffset2 = localId2 + i * workgroupSize;
		if(thisOffset2 < filterCubeLength){
			_filterCube[thisOffset2] = filters[filterCubeGlobalOffset + thisOffset2];
			}

        int thisOffset3 = localId3 + i * workgroupSize;
		if(thisOffset3 < filterCubeLength){
			_filterCube[thisOffset3] = filters[filterCubeGlobalOffset + thisOffset3];
			}

    }

    float sum0 = 0;

    float sum1 = 0;

    float sum2 = 0;

    float sum3 = 0;


    for (int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++) {
        int thisUpstreamImageOffset = (n * gInputPlanes + upstreamPlane) * gInputSizeSquared;
        __syncthreads();
        for (int i = 0; i < numUpstreamsPerThread; i++) {
    
        int thisOffset0 = workgroupSize * i + localId0;
	    if (thisOffset0 < gInputSizeSquared){
		_upstreamImage[ thisOffset0 ] = images[ thisUpstreamImageOffset + thisOffset0 ];
		}
    
        int thisOffset1 = workgroupSize * i + localId1;
	    if (thisOffset1 < gInputSizeSquared){
		_upstreamImage[ thisOffset1 ] = images[ thisUpstreamImageOffset + thisOffset1 ];
		}
    
        int thisOffset2 = workgroupSize * i + localId2;
	    if (thisOffset2 < gInputSizeSquared){
		_upstreamImage[ thisOffset2 ] = images[ thisUpstreamImageOffset + thisOffset2 ];
		}
    
        int thisOffset3 = workgroupSize * i + localId3;
	    if (thisOffset3 < gInputSizeSquared){
		_upstreamImage[ thisOffset3 ] = images[ thisUpstreamImageOffset + thisOffset3 ];
		}
    
    }
        __syncthreads();
        int filterImageOffset = upstreamPlane * gFilterSizeSquared;
    
        for (int u = minu0; u <= maxu0; u++) {
            int inputRow = outputRow0 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv0; v <= maxv0; v++) {
                int inputCol = outputCol0 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId0 < gOutputSizeSquared) {
                   sum0 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu1; u <= maxu1; u++) {
            int inputRow = outputRow1 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv1; v <= maxv1; v++) {
                int inputCol = outputCol1 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId1 < gOutputSizeSquared) {
                   sum1 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu2; u <= maxu2; u++) {
            int inputRow = outputRow2 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv2; v <= maxv2; v++) {
                int inputCol = outputCol2 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId2 < gOutputSizeSquared) {
                   sum2 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu3; u <= maxu3; u++) {
            int inputRow = outputRow3 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv3; v <= maxv3; v++) {
                int inputCol = outputCol3 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId3 < gOutputSizeSquared) {
                   sum3 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
    }

    int resultIndex0 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId0;
    if (localId0 < gOutputSizeSquared){
	 output[resultIndex0] = sum0;
 	}

    int resultIndex1 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId1;
    if (localId1 < gOutputSizeSquared){
	 output[resultIndex1] = sum1;
 	}

    int resultIndex2 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId2;
    if (localId2 < gOutputSizeSquared){
	 output[resultIndex2] = sum2;
 	}

    int resultIndex3 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId3;
    if (localId3 < gOutputSizeSquared){
	 output[resultIndex3] = sum3;
 	}

}

  





























__global__ void coarsened_convolution_8C32S(const int batchSize, const float *images, 
const float *filters, float *output,  int gOutputSize, int gPadZeros, int gEven, int gOutputSizeSquared,
int gInputSize, int gInputPlanes, int gFilterSize, int gNumFilters)
{
    __shared__ float _filterCube[100]; // < filterCubeLength
    __shared__ float _upstreamImage[2000]; // < gInputSizeSquared

    int gFilterSizeSquared = (gFilterSize*gFilterSize);
    int gHalfFilterSize = (gFilterSize/2);
    int filterCubeLength =  (gInputPlanes*gFilterSizeSquared);
    int gInputSizeSquared = (gInputSize * gInputSize);

    const int globalId = blockDim.x*blockIdx.x+threadIdx.x;
    const int workgroupId = blockIdx.x;
    const int workgroupSize = 8*blockDim.x;
    const int n = workgroupId / gNumFilters;
    const int outPlane = workgroupId % gNumFilters;

    const int localId0 = (threadIdx.x/32)*32*8 +threadIdx.x%32+0*32;
    int outputRow0 = localId0/gOutputSize;
    int outputCol0 = localId0%gOutputSize;
    const int minu0 = gPadZeros ? max(-gHalfFilterSize, -outputRow0) : -gHalfFilterSize;
    const int maxu0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow0  - gEven) : gHalfFilterSize - gEven;
    const int minv0 = gPadZeros ? max(-gHalfFilterSize, -outputCol0) : - gHalfFilterSize;
    const int maxv0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol0 - gEven) : gHalfFilterSize - gEven;

    const int localId1 = (threadIdx.x/32)*32*8 +threadIdx.x%32+1*32;
    int outputRow1 = localId1/gOutputSize;
    int outputCol1 = localId1%gOutputSize;
    const int minu1 = gPadZeros ? max(-gHalfFilterSize, -outputRow1) : -gHalfFilterSize;
    const int maxu1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow1  - gEven) : gHalfFilterSize - gEven;
    const int minv1 = gPadZeros ? max(-gHalfFilterSize, -outputCol1) : - gHalfFilterSize;
    const int maxv1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol1 - gEven) : gHalfFilterSize - gEven;

    const int localId2 = (threadIdx.x/32)*32*8 +threadIdx.x%32+2*32;
    int outputRow2 = localId2/gOutputSize;
    int outputCol2 = localId2%gOutputSize;
    const int minu2 = gPadZeros ? max(-gHalfFilterSize, -outputRow2) : -gHalfFilterSize;
    const int maxu2 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow2  - gEven) : gHalfFilterSize - gEven;
    const int minv2 = gPadZeros ? max(-gHalfFilterSize, -outputCol2) : - gHalfFilterSize;
    const int maxv2 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol2 - gEven) : gHalfFilterSize - gEven;

    const int localId3 = (threadIdx.x/32)*32*8 +threadIdx.x%32+3*32;
    int outputRow3 = localId3/gOutputSize;
    int outputCol3 = localId3%gOutputSize;
    const int minu3 = gPadZeros ? max(-gHalfFilterSize, -outputRow3) : -gHalfFilterSize;
    const int maxu3 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow3  - gEven) : gHalfFilterSize - gEven;
    const int minv3 = gPadZeros ? max(-gHalfFilterSize, -outputCol3) : - gHalfFilterSize;
    const int maxv3 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol3 - gEven) : gHalfFilterSize - gEven;

    const int localId4 = (threadIdx.x/32)*32*8 +threadIdx.x%32+4*32;
    int outputRow4 = localId4/gOutputSize;
    int outputCol4 = localId4%gOutputSize;
    const int minu4 = gPadZeros ? max(-gHalfFilterSize, -outputRow4) : -gHalfFilterSize;
    const int maxu4 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow4  - gEven) : gHalfFilterSize - gEven;
    const int minv4 = gPadZeros ? max(-gHalfFilterSize, -outputCol4) : - gHalfFilterSize;
    const int maxv4 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol4 - gEven) : gHalfFilterSize - gEven;

    const int localId5 = (threadIdx.x/32)*32*8 +threadIdx.x%32+5*32;
    int outputRow5 = localId5/gOutputSize;
    int outputCol5 = localId5%gOutputSize;
    const int minu5 = gPadZeros ? max(-gHalfFilterSize, -outputRow5) : -gHalfFilterSize;
    const int maxu5 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow5  - gEven) : gHalfFilterSize - gEven;
    const int minv5 = gPadZeros ? max(-gHalfFilterSize, -outputCol5) : - gHalfFilterSize;
    const int maxv5 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol5 - gEven) : gHalfFilterSize - gEven;

    const int localId6 = (threadIdx.x/32)*32*8 +threadIdx.x%32+6*32;
    int outputRow6 = localId6/gOutputSize;
    int outputCol6 = localId6%gOutputSize;
    const int minu6 = gPadZeros ? max(-gHalfFilterSize, -outputRow6) : -gHalfFilterSize;
    const int maxu6 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow6  - gEven) : gHalfFilterSize - gEven;
    const int minv6 = gPadZeros ? max(-gHalfFilterSize, -outputCol6) : - gHalfFilterSize;
    const int maxv6 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol6 - gEven) : gHalfFilterSize - gEven;

    const int localId7 = (threadIdx.x/32)*32*8 +threadIdx.x%32+7*32;
    int outputRow7 = localId7/gOutputSize;
    int outputCol7 = localId7%gOutputSize;
    const int minu7 = gPadZeros ? max(-gHalfFilterSize, -outputRow7) : -gHalfFilterSize;
    const int maxu7 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow7  - gEven) : gHalfFilterSize - gEven;
    const int minv7 = gPadZeros ? max(-gHalfFilterSize, -outputCol7) : - gHalfFilterSize;
    const int maxv7 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol7 - gEven) : gHalfFilterSize - gEven;

    const int numUpstreamsPerThread = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;
    //const int filterCubeLength = gInputPlanes * gFilterSizeSquared;
    const int filterCubeGlobalOffset = outPlane * filterCubeLength;
    const int numPixelsPerThread = (filterCubeLength + workgroupSize - 1) / workgroupSize;
    for (int i = 0; i < numPixelsPerThread; i++) 
    {

        int thisOffset0 = localId0 + i * workgroupSize;
		if(thisOffset0 < filterCubeLength){
			_filterCube[thisOffset0] = filters[filterCubeGlobalOffset + thisOffset0];
			}

        int thisOffset1 = localId1 + i * workgroupSize;
		if(thisOffset1 < filterCubeLength){
			_filterCube[thisOffset1] = filters[filterCubeGlobalOffset + thisOffset1];
			}

        int thisOffset2 = localId2 + i * workgroupSize;
		if(thisOffset2 < filterCubeLength){
			_filterCube[thisOffset2] = filters[filterCubeGlobalOffset + thisOffset2];
			}

        int thisOffset3 = localId3 + i * workgroupSize;
		if(thisOffset3 < filterCubeLength){
			_filterCube[thisOffset3] = filters[filterCubeGlobalOffset + thisOffset3];
			}

        int thisOffset4 = localId4 + i * workgroupSize;
		if(thisOffset4 < filterCubeLength){
			_filterCube[thisOffset4] = filters[filterCubeGlobalOffset + thisOffset4];
			}

        int thisOffset5 = localId5 + i * workgroupSize;
		if(thisOffset5 < filterCubeLength){
			_filterCube[thisOffset5] = filters[filterCubeGlobalOffset + thisOffset5];
			}

        int thisOffset6 = localId6 + i * workgroupSize;
		if(thisOffset6 < filterCubeLength){
			_filterCube[thisOffset6] = filters[filterCubeGlobalOffset + thisOffset6];
			}

        int thisOffset7 = localId7 + i * workgroupSize;
		if(thisOffset7 < filterCubeLength){
			_filterCube[thisOffset7] = filters[filterCubeGlobalOffset + thisOffset7];
			}

    }

    float sum0 = 0;

    float sum1 = 0;

    float sum2 = 0;

    float sum3 = 0;

    float sum4 = 0;

    float sum5 = 0;

    float sum6 = 0;

    float sum7 = 0;


    for (int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++) {
        int thisUpstreamImageOffset = (n * gInputPlanes + upstreamPlane) * gInputSizeSquared;
        __syncthreads();
        for (int i = 0; i < numUpstreamsPerThread; i++) {
    
        int thisOffset0 = workgroupSize * i + localId0;
	    if (thisOffset0 < gInputSizeSquared){
		_upstreamImage[ thisOffset0 ] = images[ thisUpstreamImageOffset + thisOffset0 ];
		}
    
        int thisOffset1 = workgroupSize * i + localId1;
	    if (thisOffset1 < gInputSizeSquared){
		_upstreamImage[ thisOffset1 ] = images[ thisUpstreamImageOffset + thisOffset1 ];
		}
    
        int thisOffset2 = workgroupSize * i + localId2;
	    if (thisOffset2 < gInputSizeSquared){
		_upstreamImage[ thisOffset2 ] = images[ thisUpstreamImageOffset + thisOffset2 ];
		}
    
        int thisOffset3 = workgroupSize * i + localId3;
	    if (thisOffset3 < gInputSizeSquared){
		_upstreamImage[ thisOffset3 ] = images[ thisUpstreamImageOffset + thisOffset3 ];
		}
    
        int thisOffset4 = workgroupSize * i + localId4;
	    if (thisOffset4 < gInputSizeSquared){
		_upstreamImage[ thisOffset4 ] = images[ thisUpstreamImageOffset + thisOffset4 ];
		}
    
        int thisOffset5 = workgroupSize * i + localId5;
	    if (thisOffset5 < gInputSizeSquared){
		_upstreamImage[ thisOffset5 ] = images[ thisUpstreamImageOffset + thisOffset5 ];
		}
    
        int thisOffset6 = workgroupSize * i + localId6;
	    if (thisOffset6 < gInputSizeSquared){
		_upstreamImage[ thisOffset6 ] = images[ thisUpstreamImageOffset + thisOffset6 ];
		}
    
        int thisOffset7 = workgroupSize * i + localId7;
	    if (thisOffset7 < gInputSizeSquared){
		_upstreamImage[ thisOffset7 ] = images[ thisUpstreamImageOffset + thisOffset7 ];
		}
    
    }
        __syncthreads();
        int filterImageOffset = upstreamPlane * gFilterSizeSquared;
    
        for (int u = minu0; u <= maxu0; u++) {
            int inputRow = outputRow0 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv0; v <= maxv0; v++) {
                int inputCol = outputCol0 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId0 < gOutputSizeSquared) {
                   sum0 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu1; u <= maxu1; u++) {
            int inputRow = outputRow1 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv1; v <= maxv1; v++) {
                int inputCol = outputCol1 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId1 < gOutputSizeSquared) {
                   sum1 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu2; u <= maxu2; u++) {
            int inputRow = outputRow2 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv2; v <= maxv2; v++) {
                int inputCol = outputCol2 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId2 < gOutputSizeSquared) {
                   sum2 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu3; u <= maxu3; u++) {
            int inputRow = outputRow3 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv3; v <= maxv3; v++) {
                int inputCol = outputCol3 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId3 < gOutputSizeSquared) {
                   sum3 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu4; u <= maxu4; u++) {
            int inputRow = outputRow4 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv4; v <= maxv4; v++) {
                int inputCol = outputCol4 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId4 < gOutputSizeSquared) {
                   sum4 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu5; u <= maxu5; u++) {
            int inputRow = outputRow5 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv5; v <= maxv5; v++) {
                int inputCol = outputCol5 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId5 < gOutputSizeSquared) {
                   sum5 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu6; u <= maxu6; u++) {
            int inputRow = outputRow6 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv6; v <= maxv6; v++) {
                int inputCol = outputCol6 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId6 < gOutputSizeSquared) {
                   sum6 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu7; u <= maxu7; u++) {
            int inputRow = outputRow7 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv7; v <= maxv7; v++) {
                int inputCol = outputCol7 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId7 < gOutputSizeSquared) {
                   sum7 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
    }

    int resultIndex0 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId0;
    if (localId0 < gOutputSizeSquared){
	 output[resultIndex0] = sum0;
 	}

    int resultIndex1 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId1;
    if (localId1 < gOutputSizeSquared){
	 output[resultIndex1] = sum1;
 	}

    int resultIndex2 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId2;
    if (localId2 < gOutputSizeSquared){
	 output[resultIndex2] = sum2;
 	}

    int resultIndex3 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId3;
    if (localId3 < gOutputSizeSquared){
	 output[resultIndex3] = sum3;
 	}

    int resultIndex4 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId4;
    if (localId4 < gOutputSizeSquared){
	 output[resultIndex4] = sum4;
 	}

    int resultIndex5 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId5;
    if (localId5 < gOutputSizeSquared){
	 output[resultIndex5] = sum5;
 	}

    int resultIndex6 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId6;
    if (localId6 < gOutputSizeSquared){
	 output[resultIndex6] = sum6;
 	}

    int resultIndex7 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId7;
    if (localId7 < gOutputSizeSquared){
	 output[resultIndex7] = sum7;
 	}

}

  





























__global__ void coarsened_convolution_16C32S(const int batchSize, const float *images, 
const float *filters, float *output,  int gOutputSize, int gPadZeros, int gEven, int gOutputSizeSquared,
int gInputSize, int gInputPlanes, int gFilterSize, int gNumFilters)
{
    __shared__ float _filterCube[100]; // < filterCubeLength
    __shared__ float _upstreamImage[2000]; // < gInputSizeSquared

    int gFilterSizeSquared = (gFilterSize*gFilterSize);
    int gHalfFilterSize = (gFilterSize/2);
    int filterCubeLength =  (gInputPlanes*gFilterSizeSquared);
    int gInputSizeSquared = (gInputSize * gInputSize);

    const int globalId = blockDim.x*blockIdx.x+threadIdx.x;
    const int workgroupId = blockIdx.x;
    const int workgroupSize = 16*blockDim.x;
    const int n = workgroupId / gNumFilters;
    const int outPlane = workgroupId % gNumFilters;

    const int localId0 = (threadIdx.x/32)*32*16 +threadIdx.x%32+0*32;
    int outputRow0 = localId0/gOutputSize;
    int outputCol0 = localId0%gOutputSize;
    const int minu0 = gPadZeros ? max(-gHalfFilterSize, -outputRow0) : -gHalfFilterSize;
    const int maxu0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow0  - gEven) : gHalfFilterSize - gEven;
    const int minv0 = gPadZeros ? max(-gHalfFilterSize, -outputCol0) : - gHalfFilterSize;
    const int maxv0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol0 - gEven) : gHalfFilterSize - gEven;

    const int localId1 = (threadIdx.x/32)*32*16 +threadIdx.x%32+1*32;
    int outputRow1 = localId1/gOutputSize;
    int outputCol1 = localId1%gOutputSize;
    const int minu1 = gPadZeros ? max(-gHalfFilterSize, -outputRow1) : -gHalfFilterSize;
    const int maxu1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow1  - gEven) : gHalfFilterSize - gEven;
    const int minv1 = gPadZeros ? max(-gHalfFilterSize, -outputCol1) : - gHalfFilterSize;
    const int maxv1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol1 - gEven) : gHalfFilterSize - gEven;

    const int localId2 = (threadIdx.x/32)*32*16 +threadIdx.x%32+2*32;
    int outputRow2 = localId2/gOutputSize;
    int outputCol2 = localId2%gOutputSize;
    const int minu2 = gPadZeros ? max(-gHalfFilterSize, -outputRow2) : -gHalfFilterSize;
    const int maxu2 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow2  - gEven) : gHalfFilterSize - gEven;
    const int minv2 = gPadZeros ? max(-gHalfFilterSize, -outputCol2) : - gHalfFilterSize;
    const int maxv2 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol2 - gEven) : gHalfFilterSize - gEven;

    const int localId3 = (threadIdx.x/32)*32*16 +threadIdx.x%32+3*32;
    int outputRow3 = localId3/gOutputSize;
    int outputCol3 = localId3%gOutputSize;
    const int minu3 = gPadZeros ? max(-gHalfFilterSize, -outputRow3) : -gHalfFilterSize;
    const int maxu3 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow3  - gEven) : gHalfFilterSize - gEven;
    const int minv3 = gPadZeros ? max(-gHalfFilterSize, -outputCol3) : - gHalfFilterSize;
    const int maxv3 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol3 - gEven) : gHalfFilterSize - gEven;

    const int localId4 = (threadIdx.x/32)*32*16 +threadIdx.x%32+4*32;
    int outputRow4 = localId4/gOutputSize;
    int outputCol4 = localId4%gOutputSize;
    const int minu4 = gPadZeros ? max(-gHalfFilterSize, -outputRow4) : -gHalfFilterSize;
    const int maxu4 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow4  - gEven) : gHalfFilterSize - gEven;
    const int minv4 = gPadZeros ? max(-gHalfFilterSize, -outputCol4) : - gHalfFilterSize;
    const int maxv4 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol4 - gEven) : gHalfFilterSize - gEven;

    const int localId5 = (threadIdx.x/32)*32*16 +threadIdx.x%32+5*32;
    int outputRow5 = localId5/gOutputSize;
    int outputCol5 = localId5%gOutputSize;
    const int minu5 = gPadZeros ? max(-gHalfFilterSize, -outputRow5) : -gHalfFilterSize;
    const int maxu5 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow5  - gEven) : gHalfFilterSize - gEven;
    const int minv5 = gPadZeros ? max(-gHalfFilterSize, -outputCol5) : - gHalfFilterSize;
    const int maxv5 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol5 - gEven) : gHalfFilterSize - gEven;

    const int localId6 = (threadIdx.x/32)*32*16 +threadIdx.x%32+6*32;
    int outputRow6 = localId6/gOutputSize;
    int outputCol6 = localId6%gOutputSize;
    const int minu6 = gPadZeros ? max(-gHalfFilterSize, -outputRow6) : -gHalfFilterSize;
    const int maxu6 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow6  - gEven) : gHalfFilterSize - gEven;
    const int minv6 = gPadZeros ? max(-gHalfFilterSize, -outputCol6) : - gHalfFilterSize;
    const int maxv6 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol6 - gEven) : gHalfFilterSize - gEven;

    const int localId7 = (threadIdx.x/32)*32*16 +threadIdx.x%32+7*32;
    int outputRow7 = localId7/gOutputSize;
    int outputCol7 = localId7%gOutputSize;
    const int minu7 = gPadZeros ? max(-gHalfFilterSize, -outputRow7) : -gHalfFilterSize;
    const int maxu7 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow7  - gEven) : gHalfFilterSize - gEven;
    const int minv7 = gPadZeros ? max(-gHalfFilterSize, -outputCol7) : - gHalfFilterSize;
    const int maxv7 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol7 - gEven) : gHalfFilterSize - gEven;

    const int localId8 = (threadIdx.x/32)*32*16 +threadIdx.x%32+8*32;
    int outputRow8 = localId8/gOutputSize;
    int outputCol8 = localId8%gOutputSize;
    const int minu8 = gPadZeros ? max(-gHalfFilterSize, -outputRow8) : -gHalfFilterSize;
    const int maxu8 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow8  - gEven) : gHalfFilterSize - gEven;
    const int minv8 = gPadZeros ? max(-gHalfFilterSize, -outputCol8) : - gHalfFilterSize;
    const int maxv8 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol8 - gEven) : gHalfFilterSize - gEven;

    const int localId9 = (threadIdx.x/32)*32*16 +threadIdx.x%32+9*32;
    int outputRow9 = localId9/gOutputSize;
    int outputCol9 = localId9%gOutputSize;
    const int minu9 = gPadZeros ? max(-gHalfFilterSize, -outputRow9) : -gHalfFilterSize;
    const int maxu9 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow9  - gEven) : gHalfFilterSize - gEven;
    const int minv9 = gPadZeros ? max(-gHalfFilterSize, -outputCol9) : - gHalfFilterSize;
    const int maxv9 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol9 - gEven) : gHalfFilterSize - gEven;

    const int localId10 = (threadIdx.x/32)*32*16 +threadIdx.x%32+10*32;
    int outputRow10 = localId10/gOutputSize;
    int outputCol10 = localId10%gOutputSize;
    const int minu10 = gPadZeros ? max(-gHalfFilterSize, -outputRow10) : -gHalfFilterSize;
    const int maxu10 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow10  - gEven) : gHalfFilterSize - gEven;
    const int minv10 = gPadZeros ? max(-gHalfFilterSize, -outputCol10) : - gHalfFilterSize;
    const int maxv10 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol10 - gEven) : gHalfFilterSize - gEven;

    const int localId11 = (threadIdx.x/32)*32*16 +threadIdx.x%32+11*32;
    int outputRow11 = localId11/gOutputSize;
    int outputCol11 = localId11%gOutputSize;
    const int minu11 = gPadZeros ? max(-gHalfFilterSize, -outputRow11) : -gHalfFilterSize;
    const int maxu11 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow11  - gEven) : gHalfFilterSize - gEven;
    const int minv11 = gPadZeros ? max(-gHalfFilterSize, -outputCol11) : - gHalfFilterSize;
    const int maxv11 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol11 - gEven) : gHalfFilterSize - gEven;

    const int localId12 = (threadIdx.x/32)*32*16 +threadIdx.x%32+12*32;
    int outputRow12 = localId12/gOutputSize;
    int outputCol12 = localId12%gOutputSize;
    const int minu12 = gPadZeros ? max(-gHalfFilterSize, -outputRow12) : -gHalfFilterSize;
    const int maxu12 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow12  - gEven) : gHalfFilterSize - gEven;
    const int minv12 = gPadZeros ? max(-gHalfFilterSize, -outputCol12) : - gHalfFilterSize;
    const int maxv12 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol12 - gEven) : gHalfFilterSize - gEven;

    const int localId13 = (threadIdx.x/32)*32*16 +threadIdx.x%32+13*32;
    int outputRow13 = localId13/gOutputSize;
    int outputCol13 = localId13%gOutputSize;
    const int minu13 = gPadZeros ? max(-gHalfFilterSize, -outputRow13) : -gHalfFilterSize;
    const int maxu13 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow13  - gEven) : gHalfFilterSize - gEven;
    const int minv13 = gPadZeros ? max(-gHalfFilterSize, -outputCol13) : - gHalfFilterSize;
    const int maxv13 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol13 - gEven) : gHalfFilterSize - gEven;

    const int localId14 = (threadIdx.x/32)*32*16 +threadIdx.x%32+14*32;
    int outputRow14 = localId14/gOutputSize;
    int outputCol14 = localId14%gOutputSize;
    const int minu14 = gPadZeros ? max(-gHalfFilterSize, -outputRow14) : -gHalfFilterSize;
    const int maxu14 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow14  - gEven) : gHalfFilterSize - gEven;
    const int minv14 = gPadZeros ? max(-gHalfFilterSize, -outputCol14) : - gHalfFilterSize;
    const int maxv14 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol14 - gEven) : gHalfFilterSize - gEven;

    const int localId15 = (threadIdx.x/32)*32*16 +threadIdx.x%32+15*32;
    int outputRow15 = localId15/gOutputSize;
    int outputCol15 = localId15%gOutputSize;
    const int minu15 = gPadZeros ? max(-gHalfFilterSize, -outputRow15) : -gHalfFilterSize;
    const int maxu15 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow15  - gEven) : gHalfFilterSize - gEven;
    const int minv15 = gPadZeros ? max(-gHalfFilterSize, -outputCol15) : - gHalfFilterSize;
    const int maxv15 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol15 - gEven) : gHalfFilterSize - gEven;

    const int numUpstreamsPerThread = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;
    //const int filterCubeLength = gInputPlanes * gFilterSizeSquared;
    const int filterCubeGlobalOffset = outPlane * filterCubeLength;
    const int numPixelsPerThread = (filterCubeLength + workgroupSize - 1) / workgroupSize;
    for (int i = 0; i < numPixelsPerThread; i++) 
    {

        int thisOffset0 = localId0 + i * workgroupSize;
		if(thisOffset0 < filterCubeLength){
			_filterCube[thisOffset0] = filters[filterCubeGlobalOffset + thisOffset0];
			}

        int thisOffset1 = localId1 + i * workgroupSize;
		if(thisOffset1 < filterCubeLength){
			_filterCube[thisOffset1] = filters[filterCubeGlobalOffset + thisOffset1];
			}

        int thisOffset2 = localId2 + i * workgroupSize;
		if(thisOffset2 < filterCubeLength){
			_filterCube[thisOffset2] = filters[filterCubeGlobalOffset + thisOffset2];
			}

        int thisOffset3 = localId3 + i * workgroupSize;
		if(thisOffset3 < filterCubeLength){
			_filterCube[thisOffset3] = filters[filterCubeGlobalOffset + thisOffset3];
			}

        int thisOffset4 = localId4 + i * workgroupSize;
		if(thisOffset4 < filterCubeLength){
			_filterCube[thisOffset4] = filters[filterCubeGlobalOffset + thisOffset4];
			}

        int thisOffset5 = localId5 + i * workgroupSize;
		if(thisOffset5 < filterCubeLength){
			_filterCube[thisOffset5] = filters[filterCubeGlobalOffset + thisOffset5];
			}

        int thisOffset6 = localId6 + i * workgroupSize;
		if(thisOffset6 < filterCubeLength){
			_filterCube[thisOffset6] = filters[filterCubeGlobalOffset + thisOffset6];
			}

        int thisOffset7 = localId7 + i * workgroupSize;
		if(thisOffset7 < filterCubeLength){
			_filterCube[thisOffset7] = filters[filterCubeGlobalOffset + thisOffset7];
			}

        int thisOffset8 = localId8 + i * workgroupSize;
		if(thisOffset8 < filterCubeLength){
			_filterCube[thisOffset8] = filters[filterCubeGlobalOffset + thisOffset8];
			}

        int thisOffset9 = localId9 + i * workgroupSize;
		if(thisOffset9 < filterCubeLength){
			_filterCube[thisOffset9] = filters[filterCubeGlobalOffset + thisOffset9];
			}

        int thisOffset10 = localId10 + i * workgroupSize;
		if(thisOffset10 < filterCubeLength){
			_filterCube[thisOffset10] = filters[filterCubeGlobalOffset + thisOffset10];
			}

        int thisOffset11 = localId11 + i * workgroupSize;
		if(thisOffset11 < filterCubeLength){
			_filterCube[thisOffset11] = filters[filterCubeGlobalOffset + thisOffset11];
			}

        int thisOffset12 = localId12 + i * workgroupSize;
		if(thisOffset12 < filterCubeLength){
			_filterCube[thisOffset12] = filters[filterCubeGlobalOffset + thisOffset12];
			}

        int thisOffset13 = localId13 + i * workgroupSize;
		if(thisOffset13 < filterCubeLength){
			_filterCube[thisOffset13] = filters[filterCubeGlobalOffset + thisOffset13];
			}

        int thisOffset14 = localId14 + i * workgroupSize;
		if(thisOffset14 < filterCubeLength){
			_filterCube[thisOffset14] = filters[filterCubeGlobalOffset + thisOffset14];
			}

        int thisOffset15 = localId15 + i * workgroupSize;
		if(thisOffset15 < filterCubeLength){
			_filterCube[thisOffset15] = filters[filterCubeGlobalOffset + thisOffset15];
			}

    }

    float sum0 = 0;

    float sum1 = 0;

    float sum2 = 0;

    float sum3 = 0;

    float sum4 = 0;

    float sum5 = 0;

    float sum6 = 0;

    float sum7 = 0;

    float sum8 = 0;

    float sum9 = 0;

    float sum10 = 0;

    float sum11 = 0;

    float sum12 = 0;

    float sum13 = 0;

    float sum14 = 0;

    float sum15 = 0;


    for (int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++) {
        int thisUpstreamImageOffset = (n * gInputPlanes + upstreamPlane) * gInputSizeSquared;
        __syncthreads();
        for (int i = 0; i < numUpstreamsPerThread; i++) {
    
        int thisOffset0 = workgroupSize * i + localId0;
	    if (thisOffset0 < gInputSizeSquared){
		_upstreamImage[ thisOffset0 ] = images[ thisUpstreamImageOffset + thisOffset0 ];
		}
    
        int thisOffset1 = workgroupSize * i + localId1;
	    if (thisOffset1 < gInputSizeSquared){
		_upstreamImage[ thisOffset1 ] = images[ thisUpstreamImageOffset + thisOffset1 ];
		}
    
        int thisOffset2 = workgroupSize * i + localId2;
	    if (thisOffset2 < gInputSizeSquared){
		_upstreamImage[ thisOffset2 ] = images[ thisUpstreamImageOffset + thisOffset2 ];
		}
    
        int thisOffset3 = workgroupSize * i + localId3;
	    if (thisOffset3 < gInputSizeSquared){
		_upstreamImage[ thisOffset3 ] = images[ thisUpstreamImageOffset + thisOffset3 ];
		}
    
        int thisOffset4 = workgroupSize * i + localId4;
	    if (thisOffset4 < gInputSizeSquared){
		_upstreamImage[ thisOffset4 ] = images[ thisUpstreamImageOffset + thisOffset4 ];
		}
    
        int thisOffset5 = workgroupSize * i + localId5;
	    if (thisOffset5 < gInputSizeSquared){
		_upstreamImage[ thisOffset5 ] = images[ thisUpstreamImageOffset + thisOffset5 ];
		}
    
        int thisOffset6 = workgroupSize * i + localId6;
	    if (thisOffset6 < gInputSizeSquared){
		_upstreamImage[ thisOffset6 ] = images[ thisUpstreamImageOffset + thisOffset6 ];
		}
    
        int thisOffset7 = workgroupSize * i + localId7;
	    if (thisOffset7 < gInputSizeSquared){
		_upstreamImage[ thisOffset7 ] = images[ thisUpstreamImageOffset + thisOffset7 ];
		}
    
        int thisOffset8 = workgroupSize * i + localId8;
	    if (thisOffset8 < gInputSizeSquared){
		_upstreamImage[ thisOffset8 ] = images[ thisUpstreamImageOffset + thisOffset8 ];
		}
    
        int thisOffset9 = workgroupSize * i + localId9;
	    if (thisOffset9 < gInputSizeSquared){
		_upstreamImage[ thisOffset9 ] = images[ thisUpstreamImageOffset + thisOffset9 ];
		}
    
        int thisOffset10 = workgroupSize * i + localId10;
	    if (thisOffset10 < gInputSizeSquared){
		_upstreamImage[ thisOffset10 ] = images[ thisUpstreamImageOffset + thisOffset10 ];
		}
    
        int thisOffset11 = workgroupSize * i + localId11;
	    if (thisOffset11 < gInputSizeSquared){
		_upstreamImage[ thisOffset11 ] = images[ thisUpstreamImageOffset + thisOffset11 ];
		}
    
        int thisOffset12 = workgroupSize * i + localId12;
	    if (thisOffset12 < gInputSizeSquared){
		_upstreamImage[ thisOffset12 ] = images[ thisUpstreamImageOffset + thisOffset12 ];
		}
    
        int thisOffset13 = workgroupSize * i + localId13;
	    if (thisOffset13 < gInputSizeSquared){
		_upstreamImage[ thisOffset13 ] = images[ thisUpstreamImageOffset + thisOffset13 ];
		}
    
        int thisOffset14 = workgroupSize * i + localId14;
	    if (thisOffset14 < gInputSizeSquared){
		_upstreamImage[ thisOffset14 ] = images[ thisUpstreamImageOffset + thisOffset14 ];
		}
    
        int thisOffset15 = workgroupSize * i + localId15;
	    if (thisOffset15 < gInputSizeSquared){
		_upstreamImage[ thisOffset15 ] = images[ thisUpstreamImageOffset + thisOffset15 ];
		}
    
    }
        __syncthreads();
        int filterImageOffset = upstreamPlane * gFilterSizeSquared;
    
        for (int u = minu0; u <= maxu0; u++) {
            int inputRow = outputRow0 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv0; v <= maxv0; v++) {
                int inputCol = outputCol0 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId0 < gOutputSizeSquared) {
                   sum0 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu1; u <= maxu1; u++) {
            int inputRow = outputRow1 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv1; v <= maxv1; v++) {
                int inputCol = outputCol1 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId1 < gOutputSizeSquared) {
                   sum1 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu2; u <= maxu2; u++) {
            int inputRow = outputRow2 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv2; v <= maxv2; v++) {
                int inputCol = outputCol2 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId2 < gOutputSizeSquared) {
                   sum2 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu3; u <= maxu3; u++) {
            int inputRow = outputRow3 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv3; v <= maxv3; v++) {
                int inputCol = outputCol3 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId3 < gOutputSizeSquared) {
                   sum3 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu4; u <= maxu4; u++) {
            int inputRow = outputRow4 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv4; v <= maxv4; v++) {
                int inputCol = outputCol4 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId4 < gOutputSizeSquared) {
                   sum4 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu5; u <= maxu5; u++) {
            int inputRow = outputRow5 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv5; v <= maxv5; v++) {
                int inputCol = outputCol5 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId5 < gOutputSizeSquared) {
                   sum5 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu6; u <= maxu6; u++) {
            int inputRow = outputRow6 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv6; v <= maxv6; v++) {
                int inputCol = outputCol6 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId6 < gOutputSizeSquared) {
                   sum6 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu7; u <= maxu7; u++) {
            int inputRow = outputRow7 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv7; v <= maxv7; v++) {
                int inputCol = outputCol7 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId7 < gOutputSizeSquared) {
                   sum7 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu8; u <= maxu8; u++) {
            int inputRow = outputRow8 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv8; v <= maxv8; v++) {
                int inputCol = outputCol8 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId8 < gOutputSizeSquared) {
                   sum8 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu9; u <= maxu9; u++) {
            int inputRow = outputRow9 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv9; v <= maxv9; v++) {
                int inputCol = outputCol9 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId9 < gOutputSizeSquared) {
                   sum9 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu10; u <= maxu10; u++) {
            int inputRow = outputRow10 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv10; v <= maxv10; v++) {
                int inputCol = outputCol10 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId10 < gOutputSizeSquared) {
                   sum10 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu11; u <= maxu11; u++) {
            int inputRow = outputRow11 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv11; v <= maxv11; v++) {
                int inputCol = outputCol11 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId11 < gOutputSizeSquared) {
                   sum11 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu12; u <= maxu12; u++) {
            int inputRow = outputRow12 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv12; v <= maxv12; v++) {
                int inputCol = outputCol12 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId12 < gOutputSizeSquared) {
                   sum12 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu13; u <= maxu13; u++) {
            int inputRow = outputRow13 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv13; v <= maxv13; v++) {
                int inputCol = outputCol13 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId13 < gOutputSizeSquared) {
                   sum13 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu14; u <= maxu14; u++) {
            int inputRow = outputRow14 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv14; v <= maxv14; v++) {
                int inputCol = outputCol14 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId14 < gOutputSizeSquared) {
                   sum14 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu15; u <= maxu15; u++) {
            int inputRow = outputRow15 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv15; v <= maxv15; v++) {
                int inputCol = outputCol15 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId15 < gOutputSizeSquared) {
                   sum15 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
    }

    int resultIndex0 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId0;
    if (localId0 < gOutputSizeSquared){
	 output[resultIndex0] = sum0;
 	}

    int resultIndex1 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId1;
    if (localId1 < gOutputSizeSquared){
	 output[resultIndex1] = sum1;
 	}

    int resultIndex2 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId2;
    if (localId2 < gOutputSizeSquared){
	 output[resultIndex2] = sum2;
 	}

    int resultIndex3 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId3;
    if (localId3 < gOutputSizeSquared){
	 output[resultIndex3] = sum3;
 	}

    int resultIndex4 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId4;
    if (localId4 < gOutputSizeSquared){
	 output[resultIndex4] = sum4;
 	}

    int resultIndex5 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId5;
    if (localId5 < gOutputSizeSquared){
	 output[resultIndex5] = sum5;
 	}

    int resultIndex6 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId6;
    if (localId6 < gOutputSizeSquared){
	 output[resultIndex6] = sum6;
 	}

    int resultIndex7 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId7;
    if (localId7 < gOutputSizeSquared){
	 output[resultIndex7] = sum7;
 	}

    int resultIndex8 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId8;
    if (localId8 < gOutputSizeSquared){
	 output[resultIndex8] = sum8;
 	}

    int resultIndex9 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId9;
    if (localId9 < gOutputSizeSquared){
	 output[resultIndex9] = sum9;
 	}

    int resultIndex10 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId10;
    if (localId10 < gOutputSizeSquared){
	 output[resultIndex10] = sum10;
 	}

    int resultIndex11 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId11;
    if (localId11 < gOutputSizeSquared){
	 output[resultIndex11] = sum11;
 	}

    int resultIndex12 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId12;
    if (localId12 < gOutputSizeSquared){
	 output[resultIndex12] = sum12;
 	}

    int resultIndex13 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId13;
    if (localId13 < gOutputSizeSquared){
	 output[resultIndex13] = sum13;
 	}

    int resultIndex14 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId14;
    if (localId14 < gOutputSizeSquared){
	 output[resultIndex14] = sum14;
 	}

    int resultIndex15 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId15;
    if (localId15 < gOutputSizeSquared){
	 output[resultIndex15] = sum15;
 	}

}

  
__global__ void coarsened_convolution_32C32S(const int batchSize, const float *images, 
const float *filters, float *output,  int gOutputSize, int gPadZeros, int gEven, int gOutputSizeSquared,
int gInputSize, int gInputPlanes, int gFilterSize, int gNumFilters)
{
    __shared__ float _filterCube[100];
    __shared__ float _upstreamImage[2000];

    int gFilterSizeSquared = (gFilterSize*gFilterSize);
    int gHalfFilterSize = (gFilterSize/2);
    int filterCubeLength =  (gInputPlanes*gFilterSizeSquared);
    int gInputSizeSquared = (gInputSize * gInputSize);


    const int globalId = blockDim.x*blockIdx.x+threadIdx.x;
    const int workgroupId = blockIdx.x;
    const int workgroupSize = 32*blockDim.x;
    const int n = workgroupId / gNumFilters;
    const int outPlane = workgroupId % gNumFilters;

    const int localId0 = (threadIdx.x/32)*32*32 +threadIdx.x%32+0*32;
    int outputRow0 = localId0/gOutputSize;
    int outputCol0 = localId0%gOutputSize;
    const int minu0 = gPadZeros ? max(-gHalfFilterSize, -outputRow0) : -gHalfFilterSize;
    const int maxu0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow0  - gEven) : gHalfFilterSize - gEven;
    const int minv0 = gPadZeros ? max(-gHalfFilterSize, -outputCol0) : - gHalfFilterSize;
    const int maxv0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol0 - gEven) : gHalfFilterSize - gEven;

    const int localId1 = (threadIdx.x/32)*32*32 +threadIdx.x%32+1*32;
    int outputRow1 = localId1/gOutputSize;
    int outputCol1 = localId1%gOutputSize;
    const int minu1 = gPadZeros ? max(-gHalfFilterSize, -outputRow1) : -gHalfFilterSize;
    const int maxu1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow1  - gEven) : gHalfFilterSize - gEven;
    const int minv1 = gPadZeros ? max(-gHalfFilterSize, -outputCol1) : - gHalfFilterSize;
    const int maxv1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol1 - gEven) : gHalfFilterSize - gEven;

    const int localId2 = (threadIdx.x/32)*32*32 +threadIdx.x%32+2*32;
    int outputRow2 = localId2/gOutputSize;
    int outputCol2 = localId2%gOutputSize;
    const int minu2 = gPadZeros ? max(-gHalfFilterSize, -outputRow2) : -gHalfFilterSize;
    const int maxu2 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow2  - gEven) : gHalfFilterSize - gEven;
    const int minv2 = gPadZeros ? max(-gHalfFilterSize, -outputCol2) : - gHalfFilterSize;
    const int maxv2 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol2 - gEven) : gHalfFilterSize - gEven;

    const int localId3 = (threadIdx.x/32)*32*32 +threadIdx.x%32+3*32;
    int outputRow3 = localId3/gOutputSize;
    int outputCol3 = localId3%gOutputSize;
    const int minu3 = gPadZeros ? max(-gHalfFilterSize, -outputRow3) : -gHalfFilterSize;
    const int maxu3 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow3  - gEven) : gHalfFilterSize - gEven;
    const int minv3 = gPadZeros ? max(-gHalfFilterSize, -outputCol3) : - gHalfFilterSize;
    const int maxv3 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol3 - gEven) : gHalfFilterSize - gEven;

    const int localId4 = (threadIdx.x/32)*32*32 +threadIdx.x%32+4*32;
    int outputRow4 = localId4/gOutputSize;
    int outputCol4 = localId4%gOutputSize;
    const int minu4 = gPadZeros ? max(-gHalfFilterSize, -outputRow4) : -gHalfFilterSize;
    const int maxu4 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow4  - gEven) : gHalfFilterSize - gEven;
    const int minv4 = gPadZeros ? max(-gHalfFilterSize, -outputCol4) : - gHalfFilterSize;
    const int maxv4 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol4 - gEven) : gHalfFilterSize - gEven;

    const int localId5 = (threadIdx.x/32)*32*32 +threadIdx.x%32+5*32;
    int outputRow5 = localId5/gOutputSize;
    int outputCol5 = localId5%gOutputSize;
    const int minu5 = gPadZeros ? max(-gHalfFilterSize, -outputRow5) : -gHalfFilterSize;
    const int maxu5 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow5  - gEven) : gHalfFilterSize - gEven;
    const int minv5 = gPadZeros ? max(-gHalfFilterSize, -outputCol5) : - gHalfFilterSize;
    const int maxv5 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol5 - gEven) : gHalfFilterSize - gEven;

    const int localId6 = (threadIdx.x/32)*32*32 +threadIdx.x%32+6*32;
    int outputRow6 = localId6/gOutputSize;
    int outputCol6 = localId6%gOutputSize;
    const int minu6 = gPadZeros ? max(-gHalfFilterSize, -outputRow6) : -gHalfFilterSize;
    const int maxu6 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow6  - gEven) : gHalfFilterSize - gEven;
    const int minv6 = gPadZeros ? max(-gHalfFilterSize, -outputCol6) : - gHalfFilterSize;
    const int maxv6 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol6 - gEven) : gHalfFilterSize - gEven;

    const int localId7 = (threadIdx.x/32)*32*32 +threadIdx.x%32+7*32;
    int outputRow7 = localId7/gOutputSize;
    int outputCol7 = localId7%gOutputSize;
    const int minu7 = gPadZeros ? max(-gHalfFilterSize, -outputRow7) : -gHalfFilterSize;
    const int maxu7 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow7  - gEven) : gHalfFilterSize - gEven;
    const int minv7 = gPadZeros ? max(-gHalfFilterSize, -outputCol7) : - gHalfFilterSize;
    const int maxv7 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol7 - gEven) : gHalfFilterSize - gEven;

    const int localId8 = (threadIdx.x/32)*32*32 +threadIdx.x%32+8*32;
    int outputRow8 = localId8/gOutputSize;
    int outputCol8 = localId8%gOutputSize;
    const int minu8 = gPadZeros ? max(-gHalfFilterSize, -outputRow8) : -gHalfFilterSize;
    const int maxu8 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow8  - gEven) : gHalfFilterSize - gEven;
    const int minv8 = gPadZeros ? max(-gHalfFilterSize, -outputCol8) : - gHalfFilterSize;
    const int maxv8 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol8 - gEven) : gHalfFilterSize - gEven;

    const int localId9 = (threadIdx.x/32)*32*32 +threadIdx.x%32+9*32;
    int outputRow9 = localId9/gOutputSize;
    int outputCol9 = localId9%gOutputSize;
    const int minu9 = gPadZeros ? max(-gHalfFilterSize, -outputRow9) : -gHalfFilterSize;
    const int maxu9 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow9  - gEven) : gHalfFilterSize - gEven;
    const int minv9 = gPadZeros ? max(-gHalfFilterSize, -outputCol9) : - gHalfFilterSize;
    const int maxv9 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol9 - gEven) : gHalfFilterSize - gEven;

    const int localId10 = (threadIdx.x/32)*32*32 +threadIdx.x%32+10*32;
    int outputRow10 = localId10/gOutputSize;
    int outputCol10 = localId10%gOutputSize;
    const int minu10 = gPadZeros ? max(-gHalfFilterSize, -outputRow10) : -gHalfFilterSize;
    const int maxu10 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow10  - gEven) : gHalfFilterSize - gEven;
    const int minv10 = gPadZeros ? max(-gHalfFilterSize, -outputCol10) : - gHalfFilterSize;
    const int maxv10 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol10 - gEven) : gHalfFilterSize - gEven;

    const int localId11 = (threadIdx.x/32)*32*32 +threadIdx.x%32+11*32;
    int outputRow11 = localId11/gOutputSize;
    int outputCol11 = localId11%gOutputSize;
    const int minu11 = gPadZeros ? max(-gHalfFilterSize, -outputRow11) : -gHalfFilterSize;
    const int maxu11 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow11  - gEven) : gHalfFilterSize - gEven;
    const int minv11 = gPadZeros ? max(-gHalfFilterSize, -outputCol11) : - gHalfFilterSize;
    const int maxv11 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol11 - gEven) : gHalfFilterSize - gEven;

    const int localId12 = (threadIdx.x/32)*32*32 +threadIdx.x%32+12*32;
    int outputRow12 = localId12/gOutputSize;
    int outputCol12 = localId12%gOutputSize;
    const int minu12 = gPadZeros ? max(-gHalfFilterSize, -outputRow12) : -gHalfFilterSize;
    const int maxu12 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow12  - gEven) : gHalfFilterSize - gEven;
    const int minv12 = gPadZeros ? max(-gHalfFilterSize, -outputCol12) : - gHalfFilterSize;
    const int maxv12 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol12 - gEven) : gHalfFilterSize - gEven;

    const int localId13 = (threadIdx.x/32)*32*32 +threadIdx.x%32+13*32;
    int outputRow13 = localId13/gOutputSize;
    int outputCol13 = localId13%gOutputSize;
    const int minu13 = gPadZeros ? max(-gHalfFilterSize, -outputRow13) : -gHalfFilterSize;
    const int maxu13 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow13  - gEven) : gHalfFilterSize - gEven;
    const int minv13 = gPadZeros ? max(-gHalfFilterSize, -outputCol13) : - gHalfFilterSize;
    const int maxv13 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol13 - gEven) : gHalfFilterSize - gEven;

    const int localId14 = (threadIdx.x/32)*32*32 +threadIdx.x%32+14*32;
    int outputRow14 = localId14/gOutputSize;
    int outputCol14 = localId14%gOutputSize;
    const int minu14 = gPadZeros ? max(-gHalfFilterSize, -outputRow14) : -gHalfFilterSize;
    const int maxu14 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow14  - gEven) : gHalfFilterSize - gEven;
    const int minv14 = gPadZeros ? max(-gHalfFilterSize, -outputCol14) : - gHalfFilterSize;
    const int maxv14 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol14 - gEven) : gHalfFilterSize - gEven;

    const int localId15 = (threadIdx.x/32)*32*32 +threadIdx.x%32+15*32;
    int outputRow15 = localId15/gOutputSize;
    int outputCol15 = localId15%gOutputSize;
    const int minu15 = gPadZeros ? max(-gHalfFilterSize, -outputRow15) : -gHalfFilterSize;
    const int maxu15 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow15  - gEven) : gHalfFilterSize - gEven;
    const int minv15 = gPadZeros ? max(-gHalfFilterSize, -outputCol15) : - gHalfFilterSize;
    const int maxv15 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol15 - gEven) : gHalfFilterSize - gEven;

    const int localId16 = (threadIdx.x/32)*32*32 +threadIdx.x%32+16*32;
    int outputRow16 = localId16/gOutputSize;
    int outputCol16 = localId16%gOutputSize;
    const int minu16 = gPadZeros ? max(-gHalfFilterSize, -outputRow16) : -gHalfFilterSize;
    const int maxu16 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow16  - gEven) : gHalfFilterSize - gEven;
    const int minv16 = gPadZeros ? max(-gHalfFilterSize, -outputCol16) : - gHalfFilterSize;
    const int maxv16 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol16 - gEven) : gHalfFilterSize - gEven;

    const int localId17 = (threadIdx.x/32)*32*32 +threadIdx.x%32+17*32;
    int outputRow17 = localId17/gOutputSize;
    int outputCol17 = localId17%gOutputSize;
    const int minu17 = gPadZeros ? max(-gHalfFilterSize, -outputRow17) : -gHalfFilterSize;
    const int maxu17 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow17  - gEven) : gHalfFilterSize - gEven;
    const int minv17 = gPadZeros ? max(-gHalfFilterSize, -outputCol17) : - gHalfFilterSize;
    const int maxv17 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol17 - gEven) : gHalfFilterSize - gEven;

    const int localId18 = (threadIdx.x/32)*32*32 +threadIdx.x%32+18*32;
    int outputRow18 = localId18/gOutputSize;
    int outputCol18 = localId18%gOutputSize;
    const int minu18 = gPadZeros ? max(-gHalfFilterSize, -outputRow18) : -gHalfFilterSize;
    const int maxu18 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow18  - gEven) : gHalfFilterSize - gEven;
    const int minv18 = gPadZeros ? max(-gHalfFilterSize, -outputCol18) : - gHalfFilterSize;
    const int maxv18 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol18 - gEven) : gHalfFilterSize - gEven;

    const int localId19 = (threadIdx.x/32)*32*32 +threadIdx.x%32+19*32;
    int outputRow19 = localId19/gOutputSize;
    int outputCol19 = localId19%gOutputSize;
    const int minu19 = gPadZeros ? max(-gHalfFilterSize, -outputRow19) : -gHalfFilterSize;
    const int maxu19 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow19  - gEven) : gHalfFilterSize - gEven;
    const int minv19 = gPadZeros ? max(-gHalfFilterSize, -outputCol19) : - gHalfFilterSize;
    const int maxv19 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol19 - gEven) : gHalfFilterSize - gEven;

    const int localId20 = (threadIdx.x/32)*32*32 +threadIdx.x%32+20*32;
    int outputRow20 = localId20/gOutputSize;
    int outputCol20 = localId20%gOutputSize;
    const int minu20 = gPadZeros ? max(-gHalfFilterSize, -outputRow20) : -gHalfFilterSize;
    const int maxu20 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow20  - gEven) : gHalfFilterSize - gEven;
    const int minv20 = gPadZeros ? max(-gHalfFilterSize, -outputCol20) : - gHalfFilterSize;
    const int maxv20 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol20 - gEven) : gHalfFilterSize - gEven;

    const int localId21 = (threadIdx.x/32)*32*32 +threadIdx.x%32+21*32;
    int outputRow21 = localId21/gOutputSize;
    int outputCol21 = localId21%gOutputSize;
    const int minu21 = gPadZeros ? max(-gHalfFilterSize, -outputRow21) : -gHalfFilterSize;
    const int maxu21 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow21  - gEven) : gHalfFilterSize - gEven;
    const int minv21 = gPadZeros ? max(-gHalfFilterSize, -outputCol21) : - gHalfFilterSize;
    const int maxv21 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol21 - gEven) : gHalfFilterSize - gEven;

    const int localId22 = (threadIdx.x/32)*32*32 +threadIdx.x%32+22*32;
    int outputRow22 = localId22/gOutputSize;
    int outputCol22 = localId22%gOutputSize;
    const int minu22 = gPadZeros ? max(-gHalfFilterSize, -outputRow22) : -gHalfFilterSize;
    const int maxu22 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow22  - gEven) : gHalfFilterSize - gEven;
    const int minv22 = gPadZeros ? max(-gHalfFilterSize, -outputCol22) : - gHalfFilterSize;
    const int maxv22 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol22 - gEven) : gHalfFilterSize - gEven;

    const int localId23 = (threadIdx.x/32)*32*32 +threadIdx.x%32+23*32;
    int outputRow23 = localId23/gOutputSize;
    int outputCol23 = localId23%gOutputSize;
    const int minu23 = gPadZeros ? max(-gHalfFilterSize, -outputRow23) : -gHalfFilterSize;
    const int maxu23 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow23  - gEven) : gHalfFilterSize - gEven;
    const int minv23 = gPadZeros ? max(-gHalfFilterSize, -outputCol23) : - gHalfFilterSize;
    const int maxv23 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol23 - gEven) : gHalfFilterSize - gEven;

    const int localId24 = (threadIdx.x/32)*32*32 +threadIdx.x%32+24*32;
    int outputRow24 = localId24/gOutputSize;
    int outputCol24 = localId24%gOutputSize;
    const int minu24 = gPadZeros ? max(-gHalfFilterSize, -outputRow24) : -gHalfFilterSize;
    const int maxu24 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow24  - gEven) : gHalfFilterSize - gEven;
    const int minv24 = gPadZeros ? max(-gHalfFilterSize, -outputCol24) : - gHalfFilterSize;
    const int maxv24 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol24 - gEven) : gHalfFilterSize - gEven;

    const int localId25 = (threadIdx.x/32)*32*32 +threadIdx.x%32+25*32;
    int outputRow25 = localId25/gOutputSize;
    int outputCol25 = localId25%gOutputSize;
    const int minu25 = gPadZeros ? max(-gHalfFilterSize, -outputRow25) : -gHalfFilterSize;
    const int maxu25 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow25  - gEven) : gHalfFilterSize - gEven;
    const int minv25 = gPadZeros ? max(-gHalfFilterSize, -outputCol25) : - gHalfFilterSize;
    const int maxv25 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol25 - gEven) : gHalfFilterSize - gEven;

    const int localId26 = (threadIdx.x/32)*32*32 +threadIdx.x%32+26*32;
    int outputRow26 = localId26/gOutputSize;
    int outputCol26 = localId26%gOutputSize;
    const int minu26 = gPadZeros ? max(-gHalfFilterSize, -outputRow26) : -gHalfFilterSize;
    const int maxu26 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow26  - gEven) : gHalfFilterSize - gEven;
    const int minv26 = gPadZeros ? max(-gHalfFilterSize, -outputCol26) : - gHalfFilterSize;
    const int maxv26 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol26 - gEven) : gHalfFilterSize - gEven;

    const int localId27 = (threadIdx.x/32)*32*32 +threadIdx.x%32+27*32;
    int outputRow27 = localId27/gOutputSize;
    int outputCol27 = localId27%gOutputSize;
    const int minu27 = gPadZeros ? max(-gHalfFilterSize, -outputRow27) : -gHalfFilterSize;
    const int maxu27 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow27  - gEven) : gHalfFilterSize - gEven;
    const int minv27 = gPadZeros ? max(-gHalfFilterSize, -outputCol27) : - gHalfFilterSize;
    const int maxv27 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol27 - gEven) : gHalfFilterSize - gEven;

    const int localId28 = (threadIdx.x/32)*32*32 +threadIdx.x%32+28*32;
    int outputRow28 = localId28/gOutputSize;
    int outputCol28 = localId28%gOutputSize;
    const int minu28 = gPadZeros ? max(-gHalfFilterSize, -outputRow28) : -gHalfFilterSize;
    const int maxu28 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow28  - gEven) : gHalfFilterSize - gEven;
    const int minv28 = gPadZeros ? max(-gHalfFilterSize, -outputCol28) : - gHalfFilterSize;
    const int maxv28 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol28 - gEven) : gHalfFilterSize - gEven;

    const int localId29 = (threadIdx.x/32)*32*32 +threadIdx.x%32+29*32;
    int outputRow29 = localId29/gOutputSize;
    int outputCol29 = localId29%gOutputSize;
    const int minu29 = gPadZeros ? max(-gHalfFilterSize, -outputRow29) : -gHalfFilterSize;
    const int maxu29 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow29  - gEven) : gHalfFilterSize - gEven;
    const int minv29 = gPadZeros ? max(-gHalfFilterSize, -outputCol29) : - gHalfFilterSize;
    const int maxv29 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol29 - gEven) : gHalfFilterSize - gEven;

    const int localId30 = (threadIdx.x/32)*32*32 +threadIdx.x%32+30*32;
    int outputRow30 = localId30/gOutputSize;
    int outputCol30 = localId30%gOutputSize;
    const int minu30 = gPadZeros ? max(-gHalfFilterSize, -outputRow30) : -gHalfFilterSize;
    const int maxu30 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow30  - gEven) : gHalfFilterSize - gEven;
    const int minv30 = gPadZeros ? max(-gHalfFilterSize, -outputCol30) : - gHalfFilterSize;
    const int maxv30 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol30 - gEven) : gHalfFilterSize - gEven;

    const int localId31 = (threadIdx.x/32)*32*32 +threadIdx.x%32+31*32;
    int outputRow31 = localId31/gOutputSize;
    int outputCol31 = localId31%gOutputSize;
    const int minu31 = gPadZeros ? max(-gHalfFilterSize, -outputRow31) : -gHalfFilterSize;
    const int maxu31 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow31  - gEven) : gHalfFilterSize - gEven;
    const int minv31 = gPadZeros ? max(-gHalfFilterSize, -outputCol31) : - gHalfFilterSize;
    const int maxv31 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol31 - gEven) : gHalfFilterSize - gEven;

    const int numUpstreamsPerThread = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;
    //const int filterCubeLength = gInputPlanes * gFilterSizeSquared;
    const int filterCubeGlobalOffset = outPlane * filterCubeLength;
    const int numPixelsPerThread = (filterCubeLength + workgroupSize - 1) / workgroupSize;
    for (int i = 0; i < numPixelsPerThread; i++) 
    {

        int thisOffset0 = localId0 + i * workgroupSize;
		if(thisOffset0 < filterCubeLength){
			_filterCube[thisOffset0] = filters[filterCubeGlobalOffset + thisOffset0];
			}

        int thisOffset1 = localId1 + i * workgroupSize;
		if(thisOffset1 < filterCubeLength){
			_filterCube[thisOffset1] = filters[filterCubeGlobalOffset + thisOffset1];
			}

        int thisOffset2 = localId2 + i * workgroupSize;
		if(thisOffset2 < filterCubeLength){
			_filterCube[thisOffset2] = filters[filterCubeGlobalOffset + thisOffset2];
			}

        int thisOffset3 = localId3 + i * workgroupSize;
		if(thisOffset3 < filterCubeLength){
			_filterCube[thisOffset3] = filters[filterCubeGlobalOffset + thisOffset3];
			}

        int thisOffset4 = localId4 + i * workgroupSize;
		if(thisOffset4 < filterCubeLength){
			_filterCube[thisOffset4] = filters[filterCubeGlobalOffset + thisOffset4];
			}

        int thisOffset5 = localId5 + i * workgroupSize;
		if(thisOffset5 < filterCubeLength){
			_filterCube[thisOffset5] = filters[filterCubeGlobalOffset + thisOffset5];
			}

        int thisOffset6 = localId6 + i * workgroupSize;
		if(thisOffset6 < filterCubeLength){
			_filterCube[thisOffset6] = filters[filterCubeGlobalOffset + thisOffset6];
			}

        int thisOffset7 = localId7 + i * workgroupSize;
		if(thisOffset7 < filterCubeLength){
			_filterCube[thisOffset7] = filters[filterCubeGlobalOffset + thisOffset7];
			}

        int thisOffset8 = localId8 + i * workgroupSize;
		if(thisOffset8 < filterCubeLength){
			_filterCube[thisOffset8] = filters[filterCubeGlobalOffset + thisOffset8];
			}

        int thisOffset9 = localId9 + i * workgroupSize;
		if(thisOffset9 < filterCubeLength){
			_filterCube[thisOffset9] = filters[filterCubeGlobalOffset + thisOffset9];
			}

        int thisOffset10 = localId10 + i * workgroupSize;
		if(thisOffset10 < filterCubeLength){
			_filterCube[thisOffset10] = filters[filterCubeGlobalOffset + thisOffset10];
			}

        int thisOffset11 = localId11 + i * workgroupSize;
		if(thisOffset11 < filterCubeLength){
			_filterCube[thisOffset11] = filters[filterCubeGlobalOffset + thisOffset11];
			}

        int thisOffset12 = localId12 + i * workgroupSize;
		if(thisOffset12 < filterCubeLength){
			_filterCube[thisOffset12] = filters[filterCubeGlobalOffset + thisOffset12];
			}

        int thisOffset13 = localId13 + i * workgroupSize;
		if(thisOffset13 < filterCubeLength){
			_filterCube[thisOffset13] = filters[filterCubeGlobalOffset + thisOffset13];
			}

        int thisOffset14 = localId14 + i * workgroupSize;
		if(thisOffset14 < filterCubeLength){
			_filterCube[thisOffset14] = filters[filterCubeGlobalOffset + thisOffset14];
			}

        int thisOffset15 = localId15 + i * workgroupSize;
		if(thisOffset15 < filterCubeLength){
			_filterCube[thisOffset15] = filters[filterCubeGlobalOffset + thisOffset15];
			}

        int thisOffset16 = localId16 + i * workgroupSize;
		if(thisOffset16 < filterCubeLength){
			_filterCube[thisOffset16] = filters[filterCubeGlobalOffset + thisOffset16];
			}

        int thisOffset17 = localId17 + i * workgroupSize;
		if(thisOffset17 < filterCubeLength){
			_filterCube[thisOffset17] = filters[filterCubeGlobalOffset + thisOffset17];
			}

        int thisOffset18 = localId18 + i * workgroupSize;
		if(thisOffset18 < filterCubeLength){
			_filterCube[thisOffset18] = filters[filterCubeGlobalOffset + thisOffset18];
			}

        int thisOffset19 = localId19 + i * workgroupSize;
		if(thisOffset19 < filterCubeLength){
			_filterCube[thisOffset19] = filters[filterCubeGlobalOffset + thisOffset19];
			}

        int thisOffset20 = localId20 + i * workgroupSize;
		if(thisOffset20 < filterCubeLength){
			_filterCube[thisOffset20] = filters[filterCubeGlobalOffset + thisOffset20];
			}

        int thisOffset21 = localId21 + i * workgroupSize;
		if(thisOffset21 < filterCubeLength){
			_filterCube[thisOffset21] = filters[filterCubeGlobalOffset + thisOffset21];
			}

        int thisOffset22 = localId22 + i * workgroupSize;
		if(thisOffset22 < filterCubeLength){
			_filterCube[thisOffset22] = filters[filterCubeGlobalOffset + thisOffset22];
			}

        int thisOffset23 = localId23 + i * workgroupSize;
		if(thisOffset23 < filterCubeLength){
			_filterCube[thisOffset23] = filters[filterCubeGlobalOffset + thisOffset23];
			}

        int thisOffset24 = localId24 + i * workgroupSize;
		if(thisOffset24 < filterCubeLength){
			_filterCube[thisOffset24] = filters[filterCubeGlobalOffset + thisOffset24];
			}

        int thisOffset25 = localId25 + i * workgroupSize;
		if(thisOffset25 < filterCubeLength){
			_filterCube[thisOffset25] = filters[filterCubeGlobalOffset + thisOffset25];
			}

        int thisOffset26 = localId26 + i * workgroupSize;
		if(thisOffset26 < filterCubeLength){
			_filterCube[thisOffset26] = filters[filterCubeGlobalOffset + thisOffset26];
			}

        int thisOffset27 = localId27 + i * workgroupSize;
		if(thisOffset27 < filterCubeLength){
			_filterCube[thisOffset27] = filters[filterCubeGlobalOffset + thisOffset27];
			}

        int thisOffset28 = localId28 + i * workgroupSize;
		if(thisOffset28 < filterCubeLength){
			_filterCube[thisOffset28] = filters[filterCubeGlobalOffset + thisOffset28];
			}

        int thisOffset29 = localId29 + i * workgroupSize;
		if(thisOffset29 < filterCubeLength){
			_filterCube[thisOffset29] = filters[filterCubeGlobalOffset + thisOffset29];
			}

        int thisOffset30 = localId30 + i * workgroupSize;
		if(thisOffset30 < filterCubeLength){
			_filterCube[thisOffset30] = filters[filterCubeGlobalOffset + thisOffset30];
			}

        int thisOffset31 = localId31 + i * workgroupSize;
		if(thisOffset31 < filterCubeLength){
			_filterCube[thisOffset31] = filters[filterCubeGlobalOffset + thisOffset31];
			}

    }

    float sum0 = 0;

    float sum1 = 0;

    float sum2 = 0;

    float sum3 = 0;

    float sum4 = 0;

    float sum5 = 0;

    float sum6 = 0;

    float sum7 = 0;

    float sum8 = 0;

    float sum9 = 0;

    float sum10 = 0;

    float sum11 = 0;

    float sum12 = 0;

    float sum13 = 0;

    float sum14 = 0;

    float sum15 = 0;

    float sum16 = 0;

    float sum17 = 0;

    float sum18 = 0;

    float sum19 = 0;

    float sum20 = 0;

    float sum21 = 0;

    float sum22 = 0;

    float sum23 = 0;

    float sum24 = 0;

    float sum25 = 0;

    float sum26 = 0;

    float sum27 = 0;

    float sum28 = 0;

    float sum29 = 0;

    float sum30 = 0;

    float sum31 = 0;


    for (int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++) {
        int thisUpstreamImageOffset = (n * gInputPlanes + upstreamPlane) * gInputSizeSquared;
        __syncthreads();
        for (int i = 0; i < numUpstreamsPerThread; i++) {
    
        int thisOffset0 = workgroupSize * i + localId0;
	    if (thisOffset0 < gInputSizeSquared){
		_upstreamImage[ thisOffset0 ] = images[ thisUpstreamImageOffset + thisOffset0 ];
		}
    
        int thisOffset1 = workgroupSize * i + localId1;
	    if (thisOffset1 < gInputSizeSquared){
		_upstreamImage[ thisOffset1 ] = images[ thisUpstreamImageOffset + thisOffset1 ];
		}
    
        int thisOffset2 = workgroupSize * i + localId2;
	    if (thisOffset2 < gInputSizeSquared){
		_upstreamImage[ thisOffset2 ] = images[ thisUpstreamImageOffset + thisOffset2 ];
		}
    
        int thisOffset3 = workgroupSize * i + localId3;
	    if (thisOffset3 < gInputSizeSquared){
		_upstreamImage[ thisOffset3 ] = images[ thisUpstreamImageOffset + thisOffset3 ];
		}
    
        int thisOffset4 = workgroupSize * i + localId4;
	    if (thisOffset4 < gInputSizeSquared){
		_upstreamImage[ thisOffset4 ] = images[ thisUpstreamImageOffset + thisOffset4 ];
		}
    
        int thisOffset5 = workgroupSize * i + localId5;
	    if (thisOffset5 < gInputSizeSquared){
		_upstreamImage[ thisOffset5 ] = images[ thisUpstreamImageOffset + thisOffset5 ];
		}
    
        int thisOffset6 = workgroupSize * i + localId6;
	    if (thisOffset6 < gInputSizeSquared){
		_upstreamImage[ thisOffset6 ] = images[ thisUpstreamImageOffset + thisOffset6 ];
		}
    
        int thisOffset7 = workgroupSize * i + localId7;
	    if (thisOffset7 < gInputSizeSquared){
		_upstreamImage[ thisOffset7 ] = images[ thisUpstreamImageOffset + thisOffset7 ];
		}
    
        int thisOffset8 = workgroupSize * i + localId8;
	    if (thisOffset8 < gInputSizeSquared){
		_upstreamImage[ thisOffset8 ] = images[ thisUpstreamImageOffset + thisOffset8 ];
		}
    
        int thisOffset9 = workgroupSize * i + localId9;
	    if (thisOffset9 < gInputSizeSquared){
		_upstreamImage[ thisOffset9 ] = images[ thisUpstreamImageOffset + thisOffset9 ];
		}
    
        int thisOffset10 = workgroupSize * i + localId10;
	    if (thisOffset10 < gInputSizeSquared){
		_upstreamImage[ thisOffset10 ] = images[ thisUpstreamImageOffset + thisOffset10 ];
		}
    
        int thisOffset11 = workgroupSize * i + localId11;
	    if (thisOffset11 < gInputSizeSquared){
		_upstreamImage[ thisOffset11 ] = images[ thisUpstreamImageOffset + thisOffset11 ];
		}
    
        int thisOffset12 = workgroupSize * i + localId12;
	    if (thisOffset12 < gInputSizeSquared){
		_upstreamImage[ thisOffset12 ] = images[ thisUpstreamImageOffset + thisOffset12 ];
		}
    
        int thisOffset13 = workgroupSize * i + localId13;
	    if (thisOffset13 < gInputSizeSquared){
		_upstreamImage[ thisOffset13 ] = images[ thisUpstreamImageOffset + thisOffset13 ];
		}
    
        int thisOffset14 = workgroupSize * i + localId14;
	    if (thisOffset14 < gInputSizeSquared){
		_upstreamImage[ thisOffset14 ] = images[ thisUpstreamImageOffset + thisOffset14 ];
		}
    
        int thisOffset15 = workgroupSize * i + localId15;
	    if (thisOffset15 < gInputSizeSquared){
		_upstreamImage[ thisOffset15 ] = images[ thisUpstreamImageOffset + thisOffset15 ];
		}
    
        int thisOffset16 = workgroupSize * i + localId16;
	    if (thisOffset16 < gInputSizeSquared){
		_upstreamImage[ thisOffset16 ] = images[ thisUpstreamImageOffset + thisOffset16 ];
		}
    
        int thisOffset17 = workgroupSize * i + localId17;
	    if (thisOffset17 < gInputSizeSquared){
		_upstreamImage[ thisOffset17 ] = images[ thisUpstreamImageOffset + thisOffset17 ];
		}
    
        int thisOffset18 = workgroupSize * i + localId18;
	    if (thisOffset18 < gInputSizeSquared){
		_upstreamImage[ thisOffset18 ] = images[ thisUpstreamImageOffset + thisOffset18 ];
		}
    
        int thisOffset19 = workgroupSize * i + localId19;
	    if (thisOffset19 < gInputSizeSquared){
		_upstreamImage[ thisOffset19 ] = images[ thisUpstreamImageOffset + thisOffset19 ];
		}
    
        int thisOffset20 = workgroupSize * i + localId20;
	    if (thisOffset20 < gInputSizeSquared){
		_upstreamImage[ thisOffset20 ] = images[ thisUpstreamImageOffset + thisOffset20 ];
		}
    
        int thisOffset21 = workgroupSize * i + localId21;
	    if (thisOffset21 < gInputSizeSquared){
		_upstreamImage[ thisOffset21 ] = images[ thisUpstreamImageOffset + thisOffset21 ];
		}
    
        int thisOffset22 = workgroupSize * i + localId22;
	    if (thisOffset22 < gInputSizeSquared){
		_upstreamImage[ thisOffset22 ] = images[ thisUpstreamImageOffset + thisOffset22 ];
		}
    
        int thisOffset23 = workgroupSize * i + localId23;
	    if (thisOffset23 < gInputSizeSquared){
		_upstreamImage[ thisOffset23 ] = images[ thisUpstreamImageOffset + thisOffset23 ];
		}
    
        int thisOffset24 = workgroupSize * i + localId24;
	    if (thisOffset24 < gInputSizeSquared){
		_upstreamImage[ thisOffset24 ] = images[ thisUpstreamImageOffset + thisOffset24 ];
		}
    
        int thisOffset25 = workgroupSize * i + localId25;
	    if (thisOffset25 < gInputSizeSquared){
		_upstreamImage[ thisOffset25 ] = images[ thisUpstreamImageOffset + thisOffset25 ];
		}
    
        int thisOffset26 = workgroupSize * i + localId26;
	    if (thisOffset26 < gInputSizeSquared){
		_upstreamImage[ thisOffset26 ] = images[ thisUpstreamImageOffset + thisOffset26 ];
		}
    
        int thisOffset27 = workgroupSize * i + localId27;
	    if (thisOffset27 < gInputSizeSquared){
		_upstreamImage[ thisOffset27 ] = images[ thisUpstreamImageOffset + thisOffset27 ];
		}
    
        int thisOffset28 = workgroupSize * i + localId28;
	    if (thisOffset28 < gInputSizeSquared){
		_upstreamImage[ thisOffset28 ] = images[ thisUpstreamImageOffset + thisOffset28 ];
		}
    
        int thisOffset29 = workgroupSize * i + localId29;
	    if (thisOffset29 < gInputSizeSquared){
		_upstreamImage[ thisOffset29 ] = images[ thisUpstreamImageOffset + thisOffset29 ];
		}
    
        int thisOffset30 = workgroupSize * i + localId30;
	    if (thisOffset30 < gInputSizeSquared){
		_upstreamImage[ thisOffset30 ] = images[ thisUpstreamImageOffset + thisOffset30 ];
		}
    
        int thisOffset31 = workgroupSize * i + localId31;
	    if (thisOffset31 < gInputSizeSquared){
		_upstreamImage[ thisOffset31 ] = images[ thisUpstreamImageOffset + thisOffset31 ];
		}
    
    }
        __syncthreads();
        int filterImageOffset = upstreamPlane * gFilterSizeSquared;
    
        for (int u = minu0; u <= maxu0; u++) {
            int inputRow = outputRow0 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv0; v <= maxv0; v++) {
                int inputCol = outputCol0 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId0 < gOutputSizeSquared) {
                   sum0 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu1; u <= maxu1; u++) {
            int inputRow = outputRow1 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv1; v <= maxv1; v++) {
                int inputCol = outputCol1 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId1 < gOutputSizeSquared) {
                   sum1 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu2; u <= maxu2; u++) {
            int inputRow = outputRow2 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv2; v <= maxv2; v++) {
                int inputCol = outputCol2 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId2 < gOutputSizeSquared) {
                   sum2 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu3; u <= maxu3; u++) {
            int inputRow = outputRow3 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv3; v <= maxv3; v++) {
                int inputCol = outputCol3 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId3 < gOutputSizeSquared) {
                   sum3 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu4; u <= maxu4; u++) {
            int inputRow = outputRow4 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv4; v <= maxv4; v++) {
                int inputCol = outputCol4 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId4 < gOutputSizeSquared) {
                   sum4 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu5; u <= maxu5; u++) {
            int inputRow = outputRow5 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv5; v <= maxv5; v++) {
                int inputCol = outputCol5 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId5 < gOutputSizeSquared) {
                   sum5 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu6; u <= maxu6; u++) {
            int inputRow = outputRow6 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv6; v <= maxv6; v++) {
                int inputCol = outputCol6 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId6 < gOutputSizeSquared) {
                   sum6 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu7; u <= maxu7; u++) {
            int inputRow = outputRow7 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv7; v <= maxv7; v++) {
                int inputCol = outputCol7 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId7 < gOutputSizeSquared) {
                   sum7 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu8; u <= maxu8; u++) {
            int inputRow = outputRow8 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv8; v <= maxv8; v++) {
                int inputCol = outputCol8 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId8 < gOutputSizeSquared) {
                   sum8 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu9; u <= maxu9; u++) {
            int inputRow = outputRow9 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv9; v <= maxv9; v++) {
                int inputCol = outputCol9 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId9 < gOutputSizeSquared) {
                   sum9 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu10; u <= maxu10; u++) {
            int inputRow = outputRow10 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv10; v <= maxv10; v++) {
                int inputCol = outputCol10 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId10 < gOutputSizeSquared) {
                   sum10 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu11; u <= maxu11; u++) {
            int inputRow = outputRow11 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv11; v <= maxv11; v++) {
                int inputCol = outputCol11 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId11 < gOutputSizeSquared) {
                   sum11 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu12; u <= maxu12; u++) {
            int inputRow = outputRow12 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv12; v <= maxv12; v++) {
                int inputCol = outputCol12 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId12 < gOutputSizeSquared) {
                   sum12 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu13; u <= maxu13; u++) {
            int inputRow = outputRow13 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv13; v <= maxv13; v++) {
                int inputCol = outputCol13 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId13 < gOutputSizeSquared) {
                   sum13 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu14; u <= maxu14; u++) {
            int inputRow = outputRow14 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv14; v <= maxv14; v++) {
                int inputCol = outputCol14 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId14 < gOutputSizeSquared) {
                   sum14 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu15; u <= maxu15; u++) {
            int inputRow = outputRow15 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv15; v <= maxv15; v++) {
                int inputCol = outputCol15 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId15 < gOutputSizeSquared) {
                   sum15 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu16; u <= maxu16; u++) {
            int inputRow = outputRow16 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv16; v <= maxv16; v++) {
                int inputCol = outputCol16 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId16 < gOutputSizeSquared) {
                   sum16 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu17; u <= maxu17; u++) {
            int inputRow = outputRow17 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv17; v <= maxv17; v++) {
                int inputCol = outputCol17 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId17 < gOutputSizeSquared) {
                   sum17 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu18; u <= maxu18; u++) {
            int inputRow = outputRow18 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv18; v <= maxv18; v++) {
                int inputCol = outputCol18 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId18 < gOutputSizeSquared) {
                   sum18 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu19; u <= maxu19; u++) {
            int inputRow = outputRow19 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv19; v <= maxv19; v++) {
                int inputCol = outputCol19 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId19 < gOutputSizeSquared) {
                   sum19 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu20; u <= maxu20; u++) {
            int inputRow = outputRow20 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv20; v <= maxv20; v++) {
                int inputCol = outputCol20 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId20 < gOutputSizeSquared) {
                   sum20 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu21; u <= maxu21; u++) {
            int inputRow = outputRow21 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv21; v <= maxv21; v++) {
                int inputCol = outputCol21 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId21 < gOutputSizeSquared) {
                   sum21 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu22; u <= maxu22; u++) {
            int inputRow = outputRow22 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv22; v <= maxv22; v++) {
                int inputCol = outputCol22 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId22 < gOutputSizeSquared) {
                   sum22 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu23; u <= maxu23; u++) {
            int inputRow = outputRow23 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv23; v <= maxv23; v++) {
                int inputCol = outputCol23 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId23 < gOutputSizeSquared) {
                   sum23 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu24; u <= maxu24; u++) {
            int inputRow = outputRow24 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv24; v <= maxv24; v++) {
                int inputCol = outputCol24 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId24 < gOutputSizeSquared) {
                   sum24 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu25; u <= maxu25; u++) {
            int inputRow = outputRow25 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv25; v <= maxv25; v++) {
                int inputCol = outputCol25 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId25 < gOutputSizeSquared) {
                   sum25 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu26; u <= maxu26; u++) {
            int inputRow = outputRow26 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv26; v <= maxv26; v++) {
                int inputCol = outputCol26 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId26 < gOutputSizeSquared) {
                   sum26 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu27; u <= maxu27; u++) {
            int inputRow = outputRow27 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv27; v <= maxv27; v++) {
                int inputCol = outputCol27 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId27 < gOutputSizeSquared) {
                   sum27 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu28; u <= maxu28; u++) {
            int inputRow = outputRow28 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv28; v <= maxv28; v++) {
                int inputCol = outputCol28 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId28 < gOutputSizeSquared) {
                   sum28 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu29; u <= maxu29; u++) {
            int inputRow = outputRow29 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv29; v <= maxv29; v++) {
                int inputCol = outputCol29 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId29 < gOutputSizeSquared) {
                   sum29 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu30; u <= maxu30; u++) {
            int inputRow = outputRow30 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv30; v <= maxv30; v++) {
                int inputCol = outputCol30 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId30 < gOutputSizeSquared) {
                   sum30 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu31; u <= maxu31; u++) {
            int inputRow = outputRow31 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv31; v <= maxv31; v++) {
                int inputCol = outputCol31 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId31 < gOutputSizeSquared) {
                   sum31 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
    }

    int resultIndex0 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId0;
    if (localId0 < gOutputSizeSquared){
	 output[resultIndex0] = sum0;
 	}

    int resultIndex1 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId1;
    if (localId1 < gOutputSizeSquared){
	 output[resultIndex1] = sum1;
 	}

    int resultIndex2 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId2;
    if (localId2 < gOutputSizeSquared){
	 output[resultIndex2] = sum2;
 	}

    int resultIndex3 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId3;
    if (localId3 < gOutputSizeSquared){
	 output[resultIndex3] = sum3;
 	}

    int resultIndex4 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId4;
    if (localId4 < gOutputSizeSquared){
	 output[resultIndex4] = sum4;
 	}

    int resultIndex5 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId5;
    if (localId5 < gOutputSizeSquared){
	 output[resultIndex5] = sum5;
 	}

    int resultIndex6 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId6;
    if (localId6 < gOutputSizeSquared){
	 output[resultIndex6] = sum6;
 	}

    int resultIndex7 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId7;
    if (localId7 < gOutputSizeSquared){
	 output[resultIndex7] = sum7;
 	}

    int resultIndex8 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId8;
    if (localId8 < gOutputSizeSquared){
	 output[resultIndex8] = sum8;
 	}

    int resultIndex9 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId9;
    if (localId9 < gOutputSizeSquared){
	 output[resultIndex9] = sum9;
 	}

    int resultIndex10 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId10;
    if (localId10 < gOutputSizeSquared){
	 output[resultIndex10] = sum10;
 	}

    int resultIndex11 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId11;
    if (localId11 < gOutputSizeSquared){
	 output[resultIndex11] = sum11;
 	}

    int resultIndex12 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId12;
    if (localId12 < gOutputSizeSquared){
	 output[resultIndex12] = sum12;
 	}

    int resultIndex13 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId13;
    if (localId13 < gOutputSizeSquared){
	 output[resultIndex13] = sum13;
 	}

    int resultIndex14 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId14;
    if (localId14 < gOutputSizeSquared){
	 output[resultIndex14] = sum14;
 	}

    int resultIndex15 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId15;
    if (localId15 < gOutputSizeSquared){
	 output[resultIndex15] = sum15;
 	}

    int resultIndex16 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId16;
    if (localId16 < gOutputSizeSquared){
	 output[resultIndex16] = sum16;
 	}

    int resultIndex17 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId17;
    if (localId17 < gOutputSizeSquared){
	 output[resultIndex17] = sum17;
 	}

    int resultIndex18 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId18;
    if (localId18 < gOutputSizeSquared){
	 output[resultIndex18] = sum18;
 	}

    int resultIndex19 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId19;
    if (localId19 < gOutputSizeSquared){
	 output[resultIndex19] = sum19;
 	}

    int resultIndex20 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId20;
    if (localId20 < gOutputSizeSquared){
	 output[resultIndex20] = sum20;
 	}

    int resultIndex21 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId21;
    if (localId21 < gOutputSizeSquared){
	 output[resultIndex21] = sum21;
 	}

    int resultIndex22 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId22;
    if (localId22 < gOutputSizeSquared){
	 output[resultIndex22] = sum22;
 	}

    int resultIndex23 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId23;
    if (localId23 < gOutputSizeSquared){
	 output[resultIndex23] = sum23;
 	}

    int resultIndex24 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId24;
    if (localId24 < gOutputSizeSquared){
	 output[resultIndex24] = sum24;
 	}

    int resultIndex25 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId25;
    if (localId25 < gOutputSizeSquared){
	 output[resultIndex25] = sum25;
 	}

    int resultIndex26 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId26;
    if (localId26 < gOutputSizeSquared){
	 output[resultIndex26] = sum26;
 	}

    int resultIndex27 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId27;
    if (localId27 < gOutputSizeSquared){
	 output[resultIndex27] = sum27;
 	}

    int resultIndex28 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId28;
    if (localId28 < gOutputSizeSquared){
	 output[resultIndex28] = sum28;
 	}

    int resultIndex29 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId29;
    if (localId29 < gOutputSizeSquared){
	 output[resultIndex29] = sum29;
 	}

    int resultIndex30 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId30;
    if (localId30 < gOutputSizeSquared){
	 output[resultIndex30] = sum30;
 	}

    int resultIndex31 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId31;
    if (localId31 < gOutputSizeSquared){
	 output[resultIndex31] = sum31;
 	}

}

