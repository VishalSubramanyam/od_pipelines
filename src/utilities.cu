#include <coarsened_forward_convolution.h>
#include <utilities.h>
#define gpuErrchk(ans)                                                         \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d \n", cudaGetErrorString(code),
                file, line);
        if (abort)
            exit(code);
    }
}

void error(const char *s) {
    perror(s);
    assert(0);
    exit(-1);
}

dim3 cuda_gridsize(int n) {
    unsigned int k = (n - 1) / BLOCK1 + 1;
    unsigned int x = k;
    unsigned int y = 1;
    if (x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * BLOCK1) + 1;
    }
    dim3 d = {x, y, 1};
    // printf("n=%d x=%d y=%d x*y*BLOCK1=%d\n", n, x, y, x*y*BLOCK1);
    return d;
}

void check_error(cudaError_t status) {
    // cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess) {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    }
    if (status2 != cudaSuccess) {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    }
}

// Pooling kernels--start
__global__ void forward_avgpool_layer_kernel(int n, int w, int h, int c,
                                             float *input, float *output) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n)
        return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c * b);
    output[out_index] = 0;
    for (i = 0; i < w * h; ++i) {
        int in_index = i + h * w * (k + b * c);
        output[out_index] += input[in_index];
    }
    output[out_index] /= w * h;
}

void forward_avgpool_layer_gpu(int w, int h, int c, float *input,
                               float *output) {
    size_t n = c;
    // size_t n = layer.c*layer.batch;

    forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK1>>>(n, w, h, c,
                                                               input, output);
    check_error(cudaPeekAtLastError());
}

__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w,
                                             int in_c, int stride, int size,
                                             int pad, float *input,
                                             float *output) {
    int h = (in_h + pad - size) / stride + 1;
    int w = (in_w + pad - size) / stride + 1;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n)
        return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad / 2;
    int h_offset = -pad / 2;

    int out_index = j + w * (i + h * (k + c * b));
    float max = -INFINITY;
    int max_i = -1;
    int l, m;
    for (l = 0; l < size; ++l) {
        for (m = 0; m < size; ++m) {
            int cur_h = h_offset + i * stride + l;
            int cur_w = w_offset + j * stride + m;
            int index = cur_w + in_w * (cur_h + in_h * (k + b * in_c));
            int valid =
                (cur_h >= 0 && cur_h < in_h && cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            max_i = (val > max) ? index : max_i;
            max = (val > max) ? val : max;
        }
    }
    output[out_index] = max;
}

void forward_maxpool_layer_gpu(int h, int w, int c, int stride, int size,
                               int pad, float *input, float *output) {
    int h1 = (h + pad - size) / stride + 1; // layer.out_h;
    int w1 = (w + pad - size) / stride + 1; // layer.out_w;
    // int c = layer.c;

    size_t n = h1 * w1 * c;

    forward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK1>>>(
        n, h, w, c, stride, size, pad, input, output);
    check_error(cudaPeekAtLastError());
}

// for softmax function
__device__ void softmax_device(float *input, int n, float temp, int stride,
                               float *output) {
    int i;
    float sum = 0;
    float largest = -INFINITY;
    for (i = 0; i < n; ++i) {
        int val = input[i * stride];
        largest = (val > largest) ? val : largest;
    }
    for (i = 0; i < n; ++i) {
        float e = expf(input[i * stride] / temp - largest / temp);
        sum += e;
        output[i * stride] = e;
    }
    for (i = 0; i < n; ++i) {
        output[i * stride] /= sum;
    }
}

__global__ void softmax_kernel(float *input, int n, int batch, int batch_offset,
                               int groups, int group_offset, int stride,
                               float temp, float *output) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch * groups)
        return;
    int b = id / groups;
    int g = id % groups;
    softmax_device(input + b * batch_offset + g * group_offset, n, temp, stride,
                   output + b * batch_offset + g * group_offset);
}

void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups,
                 int group_offset, int stride, float temp, float *output,
                 cudaStream_t stream) {
    softmax_kernel<<<cuda_gridsize(batch * groups), BLOCK1, 0, stream>>>(
        input, n, batch, batch_offset, groups, group_offset, stride, temp,
        output);
    check_error(cudaPeekAtLastError());
}
// for softmax function -- end

__device__ float linear_activate_kernel(float x) { return x; }
__device__ float logistic_activate_kernel(float x) {
    return 1.f / (1.f + expf(-x));
}

__device__ float activate_kernel(float x, ACTIVATION a) {
    switch (a) {
    case LINEAR:
        return linear_activate_kernel(x);
    case LOGISTIC:
        return logistic_activate_kernel(x);
    }
    return 0;
}

__device__ float leaky_activate_kernel(float x) {
    return (x > 0) ? x : .1f * x;
}

__global__ void activate_array_kernel(float *x, int n) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = leaky_activate_kernel(x[i]);
    //	if (i<n){ if (x>0) x[i]= x[i] ;
    //                 else x[i]=.1f*x[i];
    //}
}

__global__ void normalize_kernel(int N, float *x, float *mean, float *variance,
                                 int batch, int filters, int spatial) {
    int index =
        (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N)
        return;
    int f = (index / spatial) % filters;

    x[index] = (x[index] - mean[f]) / (sqrtf(variance[f] + .00001f));
}

__global__ void scale_bias_kernel(float *output, float *biases, int n,
                                  int size) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if (offset < size)
        output[(batch * n + filter) * size + offset] *= biases[filter];
}

__global__ void add_bias_kernel(float *output, float *biases, int batch, int n,
                                int size) {
    int index =
        (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n * size * batch)
        return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k * n + j) * size + i] += biases[j];
}

__global__ void copy_kernel(int N, float *X, int OFFX, int INCX, float *Y,
                            int OFFY, int INCY) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N)
        Y[i * INCY + OFFY] = X[i * INCX + OFFX];
}

void copy_gpu_offset(int N, float *X, int OFFX, int INCX, float *Y, int OFFY,
                     int INCY, cudaStream_t stream) {
    copy_kernel<<<cuda_gridsize(N), BLOCK1, 0, stream>>>(N, X, OFFX, INCX, Y,
                                                         OFFY, INCY);
    gpuErrchk(cudaPeekAtLastError());
}

void copy_gpu(int N, float *X, int INCX, float *Y, int INCY,
              cudaStream_t stream) {
    copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY, stream);
}

void normalize_gpu(float *x, float *mean, float *variance, int batch,
                   int filters, int spatial, cudaStream_t stream) {
    size_t N = batch * filters * spatial;
    normalize_kernel<<<cuda_gridsize(N), BLOCK1, 0, stream>>>(
        N, x, mean, variance, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}

void add_bias_gpu(float *output, float *biases, int batch, int n, int size,
                  cudaStream_t stream) {
    int num = n * size * batch;

    add_bias_kernel<<<cuda_gridsize(num), BLOCK1, 0, stream>>>(output, biases,
                                                               batch, n, size);
    check_error(cudaPeekAtLastError());
}

void scale_bias_gpu(float *output, float *biases, int batch, int n, int size,
                    cudaStream_t stream) {
    dim3 dimGrid((size - 1) / BLOCK1 + 1, n, batch);
    dim3 dimBlock(BLOCK1, 1, 1);

    scale_bias_kernel<<<dimGrid, dimBlock, 0, stream>>>(output, biases, n,
                                                        size);
    check_error(cudaPeekAtLastError());
}

__global__ void activate_array_kernel1(float *x, int n, ACTIVATION a) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = activate_kernel(x[i], a);
}

void activate_array_gpu1(float *x, int n, ACTIVATION a, cudaStream_t stream) {
    activate_array_kernel1<<<cuda_gridsize(n), BLOCK1, 0, stream>>>(x, n, a);
    // check_error(cudaPeekAtLastError());
    gpuErrchk(cudaPeekAtLastError());
}

void activate_array_gpu(float *x, int n, cudaStream_t stream) {
    dim3 kk = cuda_gridsize(n);
    activate_array_kernel<<<kk, BLOCK1, 0, stream>>>(x, n);
    check_error(cudaPeekAtLastError());
}

int entry_index(int batch, int location, int entry, int w, int h, int outputs,
                int coords, int classes) {
    int n = location / (w * h);
    int loc = location % (w * h);
    // printf("batch %d n  %d loc %d entry %d coords %d classes %d\n", batch,
    // n,loc,entry, coords,classes);
    return batch * outputs + n * w * h * (coords + classes + 1) +
           entry * w * h + loc;
}
// Region layer processing
void forward_region_layer_gpu(int inputs, int outputs, float *input, int batch,
                              int h, int w, int nm, int classes, int coords,
                              float *output, cudaStream_t cs) {
    int background = 0;
    int softmax = 1;
    int softmax_tree = 0;
    int b, n;
    // if(output == NULL)  printf("memory not allocated to output array");
    // printf("batch %d inputs %d outputs%d w %d h %d coords %d classes %d nm %d
    // \n",batch, inputs,outputs,w,h, coords, classes,nm);
    copy_gpu(batch * inputs, input, 1, output, 1, cs);
    for (b = 0; b < batch; ++b) {
        for (n = 0; n < nm; ++n) {
            int index =
                entry_index(b, n * w * h, 0, w, h, outputs, coords, classes);
            // printf("\n first index is %d ",index);
            activate_array_gpu1(output + index, 2 * w * h, LOGISTIC, cs);
            if (coords > 4) {
                index = entry_index(b, n * w * h, 4, w, h, outputs, coords,
                                    classes);
                // printf("index is %d ",index);
                activate_array_gpu1(output + index, (coords - 4) * w * h,
                                    LOGISTIC, cs);
            }
            index = entry_index(b, n * w * h, coords, w, h, outputs, coords,
                                classes);
            // printf("\nindex is %d ",index);
            if (!background)
                activate_array_gpu1(output + index, w * h, LOGISTIC, cs);
            index = entry_index(b, n * w * h, coords + 1, w, h, outputs, coords,
                                classes);
            // printf("\nindex is %d ",index);
            if (!softmax && !softmax_tree)
                activate_array_gpu1(output + index, classes * w * h, LOGISTIC,
                                    cs);
        }
    }
    if (softmax) {
        int index = entry_index(0, 0, coords + !background, w, h, outputs,
                                coords, classes);
        // printf(" softmax index is %d ",index);
        softmax_gpu(input + index, classes + background, batch * nm,
                    inputs / nm, w * h, 1, w * h, 1, output + index, cs);
    }
    // cuda_pull_array(output, input, batch*inputs); //check this...
    // forward_region_layer(l, net);
}

void free_detections(detection *dets, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        free(dets[i].prob);
        if (dets[i].mask)
            free(dets[i].mask);
    }
    free(dets);
}

void correct_region_boxes(detection *dets, int n, int w, int h, int netw,
                          int neth, int relative) {
    int i;
    int new_w = 0;
    int new_h = 0;
    if (((float)netw / w) < ((float)neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    } else {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i) {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j,
                   int w, int h, int stride) {
    box b;
    b.x = (i + x[index + 0 * stride]) / w;
    b.y = (j + x[index + 1 * stride]) / h;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

void get_region_detections(RegionLayerParams *p, int w, int h, float thresh,
                           int relative, detection *dets, float *output,
                           int outputs, int netw, int neth) {
    int i, j, n, background = 0;
    float biases[] = {1.08,  1.19, 3.42, 4.41,  6.63,
                      11.38, 9.42, 5.11, 16.62, 10.52};
    float *predictions = output;
    // printf("Prediction %f", predictions[0]);
    if (predictions == NULL)
        printf("Prediction NULL error");
    for (i = 0; i < p->width * p->height; ++i) {
        int row = i / p->width;
        int col = i % p->width;
        for (n = 0; n < p->num; ++n) {
            int index = n * p->width * p->height + i;
            for (j = 0; j < p->classes; ++j) {
                dets[index].prob[j] = 0;
            }
            int obj_index = entry_index(0, n * p->width * p->height + i,
                                        p->coords, p->width, p->height, outputs,
                                        p->coords, p->classes);
            int box_index =
                entry_index(0, n * p->width * p->height + i, 0, p->width,
                            p->height, outputs, p->coords, p->classes);
            int mask_index =
                entry_index(0, n * p->width * p->height + i, 4, p->width,
                            p->height, outputs, p->coords, p->classes);
            // printf("OBJ index is %d",obj_index);
            float scale = background ? 1 : predictions[obj_index];
            dets[index].bbox =
                get_region_box(predictions, biases, n, box_index, col, row,
                               p->width, p->height, p->width * p->width);
            dets[index].objectness = scale > thresh ? scale : 0;
            if (dets[index].mask) {
                for (j = 0; j < p->coords - 4; ++j) {
                    dets[index].mask[j] =
                        output[mask_index + j * p->width * p->height];
                }
            }

            int class_index = entry_index(
                0, n * p->width * p->height + i, p->coords + !background,
                p->width, p->height, outputs, p->coords, p->classes);
            ;
            if (dets[index].objectness) {
                for (j = 0; j < p->classes; ++j) {
                    int class_index = entry_index(
                        0, n * p->width * p->height + i, p->coords + 1 + j,
                        p->width, p->height, outputs, p->coords, p->classes);
                    ;
                    float prob = scale * predictions[class_index];
                    dets[index].prob[j] = (prob > thresh) ? prob : 0;
                }
            }
        }
    }
    correct_region_boxes(dets, p->width * p->height * p->num, w, h, netw, neth,
                         relative);
}

void fill_network_boxes(RegionLayerParams *p, int w, int h, float thresh,
                        int relative, detection *dets, float *output,
                        int outputs, int input_w, int input_h) {
    get_region_detections(p, w, h, thresh, relative, dets, output, outputs,
                          input_w, input_h);
    dets += p->width * p->height * p->num;
}

detection *make_network_boxes(RegionLayerParams *p, int thresh, int *num) {
    int i;
    int nboxes = p->width * p->height * p->num; // num_detections(net, thresh);
    if (num)
        *num = nboxes;
    detection *dets = (detection *)calloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = (float *)calloc(p->classes, sizeof(float));
        if (p->coords > 4) {
            dets[i].mask = (float *)calloc(p->coords - 4, sizeof(float));
        }
    }
    return dets;
}

void customCoarsenedConvolutionForward(
    float *layer_input, float *layer_output,
    // int coarsening_factor, int coarsening_stride,
    cudnnConvolutionDescriptor_t conv_desc, cudnnFilterDescriptor_t filt_desc,
    cudnnTensorDescriptor_t input_tensor, float *filt,
    cudaStream_t stream_compute) {

    int pad_h, pad_w, stride_x,
        stride_y; // padding along h and w, vertical and horizontal stride
    int dilation_h, dilation_w; // this is 1 always for both
    cudnnConvolutionMode_t mode;
    cudnnDataType_t computeType;

    cudnnGetConvolution2dDescriptor(conv_desc, &pad_h, &pad_w, &stride_x,
                                    &stride_y, &dilation_h, &dilation_w, &mode,
                                    &computeType);

    int k, c; // k = # of output channels, c = # of input channels
    cudnnDataType_t datatype;
    cudnnTensorFormat_t format;
    int kernel_h, kernel_w;
    cudnnGetFilter4dDescriptor(filt_desc, &datatype, &format, &k, &c, &kernel_h,
                               &kernel_w);

    cudnnDataType_t dataType2;

    // For example, in a minibatch of RGB images, we may have
    // X[n,c,h,w], where n is the index of an image in the
    // minibatch, c is the channel (R = 0, G = 1, B = 2), and h and w
    // index a pixel (h, w) in the image (h and w are height and width)

    int batch_size, c2, input_h, input_w;
    int nStr, cStr, hStr, wStr;

    cudnnGetTensor4dDescriptor(input_tensor, &dataType2, &batch_size, &c2,
                               &input_h, &input_w, &nStr, &cStr, &hStr, &wStr);

    // cout << pad_h << endl;

    // for debugging
    int coarsening_factor = 1;
    int coarsening_stride = 32;

    if (kernel_h != kernel_w) {
        std::cout << "ERROR: Please pass a square kernel with equal height and "
                     "width. Returning..."
                  << std::endl;
        return;
    }

    if (input_h != input_w) {
        std::cout << "ERROR: Please pass a square input with equal height and "
                     "width. Returning..."
                  << std::endl;
        return;
    }

    if (pad_h != pad_w) {
        std::cout
            << "ERROR: Padding in both directions should be equal. Returning..."
            << std::endl;
        return;
    }

    if (stride_y != stride_x || stride_y != 1) {
        std::cout << "ERROR: Please ensure both stride x and stride y are "
                     "equal to 1. Returning..."
                  << std::endl;
        return;
    }

    if (coarsening_stride != 32) {
        std::cout << "ERROR: Stride is not 32. This will break memory "
                     "coalescing pattern. Please set stride to 32. Returning..."
                  << std::endl;
        return;
    }

    float *images = layer_input;
    float *filters = filt;
    float *output = layer_output;
    float gPadZeros = pad_h;
    int gFilterSize = kernel_h;
    int gEven = (gFilterSize % 2 == 0);
    int gInputSize = input_h;
    int gInputPlanes = c;

    int gNumFilters = k;

    int stride = stride_x;
    int gOutputSize = (gInputSize - gFilterSize + 2 * gPadZeros) / stride + 1;
    int gOutputSizeSquared = gOutputSize * gOutputSize;

    int gFilterSizeSquared = (gFilterSize * gFilterSize);
    int filterCubeLength = (gInputPlanes * gFilterSizeSquared);
    int gInputSizeSquared = (gInputSize * gInputSize);

    //--std::cout<<"input_h is ...."<<input_h<<std::endl;
    //--std::cout<<"FilterCubelength is..."<<filterCubeLength<<std::endl;i
    //--std::cout<<"gInputSizeSquare is..."<<gInputSizeSquared<<std::endl;

    if (filterCubeLength >= 600) {
        std::cout << "Allocated shared memory is not enough (filter size is "
                     "too large/param::M1 in kernel generator script)."
                  << std::endl;
        std::cout << "Please regenerate the kernels with large enough shared "
                     "memory. Returning..."
                  << std::endl;
        return;
    }

    if (gInputSizeSquared >= 512) {
        std::cout << "Allocated shared memory is not enough (input size is too "
                     "large/param::M2 in kernel generator script)."
                  << std::endl;
        std::cout << "Please regenerate the kernels with large enough shared "
                     "memory. Returning..."
                  << std::endl;
        return;
    }

    int batchSize = 1;
    dim3 grid(batchSize * gNumFilters);
    // int nblocks =
    // ((gOutputSizeSquared+coarsening_factor-1)/(coarsening_factor) + 31)/32 *
    // 32;
    int nblocks = gOutputSizeSquared;
    dim3 block(nblocks);

    //--  std::cout<<"grid"<<batchSize * gNumFilters<<endl;
    // std::cout<<"grid"<<grid;
    //--  std::cout<<"block"<< nblocks<<endl;

    if (coarsening_factor == 1) {
        coarsened_convolution_1C32S<<<grid, block, 0, stream_compute>>>(
            batchSize, images, filters, output, gOutputSize, gPadZeros, gEven,
            gOutputSizeSquared, gInputSize, gInputPlanes, gFilterSize,
            gNumFilters);
        check_error(cudaPeekAtLastError());
        return;
    } else if (coarsening_factor == 2) {
        coarsened_convolution_2C32S<<<grid, block, 0, stream_compute>>>(
            batchSize, images, filters, output, gOutputSize, gPadZeros, gEven,
            gOutputSizeSquared, gInputSize, gInputPlanes, gFilterSize,
            gNumFilters);
        check_error(cudaPeekAtLastError());
        return;
    } else if (coarsening_factor == 4) {
        coarsened_convolution_4C32S<<<grid, block, 0, stream_compute>>>(
            batchSize, images, filters, output, gOutputSize, gPadZeros, gEven,
            gOutputSizeSquared, gInputSize, gInputPlanes, gFilterSize,
            gNumFilters);
        check_error(cudaPeekAtLastError());
        return;
    } else if (coarsening_factor == 8) {
        coarsened_convolution_8C32S<<<grid, block, 0, stream_compute>>>(
            batchSize, images, filters, output, gOutputSize, gPadZeros, gEven,
            gOutputSizeSquared, gInputSize, gInputPlanes, gFilterSize,
            gNumFilters);
        check_error(cudaPeekAtLastError());
        return;
    } else if (coarsening_factor == 16) {
        coarsened_convolution_16C32S<<<grid, block, 0, stream_compute>>>(
            batchSize, images, filters, output, gOutputSize, gPadZeros, gEven,
            gOutputSizeSquared, gInputSize, gInputPlanes, gFilterSize,
            gNumFilters);
        check_error(cudaPeekAtLastError());
        return;
    } else if (coarsening_factor == 32) {
        coarsened_convolution_32C32S<<<grid, block, 0, stream_compute>>>(
            batchSize, images, filters, output, gOutputSize, gPadZeros, gEven,
            gOutputSizeSquared, gInputSize, gInputPlanes, gFilterSize,
            gNumFilters);
        check_error(cudaPeekAtLastError());
        return;
    }

    std::cout << "ERROR: An invalid coarsening factor has been passed. Please "
                 "ensure coarsening factor is one of 1/2/4/8/16. Returning..."
              << std::endl;
    return;
}

// functions taken from darknet frameowrk

void malloc_error() {
    fprintf(stderr, "Malloc error\n");
    exit(-1);
}

void free_node(node *n) {
    node *next;
    while (n) {
        next = n->next;
        free(n);
        n = next;
    }
}

void free_list(list *l) {
    free_node(l->front);
    free(l);
}

void list_insert(list *l, void *val) {
    node *new1 = (node *)malloc(sizeof(node));
    new1->val = val;
    new1->next = 0;

    if (!l->back) {
        l->front = new1;
        new1->prev = 0;
    } else {
        l->back->next = new1;
        new1->prev = l->back;
    }
    l->back = new1;
    ++l->size;
}

void file_error(const char *s) {
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}

void strip(char *s) {
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for (i = 0; i < len; ++i) {
        char c = s[i];
        if (c == ' ' || c == '\t' || c == '\n')
            ++offset;
        else
            s[i - offset] = c;
    }
    s[len - offset] = '\0';
}

char *fgetl(FILE *fp) {
    if (feof(fp))
        return 0;
    size_t size = 512;
    char *line = (char *)malloc(size * sizeof(char));
    if (!fgets(line, size, fp)) {
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while ((line[curr - 1] != '\n') && !feof(fp)) {
        if (curr == size - 1) {
            size *= 2;
            line = (char *)realloc(line, size * sizeof(char));
            if (!line) {
                printf("%ld\n", size);
                malloc_error();
            }
        }
        size_t readsize = size - curr;
        if (readsize > INT_MAX)
            readsize = INT_MAX - 1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if (line[curr - 1] == '\n')
        line[curr - 1] = '\0';

    return line;
}

void **list_to_array(list *l) {
    void **a = (void **)calloc(l->size, sizeof(void *));
    int count = 0;
    node *n = l->front;
    while (n) {
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}

list *make_list() {
    list *l = (list *)malloc(sizeof(list));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}

list *get_paths(const char *filename) {
    char *path;
    FILE *file = fopen(filename, "r");
    if (!file)
        file_error(filename);
    list *lines = make_list();
    while ((path = fgetl(file))) {
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

void top_k(float *a, int n, int k, int *index) {
    int i, j;
    for (j = 0; j < k; ++j)
        index[j] = -1;
    for (i = 0; i < n; ++i) {
        int curr = i;
        for (j = 0; j < k; ++j) {
            if ((index[j] < 0) || a[curr] > a[index[j]]) {
                int swap = curr;
                curr = index[j];
                index[j] = swap;
            }
        }
    }
}

char **get_labels(const char *filename) {
    list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

char *option_find(list *l, const char *key) {
    node *n = l->front;
    while (n) {
        kvp *p = (kvp *)n->val;
        if (strcmp(p->key, key) == 0) {
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}
char *option_find_str(list *l, const char *key, char *def) {
    char *v = option_find(l, key);
    if (v)
        return v;
    if (def)
        fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}

int option_find_int(list *l, const char *key, int def) {
    char *v = option_find(l, key);
    if (v)
        return atoi(v);
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}
void option_insert(list *l, char *key, char *val) {
    kvp *p = (kvp *)malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}

int read_option(char *s, list *options) {
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for (i = 0; i < len; ++i) {
        if (s[i] == '=') {
            s[i] = '\0';
            val = s + i + 1;
            break;
        }
    }
    if (i == len - 1)
        return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

list *read_data_cfg(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == 0)
        file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    while ((line = fgetl(file)) != 0) {
        ++nu;
        strip(line);
        switch (line[0]) {
        case '\0':
        case '#':
        case ';':
            free(line);
            break;
        default:
            if (!read_option(line, options)) {
                fprintf(stderr, "Config file error line %d, could parse: %s\n",
                        nu, line);
                free(line);
            }
            break;
        }
    }
    fclose(file);
    return options;
}

void fillExecutionTime(std::ifstream &fp, std::vector<InputOperation *> dags) {
    for (auto &input_operation : dags) {
        Operation *tp = input_operation;
        while (tp != nullptr) {
            fp >> tp->time_to_start;
            fp >> tp->time_to_execute;
            tp = tp->children.back();
        }
    }
}
