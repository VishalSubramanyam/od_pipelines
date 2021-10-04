#ifndef UTILITIES_H
#define UTILITIES_H
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <dag.h>
#include <image.h>
#include <iostream>
#include <layer_params.h>
#include <vector>
#define BLOCK1 512

typedef enum { LOGISTIC, LINEAR } ACTIVATION;

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true);
void error(const char *s);
dim3 cuda_gridsize(int n);
void check_error(cudaError_t status);
__global__ void forward_avgpool_layer_kernel(int n, int w, int h, int c,
                                             float *input, float *output);
void forward_avgpool_layer_gpu(int w, int h, int c, float *input,
                               float *output);
__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w,
                                             int in_c, int stride, int size,
                                             int pad, float *input,
                                             float *output);
void forward_maxpool_layer_gpu(int h, int w, int c, int stride, int size,
                               int pad, float *input, float *output);
__device__ void softmax_device(float *input, int n, float temp, int stride,
                               float *output);
__global__ void softmax_kernel(float *input, int n, int batch, int batch_offset,
                               int groups, int group_offset, int stride,
                               float temp, float *output);
void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups,
                 int group_offset, int stride, float temp, float *output,
                 cudaStream_t stream);
__device__ float linear_activate_kernel(float x);
__device__ float logistic_activate_kernel(float x);
__device__ float activate_kernel(float x, ACTIVATION a);
__device__ float leaky_activate_kernel(float x);
__global__ void activate_array_kernel(float *x, int n);
__global__ void normalize_kernel(int N, float *x, float *mean, float *variance,
                                 int batch, int filters, int spatial);
__global__ void scale_bias_kernel(float *output, float *biases, int n,
                                  int size);
__global__ void add_bias_kernel(float *output, float *biases, int batch, int n,
                                int size);
__global__ void copy_kernel(int N, float *X, int OFFX, int INCX, float *Y,
                            int OFFY, int INCY);
void copy_gpu_offset(int N, float *X, int OFFX, int INCX, float *Y, int OFFY,
                     int INCY, cudaStream_t stream);
void copy_gpu(int N, float *X, int INCX, float *Y, int INCY,
              cudaStream_t stream);
void normalize_gpu(float *x, float *mean, float *variance, int batch,
                   int filters, int spatial, cudaStream_t stream);
void add_bias_gpu(float *output, float *biases, int batch, int n, int size,
                  cudaStream_t stream);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size,
                    cudaStream_t stream);
__global__ void activate_array_kernel1(float *x, int n, ACTIVATION a);
void activate_array_gpu1(float *x, int n, ACTIVATION a, cudaStream_t stream);
void activate_array_gpu(float *x, int n, cudaStream_t stream);
int entry_index(int batch, int location, int entry, int w, int h, int outputs,
                int coords, int classes);
void forward_region_layer_gpu(int inputs, int outputs, float *input, int batch,
                              int h, int w, int nm, int classes, int coords,
                              float *output, cudaStream_t cs);
void free_detections(detection *dets, int n);
void correct_region_boxes(detection *dets, int n, int w, int h, int netw,
                          int neth, int relative);
box get_region_box(float *x, float *biases, int n, int index, int i, int j,
                   int w, int h, int stride);
void get_region_detections(RegionLayerParams *p, int w, int h, float thresh,
                           int relative, detection *dets, float *output,
                           int outputs, int netw, int neth);
void fill_network_boxes(RegionLayerParams *p, int w, int h, float thresh,
                        int relative, detection *dets, float *output,
                        int outputs, int input_w, int input_h);
detection *make_network_boxes(RegionLayerParams *p, int thresh, int *num);
void customCoarsenedConvolutionForward(
    float *layer_input, float *layer_output,
    // int coarsening_factor, int coarsening_stride,
    cudnnConvolutionDescriptor_t conv_desc, cudnnFilterDescriptor_t filt_desc,
    cudnnTensorDescriptor_t input_tensor, float *filt,
    cudaStream_t stream_compute);
void malloc_error();
void free_node(node *n);
void free_list(list *l);
void list_insert(list *l, void *val);
void file_error(const char *s);
void strip(char *s);
char *fgetl(FILE *fp);
void **list_to_array(list *l);
list *make_list();
list *get_paths(const char *filename);
void top_k(float *a, int n, int k, int *index);
char **get_labels(const char *filename);
char *option_find(list *l, const char *key);
char *option_find_str(list *l, const char *key, char *def);
int option_find_int(list *l, const char *key, int def);
void option_insert(list *l, char *key, char *val);
int read_option(char *s, list *options);
list *read_data_cfg(const char *filename);
void fillExecutionTime(ifstream &fp, vector<InputOperation *> dags);
#endif