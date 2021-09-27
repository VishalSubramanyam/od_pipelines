#include <assert.h>
#include <cstdlib>
#include <cmath>
#include "ScheduleEngine.h"
#include "image.h"
#include "coarsened_forward_convolution.h"
#define BLOCK1 512
using namespace std;

void error(const char *s)
{
	perror(s);
	assert(0);
	exit(-1);
}

dim3 cuda_gridsize(int n)
{
	unsigned int k = (n - 1) / BLOCK1 + 1;
	unsigned int x = k;
	unsigned int y = 1;
	if (x > 65535)
	{
		x = ceil(sqrt(k));
		y = (n - 1) / (x * BLOCK1) + 1;
	}
	dim3 d = {x, y, 1};
	// printf("n=%d x=%d y=%d x*y*BLOCK1=%d\n", n, x, y, x*y*BLOCK1);
	return d;
}

void check_error(cudaError_t status)
{
	//cudaDeviceSynchronize();
	cudaError_t status2 = cudaGetLastError();
	if (status != cudaSuccess)
	{
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error: %s", s);
		error(buffer);
	}
	if (status2 != cudaSuccess)
	{
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error Prev: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error Prev: %s", s);
		error(buffer);
	}
}

//Pooling kernels--start
__global__ void forward_avgpool_layer_kernel(int n, int w, int h, int c, float *input, float *output)
{
	int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= n)
		return;

	int k = id % c;
	id /= c;
	int b = id;

	int i;
	int out_index = (k + c * b);
	output[out_index] = 0;
	for (i = 0; i < w * h; ++i)
	{
		int in_index = i + h * w * (k + b * c);
		output[out_index] += input[in_index];
	}
	output[out_index] /= w * h;
}

void forward_avgpool_layer_gpu(int w, int h, int c, float *input, float *output)
{
	size_t n = c;
	//size_t n = layer.c*layer.batch;

	forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK1>>>(n, w, h, c, input, output);
	check_error(cudaPeekAtLastError());
}

__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *input, float *output)
{
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
	for (l = 0; l < size; ++l)
	{
		for (m = 0; m < size; ++m)
		{
			int cur_h = h_offset + i * stride + l;
			int cur_w = w_offset + j * stride + m;
			int index = cur_w + in_w * (cur_h + in_h * (k + b * in_c));
			int valid = (cur_h >= 0 && cur_h < in_h &&
						 cur_w >= 0 && cur_w < in_w);
			float val = (valid != 0) ? input[index] : -INFINITY;
			max_i = (val > max) ? index : max_i;
			max = (val > max) ? val : max;
		}
	}
	output[out_index] = max;
}

void forward_maxpool_layer_gpu(int h, int w, int c, int stride, int size, int pad, float *input, float *output)
{
	int h1 = (h + pad - size) / stride + 1; //layer.out_h;
	int w1 = (w + pad - size) / stride + 1; //layer.out_w;
	//int c = layer.c;

	size_t n = h1 * w1 * c;

	forward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK1>>>(n, h, w, c, stride, size, pad, input, output);
	check_error(cudaPeekAtLastError());
}

//for softmax function
__device__ void softmax_device(float *input, int n, float temp, int stride, float *output)
{
	int i;
	float sum = 0;
	float largest = -INFINITY;
	for (i = 0; i < n; ++i)
	{
		int val = input[i * stride];
		largest = (val > largest) ? val : largest;
	}
	for (i = 0; i < n; ++i)
	{
		float e = expf(input[i * stride] / temp - largest / temp);
		sum += e;
		output[i * stride] = e;
	}
	for (i = 0; i < n; ++i)
	{
		output[i * stride] /= sum;
	}
}

__global__ void softmax_kernel(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
	int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= batch * groups)
		return;
	int b = id / groups;
	int g = id % groups;
	softmax_device(input + b * batch_offset + g * group_offset, n, temp, stride, output + b * batch_offset + g * group_offset);
}

void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
	softmax_kernel<<<cuda_gridsize(batch * groups), BLOCK1>>>(input, n, batch, batch_offset, groups, group_offset, stride, temp, output);
	check_error(cudaPeekAtLastError());
}
//for softmax function -- end

__device__ float leaky_activate_kernel(float x) { return (x > 0) ? x : .1f * x; }

__global__ void activate_array_kernel(float *x, int n)
{
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n)
		x[i] = leaky_activate_kernel(x[i]);
	//	if (i<n){ if (x>0) x[i]= x[i] ;
	//                 else x[i]=.1f*x[i];
	//}
}

__global__ void normalize_kernel(int N, float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
	int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= N)
		return;
	int f = (index / spatial) % filters;

	x[index] = (x[index] - mean[f]) / (sqrtf(variance[f] + .00001f));
}

__global__ void scale_bias_kernel(float *output, float *biases, int n, int size)
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int filter = blockIdx.y;
	int batch = blockIdx.z;

	if (offset < size)
		output[(batch * n + filter) * size + offset] *= biases[filter];
}

__global__ void add_bias_kernel(float *output, float *biases, int batch, int n, int size)
{
	int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= n * size * batch)
		return;
	int i = index % size;
	index /= size;
	int j = index % n;
	index /= n;
	int k = index;

	output[(k * n + j) * size + i] += biases[j];
}

void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial, cudaStream_t stream)
{
	size_t N = batch * filters * spatial;
	normalize_kernel<<<cuda_gridsize(N), BLOCK1, 0, stream>>>(N, x, mean, variance, batch, filters, spatial);
	check_error(cudaPeekAtLastError());
}

void add_bias_gpu(float *output, float *biases, int batch, int n, int size, cudaStream_t stream)
{
	int num = n * size * batch;

	add_bias_kernel<<<cuda_gridsize(num), BLOCK1, 0, stream>>>(output, biases, batch, n, size);
	check_error(cudaPeekAtLastError());
}

void scale_bias_gpu(float *output, float *biases, int batch, int n, int size, cudaStream_t stream)
{
	dim3 dimGrid((size - 1) / BLOCK1 + 1, n, batch);
	dim3 dimBlock(BLOCK1, 1, 1);

	scale_bias_kernel<<<dimGrid, dimBlock, 0, stream>>>(output, biases, n, size);
	check_error(cudaPeekAtLastError());
}

void activate_array_gpu(float *x, int n, cudaStream_t stream)
{
	dim3 kk = cuda_gridsize(n);
	activate_array_kernel<<<kk, BLOCK1, 0, stream>>>(x, n);
	check_error(cudaPeekAtLastError());
}

void customCoarsenedConvolutionForward(float *layer_input, float *layer_output,
									   // int coarsening_factor, int coarsening_stride,
									   cudnnConvolutionDescriptor_t conv_desc,
									   cudnnFilterDescriptor_t filt_desc,
									   cudnnTensorDescriptor_t input_tensor,
									   float *filt, cudaStream_t stream_compute)
{

	int pad_h, pad_w, stride_x, stride_y; // padding along h and w, vertical and horizontal stride
	int dilation_h, dilation_w;			  // this is 1 always for both
	cudnnConvolutionMode_t mode;
	cudnnDataType_t computeType;

	cudnnGetConvolution2dDescriptor(
		conv_desc,
		&pad_h, &pad_w,
		&stride_x, &stride_y,
		&dilation_h, &dilation_w,
		&mode,
		&computeType);

	int k, c; // k = # of output channels, c = # of input channels
	cudnnDataType_t datatype;
	cudnnTensorFormat_t format;
	int kernel_h, kernel_w;
	cudnnGetFilter4dDescriptor(
		filt_desc,
		&datatype,
		&format,
		&k, &c, &kernel_h, &kernel_w);

	cudnnDataType_t dataType2;

	// For example, in a minibatch of RGB images, we may have
	// X[n,c,h,w], where n is the index of an image in the
	// minibatch, c is the channel (R = 0, G = 1, B = 2), and h and w
	// index a pixel (h, w) in the image (h and w are height and width)

	int batch_size, c2, input_h, input_w;
	int nStr, cStr, hStr, wStr;

	cudnnGetTensor4dDescriptor(
		input_tensor,
		&dataType2,
		&batch_size, &c2, &input_h, &input_w,
		&nStr, &cStr, &hStr, &wStr);

	//cout << pad_h << endl;

	// for debugging
	int coarsening_factor = 1;
	int coarsening_stride = 32;

	if (kernel_h != kernel_w)
	{
		std::cout << "ERROR: Please pass a square kernel with equal height and width. Returning..." << std::endl;
		return;
	}

	if (input_h != input_w)
	{
		std::cout << "ERROR: Please pass a square input with equal height and width. Returning..." << std::endl;
		return;
	}

	if (pad_h != pad_w)
	{
		std::cout << "ERROR: Padding in both directions should be equal. Returning..." << std::endl;
		return;
	}

	if (stride_y != stride_x || stride_y != 1)
	{
		std::cout << "ERROR: Please ensure both stride x and stride y are equal to 1. Returning..." << std::endl;
		return;
	}

	if (coarsening_stride != 32)
	{
		std::cout << "ERROR: Stride is not 32. This will break memory coalescing pattern. Please set stride to 32. Returning..." << std::endl;
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

	if (filterCubeLength >= 600)
	{
		std::cout << "Allocated shared memory is not enough (filter size is too large/param::M1 in kernel generator script)." << std::endl;
		std::cout << "Please regenerate the kernels with large enough shared memory. Returning..." << std::endl;
		return;
	}

	if (gInputSizeSquared >= 512)
	{
		std::cout << "Allocated shared memory is not enough (input size is too large/param::M2 in kernel generator script)." << std::endl;
		std::cout << "Please regenerate the kernels with large enough shared memory. Returning..." << std::endl;
		return;
	}

	int batchSize = 1;
	dim3 grid(batchSize * gNumFilters);
	//int nblocks = ((gOutputSizeSquared+coarsening_factor-1)/(coarsening_factor) + 31)/32 * 32;
	int nblocks = gOutputSizeSquared;
	dim3 block(nblocks);

	//--  std::cout<<"grid"<<batchSize * gNumFilters<<endl;
	//std::cout<<"grid"<<grid;
	//--  std::cout<<"block"<< nblocks<<endl;

	if (coarsening_factor == 1)
	{
		coarsened_convolution_1C32S<<<grid, block, 0, stream_compute>>>(batchSize, images, filters, output, gOutputSize, gPadZeros, gEven, gOutputSizeSquared, gInputSize, gInputPlanes, gFilterSize, gNumFilters);
		check_error(cudaPeekAtLastError());
		return;
	}
	else if (coarsening_factor == 2)
	{
		coarsened_convolution_2C32S<<<grid, block, 0, stream_compute>>>(batchSize, images, filters, output, gOutputSize, gPadZeros, gEven, gOutputSizeSquared, gInputSize, gInputPlanes, gFilterSize, gNumFilters);
		check_error(cudaPeekAtLastError());
		return;
	}
	else if (coarsening_factor == 4)
	{
		coarsened_convolution_4C32S<<<grid, block, 0, stream_compute>>>(batchSize, images, filters, output, gOutputSize, gPadZeros, gEven, gOutputSizeSquared, gInputSize, gInputPlanes, gFilterSize, gNumFilters);
		check_error(cudaPeekAtLastError());
		return;
	}
	else if (coarsening_factor == 8)
	{
		coarsened_convolution_8C32S<<<grid, block, 0, stream_compute>>>(batchSize, images, filters, output, gOutputSize, gPadZeros, gEven, gOutputSizeSquared, gInputSize, gInputPlanes, gFilterSize, gNumFilters);
		check_error(cudaPeekAtLastError());
		return;
	}
	else if (coarsening_factor == 16)
	{
		coarsened_convolution_16C32S<<<grid, block, 0, stream_compute>>>(batchSize, images, filters, output, gOutputSize, gPadZeros, gEven, gOutputSizeSquared, gInputSize, gInputPlanes, gFilterSize, gNumFilters);
		check_error(cudaPeekAtLastError());
		return;
	}
	else if (coarsening_factor == 32)
	{
		coarsened_convolution_32C32S<<<grid, block, 0, stream_compute>>>(batchSize, images, filters, output, gOutputSize, gPadZeros, gEven, gOutputSizeSquared, gInputSize, gInputPlanes, gFilterSize, gNumFilters);
		check_error(cudaPeekAtLastError());
		return;
	}

	std::cout << "ERROR: An invalid coarsening factor has been passed. Please ensure coarsening factor is one of 1/2/4/8/16. Returning..." << std::endl;
	return;
}

//functions taken from darknet frameowrk

void malloc_error()
{
	fprintf(stderr, "Malloc error\n");
	exit(-1);
}

void free_node(node *n)
{
	node *next;
	while (n)
	{
		next = n->next;
		free(n);
		n = next;
	}
}

void free_list(list *l)
{
	free_node(l->front);
	free(l);
}

void list_insert(list *l, void *val)
{
	node *new1 = (node *)malloc(sizeof(node));
	new1->val = val;
	new1->next = 0;

	if (!l->back)
	{
		l->front = new1;
		new1->prev = 0;
	}
	else
	{
		l->back->next = new1;
		new1->prev = l->back;
	}
	l->back = new1;
	++l->size;
}

void file_error(const char *s)
{
	fprintf(stderr, "Couldn't open file: %s\n", s);
	exit(0);
}

void strip(char *s)
{
	size_t i;
	size_t len = strlen(s);
	size_t offset = 0;
	for (i = 0; i < len; ++i)
	{
		char c = s[i];
		if (c == ' ' || c == '\t' || c == '\n')
			++offset;
		else
			s[i - offset] = c;
	}
	s[len - offset] = '\0';
}

char *fgetl(FILE *fp)
{
	if (feof(fp))
		return 0;
	size_t size = 512;
	char *line = (char *)malloc(size * sizeof(char));
	if (!fgets(line, size, fp))
	{
		free(line);
		return 0;
	}

	size_t curr = strlen(line);

	while ((line[curr - 1] != '\n') && !feof(fp))
	{
		if (curr == size - 1)
		{
			size *= 2;
			line = (char *)realloc(line, size * sizeof(char));
			if (!line)
			{
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

void **list_to_array(list *l)
{
	void **a = (void **)calloc(l->size, sizeof(void *));
	int count = 0;
	node *n = l->front;
	while (n)
	{
		a[count++] = n->val;
		n = n->next;
	}
	return a;
}

list *make_list()
{
	list *l = (list *)malloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}

list *get_paths(const char *filename)
{
	char *path;
	FILE *file = fopen(filename, "r");
	if (!file)
		file_error(filename);
	list *lines = make_list();
	while ((path = fgetl(file)))
	{
		list_insert(lines, path);
	}
	fclose(file);
	return lines;
}

void top_k(float *a, int n, int k, int *index)
{
	int i, j;
	for (j = 0; j < k; ++j)
		index[j] = -1;
	for (i = 0; i < n; ++i)
	{
		int curr = i;
		for (j = 0; j < k; ++j)
		{
			if ((index[j] < 0) || a[curr] > a[index[j]])
			{
				int swap = curr;
				curr = index[j];
				index[j] = swap;
			}
		}
	}
}

char **get_labels(const char *filename)
{
	list *plist = get_paths(filename);
	char **labels = (char **)list_to_array(plist);
	free_list(plist);
	return labels;
}

char *option_find(list *l, const char *key)
{
	node *n = l->front;
	while (n)
	{
		kvp *p = (kvp *)n->val;
		if (strcmp(p->key, key) == 0)
		{
			p->used = 1;
			return p->val;
		}
		n = n->next;
	}
	return 0;
}
char *option_find_str(list *l, const char *key, char *def)
{
	char *v = option_find(l, key);
	if (v)
		return v;
	if (def)
		fprintf(stderr, "%s: Using default '%s'\n", key, def);
	return def;
}

int option_find_int(list *l, const char *key, int def)
{
	char *v = option_find(l, key);
	if (v)
		return atoi(v);
	fprintf(stderr, "%s: Using default '%d'\n", key, def);
	return def;
}
void option_insert(list *l, char *key, char *val)
{
	kvp *p = (kvp *)malloc(sizeof(kvp));
	p->key = key;
	p->val = val;
	p->used = 0;
	list_insert(l, p);
}

int read_option(char *s, list *options)
{
	size_t i;
	size_t len = strlen(s);
	char *val = 0;
	for (i = 0; i < len; ++i)
	{
		if (s[i] == '=')
		{
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

list *read_data_cfg(const char *filename)
{
	FILE *file = fopen(filename, "r");
	if (file == 0)
		file_error(filename);
	char *line;
	int nu = 0;
	list *options = make_list();
	while ((line = fgetl(file)) != 0)
	{
		++nu;
		strip(line);
		switch (line[0])
		{
		case '\0':
		case '#':
		case ';':
			free(line);
			break;
		default:
			if (!read_option(line, options))
			{
				fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
				free(line);
			}
			break;
		}
	}
	fclose(file);
	return options;
}

void ScheduleEngine::initMutex(void)
{
	pthread_mutex_init(&lock, NULL);
}

void ScheduleEngine::destroyMutex(void)
{
	pthread_mutex_destroy(&lock);
}

void ScheduleEngine::initCond(void)
{
	pthread_cond_init(&cond, NULL);
}

void ScheduleEngine::destroyCond(void)
{
	pthread_cond_destroy(&cond);
}

ScheduleEngine::ScheduleEngine()
{
	int leastPriority, greatestPriority;
	checkCudaErrors(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

	checkCudaErrors(cudaStreamCreateWithPriority(&compute_streams[HIGH_COMPUTE_STREAM], cudaStreamNonBlocking, greatestPriority));
	checkCudaErrors(cudaStreamCreateWithPriority(&compute_streams[LOW_COMPUTE_STREAM], cudaStreamNonBlocking, leastPriority));
	checkCudaErrors(cudaStreamCreate(&memoryStream));

	checkCUDNN(cudnnCreate(&cudnnHandles[0]));
	checkCUDNN(cudnnSetStream(cudnnHandles[0], compute_streams[0]));
	checkCUDNN(cudnnCreate(&cudnnHandles[1]));
	checkCUDNN(cudnnSetStream(cudnnHandles[1], compute_streams[1]));

	checkCUBLAS(cublasCreate(&cublasHandles[0]));
	checkCUBLAS(cublasSetStream(cublasHandles[0], compute_streams[0]));
	checkCUBLAS(cublasCreate(&cublasHandles[1]));
	checkCUBLAS(cublasSetStream(cublasHandles[1], compute_streams[1]));
	initMutex();
	initCond();
}

void ScheduleEngine::enqueue(Operation *tp)
{
	pthread_mutex_lock(&lock);
	Q.push(tp);
	pthread_cond_signal(&cond);
	pthread_mutex_unlock(&lock);
}

Operation *ScheduleEngine::dequeue()
{
	Operation *tp;
	pthread_mutex_lock(&lock);
	while (Q.empty())
	{
		printf("Wating for operations to be added in Queue\n");
		pthread_cond_wait(&cond, &lock);
	}
	tp = Q.top();
	Q.pop();
	pthread_mutex_unlock(&lock);
	return (tp);
}

//Execute opration will execute the operation on GPU. The operation performed is based on the type of layer
void ScheduleEngine::execute(Operation *tp, stream_indicator streamIndicator)
{
	cudaStream_t &compute_stream = compute_streams[streamIndicator];
	cudnnHandle_t &cudnn_handle = cudnnHandles[streamIndicator];
	cublasHandle_t &cublas_handle = cublasHandles[streamIndicator];
	int i = tp->op_layer;

	NeuralNet *nm = tp->model;
	CnmemSpace space_tracker(nm->free_bytes); //need updates here

	if (i == 0)
	{
		space_tracker.updateSpace(CnmemSpace::SUB, nm->layer_input_size[0] * nm->data_type_size);
	}
	float alpha = 1.0, beta = 0.0;
	float Salpha = 1.0, Sbeta = 0.0;
	double Dalpha = 1.0, Dbeta = 0.0;
	size_t cur_workspace_size;
	void *cur_workspace;

	//	nm->lockedcnmemMalloc(&(nm->layer_input[i + 1]), nm->layer_input_size[i+ 1] * nm->data_type_size,NULL);
	space_tracker.updateSpace(CnmemSpace::SUB, nm->layer_input_size[i + 1] * nm->data_type_size);

	if (nm->layer_type[i] == CONV)
	{
		tp->type = 'C';
		// std::cout << "conv\n";
		ConvLayerParams *cur_params = (ConvLayerParams *)nm->params[i];

		cur_workspace_size = cur_params->fwd_workspace_size;
		nm->lockedcnmemMalloc(&cur_workspace, cur_workspace_size, compute_stream); // compute stream or memory stream?
		// computation
		/*		checkCUDNN(cudnnConvolutionForward(nm->cudnn_handle, &alpha, 
				cur_params->input_tensor, nm->layer_input[i],
				cur_params->filter_desc, cur_params->W,
				cur_params->conv_desc, cur_params->fwd_algo,
				cur_workspace, cur_workspace_size,
				&beta,
				cur_params->output_tensor, nm->layer_input[i + 1]));
		 */
		//custom  coarsened cuda kernel
		customCoarsenedConvolutionForward((float *)nm->layer_input[i], (float *)nm->layer_input[i + 1], cur_params->conv_desc, cur_params->filter_desc, cur_params->input_tensor, (float *)cur_params->W, compute_stream);

		//Batch Normalization
		if (cur_params->bn == 1)
		{
			normalize_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->rolling_mean_gpu, (float *)cur_params->rolling_variance_gpu, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
			scale_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->scales_gpu, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
			add_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->b, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
		}
		else
		{
			add_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->b, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
		}
		/*--		checkCUDNN(cudnnAddTensor(nm->cudnn_handle, &alpha, 
		  cur_params->bias_desc, cur_params->b, 
		  &alpha,
		  cur_params->output_tensor, nm->layer_input[i + 1]));
		  --*/
		// if activation required
		if (cur_params->activation_mode != ACTIVATION_NONE)
		{
			//Replacing cuDNN call for relu to custom leaky relu call
			float *addr = (float *)(nm->layer_input[i + 1]);
			activate_array_gpu(addr, nm->layer_input_size[i + 1], compute_stream);

			/*checkCUDNN(cudnnActivationForward(nm->cudnn_handle, cur_params->actv_desc,
			  &alpha,
			  cur_params->output_tensor, nm->layer_input[i + 1],
			  &beta,
			  cur_params->output_tensor, nm->layer_input[i + 1]));
			 */
		}

		space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
		// std::cout << "Free bytes: " << free_bytes << std::endl;
	}
	else if (nm->layer_type[i] == FULLY_CONNECTED)
	{
		tp->type = 'F';
		// std::cout << "FC\n";
		FCLayerParams *cur_params = (FCLayerParams *)nm->params[i];
		// std::cout << "FChere" << i << std::endl;

		if (nm->data_type == CUDNN_DATA_FLOAT)
		{
			checkCUBLAS(cublasSgemm(cublas_handle,
									CUBLAS_OP_N, CUBLAS_OP_N,
									cur_params->C_out, nm->batch_size, cur_params->C_in,
									&Salpha,
									(float *)cur_params->W, cur_params->C_out,
									(float *)nm->layer_input[i], cur_params->C_in,
									&Sbeta,
									(float *)nm->layer_input[i + 1], cur_params->C_out));
			checkCUBLAS(cublasSgemm(cublas_handle,
									CUBLAS_OP_N, CUBLAS_OP_N,
									cur_params->C_out, nm->batch_size, 1,
									&Salpha,
									(float *)cur_params->b, cur_params->C_out,
									(float *)nm->one_vec, 1,
									&Salpha,
									(float *)nm->layer_input[i + 1], cur_params->C_out));
		}
		else if (nm->data_type == CUDNN_DATA_DOUBLE)
		{
			checkCUBLAS(cublasDgemm(cublas_handle,
									CUBLAS_OP_N, CUBLAS_OP_N,
									cur_params->C_out, nm->batch_size, cur_params->C_in,
									&Dalpha,
									(double *)cur_params->W, cur_params->C_out,
									(double *)nm->layer_input[i], cur_params->C_in,
									&Dbeta,
									(double *)nm->layer_input[i + 1], cur_params->C_out));
			checkCUBLAS(cublasDgemm(cublas_handle,
									CUBLAS_OP_N, CUBLAS_OP_N,
									cur_params->C_out, nm->batch_size, 1,
									&Dalpha,
									(double *)cur_params->b, cur_params->C_out,
									(double *)nm->one_vec, 1,
									&Dalpha,
									(double *)nm->layer_input[i + 1], cur_params->C_out));
		}
		if (cur_params->activation_mode != ACTIVATION_NONE)
		{
			//Replacing cuDNN call for Relu activation to custom Leaky Relu call
			//checkCUDNN(cudnnActivationForward(nm->cudnn_handle, cur_params->actv_desc,&alpha,cur_params->output_tensor, nm->layer_input[i + 1],&beta,cur_params->output_tensor, nm->layer_input[i + 1]));
			activate_array_gpu((float *)nm->layer_input[i + 1], nm->layer_input_size[i + 1], compute_stream);
		}
		// std::cout << "FChere" << i << std::endl;
	}

	else if (nm->layer_type[i] == DROPOUT)
	{
		tp->type = 'D';
		// std::cout << "Dropout\n";
		DropoutLayerParams *cur_params = (DropoutLayerParams *)nm->params[i];
		checkCUDNN(cudnnDropoutForward(cudnn_handle, cur_params->dropout_desc,
									   cur_params->input_tensor, nm->layer_input[i],
									   cur_params->input_tensor, nm->layer_input[i + 1],
									   cur_params->reserved_space,
									   cur_params->reserved_space_size));
	}
	else if (nm->layer_type[i] == BATCHNORM)
	{
		tp->type = 'B';
		// std::cout << "Batchnorm\n";
		BatchNormLayerParams *cur_params = (BatchNormLayerParams *)nm->params[i];

		checkCUDNN(cudnnBatchNormalizationForwardInference(cudnn_handle, cur_params->mode,
														   &alpha, &beta,
														   cur_params->input_tensor, nm->layer_input[i], cur_params->input_tensor, nm->layer_input[i + 1], cur_params->sbmv_desc,
														   cur_params->scale, cur_params->bias,
														   cur_params->running_mean, cur_params->running_variance,
														   cur_params->epsilon));
	}
	else if (nm->layer_type[i] == POOLING)
	{
		tp->type = 'P';
		// std::cout << "Pooling\n";
		PoolingLayerParams *cur_params = (PoolingLayerParams *)nm->params[i];
		checkCUDNN(cudnnPoolingForward(cudnn_handle, cur_params->pool_desc,
									   &alpha,
									   cur_params->input_tensor, nm->layer_input[i],
									   &beta,
									   cur_params->output_tensor, nm->layer_input[i + 1]));
	}
	else if (nm->layer_type[i] == ACTV)
	{
		tp->type = 'A';
		ActivationLayerParams *cur_params = (ActivationLayerParams *)nm->params[i];
		checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
										  &alpha,
										  cur_params->input_tensor, nm->layer_input[i],
										  &beta,
										  cur_params->input_tensor, nm->layer_input[i + 1]));
	}
	else if (nm->layer_type[i] == SOFTMAX)
	{
		tp->type = 'S';
		SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)nm->params[i];
		checkCUDNN(cudnnSoftmaxForward(cudnn_handle, cur_params->algo, cur_params->mode,
									   &alpha,
									   cur_params->input_tensor, nm->layer_input[i],
									   &beta,
									   cur_params->input_tensor, nm->layer_input[i + 1]));
		//-Copy the result produced by softmax layer from GPU to CPU

		//checkCudaErrors(cudaStreamSynchronize(nm->stream_compute)); /////-----check....
		float *result = (float *)malloc(nm->layer_input_size[i + 1] * sizeof(float));
		checkCudaErrors(cudaMemcpy(result, nm->layer_input[i + 1], nm->layer_input_size[i + 1] * sizeof(float), cudaMemcpyDeviceToHost));

		//Infer the output class
		//	int *correct_count=0;
		//	nm->compareOutputCorrect(correct_count,nm->y);
		//	checkCNMEM(cnmemFree(nm->layer_input[nm->num_layers - 1], NULL));
		//	space_tracker.updateSpace(CnmemSpace::ADD, nm->layer_input_size[nm->num_layers - 1] * nm->data_type_size);
		//--
		int top = 5;
		list *options = read_data_cfg("data/imagenet1k.data"); //specify name  of the file
		char *name_list = option_find_str(options, "names", 0);
		if (!name_list)
			name_list = option_find_str(options, "labels", "data/labels.list");
		if (top == 0)
			top = option_find_int(options, "top", 1);

		int ii = 0;
		char **names = get_labels(name_list);
		//    clock_t time;
		int *indexes = (int *)calloc(top, sizeof(int));
		// time=clock();
		top_k(result, nm->layer_input_size[i + 1], top, indexes); //check parameters of this function
		// fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
		for (ii = 0; ii < top; ++ii)
		{
			//int index = indexes[ii];
			//if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
			//else printf("%s: %f\n",names[index], predictions[index]);
			//printf("index is %d: %5.2f%%: %s\n",index, result[index]*100, names[index]);
			//printf("index is %d: %s\n",index, names[index]);
		}
	}
	if (nm->layer_type[i] == CONV)
	{
		nm->lockedcnmemFree(cur_workspace, NULL);
		space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
	}

	//kCudaErrors(cudaStreamSynchronize(nm->stream_compute));
	//free the memory allocated to layer_input[i]
	//checkCNMEM(cnmemFree(nm->layer_input[i], NULL));
	//nm->lockedcnmemFree(nm->layer_input[i], NULL);
	//space_tracker.updateSpace(CnmemSpace::ADD, nm->layer_input_size[i] * nm->data_type_size);
}

//Dispatch opration will execute the operation on GPU. The operation performed is based on the type of layer
void ScheduleEngine::dispatch(Operation *tp, stream_indicator streamIndicator)
{
	cudaStream_t &compute_stream = compute_streams[streamIndicator];
	cudnnHandle_t &cudnn_handle = cudnnHandles[streamIndicator];
	cublasHandle_t &cublas_handle = cublasHandles[streamIndicator];
	//int priority=tp->priority;	// to be used later
	int i = tp->op_layer;

	NeuralNet *nm = tp->model;
	CnmemSpace space_tracker(nm->free_bytes); //need updates here
	//--	std::cout << "here\n";
	//--	std::cout << "Free bytes: " << nm->free_bytes << std::endl;

	if (i == 0)
	{ //this is the first layer, load and resize image as per current inference pipeline
		/*	image im = load_image_color(nm->imgfname, 0, 0);
		//size? net->w in yolo
		image r = letterbox_image(im,nm->input_w, nm->input_h );
		//resize_network(net, resized.w, resized.h);
		show_image(im,"orig",5);
		show_image(r,"letterimg",5);
		//copy image data into layer_input[0]
		//memcpy(&(nm->layer_input[i]),r.data,nm->layer_input_size[i]*nm->data_type_size);
		nm->lockedcnmemMalloc(&(nm->layer_input[0]), nm->layer_input_size[0] * nm->data_type_size, NULL);*/
		space_tracker.updateSpace(CnmemSpace::SUB, nm->layer_input_size[0] * nm->data_type_size);
		//checkCudaErrors(cudaMemcpy(nm->layer_input[0], r.data, nm->batch_size * nm->input_channels * nm->input_h * nm->input_w * nm->data_type_size, cudaMemcpyHostToDevice));
	}
	float alpha = 1.0, beta = 0.0;
	float Salpha = 1.0, Sbeta = 0.0;
	double Dalpha = 1.0, Dbeta = 0.0;
	size_t cur_workspace_size;
	void *cur_workspace;

	//testingg
	/*	cudaEvent_t s,e;
		float mss;
		cudaEventCreate(&s);
		cudaEventCreate(&e);
		cudaEventRecord(s, nm->stream_compute);
	 */
	nm->lockedcnmemMalloc(&(nm->layer_input[i + 1]), nm->layer_input_size[i + 1] * nm->data_type_size, NULL);
	/*	cudaEventRecord(e, nm->stream_compute);
		cudaEventSynchronize(e);
		cudaEventElapsedTime(&mss,s,e);
		printf("%f :",mss);
	 */
	space_tracker.updateSpace(CnmemSpace::SUB, nm->layer_input_size[i + 1] * nm->data_type_size);

	if (nm->layer_type[i] == CONV)
	{
		tp->type = 'C';
		// std::cout << "conv\n";
		ConvLayerParams *cur_params = (ConvLayerParams *)nm->params[i];

		cur_workspace_size = cur_params->fwd_workspace_size;
		nm->lockedcnmemMalloc(&cur_workspace, cur_workspace_size, NULL);
		// computation
		/*		checkCUDNN(cudnnConvolutionForward(nm->cudnn_handle, &alpha, 
				cur_params->input_tensor, nm->layer_input[i],
				cur_params->filter_desc, cur_params->W,
				cur_params->conv_desc, cur_params->fwd_algo,
				cur_workspace, cur_workspace_size,
				&beta,
				cur_params->output_tensor, nm->layer_input[i + 1]));
		 */
		//custom  coarsened cuda kernel
		customCoarsenedConvolutionForward((float *)nm->layer_input[i], (float *)nm->layer_input[i + 1], cur_params->conv_desc, cur_params->filter_desc, cur_params->input_tensor, (float *)cur_params->W, compute_stream);

		//Batch Normalization
		if (cur_params->bn == 1)
		{
			normalize_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->rolling_mean_gpu, (float *)cur_params->rolling_variance_gpu, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
			scale_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->scales_gpu, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
			add_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->b, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
		}
		else
		{
			add_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->b, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
		}
		/*--		checkCUDNN(cudnnAddTensor(nm->cudnn_handle, &alpha, 
		  cur_params->bias_desc, cur_params->b, 
		  &alpha,
		  cur_params->output_tensor, nm->layer_input[i + 1]));
		  --*/
		// if activation required
		if (cur_params->activation_mode != ACTIVATION_NONE)
		{
			//Replacing cuDNN call for relu to custom leaky relu call
			float *addr = (float *)(nm->layer_input[i + 1]);
			activate_array_gpu(addr, nm->layer_input_size[i + 1], compute_stream);

			/*checkCUDNN(cudnnActivationForward(nm->cudnn_handle, cur_params->actv_desc,
			  &alpha,
			  cur_params->output_tensor, nm->layer_input[i + 1],
			  &beta,
			  cur_params->output_tensor, nm->layer_input[i + 1]));
			 */
		}

		space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
		// std::cout << "Free bytes: " << free_bytes << std::endl;
	}
	else if (nm->layer_type[i] == FULLY_CONNECTED)
	{
		tp->type = 'F';
		// std::cout << "FC\n";
		FCLayerParams *cur_params = (FCLayerParams *)nm->params[i];
		// std::cout << "FChere" << i << std::endl;

		if (nm->data_type == CUDNN_DATA_FLOAT)
		{
			checkCUBLAS(cublasSgemm(cublas_handle,
									CUBLAS_OP_N, CUBLAS_OP_N,
									cur_params->C_out, nm->batch_size, cur_params->C_in,
									&Salpha,
									(float *)cur_params->W, cur_params->C_out,
									(float *)nm->layer_input[i], cur_params->C_in,
									&Sbeta,
									(float *)nm->layer_input[i + 1], cur_params->C_out));
			checkCUBLAS(cublasSgemm(cublas_handle,
									CUBLAS_OP_N, CUBLAS_OP_N,
									cur_params->C_out, nm->batch_size, 1,
									&Salpha,
									(float *)cur_params->b, cur_params->C_out,
									(float *)nm->one_vec, 1,
									&Salpha,
									(float *)nm->layer_input[i + 1], cur_params->C_out));
		}
		else if (nm->data_type == CUDNN_DATA_DOUBLE)
		{
			checkCUBLAS(cublasDgemm(cublas_handle,
									CUBLAS_OP_N, CUBLAS_OP_N,
									cur_params->C_out, nm->batch_size, cur_params->C_in,
									&Dalpha,
									(double *)cur_params->W, cur_params->C_out,
									(double *)nm->layer_input[i], cur_params->C_in,
									&Dbeta,
									(double *)nm->layer_input[i + 1], cur_params->C_out));
			checkCUBLAS(cublasDgemm(cublas_handle,
									CUBLAS_OP_N, CUBLAS_OP_N,
									cur_params->C_out, nm->batch_size, 1,
									&Dalpha,
									(double *)cur_params->b, cur_params->C_out,
									(double *)nm->one_vec, 1,
									&Dalpha,
									(double *)nm->layer_input[i + 1], cur_params->C_out));
		}
		if (cur_params->activation_mode != ACTIVATION_NONE)
		{
			//Replacing cuDNN call for Relu activation to custom Leaky Relu call
			//checkCUDNN(cudnnActivationForward(nm->cudnn_handle, cur_params->actv_desc,&alpha,cur_params->output_tensor, nm->layer_input[i + 1],&beta,cur_params->output_tensor, nm->layer_input[i + 1]));
			activate_array_gpu((float *)nm->layer_input[i + 1], nm->layer_input_size[i + 1], compute_stream);
		}
		// std::cout << "FChere" << i << std::endl;
	}

	else if (nm->layer_type[i] == DROPOUT)
	{
		tp->type = 'D';
		// std::cout << "Dropout\n";
		DropoutLayerParams *cur_params = (DropoutLayerParams *)nm->params[i];
		checkCUDNN(cudnnDropoutForward(cudnn_handle, cur_params->dropout_desc,
									   cur_params->input_tensor, nm->layer_input[i],
									   cur_params->input_tensor, nm->layer_input[i + 1],
									   cur_params->reserved_space,
									   cur_params->reserved_space_size));
	}
	else if (nm->layer_type[i] == BATCHNORM)
	{
		tp->type = 'B';
		// std::cout << "Batchnorm\n";
		BatchNormLayerParams *cur_params = (BatchNormLayerParams *)nm->params[i];

		checkCUDNN(cudnnBatchNormalizationForwardInference(cudnn_handle, cur_params->mode,
														   &alpha, &beta,
														   cur_params->input_tensor, nm->layer_input[i], cur_params->input_tensor, nm->layer_input[i + 1], cur_params->sbmv_desc,
														   cur_params->scale, cur_params->bias,
														   cur_params->running_mean, cur_params->running_variance,
														   cur_params->epsilon));
	}
	else if (nm->layer_type[i] == POOLING)
	{
		tp->type = 'P';
		// std::cout << "Pooling\n";
		PoolingLayerParams *cur_params = (PoolingLayerParams *)nm->params[i];
		checkCUDNN(cudnnPoolingForward(cudnn_handle, cur_params->pool_desc,
									   &alpha,
									   cur_params->input_tensor, nm->layer_input[i],
									   &beta,
									   cur_params->output_tensor, nm->layer_input[i + 1]));
	}
	else if (nm->layer_type[i] == ACTV)
	{
		tp->type = 'A';
		ActivationLayerParams *cur_params = (ActivationLayerParams *)nm->params[i];
		checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
										  &alpha,
										  cur_params->input_tensor, nm->layer_input[i],
										  &beta,
										  cur_params->input_tensor, nm->layer_input[i + 1]));
	}
	else if (nm->layer_type[i] == SOFTMAX)
	{
		tp->type = 'S';
		SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)nm->params[i];
		//custom kernel call for softmax
		softmax_gpu((float *)nm->layer_input[i], cur_params->channels, cur_params->channels, (nm->layer_input_size[i]) / cur_params->channels, (cur_params->w) * (cur_params->h), 1, (cur_params->w) * (cur_params->h), 1, (float *)nm->layer_input[i + 1]);
		//cuDNN kernel call for Softmax
		/*checkCUDNN(cudnnSoftmaxForward(nm->cudnn_handle, cur_params->algo, cur_params->mode,
					&alpha,
					cur_params->input_tensor, nm->layer_input[i],
					&beta,
					cur_params->input_tensor, nm->layer_input[i + 1]));*/
		//-Copy the result produced by softmax layer from GPU to CPU

		//checkCudaErrors(cudaStreamSynchronize(nm->stream_compute)); /////-----check....
		float *result = (float *)malloc(nm->layer_input_size[i + 1] * sizeof(float));
		checkCudaErrors(cudaMemcpy(result, nm->layer_input[i + 1], nm->layer_input_size[i + 1] * sizeof(float), cudaMemcpyDeviceToHost));

		//Infer the output class
		//	int *correct_count=0;
		//	nm->compareOutputCorrect(correct_count,nm->y);
		//	checkCNMEM(cnmemFree(nm->layer_input[nm->num_layers - 1], NULL));
		//	space_tracker.updateSpace(CnmemSpace::ADD, nm->layer_input_size[nm->num_layers - 1] * nm->data_type_size);
		//--
		int top = 5;
		list *options = read_data_cfg("data/imagenet1k.data"); //specify name  of the file
		char *name_list = option_find_str(options, "names", 0);
		if (!name_list)
			name_list = option_find_str(options, "labels", "data/labels.list");
		if (top == 0)
			top = option_find_int(options, "top", 1);

		int ii = 0;
		char **names = get_labels(name_list);
		//    clock_t time;
		int *indexes = (int *)calloc(top, sizeof(int));
		// time=clock();
		top_k(result, nm->layer_input_size[i + 1], top, indexes); //check parameters of this function
		// fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
		for (ii = 0; ii < top; ++ii)
		{
			//int index = indexes[ii];
			//if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
			//else printf("%s: %f\n",names[index], predictions[index]);
			//printf("index is %d: %5.2f%%: %s\n",index, result[index]*100, names[index]);
			//printf("index is %d: %s\n",index, names[index]);
		}
	}
	if (nm->layer_type[i] == CONV)
	{
		nm->lockedcnmemFree(cur_workspace, NULL);
		space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
	}

	//kCudaErrors(cudaStreamSynchronize(nm->stream_compute));
	//free the memory allocated to layer_input[i]
	//nm->lockedcnmemFree(nm->layer_input[i], NULL);
	//space_tracker.updateSpace(CnmemSpace::ADD, nm->layer_input_size[i] * nm->data_type_size);
}
/*void ScheduleEngine::startPrefetchWeights(NeuralNet *nm){

  cudaEventCreate(&op.mem_op_start);
  cudaEventCreate(&op.mem_op_stop);
  std::map<int, Operation>::iterator it;

  for(it=pipe.begin();it!=pipe.end();it++){
  cudaEventRecord((it->second).mem_op_start, (op.model)->stream_memory);
  (pipe[0].model)->prefetchWeights(it->first);
  cudaEventRecord(op.mem_op_stop, (op.model)->stream_memory);
  }
  }
 */

void ScheduleEngine::startPrefetchWeights(NeuralNet *nm, int nos_layers_to_prefetch, cudaStream_t &memory_stream)
{

	int i, j;
	for (i = nm->cur_prefetch_layer, j = 0; i < (nm->num_layers) && j < nos_layers_to_prefetch; i++, j++)
	{
		checkCudaErrors(cudaEventRecord(nm->event_mem_op_start[i], memory_stream));
		nm->prefetchWeights(i, memory_stream);
		checkCudaErrors(cudaEventRecord(nm->event_mem_op_end[i], memory_stream));
		nm->cur_prefetch_layer += 1;
	}
}

//Schedule fucntion for profiling Co-Scheduling table
void ScheduleEngine::schedule_profile(vector<Operation> &p1, vector<Operation> &p2)
{
	int nP1 = p1.size(); //Number of compute operations in Pipeline1
	int nP2 = p2.size(); //Number of compute operations in Pipeline2
	int per;
	float y1, y2;
	FILE *cofp;
	Operation *l1, *l2;
	cofp = fopen("coSched.txt", "w");
	cudaEvent_t global_start;
	cudaEventCreate(&global_start);
	cudaEventRecord(global_start);
	//fprintf(cofp,"layer ");
	//for(i=0;i<nP1;i++)
	//	fprintf(cofp,"%6d",i);
	//fprintf(cofp,"\n");
	for (int i = 0; i < nP1; i++)
	{
		l1 = &(p1[i]);
		checkCudaErrors(cudaEventCreate(&(l1->startop)));
		checkCudaErrors(cudaEventCreate(&(l1->endop)));
		//fprintf(cofp,"layer %d   :",i);
		for (int j = 0; j < nP2; j++)
		{
			//Layer 0 for pipeline 1 and pipline2
			l2 = &(p2[j]);
			//create events
			checkCudaErrors(cudaEventCreate(&(l2->startop)));
			checkCudaErrors(cudaEventCreate(&(l2->endop)));

			checkCudaErrors(cudaEventRecord(l1->startop, compute_streams[HIGH_COMPUTE_STREAM]));
			execute(l1, HIGH_COMPUTE_STREAM);
			checkCudaErrors(cudaEventRecord(l1->endop, compute_streams[HIGH_COMPUTE_STREAM]));

			checkCudaErrors(cudaEventRecord(l2->startop, compute_streams[LOW_COMPUTE_STREAM]));
			execute(l2, LOW_COMPUTE_STREAM);
			checkCudaErrors(cudaEventRecord(l2->endop, compute_streams[LOW_COMPUTE_STREAM]));

			checkCudaErrors(cudaEventSynchronize(l1->startop));
			checkCudaErrors(cudaEventElapsedTime(&(l1->time_to_start), global_start, l1->startop));
			checkCudaErrors(cudaEventSynchronize(l1->endop));
			cudaEventElapsedTime(&(l1->time_to_execute), l1->startop, l1->endop);

			checkCudaErrors(cudaEventSynchronize(l2->startop));
			checkCudaErrors(cudaEventElapsedTime(&(l2->time_to_start), global_start, l2->startop));
			checkCudaErrors(cudaEventSynchronize(l2->endop));
			cudaEventElapsedTime(&(l2->time_to_execute), l2->startop, l2->endop);

			y1 = l1->time_to_execute;
			cudaEventElapsedTime(&y2, l2->startop, l1->endop);

			if (y2 < 0)
				per = 0;
			else
				per = (int)(100 * y2) / y1;

			/*if(per > 0)
				fprintf(cofp,"S(%d) ",per);
			else 
				fprintf(cofp,"NS    ");
			*/
			if (per > 70)
				fprintf(cofp, "1 ");
			else
				fprintf(cofp, "0 ");

			checkCudaErrors(cudaStreamSynchronize(compute_streams[HIGH_COMPUTE_STREAM]));
			checkCudaErrors(cudaStreamSynchronize(compute_streams[LOW_COMPUTE_STREAM]));
		}
		fprintf(cofp, "\n");
	}
	for (int i = 0; i < nP1; i++)
	{
		l1 = &(p1[i]);
		checkCudaErrors(cudaEventCreate(&(l1->startop)));
		checkCudaErrors(cudaEventCreate(&(l1->endop)));
		//fprintf(cofp,"layer %d   :",i);
		for (int j = 0; j < nP2; j++)
		{
			//Layer 0 for pipeline 1 and pipline2
			l2 = &(p2[j]);
			//create events
			checkCudaErrors(cudaEventCreate(&(l2->startop)));
			checkCudaErrors(cudaEventCreate(&(l2->endop)));

			checkCudaErrors(cudaEventRecord(l1->startop, compute_streams[LOW_COMPUTE_STREAM]));
			execute(l1, LOW_COMPUTE_STREAM);
			checkCudaErrors(cudaEventRecord(l1->endop, compute_streams[LOW_COMPUTE_STREAM]));

			checkCudaErrors(cudaEventRecord(l2->startop, compute_streams[HIGH_COMPUTE_STREAM]));
			execute(l2, HIGH_COMPUTE_STREAM);
			checkCudaErrors(cudaEventRecord(l2->endop, compute_streams[HIGH_COMPUTE_STREAM]));

			checkCudaErrors(cudaEventSynchronize(l1->startop));
			checkCudaErrors(cudaEventElapsedTime(&(l1->time_to_start), global_start, l1->startop));
			checkCudaErrors(cudaEventSynchronize(l1->endop));
			cudaEventElapsedTime(&(l1->time_to_execute), l1->startop, l1->endop);

			checkCudaErrors(cudaEventSynchronize(l2->startop));
			checkCudaErrors(cudaEventElapsedTime(&(l2->time_to_start), global_start, l2->startop));
			checkCudaErrors(cudaEventSynchronize(l2->endop));
			cudaEventElapsedTime(&(l2->time_to_execute), l2->startop, l2->endop);

			y1 = l1->time_to_execute;
			cudaEventElapsedTime(&y2, l2->startop, l1->endop);

			if (y2 < 0)
				per = 0;
			else
				per = (int)(100 * y2) / y1;

			/*if(per > 0)
				fprintf(cofp,"S(%d) ",per);
			else 
				fprintf(cofp,"NS    ");
			*/
			if (per > 70)
				fprintf(cofp, "1 ");
			else
				fprintf(cofp, "0 ");

			checkCudaErrors(cudaStreamSynchronize(compute_streams[HIGH_COMPUTE_STREAM]));
			checkCudaErrors(cudaStreamSynchronize(compute_streams[LOW_COMPUTE_STREAM]));
		}
		fprintf(cofp, "\n");
	}
	fclose(cofp);
}
/*
//this is sequential scheduling function
void ScheduleEngine::schedule_sequential(InputOperation *zerothLayer, FILE *fpcf)
{
	//loops over all elements of prioriy Queue and dispatch all operations on GPU
	printf("Scheduling loop started\n");
	//FILE *fpcf = fopen("stats_mem_seq.txt","a");
	Operation *tp;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	/* cudaEventRecord(start, nm->stream_memory);
	nm->loadFile(filename);
	cudaEventRecord(stop, nm->stream_memory); 
	cudaEventSynchronize(stop); */
	/* checkCudaErrors(cudaEventElapsedTime(&time_to_start, global_start, start));
	cudaEventElapsedTime(&time_to_execute, start, stop);
	fprintf(fpcf, "%d:%d:M:%f:%f\n", 0, 0, time_to_start, time_to_execute);
	//start prefetching weights of  both pipelines
	startPrefetchWeights(nm, 1); 

	while (!Q.empty())
	{
		//pop element from queue
		tp = dequeue();
		//create events
		checkCudaErrors(cudaEventCreate(&(tp->startop)));
		checkCudaErrors(cudaEventCreate(&(tp->endop)));

		if (tp->op_type == 'c')
		{
			assert(tp->parents.back()->op_type == 'm');
			checkCudaErrors(cudaStreamWaitEvent(tp->model->stream_compute, tp->parents.back()->endop, 0));
			checkCudaErrors(cudaEventSynchronize(tp->parents.back()->startop));
			checkCudaErrors(cudaEventElapsedTime(&(tp->parents.back()->time_to_start), global_start, tp->parents.back()->startop));
			checkCudaErrors(cudaEventSynchronize(tp->parents.back()->endop));
			checkCudaErrors(cudaEventElapsedTime(&(tp->parents.back()->time_to_execute), tp->parents.back()->startop, tp->parents.back()->endop));
			checkCudaErrors(cudaEventRecord(tp->startop, tp->model->stream_compute));
			dispatch(tp);
			checkCudaErrors(cudaEventRecord(tp->endop, tp->model->stream_compute));
		}
		else if (tp->op_type == 'm')
		{
			if (tp->op_layer == 0)
			{
				InputOperation *zerothLayer = static_cast<InputOperation *>(tp);
				checkCudaErrors(cudaEventRecord(tp->startop, tp->model->stream_memory));
				zerothLayer->model->loadFile(const_cast<char *>((zerothLayer->filename).c_str()));
				checkCudaErrors(cudaEventRecord(tp->endop, tp->model->stream_memory));
			}
			else
			{
				checkCudaErrors(cudaEventRecord(tp->startop, tp->model->stream_memory));
				tp->model->prefetchWeights(tp->op_layer);
				checkCudaErrors(cudaEventRecord(tp->endop, tp->model->stream_memory));
			}
		}
		timeQ.push(tp);
	}

	while (!timeQ.empty())
	{
		tp = timeQ.front();
		timeQ.pop();
		checkCudaErrors(cudaEventSynchronize(tp->startop));
		checkCudaErrors(cudaEventElapsedTime(&(tp->time_to_start), global_start, tp->startop));
		checkCudaErrors(cudaEventSynchronize(tp->endop));
		checkCudaErrors(cudaEventElapsedTime(&(tp->time_to_execute), tp->startop, tp->endop));
		// fprintf(fpcf, "%d:%d:M:%f:%f\n", tp.pipeline, tp.op_layer * 2 + 1, tp.time_to_start_mo, tp.time_to_execute_mo);
		fprintf(fpcf, "%d:%c:%d:%c:%f:%f\n", tp->pipeline, tp->op_type, tp->op_layer * 2 + (tp->op_type == 'm' ? 1 : 2), tp->type, tp->time_to_start, tp->time_to_execute);
	}
	checkCudaErrors(cudaStreamSynchronize(tp->model->stream_compute));
}

//this is parallel scheduling function
/*
void ScheduleEngine::schedule()
{
	//loops over all elements of prioriy Queue and dispatch all operations on GPU
	printf("Scheduling loop started\n");
	FILE *fpcf = fopen("stats_mem.txt", "w");
	Operation tp;
	cudaEvent_t global_start;
	cudaEventCreate(&global_start);
<<<<<<< HEAD
	cudaEventRecord(global_start);	

	//start prefetching weights of  both pipelines
	startPrefetchWeights(model1,1);
	startPrefetchWeights(model2,1);

	while(!Q.empty()){
>>>>>>> DAG
		//pop element from queue
		tp = dequeue();
		//create events
		checkCudaErrors(cudaEventCreate(&tp.startop));
		checkCudaErrors(cudaEventCreate(&tp.endop));

		//wait for prefetch is done for layer and calculate the time required to finish mem_operation
		cudaEventElapsedTime(&tp.time_to_start_mo, global_start, (tp.model)->event_mem_op_start[tp.op_layer]);
		cudaEventElapsedTime(&tp.time_to_execute_mo, (tp.model)->event_mem_op_start[tp.op_layer], (tp.model)->event_mem_op_end[tp.op_layer]);

		checkCudaErrors(cudaEventRecord(tp.startop, (tp.model)->stream_compute));
		//	printf("Scheduling %d layer:", tp.op_layer);
		dispatch(&tp);
		checkCudaErrors(cudaEventRecord(tp.endop, (tp.model)->stream_compute));
		//--printf("Operation of %d layer completed on GPU\n",tp.op_layer);
		//add the operation object in timeQ queue;
		timeQ.push(tp);
		if (tp.model == model1)
			startPrefetchWeights(model2, 1);
		else
			startPrefetchWeights(model1, 1);
	}

	while (!timeQ.empty())
	{
		tp = timeQ.front();
		timeQ.pop();
		checkCudaErrors(cudaEventSynchronize(tp.startop));
		checkCudaErrors(cudaEventElapsedTime(&tp.time_to_start, global_start, tp.startop));
		checkCudaErrors(cudaEventSynchronize(tp.endop));
		cudaEventElapsedTime(&tp.time_to_execute, tp.startop, tp.endop);
		fprintf(fpcf, "%d:%d:%c:%f:%f:%f:%f \n", tp.pipeline, tp.op_layer, tp.type, tp.time_to_start_mo, tp.time_to_execute_mo, tp.time_to_start, tp.time_to_execute);
	}
	checkCudaErrors(cudaStreamSynchronize(model1->stream_compute));
	checkCudaErrors(cudaStreamSynchronize(model2->stream_compute));
	fclose(fpcf);
}
*/
void ScheduleEngine::warmup_schedule(InputOperation *zerothLayer)
{
	startPrefetchWeights(zerothLayer->model, zerothLayer->model->num_layers, compute_streams[LOW_COMPUTE_STREAM]);
	Operation *tp;
	while (!Q.empty())
	{
		tp = dequeue();
		if (tp->op_layer == 0)
		{
			tp->model->loadFile(const_cast<char *>((zerothLayer->filename).c_str()));
		}
		if (tp->op_type == 'm')
			continue;
		dispatch(tp, LOW_COMPUTE_STREAM);
	}
	checkCudaErrors(cudaStreamSynchronize(compute_streams[LOW_COMPUTE_STREAM]));
}

void ScheduleEngine::createGlobalEvent()
{
	cudaEventCreate(&global_start);
	cudaEventRecord(global_start);
}