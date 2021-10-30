#include <assert.h>
#include <cstdlib>
#include <cmath>
#include "ScheduleEngine.h"
#include <utilities.h>

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

	checkCudaErrors(cudaStreamCreateWithPriority(&compute_streams[HIGH_COMPUTE_STREAM], cudaStreamDefault, greatestPriority));
	checkCudaErrors(cudaStreamCreateWithPriority(&compute_streams[LOW_COMPUTE_STREAM], cudaStreamDefault, leastPriority));
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
	tp = Q.front();
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
	int i = tp->op_layer - 1;

	//printf("In Execute funtion: Processing %d layer\n",i);
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

	nm->lockedcnmemMalloc(&(nm->layer_input[i + 1]), nm->layer_input_size[i+ 1] * nm->data_type_size,NULL);
	space_tracker.updateSpace(CnmemSpace::SUB, nm->layer_input_size[i + 1] * nm->data_type_size);

	if (nm->layer_type[i] == CONV)
	{
		tp->type = 'C';
		// std::cout << "conv\n";
		ConvLayerParams *cur_params = (ConvLayerParams *)nm->params[i];

		cur_workspace_size = cur_params->fwd_workspace_size;
		nm->lockedcnmemMalloc(&cur_workspace, cur_workspace_size, compute_stream); // compute stream or memory stream?
		// computation
		checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha,
				cur_params->input_tensor, nm->layer_input[i],
				cur_params->filter_desc, cur_params->W,
				cur_params->conv_desc, cur_params->fwd_algo,
				cur_workspace, cur_workspace_size,
				&beta,
				cur_params->output_tensor, nm->layer_input[i + 1]));

		//custom  coarsened cuda kernel
		/* customCoarsenedConvolutionForward((float *)nm->layer_input[i], (float *)nm->layer_input[i + 1], cur_params->conv_desc, cur_params->filter_desc, cur_params->input_tensor, (float *)cur_params->W, compute_stream);

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
		} */
		checkCUDNN(cudnnAddTensor(cudnn_handle, &alpha,
		  cur_params->bias_desc, cur_params->b,
		  &alpha,
		  cur_params->output_tensor, nm->layer_input[i + 1]));

		// if activation required
		if (cur_params->activation_mode != ACTIVATION_NONE)
		{
			//Replacing cuDNN call for relu to custom leaky relu call
			// float *addr = (float *)(nm->layer_input[i + 1]);
			// activate_array_gpu(addr, nm->layer_input_size[i + 1], compute_stream);

			checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
			  &alpha,
			  cur_params->output_tensor, nm->layer_input[i + 1],
			  &beta,
			  cur_params->output_tensor, nm->layer_input[i + 1]));

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
			checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,&alpha,cur_params->output_tensor, nm->layer_input[i + 1],&beta,cur_params->output_tensor, nm->layer_input[i + 1]));
			//activate_array_gpu((float *)nm->layer_input[i + 1], nm->layer_input_size[i + 1], compute_stream);
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
	}else if(nm->layer_type[i] == REGION) { // Processing of region layer
		tp->type = 'R';
		//printf("Processing region layer %d",i);
		//printf("Input layer size is %d output layer size is %d\n", nm->layer_input_size[i], nm->layer_input_size[i+1]);
		RegionLayerParams *cur_params =(RegionLayerParams *)nm->params[i];
		//printf("Batch size is %d\n", cur_params->batch_size);
		forward_region_layer_gpu(nm->layer_input_size[i], nm->layer_input_size[i+1], (float *)nm->layer_input[i],cur_params->batch_size, cur_params->height, cur_params->width, cur_params->num,cur_params->classes,cur_params->coords,(float*)nm->layer_input[i+1], compute_stream);

		float *result=(float *)malloc(nm->layer_input_size[i+1]*sizeof(float));
		checkCudaErrors(cudaMemcpy(result, nm->layer_input[i+1], nm->layer_input_size[i+1]*sizeof(float), cudaMemcpyDeviceToHost));

		//int nbox=0;
		//newly added block
    		//--detection *dets = make_network_boxes(cur_params,0.5, &nbox);
    		//--fill_network_boxes(cur_params,nm->img_w,nm->img_h, 0.5,0, dets, result, nm->layer_input_size[i+1], nm->input_w, nm->input_h);
       		 //print_detector_detections(fps, id, dets, num, classes, w, h);
		//----list *options = read_data_cfg("cfg/coco.data");
    		//char *name_list = option_find_str(options, "names", "data/names.list");
    		//----char *name_list = option_find_str(options, "names", "data/coco.names");
   		//--char **names = get_labels("data/coco.names");
    		//--image **alphabet = load_alphabet();
        	//--draw_detections(nm->im, dets, nbox, 0.5, names, alphabet, cur_params->classes);
            	//--save_image(nm->im, "predictions");
            	//--free_detections(dets, nbox);
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
	int i = tp->op_layer - 1;

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

	//Free memory for the previous layers
	if ( i > 0 ){
		printf("dispatch() function: In free memroy block:");
		printf("Layer being freed is  %d\n",i-1);
		//deallocate memory of the parent 
		Operation *prev_compute=(tp->parents.back())->parents.back();
		NeuralNet *nnet=prev_compute->model;
		nnet->lockedcnmemFree(
                nnet->layer_input[i-1], NULL);
		if (nnet->layer_type[i-1]==CONV){
			printf("Convolution layer parameters being freed: for layer: %d\n",i-1);
			((ConvLayerParams *) nnet->params[i-1]) ->deallocateSpace(
				nnet->free_bytes,nnet->data_type_size);
		}
	}

	//Allocate space for the next layer input
	nm->lockedcnmemMalloc(&(nm->layer_input[i + 1]), nm->layer_input_size[i + 1] * nm->data_type_size, NULL);

	space_tracker.updateSpace(CnmemSpace::SUB, nm->layer_input_size[i + 1] * nm->data_type_size);

	if (nm->layer_type[i] == CONV)
	{
		tp->type = 'C';
		// std::cout << "conv\n";
		ConvLayerParams *cur_params = (ConvLayerParams *)nm->params[i];

		cur_workspace_size = cur_params->fwd_workspace_size;
		nm->lockedcnmemMalloc(&cur_workspace, cur_workspace_size, NULL);
		// computation
		checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha,
				cur_params->input_tensor, nm->layer_input[i],
				cur_params->filter_desc, cur_params->W,
				cur_params->conv_desc, cur_params->fwd_algo,
				cur_workspace, cur_workspace_size,
				&beta,
				cur_params->output_tensor, nm->layer_input[i + 1]));

		//custom  coarsened cuda kernel
		//customCoarsenedConvolutionForward((float *)nm->layer_input[i], (float *)nm->layer_input[i + 1], cur_params->conv_desc, cur_params->filter_desc, cur_params->input_tensor, (float *)cur_params->W, compute_stream);

		//Batch Normalization
		/* if (cur_params->bn == 1)
		{
			normalize_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->rolling_mean_gpu, (float *)cur_params->rolling_variance_gpu, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
			scale_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->scales_gpu, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
			add_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->b, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
		}
		else
		{
			add_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->b, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
		} */
		checkCUDNN(cudnnAddTensor(cudnn_handle, &alpha,
		  cur_params->bias_desc, cur_params->b,
		  &alpha,
		  cur_params->output_tensor, nm->layer_input[i + 1]));

		// if activation required
		if (cur_params->activation_mode != ACTIVATION_NONE)
		{
			//Replacing cuDNN call for relu to custom leaky relu call
			//float *addr = (float *)(nm->layer_input[i + 1]);
			//activate_array_gpu(addr, nm->layer_input_size[i + 1], compute_stream);

			checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
			  &alpha,
			  cur_params->output_tensor, nm->layer_input[i + 1],
			  &beta,
			  cur_params->output_tensor, nm->layer_input[i + 1]));
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
			checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,&alpha,cur_params->output_tensor, nm->layer_input[i + 1],&beta,cur_params->output_tensor, nm->layer_input[i + 1]));
			//activate_array_gpu((float *)nm->layer_input[i + 1], nm->layer_input_size[i + 1], compute_stream);
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
	else if(nm->layer_type[i] == REGION) { // Processing of region layer
		tp->type = 'R';
		//printf("Processing region layer %d",i);
		//printf("Input layer size is %d output layer size is %d\n", nm->layer_input_size[i], nm->layer_input_size[i+1]);
		RegionLayerParams *cur_params =(RegionLayerParams *)nm->params[i];
		//printf("Batch size is %d\n", cur_params->batch_size);
		forward_region_layer_gpu(nm->layer_input_size[i], nm->layer_input_size[i+1], (float *)nm->layer_input[i],cur_params->batch_size, cur_params->height, cur_params->width, cur_params->num,cur_params->classes,cur_params->coords,(float*)nm->layer_input[i+1], compute_stream);

		float *result=(float *)malloc(nm->layer_input_size[i+1]*sizeof(float));
		checkCudaErrors(cudaMemcpy(result, nm->layer_input[i+1], nm->layer_input_size[i+1]*sizeof(float), cudaMemcpyDeviceToHost));

		//int nbox=0;
		//newly added block
    		//--detection *dets = make_network_boxes(cur_params,0.5, &nbox);
    		//--fill_network_boxes(cur_params,nm->img_w,nm->img_h, 0.5,0, dets, result, nm->layer_input_size[i+1], nm->input_w, nm->input_h);
       		 //print_detector_detections(fps, id, dets, num, classes, w, h);
		//----list *options = read_data_cfg("cfg/coco.data");
    		//char *name_list = option_find_str(options, "names", "data/names.list");
    		//----char *name_list = option_find_str(options, "names", "data/coco.names");
   		//--char **names = get_labels("data/coco.names");
    		//--image **alphabet = load_alphabet();
        	//--draw_detections(nm->im, dets, nbox, 0.5, names, alphabet, cur_params->classes);
            	//--save_image(nm->im, "predictions");
            	//--free_detections(dets, nbox);
	}
	else if (nm->layer_type[i] == SOFTMAX)
	{
		tp->type = 'S';
		SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)nm->params[i];
		//custom kernel call for softmax
		//softmax_gpu((float *)nm->layer_input[i], cur_params->channels, cur_params->channels, (nm->layer_input_size[i]) / cur_params->channels, (cur_params->w) * (cur_params->h), 1, (cur_params->w) * (cur_params->h), 1, (float *)nm->layer_input[i + 1], compute_stream);
		//cuDNN kernel call for Softmax
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
	//nm->lockedcnmemFree(nm->layer_input[i], NULL);
	//space_tracker.updateSpace(CnmemSpace::ADD, nm->layer_input_size[i] * nm->data_type_size);
}

//Dispatch function for warmup code
//Dispatch opration will execute the operation on GPU. The operation performed is based on the type of layer
void ScheduleEngine::dispatch_warmup(Operation *tp, stream_indicator streamIndicator)
{
	cudaStream_t &compute_stream = compute_streams[streamIndicator];
	cudnnHandle_t &cudnn_handle = cudnnHandles[streamIndicator];
	cublasHandle_t &cublas_handle = cublasHandles[streamIndicator];
	//int priority=tp->priority;	// to be used later
	int i = tp->op_layer - 1;

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

	//Free memory for the previous layers
	if ( i > 0 ){
		printf("dispatch() function: In free memroy block:");
		printf("Layer being freed is  %d\n",i-1);
		//deallocate memory of the parent
		nm->lockedcnmemFree(
                nm->layer_input[i-1], compute_stream);
		if (nm->layer_type[i-1]==CONV){
			printf("Convolution layer parameters being freed: for layer: %d\n",i-1);
			((ConvLayerParams *) nm->params[i-1]) ->deallocateSpace(
				nm->free_bytes,nm->data_type_size);
		}
	}

	//Allocate space for the next layer input
	nm->lockedcnmemMalloc(&(nm->layer_input[i + 1]), nm->layer_input_size[i + 1] * nm->data_type_size, compute_stream);

	space_tracker.updateSpace(CnmemSpace::SUB, nm->layer_input_size[i + 1] * nm->data_type_size);

	if (nm->layer_type[i] == CONV)
	{
		tp->type = 'C';
		// std::cout << "conv\n";
		ConvLayerParams *cur_params = (ConvLayerParams *)nm->params[i];

		cur_workspace_size = cur_params->fwd_workspace_size;
		nm->lockedcnmemMalloc(&cur_workspace, cur_workspace_size, NULL);
		// computation
		checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha,
				cur_params->input_tensor, nm->layer_input[i],
				cur_params->filter_desc, cur_params->W,
				cur_params->conv_desc, cur_params->fwd_algo,
				cur_workspace, cur_workspace_size,
				&beta,
				cur_params->output_tensor, nm->layer_input[i + 1]));

		//custom  coarsened cuda kernel
		//customCoarsenedConvolutionForward((float *)nm->layer_input[i], (float *)nm->layer_input[i + 1], cur_params->conv_desc, cur_params->filter_desc, cur_params->input_tensor, (float *)cur_params->W, compute_stream);

		//Batch Normalization
		/* if (cur_params->bn == 1)
		{
			normalize_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->rolling_mean_gpu, (float *)cur_params->rolling_variance_gpu, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
			scale_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->scales_gpu, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
			add_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->b, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
		}
		else
		{
			add_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->b, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
		} */
		checkCUDNN(cudnnAddTensor(cudnn_handle, &alpha,
		  cur_params->bias_desc, cur_params->b,
		  &alpha,
		  cur_params->output_tensor, nm->layer_input[i + 1]));

		// if activation required
		if (cur_params->activation_mode != ACTIVATION_NONE)
		{
			//Replacing cuDNN call for relu to custom leaky relu call
			//float *addr = (float *)(nm->layer_input[i + 1]);
			//activate_array_gpu(addr, nm->layer_input_size[i + 1], compute_stream);

			checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
			  &alpha,
			  cur_params->output_tensor, nm->layer_input[i + 1],
			  &beta,
			  cur_params->output_tensor, nm->layer_input[i + 1]));
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
			checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,&alpha,cur_params->output_tensor, nm->layer_input[i + 1],&beta,cur_params->output_tensor, nm->layer_input[i + 1]));
			//activate_array_gpu((float *)nm->layer_input[i + 1], nm->layer_input_size[i + 1], compute_stream);
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
	else if(nm->layer_type[i] == REGION) { // Processing of region layer
		tp->type = 'R';
		//printf("Processing region layer %d",i);
		//printf("Input layer size is %d output layer size is %d\n", nm->layer_input_size[i], nm->layer_input_size[i+1]);
		RegionLayerParams *cur_params =(RegionLayerParams *)nm->params[i];
		//printf("Batch size is %d\n", cur_params->batch_size);
		forward_region_layer_gpu(nm->layer_input_size[i], nm->layer_input_size[i+1], (float *)nm->layer_input[i],cur_params->batch_size, cur_params->height, cur_params->width, cur_params->num,cur_params->classes,cur_params->coords,(float*)nm->layer_input[i+1], compute_stream);

		float *result=(float *)malloc(nm->layer_input_size[i+1]*sizeof(float));
		checkCudaErrors(cudaMemcpy(result, nm->layer_input[i+1], nm->layer_input_size[i+1]*sizeof(float), cudaMemcpyDeviceToHost));

		//int nbox=0;
		//newly added block
    		//--detection *dets = make_network_boxes(cur_params,0.5, &nbox);
    		//--fill_network_boxes(cur_params,nm->img_w,nm->img_h, 0.5,0, dets, result, nm->layer_input_size[i+1], nm->input_w, nm->input_h);
       		 //print_detector_detections(fps, id, dets, num, classes, w, h);
		//----list *options = read_data_cfg("cfg/coco.data");
    		//char *name_list = option_find_str(options, "names", "data/names.list");
    		//----char *name_list = option_find_str(options, "names", "data/coco.names");
   		//--char **names = get_labels("data/coco.names");
    		//--image **alphabet = load_alphabet();
        	//--draw_detections(nm->im, dets, nbox, 0.5, names, alphabet, cur_params->classes);
            	//--save_image(nm->im, "predictions");
            	//--free_detections(dets, nbox);
	}
	else if (nm->layer_type[i] == SOFTMAX)
	{
		tp->type = 'S';
		SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)nm->params[i];
		//custom kernel call for softmax
		softmax_gpu((float *)nm->layer_input[i], cur_params->channels, cur_params->channels, (nm->layer_input_size[i]) / cur_params->channels, (cur_params->w) * (cur_params->h), 1, (cur_params->w) * (cur_params->h), 1, (float *)nm->layer_input[i + 1], compute_stream);
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





//dispatch_new function for SMT simulation
void ScheduleEngine::dispatch_new(Operation *tp, stream_indicator streamIndicator)
{
	cudaStream_t &compute_stream = compute_streams[streamIndicator];
	cudnnHandle_t &cudnn_handle = cudnnHandles[streamIndicator];
	cublasHandle_t &cublas_handle = cublasHandles[streamIndicator];
	//int priority=tp->priority;	// to be used later
	int i = tp->op_layer - 1;

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

	//Free memory for the previous layers
	if ( i > 0 ){
		printf("dispatch() function: In free memroy block:");
		printf("Layer being freed is  %d\n",i-1);
		//deallocate memory of the parent 
		Operation *prev_compute=(tp->parents.back())->parents.back();
		auto prev_computeStream =
                        (prev_compute->chosenStream == 'H')
                            ? ScheduleEngine::HIGH_COMPUTE_STREAM
                            : ScheduleEngine::LOW_COMPUTE_STREAM;
		NeuralNet *nnet=prev_compute->model;
		//check below statement
		nnet->lockedcnmemFree(
                nnet->layer_input[i-1], ScheduleEngine::compute_streams[prev_computeStream]);
		if (nnet->layer_type[i-1]==CONV){
			printf("Convolution layer parameters being freed: for layer: %d\n",i-1);
			((ConvLayerParams *) nnet->params[i-1]) ->deallocateSpace(
				nnet->free_bytes,nnet->data_type_size);
		}
	}

	//Allocate space for the next layer input
	Operation *next_compute= tp->children.back()->children.back();
	NeuralNet *nnet=next_compute->model;
	if (next_compute !=nullptr){
		auto next_computeStream =
                        (next_compute->chosenStream == 'H')
                            ? ScheduleEngine::HIGH_COMPUTE_STREAM
                            : ScheduleEngine::LOW_COMPUTE_STREAM;
		nnet->lockedcnmemMalloc(&(nnet->layer_input[i + 1]), nnet->layer_input_size[i + 1] * nnet->data_type_size, ScheduleEngine::compute_streams[next_computeStream]);
		space_tracker.updateSpace(CnmemSpace::SUB, nnet->layer_input_size[i + 1] * nnet->data_type_size);
	}
	else{
		nm->lockedcnmemMalloc(&(nm->layer_input[i + 1]), nm->layer_input_size[i + 1] * nm->data_type_size, compute_stream);
		space_tracker.updateSpace(CnmemSpace::SUB, nm->layer_input_size[i + 1] * nm->data_type_size);
	}
	
	if (nm->layer_type[i] == CONV)
	{
		tp->type = 'C';
		// std::cout << "conv\n";
		ConvLayerParams *cur_params = (ConvLayerParams *)nm->params[i];

		cur_workspace_size = cur_params->fwd_workspace_size;
		nm->lockedcnmemMalloc(&cur_workspace, cur_workspace_size, NULL);
		// computation
		checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha,
				cur_params->input_tensor, nm->layer_input[i],
				cur_params->filter_desc, cur_params->W,
				cur_params->conv_desc, cur_params->fwd_algo,
				cur_workspace, cur_workspace_size,
				&beta,
				cur_params->output_tensor, nm->layer_input[i + 1]));

		//custom  coarsened cuda kernel
		//customCoarsenedConvolutionForward((float *)nm->layer_input[i], (float *)nm->layer_input[i + 1], cur_params->conv_desc, cur_params->filter_desc, cur_params->input_tensor, (float *)cur_params->W, compute_stream);

		//Batch Normalization
		/* if (cur_params->bn == 1)
		{
			normalize_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->rolling_mean_gpu, (float *)cur_params->rolling_variance_gpu, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
			scale_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->scales_gpu, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
			add_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->b, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
		}
		else
		{
			add_bias_gpu((float *)nm->layer_input[i + 1], (float *)cur_params->b, 1, cur_params->C_out, cur_params->output_h * cur_params->output_w, compute_stream);
		} */
		checkCUDNN(cudnnAddTensor(cudnn_handle, &alpha,
		  cur_params->bias_desc, cur_params->b,
		  &alpha,
		  cur_params->output_tensor, nm->layer_input[i + 1]));

		// if activation required
		if (cur_params->activation_mode != ACTIVATION_NONE)
		{
			//Replacing cuDNN call for relu to custom leaky relu call
			//float *addr = (float *)(nm->layer_input[i + 1]);
			//activate_array_gpu(addr, nm->layer_input_size[i + 1], compute_stream);

			checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
			  &alpha,
			  cur_params->output_tensor, nm->layer_input[i + 1],
			  &beta,
			  cur_params->output_tensor, nm->layer_input[i + 1]));
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
			checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,&alpha,cur_params->output_tensor, nm->layer_input[i + 1],&beta,cur_params->output_tensor, nm->layer_input[i + 1]));
			//activate_array_gpu((float *)nm->layer_input[i + 1], nm->layer_input_size[i + 1], compute_stream);
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
	else if(nm->layer_type[i] == REGION) { // Processing of region layer
		tp->type = 'R';
		//printf("Processing region layer %d",i);
		//printf("Input layer size is %d output layer size is %d\n", nm->layer_input_size[i], nm->layer_input_size[i+1]);
		RegionLayerParams *cur_params =(RegionLayerParams *)nm->params[i];
		//printf("Batch size is %d\n", cur_params->batch_size);
		forward_region_layer_gpu(nm->layer_input_size[i], nm->layer_input_size[i+1], (float *)nm->layer_input[i],cur_params->batch_size, cur_params->height, cur_params->width, cur_params->num,cur_params->classes,cur_params->coords,(float*)nm->layer_input[i+1], compute_stream);

		float *result=(float *)malloc(nm->layer_input_size[i+1]*sizeof(float));
		checkCudaErrors(cudaMemcpy(result, nm->layer_input[i+1], nm->layer_input_size[i+1]*sizeof(float), cudaMemcpyDeviceToHost));

		//int nbox=0;
		//newly added block
    		//--detection *dets = make_network_boxes(cur_params,0.5, &nbox);
    		//--fill_network_boxes(cur_params,nm->img_w,nm->img_h, 0.5,0, dets, result, nm->layer_input_size[i+1], nm->input_w, nm->input_h);
       		 //print_detector_detections(fps, id, dets, num, classes, w, h);
		//----list *options = read_data_cfg("cfg/coco.data");
    		//char *name_list = option_find_str(options, "names", "data/names.list");
    		//----char *name_list = option_find_str(options, "names", "data/coco.names");
   		//--char **names = get_labels("data/coco.names");
    		//--image **alphabet = load_alphabet();
        	//--draw_detections(nm->im, dets, nbox, 0.5, names, alphabet, cur_params->classes);
            	//--save_image(nm->im, "predictions");
            	//--free_detections(dets, nbox);
	}
	else if (nm->layer_type[i] == SOFTMAX)
	{
		tp->type = 'S';
		SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)nm->params[i];
		//custom kernel call for softmax
		softmax_gpu((float *)nm->layer_input[i], cur_params->channels, cur_params->channels, (nm->layer_input_size[i]) / cur_params->channels, (cur_params->w) * (cur_params->h), 1, (cur_params->w) * (cur_params->h), 1, (float *)nm->layer_input[i + 1], compute_stream);
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


void ScheduleEngine::schedule_profile(InputOperation *z1, InputOperation *z2, vector<Operation *> &p1, vector<Operation *> &p2)
{
	int nP1 = p1.size(); //Number of compute operations in Pipeline1
	int nP2 = p2.size(); //Number of compute operations in Pipeline2
	FILE *cofp;
	ifstream fp("input/freship.txt");
	Operation *l1, *l2;
	float milliseconds = 0;
	fillExecutionTime(fp, {z1, z2});
	fp.close();

	cofp = fopen("output/coSched.txt", "w");
	cudaEvent_t global_start;
	cudaEventCreate(&global_start);
	cudaEventRecord(global_start);

	//load the input image and weight for both pipelines
	z1->model->loadFile(const_cast<char *>((z1->filename).c_str()), memoryStream);
	z2->model->loadFile(const_cast<char *>((z2->filename).c_str()), memoryStream);
	startPrefetchWeights(z1->model, z1->model->num_layers, memoryStream);
	startPrefetchWeights(z2->model, z2->model->num_layers, memoryStream);
	//wait for the completion of trasfer of weights
	cudaStreamSynchronize(memoryStream);
	cout << "\nFinished prefetching in schedule_profile()" << endl;

	cudaEvent_t start,end;


	for (int i = 0; i < nP1; i++)
	{
		l1 = p1[i];
		//fprintf(cofp,"layer %d   :",i);

		//fprintf(cofp,"layer %d   :",i);
		for (int j = 0; j < nP2; j++)
		{
			//Layer 0 for pipeline 1 and pipline2
			l2 = p2[j];
			//create events
			checkCudaErrors(cudaEventCreate(&(start)));
			checkCudaErrors(cudaEventCreate(&(end)));

			checkCudaErrors(cudaEventRecord(start));
			execute(l1, HIGH_COMPUTE_STREAM);
			execute(l2, LOW_COMPUTE_STREAM);
			checkCudaErrors(cudaEventRecord(end));
			checkCudaErrors(cudaEventSynchronize(end));

			checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, end));
			float per = (milliseconds/(l1->time_to_execute + l2->time_to_execute));


			/*if(per > 0)
				fprintf(cofp,"S(%d) ",per);
			else
				fprintf(cofp,"NS    ");
			*/

			if (per < 0.6)
				fprintf(cofp, "1 ");
			else
				fprintf(cofp, "0 ");

			checkCudaErrors(cudaStreamSynchronize(compute_streams[HIGH_COMPUTE_STREAM]));
			checkCudaErrors(cudaStreamSynchronize(compute_streams[LOW_COMPUTE_STREAM]));
			//Destroy events
			checkCudaErrors(cudaEventDestroy(start));
			checkCudaErrors(cudaEventDestroy(end));
		}
		fprintf(cofp, "\n");
	}
	for (int i = 0; i < nP1; i++)
	{
		l1 = p1[i];
		//fprintf(cofp,"layer %d   :",i);

		//fprintf(cofp,"layer %d   :",i);
		for (int j = 0; j < nP2; j++)
		{
			//Layer 0 for pipeline 1 and pipline2
			l2 = p2[j];
			//create events
			checkCudaErrors(cudaEventCreate(&(start)));
			checkCudaErrors(cudaEventCreate(&(end)));

			checkCudaErrors(cudaEventRecord(start));
			execute(l1, LOW_COMPUTE_STREAM);
			execute(l2, HIGH_COMPUTE_STREAM);
			checkCudaErrors(cudaEventRecord(end));
			checkCudaErrors(cudaEventSynchronize(end));

			checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, end));
			float per = (milliseconds/(l1->time_to_execute + l2->time_to_execute));


			/*if(per > 0)
				fprintf(cofp,"S(%d) ",per);
			else
				fprintf(cofp,"NS    ");
			*/

			if (per < 0.6)
				fprintf(cofp, "1 ");
			else
				fprintf(cofp, "0 ");

			checkCudaErrors(cudaStreamSynchronize(compute_streams[HIGH_COMPUTE_STREAM]));
			checkCudaErrors(cudaStreamSynchronize(compute_streams[LOW_COMPUTE_STREAM]));
			//Destroy events
			checkCudaErrors(cudaEventDestroy(start));
			checkCudaErrors(cudaEventDestroy(end));
		}
		fprintf(cofp, "\n");
	}
	fclose(cofp);
}


//this is sequential scheduling function
void ScheduleEngine::schedule_sequential(InputOperation *zerothLayer, FILE *fpcf)
{
	//loops over all elements of prioriy Queue and dispatch all operations on GPU
	printf("Scheduling loop started\n");
	//FILE *fpcf = fopen("stats_mem_seq.txt","a");
	Operation *tp = zerothLayer;
	float * dummy;

	checkCudaErrors( cudaMalloc( (void**)&dummy, 100 * sizeof(float) ) );
	checkCudaErrors( cudaFree( dummy ) );
	while (tp != nullptr)
	{
		//This may not be required
		// checkCudaErrors(cudaEventCreate(&(tp->startop)));
		// checkCudaErrors(cudaEventCreate(&(tp->endop)));

		if (tp->op_type == 'c')
		{
			assert(tp->parents.back()->op_type == 'm');
			//--checkCudaErrors(cudaStreamWaitEvent(memoryStream, tp->parents.back()->endop, 0));
			checkCudaErrors(cudaStreamWaitEvent(compute_streams[LOW_COMPUTE_STREAM], tp->parents.back()->endop, 0));
			checkCudaErrors(cudaStreamWaitEvent(compute_streams[LOW_COMPUTE_STREAM], tp->parents.back()->parents.back()->endop, 0));
			/* checkCudaErrors(cudaEventSynchronize(tp->parents.back()->startop));
			checkCudaErrors(cudaEventElapsedTime(&(tp->parents.back()->time_to_start), global_start, tp->parents.back()->startop));
			checkCudaErrors(cudaEventSynchronize(tp->parents.back()->endop));
			checkCudaErrors(cudaEventElapsedTime(&(tp->parents.back()->time_to_execute), tp->parents.back()->startop, tp->parents.back()->endop)); */
			checkCudaErrors(cudaEventRecord(tp->startop, compute_streams[LOW_COMPUTE_STREAM]));
			dispatch(tp, LOW_COMPUTE_STREAM);
			checkCudaErrors(cudaEventRecord(tp->endop, compute_streams[LOW_COMPUTE_STREAM]));
		}
		else if (tp->op_type == 'm')
		{
			/* if(tp->op_layer >= 2) checkCudaErrors(cudaStreamWaitEvent(compute_streams[LOW_COMPUTE_STREAM], tp->parents.back()->endop, 0));
			//if(tp->op_layer == 1) checkCudaErrors(cudaStreamWaitEvent(memoryStream, tp->parents.back()->endop, 0));
			if(tp->op_layer == 1) checkCudaErrors(cudaStreamWaitEvent(compute_streams[LOW_COMPUTE_STREAM], tp->parents.back()->endop, 0)); */
			tp->type='M';
			if (tp->op_layer == 0)
			{
				InputOperation *zerothLayer = static_cast<InputOperation *>(tp);
				/* checkCudaErrors(cudaEventRecord(tp->startop, memoryStream));
				zerothLayer->model->loadFile(const_cast<char *>((zerothLayer->filename).c_str()), memoryStream);
				checkCudaErrors(cudaEventRecord(tp->endop, memoryStream)); */
				checkCudaErrors(cudaEventRecord(tp->startop, compute_streams[LOW_COMPUTE_STREAM]));
				zerothLayer->model->loadFile(const_cast<char *>((zerothLayer->filename).c_str()), compute_streams[LOW_COMPUTE_STREAM]);
				checkCudaErrors(cudaEventRecord(tp->endop, compute_streams[LOW_COMPUTE_STREAM]));
			}
			else
			{
				/* checkCudaErrors(cudaEventRecord(tp->startop, memoryStream));
				tp->model->prefetchWeights(tp->op_layer - 1, memoryStream);
				checkCudaErrors(cudaEventRecord(tp->endop, memoryStream)); */
				checkCudaErrors(cudaEventRecord(tp->startop, compute_streams[LOW_COMPUTE_STREAM]));
				tp->model->prefetchWeights(tp->op_layer - 1, compute_streams[LOW_COMPUTE_STREAM]);
				checkCudaErrors(cudaEventRecord(tp->endop, compute_streams[LOW_COMPUTE_STREAM]));
			}
		}
		//check if tp is the last node in DAG
		if(tp->children.back() == nullptr){
			//free the last layer of the neural network
			tp->model->lockedcnmemFree(tp->model->layer_input[tp->model->num_layers-1], NULL);
		}
		tp = tp->children.back();	
		//timeQ.push(tp);
	}

	tp = zerothLayer;
	while (tp != nullptr)
	{
		checkCudaErrors(cudaEventSynchronize(tp->startop));
		checkCudaErrors(cudaEventElapsedTime(&(tp->time_to_start), global_start, tp->startop));
		checkCudaErrors(cudaEventSynchronize(tp->endop));
		checkCudaErrors(cudaEventElapsedTime(&(tp->time_to_execute), tp->startop, tp->endop));

		// fprintf(fpcf, "%d:%d:M:%f:%f\n", tp.pipeline, tp.op_layer * 2 + 1, tp.time_to_start_mo, tp.time_to_execute_mo);
		/* if(tp->op_layer==0)
			fprintf(fpcf, "%d:%c:%d:%c:%f:%f\n", tp->pipeline, tp->op_type, tp->op_layer, tp->type, tp->time_to_start, tp->time_to_execute);
		else
			fprintf(fpcf, "%d:%c:%d:%c:%f:%f\n", tp->pipeline, tp->op_type, (tp->op_layer-1) * 2 + (tp->op_type == 'm' ? 1 : 2), tp->type, tp->time_to_start, tp->time_to_execute);
 */
		if(tp->op_layer==0)
			fprintf(fpcf, "%d:%c:%d:%c:%f:%f\n", tp->pipeline, tp->op_type, tp->op_layer, tp->type, tp->time_to_start, tp->time_to_execute);
		else if( tp->children.back() != nullptr and  ((tp->children.back())->type == 'P' or (tp->children.back())->type == 'R') )
			fprintf(fpcf, "%d:%c:%d:%c:%f:0.000000\n", tp->pipeline, tp->op_type, (tp->op_layer-1) * 2 + (tp->op_type == 'm' ? 1 : 2), tp->type, tp->time_to_start);
		else
			fprintf(fpcf, "%d:%c:%d:%c:%f:%f\n", tp->pipeline, tp->op_type, (tp->op_layer-1) * 2 + (tp->op_type == 'm' ? 1 : 2), tp->type, tp->time_to_start, tp->time_to_execute);
		tp = tp->children.back();
	}
	checkCudaErrors(cudaStreamSynchronize(compute_streams[LOW_COMPUTE_STREAM]));
	//checkCudaErrors(cudaStreamSynchronize(memoryStream));
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
			tp->model->loadFile(const_cast<char *>((zerothLayer->filename).c_str()), compute_streams[LOW_COMPUTE_STREAM]);
		}
		if (tp->op_type == 'm')
			continue;
		dispatch(tp, LOW_COMPUTE_STREAM);
	}
	checkCudaErrors(cudaStreamSynchronize(compute_streams[LOW_COMPUTE_STREAM]));
	//free the last layer of the neural network
	tp->model->lockedcnmemFree(tp->model->layer_input[tp->model->num_layers-1],NULL );

}

void ScheduleEngine::createGlobalEvent()
{
	cudaEventCreate(&global_start);
	cudaEventRecord(global_start);
}
void ScheduleEngine::destroyGlobalEvents()
{
	cudaEventDestroy(global_start);
}
