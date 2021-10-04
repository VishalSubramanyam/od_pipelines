#include "coarsened_forward_convolution.h"
#include "image.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <dag.h>
#include <utilities.h>

using namespace std;
using timeMS = float;

int no_of_pipelines;
/* arrays, each element belongs to one pipeline each */
vector<cudaStream_t> stream;
vector<cudnnHandle_t> cudnnHandles;
vector<cublasHandle_t> cublasHandles;
vector<timeMS> slack;
vector<timeMS> deadline;
vector<Operation *> HEAD;

cudaEvent_t now, global_start;

static void dispatch(Operation *tp, int index) {
    assert(index < no_of_pipelines);
    cudaStream_t &compute_stream = stream[index];
    cudnnHandle_t &cudnn_handle = cudnnHandles[index];
    cublasHandle_t &cublas_handle = cublasHandles[index];
    int i = tp->op_layer - 1;

    NeuralNet *nm = tp->model;
    CnmemSpace space_tracker(nm->free_bytes); // need updates here
    //--	std::cout << "here\n";
    //--	std::cout << "Free bytes: " << nm->free_bytes << std::endl;

    if (i == 0) { // this is the first layer, load and resize image as per
                  // current inference pipeline
        /*	image im = load_image_color(nm->imgfname, 0, 0);
        //size? net->w in yolo
        image r = letterbox_image(im,nm->input_w, nm->input_h );
        //resize_network(net, resized.w, resized.h);
        show_image(im,"orig",5);
        show_image(r,"letterimg",5);
        //copy image data into layer_input[0]
        //memcpy(&(nm->layer_input[i]),r.data,nm->layer_input_size[i]*nm->data_type_size);
        nm->lockedcnmemMalloc(&(nm->layer_input[0]), nm->layer_input_size[0] *
        nm->data_type_size, NULL);*/
        space_tracker.updateSpace(CnmemSpace::SUB,
                                  nm->layer_input_size[0] * nm->data_type_size);
        // checkCudaErrors(cudaMemcpy(nm->layer_input[0], r.data, nm->batch_size
        // * nm->input_channels * nm->input_h * nm->input_w * nm->data_type_size,
        // cudaMemcpyHostToDevice));
    }
    float alpha = 1.0, beta = 0.0;
    float Salpha = 1.0, Sbeta = 0.0;
    double Dalpha = 1.0, Dbeta = 0.0;
    size_t cur_workspace_size;
    void *cur_workspace;

    // testingg
    /*	cudaEvent_t s,e;
            float mss;
            cudaEventCreate(&s);
            cudaEventCreate(&e);
            cudaEventRecord(s, nm->stream_compute);
     */
    nm->lockedcnmemMalloc(&(nm->layer_input[i + 1]),
                          nm->layer_input_size[i + 1] * nm->data_type_size,
                          NULL);
    /*	cudaEventRecord(e, nm->stream_compute);
            cudaEventSynchronize(e);
            cudaEventElapsedTime(&mss,s,e);
            printf("%f :",mss);
     */
    space_tracker.updateSpace(CnmemSpace::SUB,
                              nm->layer_input_size[i + 1] * nm->data_type_size);

    if (nm->layer_type[i] == CONV) {
        tp->type = 'C';
        // std::cout << "conv\n";
        ConvLayerParams *cur_params = (ConvLayerParams *)nm->params[i];

        cur_workspace_size = cur_params->fwd_workspace_size;
        nm->lockedcnmemMalloc(&cur_workspace, cur_workspace_size, NULL);
        // computation
        checkCUDNN(cudnnConvolutionForward(
            cudnn_handle, &alpha, cur_params->input_tensor, nm->layer_input[i],
            cur_params->filter_desc, cur_params->W, cur_params->conv_desc,
            cur_params->fwd_algo, cur_workspace, cur_workspace_size, &beta,
            cur_params->output_tensor, nm->layer_input[i + 1]));

        // custom  coarsened cuda kernel
        // customCoarsenedConvolutionForward((float *)nm->layer_input[i], (float
        // *)nm->layer_input[i + 1], cur_params->conv_desc,
        // cur_params->filter_desc, cur_params->input_tensor, (float
        // *)cur_params->W, compute_stream);

        // Batch Normalization
        /* if (cur_params->bn == 1)
        {
                normalize_gpu((float *)nm->layer_input[i + 1], (float
        *)cur_params->rolling_mean_gpu, (float
        *)cur_params->rolling_variance_gpu, 1, cur_params->C_out,
        cur_params->output_h * cur_params->output_w, compute_stream);
                scale_bias_gpu((float *)nm->layer_input[i + 1], (float
        *)cur_params->scales_gpu, 1, cur_params->C_out, cur_params->output_h *
        cur_params->output_w, compute_stream); add_bias_gpu((float
        *)nm->layer_input[i + 1], (float *)cur_params->b, 1, cur_params->C_out,
        cur_params->output_h * cur_params->output_w, compute_stream);
        }
        else
        {
                add_bias_gpu((float *)nm->layer_input[i + 1], (float
        *)cur_params->b, 1, cur_params->C_out, cur_params->output_h *
        cur_params->output_w, compute_stream);
        } */
        checkCUDNN(cudnnAddTensor(
            cudnn_handle, &alpha, cur_params->bias_desc, cur_params->b, &alpha,
            cur_params->output_tensor, nm->layer_input[i + 1]));

        // if activation required
        if (cur_params->activation_mode != ACTIVATION_NONE) {
            // Replacing cuDNN call for relu to custom leaky relu call
            // float *addr = (float *)(nm->layer_input[i + 1]);
            // activate_array_gpu(addr, nm->layer_input_size[i + 1],
            // compute_stream);

            checkCUDNN(cudnnActivationForward(
                cudnn_handle, cur_params->actv_desc, &alpha,
                cur_params->output_tensor, nm->layer_input[i + 1], &beta,
                cur_params->output_tensor, nm->layer_input[i + 1]));
        }

        space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
        // std::cout << "Free bytes: " << free_bytes << std::endl;
    } else if (nm->layer_type[i] == FULLY_CONNECTED) {
        tp->type = 'F';
        // std::cout << "FC\n";
        FCLayerParams *cur_params = (FCLayerParams *)nm->params[i];
        // std::cout << "FChere" << i << std::endl;

        if (nm->data_type == CUDNN_DATA_FLOAT) {
            checkCUBLAS(cublasSgemm(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out,
                nm->batch_size, cur_params->C_in, &Salpha,
                (float *)cur_params->W, cur_params->C_out,
                (float *)nm->layer_input[i], cur_params->C_in, &Sbeta,
                (float *)nm->layer_input[i + 1], cur_params->C_out));
            checkCUBLAS(cublasSgemm(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out,
                nm->batch_size, 1, &Salpha, (float *)cur_params->b,
                cur_params->C_out, (float *)nm->one_vec, 1, &Salpha,
                (float *)nm->layer_input[i + 1], cur_params->C_out));
        } else if (nm->data_type == CUDNN_DATA_DOUBLE) {
            checkCUBLAS(cublasDgemm(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out,
                nm->batch_size, cur_params->C_in, &Dalpha,
                (double *)cur_params->W, cur_params->C_out,
                (double *)nm->layer_input[i], cur_params->C_in, &Dbeta,
                (double *)nm->layer_input[i + 1], cur_params->C_out));
            checkCUBLAS(cublasDgemm(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cur_params->C_out,
                nm->batch_size, 1, &Dalpha, (double *)cur_params->b,
                cur_params->C_out, (double *)nm->one_vec, 1, &Dalpha,
                (double *)nm->layer_input[i + 1], cur_params->C_out));
        }
        if (cur_params->activation_mode != ACTIVATION_NONE) {
            // Replacing cuDNN call for Relu activation to custom Leaky Relu
            // call
            checkCUDNN(cudnnActivationForward(
                cudnn_handle, cur_params->actv_desc, &alpha,
                cur_params->output_tensor, nm->layer_input[i + 1], &beta,
                cur_params->output_tensor, nm->layer_input[i + 1]));
            // activate_array_gpu((float *)nm->layer_input[i + 1],
            // nm->layer_input_size[i + 1], compute_stream);
        }
        // std::cout << "FChere" << i << std::endl;
    }

    else if (nm->layer_type[i] == DROPOUT) {
        tp->type = 'D';
        // std::cout << "Dropout\n";
        DropoutLayerParams *cur_params = (DropoutLayerParams *)nm->params[i];
        checkCUDNN(cudnnDropoutForward(
            cudnn_handle, cur_params->dropout_desc, cur_params->input_tensor,
            nm->layer_input[i], cur_params->input_tensor,
            nm->layer_input[i + 1], cur_params->reserved_space,
            cur_params->reserved_space_size));
    } else if (nm->layer_type[i] == BATCHNORM) {
        tp->type = 'B';
        // std::cout << "Batchnorm\n";
        BatchNormLayerParams *cur_params =
            (BatchNormLayerParams *)nm->params[i];

        checkCUDNN(cudnnBatchNormalizationForwardInference(
            cudnn_handle, cur_params->mode, &alpha, &beta,
            cur_params->input_tensor, nm->layer_input[i],
            cur_params->input_tensor, nm->layer_input[i + 1],
            cur_params->sbmv_desc, cur_params->scale, cur_params->bias,
            cur_params->running_mean, cur_params->running_variance,
            cur_params->epsilon));
    } else if (nm->layer_type[i] == POOLING) {
        tp->type = 'P';
        // std::cout << "Pooling\n";
        PoolingLayerParams *cur_params = (PoolingLayerParams *)nm->params[i];
        checkCUDNN(cudnnPoolingForward(
            cudnn_handle, cur_params->pool_desc, &alpha,
            cur_params->input_tensor, nm->layer_input[i], &beta,
            cur_params->output_tensor, nm->layer_input[i + 1]));
    } else if (nm->layer_type[i] == ACTV) {
        tp->type = 'A';
        ActivationLayerParams *cur_params =
            (ActivationLayerParams *)nm->params[i];
        checkCUDNN(cudnnActivationForward(
            cudnn_handle, cur_params->actv_desc, &alpha,
            cur_params->input_tensor, nm->layer_input[i], &beta,
            cur_params->input_tensor, nm->layer_input[i + 1]));
    } else if (nm->layer_type[i] == REGION) { // Processing of region layer
        tp->type = 'R';
        // printf("Processing region layer %d",i);
        // printf("Input layer size is %d output layer size is %d\n",
        // nm->layer_input_size[i], nm->layer_input_size[i+1]);
        RegionLayerParams *cur_params = (RegionLayerParams *)nm->params[i];
        // printf("Batch size is %d\n", cur_params->batch_size);
        forward_region_layer_gpu(
            nm->layer_input_size[i], nm->layer_input_size[i + 1],
            (float *)nm->layer_input[i], cur_params->batch_size,
            cur_params->height, cur_params->width, cur_params->num,
            cur_params->classes, cur_params->coords,
            (float *)nm->layer_input[i + 1], compute_stream);

        float *result =
            (float *)malloc(nm->layer_input_size[i + 1] * sizeof(float));
        checkCudaErrors(cudaMemcpy(result, nm->layer_input[i + 1],
                                   nm->layer_input_size[i + 1] * sizeof(float),
                                   cudaMemcpyDeviceToHost));

        // int nbox=0;
        // newly added block
        //--detection *dets = make_network_boxes(cur_params,0.5, &nbox);
        //--fill_network_boxes(cur_params,nm->img_w,nm->img_h, 0.5,0, dets,
        //result, nm->layer_input_size[i+1], nm->input_w, nm->input_h);
        // print_detector_detections(fps, id, dets, num, classes, w, h);
        //----list *options = read_data_cfg("cfg/coco.data");
        // char *name_list = option_find_str(options, "names",
        // "data/names.list");
        //----char *name_list = option_find_str(options, "names",
        //"data/coco.names");
        //--char **names = get_labels("data/coco.names");
        //--image **alphabet = load_alphabet();
        //--draw_detections(nm->im, dets, nbox, 0.5, names, alphabet,
        //cur_params->classes);
        //--save_image(nm->im, "predictions");
        //--free_detections(dets, nbox);
    } else if (nm->layer_type[i] == SOFTMAX) {
        tp->type = 'S';
        SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)nm->params[i];
        // custom kernel call for softmax
        softmax_gpu((float *)nm->layer_input[i], cur_params->channels,
                    cur_params->channels,
                    (nm->layer_input_size[i]) / cur_params->channels,
                    (cur_params->w) * (cur_params->h), 1,
                    (cur_params->w) * (cur_params->h), 1,
                    (float *)nm->layer_input[i + 1], compute_stream);
        // cuDNN kernel call for Softmax
        /*checkCUDNN(cudnnSoftmaxForward(nm->cudnn_handle, cur_params->algo,
           cur_params->mode, &alpha, cur_params->input_tensor,
           nm->layer_input[i], &beta, cur_params->input_tensor,
           nm->layer_input[i + 1]));*/
        //-Copy the result produced by softmax layer from GPU to CPU

        // checkCudaErrors(cudaStreamSynchronize(nm->stream_compute));
        // /////-----check....
        float *result =
            (float *)malloc(nm->layer_input_size[i + 1] * sizeof(float));
        checkCudaErrors(cudaMemcpy(result, nm->layer_input[i + 1],
                                   nm->layer_input_size[i + 1] * sizeof(float),
                                   cudaMemcpyDeviceToHost));

        // Infer the output class
        //	int *correct_count=0;
        //	nm->compareOutputCorrect(correct_count,nm->y);
        //	checkCNMEM(cnmemFree(nm->layer_input[nm->num_layers - 1],
        //NULL)); 	space_tracker.updateSpace(CnmemSpace::ADD,
        //nm->layer_input_size[nm->num_layers - 1] * nm->data_type_size);
        //--
        int top = 5;
        list *options =
            read_data_cfg("data/imagenet1k.data"); // specify name  of the file
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
        top_k(result, nm->layer_input_size[i + 1], top,
              indexes); // check parameters of this function
        // fprintf(stderr, "%s: Predicted in %f seconds.\n", input,
        // sec(clock()-time));
        for (ii = 0; ii < top; ++ii) {
            // int index = indexes[ii];
            // if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index,
            // names[index], predictions[index], (net->hierarchy->parent[index]
            // >= 0) ? names[net->hierarchy->parent[index]] : "Root"); else
            // printf("%s: %f\n",names[index], predictions[index]); printf("index
            // is %d: %5.2f%%: %s\n",index, result[index]*100, names[index]);
            // printf("index is %d: %s\n",index, names[index]);
        }
    }
    if (nm->layer_type[i] == CONV) {
        nm->lockedcnmemFree(cur_workspace, NULL);
        space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
    }

    // kCudaErrors(cudaStreamSynchronize(nm->stream_compute));
    // free the memory allocated to layer_input[i]
    // nm->lockedcnmemFree(nm->layer_input[i], NULL);
    // space_tracker.updateSpace(CnmemSpace::ADD, nm->layer_input_size[i] *
    // nm->data_type_size);
}

static void loadDeadlines() {
    for (int i = 0; i < no_of_pipelines; i++) {
        deadline[i] = 80.0;
    }
}

static timeMS sumOfExecutionTimes(Operation *op) {
    timeMS sum = 0.0;
    for (; op != nullptr; op = op->children.back()) {
        sum += op->time_to_execute;
    }
    return sum;
}
static void execute(Operation *tp, int index) {
    if (tp->op_type == 'c') {
        assert(tp->parents.back()->op_type == 'm');
        dispatch(tp, index);
    } else if (tp->op_type == 'm') {
        if (tp->op_layer == 0) {
            InputOperation *zerothLayer = static_cast<InputOperation *>(tp);
            zerothLayer->model->loadFile(
                const_cast<char *>((zerothLayer->filename).c_str()));
        } else {
            tp->model->prefetchWeights(tp->op_layer, stream[index]);
        }
    }
    checkcudaStreamAddCallback(stream[index], );
}
void start(vector<InputOperation *> &dags) {
    setHEAD(dags);
    assert(!HEAD.empty());

    for (int i = 0; i < no_of_pipelines; i++) {
        cudaStreamCreate(&stream[i]);
        cudnnCreate(&cudnnHandles[i]);
        cudnnSetStream(cudnnHandles[i], stream[i]);
        cublasCreate(&cublasHandles[i]);
        cublasSetStream(cublasHandles[i], stream[i]);
    }

    loadDeadlines();

    checkCudaErrors(cudaEventCreate(&now));

    checkCudaErrors(cudaEventCreate(&global_start));

    checkCudaErrors(cudaEventRecord(global_start));
    perform_lsf(nullptr, cudaError_t::cudaSuccess, nullptr);
}

static void CUDART_CB perform_lsf(cudaStream_t stream, cudaError_t status,
                                  void *data) {
    checkCudaErrors(cudaEventRecord(now));
    timeMS currentTime;
    checkCudaErrors(cudaEventElapsedTime(&currentTime, global_start, now));
    for (int i = 0; i < no_of_pipelines; i++) {
        if (HEAD[i] != nullptr)
            slack[i] = deadline[i] - currentTime -
                       sumOfExecutionTimes(
                           HEAD[i]->children.back()); // - something else
    }
    auto index = std::min_element(slack.begin(), slack.end()) - slack.begin();
    execute(HEAD[index],
            index); // asynchronous placement of the operation on the GPU
    HEAD[index] = HEAD[index]->children.back();
    if (HEAD[index] == nullptr) {
        slack[index] =
            1.0 / 0.0; // setting the slack to infinity so that we never pick it
    }
}

static void setHEAD(vector<InputOperation *> &dags) {
    for (int i = 0; i < no_of_pipelines; i++) {
        HEAD[i] = dags[i];
    }
}