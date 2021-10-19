#include "solver.h"
#include <ScheduleEngine.h>
#include <cmath>
#include <cstdlib>
#include <dag.h>
#include <dirent.h>
#include <iostream>
#include <lsf_pipelines.h>
#include <lsf_scheduler.h>
#include <thread>
#include <utilities.h>

enum PROGRAM_TYPE {
    SEQUENTIAL = 1,
    COSCHEDULING = 2,
    LSF = 3,
    SMT = 4,
    LSF_PYTHON = 5
};

int str_ends_with(const char *s, const char *suffix) {
    size_t slen = strlen(s);
    size_t suffix_len = strlen(suffix);

    return suffix_len <= slen && !strcmp(s + slen - suffix_len, suffix);
}

// Task setup
int nos_of_tasks = 2;
int jobs_per_task[] = {2, 1};

typedef unsigned char uchar;

int pipelines = 0;
int numOfImages = 10;

int main(int argc, char *argv[]) {
    vector<NeuralNet> networks;
    if (argc != 6) {
        printf("Usage: build/pipeline.out data/weightfile1 data/weightfile2 "
               "images01/ images02 Program_type\n");
        printf("Program_Type 1:Sequential scheduling 2:Coscheduling lookup "
               "table 3:LSF scheduling 4:SMT schedule execution\n");
        exit(1);
    }
    vDNNConvAlgo vdnn_conv_algo = vDNN_MEMORY_OPTIMAL;
    vDNNType vdnn_type = vDNN_ALL;
    int batch_size = 1;
    long long dropout_seed = 1;
    float softmax_eps = 1e-8;
    float init_std_dev = 0.1;
    /*	NeuralNet vggnet(layer_specifier, DATA_FLOAT, batch_size, TENSOR_NCHW,
       dropout_seed, softmax_eps, init_std_dev, vdnn_type, vdnn_conv_algo, SGD);


            NeuralNet alexnet(layer_specifier1, DATA_FLOAT, batch_size,
       TENSOR_NCHW, dropout_seed, softmax_eps, init_std_dev, vdnn_type,
       vdnn_conv_algo, SGD);

     */

    pipelines++;
    vector<LayerSpecifier> layer_specifier3;
    // conv1
    {
        ConvDescriptor conv1;
        conv1.initializeValues(3, 16, 3, 3, 16, 16, 1, 1, 1, 1, 1, RELU);
        // conv1.initializeValues(3, 16, 3, 3, 224,224, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv1;
        layer_specifier3.push_back(temp);
    }
    // max pool 1
    {
        PoolingDescriptor poo01;
        poo01.initializeValues(16, 2, 2, 16, 16, 0, 0, 2, 2, POOLING_MAX);
        // poo01.initializeValues(16, 2, 2, 224,224 , 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo01;
        layer_specifier3.push_back(temp);
    }
    // conv2
    {
        ConvDescriptor conv2;
        conv2.initializeValues(16, 32, 3, 3, 8, 8, 1, 1, 1, 1, 1, RELU);
        // conv2.initializeValues(16, 32, 3, 3, 112, 112, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv2;
        layer_specifier3.push_back(temp);
    }
    // max pool 2
    {
        PoolingDescriptor poo02;
        poo02.initializeValues(32, 2, 2, 8, 8, 0, 0, 2, 2, POOLING_MAX);
        // poo02.initializeValues(32, 2, 2, 112,112 , 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo02;
        layer_specifier3.push_back(temp);
    }
    // conv3
    {
        ConvDescriptor conv3;
        conv3.initializeValues(32, 64, 3, 3, 4, 4, 1, 1, 1, 1, 1, RELU);
        // conv3.initializeValues(32, 64 , 3, 3, 56,56, 1, 1 , 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv3;
        layer_specifier3.push_back(temp);
    }
    // max pool 3
    {
        PoolingDescriptor poo03;
        poo03.initializeValues(64, 2, 2, 4, 4, 0, 0, 2, 2, POOLING_MAX);
        // poo03.initializeValues(64, 2, 2, 56,56 , 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo03;
        layer_specifier3.push_back(temp);
    }
    // conv4
    {
        ConvDescriptor conv4;
        conv4.initializeValues(16, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, RELU);
        // conv4.initializeValues(16,128, 3, 3, 28,28, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv4;
        layer_specifier3.push_back(temp);
    }
    // max pool 4
    {
        PoolingDescriptor poo04;
        poo04.initializeValues(128, 2, 2, 2, 2, 0, 0, 1, 1, POOLING_MAX);
        // poo04.initializeValues(128 , 2, 2, 28, 28 , 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo04;
        layer_specifier3.push_back(temp);
    }
    // conv5
    {
        ConvDescriptor conv5;
        conv5.initializeValues(128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, RELU);
        // conv5.initializeValues(128,128 ,3,3, 14, 14 , 1,1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv5;
        layer_specifier3.push_back(temp);
    }
    // max pool 5
    {
        PoolingDescriptor poo05;
        poo05.initializeValues(128, 2, 2, 2, 2, 0, 0, 1, 1, POOLING_MAX);
        // poo05.initializeValues(128 , 2, 2, 14,14 , 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo05;
        layer_specifier3.push_back(temp);
    }
    // conv6
    {
        ConvDescriptor conv6;
        conv6.initializeValues(128, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, RELU);
        // conv6.initializeValues( 128,256,3,3, 7,7 , 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv6;
        layer_specifier3.push_back(temp);
    }
    // conv7
    {
        ConvDescriptor conv7;
        conv7.initializeValues(256, 125, 1, 1, 2, 2, 1, 1, 1, 1, 1, SIGMOID);
        // conv7.initializeValues( 256 , 125, 1, 1, 7,7 , 0, 0, 1, 1, 1,RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv7;
        layer_specifier3.push_back(temp);
    }
    // Region layer
    {
        RegionDescriptor region1;
        // region1.initializeValues(channels,w,h,nuw,classes,coords);
        region1.initializeValues(125, 2, 2, 5, 20, 4);
        // region1.initializeValues(125,7,7,5,20,4);
        LayerSpecifier temp;
        temp.initPointer(REGION);
        *((RegionDescriptor *)temp.params) = region1;
        layer_specifier3.push_back(temp);
    }

    networks.push_back(NeuralNet(layer_specifier3, DATA_FLOAT, batch_size,
                                 TENSOR_NCHW, dropout_seed, softmax_eps,
                                 init_std_dev, vdnn_type, vdnn_conv_algo, SGD,
                                 argv[1], argv[3]));

    pipelines++;
    vector<LayerSpecifier> layer_specifier2;
    // conv1
    {
        ConvDescriptor conv1;
        conv1.initializeValues(3, 16, 3, 3, 16, 16, 1, 1, 1, 1, 1, RELU);
        // conv1.initializeValues(3, 16, 3, 3, 416, 416, 1, 1, 1, 1, 1, RELU);
        // conv1.initializeValues(3, 16, 3, 3, 224, 224, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv1;
        layer_specifier2.push_back(temp);
    }
    // max pool 1
    {
        PoolingDescriptor poo01;
        poo01.initializeValues(16, 2, 2, 16, 16, 0, 0, 2, 2, POOLING_MAX);
        // poo01.initializeValues(16, 2, 2, 416, 416 , 0, 0, 2, 2, POOLING_MAX);
        // poo01.initializeValues(16, 2, 2, 224,224 , 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo01;
        layer_specifier2.push_back(temp);
    }
    // conv2
    {
        ConvDescriptor conv2;
        conv2.initializeValues(16, 32, 3, 3, 8, 8, 1, 1, 1, 1, 1, RELU);
        // conv2.initializeValues(16, 32, 3, 3, 208, 208, 1, 1, 1, 1, 1, RELU);
        // conv2.initializeValues(16, 32, 3, 3, 112, 112, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv2;
        layer_specifier2.push_back(temp);
    }
    // max pool 2
    {
        PoolingDescriptor poo02;
        poo02.initializeValues(32, 2, 2, 8, 8, 0, 0, 2, 2, POOLING_MAX);
        // poo02.initializeValues(32, 2, 2, 208, 208 , 0, 0, 2, 2, POOLING_MAX);
        // poo02.initializeValues(32, 2, 2, 112, 112, 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo02;
        layer_specifier2.push_back(temp);
    }
    // conv3
    {
        ConvDescriptor conv3;
        conv3.initializeValues(32, 64, 3, 3, 4, 4, 1, 1, 1, 1, 1, RELU);
        // conv3.initializeValues(32, 64 , 3, 3, 104, 104, 1, 1 , 1, 1, 1,
        // RELU); conv3.initializeValues(32, 64 , 3, 3, 56, 56, 1, 1 , 1, 1, 1,
        // RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv3;
        layer_specifier2.push_back(temp);
    }
    // max pool 3
    {
        PoolingDescriptor poo03;
        poo03.initializeValues(64, 2, 2, 4, 4, 0, 0, 2, 2, POOLING_MAX);
        // poo03.initializeValues(64, 2, 2, 104, 104 , 0, 0, 2, 2, POOLING_MAX);
        // poo03.initializeValues(64, 2, 2, 56, 56 , 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo03;
        layer_specifier2.push_back(temp);
    }

    // conv4
    {
        ConvDescriptor conv4;
        // conv4.initializeValues(64,128, 3, 3, 2,2, 1, 1, 1, 1, 1, RELU);
        conv4.initializeValues(64, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1, RELU);
        // conv4.initializeValues(16,128, 3, 3, 52, 52, 1, 1, 1, 1, 1, RELU);
        // conv4.initializeValues(16,128, 3, 3, 28,28, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv4;
        layer_specifier2.push_back(temp);
    }
    // max pool 4
    {
        PoolingDescriptor poo04;
        // poo04.initializeValues(128 , 2, 2, 2, 2 , 0, 0, 1, 1, POOLING_MAX);
        poo04.initializeValues(64, 2, 2, 2, 2, 0, 0, 1, 1, POOLING_MAX);
        // poo04.initializeValues(128 , 2, 2, 52, 52 , 0, 0, 2, 2, POOLING_MAX);
        // poo04.initializeValues(128 , 2, 2, 28, 28, 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo04;
        layer_specifier2.push_back(temp);
    }
    // conv5
    {
        ConvDescriptor conv5;
        //--conv5.initializeValues(128,128 , 3, 3, 4,4, 1,1,1, 1, 1, RELU);
        // conv5.initializeValues(128,256 , 3, 3, 2,2, 1,1,1, 1, 1, RELU);
        conv5.initializeValues(64, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1, RELU);
        // conv5.initializeValues(128,128 , 3, 3, 26, 26, 1,1,1, 1, 1, RELU);
        // conv5.initializeValues(128,128 , 3, 3, 14, 14, 1,1,1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv5;
        layer_specifier2.push_back(temp);
    }
    // max pool 5
    {
        PoolingDescriptor poo05;
        // poo05.initializeValues(256 , 2, 2, 2,2 , 0, 0, 1, 1, POOLING_MAX);
        poo05.initializeValues(128, 2, 2, 2, 2, 0, 0, 1, 1, POOLING_MAX);
        // poo05.initializeValues(128 , 2, 2, 26, 26 , 0, 0, 2, 2, POOLING_MAX);
        // poo05.initializeValues(128 , 2, 2, 14,14 , 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo05;
        layer_specifier2.push_back(temp);
    }
    // conv6
    {
        ConvDescriptor conv6;
        //--conv6.initializeValues( 128,256,3,3, 2,2, 1, 1, 1, 1, 1, RELU);
        // conv6.initializeValues( 256,512,3,3, 2,2, 1, 1, 1, 1, 1, RELU);
        conv6.initializeValues(128, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, RELU);
        // conv6.initializeValues( 128,256,3,3, 13,13, 1, 1, 1, 1, 1, RELU);
        // conv6.initializeValues( 128,256,3,3, 7,7, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv6;
        layer_specifier2.push_back(temp);
    }
    // max pool 6
    {
        PoolingDescriptor poo06;
        //--poo06.initializeValues(256 , 2, 2, 2,2 ,0, 0, 2, 2, POOLING_MAX);
        // poo06.initializeValues(512 , 2, 2, 2,2 ,0, 0, 2, 2, POOLING_MAX);
        poo06.initializeValues(256, 2, 2, 2, 2, 0, 0, 2, 2, POOLING_MAX);
        // poo06.initializeValues(256 , 2, 2, 13, 13 ,0, 0, 1, 1, POOLING_MAX);
        // poo06.initializeValues(256 , 2, 2, 7, 7 ,0, 0, 1, 1, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo06;
        layer_specifier2.push_back(temp);
    }

    // conv7
    {
        ConvDescriptor conv7;
        // conv7.initializeValues( 512 , 1024, 3, 3, 1,1, 1, 1, 1, 1, 1,RELU);
        conv7.initializeValues(256, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, RELU);
        // conv7.initializeValues( 256 , 1024, 3, 3, 12, 12, 1, 1, 1, 1,
        // 1,RELU); conv7.initializeValues( 256 , 1024, 3, 3, 6, 6, 1, 1, 1, 1,
        // 1,RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv7;
        layer_specifier2.push_back(temp);
    }
    // conv8
    {
        ConvDescriptor conv8;
        //--conv8.initializeValues( 256 , 1024, 3, 3, 1,1, 1, 1, 1, 1, 1,RELU);
        // conv8.initializeValues( 1024 , 1024, 3, 3, 1,1, 1, 1, 1, 1, 1,RELU);
        conv8.initializeValues(512, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, RELU);
        // conv8.initializeValues( 1024 , 1024, 3, 3, 12, 12, 1, 1, 1, 1,
        // 1,RELU); conv8.initializeValues( 1024 , 1024, 3, 3, 6,6, 1, 1, 1, 1,
        // 1,RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv8;
        layer_specifier2.push_back(temp);
    }
    // conv9
    {
        ConvDescriptor conv9;
        //-conv9.initializeValues( 256 , 1024, 3, 3, 1,1, 1, 1, 1, 1, 1,RELU);
        // conv9.initializeValues( 1024 , 125, 3, 3, 1,1, 1, 1, 1, 1, 1,RELU);
        conv9.initializeValues(512, 125, 3, 3, 1, 1, 1, 1, 1, 1, 1, RELU);
        // conv9.initializeValues( 1024 , 125, 1, 1, 12, 12, 1, 1, 1, 1,
        // 1,RELU); conv9.initializeValues( 1024 , 125, 1, 1, 6, 6, 1, 1, 1, 1,
        // 1,RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv9;
        layer_specifier2.push_back(temp);
    }
    // Region layer
    {
        RegionDescriptor region1;
        // region1.initializeValues(channels,w,h,nuw,classes,coords);
        region1.initializeValues(125, 1, 1, 5, 20, 4);
        // region1.initializeValues(125,6,6,5,20,4);
        LayerSpecifier temp;
        temp.initPointer(REGION);
        *((RegionDescriptor *)temp.params) = region1;
        layer_specifier2.push_back(temp);
    }

    // cost layer as per specification

    networks.push_back(NeuralNet(layer_specifier2, DATA_FLOAT, batch_size,
                                 TENSOR_NCHW, dropout_seed, softmax_eps,
                                 init_std_dev, vdnn_type, vdnn_conv_algo, SGD,
                                 argv[2], argv[4]));

    // Now inference will start: Operations of each layer will be scheduled by
    // ScheduleEngine

    // Create an object of ScheduleEngine
    ScheduleEngine se;
    /* se.model1=&networks[0];
    se.model2=&networks[1]; */

    // warmup code starts here
    //--------------------
    for (auto &network : networks) {
        auto zerothLayer = new InputOperation("kite.jpg", &network, 0, 'm', 0);
        createLinearDAG(zerothLayer);
        printf("Starting Warm up code\n");
        Operation *currentOperation = zerothLayer;
        while (currentOperation != nullptr) {
            se.enqueue(currentOperation);
            currentOperation = currentOperation->children.back();
        }
        se.warmup_schedule(zerothLayer);
        printf("Warming up code execution completed\n");
        network.cur_prefetch_layer = 0;
        fseek(network.wfp, 0, SEEK_SET);
        destroyLinearDAG(&zerothLayer);
    }
    // warmup code ends here

    DIR *d1, *d2;
    struct dirent *dir;
    char **list1, **list2;
    int i = 0;
    float ms = 0;
    float total_time = 0;
    list1 = (char **)malloc(numOfImages * sizeof(char *));
    list2 = (char **)malloc(numOfImages * sizeof(char *));
    printf("P1: %s", networks[0].imgpath);
    printf("P2: %s", networks[1].imgpath);
    d1 = opendir(networks[0].imgpath); // arg[3] for pipeline01  image path
    d2 = opendir(networks[1].imgpath); // arg[4] for pipeline01  image path
    if (d1) {
        while ((dir = readdir(d1)) != NULL && i < numOfImages) {

            if (!strcmp(dir->d_name, "."))
                continue;
            if (!strcmp(dir->d_name, ".."))
                continue;
            if (str_ends_with(dir->d_name, ".JPEG")) {
                list1[i] =
                    (char *)malloc((strlen(dir->d_name) + 1) * sizeof(char));
                strcpy(list1[i], dir->d_name);
                printf("%s\n", list1[i]);
                i++;
            }
        }
        closedir(d1);
    } else {
        printf("Unable to open directory %s\n", networks[0].imgpath);
        exit(1);
    }

    i = 0;

    if (d2) {
        while ((dir = readdir(d2)) != NULL && i < numOfImages) {

            if (!strcmp(dir->d_name, "."))
                continue;
            if (!strcmp(dir->d_name, ".."))
                continue;
            if (str_ends_with(dir->d_name, ".JPEG")) {
                list2[i] =
                    (char *)malloc((strlen(dir->d_name) + 1) * sizeof(char));
                strcpy(list2[i], dir->d_name);
                printf("%s\n", list2[i]);
                i++;
            }
        }

        closedir(d2);
    } else {
        printf("Unable to open directory %s\n", networks[1].imgpath);
        exit(1);
    }
    assert(strlen(argv[argc - 1]) == 1);

    switch (atoi(argv[argc - 1])) {
    case PROGRAM_TYPE::SEQUENTIAL: {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        //--------------
        // Create global event
        se.createGlobalEvent();
        FILE *fpcf = fopen("output/stats_mem_seq.txt", "w");
        char filename[100];
        for (int no = 0; no < numOfImages; no++) {
            strcpy(filename, networks[0].imgpath);
            strcat(filename, list1[no]);
            auto zerothLayer1 =
                new InputOperation(filename, &networks[0], 0, 'm', 1);
            createLinearDAG(zerothLayer1);
            cudaEventRecord(start);
            se.schedule_sequential(zerothLayer1, fpcf);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            total_time += ms;

            // execution of pipeline 01 is over, so deallocate its space
            // networks[0].deallocateSpace();
            strcpy(filename, networks[1].imgpath);
            strcat(filename, list2[no]);
            auto zerothLayer2 =
                new InputOperation(filename, &networks[1], 0, 'm', 2);
            createLinearDAG(zerothLayer2);
            cudaEventRecord(start);
            se.schedule_sequential(zerothLayer2, fpcf);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            total_time += ms;
        }
        fclose(fpcf);
        printf("Total time to process %d images is %f\n", numOfImages,
               total_time);
        // networks[1].deallocateSpace();
        break;
    }
    case PROGRAM_TYPE::COSCHEDULING: {
        vector<Operation *> pipe1, pipe2;
        // Parallel execution of two pipelines
        auto zerothLayer1 =
            new InputOperation("data/dog.jpg", &networks[0], 0, 'm', 1);
        createLinearDAG(zerothLayer1);
        Operation *currentOperation = zerothLayer1;
        while (currentOperation != nullptr) {
            printf("%c  ", currentOperation->op_type);
            if (currentOperation->op_type == 'c') {
                pipe1.push_back(currentOperation);
            }
            currentOperation = currentOperation->children.back();
        }
        printf("\n");
        auto zerothLayer2 =
            new InputOperation("data/eagle.jpg", &networks[1], 0, 'm', 2);
        createLinearDAG(zerothLayer2);
        currentOperation = zerothLayer2;
        while (currentOperation != nullptr) {
            printf("%c  ", currentOperation->op_type);
            if (currentOperation->op_type == 'c') {
                pipe2.push_back(currentOperation);
            }
            currentOperation = currentOperation->children.back();
        }
        printf("\nNumber of elements in pipe1 is %ld\n", pipe1.size());
        printf("Number of elements in pipe1 is %ld\n", pipe2.size());
        // call schedule profile function
        printf("starting execution of schedule_profile for generating "
               "co-scheduling table\n");
        se.schedule_profile(zerothLayer1, zerothLayer2, pipe1, pipe2);
        printf("Finished preparing co-scheduling table");
        break;
    }
    case PROGRAM_TYPE::LSF: {
        ifstream timingFile;
        timingFile.open("output/arrival-execution.txt");
        string filename = networks[0].imgpath;
        filename += list1[0];
        auto zerothLayer1 =
            new InputOperation(filename, &networks[0], 0, 'm', 0);
        createLinearDAG(zerothLayer1);
        filename = networks[1].imgpath;
        filename += list2[0];
        auto zerothLayer2 =
            new InputOperation(filename, &networks[1], 0, 'm', 1);
        createLinearDAG(zerothLayer2);
        // loadTimings(timingFile1, zerothLayer1);
        fillExecutionTime(timingFile, {zerothLayer1, zerothLayer2});
        // Start the execution of LSF
        vector<InputOperation *> v;
        v.push_back(zerothLayer1);
        v.push_back(zerothLayer2);
        start(v);
        break;
    }
    case PROGRAM_TYPE::SMT: {
        ScheduleEngine se;
        vector<InputOperation *> zerothLayer;
        ifstream timingFile;
        timingFile.open("output/smt-arrival-stream.txt");
        for (int i = 0; i < jobs_per_task[0]; i++) {
            string filename = networks[0].imgpath;
            filename += list1[i];
            zerothLayer.push_back(
                new InputOperation(filename, &networks[0], 0, 'm', 0));
            createLinearDAG(zerothLayer.back());
        }
        for (int i = 0; i < jobs_per_task[1]; i++) {
            string filename = networks[1].imgpath;
            filename += list2[i];
            zerothLayer.push_back(
                new InputOperation(filename, &networks[1], 0, 'm', 1));
            createLinearDAG(zerothLayer.back());
        }
        fillSMTDetails(timingFile, zerothLayer);
        std::priority_queue<Operation *, std::vector<Operation *>,
                            compareStartTimings>
            operationQueue;
        for (auto &zeroLayer : zerothLayer) {
            dagToPriorityQueue(operationQueue, (Operation *)zeroLayer);
        }
        chrono::time_point<chrono::steady_clock> timeGlobalStart =
            chrono::steady_clock::now(); // globalStart
        while (!operationQueue.empty()) {
            auto currentOperation = operationQueue.top();
            operationQueue.pop();
            {
                if (currentOperation->op_type == 'c') {
                    cudaEventSynchronize(
                        currentOperation->parents.back()->endop);
                    auto currentStream =
                        (currentOperation->chosenStream == 'H')
                            ? ScheduleEngine::HIGH_COMPUTE_STREAM
                            : ScheduleEngine::LOW_COMPUTE_STREAM;
                    checkCudaErrors(
                        cudaEventRecord(currentOperation->startop,
                                        se.compute_streams[currentStream]));
                    assert(currentOperation->parents.back()->op_type == 'm');
                    se.dispatch(currentOperation, currentStream);
                    checkCudaErrors(
                        cudaEventRecord(currentOperation->endop,
                                        se.compute_streams[currentStream]));
                } else if (currentOperation->op_type == 'm') {
                    checkCudaErrors(cudaEventRecord(currentOperation->endop,
                                                    se.memoryStream));
                    if (currentOperation->op_layer == 0) {
                        InputOperation *zerothLayer =
                            static_cast<InputOperation *>(currentOperation);
                        zerothLayer->model->loadFile(
                            const_cast<char *>((zerothLayer->filename).c_str()),
                            se.memoryStream);
                    } else {
                        currentOperation->model->prefetchWeights(
                            currentOperation->op_layer - 1,
                            se.memoryStream); //-1 missing here
                    }
                    checkCudaErrors(cudaEventRecord(currentOperation->endop,
                                                    se.memoryStream));
                }
            }
            // sleep for (duration = start time of next op - start time of
            // current op)
            std::this_thread::sleep_for(
                std::chrono::duration<double, std::milli>(
                    operationQueue.top()->time_to_start -
                    currentOperation->time_to_start));
        }
        cout << "Total time for processing = "
             << (chrono::duration_cast<chrono::duration<double, std::milli>>(
                     chrono::steady_clock::now() - timeGlobalStart))
                    .count()
             << endl;
        break;
    }

    case PROGRAM_TYPE::LSF_PYTHON: {
        lsf_initialize();
        vector<InputOperation *> zerothLayer;
        ifstream timingFile;
        timingFile.open("output/lsf-arrival-stream.txt");
        for (int i = 0; i < jobs_per_task[0]; i++) {
            string filename = networks[0].imgpath;
            filename += list1[i];
            zerothLayer.push_back(
                new InputOperation(filename, &networks[0], 0, 'm', 0));
            createLinearDAG(zerothLayer.back());
        }
        for (int i = 0; i < jobs_per_task[1]; i++) {
            string filename = networks[1].imgpath;
            filename += list2[i];
            zerothLayer.push_back(
                new InputOperation(filename, &networks[1], 0, 'm', 1));
            createLinearDAG(zerothLayer.back());
        }
        fillSMTDetails(timingFile, zerothLayer);
        std::priority_queue<Operation *, std::vector<Operation *>,
                            compareStartTimings>
            operationQueue;
        for (auto &zeroLayer : zerothLayer) {
            dagToPriorityQueue(operationQueue, (Operation *)zeroLayer);
        }
        chrono::time_point<chrono::steady_clock> timeGlobalStart =
            chrono::steady_clock::now(); // globalStart
        while (!operationQueue.empty()) {
            auto currentOperation = operationQueue.top();
            operationQueue.pop();
            {
                auto currentStream = atoi(string(1, currentOperation->chosenStream).c_str());
                if (currentOperation->op_type == 'c') {
                    cudaEventSynchronize(
                        currentOperation->parents.back()->endop);
                    checkCudaErrors(cudaEventRecord(currentOperation->startop,
                                                    lsf_stream[currentStream]));
                    assert(currentOperation->parents.back()->op_type == 'm');
                    lsf_dispatch(currentOperation, currentStream);
                    checkCudaErrors(cudaEventRecord(currentOperation->endop,
                                                    lsf_stream[currentStream]));
                } else if (currentOperation->op_type == 'm') {
                    checkCudaErrors(cudaEventRecord(currentOperation->endop,
                                                    lsf_stream[currentStream]));
                    if (currentOperation->op_layer == 0) {
                        InputOperation *zerothLayer =
                            static_cast<InputOperation *>(currentOperation);
                        zerothLayer->model->loadFile(
                            const_cast<char *>((zerothLayer->filename).c_str()),
                            lsf_stream[currentStream]);
                    } else {
                        currentOperation->model->prefetchWeights(
                            currentOperation->op_layer - 1,
                            lsf_stream[currentStream]);
                    }
                    checkCudaErrors(cudaEventRecord(currentOperation->endop,
                                                    lsf_stream[currentStream]));
                }
            }
            // sleep for (duration = start time of next op - start time of
            // current op)
            std::this_thread::sleep_for(
                std::chrono::duration<double, std::milli>(
                    operationQueue.top()->time_to_start -
                    currentOperation->time_to_start));
        }
        cout << "Total time for processing = "
             << (chrono::duration_cast<chrono::duration<double, std::milli>>(
                     chrono::steady_clock::now() - timeGlobalStart))
                    .count()
             << endl;
        break;
    }
    }
}
/*

//--------------
char filename[100];
for (int no=0;no<numOfImages;no++)
{
        strcpy(filename,networks[0].imgpath);
        strcat(filename, list1[no]);
        printf("Image No: %d Processing  %s file by pipeline01\n",no,filename);
        //Load the image from the directory into pipeline01
        networks[0].loadFile(filename);

        //Parallel execution of two pipelines
        //printf("Adding operations of tiny Yolov1 in Queue\n");
        for(int i=0;i < networks[0].num_layers;i++){
                op=Operation();
                op.model=&networks[0]; op.op_layer=i;
                op.priority= i;
                op.pipeline=1;
                //se.enqueue(op);
                pipe1.push_back(op);
        }

//Uncomment this block for sequential executin
/*
        cudaEventRecord(start);
        se.schedule_sequential(&networks[0]);
        //se.schedule();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start,stop);
        total_time+=ms;
*/
/*
        strcpy(filename,networks[1].imgpath);
        strcat(filename, list2[no]);
        printf("Processing  %s file by pipeline02\n",filename);
        //Load the image from the directory into pipeline02
        networks[1].loadFile(filename);
        for(int i=0;i < networks[1].num_layers;i++){
                op=Operation();
                op.model=&networks[1]; op.op_layer=i;//+networks[0].num_layers;
                op.priority= i;
                op.pipeline=2;
                //se.enqueue(op);
                pipe2.push_back(op);
        }


        //Call schedule routine

        cudaEventRecord(start);
        se.schedule_profile(pipe1, pipe2); //comment for seqential execution
        //se.schedule_sequential(&networks[1]);//comment for parallel execution
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start,stop);
        total_time+=ms;

}
printf("Total time to process %d images is %f\n", numOfImages, total_time);
        networks[0].deallocateSpace();
        networks[1].deallocateSpace();
*/