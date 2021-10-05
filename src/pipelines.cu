#include "solver.h"
#include <ScheduleEngine.h>
#include <cmath>
#include <cstdlib>
#include <dag.h>
#include <dirent.h>
#include <iostream>
#include <lsf_scheduler.h>
#include <utilities.h>

enum PROGRAM_TYPE { SEQUENTIAL = 1, COSCHEDULING = 2, LSF = 3, SMT = 4 };

int str_ends_with(const char *s, const char *suffix) {
    size_t slen = strlen(s);
    size_t suffix_len = strlen(suffix);

    return suffix_len <= slen && !strcmp(s + slen - suffix_len, suffix);
}

typedef unsigned char uchar;

int pipelines = 0;
int numOfImages = 1;
int main(int argc, char *argv[]) {

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
    // Pipeline 01 - Create tinyYOLO network
    vector<LayerSpecifier> layer_specifier3;
    // conv1
    {
        ConvDescriptor conv1;
        conv1.initializeValues(3, 16, 3, 3, 16, 16, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv1;
        layer_specifier3.push_back(temp);
    }
    // max pool 1
    {
        PoolingDescriptor poo01;
        poo01.initializeValues(16, 2, 2, 16, 16, 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo01;
        layer_specifier3.push_back(temp);
    }
    // conv2
    {
        ConvDescriptor conv2;
        conv2.initializeValues(16, 32, 3, 3, 8, 8, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv2;
        layer_specifier3.push_back(temp);
    }
    // max pool 2
    {
        PoolingDescriptor poo02;
        poo02.initializeValues(32, 2, 2, 8, 8, 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo02;
        layer_specifier3.push_back(temp);
    }
    // conv3
    {
        ConvDescriptor conv3;
        conv3.initializeValues(32, 16, 1, 1, 4, 4, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv3;
        layer_specifier3.push_back(temp);
    }
    // conv4
    {
        ConvDescriptor conv4;
        conv4.initializeValues(16, 128, 3, 3, 4, 4, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv4;
        layer_specifier3.push_back(temp);
    }
    // conv5
    {
        ConvDescriptor conv5;
        conv5.initializeValues(128, 16, 1, 1, 4, 4, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv5;
        layer_specifier3.push_back(temp);
    }
    // conv6
    {
        ConvDescriptor conv6;
        conv6.initializeValues(16, 128, 3, 3, 4, 4, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv6;
        layer_specifier3.push_back(temp);
    }
    // max pool 3
    {
        PoolingDescriptor poo03;
        poo03.initializeValues(128, 2, 2, 4, 4, 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo03;
        layer_specifier3.push_back(temp);
    }
    // conv7
    {
        ConvDescriptor conv7;
        conv7.initializeValues(128, 32, 1, 1, 2, 2, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv7;
        layer_specifier3.push_back(temp);
    }
    // conv8
    {
        ConvDescriptor conv8;
        conv8.initializeValues(32, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv8;
        layer_specifier3.push_back(temp);
    }
    // conv9
    {
        ConvDescriptor conv9;
        conv9.initializeValues(256, 32, 1, 1, 2, 2, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv9;
        layer_specifier3.push_back(temp);
    }
    // conv10
    {
        ConvDescriptor conv10;
        conv10.initializeValues(32, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv10;
        layer_specifier3.push_back(temp);
    }
    // max pool 4
    {
        PoolingDescriptor poo03;
        poo03.initializeValues(256, 2, 2, 2, 2, 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo03;
        layer_specifier3.push_back(temp);
    }
    // conv11
    {

        ConvDescriptor conv11;
        conv11.initializeValues(256, 64, 1, 1, 1, 1, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv11;
        layer_specifier3.push_back(temp);
    }
    // conv12
    {
        ConvDescriptor conv12;
        conv12.initializeValues(64, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv12;
        layer_specifier3.push_back(temp);
    }
    // conv13
    {
        ConvDescriptor conv13;
        conv13.initializeValues(512, 64, 1, 1, 1, 1, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv13;
        layer_specifier3.push_back(temp);
    }
    // conv 14
    {
        ConvDescriptor conv14;
        conv14.initializeValues(64, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv14;
        layer_specifier3.push_back(temp);
    }
    // conv 15
    {
        ConvDescriptor conv15;
        conv15.initializeValues(512, 128, 1, 1, 1, 1, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv15;
        layer_specifier3.push_back(temp);
    }
    // conv16
    {
        ConvDescriptor conv16;
        conv16.initializeValues(128, 1000, 1, 1, 1, 1, 0, 0, 1, 1, 0, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv16;
        layer_specifier3.push_back(temp);
    }
    // max pool4

    {
        PoolingDescriptor poo04;
        // poo04.initializeValues(1000,1, 1, 1, 1,0,0,1,1,
        // POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
        poo04.initializeValues(1000, 1, 1, 1, 1, 0, 0, 1, 1,
                               POOLING_AVERAGE); // for custom kernel
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo04;
        layer_specifier3.push_back(temp);
    }
    // softmax layer
    {
        SoftmaxDescriptor s_max;
        s_max.initializeValues(SOFTMAX_ACCURATE, SOFTMAX_MODE_INSTANCE, 1000, 1,
                               1);
        LayerSpecifier temp;
        temp.initPointer(SOFTMAX);
        *((SoftmaxDescriptor *)temp.params) = s_max;
        layer_specifier3.push_back(temp);
    }

    // cost layer as per specification

    NeuralNet tinyYolov1(layer_specifier3, DATA_FLOAT, batch_size, TENSOR_NCHW,
                         dropout_seed, softmax_eps, init_std_dev, vdnn_type,
                         vdnn_conv_algo, SGD, argv[1], argv[3]);

    pipelines++;
    // Pipeline 02 - Create tinyYOLO network
    vector<LayerSpecifier> layer_specifier4;
    // conv1
    {
        ConvDescriptor conv1;
        conv1.initializeValues(3, 16, 3, 3, 16, 16, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv1;
        layer_specifier4.push_back(temp);
    }
    // max pool 1
    {
        PoolingDescriptor poo01;
        poo01.initializeValues(16, 2, 2, 16, 16, 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo01;
        layer_specifier4.push_back(temp);
    }
    // conv2
    {
        ConvDescriptor conv2;
        conv2.initializeValues(16, 32, 3, 3, 8, 8, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv2;
        layer_specifier4.push_back(temp);
    }
    // max pool 2
    {
        PoolingDescriptor poo02;
        poo02.initializeValues(32, 2, 2, 8, 8, 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo02;
        layer_specifier4.push_back(temp);
    }
    // conv3
    {
        ConvDescriptor conv3;
        conv3.initializeValues(32, 16, 1, 1, 4, 4, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv3;
        layer_specifier4.push_back(temp);
    }
    // conv4
    {
        ConvDescriptor conv4;
        conv4.initializeValues(16, 128, 3, 3, 4, 4, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv4;
        layer_specifier4.push_back(temp);
    }
    // conv5
    {
        ConvDescriptor conv5;
        conv5.initializeValues(128, 16, 1, 1, 4, 4, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv5;
        layer_specifier4.push_back(temp);
    }
    // conv6
    {
        ConvDescriptor conv6;
        conv6.initializeValues(16, 128, 3, 3, 4, 4, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv6;
        layer_specifier4.push_back(temp);
    }
    // max pool 3
    {
        PoolingDescriptor poo03;
        poo03.initializeValues(128, 2, 2, 4, 4, 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo03;
        layer_specifier4.push_back(temp);
    }
    // conv7
    {
        ConvDescriptor conv7;
        conv7.initializeValues(128, 32, 1, 1, 2, 2, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv7;
        layer_specifier4.push_back(temp);
    }
    // conv8
    {
        ConvDescriptor conv8;
        conv8.initializeValues(32, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv8;
        layer_specifier4.push_back(temp);
    }
    // conv9
    {
        ConvDescriptor conv9;
        conv9.initializeValues(256, 32, 1, 1, 2, 2, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv9;
        layer_specifier4.push_back(temp);
    }
    // conv10
    {
        ConvDescriptor conv10;
        conv10.initializeValues(32, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv10;
        layer_specifier4.push_back(temp);
    }
    // max pool 4
    {
        PoolingDescriptor poo03;
        poo03.initializeValues(256, 2, 2, 2, 2, 0, 0, 2, 2, POOLING_MAX);
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo03;
        layer_specifier4.push_back(temp);
    }
    // conv11
    {

        ConvDescriptor conv11;
        conv11.initializeValues(256, 64, 1, 1, 1, 1, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv11;
        layer_specifier4.push_back(temp);
    }
    // conv12
    {
        ConvDescriptor conv12;
        conv12.initializeValues(64, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv12;
        layer_specifier4.push_back(temp);
    }
    // conv13
    {
        ConvDescriptor conv13;
        conv13.initializeValues(512, 64, 1, 1, 1, 1, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv13;
        layer_specifier4.push_back(temp);
    }
    // conv 14
    {
        ConvDescriptor conv14;
        conv14.initializeValues(64, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv14;
        layer_specifier4.push_back(temp);
    }
    // conv 15
    {
        ConvDescriptor conv15;
        conv15.initializeValues(512, 128, 1, 1, 1, 1, 0, 0, 1, 1, 1, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv15;
        layer_specifier4.push_back(temp);
    }
    // conv16
    {
        ConvDescriptor conv16;
        conv16.initializeValues(128, 1000, 1, 1, 1, 1, 0, 0, 1, 1, 0, RELU);
        LayerSpecifier temp;
        temp.initPointer(CONV);
        *((ConvDescriptor *)temp.params) = conv16;
        layer_specifier4.push_back(temp);
    }
    // max pool4

    {
        PoolingDescriptor poo04;
        // poo04.initializeValues(1000,1,1,1,1,0,0,1,1,
        // POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
        poo04.initializeValues(1000, 1, 1, 1, 1, 0, 0, 1, 1,
                               POOLING_AVERAGE); // for custom kernel
        LayerSpecifier temp;
        temp.initPointer(POOLING);
        *((PoolingDescriptor *)temp.params) = poo04;
        layer_specifier4.push_back(temp);
    }
    // softmax layer
    {
        SoftmaxDescriptor s_max;
        s_max.initializeValues(SOFTMAX_ACCURATE, SOFTMAX_MODE_INSTANCE, 1000, 1,
                               1);
        LayerSpecifier temp;
        temp.initPointer(SOFTMAX);
        *((SoftmaxDescriptor *)temp.params) = s_max;
        layer_specifier4.push_back(temp);
    }

    // cost layer as per specification

    NeuralNet tinyYolov2(layer_specifier4, DATA_FLOAT, batch_size, TENSOR_NCHW,
                         dropout_seed, softmax_eps, init_std_dev, vdnn_type,
                         vdnn_conv_algo, SGD, argv[2], argv[4]);

    // Now inference will start: Operations of each layer will be scheduled by
    // ScheduleEngine

    // Create an object of ScheduleEngine
    ScheduleEngine se;
    /* se.model1=&tinyYolov1;
    se.model2=&tinyYolov2; */

    // warmup code starts here
    //--------------------
    {
        auto zerothLayer =
            new InputOperation("kite.jpg", &tinyYolov1, 0, 'm', 0);
        createLinearDAG(zerothLayer);
        printf("Starting Warm up code\n");
        Operation *currentOperation = zerothLayer;
        while (currentOperation != nullptr) {
            se.enqueue(currentOperation);
            currentOperation = currentOperation->children.back();
        }
        se.warmup_schedule(zerothLayer);
        printf("Warming up code execution completed\n");
        tinyYolov1.cur_prefetch_layer = 0;
        fseek(tinyYolov1.wfp, 0, SEEK_SET);
        destroyLinearDAG(&zerothLayer);
    }
    {
        auto zerothLayer =
            new InputOperation("kite.jpg", &tinyYolov2, 0, 'm', 0);
        createLinearDAG(zerothLayer);
        printf("Starting Warm up code\n");
        Operation *currentOperation = zerothLayer;
        while (currentOperation != nullptr) {
            se.enqueue(currentOperation);
            currentOperation = currentOperation->children.back();
        }
        se.warmup_schedule(zerothLayer);
        printf("Warming up code execution completed\n");
        tinyYolov2.cur_prefetch_layer = 0;
        fseek(tinyYolov2.wfp, 0, SEEK_SET);
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
    printf("P1: %s", tinyYolov1.imgpath);
    printf("P2: %s", tinyYolov2.imgpath);
    d1 = opendir(tinyYolov1.imgpath); // arg[3] for pipeline01  image path
    d2 = opendir(tinyYolov2.imgpath); // arg[4] for pipeline01  image path
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
        printf("Unable to open direcoty %s\n", tinyYolov1.imgpath);
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
        printf("Unable to open directory %s\n", tinyYolov2.imgpath);
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
            strcpy(filename, tinyYolov1.imgpath);
            strcat(filename, list1[no]);
            auto zerothLayer1 =
                new InputOperation(filename, &tinyYolov1, 0, 'm', 1);
            createLinearDAG(zerothLayer1);
            cudaEventRecord(start);
            se.schedule_sequential(zerothLayer1, fpcf);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            total_time += ms;

            // execution of pipeline 01 is over, so deallocate its space
            // tinyYolov1.deallocateSpace();
            strcpy(filename, tinyYolov2.imgpath);
            strcat(filename, list2[no]);
            auto zerothLayer2 =
                new InputOperation(filename, &tinyYolov2, 0, 'm', 2);
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
        // tinyYolov2.deallocateSpace();
        break;
    }
    case PROGRAM_TYPE::COSCHEDULING: {
        vector<Operation *> pipe1, pipe2;
        // Parallel execution of two pipelines
        auto zerothLayer1 =
            new InputOperation("data/dog.jpg", &tinyYolov1, 0, 'm', 1);
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
            new InputOperation("data/eagle.jpg", &tinyYolov2, 0, 'm', 2);
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
        string filename = tinyYolov1.imgpath;
        filename += list1[0];
        auto zerothLayer1 =
            new InputOperation(filename, &tinyYolov1, 0, 'm', 1);
        createLinearDAG(zerothLayer1);
        filename = tinyYolov2.imgpath;
        filename += list2[0];
        auto zerothLayer2 =
            new InputOperation(filename, &tinyYolov2, 0, 'm', 2);
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
    }
}
/*

//--------------
char filename[100];
for (int no=0;no<numOfImages;no++)
{
        strcpy(filename,tinyYolov1.imgpath);
        strcat(filename, list1[no]);
        printf("Image No: %d Processing  %s file by pipeline01\n",no,filename);
        //Load the image from the directory into pipeline01
        tinyYolov1.loadFile(filename);

        //Parallel execution of two pipelines
        //printf("Adding operations of tiny Yolov1 in Queue\n");
        for(int i=0;i < tinyYolov1.num_layers;i++){
                op=Operation();
                op.model=&tinyYolov1; op.op_layer=i;
                op.priority= i;
                op.pipeline=1;
                //se.enqueue(op);
                pipe1.push_back(op);
        }

//Uncomment this block for sequential executin
/*
        cudaEventRecord(start);
        se.schedule_sequential(&tinyYolov1);
        //se.schedule();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start,stop);
        total_time+=ms;
*/
/*
        strcpy(filename,tinyYolov2.imgpath);
        strcat(filename, list2[no]);
        printf("Processing  %s file by pipeline02\n",filename);
        //Load the image from the directory into pipeline02
        tinyYolov2.loadFile(filename);
        for(int i=0;i < tinyYolov2.num_layers;i++){
                op=Operation();
                op.model=&tinyYolov2; op.op_layer=i;//+tinyYolov1.num_layers;
                op.priority= i;
                op.pipeline=2;
                //se.enqueue(op);
                pipe2.push_back(op);
        }


        //Call schedule routine

        cudaEventRecord(start);
        se.schedule_profile(pipe1, pipe2); //comment for seqential execution
        //se.schedule_sequential(&tinyYolov2);//comment for parallel execution
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start,stop);
        total_time+=ms;

}
printf("Total time to process %d images is %f\n", numOfImages, total_time);
        tinyYolov1.deallocateSpace();
        tinyYolov2.deallocateSpace();
*/