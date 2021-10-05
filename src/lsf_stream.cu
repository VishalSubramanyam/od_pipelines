#include <ScheduleEngine.h>
#include <dirent.h>
#include <utilities.h>
#include "lsf.h"




int str_ends_with(const char *s, const char *suffix) {
    size_t slen = strlen(s);
    size_t suffix_len = strlen(suffix);

    return suffix_len <= slen && !strcmp(s + slen - suffix_len, suffix);
}

int pipelines = 0;
int numOfImages = 1;

int main(int argc, char *argv[]) {
    vDNNConvAlgo vdnn_conv_algo = vDNN_MEMORY_OPTIMAL;
    vDNNType vdnn_type = vDNN_ALL;
    int batch_size = 1;
    long long dropout_seed = 1;
    float softmax_eps = 1e-8;
    float init_std_dev = 0.1;

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
        printf("Warm up\n");
        Operation *currentOperation = zerothLayer;
        while (currentOperation != nullptr) {
            se.enqueue(currentOperation);
            currentOperation = currentOperation->children.back();
        }
        se.warmup_schedule(zerothLayer);
        printf("Warming up code copleted here executed\n");
        tinyYolov1.cur_prefetch_layer = 0;
        fseek(tinyYolov1.wfp, 0, SEEK_SET);
        destroyLinearDAG(&zerothLayer);
    }
    {
        auto zerothLayer =
            new InputOperation("kite.jpg", &tinyYolov2, 0, 'm', 0);
        createLinearDAG(zerothLayer);
        printf("Warm up\n");
        Operation *currentOperation = zerothLayer;
        while (currentOperation != nullptr) {
            se.enqueue(currentOperation);
            currentOperation = currentOperation->children.back();
        }
        se.warmup_schedule(zerothLayer);
        printf("Warming up code copleted here executed\n");
        tinyYolov2.cur_prefetch_layer = 0;
        fseek(tinyYolov2.wfp, 0, SEEK_SET);
        destroyLinearDAG(&zerothLayer);
    }
    // warmup code ends here

    
    se.createGlobalEvent();
    FILE *fpcf = fopen("lsf_stream.txt", "a");

    ifstream timingFile1, timingFile2;
    timingFile1.open("timingFile1.txt");
    timingFile2.open("timingFile2.txt");


    auto zerothLayer1 =
            new InputOperation("data/dog.jpg", &tinyYolov1, 0, 'm', 1);
    createLinearDAG(zerothLayer1);
        //loadTimings(timingFile1, zerothLayer1);
    fillExecutionTime(timingFile1,{zerothLayer1});

    auto zerothLayer2 =
            new InputOperation("data/eagle.jpg", &tinyYolov2, 0, 'm', 2);
    createLinearDAG(zerothLayer2);
        //loadTimings(timingFile2, zerothLayer2);
    fillExecutionTime(timingFile2,{zerothLayer2});
    
    //Start the execution of LSF
    vector <InputOperation*> v;
    v.push_back(zerothLayer1);
    v.push_back(zerothLayer2);
    start(v);
    fclose(fpcf);

    return 0;
}
