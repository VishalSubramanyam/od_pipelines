cmake_minimum_required(VERSION 3.5)
project( vDNNNeuralNet CUDA CXX )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
include_directories( include /usr/include /usr/local/cuda/include /usr/include/x86_64-linux-gnu /usr/local/cuda/samples/common/inc)

link_directories( /usr/local/cuda-10.2/lib64/ )

#cuda_add_executable( neural_net_vdnn.out src/main.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp)
#target_link_libraries( neural_net_vdnn.out -lcudnn -lcublas -lcurand -lpthread)

#cuda_add_executable( mnist_test.out src/mnist_test.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp)
#target_link_libraries( mnist_test.out -lcudnn -lcublas -lcurand -lpthread)

#cuda_add_executable( alexnet_test.out src/alexnet_test.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp)
#target_link_libraries( alexnet_test.out -lcudnn -lcublas -lcurand -lpthread)

#cuda_add_executable( vgg_test.out src/vgg_test.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp)
#target_link_libraries( vgg_test.out -lcudnn -lcublas -lcurand -lpthread)

#cuda_add_executable( slow_conv.out src/slow_conv.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp)
#target_link_libraries( slow_conv.out -lcudnn -lcublas -lcurand -lpthread)

#cuda_add_executable( pipelines.out src/pipelines.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp src/threadedScheduleEngine.cu src/image.cpp src/coarsened_forward_convolution.cu)
add_executable( pipelines.out src/pipelines.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp src/ScheduleEngine.cu src/image.cpp src/coarsened_forward_convolution.cu src/dag.cu src/utilities.cu src/lsf_scheduler.cu)
#add_executable( lsf_stream src/lsf_stream.cu src/lsf_scheduler.cu src/neural_net.cu src/layer_params.cu src/user_iface.cu src/utils.cu src/cnmem.cpp src/ScheduleEngine.cu src/image.cpp src/coarsened_forward_convolution.cu src/dag.cu src/utilities.cu)
target_link_libraries(pipelines.out -lcudnn -lcublas -lcurand -lpthread -g )
#target_link_libraries(lsf_stream -lcudnn -lcublas -lcurand -lpthread -g )
