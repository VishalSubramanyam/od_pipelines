#ifndef LSF_PIPELINES_H
#define LSF_PIPELINES_H
#include "coarsened_forward_convolution.h"
#include "image.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <dag.h>
#include <mutex>
#include <thread>
#include <utilities.h>

void lsf_initialize();
void lsf_dispatch(Operation *tp, int index);
extern vector<cudaStream_t> lsf_stream;
#endif