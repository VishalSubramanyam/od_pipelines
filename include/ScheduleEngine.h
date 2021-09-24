#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <queue>
#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <helper_cuda.h>
#include <pthread.h>
#include <map>
#include <cnmem.h>
#include "user_iface.h"
#include "layer_params.h"
#include "utils.h"
#include "neural_net.h"

using namespace std;
#ifndef SCHEDULEENGINE_H
#define SCHEDULEENGINE_H

struct Operation
{
public:
	NeuralNet *model;
	char type;
	int op_layer;
	int priority;
	int pipeline;
	float time_to_start, time_to_execute;
	cudaEvent_t startop, endop;

	vector<Operation *> parents;
	vector<Operation *> children;
	/* 
	 * 'm' - memory operation involving transfer of weights
	 * 'c' - compute operation
	 * 'i' - image transfer operation - only the zeroth layer
	 */
	char op_type;

	Operation(NeuralNet *model, char type, int op_layer, int priority, char op_type, int pipeline) : model(model),
																									 type(type),
																									 op_layer(op_layer),
																									 priority(priority),
																									 op_type(op_type),
																									 pipeline(pipeline) {}
	Operation(NeuralNet *model, int op_layer, int priority, char op_type, int pipeline) : model(model),
																						  op_layer(op_layer),
																						  priority(priority),
																						  op_type(op_type),
																						  pipeline(pipeline) {}
	Operation() = default;
};

// InputOperation - The sole operation in the zeroth layer
// Transfer of image from host to device
struct InputOperation : public Operation
{
	string filename; // Name of the image file that's to be transferred
	InputOperation(string filename, NeuralNet *model, char type, int priority, char op_type, int pipeline) : Operation(model, type, 0, priority, op_type, pipeline), filename(filename) {}
	InputOperation(string filename, NeuralNet *model, int priority, char op_type, int pipeline) : Operation(model, 0, priority, op_type, pipeline), filename(filename) {}
};

void createLinearDAG(InputOperation *zerothLayer);
void destroyLinearDAG(InputOperation **zerothLayer);

struct ComparePriority
{
	bool operator()(Operation *const &op1, Operation *const &op2)
	{
		return op1->priority > op2->priority;
	}
};

class ScheduleEngine
{

public:
	cudaEvent_t global_start;
	NeuralNet *model2;
	NeuralNet *model1;
	pthread_mutex_t lock;
	pthread_cond_t cond;
	map<int, Operation> pipe1, pipe2;
	priority_queue<Operation *, vector<Operation *>, ComparePriority> Q;
	queue<Operation> timeQ;
	enum stream_indicator
	{
		HIGH_COMPUTE_STREAM,
		LOW_COMPUTE_STREAM
	};
	cudaStream_t compute_streams[2];
	cudaStream_t memoryStream;
	cudnnHandle_t cudnnHandles[2];
	cublasHandle_t cublasHandles[2];
	int getQSize(void) { return Q.size(); }
	void enqueue(Operation *);
	Operation *dequeue();
	void dispatch(Operation *, stream_indicator);
	void execute(Operation *, stream_indicator);
	void schedule();
	void warmup_schedule(InputOperation *zerothLayer);
	void schedule_sequential(NeuralNet *nm);
	ScheduleEngine();
	void initMutex();
	void destroyMutex();
	void initCond();
	void destroyCond();
	NeuralNet *create_ALEXnet();
	NeuralNet *create_VGGnet();
	void startPrefetchWeights(NeuralNet *, int, cudaStream_t &);
	void schedule_profile(vector<Operation> &, vector<Operation> &);
	void createGlobalEvent(void);
};
#endif