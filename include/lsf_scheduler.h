
#include<dag.h>
#include<vector>
#ifndef LSF_H
#define LSF_H


using timeMS = float;
static void execute(Operation *tp, int index);
void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void *data);
static timeMS sumOfExecutionTimes(Operation *op) ;
static void loadDeadlines() ;
static void setHEAD(vector<InputOperation *> &dags) ;
static void dispatch(Operation *tp, int index);
static void perform_lsf();
void start(vector<InputOperation *> &dags) ;
#endif