#include <dag.h>
#include <vector>
using timeMS = float;
using namespace std;

class LSFScheduler {
  public:
    int no_of_pipelines;
    /* arrays, each element belongs to one pipeline each */
    vector<cudaStream_t> stream;
    vector<cudnnHandle_t> cudnnHandles;
    vector<cublasHandle_t> cublasHandles;
    vector<timeMS> slack;
    vector<timeMS> deadline;
    vector<Operation *> HEAD;

    cudaEvent_t now, global_start;

    void start();
    timeMS sumOfExecutionTimes(Operation *);
    void perform_lsf(cudaStream_t, cudaError_t, void *);
    void setHEAD(vector<InputOperation *> &);
    void execute(Operation *, int);
    void dispatch(Operation *, int);
    void loadDeadlines();
    LSFScheduler();
};