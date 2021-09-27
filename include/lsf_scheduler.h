#include "dag.h"

struct LSFScheduler{
    cudaStream_t stream1, stream2, memory_stream;
    
    void execute(InputOperation *dag1, InputOperation* dag2);
};