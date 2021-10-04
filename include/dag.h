#include <neural_net.h>
#include <string>
#include <vector>
#ifndef DAG_H
#define DAG_H
using namespace std;
struct Operation {
  public:
    NeuralNet *model;
    char type;
    int op_layer;
    int priority;
    int pipeline;
    float time_to_start = 0.0, time_to_execute = 0.0;
    cudaEvent_t startop, endop;

    vector<Operation *> parents;
    vector<Operation *> children;
    /*
     * 'm' - memory operation involving transfer of weights
     * 'c' - compute operation
     * 'i' - image transfer operation - only the zeroth layer
     */
    char op_type;

    Operation(NeuralNet *model, char type, int op_layer, int priority,
              char op_type, int pipeline)
        : model(model), type(type), op_layer(op_layer), priority(priority),
          op_type(op_type), pipeline(pipeline) {}
    Operation(NeuralNet *model, int op_layer, int priority, char op_type,
              int pipeline)
        : model(model), op_layer(op_layer), priority(priority),
          op_type(op_type), pipeline(pipeline) {}
    Operation() = default;
};

// InputOperation - The sole operation in the zeroth layer
// Transfer of image from host to device
struct InputOperation : public Operation {
    string filename; // Name of the image file that's to be transferred
    InputOperation(string filename, NeuralNet *model, char type, int priority,
                   char op_type, int pipeline)
        : Operation(model, type, 0, priority, op_type, pipeline),
          filename(filename) {}
    InputOperation(string filename, NeuralNet *model, int priority,
                   char op_type, int pipeline)
        : Operation(model, 0, priority, op_type, pipeline), filename(filename) {
    }
};

void createLinearDAG(InputOperation *zerothLayer);
void destroyLinearDAG(InputOperation **zerothLayer);
#endif