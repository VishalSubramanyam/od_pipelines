#include <dag.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>

void createLinearDAG(InputOperation *zerothLayer) {
    Operation *temporaryOperation = zerothLayer;
    zerothLayer->parents.push_back(nullptr);
    auto pipeline = zerothLayer->pipeline;
    for (int layer = 1; layer <= zerothLayer->model->num_layers; layer++) {
        auto op =
            new Operation(zerothLayer->model, layer, layer, 'm', pipeline);
        temporaryOperation->children.push_back(op);
        op->parents.push_back(temporaryOperation);
        temporaryOperation = op;
        op = new Operation(zerothLayer->model, layer, layer, 'c', pipeline);
        temporaryOperation->children.push_back(op);
        op->parents.push_back(temporaryOperation);
        temporaryOperation = op;
    }
    temporaryOperation->children.push_back(nullptr);
}

void destroyLinearDAG(InputOperation **zerothLayer) {
    assert((*zerothLayer)->children.size() == 1);
    Operation *current = *zerothLayer;
    while (current != nullptr) {
        Operation *next = current->children.back();
        delete current;
        current = next;
    }
    *zerothLayer = nullptr;
}