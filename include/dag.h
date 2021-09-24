//#include "neural_net.h"

#ifndef DAG_H
#define DAG_H
struct DAGnode{
	int gIndex;
	int lIndex;
	struct DAGnode *parent;
	struct DAGnode *child;
	int type;
};
typedef struct DAGnode node;

/*
typedef struct input_node{
	int gIndex;
	int lIndex;
	void *parent;
	void *child;
}inode;


typedef struct input_node{
	int gIndex;
	int lIndex;
	void *parent;
	void *child;
}inode;

typedef struct compute_node{
	int lIndex;
	int gIndex;
	void *parent;
	void *child;
}cnode;

typedef struct memory_node{
	int lIndex;
	int gIndex;
	void *parent;
	void *child;
}mnode;

typedef struct output_node{
	int gIndex;
	int lIndex;
	void *parent;
	void *child;
}onode;

*/

 class DAG{
	node *head, *current;
	//NeuralNet *nm;
	public:
		node *addNode(int,int,int,node *);
		/*void *addComputeNode(int, int, void *);
		void *addMemoryNode(int, int, void *);
		void *addInputNode(int, int, void *);
		void addOutputNode(int, int, void *);*/
		void init(node *);
		node *getNextNode();
		void displayDAG(void);
};

#endif
