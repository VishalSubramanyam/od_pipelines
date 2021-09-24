#include "dag.h"
#include<stdio.h>
#include<stdlib.h>
/*
void* DAG::addComputeNode(int g, int l, void *parent)
{
	cnode *p;
	p=(cnode *)malloc(sizeof(cnode));
	p->parent=parent;
	(p->parent)->child=(void *)p;
	p->child=NULL;
	p->gIndex=g;
	p->lIndex=l;
	return((void *)p);
}

void* DAG::addMemoryNode(int g, int l, void *parent){
	mnode *p;
	p=(mnode *)malloc(sizeof(mnode));
	p->parent=parent;
	(p->parent)->child=(void *)p;
	p->child=NULL;
	p->gIndex=g;
	p->lIndex=l;
	return((void *)p);

}
void* DAG::addInputNode(int g,int l, void *parent){
	inode *p;
	p=(inode *)malloc(sizeof(inode));
	p->parent=parent;
	p->child=NULL;
	p->gIndex=g;
	p->lIndex=l;
	head=p;
	current=(void *)p;
	return((void *)p);
}
void DAG::addOutputNode(int g, int l, void *parent){
	onode *p;
	p=(onode *)malloc(sizeof(onode));
	p->parent=parent;
	p->child=NULL;
	p->gIndex=g;
	p->lIndex=l;
}
*/
void DAG::init(node *p){
	head=p;
	current=p;
}

//Always returns current node from DAG
node *DAG::getNextNode(){
	node *n=current;
	current=current->child;
	return(n);
}

void DAG::displayDAG(){
	node *n=head;
	while(n !=NULL){
		printf("%d %d\n", n->gIndex,n->lIndex);
		n=n->child;
	}
}

node* DAG::addNode(int g,int l,int t, node *parent){
	node *p;
	p=(node *)malloc(sizeof(node));
	p->parent=parent;
	if(parent!=NULL){
		parent->child=p;
	}
	else{
		p->child=NULL;
	}
	p->gIndex=g;
	p->lIndex=l;
	p->type=t;
	return(p);
}
