#include <stdio.h>
#include <stdlib.h>

struct TreeNode {
    int data;
    int childCount;
    struct TreeNode** children;
};

struct TreeNode* createTreeNode(int data) {
    struct TreeNode* newNode = (struct TreeNode*)malloc(sizeof(struct TreeNode));
    newNode->data = data;
    newNode->childCount = 0;
    newNode->children = NULL;
    return newNode;
}

void addChild(struct TreeNode* parent, struct TreeNode* child) {
    parent->childCount++;
    parent->children = (struct TreeNode**)realloc(parent->children, parent->childCount * sizeof(struct TreeNode*));
    parent->children[parent->childCount - 1] = child;
}

void DFS(struct TreeNode* node) {
    if (node == NULL) {
        return;
    }
    printf("%d ", node->data);
    for (int i = 0; i < node->childCount; i++) {
        DFS(node->children[i]);
    }
}

void freeTree(struct TreeNode* node) {
    if (node == NULL) {
        return;
    }
    for (int i = 0; i < node->childCount; i++) {
        freeTree(node->children[i]);
    }
    free(node->children);
    free(node);
}

int main() {
    struct TreeNode* root = createTreeNode(0);

    struct TreeNode* child1 = createTreeNode(1);
    struct TreeNode* child2 = createTreeNode(2);
    struct TreeNode* child3 = createTreeNode(3);

    addChild(root, child1);
    addChild(root, child2);
    addChild(root, child3);

    struct TreeNode* child1_1 = createTreeNode(2);
    struct TreeNode* child1_2 = createTreeNode(3);
    struct TreeNode* child1_3 = createTreeNode(4);

    addChild(child1, child1_1);
    addChild(child1, child1_2);
    addChild(child1, child1_3); 

    struct TreeNode* child1_1_1 = createTreeNode(3);
    struct TreeNode* child1_1_2 = createTreeNode(4);
    struct TreeNode* child1_1_3 = createTreeNode(5);
 
    addChild(child1_1, child1_1_1);
    addChild(child1_1, child1_1_2);
    addChild(child1_1, child1_1_3); 

    
    struct TreeNode* child1_2_1 = createTreeNode(4);
    struct TreeNode* child1_2_2 = createTreeNode(5);
 
    addChild(child1_2, child1_2_1);
    addChild(child1_2, child1_2_2);

    struct TreeNode* child1_3_1 = createTreeNode(5);

    addChild(child1_3, child1_3_1);


    struct TreeNode* child2_1 = createTreeNode(3);
    struct TreeNode* child2_2 = createTreeNode(4);

    addChild(child2, child2_1);
    addChild(child2, child2_2);

    struct TreeNode* child2_1_1 = createTreeNode(4);
    struct TreeNode* child2_1_2 = createTreeNode(5);
 
    addChild(child2_1, child2_1_1);
    addChild(child2_1, child2_1_2);

    
    struct TreeNode* child2_2_2 = createTreeNode(5);
 
    addChild(child2_2, child2_2_2);


   struct TreeNode* child3_1 = createTreeNode(4);

   addChild(child3, child3_1);

   struct TreeNode* child3_1_1 = createTreeNode(5);

   addChild(child3_1_1, child3_1);


    printf("Depth First Traversal of the tree:\n");
    DFS(root);
    
    freeTree(root);
    return 0;
}

