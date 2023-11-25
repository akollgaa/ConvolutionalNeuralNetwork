#include "algebra.h"

struct PoolingDimension {
    int size;
    int stride;
    int padding;
};

struct ConvolutionLayer {
    struct Dimension inputDimen;
    struct Dimension filterDimen;
    struct Dimension convolutedDimen;
    struct Dimension outputDimen;
    struct PoolingDimension poolingDimen;
    int activation;
};

struct WeightLayer {
    struct Dimension aDimen; // This is before it gets multiplied by the weight.
    struct Dimension aActivationDimen; // This is the size of the dimension after it is multiplied by the weight. It is when we apply the activation function.
    struct Dimension aTransposeDimen; // This is for backprop and it is the transposed dimension of the last value in the partial derivative for the weights.
    struct Dimension weightDimen;
    struct Dimension weightTransposeDimen;
    struct Dimension biasDimen;
    int activation;
};

// I did this to make creating a network more dynamic. It resembles OOP a lot which was not on purpose. 
struct NeuralNetworkFrame {
    struct ConvolutionLayer *convolutionalLayer; 
    struct WeightLayer *fullyConnectedLayer;
    // Can not be calculated so we have to store the size of the arrays.
    int numberOfConvolutions;
    int numberOfFullyCollected;
};

// Change the name for this because it is soooo generic.
// Keeps all the matrices in memory so backprop can reference them quickly.
struct Variables {
    float *****filter;
    float ***weight;
    float ***bias;
    float ****convolutions;
    float ****pooledConvolutions;
    float ***a;
    float ***aActivation;
    float **lengthen;
    struct Dimension lengthenTransposeDimen; // This is just for backprop; Possibly put it in nn?
};

void initalizeFitlers(int numberOfFilters, float *****filters, struct Dimension *filterDimen, int divisor);

void initalizeWeights(int numberOfWeights, float ***weights, struct Dimension *weightDimen, int divisor);

void initalizeBias(int numberOfBias, float ***bias, struct Dimension *biasDimen, int divisor);

void reallocateVariables(struct NeuralNetworkFrame nn, struct Variables *var);

void freeParameters(struct NeuralNetworkFrame nn, struct Variables *var);

void freeNeuralNetworkFrame(struct NeuralNetworkFrame nn);

void printDimension(struct Dimension dimen);

struct NeuralNetworkFrame createNetwork(int numberOfConvolutions, int numberOfFullyConnected, struct Dimension inputDimen, struct Dimension *filterDimen, struct PoolingDimension *pooling, struct Dimension *weight, struct Dimension *bias);

void initializeVariables();

float** forwardProp(float ***inputImage, struct NeuralNetworkFrame nn, struct Variables *var);

float forwardPropogation(float ***image, int imageHeight, int imageWidth);

void backProp(float **y_hat, float **y_expected, float ***inputImage, struct NeuralNetworkFrame nn, struct Variables *var, int inter);

void backPropogation(float y_hat, float y_expected);