#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "propogation.h"
#include "algebra.h"
#include "image.h"
#include "dataExtract.h"

/*
Compile command:
gcc propogation.c image.c algebra.c dataExtract.c -o prop -I/Users/kevinkollgaard/Desktop/AK/CodeRunner_files/C/CNN/ -I/usr/local/Cellar/libpng/1.6.37/include/libpng16 -L/usr/local/Cellar/libpng/1.6.37/lib -lpng16

*/

void initalizeFitlers(int numberOfFilters, float *****filters, struct Dimension *filterDimen, int divisor) {
    for(int i = 0; i < numberOfFilters; i++) {
        filters[i] = random4DArray(filterDimen[i], divisor);
    }
}

void initalizeWeights(int numberOfWeights, float ***weights, struct Dimension *weightDimen, int divisor) {
    for(int i = 0; i < numberOfWeights; i++) {
        *(weights + i) = random2DArray(*(weightDimen + i), 1, divisor);
    }
}

void initalizeBias(int numberOfBias, float ***bias, struct Dimension *biasDimen, int divisor) {
    for(int i = 0; i < numberOfBias; i++) {
        *(bias + i) = random2DArray(*(biasDimen + i), 0, divisor);
    }
}

// We have to free all the variables because each will get rewritten once we iterate through everything again.
void reallocateVariables(struct NeuralNetworkFrame nn, struct Variables *var) {
    for(int i = 0; i < nn.numberOfConvolutions; i++) {
        free3D(var->convolutions[i], nn.convolutionalLayer[i].convolutedDimen);
        free3D(var->pooledConvolutions[i], nn.convolutionalLayer[i].outputDimen);
    }
    free(var->convolutions);
    free(var->pooledConvolutions);
    for(int i = 0; i < nn.numberOfFullyCollected; i++) {
        free2D(var->a[i], nn.fullyConnectedLayer[i].aDimen);
        free2D(var->aActivation[i], nn.fullyConnectedLayer[i].aActivationDimen);
    }
    free(var->a);
    free(var->aActivation);
    free2D(var->lengthen, nn.fullyConnectedLayer[0].aDimen);

    var->convolutions = (float****)malloc(sizeof(float***) * nn.numberOfConvolutions);
    var->pooledConvolutions = (float****)malloc(sizeof(float***) * nn.numberOfConvolutions);
    var->a = (float***)malloc(sizeof(float**) * nn.numberOfFullyCollected);
    var->aActivation = (float***)malloc(sizeof(float**) * nn.numberOfFullyCollected);
    var->lengthen = (float**)malloc(sizeof(float*));
}

void freeParameters(struct NeuralNetworkFrame nn, struct Variables *var) {
    for(int i = 0; i < nn.numberOfConvolutions; i++) {
        free4D(var->filter[i], nn.convolutionalLayer[i].filterDimen);
    }
    free(var->filter);
    for(int i = 0; i < nn.numberOfFullyCollected; i++) {
        free2D(var->weight[i], nn.fullyConnectedLayer[i].weightDimen);
    }
    for(int i = 0; i < nn.numberOfFullyCollected-1; i++) {
        free2D(var->bias[i], nn.fullyConnectedLayer[i].biasDimen);
    }
    free(var->weight);
    free(var->bias);
}

void freeNeuralNetworkFrame(struct NeuralNetworkFrame nn) {
    free(nn.convolutionalLayer);
    free(nn.fullyConnectedLayer);
}

// This was used for testing
void printDimension(struct Dimension dimen) {
    printf("\nStack: %d\n", dimen.stack);
    printf("Layer: %d\n", dimen.layer);
    printf("Row: %d\n", dimen.row);
    printf("Column: %d\n\n", dimen.column);
}

// Automatically turns on activation for all layers.
// Try to make less inputs because this function has too many
struct NeuralNetworkFrame createNetwork(int numberOfConvolutions, int numberOfFullyConnected, struct Dimension inputDimen, struct Dimension *filterDimen, struct PoolingDimension *pooling, struct Dimension *weight, struct Dimension *bias) {
    struct ConvolutionLayer *convoLayers = (struct ConvolutionLayer*)malloc(sizeof(struct ConvolutionLayer) * numberOfConvolutions);
    for(int i  = 0; i < numberOfConvolutions; i ++) {
        convoLayers[i].filterDimen = *(filterDimen + i);
        convoLayers[i].poolingDimen = *(pooling + i);
        convoLayers[i].activation = 1; // Activation for all layers

        if(i == 0) {
            convoLayers[i].inputDimen = inputDimen;
        } else {
            convoLayers[i].inputDimen = convoLayers[i-1].outputDimen;
        }
        // I wanted to have both dimensions for the input and output of each layer even though the output of layer 'n' is the input of layer 'n+1'.
        struct Dimension output;
        output.stack = 0;
        output.layer = convoLayers[i].filterDimen.stack;
        output.row = (((convoLayers[i].inputDimen.row - convoLayers[i].filterDimen.row + 1) - convoLayers[i].poolingDimen.size) / convoLayers[i].poolingDimen.stride) + 1;
        output.column = (((convoLayers[i].inputDimen.column - convoLayers[i].filterDimen.column + 1) - convoLayers[i].poolingDimen.size) / convoLayers[i].poolingDimen.stride) + 1;
        convoLayers[i].outputDimen = output;
        struct Dimension convolutedSize;
        convolutedSize.stack = 0;
        convolutedSize.layer = convoLayers[i].filterDimen.stack;
        convolutedSize.row = convoLayers[i].inputDimen.row - convoLayers[i].filterDimen.row + 1;
        convolutedSize.column = convoLayers[i].inputDimen.column - convoLayers[i].filterDimen.column + 1;
        convoLayers[i].convolutedDimen = convolutedSize;
    }

    struct WeightLayer *weightLayers = (struct WeightLayer*)malloc(sizeof(struct WeightLayer) * numberOfFullyConnected);
    for(int i = 0; i < numberOfFullyConnected; i++) {
        struct Dimension fullyConnectedDimen;
        fullyConnectedDimen.stack = 0;
        fullyConnectedDimen.layer = 0;
        fullyConnectedDimen.row = 1;
        fullyConnectedDimen.column = (*(weight + i)).row;
        struct Dimension fullyConnectedActivationDimen;
        fullyConnectedActivationDimen.stack = 0;
        fullyConnectedActivationDimen.layer = 0;
        fullyConnectedActivationDimen.row = 1;
        fullyConnectedActivationDimen.column = (*(weight + i)).column;
        struct Dimension fullyConnectedTransposeDimen;
        fullyConnectedTransposeDimen.stack = 0;
        fullyConnectedTransposeDimen.layer= 0;
        fullyConnectedTransposeDimen.row = (*(weight + i)).column;
        fullyConnectedTransposeDimen.column = 1;
        struct Dimension fullyConnectedWeightTransposeDimen;
        fullyConnectedWeightTransposeDimen.stack = 0;
        fullyConnectedWeightTransposeDimen.layer = 0;
        fullyConnectedWeightTransposeDimen.row = weight[i].column;
        fullyConnectedWeightTransposeDimen.column = weight[i].row;
        
        weightLayers[i].aTransposeDimen = fullyConnectedTransposeDimen;
        weightLayers[i].aDimen = fullyConnectedDimen;
        weightLayers[i].aActivationDimen = fullyConnectedActivationDimen;
        weightLayers[i].weightDimen = *(weight + i);
        weightLayers[i].weightTransposeDimen = fullyConnectedWeightTransposeDimen;
        weightLayers[i].biasDimen = *(bias + i);
        weightLayers[i].activation = 1;
    }

    struct NeuralNetworkFrame nn;
    nn.convolutionalLayer = convoLayers;
    nn.fullyConnectedLayer = weightLayers;
    nn.numberOfConvolutions = numberOfConvolutions; // You might be able to move this outside of the function
    nn.numberOfFullyCollected = numberOfFullyConnected;

    return nn;
}

// Make sure to free all partialweights
void backProp(float **y_hat, float **y_expected, float ***inputImage, struct NeuralNetworkFrame nn, struct Variables *var, int iter) {
    float learningRate = 0.01;
    
    // Inital difference definition
    // I do this now because of performance
    // Although it looks kinda ugly so possibly change the way you allocate memory for delta.
    float ***delta = (float***)malloc(sizeof(float**) * nn.numberOfFullyCollected);
    matrixElementWiseDivsion(y_expected, y_hat, nn.fullyConnectedLayer[nn.numberOfFullyCollected-1].aActivationDimen);
    float **y_hatSoftMaxDerivative = softmaxDerivative(y_hat, nn.fullyConnectedLayer[nn.numberOfFullyCollected-1].aActivationDimen);
    struct Dimension softMaxDeriDimen;
    softMaxDeriDimen.row = nn.fullyConnectedLayer[nn.numberOfFullyCollected-1].aActivationDimen.column;
    softMaxDeriDimen.column = nn.fullyConnectedLayer[nn.numberOfFullyCollected-1].aActivationDimen.column;
    delta[nn.numberOfFullyCollected-1] = matrixMultiply(y_expected, y_hatSoftMaxDerivative, nn.fullyConnectedLayer[nn.numberOfFullyCollected-1].aActivationDimen, softMaxDeriDimen);
    free2D(y_hatSoftMaxDerivative, softMaxDeriDimen);
    // float **y_hatLeakyReLUDerivative = leakyReLU2DDerivative(var->a[nn.numberOfFullyCollected-1], nn.fullyConnectedLayer[nn.numberOfFullyCollected-1].aActivationDimen);

    // delta[nn.numberOfFullyCollected-1][0] = (float*)malloc(sizeof(float));
    // float **y_hatLeakyReLUDerivative = leakyReLU2DDerivative(var->a[nn.numberOfFullyCollected-1], nn.fullyConnectedLayer[nn.numberOfFullyCollected-1].aActivationDimen); // Variable name sucks
    // delta[nn.numberOfFullyCollected-1][0][0] = (y_hat - y_expected) * y_hatLeakyReLUDerivative[0][0];
    // free y_hatLeakyReLUDerivative
    // free(*y_hatLeakyReLUDerivative);
    // free(y_hatLeakyReLUDerivative);

    // Note to self: delta dimensions equal aActivationDimen.

    // Weights
    float ***partialWeight = (float***)malloc(sizeof(float**) * nn.numberOfFullyCollected);
    // Go through and comb each function to make sure the dimensions and sizes for everything matches cleanly. 
    // You can display output of each variable and if there is a large amount of zeros you know something is wrong.

    // It might be more efficient to calculate the delta first and then the partialweight instead of the opposite.
    for(int i = nn.numberOfFullyCollected-1; i >= 0; i--) {
        if(i != 0) {
            float **transposedMatrix = matrixTranspose(var->aActivation[i-1], nn.fullyConnectedLayer[i-1].aActivationDimen);
            partialWeight[i] = matrixMultiply(transposedMatrix, delta[i], nn.fullyConnectedLayer[i-1].aTransposeDimen, nn.fullyConnectedLayer[i].aActivationDimen);
            // free transposedMatrix
            free2D(transposedMatrix, nn.fullyConnectedLayer[i-1].aTransposeDimen);
            float **otherTransposedMatrix = matrixTranspose(var->weight[i], nn.fullyConnectedLayer[i].weightDimen); // Possibly reassign transposedMatrix instead
            float **temporary = matrixMultiply(delta[i], otherTransposedMatrix, nn.fullyConnectedLayer[i].aActivationDimen, nn.fullyConnectedLayer[i].weightTransposeDimen);
            // free transposedMatrix again because we reassigned it
            free2D(otherTransposedMatrix, nn.fullyConnectedLayer[i].weightTransposeDimen);
            float **fullyConnectedLeakyReLUDerivative = leakyReLU2DDerivative(var->a[i-1], nn.fullyConnectedLayer[i-1].aActivationDimen); // I hate the name of this variable
        
            delta[i-1] = matrixElementWise(temporary, fullyConnectedLeakyReLUDerivative, nn.fullyConnectedLayer[i-1].aActivationDimen);
            // free fullyConnectedLeakyReLUDerivative
            free2D(fullyConnectedLeakyReLUDerivative, nn.fullyConnectedLayer[i-1].aActivationDimen);
        } else {
            float **transposedMatrix = matrixTranspose(var->lengthen, nn.fullyConnectedLayer[i].aDimen);
            partialWeight[i] = matrixMultiply(transposedMatrix, delta[i], var->lengthenTransposeDimen, nn.fullyConnectedLayer[i].aActivationDimen);
            
            // free transposedMatrix(I am bad at naming variables).
            free2D(transposedMatrix, var->lengthenTransposeDimen);
        }
    }
    // Filters

    // lastDelta has the dimension 1x2916 which is the same as aDimen[0]
    // Seperate the functions so they are not that long; assign some parameters as variables.

    float **transposedMatrix = matrixTranspose(var->weight[0], nn.fullyConnectedLayer[0].weightDimen);
    float **lastDelta = matrixMultiply(delta[0], transposedMatrix, nn.fullyConnectedLayer[0].aActivationDimen, nn.fullyConnectedLayer[0].weightTransposeDimen);
    // free transposedMatrix
    free2D(transposedMatrix, nn.fullyConnectedLayer[0].weightTransposeDimen);
    float ***unlengthenMatrix = unlengthen(lastDelta, nn.fullyConnectedLayer[0].aDimen, nn.convolutionalLayer[nn.numberOfConvolutions-1].outputDimen);
    // free lastDelta
    free2D(lastDelta, nn.fullyConnectedLayer[0].aDimen);
    int stride = nn.convolutionalLayer[nn.numberOfConvolutions-1].poolingDimen.stride; // This shortens the argument list in order to call to unMaxPool
    int size = nn.convolutionalLayer[nn.numberOfConvolutions-1].poolingDimen.size; // This shortens the argument list in order to call to unMaxPool
    float ***unMaxPoolLastDelta = unMaxPool(unlengthenMatrix, var->convolutions[nn.numberOfConvolutions-1], nn.convolutionalLayer[nn.numberOfConvolutions-1].convolutedDimen, stride, size);
    //ptrDisplayThree(unMaxPoolLastDelta, nn.convolutionalLayer[nn.numberOfConvolutions-1].convolutedDimen);
    // free unlenghtenMatrix
    free3D(unlengthenMatrix, nn.convolutionalLayer[nn.numberOfConvolutions-1].outputDimen);
    float ****fDelta = (float****)malloc(sizeof(float***) * nn.numberOfConvolutions);
    float ***deltaLeakyReLUDerivative = leakyReLUDerivative(var->convolutions[nn.numberOfConvolutions-1], nn.convolutionalLayer[nn.numberOfConvolutions-1].convolutedDimen);
    fDelta[nn.numberOfConvolutions-1] = matrixElementWise3D(unMaxPoolLastDelta, deltaLeakyReLUDerivative, nn.convolutionalLayer[nn.numberOfConvolutions-1].convolutedDimen);
    // free deltaLeakyReLUDerivative
    free3D(deltaLeakyReLUDerivative, nn.convolutionalLayer[nn.numberOfConvolutions-1].convolutedDimen);

    float *****partialFilter = (float*****)malloc(sizeof(float****) * nn.numberOfConvolutions);

    for(int i = nn.numberOfConvolutions-1; i >= 0; i--) {
        if(i != 0) {
            partialFilter[i] = convolutionFilterDerivative(var->pooledConvolutions[i-1], fDelta[i], nn.convolutionalLayer[i-1].outputDimen, nn.convolutionalLayer[i].convolutedDimen);

            float ***fDeltaLeakyReLUDerivative = leakyReLUDerivative(var->convolutions[i-1], nn.convolutionalLayer[i-1].convolutedDimen);

            float ***maxPooledMatrix = maxPool(fDeltaLeakyReLUDerivative, nn.convolutionalLayer[i-1].convolutedDimen, nn.convolutionalLayer[i-1].poolingDimen.stride, nn.convolutionalLayer[i-1].poolingDimen.size);

            float ***filterDerivative = inputFilterDerivative(maxPooledMatrix, fDelta[i], var->filter[i], nn.convolutionalLayer[i-1].outputDimen, nn.convolutionalLayer[i].filterDimen);
            
            float ***fDeltaLeakyReLU = leakyReLU(var->convolutions[i-1], nn.convolutionalLayer[i-1].convolutedDimen); // Seems unnecesary but I need it in order to do the unMaxPooling function.

            fDelta[i-1] = unMaxPool(filterDerivative, fDeltaLeakyReLU, nn.convolutionalLayer[i-1].convolutedDimen, nn.convolutionalLayer[i-1].poolingDimen.stride, nn.convolutionalLayer[i-1].poolingDimen.size);
            // free all the memory!
            free3D(fDeltaLeakyReLUDerivative, nn.convolutionalLayer[i-1].convolutedDimen);
            free3D(maxPooledMatrix, nn.convolutionalLayer[i-1].outputDimen);
            free3D(filterDerivative, nn.convolutionalLayer[i-1].outputDimen);
            free3D(fDeltaLeakyReLU, nn.convolutionalLayer[i-1].convolutedDimen);
        } else {
            partialFilter[i] = convolutionFilterDerivative(inputImage, fDelta[i], nn.convolutionalLayer[i].inputDimen, nn.convolutionalLayer[i].convolutedDimen);
        }
    }

    // Update weights
    for(int i = 0; i < nn.numberOfFullyCollected; i++) {
        matrixSubtraction(var->weight[i], matrixScalarMultiply(learningRate, partialWeight[i], nn.fullyConnectedLayer[i].weightDimen), nn.fullyConnectedLayer[i].weightDimen);
    }

    // Update biases
    // The partialBias is equal to the deltas so the biases are updated right off of the deltas.
    for(int i = 0; i < nn.numberOfFullyCollected-1; i++) { 
        matrixSubtraction(var->bias[i], matrixScalarMultiply(learningRate, delta[i], nn.fullyConnectedLayer[i].biasDimen), nn.fullyConnectedLayer[i].biasDimen);
    }

    // // Update filters
    for(int i = 0; i < nn.numberOfConvolutions; i++) {
        matrixSubtraction4D(var->filter[i], matrixScalar4DMultiply(learningRate, partialFilter[i], nn.convolutionalLayer[i].filterDimen), nn.convolutionalLayer[i].filterDimen);
    }

    // Here we free all the memory

    // Displays the partialfilters once the algorithm is over.
    // if(iter == 19) {
    //     struct Dimension *filterGroup = (struct Dimension*)malloc(sizeof(struct Dimension) * nn.numberOfConvolutions);
    //     for(int i = 0; i < nn.numberOfConvolutions; i++) {
    //         filterGroup[i].stack = nn.convolutionalLayer[i].filterDimen.stack;
    //         filterGroup[i].layer = nn.convolutionalLayer[i].filterDimen.layer;
    //         filterGroup[i].row = nn.convolutionalLayer[i].filterDimen.row;
    //         filterGroup[i].column = nn.convolutionalLayer[i].filterDimen.column;
    //     }
    //     outputFilterToTXT(partialFilter, filterGroup, nn.numberOfConvolutions);
    //     free(filterGroup);
    // }


    // Free partial weight
    for(int i = 0; i < nn.numberOfFullyCollected; i++) {
        free2D(partialWeight[i], nn.fullyConnectedLayer[i].weightDimen);
    }
    free(partialWeight);

    // // Free partial filter
    for(int i = 0; i < nn.numberOfConvolutions; i++) {
        free4D(partialFilter[i], nn.convolutionalLayer[i].filterDimen);
    }
    free(partialFilter);

    // Free the deltas
    for(int i = 0; i < nn.numberOfFullyCollected; i++) {
        free2D(delta[i], nn.fullyConnectedLayer[i].aActivationDimen);
    }
    free(delta);

    for(int i = 0; i < nn.numberOfConvolutions; i++) {
        free3D(fDelta[i], nn.convolutionalLayer[i].convolutedDimen);
    }
    free(fDelta);
}

// Backprop wont work because you need to make sure all the variables connect.
// Make the function 'cleaner' i.e. different variables names
float** forwardProp(float ***inputImage, struct NeuralNetworkFrame nn, struct Variables *var) {
    // This is just the convolutional layers

    for(int i = 0; i < nn.numberOfConvolutions; i++) {

        if(i == 0) {
            float ***normalizedInputImage = normalize(inputImage, nn.convolutionalLayer[i].inputDimen, 255.0);
            var->convolutions[i] = convolution(normalizedInputImage, var->filter[i], nn.convolutionalLayer[i].inputDimen, nn.convolutionalLayer[i].filterDimen);
            free3D(normalizedInputImage, nn.convolutionalLayer[i].inputDimen);
        } else {
            var->convolutions[i] = convolution(var->pooledConvolutions[i-1], var->filter[i], nn.convolutionalLayer[i].inputDimen, nn.convolutionalLayer[i].filterDimen);
        }
        // I took these apart so the intilization is not 10000 characters long.
        int stride = nn.convolutionalLayer[i].poolingDimen.stride;
        int size = nn.convolutionalLayer[i].poolingDimen.size;
        float ***convolutedLeakyReLU = leakyReLU(var->convolutions[i], nn.convolutionalLayer[i].convolutedDimen);
        var->pooledConvolutions[i] = maxPool(convolutedLeakyReLU, nn.convolutionalLayer[i].convolutedDimen, stride, size);
        free3D(convolutedLeakyReLU, nn.convolutionalLayer[i].convolutedDimen);

    }

    //Here we start the fully connected layer
    for(int i = 0; i < nn.numberOfFullyCollected; i++) {
        if(i == 0) {

            var->lengthen = lengthen(var->pooledConvolutions[nn.numberOfConvolutions-1], nn.convolutionalLayer[nn.numberOfConvolutions-1].outputDimen);
            // Sometimes there is a error here once we iterate the second time in matrixMultiply. 
            // It is a heap corruption. It could possibly be because the dimensions are wrong and it is trying to multiply numbers that do not exist.
            // However sometimes there is no error so it is weird.
            // I think it might be var-weight[i] because it is the only value that gets rewritten in backprop.
            var->a[i] = matrixMultiply(var->lengthen, var->weight[i], nn.fullyConnectedLayer[i].aDimen, nn.fullyConnectedLayer[i].weightDimen);
        } else {
            // When we iterate we have to use the 'a' matrix from the last iteration as the input to the current layer.
            var->a[i] = matrixMultiply(var->aActivation[i-1], var->weight[i], nn.fullyConnectedLayer[i-1].aActivationDimen, nn.fullyConnectedLayer[i].weightDimen);
        }

        if(i != nn.numberOfFullyCollected - 1) { // There is no bias on the last layer
            matrixAddition(var->a[i], var->bias[i], nn.fullyConnectedLayer[i].biasDimen);
            var->aActivation[i] = leakyReLU2D(var->a[i], nn.fullyConnectedLayer[i].aActivationDimen);
        } else { // Last layer we do softmax
            var->aActivation[i] = softmax(var->a[i], nn.fullyConnectedLayer[i].aActivationDimen);
        }

    }

    return var->aActivation[nn.numberOfFullyCollected-1];
}

int main() {

    // char image[] = "/Users/kevinkollgaard/Desktop/AK/CodeRunner_files/C/CNN/images/rgb.png";
    // int imageSize = sizeof(image)/sizeof(char) - 1; // Why is this -1?
    // float ***imageInput = openImage(image, imageSize, 128, 128);

    struct Dimension imageDimen;
    imageDimen.stack = 0;
    imageDimen.layer = 3;
    imageDimen.row = 32;
    imageDimen.column = 32;

    // struct Dimension f1Dimen;
    // struct Dimension f2Dimen;
    
    // f1Dimen.stack = 12;
    // f1Dimen.layer = 3;
    // f1Dimen.row = 5;
    // f1Dimen.column = 5;

    // f2Dimen.stack = 18;
    // f2Dimen.layer = 12;
    // f2Dimen.row = 3;
    // f2Dimen.column = 3;

    // struct Dimension *filterDimen = (struct Dimension*)malloc(sizeof(struct Dimension) * 2);
    // filterDimen[0] = f1Dimen;
    // filterDimen[1] = f2Dimen;

    // float***** filters = (float*****)malloc(sizeof(float****) * 2);
    // initalizeFitlers(2, filters, filterDimen, 200);

    // struct Dimension w1Dimen;
    // struct Dimension w2Dimen;
    // struct Dimension w3Dimen;

    // w1Dimen.stack = 0;
    // w1Dimen.layer = 0;
    // w1Dimen.row = 648;
    // w1Dimen.column = 50;

    // w2Dimen.stack = 0;
    // w2Dimen.layer = 0;
    // w2Dimen.row = 50;
    // w2Dimen.column = 20;

    // w3Dimen.stack = 0;
    // w3Dimen.layer = 0;
    // w3Dimen.row = 20;
    // w3Dimen.column = 1;

    // struct Dimension *weightDimen = (struct Dimension*)malloc(sizeof(struct Dimension) * 3);
    // weightDimen[0] = w1Dimen;
    // weightDimen[1] = w2Dimen;
    // weightDimen[2] = w3Dimen;
    // float*** weights = (float***)malloc(sizeof(float**) * 3);
    // initalizeWeights(3, weights, weightDimen, 200);

    // struct Dimension b1Dimen;
    // struct Dimension b2Dimen;

    // b1Dimen.stack = 0;
    // b1Dimen.layer = 0;
    // b1Dimen.row = 1;
    // b1Dimen.column = 50;

    // b2Dimen.stack = 0;
    // b2Dimen.layer = 0;
    // b2Dimen.row = 1;
    // b2Dimen.column = 20;

    // struct Dimension *biasDimen = (struct Dimension*)malloc(sizeof(struct Dimension) * 2);
    // biasDimen[0] = b1Dimen;
    // biasDimen[1] = b2Dimen;
    // float ***biases = (float***)malloc(sizeof(float**) * 2);
    // initalizeBias(2, biases, biasDimen, 200);

    // struct PoolingDimension pool1;
    // pool1.size = 2;
    // pool1.stride = 2;

    // struct PoolingDimension pool2;
    // pool2.size = 2;
    // pool2.stride = 2;

    // struct PoolingDimension *pooling = (struct PoolingDimension*)malloc(sizeof(struct PoolingDimension) * 2);
    // pooling[0] = pool1;
    // pooling[1] = pool2;

    // struct NeuralNetworkFrame nn = createNetwork(2, 3, imageDimen, filterDimen, pooling, weightDimen, biasDimen);

    // struct Variables *var = (struct Variables*)malloc(sizeof(struct Variables));
    // var->filter = filters;
    // var->weight = weights;
    // var->bias = biases;
    // var->convolutions = (float****)malloc(sizeof(float***) * 2);
    // var->pooledConvolutions = (float****)malloc(sizeof(float***) * 2);
    // var->a = (float***)malloc(sizeof(float**) * 3);
    // var->aActivation = (float***)malloc(sizeof(float**) * 3);
    // var->lengthen = (float**)malloc(sizeof(float*));
    // struct Dimension lengthenTranpose;
    // lengthenTranpose.stack = 0;
    // lengthenTranpose.layer = 0;
    // lengthenTranpose.row = nn.fullyConnectedLayer[0].aDimen.column;
    // lengthenTranpose.column = nn.fullyConnectedLayer[0].aDimen.row;
    // var->lengthenTransposeDimen = lengthenTranpose;

    // struct Dimension f1Dimen;
    // struct Dimension f2Dimen;
    // struct Dimension f3Dimen;

    // f1Dimen.stack = 12;
    // f1Dimen.layer = 3;
    // f1Dimen.row = 7;
    // f1Dimen.column = 7;

    // f2Dimen.stack = 24;
    // f2Dimen.layer = 12;
    // f2Dimen.row = 4;
    // f2Dimen.column = 4;

    // f3Dimen.stack = 36;
    // f3Dimen.layer = 24;
    // f3Dimen.row = 3;
    // f3Dimen.column = 3;

    // struct Dimension *filterDimen = (struct Dimension*)malloc(sizeof(struct Dimension) * 3);
    // filterDimen[0] = f1Dimen;
    // filterDimen[1] = f2Dimen;
    // filterDimen[2] = f3Dimen;

    // float *****filters = (float*****)malloc(sizeof(float****) * 3);
    // initalizeFitlers(3, filters, filterDimen, 100);

    // struct Dimension w1Dimen;
    // struct Dimension w2Dimen;
    // struct Dimension w3Dimen;
    // struct Dimension w4Dimen;

    // w1Dimen.stack = 0;
    // w1Dimen.layer = 0;
    // w1Dimen.row = 2916;
    // w1Dimen.column = 500;

    // w2Dimen.stack = 0;
    // w2Dimen.layer = 0;
    // w2Dimen.row = 500;
    // w2Dimen.column = 200;

    // w3Dimen.stack = 0;
    // w3Dimen.layer = 0;
    // w3Dimen.row = 200;
    // w3Dimen.column = 50;

    // w4Dimen.stack = 0;
    // w4Dimen.layer = 0;
    // w4Dimen.row = 50;
    // w4Dimen.column = 1;

    // struct Dimension *weightDimen = (struct Dimension*)malloc(sizeof(struct Dimension) * 4);
    // weightDimen[0] = w1Dimen;
    // weightDimen[1] = w2Dimen;
    // weightDimen[2] = w3Dimen;
    // weightDimen[3] = w4Dimen;
    // float ***weights = (float***)malloc(sizeof(float**) * 4);
    // initalizeWeights(4, weights, weightDimen, 100);

    // struct Dimension b1Dimen;
    // struct Dimension b2Dimen;
    // struct Dimension b3Dimen;

    // b1Dimen.stack = 0;
    // b1Dimen.layer = 0;
    // b1Dimen.row = 1;
    // b1Dimen.column = 500;

    // b2Dimen.stack = 0;
    // b2Dimen.layer = 0;
    // b2Dimen.row = 1;
    // b2Dimen.column = 200;

    // b3Dimen.stack = 0;
    // b3Dimen.layer = 0;
    // b3Dimen.row = 1;
    // b3Dimen.column = 50;

    // struct Dimension *biasDimen = (struct Dimension*)malloc(sizeof(struct Dimension) * 3);
    // biasDimen[0] = b1Dimen;
    // biasDimen[1] = b2Dimen;
    // biasDimen[2] = b3Dimen;
    // float ***biases = (float***)malloc(sizeof(float**) * 3);
    // initalizeBias(3, biases, biasDimen, 100);

    // struct PoolingDimension pool1;
    // pool1.size = 2;
    // pool1.stride = 2;

    // struct PoolingDimension pool2;
    // pool2.size = 2;
    // pool2.stride = 2;

    // struct PoolingDimension pool3;
    // pool3.size = 3;
    // pool3.stride = 3;

    // struct PoolingDimension *pooling = (struct PoolingDimension*)malloc(sizeof(struct PoolingDimension) * 3);
    // pooling[0] = pool1;
    // pooling[1] = pool2;
    // pooling[2] = pool3;

    // struct NeuralNetworkFrame nn = createNetwork(3, 4, imageDimen, filterDimen, pooling, weightDimen, biasDimen);

    // struct Variables *var = (struct Variables*)malloc(sizeof(struct Variables));
    // var->filter = filters;
    // var->weight = weights;
    // var->bias = biases;
    // var->convolutions = (float****)malloc(sizeof(float***) * 3);
    // var->pooledConvolutions = (float****)malloc(sizeof(float***) * 3);
    // var->a = (float***)malloc(sizeof(float**) * 4);
    // var->aActivation = (float***)malloc(sizeof(float**) * 4);
    // var->lengthen = (float**)malloc(sizeof(float*));
    // struct Dimension lengthenTranspose;
    // lengthenTranspose.stack = 0;
    // lengthenTranspose.layer = 0;
    // lengthenTranspose.row = nn.fullyConnectedLayer[0].aDimen.column;
    // lengthenTranspose.column = nn.fullyConnectedLayer[0].aDimen.row;
    // var->lengthenTransposeDimen = lengthenTranspose;

    // int divisor = 10;

    // struct Dimension f1Dimen;
    // struct Dimension f2Dimen;
    // struct Dimension f3Dimen;
    // struct Dimension f4Dimen;

    // f1Dimen.stack = 6;
    // f1Dimen.layer = 3;
    // f1Dimen.row = 13;
    // f1Dimen.column = 13;

    // f2Dimen.stack = 12;
    // f2Dimen.layer = 6;
    // f2Dimen.row = 7;
    // f2Dimen.column = 7;

    // f3Dimen.stack = 16;
    // f3Dimen.layer = 12;
    // f3Dimen.row = 5;
    // f3Dimen.column = 5;

    // f4Dimen.stack = 18;
    // f4Dimen.layer = 16;
    // f4Dimen.row = 3;
    // f4Dimen.column = 3;

    // struct Dimension *filterDimen = (struct Dimension*)malloc(sizeof(struct Dimension) * 4);
    // filterDimen[0] = f1Dimen;
    // filterDimen[1] = f2Dimen;
    // filterDimen[2] = f3Dimen;
    // filterDimen[3] = f4Dimen;

    // float *****filters = (float*****)malloc(sizeof(float****) * 4);
    // initalizeFitlers(4, filters, filterDimen, divisor);

    // struct Dimension w1Dimen;
    // struct Dimension w2Dimen;
    // struct Dimension w3Dimen;
    // struct Dimension w4Dimen;
    // struct Dimension w5Dimen;
    // struct Dimension w6Dimen;
    // struct Dimension w7Dimen;

    // w1Dimen.stack = 0;
    // w1Dimen.layer = 0;
    // w1Dimen.row = 162;
    // w1Dimen.column = 40;

    // w2Dimen.stack = 0;
    // w2Dimen.layer = 0;
    // w2Dimen.row = 40;
    // w2Dimen.column = 40;

    // w3Dimen.stack = 0;
    // w3Dimen.layer = 0;
    // w3Dimen.row = 40;
    // w3Dimen.column = 20;

    // w4Dimen.stack = 0;
    // w4Dimen.layer = 0;
    // w4Dimen.row = 20;
    // w4Dimen.column = 10;

    // w5Dimen.stack = 0;
    // w5Dimen.layer = 0;
    // w5Dimen.row = 10;
    // w5Dimen.column = 10;

    // w6Dimen.stack = 0;
    // w6Dimen.layer = 0;
    // w6Dimen.row = 10;
    // w6Dimen.column = 4;

    // w7Dimen.stack = 0;
    // w7Dimen.layer = 0;
    // w7Dimen.row = 4;
    // w7Dimen.column = 1;

    // struct Dimension *weightDimen = (struct Dimension*)malloc(sizeof(struct Dimension) * 7);
    // weightDimen[0] = w1Dimen;
    // weightDimen[1] = w2Dimen;
    // weightDimen[2] = w3Dimen;
    // weightDimen[3] = w4Dimen;
    // weightDimen[4] = w5Dimen;
    // weightDimen[5] = w6Dimen;
    // weightDimen[6] = w7Dimen;
    // float ***weights = (float***)malloc(sizeof(float**) * 7);
    // initalizeWeights(7, weights, weightDimen, divisor);

    // struct Dimension b1Dimen;
    // struct Dimension b2Dimen;
    // struct Dimension b3Dimen;
    // struct Dimension b4Dimen;
    // struct Dimension b5Dimen;
    // struct Dimension b6Dimen;

    // b1Dimen.stack = 0;
    // b1Dimen.layer = 0;
    // b1Dimen.row = 1;
    // b1Dimen.column = 40;

    // b2Dimen.stack = 0;
    // b2Dimen.layer = 0;
    // b2Dimen.row = 1;
    // b2Dimen.column = 40;

    // b3Dimen.stack = 0;
    // b3Dimen.layer = 0;
    // b3Dimen.row = 1;
    // b3Dimen.column = 20;

    // b4Dimen.stack = 0;
    // b4Dimen.layer = 0;
    // b4Dimen.row = 1;
    // b4Dimen.column = 10;

    // b5Dimen.stack = 0;
    // b5Dimen.layer = 0;
    // b5Dimen.row = 1;
    // b5Dimen.column = 10;

    // b6Dimen.stack = 0;
    // b6Dimen.layer = 0;
    // b6Dimen.row = 1;
    // b6Dimen.column = 4;

    // struct Dimension *biasDimen = (struct Dimension*)malloc(sizeof(struct Dimension) * 6);
    // biasDimen[0] = b1Dimen;
    // biasDimen[1] = b2Dimen;
    // biasDimen[2] = b3Dimen;
    // biasDimen[3] = b4Dimen;
    // biasDimen[4] = b5Dimen;
    // biasDimen[5] = b6Dimen;
    // float ***biases = (float***)malloc(sizeof(float**) * 6);
    // initalizeBias(6, biases, biasDimen, divisor);

    // struct PoolingDimension pool1;
    // pool1.size = 2;
    // pool1.stride = 2;

    // struct PoolingDimension pool2;
    // pool2.size = 2;
    // pool2.stride = 2;

    // struct PoolingDimension pool3;
    // pool3.size = 2;
    // pool3.stride = 2;

    // struct PoolingDimension pool4;
    // pool4.size = 3;
    // pool4.stride = 3;

    // struct PoolingDimension *pooling = (struct PoolingDimension*)malloc(sizeof(struct PoolingDimension) * 4);
    // pooling[0] = pool1;
    // pooling[1] = pool2;
    // pooling[2] = pool3;
    // pooling[3] = pool4;

    // struct NeuralNetworkFrame nn = createNetwork(4, 7, imageDimen, filterDimen, pooling, weightDimen, biasDimen);

    // struct Variables *var = (struct Variables*)malloc(sizeof(struct Variables));
    // var->filter = filters;
    // var->weight = weights;
    // var->bias = biases;
    // // We only have to allocate the first dimension of memory because in forwardProp we take care of the rest. 
    // var->convolutions = (float****)malloc(sizeof(float***) * 4);
    // var->pooledConvolutions = (float****)malloc(sizeof(float***) * 4);
    // var->a = (float***)malloc(sizeof(float**) * 7);
    // var->aActivation = (float***)malloc(sizeof(float**) * 7);
    // var->lengthen = (float**)malloc(sizeof(float*));
    // struct Dimension lengthenTranspose;
    // lengthenTranspose.stack = 0;
    // lengthenTranspose.layer = 0;
    // lengthenTranspose.row = nn.fullyConnectedLayer[0].aDimen.column;
    // lengthenTranspose.column = nn.fullyConnectedLayer[0].aDimen.row;
    // var->lengthenTransposeDimen = lengthenTranspose;

    int divisor = 100;

    struct Dimension f1Dimen;

    f1Dimen.stack = 24;
    f1Dimen.layer = 3;
    f1Dimen.row = 3;
    f1Dimen.column = 3;

    struct Dimension *filterDimen = (struct Dimension*)malloc(sizeof(struct Dimension) * 1);
    filterDimen[0] = f1Dimen;

    float *****filters = (float*****)malloc(sizeof(float****) * 1);
    initalizeFitlers(1, filters, filterDimen, divisor);

    struct Dimension w1Dimen;
    struct Dimension w2Dimen;

    w1Dimen.stack = 0;
    w1Dimen.layer = 0;
    w1Dimen.row = 2400;
    w1Dimen.column = 400;

    w2Dimen.stack = 0;
    w2Dimen.layer = 0;
    w2Dimen.row = 400;
    w2Dimen.column = 4;

    struct Dimension *weightDimen = (struct Dimension*)malloc(sizeof(struct Dimension) * 2);
    weightDimen[0] = w1Dimen;
    weightDimen[1] = w2Dimen;
    float ***weights = (float***)malloc(sizeof(float**) * 2);
    initalizeWeights(2, weights, weightDimen, divisor);

    struct Dimension b1Dimen;

    b1Dimen.stack = 0;
    b1Dimen.layer = 0;
    b1Dimen.row = 1;
    b1Dimen.column = 400;

    struct Dimension *biasDimen = (struct Dimension*)malloc(sizeof(struct Dimension) * 1);
    biasDimen[0] = b1Dimen;
    float ***biases = (float***)malloc(sizeof(float**) * 1);
    initalizeBias(1, biases, biasDimen, divisor);

    struct PoolingDimension pool1;
    pool1.size = 3;
    pool1.stride = 3;

    struct PoolingDimension *pooling = (struct PoolingDimension*)malloc(sizeof(struct PoolingDimension) * 1);
    pooling[0] = pool1;

    struct NeuralNetworkFrame nn = createNetwork(1, 2, imageDimen, filterDimen, pooling, weightDimen, biasDimen);

    struct Variables *var = (struct Variables*)malloc(sizeof(struct Variables));
    var->filter = filters;
    var->weight = weights;
    var->bias = biases;
    // We only have to allocate the first dimension of memory because in forwardProp we take care of the rest. 
    var->convolutions = (float****)malloc(sizeof(float***) * 1);
    var->pooledConvolutions = (float****)malloc(sizeof(float***) * 1);
    var->a = (float***)malloc(sizeof(float**) * 2);
    var->aActivation = (float***)malloc(sizeof(float**) * 2);
    var->lengthen = (float**)malloc(sizeof(float*));
    struct Dimension lengthenTranspose;
    lengthenTranspose.stack = 0;
    lengthenTranspose.layer = 0;
    lengthenTranspose.row = nn.fullyConnectedLayer[0].aDimen.column;
    lengthenTranspose.column = nn.fullyConnectedLayer[0].aDimen.row;
    var->lengthenTransposeDimen = lengthenTranspose;

        // FFMPEG
        // ffmpeg -i face.mp4 -vf fps=30 /Users/kevinkollgaard/Desktop/AK/CodeRunner_files/C/CNN/video/faceImage/out%d.png
        // ffmpeg -i background.mp4 -vf fps=30 /Users/kevinkollgaard/Desktop/AK/CodeRunner_files/C/CNN/video/backImage/out%d.png

        // ffmpeg -i wallSmall.mp4 -vf fps=30 /Users/kevinkollgaard/Desktop/AK/CodeRunner_files/C/CNN/video/wallImage/out%d.png

    int imageCounter[4] = {0, 0, 0, 0};

    for(int i = 0; i < 1000; i++) {

        //float number = (float)rand() / (float)RAND_MAX;
        float number = 1.0;
        int extra = 0;
        if(number > 0.75) {
            if((imageCounter[0]+1) / 10 >= 1) {
                extra = 1;
            } else if((imageCounter[0]+1) / 100 >= 1) {
                extra = 2;
            } else if((imageCounter[0]+1) / 1000 >= 1) {
                extra = 3;
            }
            imageCounter[0] += 1;
        } else if(number > 0.5) {
            if((imageCounter[1]+1) / 10 >= 1) {
                extra = 1;
            } else if((imageCounter[1]+1) / 100 >= 1) {
                extra = 2;
            } else if((imageCounter[1]+1) / 1000 >= 1) {
                extra = 3;
            }
            imageCounter[1] += 1;
        } else if(number > 0.25) {
            if((imageCounter[2]+1) / 10 >= 1) {
                extra = 1;
            } else if((imageCounter[2]+1) / 100 >= 1) {
                extra = 2;
            } else if((imageCounter[2]+1) / 1000 >= 1) {
                extra = 3;
            }
            imageCounter[2] += 1;
        } else {
            if((imageCounter[3]+1) / 10 >= 1) {
                extra = 1;
            } else if((imageCounter[3]+1) / 100 >= 1) {
                extra = 2;
            } else if((imageCounter[3]+1) / 1000 >= 1) {
                extra = 3;
            }
            imageCounter[3] += 1;
        }

        //char *image = (char*)malloc(sizeof(char) * (81 + extra));
        char *image = (char*)malloc(sizeof(char) * 71);
        char filePath[63] = "/Users/kevinkollgaard/Desktop/AK/CodeRunner_files/C/CNN/video/"; // 63 characters long
        for(int j = 0; j < 62; j++) {
            image[j] = filePath[j];
        }

        float **y = (float**)malloc(sizeof(float*));
        y[0] = (float*)malloc(sizeof(float) * 4);
        for(int j = 0; j < 4; j++) {
            y[0][j] = 0;
        }

        // Please put this in its own function. Very repetative
        if(number > 0.75) {
            y[0][0] = 1;
            char ending[9] = "redd.png";
            for(int j = 0; j < 8; j++) {
                image[62+j] = ending[j];
            }
            image[70] = '\0';
            // char ending[14] = "wallImage/out";
            // for(int j = 0; j < 13; j++) {
            //     image[62+j] = ending[j];
            // }
            // int value = imageCounter[0]; // We do not need this value integer.
            // char num[extra + 1];
            // sprintf(num, "%d", value);
            // for(int j = 0; j < (extra + 1); j++) {
            //     image[75+j] = num[j];
            // }
            // char fileFormat[5] = ".png";
            // for(int j = 0; j < 4; j++) {
            //     image[(76 + extra) + j] = fileFormat[j];
            // }
            // image[80 + extra] = '\0';
        } else if(number > 0.5) {
            y[0][1] = 1;
            char ending[9] = "blue.png";
            for(int j = 0; j < 8; j++) {
                image[62+j] = ending[j];
            }
            image[70] = '\0';
            // char ending[14] = "cubeImage/out";
            // for(int j = 0; j < 13; j++) {
            //     image[62+j] = ending[j];
            // }
            // int value = imageCounter[1]; // We do not need this value integer.
            // char num[extra + 1];
            // sprintf(num, "%d", value);
            // for(int j = 0; j < (extra + 1); j++) {
            //     image[75+j] = num[j];
            // }
            // char fileFormat[5] = ".png";
            // for(int j = 0; j < 4; j++) {
            //     image[(76 + extra) + j] = fileFormat[j];
            // }
            // image[80 + extra] = '\0';
        } else if(number > 0.25) {
            y[0][2] = 1;
            char ending[9] = "gree.png";
            for(int j = 0; j < 8; j++) {
                image[62+j] = ending[j];
            }
            image[70] = '\0';
            // char ending[14] = "tboxImage/out";
            // for(int j = 0; j < 13; j++) {
            //     image[62+j] = ending[j];
            // }
            // int value = imageCounter[2]; // We do not need this value integer.
            // char num[extra + 1];
            // sprintf(num, "%d", value);
            // for(int j = 0; j < (extra + 1); j++) {
            //     image[75+j] = num[j];
            // }
            // char fileFormat[5] = ".png";
            // for(int j = 0; j < 4; j++) {
            //     image[(76 + extra) + j] = fileFormat[j];
            // }
            // image[80 + extra] = '\0';
        } else {
            y[0][3] = 1;
            char ending[9] = "yell.png";
            for(int j = 0; j < 8; j++) {
                image[62+j] = ending[j];
            }
            image[70] = '\0';
            // char ending[14] = "drivImage/out";
            // for(int j = 0; j < 13; j++) {
            //     image[62+j] = ending[j];
            // }
            // int value = imageCounter[3]; // We do not need this value integer.
            // char num[extra + 1];
            // sprintf(num, "%d", value);
            // for(int j = 0; j < (extra + 1); j++) {
            //     image[75+j] = num[j];
            // }
            // char fileFormat[5] = ".png";
            // for(int j = 0; j < 4; j++) {
            //     image[(76 + extra) + j] = fileFormat[j];
            // }
            // image[80 + extra] = '\0';
        }

        float ***imageInput = openImage(image, imageDimen.row, imageDimen.column);

        float **y_hat = forwardProp(imageInput, nn, var);

        printf("%d | y: %0.1f %0.1f %0.1f %0.1f | y_hat: %0.3f %0.3f %0.3f %0.3f\n", i, y[0][0], y[0][1], y[0][2], y[0][3], y_hat[0][0], y_hat[0][1], y_hat[0][2], y_hat[0][3]);

        backProp(y_hat, y, imageInput, nn, var, i);

        struct Dimension yDimen;
        yDimen.row = 1;
        yDimen.column = 4;

        free2D(y, yDimen);
        free3D(imageInput, imageDimen);
        free(image);
        reallocateVariables(nn, var);
    }

    printf("Training complete\n");

    // for(int i = 0; i < 9; i++) {
    //     int extra = 0; // Change if you are going to do more images.
    //     char *image = (char*)malloc(sizeof(char) * (81 + extra));
    //     char filePath[63] = "/Users/kevinkollgaard/Desktop/AK/CodeRunner_files/C/CNN/video/"; // 63 characters long
    //     for(int i = 0; i < 62; i++) {
    //         image[i] = filePath[i];
    //     }
    //     char ending[14] = "testImage/out";
    //     for(int i = 0; i < 13; i++) {
    //         image[62+i] = ending[i];
    //     }
    //     int value = i + 1; // We do not need this value integer.
    //     char num[extra + 1];
    //     sprintf(num, "%d", value);
    //     for(int i = 0; i < (extra + 1); i++) {
    //         image[75+i] = num[i];
    //     }
    //     char fileFormat[5] = ".png";
    //     for(int i = 0; i < 4; i++) {
    //         image[(76 + extra) + i] = fileFormat[i];
    //     }
    //     image[80 + extra] = '\0';
    //     float ***imageInput = openImage(image, imageDimen.row, imageDimen.column);
        
    //     float y_hat[1][2] = forwardProp(imageInput, nn, var);

    //     printf("%d | y: 1.0 | y_hat: %f", i, y_hat);
    //     if(y_hat >= 0.5) {
    //         printf(" | Correct\n");
    //     } else {
    //         printf(" | Wrong\n");
    //     }

    //     free3D(imageInput, imageDimen);
    //     free(image);
    //     reallocateVariables(nn, var);
    // }

    struct Dimension *weightGroup = (struct Dimension*)malloc(sizeof(struct Dimension) * nn.numberOfFullyCollected);
    for(int i = 0; i < nn.numberOfFullyCollected; i++) {
        weightGroup[i].row = nn.fullyConnectedLayer[i].weightDimen.row;
        weightGroup[i].column = nn.fullyConnectedLayer[i].weightDimen.column;
    }
    
    struct Dimension *biasGroup = (struct Dimension*)malloc(sizeof(struct Dimension) * (nn.numberOfFullyCollected - 1));
    for(int i = 0; i < nn.numberOfFullyCollected - 1; i++) {
        biasGroup[i].row = nn.fullyConnectedLayer[i].biasDimen.row;
        biasGroup[i].column = nn.fullyConnectedLayer[i].biasDimen.column;
    }

    struct Dimension *filterGroup = (struct Dimension*)malloc(sizeof(struct Dimension) * nn.numberOfConvolutions);
    for(int i = 0; i < nn.numberOfConvolutions; i++) {
        filterGroup[i].stack = nn.convolutionalLayer[i].filterDimen.stack;
        filterGroup[i].layer = nn.convolutionalLayer[i].filterDimen.layer;
        filterGroup[i].row = nn.convolutionalLayer[i].filterDimen.row;
        filterGroup[i].column = nn.convolutionalLayer[i].filterDimen.column;
    }

    outputWeightsToTXT(var->weight, weightGroup, nn.numberOfFullyCollected);
    outputBiasToTXT(var->bias, biasGroup, nn.numberOfFullyCollected-1);
    outputFilterToTXT(var->filter, filterGroup, nn.numberOfConvolutions);

    free(weightGroup);
    free(biasGroup);
    free(filterGroup);
    
    freeParameters(nn, var);
    freeNeuralNetworkFrame(nn);
    free(var);
    
    return 0;
}