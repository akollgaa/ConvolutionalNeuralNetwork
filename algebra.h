#ifndef ALGEBRA_H
#define ALGEBRA_H
struct Dimension {
    int row;
    int column;
    int layer;
    int stack;
};
#endif

void free2D(float **a, struct Dimension dimen);

void free3D(float ***a, struct Dimension dimen);

void free4D(float ****a, struct Dimension dimen);

void display(int a[][4], int sizeOfI, int sizeOfJ);

void ptrDisplay(float **a, struct Dimension aDimen);

void ptrDisplayThree(float ***a, struct Dimension aDimen);

void ptrDisplayFour(float ****a, struct Dimension aDimen);

void displaySingle(int a[], int size);

float** random2DArray(struct Dimension dimen, int negative, int divisor);

float*** random3DArray(struct Dimension dimen, int divisor);

float**** random4DArray(struct Dimension dimen, int divisor);

float*** normalize(float ***a, struct Dimension dimen, float maxInput);

float** sigmoidFunction(float **a, struct Dimension aDimen);

float** sigmoidFunctionDerivative(float **a, struct Dimension aDimen);

float** softmax(float **a, struct Dimension aDimen);

float** softmaxDerivative(float **a, struct Dimension aDimen);

int** singleReLU(int **a, struct Dimension aDimen);

float*** ReLU(float ***a, struct Dimension aDimen);

float** leakyReLU2D(float **a, struct Dimension aDimen);

float*** leakyReLU(float ***a, struct Dimension aDimen);

int** singleReLUDerivative(int **a, struct Dimension aDimen);

float*** ReLUDerivative(float ***a, struct Dimension aDimen);

float** leakyReLU2DDerivative(float **a, struct Dimension aDimen);

float*** leakyReLUDerivative(float ***a, struct Dimension aDimen);

float** matrixScalarMultiply(float a, float **input, struct Dimension inputDimen);

float**** matrixScalar4DMultiply(float a, float ****input, struct Dimension inputDimen);

float** matrixMultiply(float **a, float **b, struct Dimension aDimen, struct Dimension bDimen);

float** matrixAddition(float **a, float **b, struct Dimension dimen);

float** matrixSubtraction(float **a, float **b, struct Dimension dimen);

float**** matrixSubtraction4D(float ****a, float ****b, struct Dimension dimen);

float** matrixElementWise(float **a, float **b, struct Dimension dimen);

float*** matrixElementWise3D(float ***a, float ***b, struct Dimension dimen);

float** matrixElementWiseDivsion(float **a, float **b, struct Dimension dimen);

float** singleConvolution(float **input, float **filter, struct Dimension inputDimen, struct Dimension filterDimen);

float*** convolution(float ***input, float **** filter, struct Dimension inputDimen, struct Dimension filterDimen);

float**** convolutionFilterDerivative(float ***input, float ***filter, struct Dimension inputDimen, struct Dimension filterDimen);

int** singleMaxPool(int **input, struct Dimension inputDimen, int stride, int size);

float*** maxPool(float ***input, struct Dimension inputDimen, int stride, int size);

float*** unMaxPool(float ***input, float ***previousInput, struct Dimension previousInputDimen, int stride, int size);

float** matrixTranspose(float **a, struct Dimension aDimen);

float** lengthen(float ***a, struct Dimension aDimen);

float*** unlengthen(float **a, struct Dimension aDimen, struct Dimension outputDimen);

float*** inputFilterDerivative(float ***previousInput, float ***error, float ****filter, struct Dimension previousInputDimen, struct Dimension filterDimen);