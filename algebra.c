#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "algebra.h"


/*

TODO:

    - For the functions that edit "a"(A matrix) check if you need to return them or it just edits them and no changes necessary.
    - Add comments / documentation
    - Clean up code

*/

// Part of me feels as if the "freeing" functions should not be in the algebra file
// It it needed in algebra.c and also in other files.
void free2D(float **a, struct Dimension dimen) {
    for(int i = 0; i < dimen.row; i++) {
        free(a[i]);
    }
    free(a);
}

void free3D(float ***a, struct Dimension dimen) {
    for(int i = 0; i < dimen.layer; i++) {
        for(int j = 0; j < dimen.row; j++) {
            free(a[i][j]);
        }
    }
    for(int i = 0; i < dimen.layer; i++) {
        free(a[i]);
    }
    free(a);
}

void free4D(float ****a, struct Dimension dimen) {
    for(int i = 0; i < dimen.stack; i++) {
        for(int j = 0; j < dimen.layer; j++) {
            for(int k = 0; k < dimen.row; k++) {
                free(a[i][j][k]);
            }
        }
    }
    for(int i = 0; i < dimen.stack; i++) {
        for(int j = 0; j < dimen.layer; j++) {
            free(a[i][j]);
        }
    }
    for(int i = 0; i < dimen.stack; i++) {
        free(a[i]);
    }
    free(a);
}

void display(int a[][4], int sizeOfI, int sizeOfJ) {
    for(int i = 0; i < sizeOfI; i++) {
        for(int j = 0; j < sizeOfJ; j++) {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
}

void ptrDisplay(float **a, struct Dimension aDimen) {
    for(int i = 0; i < aDimen.row; i++) {
        for(int j = 0; j < aDimen.column; j++) {
            printf("%f ", *(*(a + i) + j));
        }
        printf("\n");
    }
}

void ptrDisplayThree(float ***a, struct Dimension aDimen) {
    for(int i = 0; i < aDimen.layer; i++) {
        for(int j = 0; j < aDimen.row; j++) {
            for(int k = 0; k < aDimen.column; k++) {
                printf("%0.3f ", *(*(*(a + i) + j) + k));
            }
            printf("\n");
        }
        printf("\n");
    }
}

void ptrDisplayFour(float ****a, struct Dimension aDimen) {
    for(int i = 0; i < aDimen.stack; i++) {
        for(int j = 0; j < aDimen.layer; j++) {
            for(int k = 0; k < aDimen.row; k++) {
                for(int l = 0; l < aDimen.column; l++) {
                    printf("%f ", *(*(*(*(a + i) + j) + k) + l));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void displaySingle(int a[], int size) {
    for(int i = 0; i < size; i ++) {
        printf("%d ", a[i]);
    }
    printf("\n");
}

// We have the option to make it negative so we can only make weights negative not biases.
float** random2DArray(struct Dimension dimen, int negative, int divisor) {
    srand(time(0));
    float **output = (float**)malloc(sizeof(float*) * dimen.row);
    for(int i = 0; i < dimen.row; i ++) {
        output[i] = (float*)malloc(sizeof(float) * dimen.column);
    }

    for(int i = 0; i < dimen.row; i++) {
        for(int j = 0; j < dimen.column; j++) {
            if((float)rand() / (float)RAND_MAX > 0.5 && negative) {
                *(*(output + i) + j) = -((float)rand() / (float)RAND_MAX) / divisor;
            } else {
                *(*(output + i) + j) = ((float)rand() / (float)RAND_MAX) / divisor;
            }
        }
    }
    
    return output;
}

float*** random3DArray(struct Dimension dimen, int divisor) {
    srand(time(0));
    float ***output = (float***)malloc(sizeof(float**) * dimen.layer);
    for(int i = 0; i < dimen.layer; i++) {
        output[i] = (float**)malloc(sizeof(float*) * dimen.row);
    }
    for(int i = 0; i < dimen.layer; i++) {
        for(int j = 0; j < dimen.row; j++) {
            output[i][j] = (float*)malloc(sizeof(float) * dimen.column);
        }
    }

    for(int i = 0; i < dimen.layer; i++) {
        for(int j = 0; j < dimen.row; j++) {
            for(int k = 0; k < dimen.column; k++) {
                *(*(*(output + i) + j) + k) = ((float)rand() / (float)RAND_MAX) / divisor;
            }
        }
    }

    return output;
}

float**** random4DArray(struct Dimension dimen, int divisor) {
    srand(time(0));
    float ****output = (float****)malloc(sizeof(float***) * dimen.stack);
    for(int i = 0; i < dimen.stack; i++) {
        output[i] = (float***)malloc(sizeof(float**) * dimen.layer);
    }
    for(int i = 0; i < dimen.stack; i++) {
        for(int j = 0; j < dimen.layer; j++) {
            output[i][j] = (float**)malloc(sizeof(float*) * dimen.row);
        }
    }
    for(int i = 0; i < dimen.stack; i++) {
        for(int j = 0; j < dimen.layer; j++) {
            for(int k = 0; k < dimen.row; k++) {
                output[i][j][k] = (float*)malloc(sizeof(float) * dimen.column);
            }
        }
    }

    for(int i = 0; i < dimen.stack; i++) {
        for(int j = 0; j < dimen.layer; j++) {
            for(int k = 0; k < dimen.row; k++) {
                for(int l = 0; l < dimen.column; l++) {
                    if((float)rand() / (float)RAND_MAX >= 0.5) {
                        output[i][j][k][l] = ((float)rand() / (float)RAND_MAX) / divisor;
                    } else {
                        output[i][j][k][l] = -((float)rand() / (float)RAND_MAX) / divisor;
                    }
                }
            }
        }
    }

    return output;
}

// You could possibly have this edit a
float*** normalize(float ***a, struct Dimension dimen, float maxInput) {
    float ***output = (float***)malloc(sizeof(float**) * dimen.layer);
    for(int i = 0; i < dimen.layer; i++) {
        output[i] = (float**)malloc(sizeof(float*) * dimen.row);
    }
    for(int i = 0; i < dimen.layer; i++) {
        for(int j = 0; j < dimen.row; j++) {
            output[i][j] = (float*)malloc(sizeof(float) * dimen.column);
        }
    }

    for(int i = 0; i < dimen.layer; i++) {
        for(int j = 0; j < dimen.row; j++) {
            for(int k = 0; k < dimen.column; k++) {
                output[i][j][k] = (maxInput - *(*(*(a + i) + j) + k)) / maxInput;
            }
        }
    }

    return output;
}

float** sigmoidFunction(float **a, struct Dimension aDimen) {
    float **output = (float**)malloc(sizeof(float*) * aDimen.row);
    for(int i = 0; i < aDimen.row; i++) {
        output[i] = (float*)malloc(sizeof(float) * aDimen.column);
    }

    for(int i = 0; i < aDimen.row; i++) {
        for(int j = 0; j < aDimen.column; j++) {
            *(*(output + i) + j) = 1 / (1 + exp(-(*(*(a + i) + j))));
        }
    }

    return output;
}

float** sigmoidFunctionDerivative(float **a, struct Dimension aDimen) {
    float **output = (float**)malloc(sizeof(float*) * aDimen.row);
    for(int i = 0; i < aDimen.row; i++) {
        output[i] = (float*)malloc(sizeof(float) * aDimen.column);
    }

    for(int i = 0; i < aDimen.row; i++) {
        for(int j = 0; j < aDimen.column; j++) {
            *(*(output + i) + j) = exp(-(*(*(a + i) + j))) / pow((1 + exp(-(*(*(a + i) + j)))), 2);
        }
    }

    return output;
}

float** softmax(float **a, struct Dimension aDimen) {
    float **output = (float**)malloc(sizeof(float*) * aDimen.row);
    for(int i = 0 ; i < aDimen.row; i++) {
        output[i] = (float*)malloc(sizeof(float) * aDimen.column);
    }

    float totalDist = 0;
    for(int i = 0; i < aDimen.row; i++) {
        for(int j = 0; j < aDimen.column; j++) {
            totalDist += exp(a[i][j]);
        }
    }

    for(int i = 0; i < aDimen.row; i++) {
        for(int j = 0; j < aDimen.column; j++) {
            output[i][j] = exp(a[i][j]) / totalDist;
        }
    }

    return output;
}

// Input is the softmax matrix
float** softmaxDerivative(float **a, struct Dimension aDimen) {
    // Creates a nxn matrix
    float **output = (float**)malloc(sizeof(float*) * aDimen.column);
    for(int i = 0 ; i < aDimen.column; i++) {
        output[i] = (float*)malloc(sizeof(float) * aDimen.column);
    }

    for(int i = 0; i < aDimen.column; i++) {
        for(int j = 0; j < aDimen.column; j++) {
            if(i == j) {
                output[i][j] = a[0][j] * (1 - a[0][j]);
            } else {
                output[i][j] = -a[0][i] * a[0][j];
            }
        }
    } 

    return output;
}

int** singleReLU(int **a, struct Dimension aDimen) {
    int **output = (int **)malloc(aDimen.row * sizeof(int*));
    *output = (int *)malloc(sizeof(int) * aDimen.row * aDimen.column);
    for(int i = 0; i < aDimen.row; i++) {
        output[i] = (*output + (int)(aDimen.column*i));
    }   

    for(int i = 0; i < aDimen.row; i++) {
        for(int j = 0; j < aDimen.column; j++) {
            if (*(*(a + i) + j) <= 0) {
                *(*(output + i) + j) = 0;
            } else {
                *(*(output + i) + j) = *(*(a + i) + j);
            }
        }
    }
    return output;
}

float*** ReLU(float ***a, struct Dimension aDimen) {
    float*** output = (float***)malloc(sizeof(float**) * aDimen.layer);
    for(int i = 0; i < aDimen.layer; i++) {
        output[i] = (float**)malloc(sizeof(float*) * aDimen.row);
    }
    for(int i = 0; i < aDimen.layer; i++) {
        for(int j = 0; j < aDimen.row; j++) {
            output[i][j] = (float*)malloc(sizeof(float) * aDimen.column);
        }
    }   

    for(int i = 0; i < aDimen.layer; i++) {
        for(int j = 0; j < aDimen.row; j++) {
            for(int k = 0; k < aDimen.column; k++) {
                if(*(*(*(a + i) +j) +k) <= 0) {
                    output[i][j][k] = 0;
                } else {
                    output[i][j][k] = *(*(*(a + i) +j) +k);
                }
            }
        }
    }

    return output;
}

int** singleReLUDerivative(int **a, struct Dimension aDimen) {
    int **output = (int **)malloc(aDimen.row * sizeof(int*));
    *output = (int *)malloc(sizeof(int) * aDimen.row * aDimen.column);
    for(int i = 0; i < aDimen.row; i++) {
        output[i] = (*output + (int)(aDimen.column*i));
    }   

    for(int i = 0; i < aDimen.row; i++) {
        for(int j = 0; j < aDimen.column; j++) {
            if (*(*(a + i) + j) <= 0) {
                *(*(output + i) + j) = 0;
            } else {
                *(*(output + i) + j) = 1;
            }
        }
    }
    return output;
}

float*** ReLUDerivative(float ***a, struct Dimension aDimen) {
    float*** output = (float***)malloc(sizeof(float**) * aDimen.layer);
    for(int i = 0; i < aDimen.layer; i++) {
        output[i] = (float**)malloc(sizeof(float*) * aDimen.row);
    }
    for(int i = 0; i < aDimen.layer; i++) {
        for(int j = 0; j < aDimen.row; j++) {
            output[i][j] = (float*)malloc(sizeof(float) * aDimen.column);
        }
    }   

    for(int i = 0; i < aDimen.layer; i++) {
        for(int j = 0; j < aDimen.row; j++) {
            for(int k = 0; k < aDimen.column; k++) {
                if(*(*(*(a + i) +j) +k) <= 0) {
                    output[i][j][k] = 0;
                } else {
                    output[i][j][k] = 1;
                }
            }
        }
    }

    return output;
}

// leaky parameter set at 0.1
float** leakyReLU2D(float **a, struct Dimension aDimen) {
    float** output = (float**)malloc(sizeof(float*) * aDimen.row);
    for(int i = 0; i < aDimen.row; i++) {
        output[i] = (float*)malloc(sizeof(float) * aDimen.column);
    }

    for(int i = 0; i < aDimen.row; i++) {
        for(int j = 0; j < aDimen.column; j++) {
            if(*(*(a + i) +j) <= 0) {
                output[i][j] = *(*(a + i) + j) * 0.1;
            } else {
                output[i][j] = *(*(a + i) + j);
            }
        }
    }

    return output;
}

// leaky parameter set at 0.01
float*** leakyReLU(float ***a, struct Dimension aDimen) {
    float*** output = (float***)malloc(sizeof(float**) * aDimen.layer);
    for(int i = 0; i < aDimen.layer; i++) {
        output[i] = (float**)malloc(sizeof(float*) * aDimen.row);
    }
    for(int i = 0; i < aDimen.layer; i++) {
        for(int j = 0; j < aDimen.row; j++) {
            output[i][j] = (float*)malloc(sizeof(float) * aDimen.column);
        }
    }   

    for(int i = 0; i < aDimen.layer; i++) {
        for(int j = 0; j < aDimen.row; j++) {
            for(int k = 0; k < aDimen.column; k++) {
                if(*(*(*(a + i) +j) +k) <= 0) {
                    output[i][j][k] = 0.01 * *(*(*(a + i) +j) +k);
                } else {
                    output[i][j][k] = *(*(*(a + i) +j) +k);
                }
            }
        }
    }

    return output;
}

float** leakyReLU2DDerivative(float **a, struct Dimension aDimen) {
    float** output = (float**)malloc(sizeof(float*) * aDimen.row);
    for(int i = 0; i < aDimen.row; i++) {
        output[i] = (float*)malloc(sizeof(float) * aDimen.column);
    }

    for(int i = 0; i < aDimen.row; i++) {
        for(int j = 0; j < aDimen.column; j++) {
            if(*(*(a + i) +j) <= 0) {
                output[i][j] = 0.1;
            } else {
                output[i][j] = 1;
            }
        }
    }

    return output;
}

// leaky parameter set at 0.1
float*** leakyReLUDerivative(float ***a, struct Dimension aDimen) {
    float*** output = (float***)malloc(sizeof(float**) * aDimen.layer);
    for(int i = 0; i < aDimen.layer; i++) {
        output[i] = (float**)malloc(sizeof(float*) * aDimen.row);
    }
    for(int i = 0; i < aDimen.layer; i++) {
        for(int j = 0; j < aDimen.row; j++) {
            output[i][j] = (float*)malloc(sizeof(float) * aDimen.column);
        }
    }   

    for(int i = 0; i < aDimen.layer; i++) {
        for(int j = 0; j < aDimen.row; j++) {
            for(int k = 0; k < aDimen.column; k++) {
                if(*(*(*(a + i) +j) +k) <= 0) {
                    output[i][j][k] = 0.01;
                } else {
                    output[i][j][k] = 1;
                }
            }
        }
    }

    return output;
}

// Edits input permanantly
float** matrixScalarMultiply(float a, float **input, struct Dimension inputDimen) {
    for(int i = 0; i < inputDimen.row; i++) {
        for(int j = 0; j < inputDimen.column; j++) {
            input[i][j] *= a;
        }
    }

    return input;
}

// Edits input permanantly
float**** matrixScalar4DMultiply(float a, float ****input, struct Dimension inputDimen) {

    for(int i = 0; i < inputDimen.stack; i++) {
        for(int j = 0; j < inputDimen.layer; j++) {
            for(int k = 0; k < inputDimen.row; k++) {
                for(int l = 0; l < inputDimen.column; l++) {
                    input[i][j][k][l] *= a;
                }
            }
        }
    }

    return input;
}

// Find a better way to create the output variable. Kind of stupid.
float** matrixMultiply(float **a, float **b, struct Dimension aDimen, struct Dimension bDimen) {
    float **output = (float**)malloc(sizeof(float*) * aDimen.row);
    for(int i = 0; i < aDimen.row; i++) {
        output[i] = (float*)malloc(sizeof(float) * bDimen.column);
    }

    for(int k = 0; k < aDimen.row; k++) {
        for(int j = 0; j < bDimen.column; j++) {
            float total = 0;
            for(int i = 0; i < aDimen.column; i++) {
                total += (*(*(a+k)+i)) * (*(*(b+i)+j));
            }
            *(*(output + k) + j) = total;
        }
    }
    return output;
}

// Edits a permanantly
float** matrixAddition(float **a, float **b, struct Dimension dimen) {
    for(int i = 0; i < dimen.row; i++) {
        for(int j = 0; j < dimen.column; j++) {
            *(*(a+i)+j) += *(*(b+i)+j);
        }
    }
    return a;
}

// a - b
// Edits a permanantly
float** matrixSubtraction(float **a, float **b, struct Dimension dimen) {
    for(int i = 0; i < dimen.row; i++) {
        for(int j = 0; j < dimen.column; j++) {
            *(*(a+i)+j) -= *(*(b+i)+j);
        }
    }
    return a;
}

// a - b
// Edits a permanantly
float**** matrixSubtraction4D(float ****a, float ****b, struct Dimension dimen) {

    for(int i = 0; i < dimen.stack; i++) {
        for(int j = 0; j < dimen.layer; j++) {
            for(int k = 0; k < dimen.row; k++) {
                for(int l = 0; l < dimen.column; l++) {
                    *(*(*(*(a + i) + j) + k) + l) -=  *(*(*(*(b + i) + j) + k) + l);
                }
            }
        }
    }

    return a;
}

// Edits a permanantly
float** matrixElementWise(float **a, float **b, struct Dimension dimen) {
    
    for(int i = 0; i < dimen.row; i++) {
        for(int j = 0; j < dimen.column; j++) {
            *(*(a+i)+j) *= *(*(b+i)+j);
        }
    }
    return a;
}

// Edits a permanantly
float*** matrixElementWise3D(float ***a, float ***b, struct Dimension dimen) {
    for(int i = 0; i < dimen.layer; i++) {
        for(int j = 0; j < dimen.row; j++) {
            for(int k = 0; k < dimen.column; k++) {
                *(*(*(a + i) + j) + k) *= *(*(*(b + i) + j) + k);
            }
        }
    }
    return a;
}

// Edits a permanantly
float** matrixElementWiseDivsion(float **a, float **b, struct Dimension dimen) {
    for(int i = 0; i < dimen.row; i++) {
        for(int j = 0; j < dimen.column; j++) {
            if(b[i][j] == 0.0) { // I do not want a divide by zero error.
                a[i][j] = a[i][j] / 0.000001;
            } else {
                a[i][j] = a[i][j] / b[i][j];
            }
        }
    }

    return a;
}

float** singleConvolution(float **input, float **filter, struct Dimension inputDimen, struct Dimension filterDimen) {
    int outputRowSize = inputDimen.row - filterDimen.row + 1;
    int outputColumnSize = inputDimen.column - filterDimen.column + 1;
    
    float** output = (float**)malloc(sizeof(float*) * outputRowSize);
    for(int i = 0; i < outputRowSize; i++) {
        output[i] = (float*)malloc(sizeof(float) * outputColumnSize);
    }

    for(int i = 0; i < outputRowSize; i++) {
        for(int j = 0; j < outputColumnSize; j++) {
            float total = 0;
            for(int k = 0; k < filterDimen.row; k++) {
                for(int l = 0; l < filterDimen.column; l++) {
                    total += *(*(input + (i + k)) + (j + l)) * *(*(filter + k) + l); 
                }   
            }
            output[i][j] = total;
        }
    }

    return output;
}

float*** convolution(float ***input, float ****filter, struct Dimension inputDimen, struct Dimension filterDimen) {
    int outputRowSize = inputDimen.row - filterDimen.row + 1;
    int outputColumnSize = inputDimen.column - filterDimen.column + 1;
    
    float*** output = (float***)malloc(sizeof(float**) * filterDimen.stack);

    for(int i = 0; i < filterDimen.stack; i++) {
        output[i] = (float**)malloc(sizeof(float*) * outputRowSize);
    }

    for(int i = 0; i < filterDimen.stack; i++) {
        for(int j = 0; j < outputRowSize; j++) {
            output[i][j] = (float*)malloc(sizeof(float) * inputDimen.column);
        }
    }

    for(int i = 0; i < filterDimen.stack; i++) {
        for(int j = 0; j < outputRowSize; j++) {
            for(int k = 0; k < outputColumnSize; k++) {
                float total = 0;
                for(int l = 0; l < filterDimen.layer; l++) {
                    for(int m = 0; m < filterDimen.row; m++) {
                        for(int n = 0; n < filterDimen.column; n++) {
                            total += *(*(*(input + l) + j + m) + k + n) * *(*(*(*(filter + i) + l) + m) + n);
                        }
                    }
                }
                output[i][j][k] = total;
            }
        }
    }

    return output;
}

float**** convolutionFilterDerivative(float ***input, float ***filter, struct Dimension inputDimen, struct Dimension filterDimen) {
    struct Dimension outputDimen;
    outputDimen.stack = filterDimen.layer;
    outputDimen.layer = inputDimen.layer;
    outputDimen.row = (inputDimen.row - filterDimen.row + 1);
    outputDimen.column = (inputDimen.column - filterDimen.column + 1);

    float**** output = (float****)malloc(sizeof(float***) * outputDimen.stack);
    for(int i = 0; i < outputDimen.stack; i++) {
        output[i] = (float***)malloc(sizeof(float**) * outputDimen.layer);
    }
    for(int i = 0; i < outputDimen.stack; i++) {
        for(int j = 0; j < outputDimen.layer; j++) {
            output[i][j] = (float**)malloc(sizeof(float*) * outputDimen.row);
        }
    }
    for(int i = 0; i < outputDimen.stack; i++) {
        for(int j = 0; j < outputDimen.layer; j++) {
            for(int k = 0; k < outputDimen.row; k++) {
                output[i][j][k] = (float*)malloc(sizeof(float) * outputDimen.column);
            }
        }
    }

    for(int i = 0; i < outputDimen.stack; i++ ) {
        for(int j = 0; j < outputDimen.layer; j++) {
            for(int k = 0; k < outputDimen.row; k++) {
                for(int l = 0; l < outputDimen.column; l++) {
                    float total = 0;
                    for(int m = 0; m < filterDimen.row; m++) {
                        for(int n = 0; n < filterDimen.column; n++) {
                            total += *(*(*(input + j) + (k + m)) + (l + n)) * *(*(*(filter + i) + m) + n); 
                        }   
                    }
                    output[i][j][k][l] = total;
                }
            }
        }
    }

    return output;
}

int** singleMaxPool(int **input, struct Dimension inputDimen, int stride, int size) {
    int outputRowSize = ((inputDimen.row - size) / stride) + 1; 
    int outputColumnSize = ((inputDimen.column - size) / stride) + 1;

    int** output = (int **)malloc(sizeof(int *) * outputRowSize);
    *output = (int *)malloc(sizeof(int) * outputRowSize * outputColumnSize);
    for(int i = 0; i < outputRowSize; i++) {
        output[i] = (*output + outputColumnSize*i);
    }

    for(int i = 0; i < outputRowSize; i++) {
        for(int j = 0; j < outputColumnSize; j++) {
            int highest = *(*(input + i*stride) + j*stride);
            for(int k = 0; k < size; k++) {
                for(int l = 0; l < size; l++) {
                    if (highest < (*(*(input + i*stride + k) + j*stride + l))) {
                        highest = (*(*(input + i*stride + k) + j*stride + l));
                    }
                }
            }
            *(*(output + i) + j) = highest;
        }
    }
    return output;
}

float*** maxPool(float ***input, struct Dimension inputDimen, int stride, int size) {
    int outputRowSize = ((inputDimen.row - size) / stride) + 1; 
    int outputColumnSize = ((inputDimen.column - size) / stride) + 1;

    float*** output = (float***)malloc(sizeof(float**) * inputDimen.layer);
    for(int i = 0; i < inputDimen.layer; i++) {
        output[i] = (float**)malloc(sizeof(float*) * outputRowSize);
    }
    for(int i = 0; i < inputDimen.layer; i++) {
        for(int j = 0; j < outputRowSize; j++) {
            output[i][j] = (float*)malloc(sizeof(float) * outputColumnSize);
        }
    }   

    for(int i = 0; i < inputDimen.layer; i++) {
        for(int j = 0; j < outputRowSize; j++) {
            for(int k = 0; k < outputColumnSize; k++) {
                float highest = *(*(*(input + i) + (stride * j)) + (stride * k));
                for(int l = 0; l < size; l++) {
                    for(int m = 0; m < size; m++) {
                        if(highest <= *(*(*(input + i) + (stride * j) + l) + (stride * k) + m)) {
                            highest = *(*(*(input + i) + (stride * j) + l) + (stride * k) + m);
                        }
                    }
                }
                *(*(*(output + i) + j) + k) = highest;
            }
        }
    }

    return output;
}

float*** unMaxPool(float ***input, float ***previousInput, struct Dimension previousInputDimen, int stride, int size) {
    int outputRowSize = ((previousInputDimen.row - size) / stride) + 1; // Actually inputRowSize
    int outputColumnSize = ((previousInputDimen.column - size) / stride) + 1; // Actually inputColumnSize

    // We use calloc instead of malloc because we must start with an array where all values are initialized to zero.
    float*** output = (float***)calloc(previousInputDimen.layer, sizeof(float**));
    for(int i = 0; i < previousInputDimen.layer; i++) {
        output[i] = (float**)calloc(previousInputDimen.row, sizeof(float*));
    }
    for(int i = 0; i < previousInputDimen.layer; i++) {
        for(int j = 0; j < previousInputDimen.row; j++) {
            output[i][j] = (float*)calloc(previousInputDimen.column, sizeof(float));
        }
    }   

    // Goes through the pooling function with the previous input.
    // Then it checks which position it pooled from and then uses that position
    // as the output for the unpooled matrix. The value is equal to the value 
    // that is trying to be unpooled.

    for(int i = 0; i < previousInputDimen.layer; i++) {
        for(int j = 0; j < outputRowSize; j++) {
            for(int k = 0; k < outputColumnSize; k++) {
                float highest = *(*(*(previousInput + i) + (stride * j)) + (stride * k));
                int lposition = 0;
                int mposition = 0;
                for(int l = 0; l < size; l++) {
                    for(int m = 1; m < size; m++) { // We already accounted for the 0th position
                        if(highest <= *(*(*(previousInput + i) + (stride * j) + l) + (stride * k) + m)) {
                            highest = *(*(*(previousInput + i) + (stride * j) + l) + (stride * k) + m);
                            lposition = l;
                            mposition = m;
                        }
                    }
                }
                *(*(*(output + i) + (stride * j) + lposition) + (stride * k) + mposition) = *(*(*(input + i) + j) + k);
            }
        }
    }

    return output;
}

float** matrixTranspose(float **a, struct Dimension aDimen) {
    float** output = (float**)malloc(sizeof(float*) * aDimen.column);
    for(int i = 0; i < aDimen.column; i++) {
        output[i] = (float*)malloc(sizeof(float) * aDimen.row);
    }

    for(int i = 0; i < aDimen.column; i++) {
        for(int j = 0; j < aDimen.row; j++) {
            *(*(output + i) + j) = *(*(a + j) + i);
        }
    }

    return output;
}

float** lengthen(float ***a, struct Dimension aDimen) {
    int outputLength = aDimen.layer * aDimen.row * aDimen.column;

    float** output = (float**)malloc(sizeof(float*));
    output[0] = (float*)malloc(sizeof(float) * outputLength);

    for(int i = 0; i < aDimen.layer; i++) {
        for(int j = 0; j < aDimen.row; j++) {
            for(int k = 0; k < aDimen.column; k++) {
                int index = (i * aDimen.row * aDimen.column) + (j * aDimen.column) + k;
                output[0][index] = *(*(*(a + i) + j) + k);
            }
        }
    }

    return output;
}

float*** unlengthen(float **a, struct Dimension aDimen, struct Dimension outputDimen) {
    float*** output = (float***)malloc(sizeof(float**) * outputDimen.layer);
    for(int i = 0; i < outputDimen.layer; i++) {
        output[i] = (float**)malloc(sizeof(float*) * outputDimen.row);
    }
    for(int i = 0; i < outputDimen.layer; i++) {
        for(int j = 0; j < outputDimen.row; j++) {
            output[i][j] = (float*)malloc(sizeof(float) * outputDimen.column);
        }
    }

    for(int i = 0; i < aDimen.column; i++) {
        int ith = i / (outputDimen.row * outputDimen.column);
        int jth = (i - ((i / (outputDimen.row * outputDimen.column)) * (outputDimen.row * outputDimen.column))) / outputDimen.row;
        int kth = (i - ((i / (outputDimen.row * outputDimen.column)) * (outputDimen.row * outputDimen.column))) % outputDimen.row;
        output[ith][jth][kth] = *(*(a) + i);
    }

    return output;
}

// This is specifically used with inputFilterDerivative
// This function is now completely irrelevant, but can be used.
float**** addPadding(float ****filter, struct Dimension filterDimen, int padding) {
    float ****output = (float****)calloc(filterDimen.stack, sizeof(float***));
    for(int i = 0; i < filterDimen.stack; i++) {
        output[i] = (float***)calloc(filterDimen.layer, sizeof(float**));
    }
    
    for(int i = 0; i < filterDimen.stack; i++) {
        for(int j = 0; j < filterDimen.layer; j++) {
            output[i][j] = (float**)calloc(((padding * 2) + filterDimen.row), sizeof(float*));
        }
    }
    for(int i = 0; i < filterDimen.stack; i++) {
        for(int j = 0; j < filterDimen.layer; j++) {
            for(int k = 0; k < ((padding * 2) + filterDimen.row); k++) {
                output[i][j][k] = (float*)calloc(((padding * 2) + filterDimen.column), sizeof(float));
            }
        }
    }

    for(int i = 0; i < filterDimen.stack; i++) {
        for(int j = 0; j < filterDimen.layer; j++) {
            for(int k = (filterDimen.row - 1); k >= 0; k--) {
                for(int l = (filterDimen.column - 1); l >= 0; l--) {
                    output[i][j][k + padding][l + padding] = filter[i][j][(filterDimen.row - 1) - k][(filterDimen.column - 1) - l];
                }
            }
        }
    }

    return output;
}

// There is no need for an errorDimen because we can just calculate it based on filter and previousInput
float*** inputFilterDerivative(float ***previousInput, float ***error, float ****filter, struct Dimension previousInputDimen, struct Dimension filterDimen) {
    float*** output = (float***)malloc(sizeof(float**) * previousInputDimen.layer);
    for(int i = 0; i < previousInputDimen.layer; i++) {
        output[i] = (float**)malloc(sizeof(float*) * previousInputDimen.row);
    }
    for(int i = 0; i < previousInputDimen.layer; i++) {
        for(int j = 0; j < previousInputDimen.row; j++) {
            output[i][j] = (float*)malloc(sizeof(float) * previousInputDimen.column); 
        }
    }

    int convolutionRow = previousInputDimen.row - filterDimen.row + 1; // errorHeight / Row
    int convolutionColumn = previousInputDimen.column - filterDimen.column + 1; // errorWidth / Column
    //int padding = convolutionRow - 1;



    // Don't ask how this works because I barely know, but it works.
    // It essientally does a backwards convolution operation over the filter which is flipped 180 degrees.
    // I do a lot of indexing in order to make it faster.
    for(int i = 0; i < previousInputDimen.layer; i++) {
        for(int j = 0; j < previousInputDimen.row; j++) {
            for(int k = 0; k < previousInputDimen.column; k++) {

                int startingOutputColumn = 0;
                int startingOutputRow = 0;
                if(k > filterDimen.column - 1) {
                    startingOutputColumn = k - (filterDimen.column - 1);
                }
                if(j > filterDimen.row - 1) {
                    startingOutputRow = j - (filterDimen.row - 1); 
                }
                
                int endingOutputColumn = k;
                int endingOutputRow = j;
                if(k > convolutionColumn - 1) {
                    endingOutputColumn = convolutionColumn - 1;
                }
                if(j > convolutionRow - 1) {
                    endingOutputRow = convolutionRow - 1;
                }

                float total = 0.0;
                for(int l = startingOutputRow; l <= endingOutputRow; l++) {
                    for(int m = startingOutputColumn; m <= endingOutputColumn; m++) {

                        int filterColumn = endingOutputColumn - m;
                        int filterRow = endingOutputRow - l;
                        for(int n = 0; n < filterDimen.stack; n++) {
                            total += filter[n][i][filterRow][filterColumn] * error[n][l][m];
                        }

                    }
                }
                output[i][j][k] = total;
            }
        } 
    }

    return output;
}

// Main function only used for testing each function

// int main () {

//     // float **input = (float**)malloc(sizeof(float*) * 1);
//     // for(int i = 0; i < 1; i++) {
//     //     input[i] = (float*)malloc(sizeof(float) * 5);
//     // }

//     // struct Dimension inputDimen;
//     // inputDimen.row = 1;
//     // inputDimen.column = 5;

//     // struct Dimension otherDimen;
//     // otherDimen.row = 5;
//     // otherDimen.column = 5;

//     // for(int i = 0; i < 5; i++) {
//     //     input[0][i] = i;
//     // }

//     // ptrDisplay(input, inputDimen);
//     // float **answer = softmax(input, inputDimen);
//     // float **other = softmaxDerivative(answer, inputDimen);
//     // ptrDisplay(answer, inputDimen);
//     // ptrDisplay(other, otherDimen);

//     // free2D(input, inputDimen);
//     // free2D(answer, inputDimen);
//     // free2D(other, otherDimen);

//     // float*** input = (float***)malloc(sizeof(float**) * 2);
//     // for(int i = 0; i < 2; i++) {
//     //     input[i] = (float**)malloc(sizeof(float*) * 4);
//     // }
//     // for(int i = 0; i < 2; i ++) {
//     //     for(int j = 0; j < 4; j++) {
//     //         input[i][j] = (float*)malloc(sizeof(float) * 4);
//     //     }
//     // }

//     // for(int i = 0; i < 2; i++) {
//     //     for(int j = 0; j < 4; j++) {
//     //         for(int k = 0; k < 4; k++) {
//     //             input[i][j][k] = 1;
//     //         }
//     //     }
//     // }

//     // struct Dimension inputDimen;
//     // inputDimen.layer = 2;
//     // inputDimen.row = 4;
//     // inputDimen.column = 4;

//     // float**** filter = (float****)malloc(sizeof(float***) * 3);
//     // for(int i = 0; i < 3; i++) {
//     //     filter[i] = (float***)malloc(sizeof(float**) * 2);
//     // }
//     // for(int i = 0; i < 3; i ++) {
//     //     for(int j = 0; j < 2; j++) {
//     //         filter[i][j] = (float**)malloc(sizeof(float*) * 2);
//     //     }
//     // }
//     // for(int i = 0; i < 3; i++) {
//     //     for(int j = 0; j < 2; j++) {
//     //         for(int k = 0; k < 2; k++) {
//     //             filter[i][j][k] = (float*)malloc(sizeof(float) * 2);
//     //         }
//     //     }
//     // }

//     // for(int i = 0; i < 3; i++) {
//     //     for(int j = 0; j < 2; j++) {
//     //         for(int k = 0; k < 2; k++) {
//     //             for(int l = 0; l < 2; l++) {
//     //                 filter[i][j][k][l] = 2;
//     //             }
//     //         }
//     //     }
//     // }

//     // struct Dimension filterDimen;
//     // filterDimen.stack = 3;
//     // filterDimen.layer = 2;
//     // filterDimen.row = 2;
//     // filterDimen.column = 2;

//     // float*** error = (float***)malloc(sizeof(float**) * 3);
//     // for(int i = 0; i < 3; i++) {
//     //     error[i] = (float**)malloc(sizeof(float*) * 3);
//     // }
//     // for(int i = 0; i < 3; i ++) {
//     //     for(int j = 0; j < 3; j++) {
//     //         error[i][j] = (float*)malloc(sizeof(float) * 3);
//     //     }
//     // }

//     // struct Dimension errorDimen;
//     // errorDimen.layer = 3;
//     // errorDimen.row = 3;
//     // errorDimen.column = 3;

//     // for(int i = 0; i < 3; i++) {
//     //     for(int j = 0; j < 3; j++) {
//     //         for(int k = 0; k < 3; k++) {
//     //             error[i][j][k] = 3;
//     //         }
//     //     }
//     // }

//     // // ptrDisplayThree(input, inputDimen);
//     // // printf("------\n");
//     // // ptrDisplayFour(filter, filterDimen);
//     // // printf("------\n");
//     // // ptrDisplayThree(error, errorDimen);
//     // ptrDisplayThree(inputFilterDerivative(input, error, filter, inputDimen, filterDimen), inputDimen);

//     //ptrDisplayFour(addPadding(filter, filterDimen, 2), paddingFilter);

//     // struct Dimension dimen;
//     // dimen.layer = 3;
//     // dimen.row = 9;
//     // dimen.column = 9;
//     // float ***output = random3DArray(dimen);
//     // //ptrDisplayThree(output, dimen);
//     // printf("---BREAK---- \n");
//     // float **length = lengthen(output, dimen);
//     // struct Dimension lengthDimen;
//     // lengthDimen.row = 1;
//     // lengthDimen.column = 243;
//     // float ***unlength = unlengthen(length, lengthDimen, dimen);
//     //ptrDisplayThree(unlength, dimen);



//     // struct Dimension di;
//     // di.row = 6;
//     // di.column = 6;
//     // di.layer = 3;
//     // di.stack = 2;
//     // ptrDisplay(random2DArray(di), di);
//     // printf("--------------\n");
//     // ptrDisplayThree(random3DArray(di), di);
//     // printf("--------------\n");
//     // ptrDisplayFour(random4DArray(di), di);
//     // printf("--------------\n");
//     // int value[][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
//     // display(value, 2, 4);

//     // int num[3] = {1, 2, 3};
//     // int *out = returning(num, 3);
//     // displaySingle(out, 3);

//     // int val[][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
//     // int *vals[4] = {val[0], val[1]};
//     // int **fin = vals;
//     // int **output = doubleReturn(fin, 2, 4);

//     // otherDisplay(output, 2, 4);

//     // Find a better way to make variables although I am kinda okay with it.
    
    
//     // int aaa[][3] = {{1, 2, 3}, {4, 5, 6}};
//     // int *aa[3] = {aaa[0], aaa[1]};
//     // int **a = aa;
//     // struct Dimension aDimen;
//     // aDimen.row = 2;
//     // aDimen.column = 3;

//     // float bbb[][2] = {{7, 8}, {9, 10}, {11, 12}};
//     // float *bb[3] = {bbb[0], bbb[1], bbb[2]};
//     // float **b = bb;
//     // struct Dimension bDimen;
//     // bDimen.row = 3;
//     // bDimen.column = 2;
//     // float **outt = sigmoidFunctionDerivative(b, bDimen);
//     // struct Dimension outDimen;
//     // outDimen.row = 3;
//     // outDimen.column = 2;
//     //ptrDisplay(outt, outDimen);

//     // int ccc[][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
//     // int *cc[4] = {ccc[0], ccc[1], ccc[2], ccc[3]};
//     // int **c = cc;
//     // struct Dimension cDim;
//     // cDim.row = 4;
//     // cDim.column = 4;

//     // int ddd[][3] = {{-1, 2, 3}, {-4, -5, 6}};
//     // int *dd[2] = {ddd[0], ddd[1]};
//     // int **d = dd;
//     // struct Dimension dDim;
//     // dDim.row = 2;
//     // dDim.column = 3;

//     // int **last = matrixElementWise(a, d, dDim);
//     // ptrDisplay(last, dDim);

//     // int **fin = convolution(c, d, cDim, dDim);
//     // ptrDisplay(fin, 3, 3);

//     // int **out = maxPool(c, cDim, 2, 2);
//     // ptrDisplay(out, 2, 2);

//     // int **outt = matrixTranspose(d, dDim);
//     // ptrDisplay(outt, 3, 2);

//     //ptrDisplay(ReLU(d, dDim), 2, 3);

//     // float aaaa[][4][4] = {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}, {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}};
//     // float *aaa[][4] = {{aaaa[0][0], aaaa[0][1], aaaa[0][2], aaaa[0][3]}, {aaaa[1][0], aaaa[1][1], aaaa[1][2], aaaa[1][3]}};
//     // float **aa[2] = {aaa[0], aaa[1]};
//     // float ***a = aa;
//     // struct Dimension aDimen;
//     // aDimen.row = 4;
//     // aDimen.column = 4;
//     // aDimen.layer = 2;

//     // struct Dimension newDimen;
//     // newDimen.layer = 2;
//     // newDimen.row = 2;
//     // newDimen.column = 2;

//     // ptrDisplayThree(maxPool(a, aDimen, 2, 2), newDimen);

//     // struct Dimension dim;
//     // dim.row = 1;
//     // dim.column = 18;

//     // ptrDisplay(lengthen(a, aDimen), dim);

//     // A 4-D pointer array - Possibly find a better way to create one?
//     // float fffffilter[][2][2][2] = {{{{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}}, {{{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}}};
//     // float *ffffilter[][2][2] = {{{fffffilter[0][0][0], fffffilter[0][0][1]}, {fffffilter[0][1][0], fffffilter[0][1][1]}}, {{fffffilter[1][0][0], fffffilter[1][0][1]}, {fffffilter[1][1][0], fffffilter[1][1][1]}}};
//     // float **fffilter[][2] = {{ffffilter[0][0], ffffilter[0][1]}, {ffffilter[1][0], ffffilter[1][1]}};
//     // float ***ffilter[2] = {fffilter[0], fffilter[1]};
//     // float ****filter = ffilter;
//     // struct Dimension filterDimen;
//     // filterDimen.stack = 2;
//     // filterDimen.layer = 2;
//     // filterDimen.row = 2;
//     // filterDimen.column = 2;

//     //int ***ans = convolution(a, filter, aDimen, filterDimen);
//     // struct Dimension ansDimen;
//     // ansDimen.layer = 2;
//     // ansDimen.row = 2;
//     // ansDimen.column = 2;
//     //ptrDisplayThree(ans, ansDimen);


//     return 0;
// }