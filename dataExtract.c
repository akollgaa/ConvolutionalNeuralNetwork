#include <stdio.h>
#include <stdlib.h>
#include "algebra.h"

void outputWeightsToTXT(float*** weights, struct Dimension *dimen, int numberOfFullyConnected) {
    FILE *fp = fopen("/Users/kevinkollgaard/Desktop/AK/CodeRunner_files/C/CNN/data/weights.txt", "w+");
    if(fp == NULL) {
        printf("Error opening file");
        exit(1);
    } 

    for(int i = 0; i < numberOfFullyConnected; i++) {
        fprintf(fp, "----Weight: %d\n", i);
        for(int j = 0; j < dimen[i].row; j++) {
            for(int k = 0; k < dimen[i].column; k++) {
                if(weights[i][j][k] < 0) {
                    fprintf(fp, "%f ", weights[i][j][k]);
                } else {
                    fprintf(fp, " %f ", weights[i][j][k]);
                }
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "----------\n");
    }
    fclose(fp);
}

void outputBiasToTXT(float*** bias, struct Dimension *dimen, int numberOfFullyConnected) {
    FILE *fp = fopen("/Users/kevinkollgaard/Desktop/AK/CodeRunner_files/C/CNN/data/bias.txt", "w+");
    if(fp == NULL) {
        printf("Error opening file");
        exit(1);
    } 

    for(int i = 0; i < numberOfFullyConnected; i++) {
        fprintf(fp, "----Bias: %d\n", i);
        for(int j = 0; j < dimen[i].row; j++) {
            for(int k = 0; k < dimen[i].column; k++) {
                if(bias[i][j][k] < 0) { 
                    fprintf(fp, "%f ", bias[i][j][k]);
                } else {
                    fprintf(fp, " %f ", bias[i][j][k]);
                }
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "----------\n");
    }
    fclose(fp);
}

void outputFilterToTXT(float***** filter, struct Dimension *dimen, int numberOfConvolutions) {

    for(int i = 0; i < numberOfConvolutions; i++) {
        char fileName[80] = "/Users/kevinkollgaard/Desktop/AK/CodeRunner_files/C/CNN/data/filter/filter";
        fileName[74] = i + '0';
        // Find a better way to add this to the end of the char.
        fileName[75] = '.';
        fileName[76] = 't';
        fileName[77] = 'x';
        fileName[78] = 't';

        FILE *fp = fopen(fileName, "w+");
        if(fp == NULL) {
            printf("Error opening file.");
            exit(1);
        }   

        for(int j = 0; j < dimen[i].layer; j++) {
            for(int k = 0; k < dimen[i].row; k++) {
                for(int l = 0; l < dimen[i].stack; l++) {
                    for(int m = 0; m < dimen[i].column; m++) {
                        if(filter[i][l][j][k][m] < 0) {
                            fprintf(fp, "%f ", filter[i][l][j][k][m]);
                        } else {
                            fprintf(fp, " %f ", filter[i][l][j][k][m]);
                        }
                    }
                    fprintf(fp, " |  ");
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n\n");
        }
        
        fclose(fp);

    }
}