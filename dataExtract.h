#include "algebra.h"

void outputWeightsToTXT(float*** weightGroup, struct Dimension *dimen, int numberOfFullyConnected);

void outputBiasToTXT(float*** bias, struct Dimension *dimen, int numberOfFullyConnected);

void outputFilterToTXT(float***** filter, struct Dimension *dimen, int numberOfConvolutions);