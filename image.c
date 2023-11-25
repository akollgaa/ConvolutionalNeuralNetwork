#include <stdlib.h>
#include <stdio.h>
#include <png.h>
/*

Compiler command:
gcc image.c -o image -I/usr/local/Cellar/libpng/1.6.37/include/libpng16 -L/usr/local/Cellar/libpng/1.6.37/lib -lpng16

*/

// Change the parameter so it is not 81. (Make it dynamic)!
float*** openImage(char *image, int imageHeight, int imageWidth) {
	FILE *fp = fopen(image, "rb");
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	png_infop end_info = png_create_info_struct(png_ptr);
	
	png_set_palette_to_rgb(png_ptr);

	png_init_io(png_ptr, fp);

	png_bytepp rows;
	png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
	rows = png_get_rows(png_ptr, info_ptr);
	// Here we create our output array
	float*** output = (float***)malloc(sizeof(float**) * 3);
	for(int i = 0; i < 3; i++) {
		output[i] = (float**)malloc(sizeof(float*) * imageHeight);
	}
	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < imageHeight; j++) {
			output[i][j] = (float*)malloc(sizeof(float) * imageWidth);
		}
	}

	// Each layer in the output array corresponds to the RGB layer in the image

	// The way rows are organized is a little weird
	// Each i-th row contains j-th columns that hold the RGB pixel data. 
	// j has to be iterated over by 3 or else it would overlap with the RGB pixel data of the previous pixel.
	// However we need to make sure output is allocating to the right position so we divide by 3 so...
	// ( (j / 3) means j increases by 1 for each iteration).

	for(int i = 0; i < imageHeight; i++) {
		for(int j = 0; j < imageWidth * 3; j += 3) {
			output[0][i][j / 3] = rows[i][j];
			output[1][i][j / 3] = rows[i][j + 1];
			output[2][i][j / 3] = rows[i][j + 2];
		}
	}

	png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
	fclose(fp);

	return output;
}
