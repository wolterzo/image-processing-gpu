#include <stdio.h>
#include <time.h>
#include <Magick++.h>
#include <iostream>
#include <string>
using namespace std; 
using namespace Magick;

#define BLOCK_SIZE 32
#define RANGE 256

typedef struct pixelRGB {
  int r;
  int g;
  int b;
} pixelRGB;

__global__ void image_to_grayscale(pixelRGB* pixels) {
  //Below equation found here: http://www.mathworks.com/matlabcentral/answers/99136-how-do-i-convert-my-rgb-image-to-grayscale-without-using-the-image-processing-toolbox?
  //intensity = 0.2989*red + 0.5870*green + 0.1140*blue
  int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  pixelRGB color = pixels[index];
  double intensity = ((double) 0.2989 * color.r) + ((double) 0.5870 * color.g) + ((double) 0.1140 * color.b);
  //printf("%lf\n", intensity);
  pixels[index].r = intensity;
  pixels[index].g = intensity;
  pixels[index].b = intensity;
  //printf("%d\n", pixels[index].r);
}

__global__ void matrix_filter_image(pixelRGB* input_pixels, pixelRGB* output_pixels, int w, int h, double* filter, int fWidth, int fHeight, double factor, double bias) {
  int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  
  int x = index % w;
  int y = index / w;

  
  // Looked at 'http://lodev.org/cgtutor/filtering.html' to figure out complexfilters

  double red = 0;
  double green = 0;
  double blue = 0;

  for(int fY = 0; fY < fHeight; fY++) {
    for(int fX = 0; fX < fWidth; fX++) {
      int imageX = (x - fWidth / 2 + fX + w) % w;
      int imageY = (y - fHeight / 2 + fY + h) % h;
      red += input_pixels[imageY * w + imageX].r * filter[fY * fWidth+fX];
      green += input_pixels[imageY * w + imageX].g * filter[fY * fWidth+fX];
      blue += input_pixels[imageY * w + imageX].b * filter[fY * fWidth+fX];
    }
  }
  // printf("In loop\n");
  output_pixels[index].r = min(max(int(factor * red + bias), 0), 255 * RANGE);
  output_pixels[index].g = min(max(int(factor * green + bias), 0), 255 * RANGE);
  output_pixels[index].b = min(max(int(factor * blue + bias), 0), 255 * RANGE);
}

int main (int argc, char** argv) {
  InitializeMagick(*argv);
  
  printf("In main\n");
  
  string filename = argv[1];//("bridge.jpg");
  Image image(filename);

  //START TIMER
  clock_t start = clock();


  //Filter
  int fHeight = 9;
  int fWidth = 9;

  double cpu_filter[] =
    {
      1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1,
    };
    
    // {
    //   -1, -1,  0,
    //   -1,  0,  1,
    //    0,  1,  1
    // };
  double factor = 1.0 / 9.0;
  double bias = 0.0;

  double* gpu_filter;
  if(cudaMalloc(&gpu_filter, sizeof(double) * fWidth* fHeight) != cudaSuccess) {
    fprintf(stderr, "Failed to create filter matrix for the gpu\n");
    exit(2);
  }
  
  //Copy contents from cpu to gpu
  if(cudaMemcpy(gpu_filter, cpu_filter, sizeof(double) * fWidth * fHeight, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy filter matrix from CPU to the GPU\n");
  }
  
  int width = image.columns();
  int height = image.rows();
  printf("width: %d, height: %d\n", width, height);
  // Rounds up number of iterations
  int offset = fHeight / 2;
  int iterations = (((width * (height + offset)) + (65535 * BLOCK_SIZE))-1) / (65535 * BLOCK_SIZE);
  printf("number of iterations: %d\n", iterations);
  int modheight = height / iterations;
  int startheight = height / iterations;
  int modheight_plus;
  int startheight_plus;
  printf("modheight is: %d\n", modheight);
  pixelRGB* master_pixels;
  master_pixels = (pixelRGB*) malloc(sizeof(pixelRGB) * width * height);
  
  
  for (int i = 0; i < iterations; i++){
    int iter = i;
    int remainder = height - (modheight * i);
    if (remainder < modheight && remainder != 0) {
      printf("i is %d\n", i);
      printf("rm is %d\n", remainder);
      modheight = remainder;

      printf("On last iteration!! modheight = %d\n", modheight);
    }

    if(iterations == 1) {
      startheight_plus = 0;
      modheight_plus = modheight;
    }
    else if(iter == 0) {
      startheight_plus = 0;
      modheight_plus = modheight + offset;
    } else if(iter == iterations - 1) {
      startheight_plus = iter*startheight - offset;
      modheight_plus = modheight + offset;
    }
    else {
      startheight_plus =  iter*startheight - offset;
      modheight_plus = modheight + 2 * offset;
      printf("middle loop\n");
    }
    
    image.modifyImage();
    
    PixelPacket* cpu_packet = image.getPixels(0, startheight_plus, width, modheight_plus);
    printf("start height: %d, end height: %d\n", startheight_plus, modheight_plus);
    pixelRGB* cpu_pixels;
    cpu_pixels = (pixelRGB*) malloc(sizeof(pixelRGB) * width * modheight_plus);
    printf("Got pixels?\n");
   
    for (int i = 0; i < width; i++) {
      for(int j = 0; j < modheight_plus; j++) {
        Color color = cpu_packet[j * width + i];
        cpu_pixels[j* width + i].r = color.redQuantum();// / RANGE;
        cpu_pixels[j* width + i].g = color.greenQuantum();// / RANGE;
        cpu_pixels[j* width + i].b = color.blueQuantum();// / RANGE;
      }
    }
   

 
    pixelRGB* gpu_pixels;
    if(cudaMalloc(&gpu_pixels, sizeof(pixelRGB) * width * modheight_plus) != cudaSuccess) {
      fprintf(stderr, "Failed to create image for the gpu\n");
      exit(2);
    }
  
    //Copy contents from cpu to gpu
    if(cudaMemcpy(gpu_pixels, cpu_pixels, sizeof(pixelRGB) *  width * modheight_plus, cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "Failed to copy image from CPU to the GPU\n");
    }

    printf("Gottem\n");
   
    pixelRGB* result_pixels;
    if(cudaMalloc(&result_pixels, sizeof(pixelRGB) * width * modheight_plus) != cudaSuccess) {
      fprintf(stderr, "Failed to create image for the gpu\n");
      exit(2);
    }
  
    //Copy contents from cpu to gpu
    if(cudaMemcpy(result_pixels, cpu_pixels, sizeof(pixelRGB) *  width * modheight_plus, cudaMemcpyHostToDevice) != cudaSuccess) {
      fprintf(stderr, "Failed to copy image from CPU to the GPU\n");
    }


    int blocks = (width * modheight_plus + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matrix_filter_image<<<blocks, BLOCK_SIZE>>>(gpu_pixels, result_pixels, width, modheight, gpu_filter, fWidth, fHeight, factor, bias);
    
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
      printf("\n%s\n", cudaGetErrorString(err));
      fprintf(stderr, "\nFailed to synchronize correctly\n");
    }

    if(cudaMemcpy(cpu_pixels, result_pixels, sizeof(pixelRGB) * width * modheight_plus, cudaMemcpyDeviceToHost) != cudaSuccess) {
      fprintf(stderr, "Failed to copy gpu pixels to host\n");
    }

    printf("they've returned\n");

    int row_start = 1;
    if(iter == 0) {
      row_start = 0;
    }
    
    for (int i = row_start; i < modheight; i++) {
      for(int j = 0; j < width; j++) {
        int index = (startheight_plus * width) + (i*width) + j;
        master_pixels[index] = cpu_pixels[i* width + j];
      }
    }
    

    //image.syncPixels();
    free(cpu_pixels);
    cudaFree(gpu_pixels);
    cudaFree(result_pixels);
  }

  PixelPacket* all_packets = image.getPixels(0, 0, width, height);
  for (int i = 0; i < width; i++) {
    for(int j = 0; j < height; j++) {
      pixelRGB temp = master_pixels[j* width + i];
      all_packets[j * width + i] = Color(temp.r, temp.g, temp.b);
    }
  }
  
  image.syncPixels();
  image.write("filtered_" + filename);

  clock_t diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;

  FILE* timing = fopen("timing.csv", "a");
  fprintf(timing, "%d,%d\n", width*height, msec);
  fclose(timing);
  
  // free(cpu_pixels);
  return 0;
}
