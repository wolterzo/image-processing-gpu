#include <stdio.h>
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

__global__ void matrix_filter_image(pixelRGB* input_pixels, pixelRGB* output_pixels, int* width, int* height) {
  int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int w = width[0];
  int h = height[0];
  
  int x = index % w;
  int y = index / w;

  
  // Looked at 'http://lodev.org/cgtutor/filtering.html' to figure out complexfilters
  int fHeight = 3;
  int fWidth = 3;
  //int mult = fWidth * fHeight;
  double filter[9] =
    {
      -1, -1,  0,
      -1,  0,  1,
       0,  1,  1
    };
  double factor = 1;
  double bias = 128;

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
  
  string filename ("barce.jpg");
  Image image(filename);

  int width = image.columns();
  int height = image.rows();
  // Rounds up number of iterations
  int iterations = (((width * height) + (65535 * BLOCK_SIZE))-1) / (65535 * BLOCK_SIZE);
  printf("number of iterations: %d\n", iterations);
  int modheight = height / iterations;
  int startheight = height / iterations;
  float a = (width * height) / (65535 * BLOCK_SIZE);
  printf("modheight is: %lf\n", a);//odheight);
 
for (int i = 0; i < iterations; i++){
  if (i == iterations-1){
  int rm = height - (modheight * i);
  modheight = modheight + rm;
  printf("On last iteration!! modheight = %d\n", modheight);
  }
  image.modifyImage();
  PixelPacket* cpu_packet = image.getPixels(0, i * startheight, width, modheight-1);
  printf("width: %d, height: %d\n", width, height);
  printf("start height: %d, end height: %d\n", i * startheight, modheight);
  pixelRGB* cpu_pixels;
  cpu_pixels = (pixelRGB*) malloc(sizeof(pixelRGB) * width * height);
  printf("Got pixels?\n");
   
  for (int i = 0; i < width; i++) {
    for(int j = 0; j < height; j++) {
      Color color = cpu_packet[j * width + i];
      cpu_pixels[j* width + i].r = color.redQuantum();// / RANGE;
      cpu_pixels[j* width + i].g = color.greenQuantum();// / RANGE;
      cpu_pixels[j* width + i].b = color.blueQuantum();// / RANGE;
    }
  }
   
   
  // Color color = cpu_packet[0];
  // cout << (color.redQuantum() / range) << endl;
 
  pixelRGB* gpu_pixels;
  if(cudaMalloc(&gpu_pixels, sizeof(pixelRGB) * width * height) != cudaSuccess) {
    fprintf(stderr, "Failed to create image for the gpu\n");
    exit(2);
  }
  
  //Copy contents from cpu to gpu
  if(cudaMemcpy(gpu_pixels, cpu_pixels, sizeof(pixelRGB) *  width * height, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy image from CPU to the GPU\n");
  }

  printf("Gottem\n");
   
  int blocks = (width * height + BLOCK_SIZE - 1) / BLOCK_SIZE;
  //image_to_grayscale<<<blocks, BLOCK_SIZE>>>(gpu_pixels);

  pixelRGB* result_pixels;
  if(cudaMalloc(&result_pixels, sizeof(pixelRGB) * width * height) != cudaSuccess) {
    fprintf(stderr, "Failed to create image for the gpu\n");
    exit(2);
  }
  
  //Copy contents from cpu to gpu
  if(cudaMemcpy(result_pixels, cpu_pixels, sizeof(pixelRGB) *  width * height, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy image from CPU to the GPU\n");
  }

  int* gpu_width;
  int* gpu_height;
  if(cudaMalloc(&gpu_width, sizeof(int)) != cudaSuccess) {
    fprintf(stderr, "Failed to create width for the gpu\n");
    exit(2);
  }
  
  //Copy contents from cpu to gpu
  if(cudaMemcpy(gpu_width, &width, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy width from CPU to the GPU\n");
  }
  if(cudaMalloc(&gpu_height, sizeof(int)) != cudaSuccess) {
    fprintf(stderr, "Failed to create height for the gpu\n");
    exit(2);
  }
  
  //Copy contents from cpu to gpu
  if(cudaMemcpy(gpu_height, &height, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy width from CPU to the GPU\n");
  }
  matrix_filter_image<<<blocks, BLOCK_SIZE>>>(gpu_pixels, result_pixels, gpu_width, gpu_height);
  cudaError_t err = cudaDeviceSynchronize();
  if(err != cudaSuccess) {
    printf("\n%s\n", cudaGetErrorString(err));
    fprintf(stderr, "\nFailed to synchronize correctly\n");
  }

  // image_to_grayscale<<<blocks, BLOCK_SIZE>>>(result_pixels);
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess) {
    printf("\n%s\n", cudaGetErrorString(err));
    fprintf(stderr, "\nFailed to synchronize correctly\n");
  }

  if(cudaMemcpy(cpu_pixels, result_pixels, sizeof(pixelRGB) * width * height, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy gpu pixels to host\n");
  }

  printf("they've returned\n");
   
  for (int i = 0; i < width; i++) {
    for(int j = 0; j < height; j++) {
      pixelRGB temp = cpu_pixels[j* width + i];
      cpu_packet[j * width + i] = Color(temp.r, temp.g, temp.b);
    }
  }

  image.syncPixels();
  free(cpu_pixels);
 }
  image.write("filtered_" + filename);
  // free(cpu_pixels);
  return 0;
}
