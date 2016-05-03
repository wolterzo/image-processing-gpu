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


int main (int argc, char** argv) {
  InitializeMagick(*argv);
  
  printf("In main\n");
  
  string filename ("bowtie.jpg");
  Image image(filename);

  int width = image.columns();
  int height = image.rows();
  printf("width: %d, height: %d\n", width, height);
  image.modifyImage();
  PixelPacket* cpu_packet = image.getPixels(0, 0, width, height);
  pixelRGB cpu_pixels[width*height];
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
  image_to_grayscale<<<blocks, BLOCK_SIZE>>>(gpu_pixels);

  cudaError_t err = cudaDeviceSynchronize();
  if(err != cudaSuccess) {
    printf("\n%s\n", cudaGetErrorString(err));
    fprintf(stderr, "\nFailed to synchronize correctly\n");
  }

  if(cudaMemcpy(cpu_pixels, gpu_pixels, sizeof(pixelRGB) * width * height, cudaMemcpyDeviceToHost) != cudaSuccess) {
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

  image.write("grayscale_" + filename);
   
  return 0;
}
