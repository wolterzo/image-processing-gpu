#include <stdio.h>
#include <Magick++.h>
#include <iostream>
#include <string>
using namespace std; 
using namespace Magick;

#define BLOCK_SIZE 32
#define RANGE 256

__global__ void image_to_grayscale(PixelPacket* packet) {
  //Below equation found here: http://www.mathworks.com/matlabcentral/answers/99136-how-do-i-convert-my-rgb-image-to-grayscale-without-using-the-image-processing-toolbox?
  //intensity = 0.2989*red + 0.5870*green + 0.1140*blue
  int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  Color color = packet[index];
  int intensity = ((0.2989 * color.redQuantum()) / RANGE) +
    ((0.5870 * color.greenQuantum()) / RANGE) +
    ((0.1140 * color.blueQuantum()) / RANGE);
  packet[index] = Color(intensity, intensity, intensity);
  
}


int main (int argc, char** argv) {
  InitializeMagick(*argv);

  Image image;
  string filename ("cat.jpg");
  try {
    // Read a file into image object
    image.read(filename);
  }
  catch(Exception &error_) {
    cout << "Caught exception: " << error_.what() << endl;
    return 1;
  }
  Pixels view(image);
   *(view.get(108,94,1,1)) = Color("red");

   int width = image.columns();
   int height = image.rows();
   //int range = 256;
   PixelPacket* cpu_packet = image.getPixels(0, 0, width, height);
   
   // Color color = cpu_packet[0];
   // cout << (color.redQuantum() / range) << endl;
 
   PixelPacket* gpu_packet;
   if(cudaMalloc(&gpu_packet, sizeof(PixelPacket) * width * height) != cudaSuccess) {
    fprintf(stderr, "Failed to create image for the gpu\n");
    exit(2);
  }
  
  //Copy contents from cpu to gpu
   if(cudaMemcpy(gpu_packet, cpu_packet, sizeof(PixelPacket) *  width * height, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy image from CPU to the GPU\n");
      }

   int blocks = (width * height + BLOCK_SIZE - 1) / BLOCK_SIZE;
   image_to_grayscale<<<blocks, BLOCK_SIZE>>>(gpu_packet);

   cudaError_t err = cudaDeviceSynchronize();
   if(err != cudaSuccess) {
     printf("\n%s\n", cudaGetErrorString(err));
     fprintf(stderr, "\nFailed to synchronize correctly\n");
   }

   image.syncPixels();

   image.write("grayscale_" + filename);
   
  return 0;
}
