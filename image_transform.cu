#include <stdio.h>
#include <Magick++.h>
#include <iostream> 
using namespace std; 
using namespace Magick;

int main (int argc, char** argv) {
  InitializeMagick(*argv);
  
  Image image;
  try {
    // Read a file into image object
    image.read("cat.jpg");
  }
  catch(Exception &error_) {
    cout << "Caught exception: " << error_.what() << endl;
    return 1;
  }
  PixelPacket* cpu_packet = image.getPixels(0, 0, image.columns, image.rows);
 
  PixelPacket* gpu_packet;
  if(cudaMalloc(&gpu_packet, sizeof(PixelPacket) * image.columns * image.rows) != cudaSuccess) {
    fprintf(stderr, "Failed to create image for the gpu\n");
    exit(2);
  }
  
  //Copy contents from cpu to gpu
  if(cudaMemcpy(gpu_packet, cpu_packet, sizeof(PixelPacket) * image.columns * image.rows , cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy image from CPU to the GPU\n");
      }
  return 0;
}
