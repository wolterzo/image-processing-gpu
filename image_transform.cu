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
  Pixels view(image);
   *(view.get(108,94,1,1)) = Color("red");

   int width = image.columns();
   int height = image.rows();
   int range = 256;
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
  
  return 0;
}
