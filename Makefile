NVCC := nvcc -arch sm_20
CFLAGS := -I/usr/include/x86_64-linux-gnu//ImageMagick-6 -I/usr/include/ImageMagick-6 -g -O2 -DMAGICKCORE_QUANTUM_DEPTH=16 -DMAGICKCORE_HDRI_ENABLE=0
LDFLAGS := `/home/curtsinger/bin/Magick++-config --ldflags`

SRCS := $(wildcard *.c) $(wildcard *.cc) $(wildcard *.cu)
HEADERS := $(wildcard *.h) $(wildcard *.hh)

image_transform: image_transform.cu
	$(NVCC) $(CFLAGS) -o $@ image_transform.cu $(LDFLAGS)
