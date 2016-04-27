CFLAGS := `/home/curtsinger/bin/Magick-config --cflags` -g -O2
LDFLAGS := `/home/curtsinger/bin/Magick-config --ldflags`

SRCS := $(wildcard %.c) $(wildcard %.cc)
HEADERS := $(wildcard %.h) $(wildcard %.hh)

image_transform: $(SRCS) $(HEADERS)
	clang $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)
