
## Compile options for mac/linux
## mac uses darwin ports copy of gcc with OpenMP support
## linux uses gcc as cc, so just comment out the CC definition
CC=/opt/local/bin/gcc-mp-4.8
CFLAGS=-std=c99 -fopenmp -O3 -fPIC
LD=$(CC)
LDFLAGS=-shared
LIBS=-lm -lgomp

# Use C++ rather than C compiler, and no OpenMP
# Switch between USE_FLOAT2 and USE_COMPLEX to debug differences in how
# arithmetic is evaluated.  On mac, NO_OPENMP suppresses the OpenMP runtime.
#CC=c++
##CFLAGS=-O3 -DUSE_FLOAT2 -DNO_OPENMP
#CFLAGS=-O3
#LD=$(CC)
#LDFLAGS=-shared
#LIBS=-lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all: libga3.so

libga3.so: ba.o magnetic_refl.o
	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS)

ba.o: ba.c cl_fake.h
magnetic_refl.o: magnetic_refl.c cl_fake.h

clean:
	rm *.o *.so *~
