
## Compile options for mac/linux
## mac uses darwin ports copy of gcc with OpenMP support
## linux uses gcc as cc, so just comment out the CC definition
CC=/opt/local/bin/gcc-mp-4.8
CFLAGS=-std=c99 -fopenmp -O3 -fPIC
LD=$(CC)
LDFLAGS=-shared
LIBS=-lm -lgomp

## define EMULATE_COMPLEX to test the cl_complex.h library instead of using
## the c99/c++ complex number type.
#DEFINES=-DEMULATE_COMPLEX


#CC=/opt/local/bin/g++-mp-4.8
#CFLAGS=-O3 -fPIC
#LD=$(CC)
#LDFLAGS=-shared
#LIBS=-lm -lgomp

%.o: %.c
	$(CC) $(CFLAGS) $(DEFINES) -c $< -o $@

all: libga3.so oldmag.so

libga3.so: ba.o magnetic_refl.o
	$(LD) $(LDFLAGS) $^ -o $@ $(LIBS)

ba.o: ba.c cl_env.h cl_complex.h
magnetic_refl.o: magnetic_refl.c cl_env.h cl_complex.h

oldmag.so: magnetic.cc
	c++ -shared -lm -O3 magnetic.cc -o oldmag.so

clean:
	rm *.o *.so *~
