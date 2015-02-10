

all: libga3p.so

libga3p-single.so: 9_SimulateGeneralAmplitude_ordered_periodic.cpp
	c++ -c $< -O3 -o ga3p.o
	c++ -shared -lm -o $@ ga3p.o

libga3p.so: 9_SimulateGeneralAmplitude_ordered_periodic.cpp
	/opt/local/bin/g++-mp-4.8 -fopenmp -c $< -O3 -o ga3p.o
	/opt/local/bin/g++-mp-4.8 -shared -lm -o $@ ga3p.o -lgomp
