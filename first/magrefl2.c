// This program is public domain.
#ifndef __OPENCL_VERSION__
#include <stdio.h>
#include <omp.h>
#include "cl_fake.h"
#endif

#define EPSILON 1.0e-10
#define B2SLD 2.31604654e-6
#define PI4 12.566370614359172

typedef struct {
    cdouble rhoN;
    cdouble u1;
    cdouble u3;
    double rhoB;
    double dz;
} Layer;

kernel void
//magneticR( global const double *in ) { global const double K[][8] = (global const double [][8])in;
//magneticR( global const double *L ) {
//magneticR( global const double *L ) {  global const cdouble *M = (global const cdouble*)L;
//magneticR( int Nk, global const double2 *M, global double2 *N ) {
magneticR( int Nk, global const cdouble *M, global cdouble *N ) {
//magneticR( global cdouble *M ) {
//magneticR( global const double *L ) {  global const Layer *Q = (global const Layer*)L;
//magneticR( global const cdouble *M ) { global const Layer *Q = (global const Layer*)M;
//magneticR( global const Layer *Q ) {

    const int i = get_global_id(0);
    //printf("> L: %p\n",L);
    if (i==0)  {
        printf("=== start\n");
        for (int j=0; j<4; j++) {
            printf("> L1: [(%g,%g)  (%g,%g), (%g,%g), (%g,%g)]\n",
           //K[j][0], K[j][1], K[j][2], K[j][3], K[j][4], K[j][5], K[j][6], K[j][7]);
           //L[8*j+0], L[8*j+1], L[8*j+2], L[8*j+3], L[8*j+4], L[8*j+5], L[8*j+6], L[8*j+7]);
           M[4*j+0],M[4*j+1],M[4*j+2],M[4*j+3]);
           //Q[j].rhoN, Q[j].u1, Q[j].u3, Q[j].rhoB, Q[j].dz);
        }
        printf("=== stop\n");
    }
    if (i < Nk) { N[i] = M[i]; }
}
