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

void
calculateB(
    const int Lstride,
    const int Bstride,
    const int Nlayers,
    const double kzi,
    global const Layer *L,
    global cdouble *B,
    const double plus_in);
void
calculateB(
    const int Lstride,
    const int Bstride,
    const int Nlayers,
    const double kzi,
    global const Layer *L,
    global cdouble *B,
    const double plus_in)
{
/*
    Calculation of reflectivity in magnetic sample in framework that
    also yields the wavefunction in each layer for DWBA.
    Arguments:

     kzi is the incoming momentum along the z direction
     dz is the thickness of each layer (top and bottom thickness ignored)
     rhoN is nuclear scattering length density array
     rhoB is magnetic scattering length density array
     bx, by, bz are components of unit vector along M for layer
     if plus_in, the B calculated will be valid for r+- and r++...
        plus_in = False, B will be valid for r-+ and r--
    ###################################################################
    #  all of dz, rhoN, rhoB, bx, by, bz should have length equal to  #
    #  the number of layers in the sample including fronting and      #
    #  substrate.                                                     #
    ###################################################################
     AGUIDE is the angle of the z-axis of the sample with respect to the
     z-axis (quantization) of the neutron in the lab frame.
     AGUIDE = 0 for perpendicular films, 270 for field along y
*/
    // sld is array([[sld, thickness, mu], [...], ...])
    // ordered from top (vacuum usually) to bottom (substrate)
    const cdouble onehalf = cplx(0.5,0.0);
    const cdouble one = cplx(1.0,0.0);


    cdouble A11, A12, A13, A14, A21, A22, A23, A24,
            A31, A32, A33, A34, A41, A42, A43, A44;
    cdouble B11, B12, B13, B14, B21, B22, B23, B24,
            B31, B32, B33, B34, B41, B42, B43, B44;
    cdouble D1, D2, D3, D4;
    cdouble U1p, U3p, U1n, U3n;
    cdouble S1p, S3p, S1n, S3n;

    //printf("L: %p\n",L);
    //printf("L1: [%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",L[1].rhoN, L[1].rhoB, L[1].u1, L[1].u3);

    // fronting medium removed from effective kzi
    const int surface = (kzi >=  0 ? 0 : Nlayers-1);
    const cdouble E0 = cplx(kzi*kzi,0.0) + PI4*(L[surface].rhoN + plus_in*L[surface].rhoB);

    U1n = L[1].u1;
    U3n = L[1].u3;
    S1p = -csqrt(PI4*(L[0].rhoN+L[0].rhoB)-E0 - cplx(0.0,EPSILON));
    S3p = -csqrt(PI4*(L[0].rhoN-L[0].rhoB)-E0 - cplx(0.0,EPSILON));
    S1n = -csqrt(PI4*(L[1].rhoN+L[1].rhoB)-E0 - cplx(0.0,EPSILON));
    S3n = -csqrt(PI4*(L[1].rhoN-L[1].rhoB)-E0 - cplx(0.0,EPSILON));

    // First B matrix is slightly different
    {
        const cdouble delta = cdiv(onehalf, U3n-U1n);
        const cdouble FS11 = cdiv(S1p,S1n);
        const cdouble FS31 = cdiv(S3p,S1n);
        const cdouble DU13 = cmul(delta, U3n);
        const cdouble DU33 = cmul(delta, -one);
        B11 = cmul(DU13, one + FS11);
        B12 = cmul(DU13, one - FS11);
        B13 = cmul(DU33, one + FS31);
        B14 = cmul(DU33, one - FS31);
        B21 = B12;
        B22 = B11;
        B23 = B14;
        B24 = B13;
        const cdouble FS13 = cdiv(S1p,S3n);
        const cdouble FS33 = cdiv(S3p,S3n);
        const cdouble DU11 = cmul(delta, -U1n);
        const cdouble DU31 = cmul(delta, one);
        B31 = cmul(DU11, one + FS13);
        B32 = cmul(DU11, one - FS13);
        B33 = cmul(DU31, one + FS33);
        B34 = cmul(DU31, one - FS33);
        B41 = B32;
        B42 = B31;
        B43 = B34;
        B44 = B33;
    }

    double z = 0;
    for (int i=2 ;; i++) {
#if 0
        // Show B
        printf("kzi=%g layer=%d B11=%g\n",kzi,i-1,B11);
        printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",B11,B12,B13,B14);
        printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",B21,B22,B23,B24);
        printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",B31,B32,B33,B34);
        printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",B41,B42,B43,B44);
#endif

        // Save B
        B[0*Bstride] = B11;
        B[1*Bstride] = B12;
        B[2*Bstride] = B13;
        B[3*Bstride] = B14;
        B[4*Bstride] = B21;
        B[5*Bstride] = B22;
        B[6*Bstride] = B23;
        B[7*Bstride] = B24;
        B[8*Bstride] = B31;
        B[9*Bstride] = B32;
        B[10*Bstride] = B33;
        B[11*Bstride] = B34;
        B[12*Bstride] = B41;
        B[13*Bstride] = B42;
        B[14*Bstride] = B43;
        B[15*Bstride] = B44;
        B += Lstride;
        if (i == Nlayers) { break; }


        // Move to the next layer
        z +=  L[i-1].dz;
        U1p = U1n; U3p = U3n;
        S1p = S1n; S3p = S3n;
        U1n = L[i].u1;
        U3n = L[i].u3;
        S1n = -csqrt(PI4*(L[i].rhoN+L[i].rhoB)-E0 -cplx(0.0,EPSILON));
        S3n = -csqrt(PI4*(L[i].rhoN-L[i].rhoB)-E0 -cplx(0.0,EPSILON));

        // Build (invS invX X S) matrix to transport C
        const cdouble delta = cdiv(onehalf, U3n-U1n);
        const cdouble FS11 = cdiv(S1p,S1n);
        const cdouble FS31 = cdiv(S3p,S1n);
        const cdouble DU13 = cmul(delta, U3n-U1p);
        const cdouble DU33 = cmul(delta, U3n-U1p);
        A11 = cmul(DU13, one + FS11);
        A12 = cmul(DU13, one - FS11);
        A13 = cmul(DU33, one + FS31);
        A14 = cmul(DU33, one - FS31);
        A21 = A12;
        A22 = A11;
        A23 = A14;
        A24 = A13;
        const cdouble FS13 = cdiv(S1p,S3n);
        const cdouble FS33 = cdiv(S3p,S3n);
        const cdouble DU11 = cmul(delta, U1p-U1n);
        const cdouble DU31 = cmul(delta, U3p-U1n);
        A31 = cmul(DU11, one + FS13);
        A32 = cmul(DU11, one - FS13);
        A33 = cmul(DU31, one + FS33);
        A34 = cmul(DU31, one - FS33);
        A41 = A32;
        A42 = A31;
        A43 = A34;
        A44 = A33;

        A11 = cmul(cexp(( S1p-S1n)*z),A11);
        A12 = cmul(cexp((-S1p-S1n)*z),A12);
        A13 = cmul(cexp(( S3p-S1n)*z),A13);
        A14 = cmul(cexp((-S3p-S1n)*z),A14);
        A21 = cmul(cexp(( S1p+S1n)*z),A21);
        A22 = cmul(cexp((-S1p+S1n)*z),A22);
        A23 = cmul(cexp(( S3p+S1n)*z),A23);
        A24 = cmul(cexp((-S3p+S1n)*z),A24);
        A31 = cmul(cexp(( S1p-S3n)*z),A31);
        A32 = cmul(cexp((-S1p-S3n)*z),A32);
        A33 = cmul(cexp(( S3p-S3n)*z),A33);
        A34 = cmul(cexp((-S3p-S3n)*z),A34);
        A41 = cmul(cexp(( S1p+S3n)*z),A41);
        A42 = cmul(cexp((-S1p+S3n)*z),A42);
        A43 = cmul(cexp(( S3p+S3n)*z),A43);
        A44 = cmul(cexp((-S3p+S3n)*z),A44);

        // B = A * B
        D1 = cmul(A11,B11) + cmul(A12,B21) + cmul(A13,B31) + cmul(A14,B41);
        D2 = cmul(A21,B11) + cmul(A22,B21) + cmul(A23,B31) + cmul(A24,B41);
        D3 = cmul(A31,B11) + cmul(A32,B21) + cmul(A33,B31) + cmul(A34,B41);
        D4 = cmul(A41,B11) + cmul(A42,B21) + cmul(A43,B31) + cmul(A44,B41);
        B11 = D1;
        B21 = D2;
        B31 = D3;
        B41 = D4;

        D1 = cmul(A11,B12) + cmul(A12,B22) + cmul(A13,B32) + cmul(A14,B42);
        D2 = cmul(A21,B12) + cmul(A22,B22) + cmul(A23,B32) + cmul(A24,B42);
        D3 = cmul(A31,B12) + cmul(A32,B22) + cmul(A33,B32) + cmul(A34,B42);
        D4 = cmul(A41,B12) + cmul(A42,B22) + cmul(A43,B32) + cmul(A44,B42);
        B12 = D1;
        B22 = D2;
        B32 = D3;
        B42 = D4;

        D1 = cmul(A11,B13) + cmul(A12,B23) + cmul(A13,B33) + cmul(A14,B43);
        D2 = cmul(A21,B13) + cmul(A22,B23) + cmul(A23,B33) + cmul(A24,B43);
        D3 = cmul(A31,B13) + cmul(A32,B23) + cmul(A33,B33) + cmul(A34,B43);
        D4 = cmul(A41,B13) + cmul(A42,B23) + cmul(A43,B33) + cmul(A44,B43);
        B13 = D1;
        B23 = D2;
        B33 = D3;
        B43 = D4;

        D1 = cmul(A11,B14) + cmul(A12,B24) + cmul(A13,B34) + cmul(A14,B44);
        D2 = cmul(A21,B14) + cmul(A22,B24) + cmul(A23,B34) + cmul(A24,B44);
        D3 = cmul(A31,B14) + cmul(A32,B24) + cmul(A33,B34) + cmul(A34,B44);
        D4 = cmul(A41,B14) + cmul(A42,B24) + cmul(A43,B34) + cmul(A44,B44);
        B14 = D1;
        B24 = D2;
        B34 = D3;
        B44 = D4;
    }
}

void
calculateR_sam(
    const int Bstride,
    const double kz,
    global const cdouble *B,
    global cdouble *R);
void
calculateR_sam(
    const int Bstride,
    const double kzi,
    global const cdouble *B,
    global cdouble *R)
{
    const cdouble one = cplx(1.0,0.0);
    const cdouble B22 = B[5*Bstride];
    const cdouble B24 = B[7*Bstride];
    const cdouble B42 = B[13*Bstride];
    const cdouble B44 = B[15*Bstride];
    const cdouble denom = cdiv(one, cmul(B44,B22) - cmul(B24,B42));
    //printf("[%10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj]\n",B22,B24,B42,B44);
    if (kzi > 0) {
        const cdouble B21 = B[4*Bstride];
        const cdouble B23 = B[6*Bstride];
        const cdouble B41 = B[12*Bstride];
        const cdouble B43 = B[14*Bstride];
        //printf("[%10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj]\n",B21,B23,B41,B43);
        R[0] = cmul(denom, cmul(B24,B41) - cmul(B21,B44));
        R[1] = cmul(denom, cmul(B21,B42) - cmul(B41,B22));
        R[2] = cmul(denom, cmul(B24,B43) - cmul(B23,B44));
        R[3] = cmul(denom, cmul(B23,B42) - cmul(B43,B22));
    } else {
        const cdouble B12 = B[1*Bstride];
        const cdouble B14 = B[3*Bstride];
        const cdouble B32 = B[9*Bstride];
        const cdouble B34 = B[11*Bstride];
        R[0] = cmul(denom, cmul(B12,B44) - cmul(B14,B42));
        R[1] = cmul(denom, cmul(B32,B44) - cmul(B34,B42));
        R[2] = cmul(denom, cmul(B12,B24) - cmul(B24,B22));
        R[3] = cmul(denom, cmul(B32,B24) - cmul(B24,B22));
    }
}

kernel void
magneticR(
    double plus_in,
#ifdef USE_OPENCL
    const int kz_offset,
#endif
    const int Nkz,
    const int Nlayers,
    global const double *kz,
    global const Layer *L,
    global cdouble *B,
    global cdouble *R)
{
    const cdouble zero = cplx(0.0,0.0);
    const cdouble minusone = cplx(-1.0,0.0);

#ifdef USE_OPENCL
    //printf("> offset:%d Nkz:%d Nlayers:%d kz[0]:%g\n",kz_offset,Nkz,Nlayers,kz[0]);
    const int Kstride = 1;
    const int Bstride = get_global_size(0)*Kstride;
    const int Lstride = 16*Bstride;
    //const int Bstride = 1;
    //const int Lstride = 16*Bstride;
    //const int Kstride = (Nlayers-1) * Lstride;

    const int Boffset = Kstride*get_global_id(0);
    const int i = kz_offset + get_global_id(0);
    //printf("> L: %p\n",L);
    if (i==0)
    printf("> L1: [%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",
           L[1].rhoN, L[1].u1, L[1].u3, L[1].rhoB, L[1].dz);
    {
#else
    const int Bstride = 1;
    const int Lstride = 16*Bstride;
    const int Kstride = (Nlayers-1) * Lstride;

    #pragma omp parallel for
    for (int i=0; i<Nkz; i++) {
        const int Boffset = Kstride * omp_get_thread_num();
#endif
        const double kzi = kz[i];
        if (fabs(kzi) > EPSILON) {
            calculateB(Lstride, Bstride,  Nlayers, kzi, L, B+Boffset, plus_in);
            calculateR_sam(Bstride, kzi, B+Boffset+Lstride*(Nlayers-2), R+i*4);
        } else {
            R[i*4] = R[i*4+3] = minusone;
            R[i*4+1] = R[i*4+2] = zero;
        }
    }
}


// $Id: magnetic_refl.c 2015-01-13 bbm $
