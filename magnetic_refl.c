// This program is public domain.
#ifndef __OPENCL_VERSION__
#include <stdio.h>
#include <omp.h>
#include "cl_env.h"
#endif

#define EPSILON 1.0e-10
#define PI4 12.566370614359172

#define DEBUG 0
#if DEBUG
int print_kzi = 0;
#define printrow(A1,A2,A3,A4) printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",A1,A2,A3,A4);
#endif

// Define some accessor macros to retrieve the parts of L
cdouble rhoN(global const cdouble *L, int j);
cdouble beta(global const cdouble *L, int j);
cdouble gamm(global const cdouble *L, int j);
double rhoB(global const cdouble *L, int j);
double dz(global const cdouble *L, int j);
cdouble rhoN(global const cdouble *L, int j) { return L[j*4]; }
cdouble beta(global const cdouble *L, int j)   { return L[j*4+1]; }
cdouble gamm(global const cdouble *L, int j)   { return L[j*4+2]; }
double rhoB(global const cdouble *L, int j) { return creal(L[j*4+3]); }
double dz(global const cdouble *L, int j)   { return cimag(L[j*4+3]); }

inline cdouble cdot4(
    const cdouble a1, const cdouble a2, const cdouble a3, const cdouble a4,
    const cdouble b1, const cdouble b2, const cdouble b3, const cdouble b4)
{
    return cadd(cadd(cadd(cmul(a1,b1),cmul(a2,b2)),cmul(a3,b3)),cmul(a4,b4));
}

void
calculateB(
    const int Lstride,
    const int Bstride,
    const int Nlayers,
    const double kzi,
    global const cdouble *L,
    global cdouble *B,
    const double spin);
void
calculateB(
    const int Lstride,
    const int Bstride,
    const int Nlayers,
    const double kzi,
    global const cdouble *L,
    global cdouble *B,
    const double spin)
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
     if spin = +1, B will be valid for r+- and r++
        spin = -1, B will be valid for r-+ and r--
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
    cdouble A11, A12, A13, A14, A21, A22, A23, A24,
            A31, A32, A33, A34, A41, A42, A43, A44;
    cdouble B11, B12, B13, B14, B21, B22, B23, B24,
            B31, B32, B33, B34, B41, B42, B43, B44;
    cdouble C1, C2, C3, C4;
    cdouble S1p, S3p, S1n, S3n;
    cdouble Bp, Gp, Bn, Gn;

    //printf("L: %p\n",L);
    //printf("L1: [%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",L[1].rhoN, L[1].rhoB, L[1].beta, L[1].gamma);

    // fronting medium is removed from the effective kzi
    const int surface = (kzi >=  0.0 ? 0 : Nlayers-1);
    // no absorption in incident medium
    //const cdouble E0 = radd(kzi*kzi + spin*PI4*rhoB(L,surface),
    //                        rmul(PI4, rhoN(L,surface)));
    const double E0 = kzi*kzi
                      + PI4*(creal(rhoN(L,surface)) + spin*rhoB(L,surface));

#if 0
    if (print_kzi) {
        printf("====== sqrt =======\n");
        int N = 13;
        for (int i=0; i < N; i++) {
            cdouble C = rmul(0.2, cexp(cplx(0.0,-M_PI + 2.0*M_PI/(double)N*i)));
            cdouble rC = csqrt(C);
            printf("%10.6g%+10.6gj %10.6g%+10.6gj\n", C, rC);
        }
        {
            cdouble C = csub(rmul(PI4,radd(-rhoB(L,0), rhoN(L,0))), E0);
            cdouble rC = csqrt(C);
            printf("%10.6g%+10.6gj\n",rhoN(L,0));
            printf("%10.6g%+10.6gj %10.6g%+10.6gj\n", C, rC);
        }
    }
#endif


    S1p = cneg(csqrt(subr(rmul(PI4,radd(+rhoB(L,0), rhoN(L,0))), E0)));
    S3p = cneg(csqrt(subr(rmul(PI4,radd(-rhoB(L,0), rhoN(L,0))), E0)));
    S1n = cneg(csqrt(subr(rmul(PI4,radd(+rhoB(L,1), rhoN(L,1))), E0)));
    S3n = cneg(csqrt(subr(rmul(PI4,radd(-rhoB(L,1), rhoN(L,1))), E0)));

    Bn = beta(L,1);
    Gn = gamm(L,1);

    // First B matrix is slightly different
    {
        const cdouble FS1S1 = cdiv(S1p,S1n);
        const cdouble FS3S1 = cdiv(S3p,S1n);
        const cdouble FS1S3 = cdiv(S1p,S3n);
        const cdouble FS3S3 = cdiv(S3p,S3n);

        const cdouble delta = rdiv(0.5, rsub(1.0, cmul(Bn,Gn)));

        B11 = B22 = cmul(delta, radd(1.0, FS1S1));
        B12 = B21 = cmul(delta, rsub(1.0, FS1S1));
        B13 = B24 = cmul(cmul(delta, cneg(Gn)), radd(1.0, FS3S1));
        B14 = B23 = cmul(cmul(delta, cneg(Gn)), rsub(1.0, FS3S1));

        B31 = B42 = cmul(cmul(delta, cneg(Bn)), radd(1.0, FS1S3));
        B32 = B41 = cmul(cmul(delta, cneg(Bn)), rsub(1.0, FS1S3));
        B33 = B44 = cmul(delta, radd(1.0, FS3S3));
        B34 = B43 = cmul(delta, rsub(1.0, FS3S3));
#if DEBUG
        if (print_kzi) {
            printf("===== fronting =====\n");
            printf("[ %10.6g%+10.6gj  %g %g]\n", rhoN(L,0), rhoB(L,0), kzi);
            printf("[ %10.6g%+10.6gj  %g %g]\n", rhoN(L,1), rhoB(L,1), spin);
            printrow(cplx(E0,0.), delta, Bn, Gn);
            printrow(subr(rmul(PI4,radd(+rhoB(L,0), rhoN(L,0))), E0),
                     subr(rmul(PI4,radd(-rhoB(L,0), rhoN(L,0))), E0),
                     subr(rmul(PI4,radd(+rhoB(L,1), rhoN(L,1))), E0),
                     subr(rmul(PI4,radd(-rhoB(L,1), rhoN(L,1))), E0));
            printrow(S1p, S3p, S1n, S3n);
            printrow(FS1S1, FS3S1, FS1S3, FS3S3);
        }
#endif
    }

    double z = dz(L,1);
    for (int i=2 ;; i++) {
#if DEBUG
        if (print_kzi) {
            printf("===== B ===== kzi=%g layer=%d\n", kzi, i-1);
            printrow(B11,B12,B13,B14);
            printrow(B21,B22,B23,B24);
            printrow(B31,B32,B33,B34);
            printrow(B41,B42,B43,B44);
        }
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
        Bp = Bn;
        Gp = Gn;
        S1p = S1n;
        S3p = S3n;
        Bn = beta(L,i);
        Gn = gamm(L,i);

        // Build (invS invX X S) matrix to transport C
        S1n = cneg(csqrt(subr(rmul(PI4,radd(+rhoB(L,i), rhoN(L,i))), E0)));
        S3n = cneg(csqrt(subr(rmul(PI4,radd(-rhoB(L,i), rhoN(L,i))), E0)));

        // FSiSj = Si/Sj
        const cdouble FS1S1 = cdiv(S1p, S1n);
        const cdouble FS3S1 = cdiv(S3p, S1n);
        const cdouble FS1S3 = cdiv(S1p, S3n);
        const cdouble FS3S3 = cdiv(S3p, S3n);

        const cdouble delta = rdiv(0.5, rsub(1.0, cmul(Bn,Gn)));

        // DBB = delta (Bn - Bp)
        const cdouble DBB = cmul(csub(Bp, Bn), delta);
        const cdouble DBG = cmul(rsub(1.0, cmul(Bp, Gn)), delta);
        const cdouble DGB = cmul(rsub(1.0, cmul(Gp, Bn)), delta);
        const cdouble DGG = cmul(csub(Gp, Gn), delta);

        // EPS1p = exp(+S1[prev])*z)
        // EMS3n = exp(-S3[next])*z)
        const cdouble EPS1p = cexp(rmul(z, S1p));
        const cdouble EMS1p = rdiv(1.0, EPS1p);
        const cdouble EPS1n = cexp(rmul(z, S1n));
        const cdouble EMS1n = rdiv(1.0, EPS1n);
        const cdouble EPS3p = cexp(rmul(z, S3p));
        const cdouble EMS3p = rdiv(1.0, EPS3p);
        const cdouble EPS3n = cexp(rmul(z, S3n));
        const cdouble EMS3n = rdiv(1.0, EPS3n);

#if DEBUG
        if (print_kzi) {
            printf("===== A parts =====\n");
            printrow(Bn,Gn,S1n,S3n);
            printrow(DBB,DBG,DGB,DGG);
            printrow(FS1S1,FS3S1,FS1S3,FS3S3);
            printrow(EPS1p,EMS1p,EPS1n,EMS1n);
        }
#endif

        A11 = A22 = cmul(DBG, radd(1.0, FS1S1));
        A12 = A21 = cmul(DBG, rsub(1.0, FS1S1));
        A13 = A24 = cmul(DGG, radd(1.0, FS3S1));
        A14 = A23 = cmul(DGG, rsub(1.0, FS3S1));
        A31 = A42 = cmul(DBB, radd(1.0, FS1S3));
        A32 = A41 = cmul(DBB, rsub(1.0, FS1S3));
        A33 = A44 = cmul(DGB, radd(1.0, FS3S3));
        A34 = A43 = cmul(DGB, rsub(1.0, FS3S3));

        A11 = cmul(A11, cmul(EPS1p, EMS1n));
        A22 = cmul(A22, cmul(EMS1p, EPS1n));
        A12 = cmul(A12, cmul(EMS1p, EMS1n));
        A21 = cmul(A21, cmul(EPS1p, EPS1n));
        A13 = cmul(A13, cmul(EPS3p, EMS1n));
        A24 = cmul(A24, cmul(EMS3p, EPS1n));
        A14 = cmul(A14, cmul(EMS3p, EMS1n));
        A23 = cmul(A23, cmul(EPS3p, EPS1n));

        A31 = cmul(A31, cmul(EPS1p, EMS3n));
        A42 = cmul(A42, cmul(EMS1p, EPS3n));
        A32 = cmul(A32, cmul(EMS1p, EMS3n));
        A41 = cmul(A41, cmul(EPS1p, EPS3n));
        A33 = cmul(A33, cmul(EPS3p, EMS3n));
        A44 = cmul(A44, cmul(EMS3p, EPS3n));
        A34 = cmul(A34, cmul(EMS3p, EMS3n));
        A43 = cmul(A43, cmul(EPS3p, EPS3n));

#if DEBUG
        if (print_kzi) {
            printf("===== A =====\n");
            printrow(A11,A12,A13,A14);
            printrow(A21,A22,A23,A24);
            printrow(A31,A32,A33,A34);
            printrow(A41,A42,A43,A44);
        }
#endif

        // B = A * B
        C1 = cdot4(A11,A12,A13,A14,B11,B21,B31,B41);
        C2 = cdot4(A21,A22,A23,A24,B11,B21,B31,B41);
        C3 = cdot4(A31,A32,A33,A34,B11,B21,B31,B41);
        C4 = cdot4(A41,A42,A43,A44,B11,B21,B31,B41);
        B11 = C1;
        B21 = C2;
        B31 = C3;
        B41 = C4;

        C1 = cdot4(A11,A12,A13,A14,B12,B22,B32,B42);
        C2 = cdot4(A21,A22,A23,A24,B12,B22,B32,B42);
        C3 = cdot4(A31,A32,A33,A34,B12,B22,B32,B42);
        C4 = cdot4(A41,A42,A43,A44,B12,B22,B32,B42);
        B12 = C1;
        B22 = C2;
        B32 = C3;
        B42 = C4;

        C1 = cdot4(A11,A12,A13,A14,B13,B23,B33,B43);
        C2 = cdot4(A21,A22,A23,A24,B13,B23,B33,B43);
        C3 = cdot4(A31,A32,A33,A34,B13,B23,B33,B43);
        C4 = cdot4(A41,A42,A43,A44,B13,B23,B33,B43);
        B13 = C1;
        B23 = C2;
        B33 = C3;
        B43 = C4;

        C1 = cdot4(A11,A12,A13,A14,B14,B24,B34,B44);
        C2 = cdot4(A21,A22,A23,A24,B14,B24,B34,B44);
        C3 = cdot4(A31,A32,A33,A34,B14,B24,B34,B44);
        C4 = cdot4(A41,A42,A43,A44,B14,B24,B34,B44);
        B14 = C1;
        B24 = C2;
        B34 = C3;
        B44 = C4;

        z +=  dz(L,i);
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
    const cdouble B22 = B[5*Bstride];
    const cdouble B24 = B[7*Bstride];
    const cdouble B42 = B[13*Bstride];
    const cdouble B44 = B[15*Bstride];
    const cdouble det = rdiv(1.0, csub(cmul(B44,B22),cmul(B24,B42)));
//printf("denom: %10.6g%+10.6gj\n",denom);
//printf("[%10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj]\n",B22,B24,B42,B44);
    if (kzi > 0) {
        const cdouble B21 = B[4*Bstride];
        const cdouble B23 = B[6*Bstride];
        const cdouble B41 = B[12*Bstride];
        const cdouble B43 = B[14*Bstride];
//printf("[%10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj]\n",B21,B23,B41,B43);
        R[0] = cmul(det, csub(cmul(B24,B41),cmul(B21,B44)));
        R[1] = cmul(det, csub(cmul(B21,B42),cmul(B41,B22)));
        R[2] = cmul(det, csub(cmul(B24,B43),cmul(B23,B44)));
        R[3] = cmul(det, csub(cmul(B23,B42),cmul(B43,B22)));
//printf("[%10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj]\n",R[0],R[1],R[2],R[3]);
//printf("[%10.6g,  %10.6g,  %10.6g,  %10.6g]\n",cabs(R[0]),cabs(R[1]),cabs(R[2]),cabs(R[3]));
    } else {
        const cdouble B12 = B[1*Bstride];
        const cdouble B14 = B[3*Bstride];
        const cdouble B32 = B[9*Bstride];
        const cdouble B34 = B[11*Bstride];
        R[0] = cmul(det, csub(cmul(B12,B44),cmul(B14,B42)));
        R[1] = cmul(det, csub(cmul(B32,B44),cmul(B34,B42)));
        R[2] = cmul(det, csub(cmul(B12,B24),cmul(B24,B22)));
        R[3] = cmul(det, csub(cmul(B32,B24),cmul(B24,B22)));
    }
}

kernel void
magneticR(
    double spin,
#ifdef USE_OPENCL
    const int kz_offset,
#endif
    const int Nkz,
    const int Nlayers,
    global const double *kz,
    global const cdouble *L,
    global cdouble *B,
    global cdouble *R)
{
    const cdouble zero = cplx(0.0,0.0);
    const cdouble minusone = cplx(-1.0,0.0);

    const int Bstride = 1;
    const int Lstride = 16*Bstride;
    const int Kstride = (Nlayers-1) * Lstride;

#ifdef USE_OPENCL
    //if (get_global_id(0) != 0) return;
    const int Boffset = Kstride * get_global_id(0);
    const int i = kz_offset + get_global_id(0);
    {
#else
    #pragma omp parallel for
    for (int i=0; i<Nkz; i++) {
        const int Boffset = Kstride * omp_get_thread_num();
#if DEBUG
        print_kzi = (i == 2);
#endif
#endif
        const double kzi = kz[i];
        if (fabs(kzi) > EPSILON) {
            calculateB(Lstride, Bstride,  Nlayers, kzi, L, B+Boffset, spin);
            calculateR_sam(Bstride, kzi, B+Boffset+Lstride*(Nlayers-2), R+i*4);
        } else {
            R[i*4] = R[i*4+3] = minusone;
            R[i*4+1] = R[i*4+2] = zero;
        }
    }
}


// $Id: magnetic_refl.c 2015-01-13 bbm $
