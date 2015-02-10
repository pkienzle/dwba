// This program is public domain.
#ifndef __OPENCL_VERSION__
#include <stdio.h>
#include <omp.h>
#include "cl_env.h"
#endif

#define EPSILON 1.0e-10
#define B2SLD 2.31604654e-6
#define PI4 12.566370614359172

#define PRINT_KZI 0.001
// L is the following struct, but intel driver was unable to handle float2
// objects sent as kernel parameters, or cast from kernel parameters, so
// we need to transfer L as a simple double array and compose the real and
// complex numbers by hand.
/*
typedef struct {
    cdouble rhoN;
    cdouble u1;
    cdouble u3;
    double rhoB;
    double dz;
} Layer;
*/
// Instead, define some accessor macros to retrieve the parts of L
cdouble rhoN(global const cdouble *L, const int j);
cdouble U1(global const cdouble *L, const int j);
cdouble U3(global const cdouble *L, const int j);
double rhoB(global const cdouble *L, const int j);
double dz(global const cdouble *L, const int j);
cdouble rhoN(global const cdouble *L, const int j) { return L[LAYER_STRIDE*j]; }
cdouble U1(global const cdouble *L, const int j)   { return L[LAYER_STRIDE*j+1]; }
cdouble U3(global const cdouble *L, const int j)   { return L[LAYER_STRIDE*j+2]; }
double rhoB(global const cdouble *L, const int j) { return creal(L[LAYER_STRIDE*j+3]); }
double dz(global const cdouble *L, const int j)   { return cimag(L[LAYER_STRIDE*j+3]); }


// Setters/getters for components of the work vector
cdouble S1(global const cdouble *work, const int j) { return work[WORK_STRIDE*j + S1_OFFSET]; }
cdouble S3(global const cdouble *work, const int j) { return work[WORK_STRIDE*j + S3_OFFSET]; }
cdouble setS1(global const cdouble *work, const int j, const double v) { return work[WORK_STRIDE*j + S1_OFFSET] = v; }
cdouble setS3(global const cdouble *work, const int j, const double v) { return work[WORK_STRIDE*j + S3_OFFSET] = v; }

//cdouble B(global const cdouble *work, const int j, const int k)  { return work[WORK_STRIDE*j + k]; }
//cdouble setB(global const cdouble *work, const int j, const int k, const double v) { work[WORK_STRIDE*j + k] = v; }

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
    global const double *L,
    global cdouble *B,
    const double spin);
void
calculateB(
    const int Lstride,
    const int Bstride,
    const int Nlayers,
    const double kzi,
    global const double *L,
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
     if spin, the B calculated will be valid for r+- and r++...
        spin = False, B will be valid for r-+ and r--
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
    cdouble D1, D2, D3, D4;
    cdouble S1p, S3p, S1n, S3n;
    cdouble Bp, Gp, Bn, Gn;

    //printf("L: %p\n",L);
    //printf("L1: [%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",L[1].rhoN, L[1].rhoB, L[1].u1, L[1].u3);

    // fronting medium removed from effective kzi
    const int surface = (kzi >=  0.0 ? 0 : Nlayers-1);
    // E0 = kzi**2 + spin*4*pi*rhoB[surface] + 4*pi*rhoN[surface];  // rhoN is complex
    // const cdouble E0 = radd(kzi*kzi + spin*PI4*rhoB(L,surface),
    //                        rmul(PI4, rhoN(L,surface)));
    const cdouble E0 = radd(kzi*kzi + spin*PI4*rhoB(L,surface),
                            rmul(PI4, rhoN(L,surface)));

    S1p = S1(B,0);
    S3p = S3(B,0);

    if (cabs(U1(L,1)) <= 1.0) {
        Bn = U1(L,1);
        Gn = rdiv(1,u3(L,1));
        S1n = S1(B,1);
        S3n = S3(B,1);
    } else {
        Bn = U3(L,1);
        Gn = rdiv(1,u1(L,1));
        S1n = S3(B,1);
        S3n = S3(B,0);
    }

    // First B matrix is slightly different
    {
        const cdouble delta = rdiv(0.5, rsub(1.0, cmul(Bn,Gn)));
        {
            const cdouble FS1S1 = cdiv(S1p,S1n);
            const cdouble T = delta;
            B22 = B11 = cmul(T, radd(1.0,FS1S1));
            B21 = B12 = cmul(T, rsub(1.0,FS1S1));
#if 0
if (fabs(kzi-PRINT_KZI)<1e-8) {
printf("[%g  %g  %g  %g]\n",kzi, creal(rhoN(L,0)),cimag(rhoN(L,0)),rhoB(L,0));
printf("[%g  %g  %g  %g]\n",spin, creal(rhoN(L,1)),cimag(rhoN(L,0)),rhoB(L,1));
printf("[%g%+gj %g%+gj %g%+gj %g%+gj]\n",creal(E0),cimag(E0),creal(Bn),cimag(Bn),creal(Gn),cimag(Gn),creal(FS1S1),cimag(FS1S1));
printf("[%g%+gj %g%+gj %g%+gj %g%+gj]\n",S1p,S1n,delta,B11);
}
#endif
        }
        {
            const cdouble FS3S1 = cdiv(S3p,S1n);
            const cdouble T = cmul(delta, cneg(Gn));
            B24 = B13 = cmul(T, radd(1.0,FS3S1));
            B23 = B14 = cmul(T, rsub(1.0,FS3S1));
        }
        {
            const cdouble FS1S3 = cdiv(S1p,S3n);
            const cdouble T = cmul(delta, cneg(Bn));
            B42 = B31 = cmul(T, radd(1.0,FS1S3));
            B41 = B32 = cmul(T, rsub(1.0,FS1S3));
        }
        {
            const cdouble FS3S3 = cdiv(S3p,S3n);
            const cdouble T = delta;
            B44 = B33 = cmul(T, radd(1.0,FS3S3));
            B43 = B34 = cmul(T, rsub(1.0,FS3S3));
        }
    }

    double z = 0;
    for (int i=2 ;; i++) {
// Show B
#if 0
if (fabs(kzi-PRINT_KZI)<1e-8) {
printf("===== B ===== kzi=%g layer=%d B11=%g\n",kzi,i-1,B11);
printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",B11,B12,B13,B14);
printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",B21,B22,B23,B24);
printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",B31,B32,B33,B34);
printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",B41,B42,B43,B44);
}
#endif

        // Save B
        B[0] = B11;
        B[1] = B12;
        B[2] = B13;
        B[3] = B14;
        B[4] = B21;
        B[5] = B22;
        B[6] = B23;
        B[7] = B24;
        B[8] = B31;
        B[9] = B32;
        B[10] = B33;
        B[11] = B34;
        B[12] = B41;
        B[13] = B42;
        B[14] = B43;
        B[15] = B44;
        B += WORK_STRIDE;
        if (i == Nlayers) { break; }


        // Move to the next layer
        Bp = Bn;
        Gp = Gn;
        S1p = S1n;
        S3p = S3n;
        if (cabs(u1(L,i)) <= 1.0) {
            Bn = u1(L,i);
            Gn = rdiv(1.0,u3(L,i));
            S1n = cneg(csqrt(csub(rmul(PI4,radd(+rhoB(L,i), rhoN(L,i))), E0)));
            S3n = cneg(csqrt(csub(rmul(PI4,radd(-rhoB(L,i), rhoN(L,i))), E0)));
        } else {
            Bn = u3(L,i);
            Gn = rdiv(1.0,u1(L,i));
            S1n = cneg(csqrt(csub(rmul(PI4,radd(-rhoB(L,i), rhoN(L,i))), E0)));
            S3n = cneg(csqrt(csub(rmul(PI4,radd(+rhoB(L,i), rhoN(L,i))), E0)));
        }
        z +=  dz(L,i-1);

        // Build (invS invX X S) matrix to transport C
        const cdouble delta = rdiv(0.5, rsub(1.0, cmul(Bn,Gn)));
        const cdouble DBB = cmul(csub(Bp, Bn), delta);
        const cdouble DBG = cmul(rsub(1.0, cmul(Bp, Gn)), delta);
        const cdouble DGB = cmul(rsub(1.0, cmul(Gp, Bn)), delta);
        const cdouble DGG = cmul(csub(Gp, Gn), delta);

        // exp(+S1[prev])*z)  => EPS1p
        // exp(-S3[next])*z)  => EMS3n
        const cdouble EPS1p = cexp(rmul(z, S1p));
        const cdouble EMS1p = rdiv(1.0, EPS1p);
        const cdouble EPS1n = cexp(rmul(z, S1n));
        const cdouble EMS1n = rdiv(1.0, EPS1n);
        const cdouble EPS3p = cexp(rmul(z, S3p));
        const cdouble EMS3p = rdiv(1.0, EPS3p);
        const cdouble EPS3n = cexp(rmul(z, S3n));
        const cdouble EMS3n = rdiv(1.0, EPS3n);

        const cdouble FS1S1 = cdiv(S1p, S1n);
        const cdouble FS3S1 = cdiv(S3p, S1n);
        const cdouble FS1S3 = cdiv(S1p, S3n);
        const cdouble FS3S3 = cdiv(S3p, S3n);

#if 0
if (fabs(kzi-PRINT_KZI)<1e-8) {
printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",DBG,FS1S1,Bn,Gn);
printf("[%10.6g %10.6g%+10.6gj\n",z,rmul(z,S1p));
printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",EPS1p,EMS1p,EPS1n,EMS1n);
}
#endif
        { const cdouble T = cmul(DBG, radd(1.0, FS1S1));
        A22 = cmul(T, cmul(EMS1p, EPS1n));
        A11 = cmul(T, cmul(EPS1p, EMS1n)); }
        { const cdouble T = cmul(DBG, rsub(1.0, FS1S1));
        A21 = cmul(T, cmul(EPS1p, EPS1n));
        A12 = cmul(T, cmul(EMS1p, EMS1n)); }
        { const cdouble T = cmul(DGG, radd(1.0, FS3S1));
        A24 = cmul(T, cmul(EMS3p, EPS1n));
        A13 = cmul(T, cmul(EPS3p, EMS1n)); }
        { const cdouble T = cmul(DGG, rsub(1.0, FS3S1));
        A14 = cmul(T, cmul(EMS3p, EMS1n));
        A23 = cmul(T, cmul(EPS3p, EPS1n)); }

        { const cdouble T = cmul(DBB, radd(1.0, FS1S3));
        A31 = cmul(T, cmul(EPS1p, EMS3n));
        A42 = cmul(T, cmul(EMS1p, EPS3n)); }
        { const cdouble T = cmul(DBB, rsub(1.0, FS1S3));
        A32 = cmul(T, cmul(EMS1p, EMS3n));
        A41 = cmul(T, cmul(EPS1p, EPS3n)); }
        { const cdouble T = cmul(DGB, radd(1.0, FS3S3));
        A33 = cmul(T, cmul(EPS3p, EMS3n));
        A44 = cmul(T, cmul(EMS3p, EPS3n)); }
        { const cdouble T = cmul(DGB, rsub(1.0, FS3S3));
        A34 = cmul(T, cmul(EMS3p, EMS3n));
        A43 = cmul(T, cmul(EPS3p, EPS3n)); }

#if 0
if (fabs(kzi-PRINT_KZI)<1e-8) {
printf("===== A =====\n");
printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",A11,A12,A13,A14);
printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",A21,A22,A23,A24);
printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",A31,A32,A33,A34);
printf("[%10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj %10.6g%+10.6gj]\n",A41,A42,A43,A44);
}
#endif

        // B = A * B
        D1 = cdot4(A11,A12,A13,A14,B11,B21,B31,B41);
        D2 = cdot4(A21,A22,A23,A24,B11,B21,B31,B41);
        D3 = cdot4(A31,A32,A33,A34,B11,B21,B31,B41);
        D4 = cdot4(A41,A42,A43,A44,B11,B21,B31,B41);
        B11 = D1;
        B21 = D2;
        B31 = D3;
        B41 = D4;

        D1 = cdot4(A11,A12,A13,A14,B12,B22,B32,B42);
        D2 = cdot4(A21,A22,A23,A24,B12,B22,B32,B42);
        D3 = cdot4(A31,A32,A33,A34,B12,B22,B32,B42);
        D4 = cdot4(A41,A42,A43,A44,B12,B22,B32,B42);
        B12 = D1;
        B22 = D2;
        B32 = D3;
        B42 = D4;

        D1 = cdot4(A11,A12,A13,A14,B13,B23,B33,B43);
        D2 = cdot4(A21,A22,A23,A24,B13,B23,B33,B43);
        D3 = cdot4(A31,A32,A33,A34,B13,B23,B33,B43);
        D4 = cdot4(A41,A42,A43,A44,B13,B23,B33,B43);
        B13 = D1;
        B23 = D2;
        B33 = D3;
        B43 = D4;

        D1 = cdot4(A11,A12,A13,A14,B14,B24,B34,B44);
        D2 = cdot4(A21,A22,A23,A24,B14,B24,B34,B44);
        D3 = cdot4(A31,A32,A33,A34,B14,B24,B34,B44);
        D4 = cdot4(A41,A42,A43,A44,B14,B24,B34,B44);
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
    const cdouble B22 = B[5*Bstride];
    const cdouble B24 = B[7*Bstride];
    const cdouble B42 = B[13*Bstride];
    const cdouble B44 = B[15*Bstride];
    const cdouble denom = rdiv(1.0, csub(cmul(B44,B22),cmul(B24,B42)));
//printf("denom: %10.6g%+10.6gj\n",denom);
//printf("[%10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj]\n",B22,B24,B42,B44);
    if (kzi > 0) {
        const cdouble B21 = B[4*Bstride];
        const cdouble B23 = B[6*Bstride];
        const cdouble B41 = B[12*Bstride];
        const cdouble B43 = B[14*Bstride];
//printf("[%10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj]\n",B21,B23,B41,B43);
        R[0] = cmul(denom, csub(cmul(B24,B41),cmul(B21,B44)));
        R[1] = cmul(denom, csub(cmul(B21,B42),cmul(B41,B22)));
        R[2] = cmul(denom, csub(cmul(B24,B43),cmul(B23,B44)));
        R[3] = cmul(denom, csub(cmul(B23,B42),cmul(B43,B22)));
//printf("[%10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj,  %10.6g%+10.6gj]\n",R[0],R[1],R[2],R[3]);
//printf("[%10.6g,  %10.6g,  %10.6g,  %10.6g]\n",cabs(R[0]),cabs(R[1]),cabs(R[2]),cabs(R[3]));
    } else {
        const cdouble B12 = B[1*Bstride];
        const cdouble B14 = B[3*Bstride];
        const cdouble B32 = B[9*Bstride];
        const cdouble B34 = B[11*Bstride];
        R[0] = cmul(denom, csub(cmul(B12,B44),cmul(B14,B42)));
        R[1] = cmul(denom, csub(cmul(B32,B44),cmul(B34,B42)));
        R[2] = cmul(denom, csub(cmul(B12,B24),cmul(B24,B22)));
        R[3] = cmul(denom, csub(cmul(B32,B24),cmul(B24,B22)));
    }
}

/* Compute S1/S3 at kzi,spin for all layers.
*/
void calculateS(global const cdouble *L,
                const int Nlayers,
                const double kzi,
                const double spin,
                global cdouble *work);
void calculateS(global const cdouble *L,
                const int Nlayers,
                const double kzi,
                const double spin,
                global cdouble *work)
{
    // fronting medium removed from effective kzi
    const int surface = (kzi >=  0.0 ? 0 : Nlayers-1);
    // E0 = kzi**2 + spin*4*pi*rhoB[surface] + 4*pi*rhoN[surface];  // rhoN is complex
    // const cdouble E0 = radd(kzi*kzi + spin*PI4*rhoB(L,surface),
    //                        rmul(PI4, rhoN(L,surface)));
    const cdouble E0 = radd(kzi*kzi + spin*PI4*rhoB(L,surface),
                            rmul(PI4, rhoN(L,surface)));

    for (int k=0; k < Nlayers; k++) {
        setS1(work, k, cneg(csqrt(csub(rmul(PI4,radd(+rhoB(L,k), rhoN(L,k))), E0))));
        setS3(work, k, cneg(csqrt(csub(rmul(PI4,radd(-rhoB(L,k), rhoN(L,k))), E0))));
    }
}

kernel void
dwba(
#ifdef USE_OPENCL
    const int kz_offset,
#endif
    const int Nkz,
    const int Nlayers,
    global const double *ki,
    global const double *kf,
    global const double *L,
    global cdouble *ba,
    global cdouble *R,
    global cdouble *work,
    )
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
