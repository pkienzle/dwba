// ===== Open CL / C99 / C++  common header ======
/*
Provides a common environment for opencl/c++/c99 so that
*/

#ifdef __OPENCL_VERSION__
# define USE_OPENCL
#endif

// If opencl is not available, then we are compiling a C function
#ifndef USE_OPENCL
   #define SINCOS(angle,svar,cvar) do {svar=sin(angle);cvar=cos(angle);} while (0)

   // Support for complex numbers, using c++ and/or c99
   #ifdef EMULATE_COMPLEX
       #ifdef __cplusplus
           #include <cmath>
       #else
           #include <math.h>
           #define cabs fake_cabs
           #define creal fake_creal
           #define cimag fake_cimag
           #define carg fake_carg
           #define cexp fake_cexp
           #define csqrt fake_csqrt
           #define csin fake_csin
           #define ccos fake_ccos
           #define ctan fake_ctan
           #define csinh fake_csinh
           #define ccosh fake_ccosh
           #define ctanh fake_ctanh
       #endif
       #include "cl_complex.h"
   #else
       #ifdef __cplusplus
           #include <cmath>
           #include <complex>
           #include <iostream>
           #define cexp exp
           #define csqrt sqrt
           #define csin sin
           #define ccos cos
           #define ctan tan
           #define csinh sinh
           #define ccosh cosh
           #define ctanh tanh
           #define cabs abs
           #define creal real
           #define cimag imag
           typedef std::complex<double> cdouble;
           const cdouble I(0.0, 1.0);
       #else
           #include <math.h>
           #include <complex.h>
           typedef double complex cdouble;
       #endif
       inline cdouble cplx(const double x, const double y) { return x + I*y; }
       inline cdouble cadd(const cdouble x, const cdouble y) { return x+y; }
       inline cdouble csub(const cdouble x, const cdouble y) { return x-y; }
       inline cdouble cmul(const cdouble x, const cdouble y) { return x*y; }
       inline cdouble cdiv(const cdouble x, const cdouble y) { return x/y; }
       inline cdouble cneg(const cdouble x) { return -x; }
       inline cdouble radd(const double x, const cdouble y) { return x+y; }
       inline cdouble rsub(const double x, const cdouble y) { return x-y; }
       inline cdouble rmul(const double x, const cdouble y) { return x*y; }
       inline cdouble rdiv(const double x, const cdouble y) { return x/y; }
   #endif

   // opencl defines
   #ifdef __cplusplus
   #  define kernel extern "C"
   #else
   #  define kernel
   #endif
   #define constant const
   #define global
   #define local
   #define powr(a,b) pow(a,b)
#else
   #ifdef USE_SINCOS
   #  define SINCOS(angle,svar,cvar) svar=sincos(angle,&cvar)
   #else
   #  define SINCOS(angle,svar,cvar) do {svar=sin(angle);cvar=cos(angle);} while (0)
   #endif
#endif
// Standard mathematical constants:
//   M_E, M_LOG2E, M_LOG10E, M_LN2, M_LN10, M_PI, M_PI_2=pi/2, M_PI_4=pi/4,
//   M_1_PI=1/pi, M_2_PI=2/pi, M_2_SQRTPI=2/sqrt(pi), SQRT2, SQRT1_2=sqrt(1/2)
// OpenCL defines M_constant_F for float constants, and nothing if double
// is not enabled on the card, which is why these constants may be missing
#ifndef M_PI
#  define M_PI 3.141592653589793
#endif
#ifndef M_PI_2
#  define M_PI_2 1.570796326794897
#endif
#ifndef M_PI_4
#  define M_PI_4 0.7853981633974483
#endif
