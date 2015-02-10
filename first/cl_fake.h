// ===== Open CL / C99 / C++  common header ======
/*
Provides a common environment for opencl/c++/c99 so that
*/

#ifdef __OPENCL_VERSION__
# define USE_OPENCL
#endif

// If opencl is not available, then we are compiling a C function
// Note: if using a C++ compiler, then define kernel as extern "C"
#ifndef USE_OPENCL
   // Support for complex numbers, using c++ and/or c99
   #ifdef __cplusplus
     #include <cmath>
     // Define EMULATE_FLOAT2 to force use of the cl_complex.h header and
     // functions instead of the C/C99 complex math libraries.  The main
     // difference is in the arithmetic:
     //    s+(a,b) is (s,s)+(a,b) instead of (s,0)+(a,b) and
     //    (a,b)*(c,d) gives (a*b,c*d) instead of (a*c-b*d,a*d+b*c).
     // There may also be speed and precision differences due to different
     // algorithms.
     #if defined(USE_FLOAT2)
       #include "float2.hpp"
       #include "cl_complex.h"
     #else // USE_COMPLEX
       #include <complex>
       #include <iostream>
       #define cexp exp
       #define csqrt sqrt
       #define cexp exp
       #define cexp exp
       typedef std::complex<double> cdouble;
       const cdouble I(0.0, 1.0);
       inline cdouble cplx(double x, double y) { return cdouble(x,y); }
       inline cdouble cmul(cdouble x, cdouble y) { return x*y; }
       inline cdouble cdiv(cdouble x, cdouble y) { return x/y; }
     #endif
   #else
     #include <math.h>
     #include <complex.h>
     typedef double complex cdouble;
     inline cdouble cplx(double x, double y) { return x + I*y; }
     inline cdouble cmul(cdouble x, cdouble y) { return x*y; }
     inline cdouble cdiv(cdouble x, cdouble y) { return x/y; }
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
   #define SINCOS(angle,svar,cvar) do {svar=sin(angle);cvar=cos(angle);} while (0)
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
