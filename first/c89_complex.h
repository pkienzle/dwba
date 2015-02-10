#warning c89_complex.h doesn't work

/*
Simple complex library for OpenCL

TODO: use more robust algorithms

declare:     cdouble x
define:      cplx(real, imag)
1i:          I
-x:          -x
x + scalar:  x + cplx(scalar, 0.0)
x - scalar:  x - cplx(scalar, 0.0)
x * scalar:  x * scalar
scalar * x:  scalar * x
x / scalar:  x / scalar
scalar / x:  cdiv(cplx(scalar, 0.0), x)
x + y:       x + y
x - y:       x - y
x * y:       cmul(x,y)
x / y:       cdiv(x,y)
abs(x):      cmod(x)
angle(x):    carg(x)
sqrt(x):     csart(x)
exp(x):      cexp(x)

*/

// two component vector to hold the real and imaginary parts of a complex number:
// Note: the double->float converter in cl_util.py needs the extra space in
// order to convert this to "typedef float2  cfloat;"
typedef struct { double x, y; } cdouble;

inline cdouble cplx(double x, double y) {
    const cdouble ret = {x,y};
    return ret;
}

#define I (cplx(0.0, 1.0))


/*
 * Return Real (Imaginary) component of complex number:
 */
inline double  real(cdouble a){
     return a.x;
}
inline double  imag(cdouble a){
     return a.y;
}

/*
 * Get the modulus of a complex number (its length):
 */
inline double cmod(cdouble a){
    return (sqrt(a.x*a.x + a.y*a.y));
}

/*
 * Get the argument of a complex number (its angle):
 * http://en.wikipedia.org/wiki/Complex_number#Absolute_value_and_argument
 */
inline double carg(cdouble a){
    return atan2(a.y, a.x);
}

/*
 * Multiply two complex numbers:
 *
 *  a = (aReal + I*aImag)
 *  b = (bReal + I*bImag)
 *  a * b = (aReal + I*aImag) * (bReal + I*bImag)
 *        = aReal*bReal +I*aReal*bImag +I*aImag*bReal +I^2*aImag*bImag
 *        = (aReal*bReal - aImag*bImag) + I*(aReal*bImag + aImag*bReal)
 */
inline cdouble  cmul(cdouble a, cdouble b){
    const cdouble ret = { a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
    return ret;
}


/*
 * Divide two complex numbers:
 *
 *  aReal + I*aImag     (aReal + I*aImag) * (bReal - I*bImag)
 * ----------------- = ---------------------------------------
 *  bReal + I*bImag     (bReal + I*bImag) * (bReal - I*bImag)
 *
 *        aReal*bReal - I*aReal*bImag + I*aImag*bReal - I^2*aImag*bImag
 *     = ---------------------------------------------------------------
 *            bReal^2 - I*bReal*bImag + I*bImag*bReal  -I^2*bImag^2
 *
 *        aReal*bReal + aImag*bImag         aImag*bReal - Real*bImag
 *     = ---------------------------- + I* --------------------------
 *            bReal^2 + bImag^2                bReal^2 + bImag^2
 *
 */
inline cdouble cdiv(cdouble a, cdouble b){
    const double scale = 1.0/cmod(b);
    const cdouble ret = {scale*(a.x*b.x + a.y*b.y), scale*(a.y*b.x - a.x*b.y)};
    return ret;
}


/*
 *  Square root of complex number.
 *  Although a complex number has two square roots, numerically we will
 *  only determine one of them -the principal square root, see wikipedia
 *  for more info:
 *  http://en.wikipedia.org/wiki/Square_root#Principal_square_root_of_a_complex_number
 */
 inline cdouble csqrt(cdouble a){
     const double arg = 0.5*carg(a), mod=sqrt(cmod(a));
     const cdouble ret = { mod * cos(arg),   mod * sin(arg)};
     return ret;
 }

/* e^(a+bi) = e^a e^bi = e^a (cos b + i sin b) = e^a cos b + i e^a sin b
 */
inline cdouble cexp(cdouble a){
    const double mod = exp(a.x);
    const cdouble ret = { mod * cos(a.y),   mod * sin(a.y) };
    return ret;
}
