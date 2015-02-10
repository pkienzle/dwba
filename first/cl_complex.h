/*
Simple complex library for OpenCL

TODO: use more robust algorithms

declare:     cdouble x
define:      cplx(real, imag)
1j:          I
-x:          rmul(-1.0, x)
real + x:    radd(real, x)
x + real:    radd(real, x)
x - real:    radd(-real, x)
x * real:    rmul(real, x)
real * x:    rmul(real, x)
real - x:    rsub(real, x)
x / real:    divr(x, real)
real / x:    rdiv(real, x)
x + y:       cadd(x,y)
x - y:       csub(x,y)
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
typedef double2  cdouble;

#define I (cplx(0.0, 1.0))

inline cdouble cplx(double x, double y) { return (cdouble)(x,y); }
// For C++, we can use the following constructor since the opencl form
// will not work.
//inline cdouble cplx(double x, double y) { return V2<double>(x,y); }

/*
 * Return Real (Imaginary) component of complex number:
 */
inline double  real(cdouble a) { return a.x; }
inline double  imag(cdouble a) { return a.y; }

/*
 * Get the modulus of a complex number (its length):
 */
inline double cmod(cdouble a) { return length(a); }

/*
 * Get the argument of a complex number (its angle):
 * http://en.wikipedia.org/wiki/Complex_number#Absolute_value_and_argument
 */
inline double carg(cdouble a) { return atan2(a.y, a.x); }

inline cdouble cpolar(double r, double theta) {
    //double si, ci;
    //SINCOS(theta, si, ci);
    //return (cdouble)(r*ci, r*si);
    return cplx(r*cos(theta), r*sin(theta));
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
inline cdouble  cmul(cdouble a, cdouble b) {
    return cplx( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
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
inline cdouble cdiv(cdouble a, cdouble b) {
    const double scale = 1.0/dot(b,b);
    return cplx(scale*dot(a,b), scale*(a.y*b.x - a.x*b.y));
}


/*
 *  Square root of complex number.
 *  Although a complex number has two square roots, numerically we will
 *  only determine one of them -the principal square root, see wikipedia
 *  for more info:
 *  http://en.wikipedia.org/wiki/Square_root#Principal_square_root_of_a_complex_number
 */
 inline cdouble csqrt(cdouble a) {
     const double mod = sqrt(cmod(a));
     const double arg = 0.5*carg(a);
     //return cpolar(mod, arg);
     return cplx(mod * cos(arg),  mod * sin(arg));
 }

/* e^(a+bi) = e^a e^bi = e^a (cos b + i sin b) = e^a cos b + i e^a sin b
 */
inline cdouble cexp(cdouble a) {
    double mod = exp(a.x);
    //return cpolar(mod, a.y);
    return cplx(mod * cos(a.y), mod * sin(a.y));
}
