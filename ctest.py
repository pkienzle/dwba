"""
Compare algorithms for complex division and square root.
"""
import math
from math import sqrt, fabs
import numpy as np

class Complex:
    def __init__(self, y):
        self.x,self.y = y.real,y.imag

def cplx(x,y): return x+1j*y

def cdiv(a,b):
    return np.array([_cdiv(ai,bi) for ai,bi in zip(a,b)])

def _cdiv(a,b):
    a,b = Complex(a),Complex(b)
    if (fabs(b.x) >= fabs(b.y)):
        t = b.y/b.x;
        den = b.x + b.y*t;
        u = (a.x + a.y*t)/den;
        v = (a.y - a.x*t)/den;
        return cplx(u,v);
    else:
        t = b.x/b.y;
        den = b.x*t + b.y;
        u = (a.x*t + a.y)/den;
        v = (a.y*t - a.x)/den;
        return cplx(u,v);
def _cdiv(a,b):
    a,b = Complex(a),Complex(b)
    scale = 1.0/(b.x*b.x  + b.y*b.y);
    return cplx(scale*(a.x*b.x+a.y*b.y), scale*(a.y*b.x - a.x*b.y));

def rdiv(a,b):
    return np.array([_rdiv(ai,bi) for ai,bi in zip(a,b)])
def _rdiv(a,b):
    b = Complex(b)
    if (fabs(b.x) >= fabs(b.y)):
        t = b.y/b.x;
        den = b.x + b.y*t;
        u = a/den;
        return cplx(u,-t*u);
    else:
        t = b.x/b.y;
        den = b.x*t + b.y;
        v = a/den;
        return cplx(t*v,-v);
def _rdiv(a,b):
    b = Complex(b)
    scale = a/(b.x*b.x  + b.y*b.y);
    return cplx(scale*b.x, -scale*b.y);

def csqrt(a):
    return np.array([_csqrt(ai) for ai in a])
def _csqrt(z):
    z = Complex(z)
    a = z.x;
    b = z.y;
    if (a == 0.0):
        real = sqrt(0.5*fabs(b));
        imag = -real if a < 0.0 else real;
        return cplx(real, imag);
    else:
        t = sqrt(2.0 * (abs(a+b*1j) + fabs(a)));
        u = 0.5*t;
        if (a > 0.0):
            return cplx(u,b/t);
        else:
            real = fabs(b)/t;
            imag = -u if b<0.0 else u;
            return cplx(real,imag);
def _cabs(a):
    return sqrt(a.x*a.x + a.y*a.y);
def _carg(a):
    return math.atan2(a.y, a.x);
def _cqrt(z):
    a = Complex(z)
    mod = sqrt(_cabs(a));
    arg = 0.5*_carg(a);
    return cplx(mod * math.cos(arg),  mod * math.sin(arg));

def test():
    import numpy as np
    N = 1000
    s = 1e15
    a = s*(np.random.rand(N)-0.5) + s*1j*(np.random.rand(N)-0.5)
    b = s*(np.random.rand(N)-0.5) + 1/s*1j*(np.random.rand(N)-0.5)
    bad = ~(abs(csqrt(a)**2-a)<abs(a)*1e-15)
    print "sqrt**2","\n".join(str(i) for i in zip(a[bad],csqrt(a)[bad],csqrt(a)[bad]**2))
    bad = ~(abs(csqrt(a)-np.sqrt(a))<np.sqrt(abs(a))*1e-15)
    print "np.sqrt","\n".join(str(i) for i in zip(a[bad],csqrt(a)[bad]))

    bad = ~(abs(cdiv(a,b) - a/b)<(abs(a)/abs(b))*1e-15)
    print "a/b","\n".join(str(i) for i in zip(a[bad],b[bad],cdiv(a,b)[bad]))

    bad = ~(abs(rdiv(a.real,b) - a.real/b)<(abs(a.real)/abs(b))*1e-15)
    print "a.real/b","\n".join(str(i) for i in zip(a.real[bad],b[bad],rdiv(a.real,b)[bad]))


if __name__ == "__main__":
    test()
