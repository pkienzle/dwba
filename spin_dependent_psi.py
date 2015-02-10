import numpy as np
from numpy import pi, sqrt, exp, sin, cos, radians
from numpy import zeros, array, matrix, eye, diag, ones_like

#np.set_printoptions(linewidth=1320,precision=6)

EPSILON = 1e-10
B2SLD = 2.31604654e-6
PI4 = complex(pi * 4.0)

cdouble = np.dtype('complex128')
rdouble = np.dtype('float64')
class Old_Calculator:
    def __init__(self, dtype=rdouble):
        import os, multiprocessing
        import ctypes as ct
        #print "Nthreads",self.Nthreads
        lib = np.ctypeslib.load_library('oldmag.so', '.')
        lib.magnetic_amplitude.argtypes = [
            ct.c_int,   # layers
            np.ctypeslib.ndpointer(dtype=rdouble), # d
            np.ctypeslib.ndpointer(dtype=rdouble), # sigma
            np.ctypeslib.ndpointer(dtype=rdouble), # rho
            np.ctypeslib.ndpointer(dtype=rdouble), # irho
            np.ctypeslib.ndpointer(dtype=rdouble), # rhoM
            np.ctypeslib.ndpointer(dtype=cdouble), # u1
            np.ctypeslib.ndpointer(dtype=cdouble), # u3
            ct.c_double, # Aguide
            ct.c_int, # points
            np.ctypeslib.ndpointer(dtype=rdouble), # kz
            ct.c_void_p, # rho_index
            np.ctypeslib.ndpointer(dtype=cdouble), # Ra
            np.ctypeslib.ndpointer(dtype=cdouble), # Rb
            np.ctypeslib.ndpointer(dtype=cdouble), # Rc
            np.ctypeslib.ndpointer(dtype=cdouble), # Rd
        ]
        self.fn = lib.magnetic_amplitude

    def __call__(self, kz, dz, rhoN, bx, by, bz, AGUIDE):
        Nlayers = len(dz)
        Nkz = kz.size
        sigma = np.zeros(Nlayers)
        irho = np.zeros(Nlayers)
        rhoB,U1,U3 = calculateU(bx, by, bz, AGUIDE)
        R = [np.empty(Nkz, dtype=cdouble) for _ in range(4)]
        self.fn(Nlayers, dz, sigma, rhoN, irho, rhoB, U1, U3, AGUIDE, Nkz, kz, 0, *R)
        Rp = np.vstack(R)
        return Rp


def calculate(kz_array, dz, rhoN, bx, by, bz, AGUIDE):
    rhoB,U1,U3 = calculateU(bx, by, bz, AGUIDE)

    result = [calculateRB(kzi, dz, rhoN, rhoB, U1, U3)
              for kzi in kz_array]
    return [array(v) for v in zip(*result)]


def calculateB(dz, S1, S3, U1, U3):
    """
    Calculation of reflectivity in magnetic sample in framework that 
    also yields the wavefunction in each layer for DWBA.  
    Arguments:
    
     kz is the incoming momentum along the z direction
     dz is the thickness of each layer (top and bottom thickness ignored)
     rhoN is nuclear scattering length density for each layer
     rhoB is magnetic scattering length density for each layer
     bx, by, bz are components of unit vector along M for layer
     if spin=1.0, the B calculated will be valid for r+- and r++...
        spin=-1.0, B will be valid for r-+ and r--
    ###################################################################
    #  All of dz, rhoN, rhoB, bx, by, bz should have length equal to  #
    #  the number of layers in the sample including fronting and      #
    #  substrate. Layers are ordered top to bottom                    #
    ###################################################################
     AGUIDE is the angle of the z-axis of the sample with respect to the
     z-axis (quantization) of the neutron in the lab frame.
     AGUIDE = 0 for perpendicular films (= 270 for field along y)

    """
    N = len(dz)

    #global B # matrices for calculating R
    B = zeros((N-1,4,4), dtype='complex') # one matrix per interface

    #print "init",U1[1],S1[0],S1[0]/S1[1]
    # apply chi at layer 0
    z = 0.0
    S  = array([exp(S1[0]*z), exp(-S1[0]*z), exp(S3[0]*z), exp(-S3[0]*z)])
    chi = matrix(
        [[  1,      1,      0,      0   ],
         [  0,      0,      1,      1   ],
         [S1[0], -S1[0],    0,      0   ],
         [  0,      0,    S3[0], -S3[0] ]])
    #newB = chi * matrix(diag(S)) * eye(4)
    newB = chi  # z=0 => diag(S)=eye(4)
    for L in range(1,N):
        # apply inverse chi at layer L+1
        S_inv = array([exp(-S1[L]*z), exp(S1[L]*z), exp(-S3[L]*z), exp(S3[L]*z)])
        chi_inv = array(
            [[ U3[L], -1,  U3[L]/S1[L], -1/S1[L] ],
             [ U3[L], -1, -U3[L]/S1[L],  1/S1[L] ],
             [ -U1[L], 1, -U1[L]/S3[L],  1/S3[L] ],
             [ -U1[L], 1,  U1[L]/S3[L], -1/S3[L] ]])
        #chi_inv *= (1.0 + 0j) / (2.0*(U3[L] - U1[L]))
        #A = np.dot(diag(S_inv)),  chi_inv)
        A =(S_inv[:,None] * (0.5/(U3[L] - U1[L]))) * chi_inv
        newB = np.dot(A, newB)
        #print "========= A ========="
        #print matrix(diag(S_inv))*chi_inv*chi*matrix(diag(S))
        #print "======  end A ======="

        B[L-1] = newB
        #print "========= B ========="
        #print "layer=%d B11=%g"%(L, newB[0,0])
        #print (newB)

        # apply chi at layer L+1
        z += dz[L]
        S  = array([exp(S1[L]*z),exp(-S1[L]*z), exp(S3[L]*z), exp(-S3[L]*z)])
        chi = array(
            [[1,               1,             1,              1            ],
             [U1[L],          U1[L],         U3[L],          U3[L]        ],
             [S1[L],         -S1[L],         S3[L],         -S3[L]        ],
             [U1[L]*S1[L],   -U1[L]*S1[L],   U3[L]*S3[L],   -U3[L]*S3[L]] ])
        #A = np.dot(chi, diag(S))
        A = chi * S[None,:]
        newB = np.dot(A, newB)

    return B



def calculateC_sam(kz, C0_sam, B):
    """ take 4x1 matrix (row vector) of initial C
    and get C in each layer """
    layers = B.shape[0]
    C_all = zeros((layers, 4), dtype='complex')
    for L in range(layers):
        C_all[L] = B[L] * C0_sam
    return C_all

def calculateRB(kz, dz, rhoN, rhoB, U1, U3):

    if abs(kz) < EPSILON:
        return  -1.0, 0.0, 0.0, -1.0, eye(4)


    #print "kzi",kz
    S1,S3 = calculateS(kz, rhoN, rhoB, spin=1.0)
    Bp = calculateB(dz, S1, S3, U1, U3)
    Rp_lab = calculateR_sam(kz, Bp[-1]) # we rotated magnetization b-vector, so Rlab = Rsam
    #print "Rp",np.array(Rp_lab)
    S1,S3 = calculateS(kz, rhoN, rhoB, spin=-1.0)
    Bm = calculateB(dz, S1, S3, U1, U3)
    Rm_lab = calculateR_sam(kz, Bm[-1])

    return Rp_lab[0], Rp_lab[1], Rm_lab[2], Rm_lab[3] #, Bp


def calculateR_sam(kz, B):
    denom = complex(1.0) / ((B[3,3] * B[1,1]) - (B[1,3] * B[3,1]))
    #print "denom:",denom
    #print np.array([B[1,1],B[1,3],B[3,1],B[3,3]])
    #print "=== B ==="; print B; print "=== end B ==="
    if kz > 0:
        #print np.array([B[1,0],B[1,2],B[3,0],B[3,2]])
        rpp = ((B[1,3] * B[3,0]) - (B[1,0] * B[3,3]))*denom
        rpm = ((B[1,0] * B[3,1]) - (B[3,0] * B[1,1]))*denom
        rmp = ((B[1,3] * B[3,2]) - (B[1,2] * B[3,3]))*denom
        rmm = ((B[1,2] * B[3,1]) - (B[3,2] * B[1,1]))*denom
        #print np.array([rpp,rpm,rmp,rmm])
        #print np.abs(np.array([rpp,rpm,rmp,rmm]))
    else:
        rpp = ((B[0,1] * B[3,3]) - (B[0,3] * B[3,1]))*denom
        rpm = ((B[2,1] * B[3,3]) - (B[2,3] * B[3,1]))*denom
        rmp = ((B[0,1] * B[1,3]) - (B[0,3] * B[1,1]))*denom
        rmm = ((B[2,1] * B[1,3]) - (B[2,3] * B[1,1]))*denom
    return rpp, rpm, rmp, rmm

def calculateU(bx, by, bz, AGUIDE):
    # rotate bx,by,bz
    bx, by, bz = rotateYZ(bx, by, bz, AGUIDE)

    # precompute rhoB, u1, u3, independent of kz
    b_tot = sqrt(bx**2 + by**2 + bz**2)
    b_nz = (b_tot != 0)
    u1 = ones_like(b_tot, dtype='complex')
    u3 = -ones_like(b_tot, dtype='complex')
    u1[b_nz] = ( b_tot + bx + 1j*by - bz )[b_nz] / ( b_tot + bx - 1j*by + bz )[b_nz]
    u3[b_nz] = (-b_tot + bx + 1j*by - bz )[b_nz] / (-b_tot + bx - 1j*by + bz )[b_nz]
    rhoB = B2SLD * b_tot
    return rhoB, u1, u3

def calculateS(kz, rhoN, rhoB, spin=1.0):
    # fronting medium removed from effective kz
    surface = 0 if kz >= 0 else -1
    E0 = kz**2 + PI4*rhoN[surface] + PI4*spin*rhoB[surface] + EPSILON*1j

    #print "L1:",np.array([rhoN[1],rhoB[1],u1[1],u3[1]])
    #S1 = -sqrt(PI4*(rhoN + rhoB)-E0 -EPSILON*1j)
    #S3 = -sqrt(PI4*(rhoN - rhoB)-E0 -EPSILON*1j)
    S1 = -sqrt(PI4*(rhoN + rhoB) - E0)
    S3 = -sqrt(PI4*(rhoN - rhoB) - E0)
    return S1,S3

def rotateYZ(bx, by, bz, AGUIDE):
    # get B in lab frame, from sample frame
    C = cos(radians(AGUIDE))
    S = sin(radians(AGUIDE))
    bxr = bx
    byr = bz * S + by * C
    bzr = bz * C - by * S
    return bxr, byr, bzr

def plot(kz, R):
    import pylab
    rpp, rpm, rmp, rmm = R
    pylab.plot(2*kz, abs(rpp)**2, label="r++")
    pylab.plot(2*kz, abs(rmp)**2, label="r-+", hold=True)
    pylab.plot(2*kz, abs(rpm)**2, label="r+-", hold=True)
    pylab.plot(2*kz, abs(rmm)**2, label="r--", hold=True)
    pylab.yscale('log')
    pylab.legend()

# ==== UNUSED CODE ====

def _calculateR_lab(R_sam, AGUIDE):
    # this only works when rhoB is zero in the fronting (and backing?)!
    r_lab = _unitary_LAB_SAM_LAB2(matrix([[R_sam[0], R_sam[2]], [R_sam[1], R_sam[3]]]), AGUIDE);

    YA_lab = r_lab[0,0]; # r++
    YB_lab = r_lab[1,0]; # r+-
    YC_lab = r_lab[0,1]; # r-+
    YD_lab = r_lab[1,1]; # r--

    return YA_lab, YB_lab, YC_lab, YD_lab

def _get_U_sam_lab(AGUIDE):
    C = complex(cos(AGUIDE/2.0*pi/180.))
    IS = 1j * sin(AGUIDE/2.0*pi/180.)
    U = matrix([ [C , IS, 0 , 0 ],
                 [IS, C , 0 , 0 ],
                 [0 , 0 , C , IS],
                 [0 , 0 , IS, C ] ])
    return U

def _get_Uinv_sam_lab(AGUIDE):
    C = complex(cos(AGUIDE/2.0*pi/180.))
    NS = 1j * -sin(AGUIDE/2.0*pi/180.)
    Uinv = matrix([ [C , NS, 0 , 0 ],
                    [NS, C , 0 , 0 ],
                    [0 , 0 , C , NS],
                    [0 , 0 , NS, C ] ])
    return Uinv

def _get_U_sam_lab2(AGUIDE):
    C = complex(cos(AGUIDE/2.0*pi/180.))
    IS = 1j * sin(AGUIDE/2.0*pi/180.)
    U = matrix([ [C , IS],
                 [IS, C ] ])
    return U

def _get_Uinv_sam_lab2(AGUIDE):
    C = complex(cos(AGUIDE/2.0*pi/180.))
    NS = 1j * -sin(AGUIDE/2.0*pi/180.)
    Uinv = matrix([ [C , NS],
                    [NS, C ] ])
    return Uinv

def _unitary_LAB_SAM_LAB(A, AGUIDE):
    """ perform rotation of coordinate system on one side of matrix
    and perform inverse on the other side """
    U = _get_U_sam_lab(AGUIDE)
    Uinv = _get_Uinv_sam_lab(AGUIDE)
    CST =  (U * A) * Uinv
    return CST

def _unitary_LAB_SAM_LAB2(A, AGUIDE):
    """ perform rotation of coordinate system on one side of matrix
    and perform inverse on the other side """
    U = _get_U_sam_lab2(AGUIDE)
    Uinv = _get_Uinv_sam_lab2(AGUIDE)
    CST =  (U * A) * Uinv
    return CST

# ==== UNUSED CODE ====

def _afm_model():
    dz_mult = 5
    dz =   array([1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0, 1.0]) * dz_mult
    rhoN_mult = 2e-4
    rhoN = array([0.0, 2.0, 1.0,  2.0, 1.0, 2.0, 1.0, 0.0]) * rhoN_mult
    #rhoB_mult = 2e-4 / 2.31604654e-6
    #rhoB = array([0.0, 1.0, 0.0,  1.0, 0.0, 1.0, 0.0, 0.0]) * rhoB_mult
    by =   array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    bx =   array([0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0])
    bz =   array([0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0])
    return dz, rhoN, bx, by, bz

def _Yaohua_model(H=0.4):
    """
    layers = [
        # depth rho rhoM thetaM phiM
        [ 0, 0.0, rhoB, 90, 0.0],
        [ 200, 4.0, rhoB + 1.0, np.degrees(np.arctan2(rhoB, 1.0)), 0.0],
        [ 200, 2.0, rhoB + 1.0, 90, 0.0],
        [ 0, 4.0, rhoB, 90 , 0.0],
        ]
    """
    B2SLD = 2.31929 # *1e-6
    dz_mult = 200.0
    dz =   array([      1.0,           1.0,           1.0,       1.0]) * dz_mult
    rhoN_mult = 1e-6
    rhoN = array([      0.0,           4.0,           2.0,       4.0]) * rhoN_mult
    by =   array([        H,             H, (1.0/B2SLD)+H,         H])
    bx =   array([      0.0,     1.0/B2SLD,           0.0,       0.0])
    bz =   array([      0.0,           0.0,           0.0,       0.0])
    return dz, rhoN, bx, by, bz

def _random_model(Nlayers=200, H=0.4, seed=None):
    if seed is None: seed = np.random.randint(1000000)
    print "Seed:",seed
    np.random.seed(seed)
    dz = np.random.rand(Nlayers)*800./Nlayers
    rhoN = np.random.rand(Nlayers)*1e-5
    by = np.ones_like(dz)*H + np.random.rand(Nlayers)*0.1
    bx = np.random.rand(Nlayers)*0.1
    bx[0] = bx[-1] = 0
    bz = np.zeros_like(dz)
    return dz, rhoN, bx, by, bz

def _demo(model, reversed=False, rotated=True):
    import time

    dz, rhoN, bx, by, bz = model
    kz= np.linspace(0.0001, 0.0151, 201)
    AGUIDE = 270.0 # 90.0 would be along y
    if not rotated:
        AGUIDE = 0.0
        bx,by,bz = bz+EPSILON,bx,by
        #bx,by,bz = bz,bx,by
    if reversed:
        kz= -kz
        dz, rhoN, bx, by, bz = [v[::-1] for v in dz, rhoN, bx, by, bz]
    t0 = time.time()
    R = calculate(kz, dz, rhoN, bx, by, bz, AGUIDE)
    print "done with time = %g" % (time.time() - t0)
    plot(abs(kz), R)


if __name__ == '__main__':
    import sys
    rotated = "unrotated" not in sys.argv
    reversed = "reversed" in sys.argv
    if "cpu" in sys.argv: calculate = Old_Calculator()
    #model = _afm_model()
    model = _Yaohua_model(H=0.4)
    model = _random_model(Nlayers=4000, H=0.4, seed=10248)
    #for p,v in zip("dz Nr bx by bz".split(), model): print p,v
    _demo(model, reversed=reversed, rotated=rotated)
    import pylab; pylab.show()

