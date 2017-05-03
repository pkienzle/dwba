import numpy as np
from numpy import pi, sqrt, cos, sin, radians

#np.set_printoptions(linewidth=1320,precision=6)

EPSILON = 1e-10
B2SLD = 2.31604654e-6

cdouble = np.dtype('complex128')
cfloat = np.dtype('complex64')
rdouble = np.dtype('float64')
rfloat = np.dtype('float32')

class CPU_Calculator:
    def __init__(self, dtype=rdouble):
        import os, multiprocessing
        import ctypes as ct
        self.Nthreads = int(os.environ.get("OMP_NUM_THREADS", str(multiprocessing.cpu_count())))
        os.environ["OMP_NUM_THREADS"] = str(self.Nthreads)
        #print "Nthreads",self.Nthreads
        lib = np.ctypeslib.load_library('libga3.so', '.')
        lib.magneticR.argtypes = [
            ct.c_double,
            ct.c_int,
            ct.c_int,
            np.ctypeslib.ndpointer(dtype=rdouble),
            np.ctypeslib.ndpointer(dtype=cdouble),
            np.ctypeslib.ndpointer(dtype=cdouble),
            np.ctypeslib.ndpointer(dtype=cdouble),
            ]
        self.fn = lib.magneticR
        self.Nlayers = 0

    def __call__(self, kz, dz, rhoN, bx, by, bz, Aguide):
        Nlayers = len(dz)
        Nkz = kz.size
        self.allocate_B(Nlayers)
        L = layer_data(dz, rhoN, bx, by, bz, Aguide, dtype=rdouble)
        #print "L:",L
        Rp = np.empty((kz.size,4), dtype=cdouble)
        Rm = np.empty((kz.size,4), dtype=cdouble)
        #print "="*20
        #print "dwba on cpu"
        #print "="*20
        self.fn(1.0, Nkz, Nlayers, kz, L, self.B, Rp)
        self.fn(-1.0, Nkz, Nlayers, kz, L, self.B, Rm)
        Rp[:,2:] = Rm[:,2:]
        return Rp.T

    def allocate_B(self, Nlayers):
        """
        Allocate a B matrix from running the compute kernel using OpenMP directives.

        This matrix must be handed to the compiled C kernel as the temporary B
        matrix.  Since the complete B matrix may be large to hold in memory
        depending on the number of layers and the length of kz, the size is
        limited to one kz value per parallel thread.

        Uses OMP_NUM_THREADS if it is declared in the environment, otherwise
        defaults to the number of processors on the CPU.
        """
        if self.Nlayers < Nlayers:
            self.B = np.empty((self.Nthreads, Nlayers-1, 4, 4),  dtype="complex")
            self.Nlayers = Nlayers


class GPU_Calculator:
    def __init__(self, dtype=rfloat):
        import pyopencl as cl
        from cl_util import get_env, get_warp
        env = get_env()
        mf = cl.mem_flags
        self.queue = env.queues[0]
        self.T = env.dtype(dtype)

        # compile the scattering kernel
        source = open('magnetic_refl.c').read()
        program = env.compile_program('magnetic_refl', source, self.T.real)
        self.kernel = program.magneticR
        self.Nkz = 0
        self.Nlayers = 0
        self.buffers = []

    def __call__(self, kz, dz, rhoN, bx, by, bz, Aguide):
        import pyopencl as cl
        T = self.T

        self.allocate_buffers(len(dz), kz.size)
        gkz, gL, gB, gR = self.buffers
        kz = np.ascontiguousarray(kz, dtype=T.real)
        L = layer_data(dz, rhoN, bx, by, bz, Aguide, dtype=T.real)
        #print "dz",L[:,3].imag
        #print "rhoN",L[:,0]
        #print "beta",L[:,1]
        #print "gamma",L[:,2]
        #print "rhoB",L[:,3].real
        if 1:
            cl.enqueue_copy(self.queue, gL, L, is_blocking=False)
            cl.enqueue_copy(self.queue, gkz, kz, is_blocking=False)
        else:
            from pyopencl import mem_flags as mf
            [b.release() for b in self.buffers[:2]]
            context = self.queue.context
            self.buffers[:2] = [
                cl.Buffer(context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=kz),
                cl.Buffer(context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=L),
            ]
        Rp = self._calculate_R(1.0)
        Rm = self._calculate_R(-1.0)
        Rp[:,2:] = Rm[:,2:]
        #print Rp[-8:,:]
        return Rp.T

    def _calculate_R(self, spin):
        import pyopencl as cl
        T = self.T
        gkz, gL, gB, gR = self.buffers

        # Process kz piecewise since we are limited by the amount of memory
        # available for the B buffer.  Instead we process the data in chunks
        # of Q, with chunk size determined by allocate_buffers.
        offset = 0
        while offset < self.Nkz:
            part = min(self.Bwidth , self.Nkz-offset)
            #print "starting kernel with",gL,T.real,T.int
            self.kernel(self.queue,  [self.Nwarp*(part//self.Nwarp)], None,
                        T.real(spin),
                        T.int(offset), T.int(part), T.int(self.Nlayers),
                        gkz, gL, gB, gR)
            offset += part
            #print "offset",offset,"Nkz",self.Nkz
            break

        # Get the return value
        R = np.empty((self.Nkz,4), dtype=T.complex)
        cl.enqueue_copy(self.queue, R, gR)

        return R


    def allocate_buffers(self, Nlayers, Nkz):
        """
        Allocate a B matrix for running the compute kernel on the given queue.

        Determine the number of kz values that can be computed simultaneously
        on the card based on the available memory, the warp, the number of layers
        and the data type.

        *memory* is the amount of memory available for storing the B array.  This
        will be *queue.device.global_mem_size/4* so that you don't have to
        worry too much about running out of memory, even if you are keeping two
        kz sets active at once.

        *warp* is the number of kz values that can run together in the same warp,
        as returned from cl_util.warp(*kernel*, *queue*).

        *layers* is the number of layers required for the intermediate B matrix.

        *dtype* is the floating point type of the B matrix elements.

        Returns the number of kz values that should by scheduled together and
        the temporary B matrix that can be used for the kernel.
        """
        #raise ValueError("fails for Nkz not a multiple of 8 on mac")
        if self.Nlayers < Nlayers or self.Nkz < Nkz:
            import pyopencl as cl
            from pyopencl import mem_flags as mf
            import cl_util
            T = self.T

            context = self.queue.context
            [v.release() for v in self.buffers]

            max_memory_size = self.queue.device.global_mem_size//4
            threads_per_warp = cl_util.get_warp(self.kernel, self.queue)
            warp_size = (
                T.complex_size  # bytes per complex
                * 16 # B matrix size
                * (Nlayers-1) # number of B matrices
                * threads_per_warp # number of qz per warp
            )
            number_of_warps = min([max_memory_size//warp_size,
                                   (Nkz+threads_per_warp-1)//threads_per_warp,
                                   ])
            self.Bwidth = number_of_warps * threads_per_warp
            self.Nwarp = threads_per_warp
            self.Nkz = Nkz
            self.Nlayers = Nlayers
            gkz = cl.Buffer(context, mf.READ_ONLY, T.real_size*Nkz)
            gL = cl.Buffer(context, mf.READ_ONLY, 4*T.complex_size*Nlayers)
            gB = cl.Buffer(context, mf.READ_WRITE, number_of_warps*warp_size)
            gR = cl.Buffer(context, mf.WRITE_ONLY, 4*T.complex_size*Nkz)
            self.buffers = [gkz, gL, gB, gR]



def layer_data(dz, rhoN, bx, by, bz, Aguide, dtype=rdouble):
    print "Aguide", Aguide
    print "d", dz
    print "rho", rhoN.real
    print "bx", bx
    print "by", by
    print "bz", bz
    bx, by, bz = rotateYZ(bx, by, bz, Aguide)
    print "bx", bx
    print "by", by
    print "bz", bz
    rhoB, beta, gamma = rhoB_beta_gamma(bx, by, bz)
    print "beta", beta
    print "gamma", gamma

    dtype = cdouble if np.dtype(dtype) == rdouble else cfloat
    L = np.empty((len(rhoN),4), dtype=dtype)
    L[:,0] = rhoN + -1e-12j  # force some absorption to exist
    L[:,1] = beta
    L[:,2] = gamma
    L[:,3] = rhoB + 1j*dz
    return L

def rhoB_u1_u3(bx, by, bz):
    raise Exception("switched to beta/gamma repn")
    # precompute rhoB, u1, u3, independent of kz
    b_tot = sqrt(bx**2 + by**2 + bz**2)
    b_nz = (b_tot != 0)
    u1 = np.ones_like(b_tot, dtype='complex')
    u3 = -np.ones_like(b_tot, dtype='complex')
    u1[b_nz] = ( b_tot + bx + 1j*by - bz )[b_nz] / ( b_tot + bx - 1j*by + bz )[b_nz]
    u3[b_nz] = (-b_tot + bx + 1j*by - bz )[b_nz] / (-b_tot + bx - 1j*by + bz )[b_nz]
    rhoB = B2SLD * b_tot
    return rhoB, u1, u3

def rhoB_beta_gamma(bx, by, bz):
    # precompute rhoB, u1, u3, independent of kz
    b_tot = sqrt(bx**2 + by**2 + bz**2)
    b_nz = (b_tot != 0)
    u1 = np.ones_like(b_tot, dtype='complex')
    u3 = -np.ones_like(b_tot, dtype='complex')
    u1[b_nz] = ( b_tot + bx + 1j*by - bz )[b_nz] / ( b_tot + bx - 1j*by + bz )[b_nz]
    u3[b_nz] = (-b_tot + bx + 1j*by - bz )[b_nz] / (-b_tot + bx - 1j*by + bz )[b_nz]
    rhoB = B2SLD * b_tot
    #return rhoB, u1, u3

    Bn,Gn = u1, 1/u3
    #print "beta",Bn
    #print "gamma",Gn

    #b_reverse = b_nz & (abs(u1)<=1.0)
    #rhoB[b_reverse] = -rhoB[b_reverse]
    #Bn[b_reverse],Gn[b_reverse] = u3[b_reverse],1/u1[b_reverse]

    return rhoB, Bn, Gn


def rotateYZ(bx, by, bz, Aguide):
    # get B in lab frame, from sample frame
    C = cos(radians(Aguide))
    S = sin(radians(Aguide))
    bxr = bx
    byr = bz * S + by * C
    bzr = bz * C - by * S
    return bxr, byr, bzr

def plot(kz, R, **kw):
    import pylab
    rpp, rpm, rmp, rmm = R
    pylab.plot(2*kz, abs(rpp)**2, label="r++", **kw)
    pylab.plot(2*kz, abs(rmp)**2, label="r-+", hold=True, **kw)
    pylab.plot(2*kz, abs(rpm)**2, label="r+-", hold=True, **kw)
    pylab.plot(2*kz, abs(rmm)**2, label="r--", hold=True, **kw)
    pylab.yscale('log')
    pylab.legend()

# =============== Demo code ====================

def _afm_model():
    dz_mult = 5
    dz =   np.array([1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0, 1.0]) * dz_mult
    rhoN_mult = 2e-4
    rhoN = np.array([0.0, 2.0, 1.0,  2.0, 1.0, 2.0, 1.0, 0.0]) * rhoN_mult
    #rhoB_mult = 2e-4 / 2.31604654e-6
    #rhoB = array([0.0, 1.0, 0.0,  1.0, 0.0, 1.0, 0.0, 0.0]) * rhoB_mult
    by =   np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    bx =   np.array([0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0])
    bz =   np.array([0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0])
    Aguide = 270.
    return dz, rhoN, bx, by, bz, Aguide

def _Yaohua_model(H=0.4):
    """
    layers = [
        # depth rho rhoM thetaM phiM
        [   0, 0.0, rhoB,       90, 0.0],
        [ 200, 4.0, rhoB + 1.0, np.degrees(np.arctan2(rhoB, 1.0)), 0.0],
        [ 200, 2.0, rhoB + 1.0, 90, 0.0],
        [   0, 4.0, rhoB,       90, 0.0],
        ]
    """
    B2SLD = 2.31929 # *1e-6
    dz_mult = 200.0
    dz =   np.array([      1.0,           1.0,           1.0,       1.0]) * dz_mult
    rhoN_mult = 1e-6
    rhoN = np.array([      0.0,           4.0,           2.0,       4.0]) * rhoN_mult
    by =   np.array([        H,             H, (1.0/B2SLD)+H,         H])
    bx =   np.array([      0.0,     1.0/B2SLD,           0.0,       0.0])
    bz =   np.array([      0.0,           0.0,           0.0,       0.0])
    #Aguide = 270.
    Aguide = 300.
    return dz, rhoN, bx, by, bz, Aguide

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
    Aguide = np.random.rand()*360.
    #Aguide = 270
    return dz, rhoN, bx, by, bz, Aguide

def _run_model(model, kz, reversed=False, rotated=True, cpu=False, dtype='f'):
    import time
    dz, rhoN, bx, by, bz, Aguide = model
    if not rotated:
        Aguide = 0.0
        bx,by,bz = bz+EPSILON,bx,by
        #bx,by,bz = bz,bx,by
    if reversed:
        kz = -kz
        dz, rhoN, bx, by, bz = [v[::-1] for v in dz, rhoN, bx, by, bz]
    calculator = CPU_Calculator() if cpu else GPU_Calculator(dtype)
    dz, rhoN, bx, by, bz = [np.asarray(v) for v in (dz, rhoN, bx, by, bz)]
    rhoN = rhoN - 1e-12j
    #N=30
    N=1
    t0 = time.time()
    for i in range(N):
       R = calculator(kz, dz, rhoN, bx, by, bz, Aguide)
    print "done with time = %g" % ((time.time() - t0)/N)
    return R

def _refl1d_model(model, kz):
    from refl1d import reflmodule
    d, rho, bx, by, bz, Aguide = model
    rho = np.asarray(rho)
    irho = np.zeros(len(rho))
    sigma = np.zeros(len(rho)-1)

    print "Aguide", Aguide
    print "d", d
    print "rho", rho
    print "bx", bx
    print "by", by
    print "bz", bz
    bx, by, bz = rotateYZ(bx, by, bz, Aguide)
    print "bx", bx
    print "by", by
    print "bz", bz
    rhoB, beta, gamma = rhoB_beta_gamma(bx, by, bz)
    print "beta", beta
    print "gamma", gamma
    rho_index = np.zeros(kz.shape, 'i')
    R = [np.empty(len(kz), 'D') for pol in (1,2,3,4)]
    rho, irho, rhoB = [v*1e6 for v in rho, irho, rhoB]
    irho += 1e-6  # force absorption to exist
    reflmodule._magnetic_amplitude(d, sigma, rho, irho, rhoB, beta, gamma,
                                   Aguide, kz, rho_index, *R)
    return R


def demo():
    import pylab
    import sys
    rotated = "unrotated" not in sys.argv
    reversed = "reversed" in sys.argv
    cpu = "cpu" in sys.argv
    dtype = "d" if "double" in sys.argv else "f"
    #model = _afm_model()
    #model = _Yaohua_model(H=0.4)
    #model = _random_model(Nlayers=4000, H=0.4, seed=10248)
    model = _random_model(Nlayers=40, H=0.4, seed=None)
    #for p,v in zip("dz Nr bx by bz".split(), model): print p,v

    kz = np.linspace(0.001, 0.0301, 208)
    if 1:
        R = _run_model(model, kz, reversed=reversed, rotated=rotated, cpu=cpu, dtype=dtype)
        #print kz[0:3],R[:,0:3]
        plot(kz,R)
        #pylab.yscale('linear')

    if 1:
        R = _refl1d_model(model, kz)
        plot(kz, R, linestyle='dashed')

    #pylab.yscale('linear')
    pylab.show()


if __name__ == '__main__':
    demo()
