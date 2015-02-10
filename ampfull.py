#!/usr/bin/env python
import ctypes as ct

from numpy import zeros, sum, array, float32, float64
import numpy as np

import omf_loader

class CPU_Calculator:
    def __init__(self, dtype=float64):
        lib = np.ctypeslib.load_library('libga3.so', '.')
        lib.born_approximation.argtypes = [
            self.fn(Nx, Ny, Nz, dx, dy, dz, Form,
                    N_backing, N_fronting,
                    A[0], A[1], A[2],
                    B[0], B[1], B[2],
                    C[0], C[1], C[2],
                    Qx_points, Qx_min, Qx_max,
                    Qy_points, Qy_min, Qy_max,
                    Qz_points, Qz_min, Qz_max,
                    Nx, Ny, Nz,
                    R)
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            np.ctypeslib.ndpointer(dtype=float64),
            np.ctypeslib.ndpointer(dtype=ct.c_int),
            ct.c_double,
            ct.c_double,
            ct.c_double, ct.c_double, ct.c_double,
            ct.c_double, ct.c_double, ct.c_double,
            ct.c_double, ct.c_double, ct.c_double,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_double,
            ct.c_double,
            np.ctypeslib.ndpointer(dtype=np.complex128),
            np.ctypeslib.ndpointer(dtype=np.complex128),
            np.ctypeslib.ndpointer(dtype=np.complex128),
            np.ctypeslib.ndpointer(dtype=np.complex128)]
        self.fn = lib.born_approximation

    def __call__(self, Nsteps, steps, Form,
                 N_backing, N_fronting, ABC,
                 Qx_points = 128, Qx_min = -0.005, Qx_max = 0.005,
                 Qy_points = 128, Qy_min = -0.005, Qy_max = 0.005,
                 Qz_points = 1, Qz_min = 0.00, Qz_max = 0.00):

        Nform = Form.shape[0]
        Nx, Ny, Nz = Nsteps
        dx, dy, dz = steps
        A, B, C = ABC

        # here we're creating the empty matrices that will be filled by the C++ function
        R = np.empty([Qx_points, Qy_points, Qz_points, 4], dtype=np.complex128)

        # and we're calling the C++ library...
        self.fn(Nx, Ny, Nz, dx, dy, dz, Form,
                N_backing, N_fronting,
                A[0], A[1], A[2],
                B[0], B[1], B[2],
                C[0], C[1], C[2],
                Qx_points, Qx_min, Qx_max,
                Qy_points, Qy_min, Qy_max,
                Qz_points, Qz_min, Qz_max,
                Nx, Ny, Nz,
                R)

        return { 'pp':R[...,0], 'mm':R[...,3],
                 'mp':R[...,2], 'pm':R[...,1] }
class GPU_Calculator:
    def __init__(self, dtype=float64):
        import pyopencl as cl
        from cl_util import get_env
        env = get_env()
        self.queue = env.queues[0]
        self.T = env.dtype(dtype)

        # Compile the scattering kernel
        source = open('ba.c').read()
        program = env.compile_program('scattering', source, self.T.real)
        self.kernel = program.born_approximation
        self.Nq = 0
        self.Nform = 0
        self.gamp = []
        self.gform = []

    def make_Q_buffers(self, Nqx, Nqy, Nqz):
        """
        Allocate space for the returned I(Q) vectors.

        If the requested space is not equal to the currently
        allocated space, then the buffers are reallocated.  This allows the
        buffers to be used within a fitting context without reallocating
        buffers each time.

        This should work for simultaneous fitting as well, with the buffer
        size eventually becoming the size of the largest buffer.
        """
        Nq = Nqx*Nqy*Nqz
        if self.Nq < Nq:
            import pyopencl as cl
            from cl_util import get_warp
            T = self.T
            mf = cl.mem_flags
            _ = [b.release() for b in self.gamp]

            # Declare arrays on the gpu
            context = self.queue.context
            self.gamp = cl.Buffer(context, mf.WRITE_ONLY, T.complex_size*Nq*4)

            Nwarp = get_warp(self.kernel, self.queue)
            self.kernel_size = [Nwarp * (Nq//Nwarp)]
            self.Nq = Nq

    def make_form_buffers(self, Nx, Ny, Nz):
        """
        Allocate space for a fully occupied unit cell.

        If the requested space is less than the currently allocated
        space, then the buffers are reallocated.  This allows the calculator
        to be used within a fitting context without reallocating buffers
        each time.

        This should work for simultaneous fitting as well, with the form
        size eventually becoming the size of the largest form.
        """
        Nform = Nx*Ny*Nz
        if self.Nform < Nform:
            import pyopencl as cl
            from cl_util import get_warp
            T = self.T
            mf = cl.mem_flags
            [b.release() for b in self.gform]

            # Declare arrays on the gpu
            context = self.queue.context
            gForm = cl.Buffer(context, mf.READ_ONLY, T.real_size*6*Nform)
            self.gform = [gForm]

    def __call__(self, Nsteps, steps, Form,
                 N_backing, N_fronting, ABC,
                 Qx_points = 128, Qx_min = -0.005, Qx_max = 0.005,
                 Qy_points = 128, Qy_min = -0.005, Qy_max = 0.005,
                 Qz_points = 1, Qz_min = 0.00, Qz_max = 0.00):

        Nform = Form.shape[0]
        Nx, Ny, Nz = Nsteps
        dx, dy, dz = steps
        A, B, C = ABC

        import pyopencl as cl
        T = self.T

        # Allocate GPU memory and copy inputs
        if 0: # Don't understand why this is failing --- instead allocate each time
            self.make_form_buffers(Nx, Ny, Nz)
            cl.enqueue_copy(self.queue, self.gform, Form)
        else:
            from pyopencl import mem_flags as mf
            context = self.queue.context
            [b.release() for b in self.gform]
            self.gform =  cl.Buffer(
                context, mf.READ_ONLY|mf.COPY_HOST_PTR,
                hostbuf=np.ascontiguousarray(Form,T.real))
        # Allocate space for outputs
        self.make_Q_buffers(Qx_points, Qy_points, Qz_points)

        # Prepare the kernel arguments
        args = [
            T.int(Nx), T.int(Ny), T.int(Nz),
            T.real(dx), T.real(dy), T.real(dz),
            T.real(N_backing), T.real(N_fronting),
            self.gform,
            T.real(A[0]), T.real(A[1]), T.real(A[2]),
            T.real(B[0]), T.real(B[1]), T.real(B[2]),
            T.real(C[0]), T.real(C[1]), T.real(C[2]),
            T.int(Nform),
            T.int(Qx_points), T.real(Qx_min), T.real(Qx_max),
            T.int(Qy_points), T.real(Qy_min), T.real(Qy_max),
            T.int(Qz_points), T.real(Qz_min), T.real(Qz_max),
            self.gamp,
        ]

        # call the kernel
        self.kernel(self.queue, self.kernel_size, None, *args)


        # Retrieve memory from the card
        R = np.empty((Qx_points, Qy_points, Qz_points, 4), dtype=self.T.complex)
        cl.enqueue_copy(self.queue, R, self.gamp)

        return { 'pp':R[...,0], 'mm':R[...,3],
                 'mp':R[...,2], 'pm':R[...,1] }

def plot(amp, extent):
    pp,pm,mp,mm = [amp[k] for k in ('pp','pm','mp','mm')]
    from pylab import imshow, colorbar, title, xlabel, ylabel, show, figure
    vmin = (abs(pm[np.isfinite(pm)])**2).min()
    vmax = (abs(mm[np.isfinite(mm)])**2).max()
    vmin = np.log10(vmin)
    vmax = np.log10(vmax)
    figure()
    imshow(np.log10(abs(pp[:,:,0])**2).T, aspect='equal', interpolation='nearest', extent = extent, vmin=vmin, vmax=vmax, origin='lower')
    title("Plus-Plus")
    xlabel("Qx")
    ylabel("Qy")
    colorbar()
    figure()
    imshow(np.log10(abs(mm[:,:,0])**2).T, aspect='equal', interpolation='nearest', extent = extent, vmin=vmin, vmax=vmax, origin='lower')
    title("Minus-Minus")
    xlabel("Qx")
    ylabel("Qy")
    colorbar()
    figure()
    imshow(np.log10(abs(pm[:,:,0])**2).T, aspect='equal', interpolation='nearest', extent = extent, vmin=vmin, vmax=vmax, origin='lower')
    title("Plus-Minus")
    xlabel("Qx")
    ylabel("Qy")
    colorbar()
    figure()
    imshow(np.log10(abs(mp[:,:,0])**2).T, aspect='equal', interpolation='nearest', extent = extent, vmin=vmin, vmax=vmax, origin='lower')
    title("Minus-Plus")
    xlabel("Qx")
    ylabel("Qy")
    colorbar()

#            figure()
#            imshow(ff.mx[:,:,2], interpolation='nearest')
#            title("structure")
#            show()


def formFromOMF(omf_mag_filename, omf_nuc_filename):
    # requires one OOMMF simulation result (Magnetization saved)
    # as well as an identical .omf file where M=rhoNuclear
    # this loads the OOMMF simulation result into the ff object
    ff = omf_loader.Omf(omf_mag_filename)
    fn = omf_loader.Omf(omf_nuc_filename)
    print "fn.shape",fn.M.shape, np.prod(fn.M.shape), np.prod(ff.M.shape)
    # TODO: must include zi == 0/z_last_index if N_fronting/N_backing != 0
    Form = zeros((sum(nonempty_mask), 6), dtype=float64, order="C")
    fidx = np.indices(fn.M.shape, dtype=ct.c_int)

    Form[:,0] = (fn.M != 0.0) + (ff.M != 0); // 0, 1 or 2
    Form[:,1] = (ff.mx * ff.rhoM) # mx, my, mz are unit vectors
    Form[:,2] = (ff.my * ff.rhoM)
    Form[:,3] = (ff.mz * ff.rhoM)
    Form[:,4] = fn.M
    Form[:,5] = 0.0 # imaginary rho

    # convert distances to Angstroms
    dx = float(ff.parameters['xstepsize']) * 1e10
    dy = float(ff.parameters['ystepsize']) * 1e10
    dz = float(ff.parameters['zstepsize']) * 1e10

    Nx = int(mag.parameters['xnodes']) - 1
    Ny = int(mag.parameters['ynodes']) - 1
    Nz = int(mag.parameters['znodes']) - 1

    return Form, [Nx,Ny,Nz], [dx, dy, dz]


def demo():
    import sys
    import time

    version = sys.argv[1] if len(sys.argv) > 1 else 'gpu'
    dtype = float32 if version.endswith('f') else float64
    if version.startswith('cpu'):
        calc = CPU_Calculator(dtype=dtype)
    else:
        calc = GPU_Calculator(dtype=dtype)
    #SLD_N = 9.06e-6 #
    #SLD_M = 2.46e-6
    
    Qx_min = -0.002
    Qx_max = 0.002

    Qy_min = -0.002
    Qy_max = 0.002

    Qz_min = 0.00002
    Qz_max = 0.04

    Qx_points, Qy_points, Qz_points = 16,8,1
    Qx_points, Qy_points, Qz_points = 32,32,1
    #Qx_points, Qy_points, Qz_points = 64,64,1
    #Qx_points, Qy_points, Qz_points = 128,128,1
    #Qx_points, Qy_points, Qz_points = 256,256,1

    #phi = 0.0 #0 deg
    #A = array([-sin(phi),cos(phi),0], dtype=float64)
    #B = array([0,0,1], dtype=float64) # z
    #C = array([cos(phi),sin(phi),0], dtype=float64)
    #A = array([0,0,1], dtype=float64) # quantization is along z
    #B = array([1,0,0], dtype=float64)
    #C = array([0,1,0], dtype=float64)   
    A = array([0,1,0], dtype=float64) # quantization is along y
    B = array([0,0,1], dtype=float64)
    C = array([1,0,0], dtype=float64)
    
    mag = omf_loader.Omf('base/askyrmion-Oxs_MinDriver-Magnetization-13-0024977.omf')
    nuc = omf_loader.Omf('base/askyrmion_nuc-Oxs_MinDriver-Magnetization-00-0000000.omf')
    rho_substrate = 0.0 #2.069e-6 # sld
    rho_fronting = 0.0 # vacuum
    form, Nsteps, steps= formFromOMF('base/askyrmion_irrad_8-Oxs_MinDriver-Magnetization-13-0024834.omf',
                                     'base/askyrmion_nuc-Oxs_MinDriver-Magnetization-00-0000000.omf')
    t0 = time.time()
    remn_scatt = calc(Nsteps, steps, form,
                      rho_substrate, rho_fronting, [A, B, C],
                      Qx_points, Qx_min, Qx_max,
                      Qy_points, Qy_min, Qy_max,
                      Qz_points, Qz_min, Qz_max)
    print "done with time = %g" % (time.time() - t0)
    try:
        import pylab
    except:
        print "no matplotlib - won't make figures"
        return
    plot(remn_scatt, extent = [Qx_min, Qx_max, Qy_min, Qy_max])
    pylab.show()


if __name__ == '__main__':
    demo()
