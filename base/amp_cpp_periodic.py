#!/usr/bin/env python

from numpy import zeros, abs, sum, linspace, empty, empty_like, array, sin, cos, float64
import omf_loader
#from SimulateGeneralScattering import GeneralScattering
import time

import numpy

''' Setup for this: need to compile C++ file as library.
On Windows, this would be a .dll file that you would have to keep 
in the same directory as the python file you're running 

These are the steps to compile in linux: (i'm using lgs3.o as intermediate compiled file)

g++ -c -lm -Ofast -fopenmp 6_SimulateGeneralAmplitude.cpp -o lga3.o
ar rcs libga3.a lga3.o
g++ -c -fPIC -lm -Ofast -fopenmp 6_SimulateGeneralAmplitude.cpp -o lga3.o
g++ -shared -lm -Ofast -fopenmp -Wl,-soname,libga3.so.1 -o libga3.so.1.0.1 lga3.o
ln -s libga3.so.1.0.1 libga3.so
'''

import numpy as nm
import ctypes as ct

def recompile_lib():
    import os
    os.system('g++ -c -lm -O3 -fopenmp 9_SimulateGeneralAmplitude_ordered_periodic.cpp -o lga3p.o')
    os.system('ar rcs libga3p.a lga3p.o')
    os.system('g++ -c -fPIC -lm -O3 -fopenmp 9_SimulateGeneralAmplitude_ordered_periodic.cpp -o lga3p.o')
    os.system('g++ -shared -lm -O3 -fopenmp -Wl,-soname,libga3p.so.1 -o libga3p.so.1.0.1 lga3p.o')
    os.system('ln -s libga3p.so.1.0.1 libga3p.so')

"""                    double *Form,
                       int *Form_indices,
                       double dx,
                       double dy,
                       double dz,
                       double *A,
                       double *B,
                       double *C,
                       int FormFactor_ScatteringCenters,
                       int Qx_points,
                       double Qx_min,
                       double Qx_max,
                       int Qy_points,
                       double Qy_min,
                       double Qy_max,
                       int Qz_points,
                       double Qz_min, 
                       double Qz_max, 
                       int last_x_index,
                       int last_y_index,
                       int last_z_index,
                       double N_backing,
                       double N_fronting,
                       dcmplx *Amplitude_PP, 
                       dcmplx *Amplitude_MM, 
                       dcmplx *Amplitude_MP, 
                       dcmplx *Amplitude_PM);
"""

argtypes = [nm.ctypeslib.ndpointer(dtype=nm.double),\
                                     nm.ctypeslib.ndpointer(dtype=ct.c_int),\
                                     ct.c_double,\
                                     ct.c_double,\
                                     ct.c_double,\
                                     nm.ctypeslib.ndpointer(dtype=nm.double),\
                                     nm.ctypeslib.ndpointer(dtype=nm.double),\
                                     nm.ctypeslib.ndpointer(dtype=nm.double),\
                                     ct.c_int,\
                                     ct.c_int,\
                                     ct.c_double,\
                                     ct.c_double,\
                                     ct.c_int,\
                                     ct.c_double,\
                                     ct.c_double,\
                                     ct.c_int,\
                                     ct.c_double,\
                                     ct.c_double,\
                                     ct.c_int,\
                                     ct.c_int,\
                                     ct.c_int,\
                                     ct.c_double,\
                                     ct.c_double,\
                                     nm.ctypeslib.ndpointer(dtype=nm.complex),\
                                     nm.ctypeslib.ndpointer(dtype=nm.complex),\
                                     nm.ctypeslib.ndpointer(dtype=nm.complex),\
                                     nm.ctypeslib.ndpointer(dtype=nm.complex)]


    
def formFromOMF(omf_mag_filename, omf_nuc_filename):
    # requires one OOMMF simulation result (Magnetization saved)
    # as well as an identical .omf file where M=rhoNuclear
    # this loads the OOMMF simulation result into the ff object
    ff = omf_loader.Omf(omf_mag_filename)
    fn = omf_loader.Omf(omf_nuc_filename)
    nonempty_mask = numpy.logical_or(abs(fn.M) > 0, abs(ff.M) > 0)
    FormFactor_ScatteringCenters = sum(nonempty_mask)
    Form = zeros((sum(nonempty_mask), 7), dtype=nm.double, order="C")
    Form_indices = zeros((FormFactor_ScatteringCenters, 3), dtype=ct.c_int, order="C")
    fidx = nm.indices(fn.M.shape, dtype=ct.c_int)

    Form[:,0] = ff.node_x[nonempty_mask] * 1e10 # convert m to Angstroms
    Form[:,1] = ff.node_y[nonempty_mask] * 1e10
    Form[:,2] = ff.node_z[nonempty_mask] * 1e10
    Form[:,3] = fn.M[nonempty_mask]
    Form[:,4] = (ff.mx * ff.rhoM)[nonempty_mask] # mx, my, mz are unit vectors
    Form[:,5] = (ff.my * ff.rhoM)[nonempty_mask]
    Form[:,6] = (ff.mz * ff.rhoM)[nonempty_mask]
    
    Form_indices[:,0] = fidx[0][nonempty_mask] # x index
    Form_indices[:,1] = fidx[1][nonempty_mask] # y
    Form_indices[:,2] = fidx[2][nonempty_mask] # z
    
    dx = float(ff.parameters['xstepsize']) * 1e10
    dy = float(ff.parameters['ystepsize']) * 1e10
    dz = float(ff.parameters['zstepsize']) * 1e10
    
    return Form, Form_indices, [dx, dy, dz]
    
def calc_scattering(Form, Form_indices, stepsizes,
                    A, B, C,
                    Qx_points = 128, Qx_min = -0.005, Qx_max = 0.005,
                    Qy_points = 128, Qy_min = -0.005, Qy_max = 0.005, 
                    Qz_points = 1, Qz_min = 0.00, Qz_max = 0.00,
                    last_x_index = 1, last_y_index = 1, last_z_index = 1,
                    N_backing=0.0, N_fronting=0.0, 
                    plot_results=True):
                    
    FormFactor_ScatteringCenters = Form.shape[0]
    dx, dy, dz = stepsizes
    
    # this loads up the compiled library: will be .dll in Windows, not .so
    _libGeneralScattering = nm.ctypeslib.load_library('libga3p.so', '.')
    _libGeneralScattering.GeneralScattering_xfirst.argtypes = argtypes
    _libGeneralScattering.GeneralScattering_yfirst.argtypes = argtypes
    _libGeneralScattering.GeneralScattering_zfirst.argtypes = argtypes

    _libGeneralScattering.GeneralScattering_xfirst.restype  = ct.c_void_p
    _libGeneralScattering.GeneralScattering_yfirst.restype  = ct.c_void_p
    _libGeneralScattering.GeneralScattering_zfirst.restype  = ct.c_void_p

    #t0 = time.time()
    #pp, mm, mp, pm = GeneralScattering(Form, Qx_points, Qx_min, Qx_max, Qy_points, Qy_min, Qy_max, Qz_points, Qz_min, Qz_max)
    #print "done with Python version, time = %g" % (time.time() - t0)

    # here we're creating the empty matrices that will be filled by the C++ function
    Amplitude_PP = zeros((Qx_points, Qy_points, Qz_points), dtype=nm.complex128)
    Amplitude_MM = zeros((Qx_points, Qy_points, Qz_points), dtype=nm.complex128)
    Amplitude_MP = zeros((Qx_points, Qy_points, Qz_points), dtype=nm.complex128)
    Amplitude_PM = zeros((Qx_points, Qy_points, Qz_points), dtype=nm.complex128)
    
    # and we're calling the C++ library...
    t0 = time.time()
    libs = [
        _libGeneralScattering.GeneralScattering_xfirst,
        _libGeneralScattering.GeneralScattering_yfirst,
        _libGeneralScattering.GeneralScattering_zfirst
    ]
    lib_chooser = [Qx_points, Qy_points, Qz_points]
    lib_index = lib_chooser.index(max(lib_chooser))
    active_lib = libs[lib_index]
    print "using lib %d" % (lib_index,)
    active_lib(Form,\
        Form_indices,\
        dx, dy, dz,\
        A, B, C, \
        FormFactor_ScatteringCenters,\
        nm.intc(Qx_points),\
        nm.double(Qx_min),\
        nm.double(Qx_max),\
        nm.intc(Qy_points),\
        nm.double(Qy_min),\
        nm.double(Qy_max),\
        nm.intc(Qz_points),\
        nm.double(Qz_min),\
        nm.double(Qz_max),\
        nm.intc(last_x_index),\
        nm.intc(last_y_index),\
        nm.intc(last_z_index),\
        nm.double(N_backing),\
        nm.double(N_fronting),\
        Amplitude_PP, Amplitude_MM, Amplitude_MP, Amplitude_PM)
                                            
    
    print "done with C++ version, time = %g" % (time.time() - t0)
    
    if plot_results:
        try: 
            from pylab import imshow, colorbar, title, xlabel, ylabel, show, figure
            vmin = (abs(Amplitude_PM[nm.isfinite(Amplitude_PM)])**2).min()
            vmax = (abs(Amplitude_MM[nm.isfinite(Amplitude_MM)])**2).max()
            vmin = nm.log10(vmin)
            vmax = nm.log10(vmax)
            extent = [Qx_min, Qx_max, Qy_min, Qy_max]
            figure()
            imshow(nm.log10(abs(Amplitude_PP[:,:,0])**2).T, aspect='equal', interpolation='nearest', extent = extent, vmin=vmin, vmax=vmax, origin='lower')
            title("Plus-Plus")
            xlabel("Qx")
            ylabel("Qy")
            colorbar()
            figure()
            imshow(nm.log10(abs(Amplitude_MM[:,:,0])**2).T, aspect='equal', interpolation='nearest', extent = extent, vmin=vmin, vmax=vmax, origin='lower')
            title("Minus-Minus")
            xlabel("Qx")
            ylabel("Qy")
            colorbar()
            figure()
            imshow(nm.log10(abs(Amplitude_PM[:,:,0])**2).T, aspect='equal', interpolation='nearest', extent = extent, vmin=vmin, vmax=vmax, origin='lower')
            title("Plus-Minus")
            xlabel("Qx")
            ylabel("Qy")
            colorbar()
            figure()
            imshow(nm.log10(abs(Amplitude_MP[:,:,0])**2).T, aspect='equal', interpolation='nearest', extent = extent, vmin=vmin, vmax=vmax, origin='lower')
            title("Minus-Plus")
            xlabel("Qx")
            ylabel("Qy")
            colorbar()
            
            show()
            
#            figure()
#            imshow(ff.mx[:,:,2], interpolation='nearest')
#            title("structure")
#            show()
            
        except:
            print "no matplotlib - won't make figures"
            
    return {'pp': Amplitude_PP, 'mm': Amplitude_MM, 'mp': Amplitude_MP, 'pm':Amplitude_PM}
            
if __name__ == '__main__':
    #from SimulateGeneralScattering import GeneralScatteringAmp
    from pylab import figure, imshow
    #SLD_N = 9.06e-6 # 
    #SLD_M = 2.46e-6
    
    Qx_points = 32
    Qx_min = -0.002
    Qx_max = 0.002

    Qy_points = 32
    Qy_min = -0.002
    Qy_max = 0.002

    Qz_points = 1
    Qz_min = 0.00002
    Qz_max = 0.04

    Qx_points, Qy_points = 32,32
    
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
    
    mag = omf_loader.Omf('askyrmion-Oxs_MinDriver-Magnetization-13-0024977.omf')
    nuc = omf_loader.Omf('askyrmion_nuc-Oxs_MinDriver-Magnetization-00-0000000.omf')
    last_x_index = int(mag.parameters['xnodes']) - 1
    last_y_index = int(mag.parameters['ynodes']) - 1
    last_z_index = int(mag.parameters['znodes']) - 1
    rho_substrate = 0.0 #2.069e-6 # sld
    rho_fronting = 0.0 # vacuum
    form, form_indices, stepsizes = formFromOMF('askyrmion_irrad_8-Oxs_MinDriver-Magnetization-13-0024834.omf', 'askyrmion_nuc-Oxs_MinDriver-Magnetization-00-0000000.omf')
    remn_scatt = calc_scattering(form, form_indices, stepsizes, A, B, C, Qx_points, Qx_min, Qx_max, Qy_points, Qy_min, Qy_max, Qz_points, Qz_min, Qz_max, last_x_index, last_y_index, last_z_index, rho_substrate, rho_fronting)
    

