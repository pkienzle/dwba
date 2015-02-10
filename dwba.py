import numpy as np
from numpy import array, exp, sqrt, vstack, cumsum

from .refl import calculateU, calculateS, calculateB, calculate_Rsam

B2SLD = 2.31604654e-6


def calculateU(bx, by, bz):
    # precompute rhoB, u1, u3, independent of kz
    b_tot = sqrt(bx**2 + by**2 + bz**2)
    b_nz = (b_tot != 0)
    u1 = np.ones_like(b_tot, dtype='complex')
    u3 = -np.ones_like(b_tot, dtype='complex')
    u1[b_nz] = ( b_tot + bx + 1j*by - bz )[b_nz] / ( b_tot + bx - 1j*by + bz )[b_nz]
    u3[b_nz] = (-b_tot + bx + 1j*by - bz )[b_nz] / (-b_tot + bx - 1j*by + bz )[b_nz]
    b_reverse = b_nz & (abs(u1)<=1.0)

    rhoB = B2SLD * b_tot
    rhoB[b_reverse] = -rhoB[b_reverse]
    Bn,Gn = u1, 1/u3
    Bn[b_reverse],Gn[b_reverse] = u3[b_reverse],1/u1[b_reverse]

    return rhoB, Bn,Gn

def distorted_wave(kz, dz, rhoN, rhoB, U1, U3, spin):
    S1, S3 = calculateS(kz, rhoN, rhoB, spin)
    B  = calculateB(dz, S1, S3, U1, U3)
    R = calculate_Rsam(kz, B)
    if spin == 1.0:
        C0 = array([[1.0], [R[0]], [0.0], [R[1]]])
    else:
        C0 = array([[0.0], [R[2]], [1.0], [R[3]]])
    S = vstack([S1, -S1, S3, -S3])

    return S, B, C0

def distorted_wave_ba(z, dwi, dwf, ba):
    Si,Bi,CiL = dwi
    Sf,Bf,CfL = dwf
    dwba = 0.0 + 0j
    for L in range(len(z)):
        CiL = np.dot(Bi[L], CiL)
        CfL = np.dot(Bf[L], CfL)
        dwba += sum( (CiLk*CfLk)/(SiLk+SfLk) * (exp((SiLk+SfLk)*z[L+1]) - exp((SiLk+SfLk)*z[L])) * ba[L]
                     for CiLk,SiLk in zip(CiL,Si[L])
                     for CfLk,SfLk in zip(CfL,Sf[L]) )
    return dwba


def calculateDWBA(dz, sld_z, ft, ki, kf, ba):
   # inplane FT should have shape (numlayers, 4) where
   # in each layer it is (FT++, FT+-, FT-+, FT--)
   rhoN, irhoN, bx, by, bz = sld_z
   rhoN = rhoN + 1j*irhoN
   rhoB,U1,U3 = calculateU(bx, by, bz, AGUIDE)

   dwip = distorted_wave(ki[2], dz, rhoN, rhoB, U1, U3, +1.0)
   dwim = distorted_wave(ki[2], dz, rhoN, rhoB, U1, U3, -1.0)
   dwfp = distorted_wave(kf[2], dz, rhoN, rhoB, U1, U3, +1.0)
   dwfm = distorted_wave(kf[2], dz, rhoN, rhoB, U1, U3, -1.0)

   z = cumsum(dz[:-1]) - dz[0] # remove thickness of first layer, which is fronting
   return [distorted_wave_ba(z, dwi, dwf, baxs)
           for dwi,dwf,baxs in (
           (dwip, dwfp, ba[0]),
           (dwip, dwfm, ba[1]),
           (dwim, dwfp, ba[2]),
           (dwim, dwfm, ba[3]))]

def dwba(x, y, z, sld, surround, H, ABC, kin, kout, Aguide):
    """
    Sample description:

    x,y,z are the boundaries of a rectilinear grid of sample voxels
    sld = (rho, irho, mx, my, mz) are the SLDs within each voxel

    Sample environment:

    surround = (sld, sld) are the SLDs of the surround
    H = (Hx,Hy,Hz) is the applied field
    ABC = [vector, vector, vector] is the quantization axis

    Measured points:

    kin, kout = (kx, ky, kz, spin) is the list of points at which the scattering is to be computed
    """

    sld = add_H(sld, H)
    surround = add_H(surround, H)
    sld_z = add_surround(planar_average(sld), surround)
    dz = np.hstack([0, np.diff(z), 0])
    sld = sld - sld_z[1:-1]  # remove DC offset for fourier transform

    result = []
    for ki,kf in zip(kin,kout):
        ft = calculateBA(x, y, z, sld, ABC, kf-ki)
        result.append(calculateDWBA(dz, sld_z, ft, ki, kf))

    return np.array(result)
