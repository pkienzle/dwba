import numpy as np
from numpy import array, exp, vstack, cumsum

from .refl import calculateU, calculateS, calculateB, calculate_Rsam

def calculateDWBA(dz, sld_z, ft, ki, kf):
   # inplane FT should have shape (numlayers, 4) where
   # in each layer it is (FT++, FT+-, FT-+, FT--)
   rhoB,U1,U3 = calculateU(bx, by, bz, AGUIDE)
   S1_in, S3_in =  calculateS(kz_in, rhoN, rhoB, polarization=pol_in)
   S1_out, S3_out =  calculateS(kz_out, rhoN, rhoB, polarization=pol_out)
   B_in  = calculateB(dz, S1_in, S3_in, U1, U3)
   B_out  = calculateB(dz, S1_out, S3_out, U1, U3)
   r_in = calculate_Rsam(kz_in, B_in)
   r_out = calculate_Rsam(kz_out, B_out)
   inplane_xs_index = 0
   if pol_in:
       C_0_in = array([[1.0], [r_in[0]], [0.0], [r_in[1]]])
   else:
       C_0_in = array([[0.0], [r_in[2]], [1.0], [r_in[3]]])
       inplane_xs_index += 2
   if pol_out:
       C_0_out = array([[1.0], [r_out[0]], [0.0], [r_out[1]]])
   else:
       C_0_out = array([[0.0], [r_out[2]], [1.0], [r_out[3]]])
       inplane_xs_index += 1
       # pick the inplane FT corresponding to plus_in and plus_out
   inplane_xs = inplane_FT[:, inplane_xs_index]
   z = cumsum(dz) - dz[0] # remove thickness of first layer, which is fronting
   S_in = vstack([S1_in, -S1_in, S3_in, -S3_in])
   S_out = vstack([S1_out, -S1_out, S3_out, -S3_out])
   dwba = 0.0 + 0j
   for layer in range(len(rhoN)):
       C_l_in = B_in[layer] * C_0_in
       C_l_out = B_out[layer] * C_0_out
       S_l_in = S_in[layer]
       S_l_out = S_out[layer]
       dwba += sum( (Ci*Co)/(Si+So) * (exp((Si+So)*z[layer+1]) - exp((Si+So)*z[layer])) * inplane_xs
                    for Ci,Si in zip(C_l_in,S_l_in)
                    for Co,So in zip(C_l_out,S_l_out) )
   return dwba


def dwba(x, y, z, sld, surround, H,  A, B, C, kin, kout):
    """
    Sample description:

    x,y,z are the boundaries of a rectilinear grid of sample points
    sld = (complex nuclear, mx, my, mz) are the constant SLDs within each voxel

    Sample environment:

    surround = (sld, sld) are the SLDs of the surround
    H = (Hx,Hy,Hz) is the applied field
    A, B, C = (vector, vector, vector) is the quantization axis

    Measured points:

    kin, kout = (kx, ky, kz, spin) is the list of points at which the scattering is to be computed
    """

    sld = add_H(sld, H)
    surround = add_H(surround, H)
    sld_z = add_surround(planar_average(sld), surround)
    dz = np.hstack([0, np.diff(z), 0])
    sld = sld - sld_z[1:-1]

    result = []
    for ki,kf in zip(kin,kout):
        ft = planar_transform(x, y, z, sld, A, B, C, kf[:3]-ki[:3])
        result.append(calculateDWBA(dz, sld_z, ft, ki, kf))

    return np.array(result)
