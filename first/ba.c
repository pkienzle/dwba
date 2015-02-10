//Written by K. Krycka and B. Maranville
#ifndef __OPENCL_VERSION__
#include "cl_fake.h"
#endif

// OS/X OpenCL driver wants a prototype
void scatt_kernel(
    global double *Form, global int *Form_indices,
    int FormFactor_ScatteringCenters,
    double xmax, double ymax, double zmax,
    double dx, double dy, double dz,
    double Qx, double Qy, double Qz,
    int last_x_index, int last_y_index, int last_z_index,
    double N_backing, double N_fronting,
    cdouble *N, cdouble *HJX, cdouble *HJY, cdouble *HJZ);
void scatt_kernel(
    global double *Form, global int *Form_indices,
    int FormFactor_ScatteringCenters,
    double xmax, double ymax, double zmax,
    double dx, double dy, double dz,
    double Qx, double Qy, double Qz,
    int last_x_index, int last_y_index, int last_z_index,
    double N_backing, double N_fronting,
    cdouble *N, cdouble *HJX, cdouble *HJY, cdouble *HJZ)
{
    // Note: don't need cmul when multiplying by I
    const cdouble laue_x = cexp(I * Qx * dx) - cplx(1.0,0.0); // correction factor not including I*Qx
    const cdouble laue_y = cexp(I * Qy * dy) - cplx(1.0,0.0);
    const cdouble laue_z = cexp(I * Qz * dz) - cplx(1.0,0.0);

    const double Qx_Sqr = Qx*Qx;
    const double Qy_Sqr = Qy*Qy;
    const double Qz_Sqr = Qz*Qz;
    const double Q_Sqr = Qx_Sqr + Qy_Sqr + Qz_Sqr;

    const cdouble x_int_factor = (Qx != 0.0 ? cmul(laue_x, (I / -Qx)) : dx);
    const cdouble y_int_factor = (Qy != 0.0 ? cmul(laue_y, (I / -Qy)) : dy);
    const cdouble z_int_factor = (Qz != 0.0 ? cmul(laue_z, (I / -Qz)) : dz);
    const cdouble laue_factor = cmul(x_int_factor, cmul(y_int_factor, z_int_factor));

    cdouble Amp_N = 0.0;
    cdouble Amp_Mx = 0.0;
    cdouble Amp_My = 0.0;
    cdouble Amp_Mz = 0.0;

    for(int c=0; c<FormFactor_ScatteringCenters; c++){
        const int co = c*8;
        const double x = Form[co];
        const double y = Form[co+1];
        const double z = Form[co+2];
        const double rMx = Form[co+3];
        const double rMy = Form[co+4];
        const double rMz = Form[co+5];
        const cdouble rN = cplx(Form[co+6], Form[co+7]);

        // Computing p = eigxyz + x_inv_l + y_inv_l
        // Later, compute Amp_# += r# * (eigxyz + x_inv_l + y_inv_l)
        const double qxyz = Qx*x + Qy*y + Qz*z;
        cdouble p = cplx(cos(qxyz), sin(qxyz));
        //cdouble p = cpolar(1.0, qxyz);

#if 1
        double edge_scale;
        const int ci=c*4;
        const int xi = Form_indices[ci];
        const int yi = Form_indices[ci+1];
        const int zi = Form_indices[ci+2];

        // Add fronting and backing media; requires a complete layer in x,y if
        // the rho for the medium is not zero.
        // Note: done before x/y since uses p = eigxyz for original xyz
        edge_scale = (zi==0)*N_backing + (zi==last_z_index)*(-N_fronting);
        if (edge_scale != 0.0)
        {
            Amp_N += (Qz != 0.0 ? cdiv((edge_scale*Qz_Sqr)*p, Q_Sqr*laue_z) : 0.0);
        }

        // Handle edges in x and y
        edge_scale = (xi==0)*-1.0 + (xi==last_x_index)*1.0;
        if (edge_scale != 0.0)
        {
            const double opposite_edge = (xi==0)*xmax + (xi==last_x_index)*0.0;
            const double qxyz = Qx*opposite_edge + Qy*y + Qz*z;
            //const cdouble eiqxyz = cpolar(1.0, qxyz);
            const cdouble eiqxyz = cplx(cos(qxyz), sin(qxyz));
            p += (Qx != 0.0 ? cdiv((edge_scale*Qx_Sqr)*eiqxyz, Q_Sqr*laue_x) : 0.0);
        }

        edge_scale = (yi==0)*-1.0 + (yi==last_y_index)*1.0;
        if (edge_scale != 0.0)
        {
            const double opposite_edge = (yi==0)*ymax + (yi==last_y_index)*0.0;
            const double qxyz = Qx*x + Qy*opposite_edge + Qz*z;
            //const cdouble eiqxyz = cpolar(1.0, qxyz);
            const cdouble eiqxyz = cplx(cos(qxyz), sin(qxyz));
            p += (Qy != 0.0 ? cdiv((edge_scale*Qy_Sqr)*eiqxyz, Q_Sqr*laue_y) : 0.0);
        }
#endif

        Amp_N += cmul(rN, p);
        Amp_Mx += rMx * p;
        Amp_My += rMy * p;
        Amp_Mz += rMz * p;

    }//end c loop

    Amp_N = cmul(Amp_N, laue_factor);
    Amp_Mx = cmul(Amp_Mx, laue_factor);
    Amp_My = cmul(Amp_My, laue_factor);
    Amp_Mz = cmul(Amp_Mz, laue_factor);

    *N = Amp_N;
    *HJX = (Amp_Mx*(Q_Sqr-Qx_Sqr) - Amp_My*(Qy*Qx) - Amp_Mz*(Qz*Qx))/Q_Sqr;
    *HJY = (Amp_My*(Q_Sqr-Qy_Sqr) - Amp_Mx*(Qx*Qy) - Amp_Mz*(Qz*Qy))/Q_Sqr;
    *HJZ = (Amp_Mz*(Q_Sqr-Qz_Sqr) - Amp_Mx*(Qx*Qz) - Amp_My*(Qy*Qz))/Q_Sqr;
}

kernel void GeneralScattering(
    global double *Form,
    global int *Form_indices,
    double dx, double dy, double dz,
    double Ax, double Ay, double Az,
    double Bx, double By, double Bz,
    double Cx, double Cy, double Cz,
    int FormFactor_ScatteringCenters,
    int Qx_points, double Qx_min, double Qx_max,
    int Qy_points, double Qy_min, double Qy_max,
    int Qz_points, double Qz_min, double Qz_max,
    int last_x_index, int last_y_index, int last_z_index,
    double N_backing, double N_fronting,
    global cdouble *Amplitude_PP,
    global cdouble *Amplitude_MM,
    global cdouble *Amplitude_MP,
    global cdouble *Amplitude_PM)
{

    double Qx_step = (Qx_points > 1 ? (Qx_max - Qx_min)/(Qx_points - 1) : 0);
    double Qy_step = (Qy_points > 1 ? (Qy_max - Qy_min)/(Qy_points - 1) : 0);
    double Qz_step = (Qz_points > 1 ? (Qz_max - Qz_min)/(Qz_points - 1) : 0);

    int Qx_stride = Qy_points * Qz_points;
    int Qy_stride = Qz_points;
    int Qz_stride = 1;

    double xmax = (last_x_index + 1) * dx;
    double ymax = (last_y_index + 1) * dy;
    double zmax = (last_z_index + 1) * dz;

#ifdef USE_OPENCL
    int offset = get_global_id(0);
    if (offset < Qx_points*Qy_points*Qz_points) {
        int Qxi = offset / Qx_stride;
        int Qyi = (offset % Qx_stride) / Qy_stride;
        int Qzi = offset % Qz_stride;
        double Qx = Qx_min + (Qxi * Qx_step);
        double Qy = Qy_min + (Qyi * Qy_step);
        double Qz = Qz_min + (Qzi * Qz_step);
#else
    //double Qz = 0.0035; //0.004
    #pragma omp parallel for
    for(int Qxi=0; Qxi<Qx_points; Qxi++){
        double Qx = Qx_min + (Qxi * Qx_step);
        for(int Qyi=0; Qyi<Qy_points; Qyi++){
            double Qy = Qy_min + (Qyi * Qy_step);
            for (int Qzi=0; Qzi<Qz_points; Qzi++) {
                double Qz = Qz_min + (Qzi * Qz_step);
                int offset = Qx_stride * Qxi + Qy_stride * Qyi + Qz_stride * Qzi;
#endif
                cdouble N, HJX, HJY, HJZ;

                scatt_kernel(Form, Form_indices, FormFactor_ScatteringCenters,
                    xmax, ymax, zmax, dx, dy, dz,
                    Qx, Qy, Qz,
                    last_x_index, last_y_index, last_z_index,
                    N_backing, N_fronting,
                    &N, &HJX, &HJY, &HJZ);

                cdouble HJA = HJX*Ax + HJY*Ay + HJZ*Az;
                cdouble HJB = HJX*Bx + HJY*By + HJZ*Bz;
                cdouble HJC = HJX*Cx + HJY*Cy + HJZ*Cz;

                Amplitude_PP[offset] = N + HJA;
                Amplitude_MM[offset] = N - HJA;
                Amplitude_MP[offset] = HJB - cmul(I, HJC);
                Amplitude_PM[offset] = HJB + cmul(I, HJC);
#ifdef USE_OPENCL
    }//end if index in Q vector
#else
            }//end Qz loop
            //*Scale*FormFactor_Volume*1E8
        }//end Qy loop
    }//end Qx loop
#endif

} // end of function GeneralScattering
