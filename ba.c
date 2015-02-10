//Written by K. Krycka and B. Maranville
#ifndef __OPENCL_VERSION__
#include "cl_env.h"
#endif

// OS/X OpenCL driver wants a prototype
void magneticBA(
    global double *Form, global int *Form_indices,
    int FormFactor_ScatteringCenters,
    double xmax, double ymax, double zmax,
    double dx, double dy, double dz,
    double Qx, double Qy, double Qz,
    int last_x_index, int last_y_index, int last_z_index,
    double N_backing, double N_fronting,
    double Ax, double Ay, double Az,
    double Bx, double By, double Bz,
    double Cx, double Cy, double Cz,
    global cdouble *Rpp,
    global cdouble *Rpm,
    global cdouble *Rmp,
    global cdouble *Rmm);
void magneticBA(
    global double *Form, global int *Form_indices,
    int FormFactor_ScatteringCenters,
    double xmax, double ymax, double zmax,
    double dx, double dy, double dz,
    double Qx, double Qy, double Qz,
    int last_x_index, int last_y_index, int last_z_index,
    double N_backing, double N_fronting,
    double Ax, double Ay, double Az,
    double Bx, double By, double Bz,
    double Cx, double Cy, double Cz,
    global cdouble *Rpp,
    global cdouble *Rpm,
    global cdouble *Rmp,
    global cdouble *Rmm)
{
    // Note: don't need cmul when multiplying by I
    const cdouble laue_x = cplx(-1.0 + cos(Qx*dx), sin(Qx*dx)); // correction factor not including I*Qx
    const cdouble laue_y = cplx(-1.0 + cos(Qy*dy), sin(Qy*dy));
    const cdouble laue_z = cplx(-1.0 + cos(Qz*dz), sin(Qz*dz));

    const double Qx_Sqr = Qx*Qx;
    const double Qy_Sqr = Qy*Qy;
    const double Qz_Sqr = Qz*Qz;
    const double Q_Sqr = Qx_Sqr + Qy_Sqr + Qz_Sqr;

    const cdouble x_int_factor = (Qx != 0.0 ? cmul(laue_x, cplx(0.0, -1.0/Qx)) : cplx(dx,0.0));
    const cdouble y_int_factor = (Qy != 0.0 ? cmul(laue_y, cplx(0.0, -1.0/Qy)) : cplx(dy,0.0));
    const cdouble z_int_factor = (Qz != 0.0 ? cmul(laue_z, cplx(0.0, -1.0/Qz)) : cplx(dz,0.0));
    const cdouble laue_factor = cmul(x_int_factor, cmul(y_int_factor, z_int_factor));

    cdouble Amp_N = cplx(0.0,0.0);
    cdouble Amp_Mx = cplx(0.0,0.0);
    cdouble Amp_My = cplx(0.0,0.0);
    cdouble Amp_Mz = cplx(0.0,0.0);

    for(int c=0; c<FormFactor_ScatteringCenters; c++){
        const int co = c*8;
        const double x = Form[co];
        const double y = Form[co+1];
        const double z = Form[co+2];
        const double rBx = Form[co+3];
        const double rBy = Form[co+4];
        const double rBz = Form[co+5];
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
            Amp_N = (Qz != 0.0 ? cadd(Amp_N, rmul(edge_scale*Qz_Sqr/Q_Sqr, cdiv(p,laue_z))) : Amp_N );
        }

        // Handle edges in x and y
        edge_scale = (xi==0)*-1.0 + (xi==last_x_index)*1.0;
        if (edge_scale != 0.0)
        {
            const double opposite_edge = (xi==0)*xmax + (xi==last_x_index)*0.0;
            const double qxyz = Qx*opposite_edge + Qy*y + Qz*z;
            //const cdouble eiqxyz = cpolar(1.0, qxyz);
            const cdouble eiqxyz = cplx(cos(qxyz), sin(qxyz));
            p = (Qx != 0.0 ? cadd(p, rmul(edge_scale*Qx_Sqr/Q_Sqr, cdiv(eiqxyz,laue_x))) : p );
        }

        edge_scale = (yi==0)*-1.0 + (yi==last_y_index)*1.0;
        if (edge_scale != 0.0)
        {
            const double opposite_edge = (yi==0)*ymax + (yi==last_y_index)*0.0;
            const double qxyz = Qx*x + Qy*opposite_edge + Qz*z;
            //const cdouble eiqxyz = cpolar(1.0, qxyz);
            const cdouble eiqxyz = cplx(cos(qxyz), sin(qxyz));
            p = (Qy != 0.0 ? cadd(p, rmul(edge_scale*Qy_Sqr/Q_Sqr, cdiv(eiqxyz,laue_y))) : p );
        }
#endif

        Amp_N = cadd(Amp_N, cmul(rN, p));
        Amp_Mx = cadd(Amp_Mx, rmul(rBx, p));
        Amp_My = cadd(Amp_My, rmul(rBy, p));
        Amp_Mz = cadd(Amp_Mz, rmul(rBz, p));

    }//end c loop

    Amp_N = cmul(Amp_N, laue_factor);
    Amp_Mx = cmul(Amp_Mx, laue_factor);
    Amp_My = cmul(Amp_My, laue_factor);
    Amp_Mz = cmul(Amp_Mz, laue_factor);

    const cdouble N = Amp_N;
    const cdouble HJX = cadd(cadd(rmul(1.0-Qx_Sqr/Q_Sqr,Amp_Mx), rmul(-Qy*Qx/Q_Sqr,Amp_My)), rmul(-Qz*Qx/Q_Sqr,Amp_Mz));
    const cdouble HJY = cadd(cadd(rmul(-Qx*Qy/Q_Sqr,Amp_Mx), rmul(1.0-Qy_Sqr/Q_Sqr,Amp_My)), rmul(-Qz*Qy/Q_Sqr,Amp_Mz));
    const cdouble HJZ = cadd(cadd(rmul(-Qx*Qz/Q_Sqr,Amp_Mx), rmul(-Qy*Qz/Q_Sqr,Amp_My)), rmul(1.0-Qz_Sqr/Q_Sqr,Amp_Mz));

    const cdouble HJA = cadd(cadd(rmul(Ax,HJX), rmul(Ay,HJY)), rmul(Az,HJZ));
    const cdouble HJB = cadd(cadd(rmul(Bx,HJX), rmul(By,HJY)), rmul(Bz,HJZ));
    const cdouble HJC = cadd(cadd(rmul(Cx,HJX), rmul(Cy,HJY)), rmul(Cz,HJZ));

    *Rpp = cadd(N, HJA);
    *Rmm = csub(N, HJA);
    *Rmp = csub(HJB, cmul(I, HJC));
    *Rpm = cadd(HJB, cmul(I, HJC));
}

kernel void born_approximation(
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
                magneticBA(Form, Form_indices, FormFactor_ScatteringCenters,
                    xmax, ymax, zmax, dx, dy, dz,
                    Qx, Qy, Qz,
                    last_x_index, last_y_index, last_z_index,
                    N_backing, N_fronting,
                    Ax, Ay, Az,
                    Bx, By, Bz,
                    Cx, Cy, Cz,
                    Amplitude_PP+offset, Amplitude_PM+offset,
                    Amplitude_MP+offset, Amplitude_MM+offset);

#ifdef USE_OPENCL
    }//end if index in Q vector
#else
            }//end Qz loop
            //*Scale*FormFactor_Volume*1E8
        }//end Qy loop
    }//end Qx loop
#endif

} // end of function GeneralScattering
