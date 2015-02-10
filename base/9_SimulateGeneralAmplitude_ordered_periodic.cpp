#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cstdlib>
#include <complex>
using namespace std;

//Written by K. Krycka and B. Maranville

#ifdef __cplusplus
 extern "C" { 
#endif

typedef complex<double> dcmplx;

const dcmplx I = dcmplx(0.0, 1.0);

void scatt_kernel(double *Form, 
                            int *Form_indices,
                            int FormFactor_ScatteringCenters,
                            double xmax, 
                            double ymax,
                            double zmax,
                            double dx,
                            double dy,
                            double dz,
                            double *A, 
                            double *B,
                            double *C,
                            double Qx,
                            double Qy,
                            double Qz,
                            double Q_Sqr,
                            int xx,
                            int yy,
                            int zz,
                            dcmplx laue_factor,
                            int xx_stride,
                            int yy_stride, 
                            int zz_stride,
                            int last_x_index,
                            int last_y_index,
                            int last_z_index,
                            double N_backing,
                            double N_fronting,
                            dcmplx *Amplitude_PP, 
                            dcmplx *Amplitude_MM, 
                            dcmplx *Amplitude_MP, 
                            dcmplx *Amplitude_PM) {
    dcmplx Amp_N = 0.0;
    dcmplx Amp_Mx = 0.0;
    dcmplx Amp_My = 0.0;
    dcmplx Amp_Mz = 0.0;
    
    dcmplx laue_x = exp(I * Qx * dx) - 1.0; // correction factor not including I*Qx
    dcmplx laue_y = exp(I * Qy * dy) - 1.0;
    dcmplx laue_z = exp(I * Qz * dz) - 1.0;
    
    dcmplx Amp_N_bound = 0.0;  // somewhere to put values on the boundary,
    dcmplx Amp_Mx_bound = 0.0; // that won't be multiplied by the Laue factor
    dcmplx Amp_My_bound = 0.0;
    dcmplx Amp_Mz_bound = 0.0;
    
    dcmplx HJX, HJY, HJZ, N, HJA, HJB, HJC, eiqxyz, inv_l, boundary_divisor, rN, rMx, rMy, rMz;
    double msin, mcos, qxyz, x, y, z;
    int offset, c, co, ci, xi, yi, zi;
    //int right_x=0, left_x=0, right_y=0, left_y=0;

    for(c=0; c<FormFactor_ScatteringCenters; c++){
        co = c*7;
        x = Form[co++];
        y = Form[co++];
        z = Form[co++];
        rN = Form[co++];
        rMx = Form[co++];
        rMy = Form[co++];
        rMz = Form[co++];
        
        ci=c*3;
        xi = Form_indices[ci++];
        yi = Form_indices[ci++];
        zi = Form_indices[ci++];
        qxyz = Qx*x + Qy*y + Qz*z;
        msin = sin(qxyz);
        mcos = cos(qxyz);
        eiqxyz = dcmplx(mcos, msin);
        if (zi == 0 && Qz != 0) {
            Amp_N += N_backing * eiqxyz * Qz*Qz / (Q_Sqr * laue_z);
        }
        if (zi == last_z_index && Qz != 0) {
            Amp_N -= N_fronting * eiqxyz * Qz*Qz / (Q_Sqr * laue_z);
        }
        
        Amp_N += rN * eiqxyz;
        Amp_Mx += rMx * eiqxyz;
        Amp_My += rMy * eiqxyz;
        Amp_Mz += rMz * eiqxyz;
        
        if (xi ==  0 && Qx != 0.0) { // x==0, subtract piece from last element
            qxyz = Qx*xmax + Qy*y + Qz*z; // x is at xmax
            msin = sin(qxyz);
            mcos = cos(qxyz);
            eiqxyz = dcmplx(mcos, msin);
            inv_l = eiqxyz * Qx * Qx / (Q_Sqr * laue_x);
            Amp_N -= rN * inv_l;
            Amp_Mx -= rMx * inv_l;
            Amp_My -= rMy * inv_l;
            Amp_Mz -= rMz * inv_l;
        }
        if (xi == last_x_index && Qx != 0.0) { // x is last element, add piece to first element
            qxyz = Qy*y + Qz*z; // x = 0 at first element
            msin = sin(qxyz);
            mcos = cos(qxyz);
            eiqxyz = dcmplx(mcos, msin);   
            inv_l = eiqxyz * Qx * Qx / (Q_Sqr * laue_x);
            Amp_N += rN * inv_l;
            Amp_Mx += rMx * inv_l;
            Amp_My += rMy * inv_l;
            Amp_Mz += rMz * inv_l;
        }
        if (yi ==  0 && Qy != 0.0) { // y==0, subtract piece from last element
            qxyz = Qx*x + Qy*ymax + Qz*z; // y is at ymax
            msin = sin(qxyz);
            mcos = cos(qxyz);
            eiqxyz = dcmplx(mcos, msin);
            inv_l = eiqxyz * Qy * Qy / (Q_Sqr * laue_y);
            Amp_N -= rN * inv_l;
            Amp_Mx -= rMx * inv_l;
            Amp_My -= rMy * inv_l;
            Amp_Mz -= rMz * inv_l;
        }
        if (yi == last_y_index && Qy != 0.0) { // y is last element, add piece to first element
            qxyz = Qx*x + Qz*z; // y = 0 at first element
            msin = sin(qxyz);
            mcos = cos(qxyz);
            eiqxyz = dcmplx(mcos, msin);   
            inv_l = eiqxyz * Qy * Qy / (Q_Sqr * laue_y);
            Amp_N += rN * inv_l;
            Amp_Mx += rMx * inv_l;
            Amp_My += rMy * inv_l;
            Amp_Mz += rMz * inv_l;
        }
        
                
    }//end c loop
    N = Amp_N * laue_factor;
    Amp_Mx *= laue_factor;
    Amp_My *= laue_factor;
    Amp_Mz *= laue_factor;
    
    HJX = (Amp_Mx*(1-Qx*Qx/Q_Sqr) - Amp_My*(Qy*Qx/Q_Sqr) - Amp_Mz*(Qz*Qx/Q_Sqr));
    HJY = (Amp_My*(1-Qy*Qy/Q_Sqr) - Amp_Mx*(Qx*Qy/Q_Sqr) - Amp_Mz*(Qz*Qy/Q_Sqr));
    HJZ = (Amp_Mz*(1-Qz*Qz/Q_Sqr) - Amp_Mx*(Qx*Qz/Q_Sqr) - Amp_My*(Qy*Qz/Q_Sqr));
    
    HJA = HJX * A[0] + HJY * A[1] + HJZ * A[2];
    HJB = HJX * B[0] + HJY * B[1] + HJZ * B[2];
    HJC = HJX * C[0] + HJY * C[1] + HJZ * C[2];

    offset = xx_stride * xx + yy_stride * yy + zz_stride * zz;
    Amplitude_PP[offset] += N + HJA;
    Amplitude_MM[offset] += N - HJA;

    Amplitude_MP[offset] += ( HJB - (I * HJC));
    Amplitude_PM[offset] += ( HJB + (I * HJC));
}

                       
void GeneralScattering_zfirst(double *Form,
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
                       dcmplx *Amplitude_PM) {
    double scatt_num = FormFactor_ScatteringCenters;
    double Qx_range = Qx_max - Qx_min;
    double Qx_step = 0;
    if (Qx_points > 1) { Qx_step = Qx_range / (Qx_points - 1); }
    double Qy_range = Qy_max - Qy_min;
    double Qy_step = 0;
    if (Qy_points > 1) { Qy_step = Qy_range / (Qy_points - 1); } 
    double Qz_range = Qz_max - Qz_min;
    double Qz_step = 0;
    if (Qz_points > 1) { Qz_step = Qz_range / (Qz_points - 1); } 
    
    int xx_stride = Qy_points * Qz_points;
    int yy_stride = Qz_points;
    int zz_stride = 1;
    
    double xmax = (last_x_index + 1) * dx;
    double ymax = (last_y_index + 1) * dy;
    double zmax = (last_z_index + 1) * dz;
    
    //double Qz = 0.0035; //0.004
    #pragma omp parallel for
    for(int zz=0; zz<Qz_points; zz++){    
        // conversion from integral to sum needs this correction:
        // (but variable decl. needs to be inside parallelized for loop)
        dcmplx x_int_factor, y_int_factor, z_int_factor, laue_factor;
        double Qx, Qy, Qz, Q_Sqr;
        
        Qz = Qz_min + (zz * Qz_step);
        z_int_factor = (Qz != 0.0 ? (1.0 - exp(I * Qz * dz)) * (I / Qz) : dz);
        
        int xx, yy, c;       
        for(xx=0; xx<Qx_points; xx++){
            Qx = Qx_min + (xx * Qx_step);
            x_int_factor = (Qx != 0.0 ? (1.0 - exp(I * Qx * dx)) * (I / Qx) : dx);            

            for (yy=0; yy<Qy_points; yy++) {
                Qy = Qy_min + (yy * Qy_step);
                y_int_factor = (Qy != 0.0 ? (1.0 - exp(I * Qy * dy)) * (I / Qy) : dy);
                Q_Sqr = Qx*Qx + Qy*Qy + Qz*Qz;
                
                laue_factor = x_int_factor * y_int_factor * z_int_factor;
                scatt_kernel(Form, Form_indices, FormFactor_ScatteringCenters, xmax, ymax, zmax, dx, dy, dz, A, B, C, Qx, Qy, Qz, Q_Sqr, xx, yy, zz, laue_factor, xx_stride, yy_stride, zz_stride,
                            last_x_index, last_y_index, last_z_index, N_backing, N_fronting,
                            Amplitude_PP, 
                            Amplitude_MM, 
                            Amplitude_MP, 
                            Amplitude_PM);
            }
            //*Scale*FormFactor_Volume*1E8
        }//end yy loop
    }//end xx loop

} // end of function GeneralScattering_zfirst

void GeneralScattering_yfirst(double *Form, 
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
                       dcmplx *Amplitude_PM) {
    double scatt_num = FormFactor_ScatteringCenters;
    double Qx_range = Qx_max - Qx_min;
    double Qx_step = 0;
    if (Qx_points > 1) { Qx_step = Qx_range / (Qx_points - 1); }
    double Qy_range = Qy_max - Qy_min;
    double Qy_step = 0;
    if (Qy_points > 1) { Qy_step = Qy_range / (Qy_points - 1); } 
    double Qz_range = Qz_max - Qz_min;
    double Qz_step = 0;
    if (Qz_points > 1) { Qz_step = Qz_range / (Qz_points - 1); } 
    
    int xx_stride = Qy_points * Qz_points;
    int yy_stride = Qz_points;
    int zz_stride = 1;
    
    double xmax = (last_x_index + 1) * dx;
    double ymax = (last_y_index + 1) * dy;
    double zmax = (last_z_index + 1) * dz;
    
    //double Qz = 0.0035; //0.004
    #pragma omp parallel for
    for(int yy=0; yy<Qy_points; yy++){
    
        // conversion from integral to sum needs this correction:
        // (but variable decl. needs to be inside parallelized for loop)
        dcmplx x_int_factor, y_int_factor, z_int_factor, laue_factor;
        double Qx, Qy, Qz, Q_Sqr;
        dcmplx HJX, HJY, HJZ, N, HJA, HJB, HJC, eiqxyz;
        double msin, mcos, qxyz;
        int offset;
        
        dcmplx Amp_N = 0.0;
        dcmplx Amp_Mx = 0.0;
        dcmplx Amp_My = 0.0;
        dcmplx Amp_Mz = 0.0;
        Qy = Qy_min + (yy * Qy_step);
        y_int_factor = (Qy != 0.0 ? (1.0 - exp(I * Qy * dy)) * (I / Qy) : dy);
        
        int xx, zz, c;
        
        for(xx=0; xx<Qx_points; xx++){
            Qx = Qx_min + (xx * Qx_step);
            x_int_factor = (Qx != 0.0 ? (1.0 - exp(I * Qx * dx)) * (I / Qx) : dx);            

            for (zz=0; zz<Qz_points; zz++) {
                Qz = Qz_min + (zz * Qz_step);
                z_int_factor = (Qz != 0.0 ? (1.0 - exp(I * Qz * dz)) * (I / Qz) : dz);
                Q_Sqr = Qx*Qx + Qy*Qy + Qz*Qz;
                
                laue_factor = x_int_factor * y_int_factor * z_int_factor;
                scatt_kernel(Form, Form_indices, FormFactor_ScatteringCenters, xmax, ymax, zmax, dx, dy, dz, A, B, C, Qx, Qy, Qz, Q_Sqr, xx, yy, zz, laue_factor, xx_stride, yy_stride, zz_stride,
                            last_x_index, last_y_index, last_z_index, N_backing, N_fronting,
                            Amplitude_PP, 
                            Amplitude_MM, 
                            Amplitude_MP, 
                            Amplitude_PM);
            }
            //*Scale*FormFactor_Volume*1E8
        }//end yy loop
    }//end xx loop

} // end of function GeneralScattering_yfirst

void GeneralScattering_xfirst(double *Form, 
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
                       dcmplx *Amplitude_PM) {
    double scatt_num = FormFactor_ScatteringCenters;
    double Qx_range = Qx_max - Qx_min;
    double Qx_step = 0;
    if (Qx_points > 1) { Qx_step = Qx_range / (Qx_points - 1); }
    double Qy_range = Qy_max - Qy_min;
    double Qy_step = 0;
    if (Qy_points > 1) { Qy_step = Qy_range / (Qy_points - 1); } 
    double Qz_range = Qz_max - Qz_min;
    double Qz_step = 0;
    if (Qz_points > 1) { Qz_step = Qz_range / (Qz_points - 1); } 
    
    int xx_stride = Qy_points * Qz_points;
    int yy_stride = Qz_points;
    int zz_stride = 1;
    
    double xmax = (last_x_index + 1) * dx;
    double ymax = (last_y_index + 1) * dy;
    double zmax = (last_z_index + 1) * dz;
    
    //double Qz = 0.0035; //0.004
    #pragma omp parallel for
    for(int xx=0; xx<Qx_points; xx++){
        // conversion from integral to sum needs this correction:
        // (but variable decl. needs to be inside parallelized for loop)
        dcmplx x_int_factor, y_int_factor, z_int_factor, laue_factor;
        double Qx, Qy, Qz, Q_Sqr;
        dcmplx HJX, HJY, HJZ, N, HJA, HJB, HJC, eiqxyz;
        double msin, mcos, qxyz;
        int offset;
        
        dcmplx Amp_N = 0.0;
        dcmplx Amp_Mx = 0.0;
        dcmplx Amp_My = 0.0;
        dcmplx Amp_Mz = 0.0;
        
        Qx = Qx_min + (xx * Qx_step);
        x_int_factor = (Qx != 0.0 ? (1.0 - exp(I * Qx * dx)) * (I / Qx) : dx);
        int yy, zz, c;
        
        for(yy=0; yy<Qy_points; yy++){
            Qy = Qy_min + (yy * Qy_step);
            y_int_factor = (Qy != 0.0 ? (1.0 - exp(I * Qy * dy)) * (I / Qy) : dy);

            for (zz=0; zz<Qz_points; zz++) {
                Qz = Qz_min + (zz * Qz_step);
                z_int_factor = (Qz != 0.0 ? (1.0 - exp(I * Qz * dz)) * (I / Qz) : dz);
                Q_Sqr = Qx*Qx + Qy*Qy + Qz*Qz;
                
                laue_factor = x_int_factor * y_int_factor * z_int_factor;
                scatt_kernel(Form, Form_indices, FormFactor_ScatteringCenters, xmax, ymax, zmax, dx, dy, dz, A, B, C, Qx, Qy, Qz, Q_Sqr, xx, yy, zz, laue_factor, xx_stride, yy_stride, zz_stride,
                            last_x_index, last_y_index, last_z_index, N_backing, N_fronting,
                            Amplitude_PP, 
                            Amplitude_MM, 
                            Amplitude_MP, 
                            Amplitude_PM);
            }
            //*Scale*FormFactor_Volume*1E8
        }//end yy loop
    }//end xx loop

} // end of function GeneralScattering_xfirst
#ifdef __cplusplus
}
#endif


