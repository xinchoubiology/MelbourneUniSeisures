#include <R.h>
#include <cufft.h>
/* This function is written for R to compute 1D FFT.
   n - [IN] the number of complex we want to compute
   inverse - [IN] set to 1 if use inverse mode
   h_idata_re - [IN] input data from host (R, real part)
   h_idata_im - [IN] input data from host (R, imaginary part)
   h_odata_re - [OUT] results (real) allocated by caller
   h_odata_im - [OUT] results (imaginary) allocated by caller
*/
extern "C"
void cufft(int *n, int *inverse, double *h_idata_re,
           double *h_idata_im, double *h_odata_re, double *h_odata_im)
{
  cufftHandle plan;
  cufftDoubleComplex *d_data, *h_data;
  cudaMalloc((void**)&d_data, sizeof(cufftDoubleComplex)*(*n));
  h_data = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * (*n));

  // Convert data to cufftDoubleComplex type
  for(int i=0; i< *n; i++) {
    h_data[i].x = h_idata_re[i];
    h_data[i].y = h_idata_im[i];
  }
 
  cudaMemcpy(d_data, h_data, sizeof(cufftDoubleComplex) * (*n), 
             cudaMemcpyHostToDevice);
  // Use the CUFFT plan to transform the signal in place.
  cufftPlan1d(&plan, *n, CUFFT_Z2Z, 1);
  if (!*inverse ) {
    cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD);
  } else {
    cufftExecZ2Z(plan, d_data, d_data, CUFFT_INVERSE);
  }

  cudaMemcpy(h_data, d_data, sizeof(cufftDoubleComplex) * (*n), 
  cudaMemcpyDeviceToHost);
  // split cufftDoubleComplex to double array
  for(int i=0; i<*n; i++) {
    h_odata_re[i] = h_data[i].x;
    h_odata_im[i] = h_data[i].y;
  }
 
  // Destroy the CUFFT plan and free memory.
  cufftDestroy(plan);
  cudaFree(d_data);
  free(h_data);
}