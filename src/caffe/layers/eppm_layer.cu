#include <vector>
#include <stdio.h>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/output.hpp"

#include "caffe/layers/eppm_layer.hpp"

#include "caffe/caffe.hpp"

#define ROUND_OFF 50000

#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

#define LAMDA_R 0.1f
#define LAMBDA_AD 0.1f  


namespace caffe {

// == Correlation Kernel
template <typename Dtype> 
__global__ void CorrelateData(const int nthreads, int num, int topwidth, int topheight, int topchannels, int topcount,
  int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width, int kernel_radius, int kernel_size, int stride1, int stride2,
  int bottomwidth, int bottomheight, int bottomchannels,
  const Dtype *bottom0, const Dtype *bottom1, Dtype *top, float* SpatialGaussian, int eppm_type) 
{
  extern __shared__ char patch_data_char[];
  
  Dtype *patch_data = (Dtype *)patch_data_char;
  
    // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
  int x1 = blockIdx.x*stride1 + max_displacement;
  int y1 = blockIdx.y*stride1 + max_displacement;
  int item = blockIdx.z;
  int ch_off = threadIdx.x;
  
  // Load 3D patch into shared shared memory
  for(int j = 0; j < kernel_size; j++) { // HEIGHT
    for(int i = 0; i < kernel_size; i++) { // WIDTH
      int ji_off = ((j * kernel_size) + i) * bottomchannels;
      for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
          int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + ch;
          int idxPatchData = ji_off + ch;
          patch_data[idxPatchData] = bottom0[idx1];
      }
    }
  }
  
  __syncthreads();
  
  __shared__ Dtype sum[WARPS_PER_BLOCK*THREADS_PER_WARP];
  
  // Compute correlation
  for(int top_channel = 0; top_channel < topchannels; top_channel++) {
    sum[ch_off] = 0;
  
    int s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    int s2p = (top_channel / neighborhood_grid_width - neighborhood_grid_radius) * stride2;

    
  for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
    Dtype kernel_sum = 0;
    Dtype weight_sum = 0;
    for(int j = 0; j < kernel_size; j++) { // HEIGHT
      for(int i = 0; i < kernel_size; i++) { // WIDTH
        int ji_off = ((j * kernel_size) + i) * bottomchannels;
          int x2 = x1 + s2o;
          int y2 = y1 + s2p;
          
          int idxPatchData = ji_off + ch;
          int idx2 = ((item * bottomheight + y2+j) * bottomwidth + x2+i) * bottomchannels + ch;

          int idx1_center = ((item * bottomheight + y1) * bottomwidth + x1) * bottomchannels + ch;
          int idx2_center = ((item * bottomheight + y2) * bottomwidth + x2) * bottomchannels + ch;

          Dtype weight0 = patch_data[idxPatchData] * bottom0[idx1_center];
          Dtype weight1 = bottom1[idx2] * bottom1[idx2_center];
          Dtype weight = 0;
          if (eppm_type == 1) {
            weight = weight0 + weight1;
          }
          else {
            weight = (weight0 + weight1) * SpatialGaussian[i] * SpatialGaussian[j];
          }
          weight_sum += weight;
          kernel_sum += patch_data[idxPatchData] * bottom1[idx2] * weight;
        }
      }
      if (weight_sum == 0)
        weight_sum = 1;
      sum[ch_off] += kernel_sum / (float)weight_sum;
    }
    
    __syncthreads();
    
    if(ch_off == 0) {
        Dtype total_sum = 0;
        Dtype total_weight = 0;
        for(int idx = 0; idx < WARPS_PER_BLOCK*THREADS_PER_WARP; idx++) {
            total_sum += sum[idx];
        }
        const int sumelems = kernel_size*kernel_size*bottomchannels;
        const int index = ((top_channel*topheight + blockIdx.y)*topwidth)+blockIdx.x;
        top[index + item*topcount] = total_sum / (float)sumelems;
    }
  }
  
  
  // Aggregate  
}

// == Correlation Backward Pass Kernel (For Blob 0)
template <typename Dtype> 
__global__ void CorrelateDataBackward0(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  const Dtype *bottom0, const Dtype *bottom1, Dtype *bottom0diff, const Dtype *topdiff, float* SpatialGaussian, int eppm_type) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size; //h-pos

    //Get X,Y ranges and clamp
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
    int xmin = (l - 2 * kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - 2 * kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    
    // Same here:
    int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
    int ymax = (m - max_displacement + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1
    

    Dtype sum = 0;
    if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
    {
        xmin = max(0,xmin);
        xmax = min(topwidth-1,xmax);

        ymin = max(0,ymin);
        ymax = min(topheight-1,ymax);

        for(int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
          for(int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {

            // Get bottom1 data:
            int s2o = stride2 * o;
            int s2p = stride2 * p;
            int idxbot1 = ((item * pbottomheight + (m+s2p)) * pbottomwidth + (l+s2o)) * bottomchannels + n;
            // Dtype bot1tmp = bottom1[idxbot1]; // bottom1[l+s2o,m+s2p,n]

            // Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
            int idxopoffset = (item * topchannels + op);

            Dtype kernel_sum = 0;
            Dtype weight_sum = 0;
            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]

                int idxbot0 = ((item * pbottomheight + (m+y)) * pbottomwidth + (l+x)) * bottomchannels + n;
                int common_center = ((item * pbottomheight + m) * pbottomwidth + l) * bottomchannels + n;
                float spatialWeight = 0;
                if (eppm_type == 1) {
                  spatialWeight = 1.f;
                }
                else {
                  spatialWeight = SpatialGaussian[abs(y-ymin)] * SpatialGaussian[abs(x-xmin)];
                }
                // f = cost * weight
                Dtype cost = bottom0[idxbot0] * bottom1[idxbot1] * spatialWeight;
                Dtype weight = (bottom0[idxbot0] * bottom0[common_center] + bottom1[idxbot1] * bottom1[common_center]) * spatialWeight;

                Dtype dcost = weight * topdiff[idxtopdiff];
                Dtype dweight = cost * topdiff[idxtopdiff];
                
                // cost BP
                Dtype dx = bottom1[idxbot1] * dcost;
                // weight BP
                dx += bottom0[common_center] * dweight;

                kernel_sum += dx;
                weight_sum += weight;
              }
            }
            if (weight_sum == 0)
              weight_sum = 1;
            sum += kernel_sum / (float)weight_sum;
          }
        }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    const int bot0index = ((n * bottomheight) + (m-pad_size)) * bottomwidth + (l-pad_size);
    bottom0diff[bot0index + item*bottomcount] = sum / (float)sumelems;
  }

}



// == Correlation Backward Pass Kernel (For Blob 1)
template <typename Dtype> 
__global__ void CorrelateDataBackward1(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  const Dtype *bottom0, const Dtype *bottom1, Dtype *bottom1diff, const Dtype *topdiff, float* SpatialGaussian, int eppm_type) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    //int l = index % bottomwidth + pad_size; //w-pos
    //int m = (index / bottomwidth) % bottomheight + pad_size; //h-pos
    //int n = (index / bottomwidth / bottomheight) % bottomchannels; //channels
    int n = index % bottomchannels; //channels
    int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size; //h-pos

    
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    Dtype sum = 0;
    for(int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
      for(int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
        
        int s2o = stride2 * o;
        int s2p = stride2 * p;
        
        //Get X,Y ranges and clamp
        // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
        int xmin = (l - 2 * kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        int ymin = (m - 2 * kernel_radius - max_displacement - s2p + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        
        // Same here:
        int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
        int ymax = (m - max_displacement - s2p + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1

        if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
        {
            xmin = max(0,xmin);
            xmax = min(topwidth-1,xmax);

            ymin = max(0,ymin);
            ymax = min(topheight-1,ymax);

            // Get bottom0 data:
            int idxbot0 = ((item * pbottomheight + (m-s2p)) * pbottomwidth + (l-s2o)) * bottomchannels + n;
            // Dtype bot0tmp = bottom0[idxbot0]; // bottom1[l+s2o,m+s2p,n]

            // Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
            int idxOpOffset = (item * topchannels + op);

            Dtype weight_sum = 0;
            Dtype kernel_sum = 0;
            for(int y = ymin; y <= ymax; y++) {
              for(int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]

                int idxbot1 = ((item * pbottomheight + (m+y)) * pbottomwidth + (l+x)) * bottomchannels + n;
                int common_center = ((item * pbottomheight + m) * pbottomwidth + l) * bottomchannels + n;

                float spatialWeight = 0;
                if (eppm_type == 1) {
                  spatialWeight = 1.f;
                }
                else {
                  spatialWeight = SpatialGaussian[abs(y-ymin)] * SpatialGaussian[abs(x-xmin)];
                }
                // f = cost * weight
                Dtype cost = bottom0[idxbot0] * bottom1[idxbot1] * spatialWeight;
                Dtype weight = (bottom0[idxbot0] * bottom0[common_center] + bottom1[idxbot1] * bottom1[common_center]) * spatialWeight;

                Dtype dcost = weight * topdiff[idxtopdiff];
                Dtype dweight = cost * topdiff[idxtopdiff];
                
                // cost BP
                Dtype dy = bottom0[idxbot0] * dcost;
                // weight BP
                dy += bottom1[common_center] * dweight;

                kernel_sum += dy;
                weight_sum += weight;
              }
            }
            if (weight_sum == 0)
              weight_sum = 1;
            sum += kernel_sum / weight_sum;
        }
      }
    }
    const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    const int bot1index = ((n * bottomheight) + (m-pad_size)) * bottomwidth + (l-pad_size);
    bottom1diff[bot1index + item*bottomcount] = sum / (float)sumelems;
  }

}

//////////////////////////////////////////////////////////////////////////////////////////////////
  

// generate spatial gaussian table
void cuda_init_gaussian_Lookup_table(float* SpatialGaussian, int kernel_radius) {
  float lamda_k = kernel_radius;
  float* fGaussian = new float[2 * kernel_radius + 1];
  for (int i = 0; i <= 2 * kernel_radius; i++) {
        fGaussian[i] = expf(-(i*i) / (lamda_k * lamda_k));
    }
    // cudaMemcpyToSymbol(cSpatialGaussian, fGaussian, (PATCH_R+1) * sizeof(float));
    cudaMemcpy(SpatialGaussian, fGaussian, (2 * kernel_radius+1) * sizeof(float), cudaMemcpyHostToDevice);
}

template <typename Dtype>
__global__ void blob_rearrange_kernel2(const Dtype* in, Dtype* out, int num, int channels, int width, int height, int widthheight, int padding, int pwidthheight)
{
    int xy = blockIdx.x*blockDim.x + threadIdx.x;
    if(xy>=widthheight)
        return;

    int ch = blockIdx.y;
    int n  = blockIdx.z;

    Dtype value=in[(n*channels+ch)*widthheight+xy];

    __syncthreads();

    int xpad  = (xy % width + padding);
    int ypad  = (xy / width + padding);
    int xypad = ypad * (width+2*padding) + xpad;

    out[(n*pwidthheight+xypad)*channels + ch] = value;
}

// == EPPM Kernel
template <typename Dtype> 
__global__ void computeMatchKernel(const int nthreads, int num, int topwidth, int topheight, int topchannels, int topcount,
  int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width, int kernel_radius, int kernel_size, 
  int border_size, int stride1, int stride2,
  int bottomwidth, int bottomheight, int bottomchannels,
  const Dtype *bottom0, const Dtype *bottom1, Dtype *top,
  float* SpatialGaussian) 
{
  // extern __shared__ char patch_data_char[];
  // // Dtype *patch_data = (Dtype *)patch_data_char;

  // init gaussian table
  int x1 = blockIdx.x*stride1 + border_size;
  int y1 = blockIdx.y*stride1 + border_size;
  int item = blockIdx.z;
  int ch_off = threadIdx.x;

  // start patch loop
  for(int top_channel = 0; top_channel < topchannels; top_channel++) {
    
    int s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    int s2p = (top_channel / neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    int x2 = x1 + s2o;
    int y2 = y1 + s2p;


    __shared__ Dtype result_sum[WARPS_PER_BLOCK*THREADS_PER_WARP];
  for(int ch = ch_off; ch < bottomchannels; ch += (WARPS_PER_BLOCK*THREADS_PER_WARP)) { // CHANNELS
    Dtype cost_sum = 0;
    // Dtype weight_sum = 0;
    for (int kernel_h = -kernel_radius; kernel_h < kernel_size; ++kernel_h) {
      for (int kernel_w = -kernel_radius; kernel_w < kernel_size; ++kernel_w) {
          //////////////////////////////////////////////////// for debug purpose
          int img0_idx = ((item * bottomheight + y1 + kernel_h) * bottomwidth + x1 + kernel_w) * bottomchannels + ch;
          int img1_idx = ((item * bottomheight + y2 + kernel_h) * bottomwidth + x2 + kernel_w) * bottomchannels + ch;
          Dtype img0_pixel = bottom0[img0_idx];
          Dtype img1_pixel = bottom1[img1_idx];
          cost_sum += img0_pixel + img1_pixel;
          //////////////////////////////////////////////////// debug end

          /*
          // step2: compute the cost.
          Dtype cost_sum = 0;
          // (x-y)^2
          // step2.1: get current pixel of image0
          int img0_idx = ((item * bottomheight + y1 + kernel_h) * bottomwidth + x1 + kernel_w) * bottomchannels + ch;

          // step2.2: get current pixel of image1
          int img1_idx = ((item * bottomheight + y2 + kernel_h) * bottomwidth + x2 + kernel_w) * bottomchannels + ch;
          
          Dtype cost = bottom0[img0_idx] - bottom1[img1_idx];
          cost *= cost;

          // 1 - e(-(x-y)^2 / delta^2
          cost = 1 - __expf(-cost / (LAMBDA_AD * LAMBDA_AD)); 

          // step3: compute the weight
          // get center pixel of image0 and image1
          int img0_center_idx = ((item * bottomheight + y1) * bottomwidth + x1) * bottomchannels + ch;
          int img1_center_idx = ((item * bottomheight + y2) * bottomwidth + x2) * bottomchannels + ch;
          // image0 weight
          Dtype img0_weight = bottom0[img0_idx] - bottom0[img0_center_idx];
          img0_weight *= img0_weight;
          // image1 weight
          Dtype img1_weight = bottom1[img1_idx] - bottom1[img1_center_idx];
          img1_weight *= img1_weight;
          // (x - xo)^2 + (y - yo)^2
          Dtype both_weight = img0_weight + img1_weight;

          // e((-(x - xo)^2 + (y - yo)^2) / lamda^2)
          Dtype exp_weight = __expf(-both_weight / (LAMDA_R * LAMDA_R));

          // e(((x - xo)^2 + (y - yo)^2) / lamda^2) * e(-o^2/lamda^2)
          Dtype weight = exp_weight * SpatialGaussian[abs(kernel_w)] * SpatialGaussian[abs(kernel_h)];

          cost_sum += cost * weight;

          weight_sum += weight;
          */
        
      } // kernel width
    } // kernel height
      // result_sum[ch_off] += cost_sum / weight_sum;
    //////////////////////////////////////////////////// for debug purpose
    result_sum[ch_off] += cost_sum;
    //////////////////////////////////////////////////// debug end
  } // channels

    __syncthreads();

    
    if(ch_off == 0) {
        Dtype total_result_sum = 0;
        for(int idx = 0; idx < WARPS_PER_BLOCK*THREADS_PER_WARP; idx++) {
            total_result_sum += result_sum[idx];
        }
        // const int sumelems = kernel_size*kernel_size*bottomchannels;
        const int index = ((top_channel*topheight + blockIdx.y)*topwidth)+blockIdx.x;
        top[index + item*topcount] = total_result_sum; // /sumelems?
    }
  } // top channels
}

// == Patch Match Backward Pass Kernel (For Blob 0)
template <typename Dtype>
__global__ void PatchMatchBackWard(const int nthreads, int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int neighborhood_grid_radius, int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, int pad_size,
  const Dtype *bottom0, const Dtype *bottom1, Dtype *bottom0diff, Dtype* bottom1diff, const Dtype *topdiff, float* SpatialGaussian) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index % bottomchannels; //channel pos
    int l = (index / bottomchannels) % bottomwidth + pad_size; //w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size; //h-pos

    //Get X,Y ranges and clamp
    // round_off is a trick to enable integer division with ceil, even for negative numbers
    // We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    
    // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
    int xmin = (l - kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
    
    // Same here:
    int xmax = (l + kernel_radius - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
    int ymax = (m + kernel_radius - max_displacement + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1
    

    Dtype sum_x = 0;
    Dtype sum_y = 0;
    
    if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
    {
        xmin = max(0,xmin);
        xmax = min(topwidth-1,xmax);

        ymin = max(0,ymin);
        ymax = min(topheight-1,ymax);

        for(int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
          for(int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {

            // Get bottom1 data:
            int s2o = stride2 * o;
            int s2p = stride2 * p;

            // Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
            int idxopoffset = (item * topchannels + op);

            // get center pixel of image0
            int img0_center_idx = ((item * pbottomheight + m) * pbottomwidth + l) * bottomchannels + n;
            Dtype img0_center = bottom0[img0_center_idx];

            int img1_center_idx = ((item * pbottomheight + m + s2p) * pbottomwidth + l + s2o) * bottomchannels + n;
            Dtype img1_center = bottom1[img1_center_idx];

            Dtype weight_sum = 0;
            Dtype sum_kernel_x = 0;
            Dtype sum_kernel_y = 0;
            // start BP from each kernel element
            int kr_y = (ymax - ymin) / 2;
            int kr_x = (xmax - xmin) / 2;
            for(int y = -kr_y; y <= kr_y; y++) {
              for(int x = -kr_x; x <= kr_x; x++) {

                //////////////////////////////////////////////////// for debug purpose
                int img0_idx = ((item * pbottomheight + m + y) * pbottomwidth + l + x) * bottomchannels + n;
                Dtype img0_pixel = bottom0[img0_idx];
                int img1_idx = ((item * pbottomheight + m + s2p + y) * pbottomwidth + l + s2o + x) * bottomchannels + n;
                Dtype img1_pixel = bottom1[img1_idx];

                int idxtopdiff = (idxopoffset * topheight + y + kr_y) * topwidth + x + kr_x;
                Dtype deltaTop = topdiff[idxtopdiff];

                Dtype dx = deltaTop;
                Dtype dy = deltaTop;

                sum_kernel_x += dx;
                sum_kernel_y += dy;
                //////////////////////////////////////////////////// debug end

                /*
                // step0: get current pixel
                int img0_idx = ((item * pbottomheight + m + y) * pbottomwidth + l + x) * bottomchannels + n;
                Dtype img0_pixel = bottom0[img0_idx];
                int img1_idx = ((item * pbottomheight + m + s2p + y) * pbottomwidth + l + s2o + x) * bottomchannels + n;
                Dtype img1_pixel = bottom1[img1_idx];

                // step1: get deltaX idx
                int idxtopdiff = (idxopoffset * topheight + y + kr_y) * topwidth + x + kr_x;
                Dtype deltaTop = topdiff[idxtopdiff];
                
                // f = weight * cost * C
                // C: e(-o^2/lamda^2)
                float spaceC = SpatialGaussian[abs(y - (ymax + ymin) / 2)] * SpatialGaussian[abs(x - (xmax + xmin) / 2)];
                // weight: e(((x - xo)^2 + (y - yo)^2) / lamda^2) * e(-o^2/lamda^2)
                Dtype weight = __expf(-(pow(img0_pixel - img0_center, 2) + pow(img1_pixel - img1_center, 2)) / (LAMDA_R * LAMDA_R));
                // cost: 1 - e(-(x-y)^2 / delta^2)
                Dtype cost = 1 - __expf(-pow(img0_pixel - img1_pixel, 2) / (LAMBDA_AD * LAMBDA_AD));
                // BP
                Dtype dcost = spaceC * weight * deltaTop;
                Dtype dweight = spaceC * cost * deltaTop;

                // cost = 1 - e(-distsquaCom)
                // distsquaCom : (x - y)^2 / delta^2
                Dtype distsquaCom = pow(img0_pixel - img1_pixel, 2) / (LAMBDA_AD * LAMBDA_AD);
                // BP
                Dtype ddistsquaCom = __expf(-distsquaCom) * dcost;

                // distsquaCom = distxysqua / delta^2
                // distxysqua: (x - y)^2
                // Dtype distxysqua = pow(img0_pixel - img1_pixel, 2);
                // BP
                Dtype ddistxysqua = 1 / (LAMBDA_AD * LAMBDA_AD) * ddistsquaCom;

                // distxysqua = distxy^2
                // distxy = x - y
                Dtype distxy = img0_pixel - img1_pixel;
                // BP
                Dtype ddistxy = 2 * distxy * ddistxysqua;

                // distxy = x - y
                // BP
                Dtype dx = ddistxy;
                Dtype dy = -ddistxy;

                // weight = e(-weightxyCom)
                // weightxyCom: ((x - xo)^2 + (y - yo)^2)/delta^2
                Dtype weightxyCom = (pow(img0_pixel - img0_center, 2) + pow(img1_pixel - img1_center, 2)) / (LAMDA_R * LAMDA_R);
                // BP
                Dtype dweightxyCom = -__expf(-weightxyCom) * dweight;

                // weightxyCom = weightxy / delta^2
                // weightxy = (x - xo)^2 + (y - yo)^2
                // Dtype weightxy = pow(img0_pixel - img0_center, 2) + pow(img1_pixel - img1_center, 2);
                // BP
                Dtype dweightxy = 1 / (LAMDA_R * LAMDA_R) * dweightxyCom;

                // weightxy = distxsq + distysq
                // distysq = (y - yo)^2
                // Dtype distysq = pow(img1_pixel - img1_center, 2);
                // BP
                Dtype ddistxsq = dweightxy;
                // distxsq = (x - xo)^2
                // Dtype distxsq = pow(img0_pixel - img0_center, 2);
                // BP
                Dtype ddistysq = dweightxy;

                // distxsq = distx^2
                // distx: (x - xo)
                Dtype distx = img0_pixel - img0_center;
                // BP
                Dtype ddistx = 2 * distx * ddistxsq;

                // distysq = disty^2
                // disty: (y - yo)
                Dtype disty = img1_pixel - img1_center;
                // BP
                Dtype ddisty = 2 * disty * ddistysq;

                // distx = x - xo
                dx += ddistx;
                // disty = y - yo
                dy += ddisty;

                sum_kernel_x += dx;
                sum_kernel_y += dy;

                // compute the sum weight of this kernel
                weight_sum += weight;
                */
              } // end kernel width
            } // end kernel height
            //////////////////////////////////////////////////// for debug purpose
            sum_x += sum_kernel_x;
            sum_y += sum_kernel_y;
            //////////////////////////////////////////////////// debug end
            // sum_x += sum_kernel_x / (float)weight_sum;
            // sum_y += sum_kernel_y / (float)weight_sum;
          } // neighbour width
        } // neightbour height
    }
    // const int sumelems = (kernel_radius*2+1)*(kernel_radius*2+1)*bottomchannels;
    const int bot0index = ((n * bottomheight) + (m-pad_size)) * bottomwidth + (l-pad_size);
    bottom0diff[bot0index + item*bottomcount] = sum_x;
    bottom1diff[bot0index + item*bottomcount] = sum_y;
  }
}


// == Forward 

template <typename Dtype>
void EppmLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    CHECK_EQ(bottom.size(),2);
    CHECK_EQ(top.size(),1);

    const int bnum = bottom[0]->num();
    const int bchannels = bottom[0]->channels();
    const int bheight = bottom[0]->height();
    const int bwidth = bottom[0]->width();
    const int bwidthheight = bwidth * bheight;
    int eppm_type = 0;
    if(eppm_type_ == EppmParameter_EppmType_NO_SPATIAL)
      eppm_type = 1; // without spatial
    else
      eppm_type = 0; // with spatial

    const int topcount = top_width_ * top_height_ * top_channels_;
    
    dim3 threadsPerBlock(THREADS_PER_WARP * WARPS_PER_BLOCK); 
    
    cudaMemset(rbot1_->mutable_gpu_data(), 0, rbot1_->count()*sizeof(Dtype));
    cudaMemset(rbot2_->mutable_gpu_data(), 0, rbot2_->count()*sizeof(Dtype));
    
    int threads_per_block=16;
    dim3 totalBlocksRearr((bwidthheight-1)/threads_per_block+1, bchannels, bnum);
    const int pwidthheight = (bwidth + 2 * pad_size_) * (bheight + 2 * pad_size_);
    
    blob_rearrange_kernel2<Dtype><<<totalBlocksRearr,threads_per_block>>>
            (bottom[0]->gpu_data(),rbot1_->mutable_gpu_data(),bnum,bchannels,bwidth,bheight,bwidthheight,pad_size_,pwidthheight);
    
    blob_rearrange_kernel2<Dtype><<<totalBlocksRearr,threads_per_block>>>
            (bottom[1]->gpu_data(),rbot2_->mutable_gpu_data(),bnum,bchannels,bwidth,bheight,bwidthheight,pad_size_,pwidthheight);
    
    const int num = bnum;
    const int channels = bchannels;
    const int height = bheight + 2*pad_size_;
    const int width = bwidth + 2*pad_size_;
    
    const int shared_memory_per_block = (kernel_size_*kernel_size_)*bchannels;
    float* SpatialGaussian;
    cudaMalloc((void**)&SpatialGaussian, (2 * kernel_radius_ + 1) * sizeof(float));
    cuda_init_gaussian_Lookup_table(SpatialGaussian, kernel_radius_);

    // CorrelationLayer
    int topThreadCount = topcount;
    
    dim3 totalBlocksCorr(top_width_, top_height_, num);


    // computeMatchKernel<Dtype><<<totalBlocksCorr, threadsPerBlock>>>(
    //     topThreadCount,
    //     num, top_width_, top_height_, top_channels_, topcount,
    //     max_displacement_, neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_, kernel_size_, border_size_,
    //     stride1_, stride2_,
    //     width, height, channels,
    //     rbot1_->gpu_data(), rbot2_->gpu_data(), top[0]->mutable_gpu_data(),
    //     SpatialGaussian
    //     );

    CorrelateData<Dtype><<<totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(Dtype)>>>(
            topThreadCount,
            num, top_width_, top_height_, top_channels_, topcount,
            max_displacement_, neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_, kernel_size_,
            stride1_, stride2_,
            width, height, channels,
            rbot1_->gpu_data(), rbot2_->gpu_data(), top[0]->mutable_gpu_data(),
            SpatialGaussian, eppm_type
            );

    CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void EppmLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{

    // Get top diff, compute bottom diff
    const Dtype* top_diff = top[0]->gpu_diff();

    Dtype* bottom0_diff = bottom[0]->mutable_gpu_diff();
    Dtype* bottom1_diff = bottom[1]->mutable_gpu_diff();

    const Dtype* bottom0_data = bottom[0]->gpu_data();
    const Dtype* bottom1_data = bottom[1]->gpu_data();

    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();

    const int paddedheight = height + 2*pad_size_;
    const int paddedwidth = width + 2*pad_size_;

    const int bottomcount = channels * height * width;

    int botThreadCount = bottomcount;

    int eppm_type = 0;
    if(eppm_type_ == EppmParameter_EppmType_NO_SPATIAL)
      eppm_type = 1;
    else
      eppm_type = 0;
        
    // == Run kernel Backward 0
    dim3 totalBlocksBackward0(width, height, channels * num); //First dim is fastest
    dim3 threadsPerBlockBackward0(THREADS_PER_WARP * WARPS_PER_BLOCK); 
    const int buffer_size_backw0 = ((int)ceil((float)(2 * kernel_radius_) / (float)stride1_) + 1) * top_channels_;

    // init gaussian table
    float* SpatialGaussian;
    cudaMalloc((void**)&SpatialGaussian, (2 * kernel_radius_ + 1) * sizeof(float));
    cuda_init_gaussian_Lookup_table(SpatialGaussian, kernel_radius_);
    // run patch match backward kernel
    // for(int n = 0; n < num; n++) {
    // PatchMatchBackWard<Dtype><<<CAFFE_GET_BLOCKS(botThreadCount), CAFFE_CUDA_NUM_THREADS>>>(
    //     botThreadCount,
    //     num, n, top_width_, top_height_, top_channels_,
    //     max_displacement_, neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_,
    //     stride1_, stride2_,
    //     width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
    //     rbot1_->gpu_data(), rbot2_->gpu_data(),
    //     bottom0_diff,  bottom1_diff, top_diff,
    //     SpatialGaussian
    //     ); 

    // CUDA_POST_KERNEL_CHECK;
    // }

    for(int n = 0; n < num; n++) {
        //Bottom0:
        CorrelateDataBackward0<Dtype><<<CAFFE_GET_BLOCKS(botThreadCount), CAFFE_CUDA_NUM_THREADS>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            rbot1_->gpu_data(), rbot2_->gpu_data(), bottom0_diff, top_diff,
            SpatialGaussian, eppm_type
            ); 
    
        CUDA_POST_KERNEL_CHECK;
        }
        
        // == Run kernel Backward 1
        for(int n = 0; n < num; n++) {
        CorrelateDataBackward1<Dtype><<<CAFFE_GET_BLOCKS(botThreadCount), CAFFE_CUDA_NUM_THREADS>>>(
            botThreadCount,
            num, n, top_width_, top_height_, top_channels_,
            max_displacement_, neighborhood_grid_radius_, neighborhood_grid_width_, kernel_radius_,
            stride1_, stride2_,
            width, height, paddedwidth, paddedheight, channels, bottomcount, pad_size_,
            rbot1_->gpu_data(), rbot2_->gpu_data(), bottom1_diff, top_diff,
            SpatialGaussian, eppm_type
            );
    
        CUDA_POST_KERNEL_CHECK;
        }
}


INSTANTIATE_LAYER_GPU_FUNCS(EppmLayer);

}  // namespace caffe
