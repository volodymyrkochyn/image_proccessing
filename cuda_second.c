%%cu

// BMP-related data types based on Microsoft's own

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// aliases for C/C++ primitive data types
// https://msdn.microsoft.com/en-us/library/cc230309.aspx
typedef uint8_t  BYTE;
typedef uint32_t DWORD;
typedef int32_t  LONG;
typedef uint16_t WORD;

// information about the type, size, and layout of a file
// https://msdn.microsoft.com/en-us/library/dd183374(v=vs.85).aspx
typedef struct
{
    WORD bfType;
    DWORD bfSize;
    WORD bfReserved1;
    WORD bfReserved2;
    DWORD bfOffBits;
} __attribute__((__packed__))
BITMAPFILEHEADER;

// information about the dimensions and color format
// https://msdn.microsoft.com/en-us/library/dd183376(v=vs.85).aspx
typedef struct
{
    DWORD biSize;
    LONG biWidth;
    LONG biHeight;
    WORD biPlanes;
    WORD biBitCount;
    DWORD biCompression;
    DWORD biSizeImage;
    LONG biXPelsPerMeter;
    LONG biYPelsPerMeter;
    DWORD biClrUsed;
    DWORD biClrImportant;
} __attribute__((__packed__))
BITMAPINFOHEADER;

// relative intensities of red, green, and blue
// https://msdn.microsoft.com/en-us/library/dd162939(v=vs.85).aspx
typedef struct
{
    BYTE rgbtBlue;
    BYTE rgbtGreen;
    BYTE rgbtRed;
} __attribute__((__packed__))
RGBTRIPLE;


unsigned char* loadImage(const char *fileNama, size_t &size)
{
    FILE *inptr = fopen(fileNama, "r");
    if (inptr == NULL)
    {
        fprintf(stderr, "Could not open %s.\n", fileNama);
        return NULL;
    }

    // read infile's BITMAPFILEHEADER
    BITMAPFILEHEADER bf;
    fread(&bf, sizeof(BITMAPFILEHEADER), 1, inptr);

    // read infile's BITMAPINFOHEADER
    BITMAPINFOHEADER bi;
    fread(&bi, sizeof(BITMAPINFOHEADER), 1, inptr);

    // ensure infile is (likely) a 24-bit uncompressed BMP 4.0
    if (bf.bfType != 0x4d42 || bf.bfOffBits != 54 || bi.biSize != 40 ||
        bi.biBitCount != 24 || bi.biCompression != 0)
    {
        fclose(inptr);
        fprintf(stderr, "Unsupported file format.\n");
        return NULL;
    }

    // determine padding for scanlines
    int padding = (4 - (bi.biWidth * sizeof(RGBTRIPLE)) % 4) % 4;
    
    size = abs(bi.biHeight) * bi.biWidth;
    unsigned char* data = (unsigned char*)malloc(size);

    // iterate over infile's scanlines
    for (int i = 0, biHeight = abs(bi.biHeight); i < biHeight; i++)
    {
        fread(data + i, sizeof(unsigned char), bi.biWidth, inptr);
        // skip over padding, if any
        fseek(inptr, padding, SEEK_CUR);
    }
    return data;
}

int sum(const unsigned char* image, size_t size)
{
    clock_t start_t;
    clock_t end_t;
    clock_t clock_delta;
    double clock_delta_ms;
    
    start_t = clock();
    int sum = 0;
    // count only one color channel
    for (size_t i = 0; i < size; i+=3)
    {
        sum += image[i];
    }
    end_t = clock();

    clock_delta = end_t - start_t;
    clock_delta_ms = ((double)clock_delta/CLOCKS_PER_SEC)*1000;
    printf("Sum time, ms\t %.4f \t\n", clock_delta_ms);
    return sum;
}


#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error at runtime: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

  
__global__ void sum_simple(unsigned char *g_ivec,  int *g_ovec, int index)
{
    extern __shared__ int sdata[];
    
    //each thread load s one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i =  (index * 1024 + blockIdx.x * blockDim.x + threadIdx.x)*3;
    sdata[tid] = g_ivec[i];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=1; s <  blockDim.x ; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_ovec[index + blockIdx.x] = sdata[0];
}

int main()
{
    const char *file = "/content/drive/My Drive/111.bmp";
    size_t size = 0;
    unsigned char* image = loadImage(file, size);
    if (image == NULL)
        return 1;
    
    printf("Image size: %zu\n", size);
    printf("Pixel sum: %d\n", sum(image, size));
    
    /*-----------------------------------------*/
    
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //ALLOCATE HOST MEM
    size_t threadsPerBlock = 1024;
    // divired by 3 means one color channel
    const size_t resultSize = size / threadsPerBlock / 3;
    int *h_result = (int *) malloc(sizeof(int) * resultSize);
    
    //ALLOCATE MEM
    int *d_result;
    unsigned char *d_image;
    cudaMalloc(&d_image, size);
    cudaMalloc(&d_result, sizeof(int) * resultSize);
    cudaCheckErrors("cudaMalloc fail \n");
    
    cudaEventRecord(start, 0);
 
    cudaCheckErrors("Kernel CALL fail \n");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    // copy from host to device
    cudaMemcpy(d_image, image, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
	  cudaCheckErrors("Memory copying filled image fail \n");
    
    cudaEventRecord(start, 0);
    
    int bound = resultSize / 1024;
    for (int i = 0; i < bound; ++i)
      sum_simple <<< 1024, threadsPerBlock, 1024*sizeof(int) >>> (d_image, d_result, i);
    cudaCheckErrors("Kernel sum_reduce_simple CALL fail \n");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf ("Time for the sum_reduce_simple kernel: %f ms\n", time);
    
    cudaMemcpy(h_result, d_result, sizeof(int) * resultSize, cudaMemcpyDeviceToHost);
	  cudaCheckErrors("Memory copying result fail \n");
    
     for (int i = 1; i < resultSize; ++i)
         h_result[0] += h_result[i];
    
    //FREE MEM
    cudaFree(d_image);
    cudaFree(d_result);
    cudaCheckErrors("cudaFree fail \n");
    
    printf ("SUM is: %d\n",h_result[0]);  
    
    free(image);
    free(h_result);
    return 0;
}
