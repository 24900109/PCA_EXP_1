# PCA: EXP-1  SUM ARRAY GPU
<h3>ENTER YOUR NAME : KIRUTHIKA N</h3>
<h3>ENTER YOUR REGISTER NO : 212224230127</h3>
<h3>DATE : 27/10/2026</h3>
<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:
```
%%writefile vector_add.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define CHECK(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("CUDA Error: %s\n", cudaGetErrorString(error)); \
        exit(1); \
    } \
}

double seconds()
{
    return (double)clock() / CLOCKS_PER_SEC;
}

void initialData(float *A, int N)
{
    for (int i = 0; i < N; i++)
        A[i] = (float)(rand() % 100) / 10.0f;
}

void sumArraysOnHost(float *A, float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}

int main()
{
    printf("CUDA Vector Addition Starting...\n");

    int dev = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    CHECK(cudaSetDevice(dev));

    printf("GPU: %s\n", prop.name);

    int N = 1 << 24;
    size_t bytes = (size_t)N * sizeof(float);

    float *A = (float *)malloc(bytes);
    float *B = (float *)malloc(bytes);
    float *C = (float *)malloc(bytes);
    float *G = (float *)malloc(bytes);

    if (!A || !B || !C || !G)
    {
        printf("Host memory allocation failed!\n");
        return 1;
    }

    srand((unsigned int)time(NULL));
    initialData(A, N);
    initialData(B, N);

    clock_t start = clock();
    sumArraysOnHost(A, B, C, N);
    double cpuTime = (double)(clock() - start) / CLOCKS_PER_SEC;

    printf("CPU Runtime: %.6f seconds\n", cpuTime);

    float *d_A, *d_B, *d_C;

    CHECK(cudaMalloc((void **)&d_A, bytes));
    CHECK(cudaMalloc((void **)&d_B, bytes));
    CHECK(cudaMalloc((void **)&d_C, bytes));

    CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

    // Change this value to 256, 512, or 1024
    int threads = 256;

    dim3 block(threads);
    dim3 grid((N + threads - 1) / threads);

    printf("Threads per block: %d\n", threads);

    // Warm-up GPU
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Measure GPU runtime
    cudaEvent_t begin, end;
    CHECK(cudaEventCreate(&begin));
    CHECK(cudaEventCreate(&end));

    CHECK(cudaEventRecord(begin));
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));

    float gpuTime;
    CHECK(cudaEventElapsedTime(&gpuTime, begin, end));

    printf("GPU Kernel Runtime: %.6f seconds\n", gpuTime / 1000.0);

    CHECK(cudaMemcpy(G, d_C, bytes, cudaMemcpyDeviceToHost));

    bool match = true;
    for (int i = 0; i < N; i++)
    {
        if (fabs(C[i] - G[i]) > 1e-5)
        {
            match = false;
            printf("Mismatch at index %d\n", i);
            break;
        }
    }

    printf("%s\n", match ? "Arrays match!" : "Arrays do not match!");

    CHECK(cudaEventDestroy(begin));
    CHECK(cudaEventDestroy(end));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    free(A);
    free(B);
    free(C);
    free(G);

    return 0;
}
```

## OUTPUT:
<img width="387" height="141" alt="1" src="https://github.com/user-attachments/assets/4f06dcc8-d329-45ff-a537-c71e95b82fc3" />


## RESULT:
Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
