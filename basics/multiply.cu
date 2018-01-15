// https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/
#include <iostream>
#include <math.h>
#include <random>

__global__
void multiply(int n, float *x, float *y, float *z) {
    int index = blockIdx.x * blockDim.x +  threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        z[i] = x[i] * y[i];
    }
}

int main() {
    int N = 1 << 20;
    float *x, *y, *z;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&z, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = (float)(random() % N) / (float) N;
        y[i] = (float)(random() % N) / (float) N;
    }

    multiply<<<1, 1>>>(N, x, y, z);
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(z[i] - (x[i] * y[i])));
    }
    std::cout << "Max error " << maxError << std::endl;
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
}