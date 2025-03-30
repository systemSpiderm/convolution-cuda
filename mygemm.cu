#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 32

double get_random_double(double lower, double upper) {
    int random_int = rand();
    return lower + (double)(random_int / (RAND_MAX + 1.0)) * (upper - lower);
}

void initialize_matrix(double *matrix, int N) {
    for (int i = 0; i < N; i++) {
        matrix[i] = get_random_double(100.0, 100000.0);
    }
}

__global__ void matrix_multiply_shared(double* A, double* B, double* C, int M, int N, int K) {
    __shared__ double shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double shared_B[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    double sum = 0.0f;

    // 循环分块计算
    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        
        if (row < M && t * BLOCK_SIZE + tx < N) 
            shared_A[ty][tx] = A[row * N + t * BLOCK_SIZE + tx];
        else 
            shared_A[ty][tx] = 0.0f;

        if (col < K && t * BLOCK_SIZE + ty < N) 
            shared_B[ty][tx] = B[(t * BLOCK_SIZE + ty) * K + col];
        else 
            shared_B[ty][tx] = 0.0f;
        
        __syncthreads(); 

        // 计算子块
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += shared_A[ty][i] * shared_B[i][tx];
        }

        __syncthreads(); // 确保共享内存不被提前覆盖
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

void perform_matrix_multiply(double* A, double* B, double* C, int M, int N, int K) {
    double *d_A, *d_B, *d_C;
    size_t size_A = M * N * sizeof(double);
    size_t size_B = N * K * sizeof(double);
    size_t size_C = M * K * sizeof(double);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_multiply_shared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


