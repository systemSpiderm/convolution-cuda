#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 32 

//从lower到upper的随机浮点数生成器
double get_random_double(double lower, double upper);

// 随机生成矩阵，100到100000
void initialize_matrix(double* mat, int size);

__global__ void convolution2D(const double *input, const double *kernel, double *output, 
                                int input_size, int output_size, int kernel_size, 
                                int stride, int padding);


int main(int argc, char* argv[]) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // kernel_size 默认为3，padding 默认为1
    int input_size, kernel_size = 3, stride, padding = 1;
    if (argc != 3) {
        fprintf(stderr, "The program %s didnot get enough parameters, please enter input_size and stride\n", argv[0]);
        exit(1);
    }

    input_size = atoi(argv[1]);
    stride = atoi(argv[2]);

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;

    // 默认1个3x3核函数，输入通道数为3，将输入通道的卷积结果相加，输出通道数为1
    size_t input_bytes = 3 * input_size * input_size * sizeof(double);
    size_t output_bytes = output_size * output_size * sizeof(double);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(double);

    double *h_input = (double*)malloc(input_bytes);
    double *h_kernel = (double*)malloc(kernel_bytes);
    double *h_output = (double*)malloc(output_bytes);

    // 初始化输入和核函数
    initialize_matrix(h_input, 3 * input_size * input_size);
    for (int i = 0; i < kernel_size * kernel_size; i++) h_kernel[i] = 1.0f;

    cudaEventRecord(start, 0);

    double *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_kernel, kernel_bytes);
    cudaMalloc(&d_output, output_bytes);

    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((input_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE);


    
    convolution2D<<<grid_dim, block_dim>>>(d_input, d_kernel, d_output, input_size, 
                                         output_size, kernel_size, stride, padding);
    

    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    printf("Elapsed time: %f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_input);
    free(h_kernel);
    free(h_output);

    

    return 0;
}

double get_random_double(double lower, double upper) {
    int random_int = rand();
    return lower + (double)(random_int / (RAND_MAX + 1.0)) * (upper - lower);
}

void initialize_matrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = get_random_double(100.0, 100000.0);
    }
}

__global__ void convolution2D(const double *input, const double *kernel, double *output, 
                                int input_size, int output_size, int kernel_size, 
                                int stride, int padding) {
    // 计算线程的全局索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_i = blockIdx.y * blockDim.y + ty - padding;
    int col_i = blockIdx.x * blockDim.x + tx - padding;
    int row_o = (row_i + padding) / stride;
    int col_o = (col_i + padding) / stride;

    // 定义共享内存
    __shared__ double shared_input[BLOCK_SIZE + 2][BLOCK_SIZE + 2][3];
    __shared__ double shared_kernel[3][3];

    // 将输入数据加载到共享内存
    for (int c = 0; c < 3; c++) {
        if (row_i >= 0 && row_i < input_size && col_i >= 0 && col_i < input_size) {
            shared_input[ty][tx][c] = input[row_i * input_size * 3 + col_i * 3 + c];
        } else {
            shared_input[ty][tx][c] = 0.0;
        }
    }

    // 将核函数加载到共享内存
    if (tx < kernel_size && ty < kernel_size) {
        shared_kernel[ty][tx] = kernel[ty * kernel_size + tx];
    }

    // 同步线程，确保所有数据已加载到共享内存
    __syncthreads();

    // 执行卷积操作，只有stride整除row_o和col_o的参与运算
    if (row_o < output_size && col_o < output_size && row_i % stride == 0 && col_i % stride == 0) {
        double value = 0.0;
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                for (int c = 0; c < 3; ++c) {
                    value += shared_input[ty + i][tx + j][c] * shared_kernel[i][j];
                }
            }
        }
        output[row_o * output_size + col_o] = value;
    }
}