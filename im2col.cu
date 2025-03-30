#include "mygemm.cu"


// im2col实现，将输入数据转为patched matrix
__global__ void im2col_kernel(const double *input, double *patched_matrix, int input_size, 
                                int output_size, int kernel_size, int stride, 
                                int padding, int patched_matrix_rows, int patched_matrix_cols);

int main(int argc, char* argv[]) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // kernel_size 默认为3，padding 默认为1
    int input_size, kernel_size = 3, stride, padding = 1;
    if (argc != 3) {
        fprintf(stderr, "The program %s did not get enough parameters, please enter input_size and stride\n", argv[0]);
        exit(1);
    }

    input_size = atoi(argv[1]);
    stride = atoi(argv[2]);

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;
    int patched_matrix_cols = 3 * kernel_size * kernel_size;
    int patched_matrix_rows = output_size * output_size;

    // 默认输入通道数为3，将输入通道的卷积结果相加，输出通道数为1
    size_t input_bytes = 3 * input_size * input_size * sizeof(double);
    size_t output_bytes = output_size * output_size * sizeof(double);
    size_t kernel_bytes = kernel_size * kernel_size * 3 * sizeof(double);
    size_t patched_matrix_bytes = output_size * output_size * kernel_size * kernel_size * 3 * sizeof(double);

    double *h_input = (double*)malloc(input_bytes);
    double *h_kernel = (double*)malloc(kernel_bytes);
    double *h_output = (double*)malloc(output_bytes);

    // 初始化输入和核函数
    initialize_matrix(h_input, 3 * input_size * input_size);
    for (int i = 0; i < kernel_size * kernel_size * 3; i++) h_kernel[i] = 1.0f;

    cudaEventRecord(start, 0);

    double *d_input, *d_kernel, *d_output, *d_patched_matrix;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_kernel, kernel_bytes * 3);
    cudaMalloc(&d_output, output_bytes);
    cudaMalloc(&d_patched_matrix, patched_matrix_bytes);

    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
    // 在GPU中，kernel需要拷贝三份，对应三个通道求得卷积和
    cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel + kernel_size * kernel_size, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel + kernel_size * kernel_size * 2, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);

    dim3 block_dim(patched_matrix_cols, BLOCK_SIZE);
    dim3 grid_dim(1, (patched_matrix_rows + BLOCK_SIZE - 1) / BLOCK_SIZE); // grid 在行方向划分线程块

    
    im2col_kernel<<<grid_dim, block_dim>>>(d_input, d_patched_matrix, input_size, output_size, kernel_size, 
                                           stride, padding, patched_matrix_rows, patched_matrix_cols);
    
    // 矩阵乘法
    perform_matrix_multiply(d_patched_matrix, d_kernel, d_output, output_size * output_size, kernel_size * kernel_size * 3, 1);

    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);


    free(h_input);
    free(h_kernel);
    free(h_output);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_patched_matrix);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("im2col execution time: %f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

__global__ void im2col_kernel(const double *input, double *patched_matrix, int input_size, 
                              int output_size, int kernel_size, int stride, 
                              int padding, int patched_matrix_rows, int patched_matrix_cols) {
    // 线程索引
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 当前处理的 patched matrix 行
    int col = threadIdx.x;                          // 当前处理的 patched matrix 列

    if (row >= 0 && row < patched_matrix_rows && col >= 0 && col < patched_matrix_cols) {
        int channel = col / (kernel_size * kernel_size);
        // 对应的核索引
        int kernel_row = (col / kernel_size) % kernel_size;
        int kernel_col = col % kernel_size;

        // 获取输出特征图中的行列位置
        int output_row = row / output_size;
        int output_col = row % output_size;

        // 输入图像中对应的起始位置
        int input_row = output_row * stride - padding + kernel_row;
        int input_col = output_col * stride - padding + kernel_col;

        // 检查是否越界
        if (input_row >= 0 && input_row < input_size && input_col >= 0 && input_col < input_size) {
            patched_matrix[row * patched_matrix_cols + col] = 
                input[channel * input_size * input_size + input_row * input_size + input_col];
        } else {
            patched_matrix[row * patched_matrix_cols + col] = 0.0; // 超出范围填充为0
        }
    }

    
}
