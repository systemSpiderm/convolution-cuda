#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>

//从lower到upper的随机浮点数生成器
double get_random_double(double lower, double upper);

// 随机生成矩阵，100到100000
void initialize_matrix(double* mat, int size);


int main(int argc, char* argv[]) {

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    int input_size, kernel_size = 3, stride, padding = 1;
    if (argc != 3) {
        fprintf(stderr, "The program %s did not get enough parameters, please enter input_size and stride\n", argv[0]);
        exit(1);
    }

    input_size = atoi(argv[1]);
    stride = atoi(argv[2]);

    // 初始化 cuDNN
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // 输入和输出的维度
    int input_n = 1; 
    int input_c = 3; 
    int input_h = input_size;
    int input_w = input_size; 

    int kernel_n = 1;
    int kernel_c = 3;
    int kernel_h = kernel_size;
    int kernel_w = kernel_size; 

    int pad_h = padding;
    int pad_w = padding;
    int stride_h = stride;
    int stride_w = stride; 

    int output_n, output_c, output_h, output_w;

    // 创建输入和输出的张量描述符
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;

    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
                                input_n, input_c, input_h, input_w);

    cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, 
                                kernel_n, kernel_c, kernel_h, kernel_w);

    cudnnSetConvolution2dDescriptor(convolution_descriptor, pad_h, pad_w, stride_h, stride_w, 1, 1,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE);

    // 获取输出张量的维度
    cudnnGetConvolution2dForwardOutputDim(convolution_descriptor, input_descriptor, kernel_descriptor,
                                            &output_n, &output_c, &output_h, &output_w);

    cudnnSetTensor4dDescriptor(output_descriptor,CUDNN_TENSOR_NCHW,CUDNN_DATA_DOUBLE,
                                output_n, output_c, output_h, output_w);

    // 分配内存
    size_t input_size_bytes = input_n * input_c * input_h * input_w * sizeof(double);
    size_t output_size_bytes = output_n * output_c * output_h * output_w * sizeof(double);
    size_t kernel_size_bytes = kernel_n * kernel_c * kernel_h * kernel_w * sizeof(double);

    double *d_input, *d_output, *d_kernel;
    cudaMalloc(&d_input, input_size_bytes);
    cudaMalloc(&d_output, output_size_bytes);
    cudaMalloc(&d_kernel, kernel_size_bytes);

    // 初始化输入和卷积核
    double *h_input = (double*)malloc(input_size_bytes);
    double *h_kernel = (double*)malloc(kernel_size_bytes);

    initialize_matrix(h_input, input_n * input_c * input_h * input_w);
    initialize_matrix(h_kernel, kernel_n * kernel_c * kernel_h * kernel_w);

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);


    cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size_bytes, cudaMemcpyHostToDevice);

    // 创建卷积前向算法描述符
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    cudnnGetConvolutionForwardAlgorithm(cudnn, input_descriptor, kernel_descriptor, 
                                        convolution_descriptor, output_descriptor, 
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolution_algorithm);

    // 获取卷积前向操作的工作空间大小
    size_t workspace_bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn,input_descriptor, kernel_descriptor,
                                            convolution_descriptor, output_descriptor,
                                            convolution_algorithm, &workspace_bytes);

    void* d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspace_bytes);

    // 执行卷积前向操作
    const double alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn,&alpha,input_descriptor,d_input,kernel_descriptor,
                            d_kernel,convolution_descriptor,convolution_algorithm,
                            d_workspace,workspace_bytes,&beta,output_descriptor,d_output);

    double *h_output = (double*)malloc(output_size_bytes);
    cudaMemcpy(h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);

    // 释放资源
    free(h_input);
    free(h_kernel);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    printf("Elapsed time: %f ms\n", elapsedTime);


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