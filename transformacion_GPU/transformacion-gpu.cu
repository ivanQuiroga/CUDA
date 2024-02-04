#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cfloat>
#include <math.h>

using namespace std;

#define Bsize_addition 256
#define Bsize_minimum   128

float maxValue = FLT_MAX;

void initializeArrays(float *A, float *B, int N) {
    for (int i = 0; i < N; i++) {
        A[i] = (float)(1.5 * (1 + (5 * i) % 7) / (1 + i % 5));
        B[i] = (float)(2.0 * (2 + i % 5) / (1 + i % 7));
    }
}

__global__ void computeC_kernel_shared(float *A, float *B, float *C, int N, int Bsize) {
    extern __shared__ float sharedData[];
    float *s_A = sharedData;      // Apunta al inicio de sharedData
    float *s_B = &sharedData[Bsize]; // Apunta a la mitad de sharedData

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x; // Identificador del hilo dentro del bloque
    int blockStart = blockIdx.x * blockDim.x; // Inicio del bloque en el array global

    // Cargar los datos de A y B en la memoria compartida
    if (i < N) {
        s_A[tid] = A[i];
        s_B[tid] = B[i];
    } else {
        s_A[tid] = 0.0f;
        s_B[tid] = 0.0f;
    }
    __syncthreads();

    // Realizar el cálculo de C[i] usando la memoria compartida
    if (i < N) {
        float cValue = 0.0;
        for (int j = 0; j < Bsize; j++) {
            if (blockStart + j < N) {
                float a = s_A[j];
                float b = s_B[j];
                float a_times_i = a * i;
                if (((int)ceilf(a_times_i)) % 2 == 0) {
                    cValue += a_times_i + b;
                } else {
                    cValue -= a_times_i - b;
                }
            }
        }
        C[i] = cValue;
    }
}

__global__ void computeC_kernel_noShared(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float a = A[i];
        float b = B[i];
        float a_times_i = a * i;
        if (((int)ceilf(a_times_i)) % 2 == 0) {
            C[i] = a_times_i + b;
        } else {
            C[i] = a_times_i - b;
        }
    }
}

__global__ void computeD_kernel(float *C, float *D, int N, int Bsize) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Cargar C en la memoria compartida
    if (i < N) {
        sdata[tid] = C[i];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // Realizar la reducción en la memoria compartida
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Escribir el resultado de este bloque a D
    if (tid == 0) {
        D[blockIdx.x] = sdata[0];
    }
}

__global__ void reduceMax_kernel(float *C, float *maxVal, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Cargar C en la memoria compartida y asignar un valor inicial para maxVal
    if (i < N) {
        sdata[tid] = C[i];
    } else {
        sdata[tid] = -FLT_MAX; // Usar un valor muy bajo para inicializar el máximo
    }
    __syncthreads();

    // Realizar la reducción en la memoria compartida
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Escribir el resultado del bloque al array maxVal
    if (tid == 0) {
        maxVal[blockIdx.x] = sdata[0];
    }
}

float computeMax(float *C, int N) {
    float mx = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        mx = std::max(mx, C[i]);
    }
    return mx;
}

void printResults(float *D, float mx, int NBlocks, double computationTime, int N, int Bsize, int shared) {
    //for (int i=0; i<N;i++)   cout<<"C["<<i<<"]="<<C[i]<<endl;
    cout << "................................." << endl;
    for (int k = 0; k < NBlocks; k++)
        cout << "D[" << k << "]=" << D[k] << endl;
    cout << "................................." << endl
         << "El valor máximo en C es:  " << mx << endl;
    if(shared==1){
        cout << "Se utilizó memoria compartida" << endl;
    }else{
        cout << "No se utilizó memoria compartida" << endl;
    }
    cout << "................................." << endl;

    cout << endl
         << "N=" << N << "= " << Bsize << "*" << NBlocks << "  ........  Tiempo algoritmo secuencial (GPU)= " << computationTime << endl
         << endl;
}

int main(int argc, char *argv[]) {
    int Bsize, NBlocks, shared;
    if (argc != 4) {
        std::cout << "Uso: transformacion Num_bloques Tam_bloque Shared(0-falso o 1-verdadero)" << std::endl;
        return 0;
    } else {
        NBlocks = std::atoi(argv[1]);
        Bsize = std::atoi(argv[2]);
        shared = std::atoi(argv[3]);
        if (shared != 0 && shared != 1) {
            std::cout << "El valor de Shared debe ser 0 (falso) o 1 (verdadero)." << std::endl;
            return 0;
        }
    }

    const int N = Bsize * NBlocks;

    // Allocate arrays on host
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];
    float *D = new float[NBlocks];

    initializeArrays(A, B, N);

    float *A_device, *B_device, *C_device, *D_device, *maxVal_device; 
 

    dim3 dimBlock(Bsize_addition);
    dim3 dimGrid(ceil((float) N / dimBlock.x));
    dim3 threadsPerBlock(Bsize_minimum, 1);
    dim3 numBlocks(ceil((float) N / threadsPerBlock.x), 1);
    int sharedMemSize = threadsPerBlock.x * sizeof(float);

    // Allocate memory on the GPU
    cudaMalloc(&A_device, N * sizeof(float));
    cudaMalloc(&B_device, N * sizeof(float));
    cudaMalloc(&C_device, N * sizeof(float));
    cudaMalloc(&D_device, NBlocks * sizeof(float));
    cudaMalloc(&maxVal_device, numBlocks.x * numBlocks.y * sizeof(float));
    
    // Copy arrays A and B to the GPU
    cudaMemcpy(A_device, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Time measurement
    double t1 = clock();
    cudaDeviceSynchronize();

    // computeC_kernel
    if(shared==1){
        computeC_kernel_shared<<<numBlocks, threadsPerBlock, sharedMemSize>>>(A_device, B_device, C_device, N, Bsize);
    }else{
        computeC_kernel_noShared<<<numBlocks, threadsPerBlock>>>(A_device, B_device, C_device, N);
    }

    // Verificar errores de CUDA después de las operaciones de CUDA
    cudaError_t cudaError;
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "Error en computeC_kernel: " << cudaGetErrorString(cudaError) << std::endl;
        return 1;
    }

    // reduceMax_kernel
    reduceMax_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(C_device, maxVal_device, N);
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "Error en reduceMax_kernel: " << cudaGetErrorString(cudaError) << std::endl;
        return 1;
    }

    // computeD_kernel
    computeD_kernel<<<NBlocks, Bsize, sharedMemSize>>>(C_device, D_device, N, Bsize);
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "Error en computeD_kernel: " << cudaGetErrorString(cudaError) << std::endl;
        return 1;
    }

    // Copiar los resultados de C y D al host
    cudaMemcpy(C, C_device, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(D, D_device, NBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Liberar memoria de la GPU
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
    cudaFree(D_device);
    cudaFree(maxVal_device); 

    // Calcular máximo en el host ya que se copió C_device a C
    float mx = computeMax(C, N);
    cudaDeviceSynchronize();
    double t2 = (clock() - t1) / CLOCKS_PER_SEC;

    printResults(D, mx, NBlocks, t2, N, Bsize, shared);

    // Liberar la memoria del host
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;

    return 0;
}
