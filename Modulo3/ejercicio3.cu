#include <stdio.h>
#include <cuda_runtime.h>

#define VALOR_PRUEBA 1.0

__global__ void modifyAndSumMatrix(double *A, double *sum, int M, int N) {
    // Cálculo de índices: Descomposición de dominio
    // Cada hilo obtiene su índice único para trabajar en un elemento específico de la matriz.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Modificación de la matriz en paralelo: Descomposición Funcional
    // Esta sección es responsable de la modificación de la matriz.
    if (row < M-1 && col < N) {
        int idx = row * N + col;
        A[idx] *= 2.0 * A[(row+1) * N + col];
    }
    // Asegura que todas las modificaciones de la matriz se han completado.
    __syncthreads();

    // Reducción paralela en bloques de hilos: Descomposición de dominio
    // Cada hilo suma independientemente un elemento de la matriz a un sumador global de manera atómica.
    // La tarea de sumar toda la matriz se divide en pequeñas sumas individuales.
    if (row < M && col < N) {
        atomicAdd(sum, A[row * N + col]);
    }
}


double modifySumMatrix(double * A, int M, int N) {
    for (int i=0; i<M-1; i++) {
        for (int j=0; j<N; j++) {
            A[i*N+j] = 2.0 * A[(i+1)*N+j];
        }
    }
    double suma = 0.0;
    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            suma = suma + A[i*N+j];
        }
    }
    return suma;
}

void initializeVector(double** vector_A, int M, int N) {
    size_t size = M * N * sizeof(double);
    *vector_A = new double[size];

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            (*vector_A)[i * N + j] = VALOR_PRUEBA; 
        }
    }
}

int main() {
    int M = 2;
    int N = 8;
    size_t size = M * N * sizeof(double);
    
    // Alojar memoria en el host
    double *h_A = (double *)malloc(size);
    double h_sum = 0.0;

    // Inicializar la matriz con valores de ejemplo
    for(int i = 0; i < M*N; ++i) {
        h_A[i] = VALOR_PRUEBA;
    }

    // Alojar memoria en el dispositivo
    double *d_A, *d_sum;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_sum, sizeof(double));

    // Copiar la matriz del host al dispositivo
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(double), cudaMemcpyHostToDevice);

    // Definir la configuración del kernel
    dim3 threadsPerBlock(N, 1);
    dim3 blocksPerGrid(1, M);
    
    // Lanzar el kernel
    modifyAndSumMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_sum, M, N);
    
    // Copiar el resultado de vuelta al host
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

    // Liberar la memoria del dispositivo
    cudaFree(d_A);
    cudaFree(d_sum);

    double* vector_A;
    initializeVector(&vector_A, M, N);
    double sum_A = modifySumMatrix(vector_A, M, N);

    // Imprimir la suma
    printf("La suma de los elementos paralelos de la matriz es: %f\n", h_sum);
    printf("La suma de los elementos de la matriz es: %f\n", sum_A);

    // Liberar la memoria del host
    free(h_A);

    return 0;
}
