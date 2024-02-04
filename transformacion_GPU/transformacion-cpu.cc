#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace std;

void initializeArrays(float *A, float *B, int N) {
    // Initialize arrays A and B
    for (int i = 0; i < N; i++) {
        A[i] = (float)(1.5 * (1 + (5 * i) % 7) / (1 + i % 5));
        B[i] = (float)(2.0 * (2 + i % 5) / (1 + i % 7));
    }
}

void computeCandD(float *A, float *B, float *C, float *D, int NBlocks, int Bsize) {
    // Compute C[i], d[K] and mx
    for (int k = 0; k < NBlocks; k++) {
        int istart = k * Bsize;
        int iend = istart + Bsize;
        for (int i = istart; i < iend; i++) {
            C[i] = 0.0;
            for (int j = istart; j < iend; j++) {
                float a = A[j] * i;
                if ((int)ceil(a) % 2 == 0)
                    C[i] += a + B[j];
                else
                    C[i] += a - B[j];
            }
        }
    }
}

float computeMax(float *C, int N) {
    // Compute mx
    float mx = C[0];
    for (int i = 1; i < N; i++) {
        mx = max(C[i], mx);
    }
    return mx;
}

void computeDblocks(float *C, float *D, int NBlocks, int Bsize) {
    // Compute d[K]
    for (int k = 0; k < NBlocks; k++) {
        int istart = k * Bsize;
        int iend = istart + Bsize;
        D[k] = 0.0;
        for (int i = istart; i < iend; i++) {
            D[k] += C[i];
        }
    }
}

void printResults(float *D, float mx, int NBlocks, double computationTime, int N, int Bsize) {
    //for (int i=0; i<N;i++)   cout<<"C["<<i<<"]="<<C[i]<<endl;
    cout << "................................." << endl;
    for (int k = 0; k < NBlocks; k++)
        cout << "D[" << k << "]=" << D[k] << endl;
    cout << "................................." << endl
         << "El valor máximo en C es:  " << mx << endl;

    cout << endl
         << "N=" << N << "= " << Bsize << "*" << NBlocks << "  ........  Tiempo algoritmo secuencial (solo CPU)= " << computationTime << endl
         << endl;
}

int main(int argc, char *argv[]) {
    int Bsize, NBlocks;
    if (argc != 3) {
        cout << "Uso: transformacion Num_bloques Tam_bloque  " << endl;
        return (0);
    } else {
        NBlocks = atoi(argv[1]);
        Bsize = atoi(argv[2]);
    }

    const int N = Bsize * NBlocks;

    //* pointers to host memory */
    //* Allocate arrays a, b and c on host*/
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];
    float *D = new float[NBlocks];
    float mx; // maximum of C

    initializeArrays(A, B, N);

  
    // Time measurement  
    double t1 = clock();

    computeCandD(A, B, C, D, NBlocks, Bsize);

    double t2 = (clock() - t1) / CLOCKS_PER_SEC;

    mx = computeMax(C, N);

    computeDblocks(C, D, NBlocks, Bsize);

    printResults(D, mx, NBlocks, t2, N, Bsize);

    //* Free the memory */
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;

    return 0;
}
