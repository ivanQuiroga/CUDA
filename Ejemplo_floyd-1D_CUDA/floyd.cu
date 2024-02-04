#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"

using namespace std;

#define default_blocksize 6

__global__ void floyd_kernel(int * M, const int nverts, const int k) {
    //printf("CUDA kernel launch \n");
    int ij = threadIdx.x + blockDim.x * blockIdx.x;
    int i= ij / nverts;
    int j= ij - i * nverts;

    if (i < nverts && j < nverts) {
        int Mij = M[ij];
        if (i != j && i != k && j != k) {
            int Mikj = M[i * nverts + k] + M[k * nverts + j];
            Mij = (Mij > Mikj) ? Mikj : Mij;
            M[ij] = Mij;
        }
    }
}

__global__ void floyd_kernel_2D(int * M, const int nverts, const int k) {
    //printf("CUDA kernel2D launch \n");
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int ij = iy * nverts + ix;

    if (ix < nverts && iy < nverts) {
        int Mij = M[ij];
        if (ix != iy && ix != k && iy != k) {
            int Mikj = M[iy * nverts + k] + M[k * nverts + ix];
            Mij = (Mij > Mikj) ? Mikj : Mij;
            M[ij] = Mij;
        }
    }
}

void init_cuda(int &devID, cudaDeviceProp &props) {
    int num_devices;
    cudaError_t err;

    err = cudaGetDeviceCount(&num_devices);
	if (err != cudaSuccess) { 
        cerr << "ERROR detecting CUDA devices......" << endl;
        exit(-1);
    }

    cout<<endl<<"....................................................."<<endl;
    cout << num_devices <<" CUDA-enabled  GPUs detected in this computer system"<<endl;
    cout<<"....................................................."<<endl;

    for (int i = 0; i < num_devices; i++) {
        devID = i;
        err = cudaGetDeviceProperties(&props, devID);
        cout<<"Device "<<devID<<": "<< props.name <<" with Compute Capability: "<<props.major<<"."<<props.minor<<endl;
        if (err != cudaSuccess) {
            cerr << "ERROR getting CUDA devices" << endl;
            exit(-1);
        }
    }
    devID = 0;  // Usando el dispositivo 0 por defecto
	cout<<"Using Device "<<devID<<endl;
	cout<<"....................................................."<<endl;
    err = cudaSetDevice(devID);
    if (err != cudaSuccess) {
        cerr << "ERROR setting CUDA device" << devID << endl;
        exit(-1);
    }
}

Graph read_graph(char* filename) {
    Graph G;
    G.lee(filename);
	//cout << "The input Graph:"<<endl;
	//G.imprime();
    return G;
}

double run_floyd_gpu(int *d_In_M, int *c_Out_M, int nverts, int nverts2, int size, int blocksize, int *A) {
    double time = clock();
	const int niters = nverts;
	cudaError_t err;
	
    err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 

    int threadsPerBlock = blocksize;
    int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int k = 0; k < niters; k++) {
        // Kernel Launch
        floyd_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_In_M, nverts, k);

		err = cudaGetLastError();
	    if (err != cudaSuccess) {
	  	    fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
	  	    exit(EXIT_FAILURE);
		}
    }

	err = cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR CUDA MEM. COPY" << endl;
	} 
    return (clock() - time) / CLOCKS_PER_SEC;
}

double run_floyd_gpu_2D(int *d_In_M, int *c_Out_M, int nverts, int blocksize, int *A) {
    double time = clock();
    const int niters = nverts;
    int size = nverts * nverts * sizeof(int);
    cudaError_t err;

    err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "ERROR CUDA 2D MEM. COPY" << endl;
    }

    dim3 threadsPerBlock(blocksize, blocksize);
    dim3 blocksPerGrid((nverts + blocksize - 1) / blocksize, (nverts + blocksize - 1) / blocksize);

    for (int k = 0; k < niters; k++) {
        floyd_kernel_2D<<<blocksPerGrid, threadsPerBlock>>>(d_In_M, nverts, k);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cerr << "Failed to launch kernel 2D! ERROR= " << err << endl;
            exit(EXIT_FAILURE);
        }
    }

    err = cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "ERROR CUDA 2D MEM. COPY" << endl;
    }
    return (clock() - time) / CLOCKS_PER_SEC;
}

double run_floyd_cpu(int *A, int nverts) {
    double time = clock();	
	const int niters = nverts;
	// BUCLE PPAL DEL ALGORITMO
    int inj, in, kn;
    for (int k = 0; k < niters; k++) {
        kn = k * nverts;
        for (int i = 0; i < nverts; i++) {
            in = i * nverts;
            for (int j = 0; j < nverts; j++)
                if (i != j && i != k && j != k) {
                    inj = in + j;
                    A[inj] = min(A[in + k] + A[kn + j], A[inj]);
                }
        }
    }
    return (clock() - time) / CLOCKS_PER_SEC;
}

bool check_errors(int *c_Out_M, Graph &G, int nverts) {
    bool errors = false;
    for (int i = 0; i < nverts; i++) {
        for (int j = 0; j < nverts; j++) {
            if (abs(c_Out_M[i * nverts + j] - G.arista(i, j)) > 0) {
                cout << "Error (" << i << "," << j << ")   " << c_Out_M[i * nverts + j] << "..." << G.arista(i, j) << endl;
                errors = true;
            }
        }
    }
    return errors;
}

int main(int argc, char *argv[]) {

	int blocksize;
    if (argc == 2) {
        blocksize = default_blocksize;  
    } else if (argc == 3) {
        blocksize = stoi(argv[2]);  
    } else{
        cerr << "Sintaxis: " << argv[0] << " <archivo de grafo> <blocksize>" << endl;
        return -1;
    }

    //Get GPU information
    int devID;
    cudaDeviceProp props;
	cudaError_t err;
    init_cuda(devID, props);

	// Declaration of the Graph object
    Graph G = read_graph(argv[1]);

    const int nverts = G.vertices;
	const int nverts2 = nverts * nverts;
	
    //1D
	int *c_Out_M = new int[nverts2];
    //2D
    int *c_Out_M2D = new int[nverts2];

	int size = nverts2*sizeof(int);
    int *d_In_M = NULL;

	err = cudaMalloc((void **) &d_In_M, size);
	if (err != cudaSuccess) {
		cerr << "ERROR MALLOC" << endl;
	}

	// Get the integer 2D array for the dense graph
    int *A = G.Get_Matrix();

    //**************************************************************************
	// GPU phase
	//**************************************************************************	
	cout<< "Blocksize= " << blocksize << endl;
    double Tgpu = run_floyd_gpu(d_In_M, c_Out_M, nverts, nverts2, size, blocksize, A);  
    double Tgpu2D = run_floyd_gpu_2D(d_In_M, c_Out_M2D, nverts, blocksize, A);
	
    //**************************************************************************
	// CPU phase
	//**************************************************************************
    double Tcpu = run_floyd_cpu(A, nverts);

    double SGPU1D = Tcpu / Tgpu;
    double SGPU2D = Tcpu / Tgpu2D;

    cout <<"....................................................."<<endl;
    cout <<"....................................................."<<endl;
    cout << "Time spent on GPU= " << Tgpu << endl;
    cout << "Time spent on GPU(2D)= " << Tgpu2D << endl;
    cout << "Time spent on CPU= " << Tcpu << endl;
    cout <<"....................................................."<<endl;
    cout << "Speedup TCPU/TGPU= " << SGPU1D << endl;
    cout << "Speedup TCPU/TGPU(2D)= " << SGPU2D << endl;

    bool error = check_errors(c_Out_M, G, nverts);
    bool error2D = check_errors(c_Out_M2D, G, nverts);

	cout<<"....................................................."<<endl;
	cout<<"....................................................."<<endl;
    if (!error) 
		cout<<"No errors found ............................"<<endl;    
    if (!error2D)
	    cout<<"No errors found in 2D ......................"<<endl;
    cout<<"....................................................."<<endl<<endl;

    cudaFree(d_In_M);
    delete[] c_Out_M;
    delete[] c_Out_M2D;
    return 0;
}
