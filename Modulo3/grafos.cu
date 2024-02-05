
__device__ void task1() { /* ... */ }
__device__ void task2() { /* ... */ }
__device__ void task3() { /* ... */ }
__device__ void task4() { /* ... */ }
__device__ void task5() { /* ... */ }
__device__ void task6() { /* ... */ }
__device__ void task7() { /* ... */ }
__device__ void task8() { /* ... */ }
__device__ void task9() { /* ... */ }

__global__ void graphExecutionKernel(bool *taskCompleted) {
    int threadId = threadIdx.x;
    
    if (threadId == 0) {
        task8();
        taskCompleted[8] = true;
        __threadfence(); 
        __syncthreads();

        task5();
        taskCompleted[5] = true;
        __threadfence(); 
        __syncthreads(); 

        task2();
        taskCompleted[2] = true;
        __threadfence(); // Ensures that the write to taskCompleted[2] is visible to all threads.
    }
    
    if (threadId == 1) {
        task9();
        taskCompleted[9] = true;
        __threadfence(); 
        __syncthreads();

        task6();
        taskCompleted[6] = true;
        __threadfence(); 
    }
    
    if (threadId == 2) {
        while (!taskCompleted[6]) { 
            __threadfence(); 
        }
        task7();
        taskCompleted[7] = true;
        __threadfence(); 
    }
    
    if (threadId == 3) {
        while (!taskCompleted[2]) { 
            __threadfence(); 
        }
        task4();
        taskCompleted[4] = true;
        __threadfence(); 
    }
    
    __syncthreads(); 

  
    if (threadId == 0) {
        while (!taskCompleted[2] || !taskCompleted[3] || !taskCompleted[7]) {
            __threadfence(); 
        }
        task1();
    }
}

int main() {
  
    bool *d_taskCompleted;
    cudaMalloc((void **)&d_taskCompleted, 10 * sizeof(bool)); 
    cudaMemset(d_taskCompleted, 0, 10 * sizeof(bool)); 
    
   
    graphExecutionKernel<<<1, 4>>>(d_taskCompleted);
    
  
    cudaDeviceSynchronize();

  
    cudaFree(d_taskCompleted);
    
    return 0;
}
