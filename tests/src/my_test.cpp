
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>

__global__ void myker(int* arr_a){
    int a = 0;
    
    int compare = arr_a[1] + 1;
    printf("val at address 1: %d \n", compare);
    a = atomicCAS((arr_a+1), compare, 100);

    printf("a is: %d\n",a);
    printf("new val is: %d\n ",arr_a[1]);
}

int main(){
    int *host_a = new int[1000];
    int *dev_a;

    for(int i = 0; i < 1000; i++){
        host_a[i] = i*76;
    }

    hipMalloc(&dev_a, sizeof(int)*1000);
    hipMemcpy(dev_a, host_a, sizeof(int)*1000, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(myker, dim3(1), dim3(1), 0, 0, dev_a);

    return 0;

}