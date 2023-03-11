#ifndef POGGERS_LAYER 
#define POGGERS_LAYER


#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <assert.h>

//#include <hip/hip_cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace ml {






//Sparse neuron ML layer
//this contains a table and a list of neurons, and allows for HOGWILD lookup of neurons in the forward pass.
struct layer {


};

}

}


#endif //Layer