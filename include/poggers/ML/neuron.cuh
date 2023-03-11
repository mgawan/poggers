#ifndef POGGERS_NEURON
#define POGGERS_NEURON


#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <assert.h>

//#include <hip/hip_cooperative_groups.h>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace ml {






//Sparse neuron
//contains weights, activation, bias, and an activation function

template <typename activation_function>
struct neuron {




};

}

}


#endif //Layer