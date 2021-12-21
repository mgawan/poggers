/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <random>
#include <assert.h>
#include <chrono>
#include <iostream>

#include <fstream>
#include <string>
#include <algorithm>
#include <bitset>


#include "include/optimized_vqf.cuh"
//#include "include/f_team_block.cuh"
#include "include/gpu_block.cuh"
#include "include/metadata.cuh"

#include <openssl/rand.h>


__global__ void test_insert_kernel(optimized_vqf* my_vqf, uint64_t * vals, bool * inserts, uint64_t nvals, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;


	uint64_t teamID = tid / 32;
	int warpID = tid % 32;

	//if (tid > 0) return;
	if (teamID >= nvals) return;

	//vals[teamID] = teamID;

	if (!my_vqf->insert(warpID, vals[teamID], false)){



		if (warpID == 0){

			atomicAdd( (unsigned long long int *) misses, 1);

			inserts[teamID] = 1;

		}
		
	} //else {


	// 	// if (!my_vqf->query(warpID, vals[teamID])){
	// 	// 	assert(my_vqf->query(warpID, vals[teamID]));
	// 	// }
		
	// }



	//printf("tid %llu done\n", tid);

	// //does a single thread have this issue?
	// for (uint64_t i =0; i< nvals; i++){

	// 	assert(vals[i] != 0);

	// 	my_vqf->insert(vals[i]);

	// }
	
}


__global__ void bulk_insert_kernel(optimized_vqf* my_vqf, uint64_t * vals, bool * inserts, uint64_t nvals, uint64_t * misses){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	uint64_t teamID = (tid/32)*2;


	int warpID = tid % 32;

	if (teamID < my_vqf->num_blocks){


	my_vqf->buffer_insert(warpID, teamID);

	if (warpID == 0)
	atomicAdd( (unsigned long long int *) misses, my_vqf->buffer_sizes[teamID]);



	teamID +=1;

	if (teamID < my_vqf->num_blocks){


	my_vqf->buffer_insert(warpID, teamID);

	if (warpID == 0)
	atomicAdd( (unsigned long long int *) misses, my_vqf->buffer_sizes[teamID]);


	//dump remainder

	}



	}


}


__global__ void shared_bulk_insert_kernel(optimized_vqf* my_vqf, uint64_t * vals, bool * inserts, uint64_t nvals, uint64_t * misses){


	//__shared__ vqf_block extern_blocks[WARPS_PER_BLOCK];

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	uint64_t teamID = (tid/32)*REGIONS_PER_WARP;

	uint64_t trueID = (tid/32);


	int warpID = tid % 32;

	//my_vqf->buffer_insert(warpID, teamID);

 	my_vqf->multi_buffer_insert(warpID, trueID % WARPS_PER_BLOCK, teamID);


 	//this clips and breaks on the last one
	// for (int i = 0; i < REGIONS_PER_WARP; i++){

	// 	if (warpID == 0)
	// 	atomicAdd( (unsigned long long int *) misses, my_vqf->buffer_sizes[teamID + i]);

	// }


}


__global__ void bulk_query_kernel(optimized_vqf* my_vqf, uint64_t * vals, bool * inserts, uint64_t nvals, uint64_t * misses){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	uint64_t teamID = (tid/32)*2;


	int warpID = tid % 32;

	if (teamID < my_vqf->num_blocks){


	int query_misses = my_vqf->buffer_query(warpID, teamID);

	if (warpID == 0)
	atomicAdd( (unsigned long long int *) misses, query_misses);



	teamID +=1;

	if (teamID < my_vqf->num_blocks){


	query_misses = my_vqf->buffer_query(warpID, teamID);

	if (warpID == 0)
	atomicAdd( (unsigned long long int *) misses, query_misses);


	//dump remainder

	}



	}


}




//BUG: This hashes the inputs twice, gets really fucky answers because of that - pass a preshashed flag to the vqf

__global__ void bulk_finish_kernel(optimized_vqf * my_vqf, uint64_t * vals, bool * inserts, uint64_t nvals, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	uint64_t teamID = (tid/32)*2;


	int warpID = tid % 32;

	if (teamID < my_vqf->num_blocks){


	int size = my_vqf->buffer_sizes[teamID];
	for (int i =0; i < size; i++){



		if (!my_vqf->insert(warpID, my_vqf->buffers[teamID][i], true)){

		if (warpID == 0)
		atomicAdd( (unsigned long long int *) misses, my_vqf->buffer_sizes[teamID]);


		}

	}

	


	teamID +=1;

	if (teamID < my_vqf->num_blocks){


	size = my_vqf->buffer_sizes[teamID];
	for (int i =0; i < size; i++){

		if (!my_vqf->insert(warpID, my_vqf->buffers[teamID][i], true)){

		if (warpID == 0)
		atomicAdd( (unsigned long long int *) misses, my_vqf->buffer_sizes[teamID]);


		}

	}

	//dump remainder

	}



	}

}

__global__ void test_query_kernel(optimized_vqf* my_vqf, uint64_t * vals, bool * inserts, uint64_t nvals, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;


	uint64_t teamID = tid / 32;
	int warpID = tid % 32;

	//if (tid > 0) return;
	if (teamID >= nvals) return;


	if (!inserts[teamID]){



	if(!my_vqf->query(warpID, vals[teamID])){

		my_vqf->query(warpID, vals[teamID]);

		if (warpID == 0)
		atomicAdd( (unsigned long long int *) misses, 1);
	}

	}


	//printf("tid %llu done\n", tid);

	// //does a single thread have this issue?
	// for (uint64_t i =0; i< nvals; i++){

	// 	assert(vals[i] != 0);

	// 	my_vqf->insert(vals[i]);

	// }
	
}


__global__ void test_remove_kernel(optimized_vqf* my_vqf, uint64_t * vals, bool * inserts, uint64_t nvals, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;


	uint64_t teamID = tid / 32;
	int warpID = tid % 32;

	//if (tid > 0) return;
	if (teamID >= nvals) return;


	if (!inserts[teamID]){



	if(!my_vqf->remove(warpID, vals[teamID])){
		if (warpID == 0)
		atomicAdd( (unsigned long long int *) misses, 1);
	}


	}



	//printf("tid %llu done\n", tid);

	// //does a single thread have this issue?
	// for (uint64_t i =0; i< nvals; i++){

	// 	assert(vals[i] != 0);

	// 	my_vqf->insert(vals[i]);

	// }
	
}


__global__ void wipe_vals(uint64_t * vals, uint64_t nvals){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >=nvals) return;

	vals[tid] = tid;
}



__host__ void insert_timing(optimized_vqf* my_vqf, uint64_t * vals, bool * inserts, uint64_t nvals, uint64_t * misses){

	auto start = std::chrono::high_resolution_clock::now();


	test_insert_kernel<<<(32*nvals/REGIONS_PER_WARP) / (32 * WARPS_PER_BLOCK) + 1, (32 * WARPS_PER_BLOCK)>>>(my_vqf, vals, inserts, nvals, misses);


	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Inserted " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();
}


__host__ void bulk_insert_timing(optimized_vqf* my_vqf, uint64_t * vals, bool * inserts, uint64_t nvals, uint64_t * misses){

	auto start = std::chrono::high_resolution_clock::now();

	my_vqf->attach_buffers(vals, nvals);

	uint64_t num_buffers = my_vqf->get_num_buffers();

	cudaDeviceSynchronize();

	auto mid = std::chrono::high_resolution_clock::now();


	//shared_bulk_insert_kernel<<<(32*num_buffers -1) / (32 * WARPS_PER_BLOCK) + 1, (32 * WARPS_PER_BLOCK)>>>(my_vqf, vals, inserts, nvals, misses);


	bulk_insert_kernel<<<(32*num_buffers -1) / (32 * WARPS_PER_BLOCK) + 1, (32 * WARPS_PER_BLOCK)>>>(my_vqf, vals, inserts, nvals, misses);


	//bulk_finish_kernel<<<(32*num_buffers -1)/ BLOCK_SIZE + 1, BLOCK_SIZE>>>(my_vqf, vals, inserts, nvals, misses);


	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::chrono::duration<double> sort_diff = mid - start;

  	std::chrono::duration<double> insert_diff = end - mid;




  	std::cout << "Bulk Inserted " << nvals << " in " << diff.count() << " seconds\n";

  	std::cout << "Sorted " << nvals << " in " << sort_diff.count() << " seconds\n";

  	std::cout << "Items Inserted " << nvals << " in " << insert_diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nvals/diff.count());


  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();
}


__host__ void bulk_query_timing(optimized_vqf* my_vqf, uint64_t * vals, bool * inserts, uint64_t nvals, uint64_t * misses){

	auto start = std::chrono::high_resolution_clock::now();

	my_vqf->attach_buffers(vals, nvals);

	uint64_t num_buffers = my_vqf->get_num_buffers();

	cudaDeviceSynchronize();

	auto mid = std::chrono::high_resolution_clock::now();



	bulk_query_kernel<<<(32*num_buffers -1) / (32 * WARPS_PER_BLOCK) + 1, (32 * WARPS_PER_BLOCK)>>>(my_vqf, vals, inserts, nvals, misses);


	//bulk_finish_kernel<<<(32*num_buffers -1)/ BLOCK_SIZE + 1, BLOCK_SIZE>>>(my_vqf, vals, inserts, nvals, misses);


	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::chrono::duration<double> sort_diff = mid - start;

  	std::chrono::duration<double> insert_diff = end - mid;




  	std::cout << "Bulk Inserted " << nvals << " in " << diff.count() << " seconds\n";

  	std::cout << "Sorted " << nvals << " in " << sort_diff.count() << " seconds\n";

  	std::cout << "Items Inserted " << nvals << " in " << insert_diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nvals/diff.count());


  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();
}



__host__ void query_timing(optimized_vqf* my_vqf, uint64_t * vals, bool * inserts, uint64_t nvals, uint64_t * misses){

	auto start = std::chrono::high_resolution_clock::now();


	test_query_kernel<<<(32*nvals -1) / (32 * WARPS_PER_BLOCK) + 1, (32 * WARPS_PER_BLOCK)>>>(my_vqf, vals, inserts, nvals, misses);


	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Queries per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();
}


__host__ void remove_timing(optimized_vqf* my_vqf, uint64_t * vals, bool * inserts, uint64_t nvals, uint64_t * misses){

	auto start = std::chrono::high_resolution_clock::now();


	test_remove_kernel<<<(32*nvals -1) / (32 * WARPS_PER_BLOCK) + 1, (32 * WARPS_PER_BLOCK)>>>(my_vqf, vals, inserts, nvals, misses);


	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "removed " << nvals << " in " << diff.count() << " seconds\n";

  	printf("removes per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
	

	uint64_t nbits = atoi(argv[1]);


	uint64_t nitems = (1ULL << nbits) * .8;

	uint64_t * vals;
	uint64_t * dev_vals;

	uint64_t * other_vals;
	uint64_t * dev_other_vals;

	vals = (uint64_t*) malloc(nitems*sizeof(vals[0]));

	RAND_bytes((unsigned char *)vals, sizeof(*vals) * nitems);


	// other_vals = (uint64_t*) malloc(nitems*sizeof(other_vals[0]));

	// RAND_bytes((unsigned char *)other_vals, sizeof(*other_vals) * nitems);




	cudaMalloc((void ** )& dev_vals, nitems*sizeof(vals[0]));

	cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);


	bool * inserts;


	cudaMalloc((void ** )& inserts, nitems*sizeof(bool));

	cudaMemset(inserts, 0, nitems*sizeof(bool));



	// cudaMalloc((void ** )& dev_other_vals, nitems*sizeof(other_vals[0]));

	// cudaMemcpy(dev_other_vals, other_vals, nitems * sizeof(other_vals[0]), cudaMemcpyHostToDevice);


	//allocate misses counter
	uint64_t * misses;
	cudaMallocManaged((void **)& misses, sizeof(uint64_t));

	misses[0] = 0;


	optimized_vqf * my_vqf =  build_vqf(1 << nbits);


	printf("Setup done\n");

	//wipe_vals<<<nitems/32+1, 32>>>(dev_vals, nitems);


	cudaDeviceSynchronize();

	


	cudaDeviceSynchronize();

	
	bulk_insert_timing(my_vqf, dev_vals, inserts, nitems,  misses);

	cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	bulk_query_timing(my_vqf, dev_vals, inserts, nitems,  misses);


	cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();


	//remove_timing(my_vqf, dev_vals, inserts, nitems,  misses);

	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


	free(vals);

	cudaFree(dev_vals);

	cudaFree(misses);

	

	return 0;

}
