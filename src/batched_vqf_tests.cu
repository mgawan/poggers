/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
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


#include "include/sorted_block_vqf.cuh"
#include "include/metadata.cuh"

#include <openssl/rand.h>






__host__ std::chrono::duration<double> insert_timing(optimized_vqf * my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){

	auto start = std::chrono::high_resolution_clock::now();


	my_vqf->sorted_bulk_insert(vals, nvals, misses);
	

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

  	return diff;
}


__host__ std::chrono::duration<double> split_insert_timing(optimized_vqf * my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){

	auto start = std::chrono::high_resolution_clock::now();

	my_vqf->attach_buffers(vals, nvals);

	cudaDeviceSynchronize();


	auto midpoint = std::chrono::high_resolution_clock::now();


	my_vqf->sorted_bulk_insert_buffers_preattached(misses);
	

	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


	std::chrono::duration<double> attach_diff = midpoint-start;
  	std::chrono::duration<double> insert_diff = end-midpoint;	
  	std::chrono::duration<double> diff = end-start;



  	std::cout << "attached in " << attach_diff.count() << ", inserted in " << insert_diff.count() << ".\n";

  	std::cout << "Inserted " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();

  	return diff;
}


__global__ void test_query_kernel(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;


	uint64_t teamID = tid / 32;
	int warpID = tid % 32;

	//if (tid > 0) return;
	if (teamID >= nvals) return;




	if(!my_vqf->query(warpID, vals[teamID])){

		my_vqf->query(warpID, vals[teamID]);

		if (warpID == 0)
		atomicAdd( (unsigned long long int *) misses, 1);
	}



	//printf("tid %llu done\n", tid);

	// //does a single thread have this issue?
	// for (uint64_t i =0; i< nvals; i++){

	// 	assert(vals[i] != 0);

	// 	my_vqf->insert(vals[i]);

	// }
	
}


__global__ void test_full_query_kernel(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;


	uint64_t teamID = tid / 32;
	int warpID = tid % 32;

	//if (tid > 0) return;
	if (teamID >= nvals) return;




	if(!my_vqf->full_query(warpID, vals[teamID])){

		my_vqf->full_query(warpID, vals[teamID]);

		if (warpID == 0)
		atomicAdd( (unsigned long long int *) misses, 1);
	}



	//printf("tid %llu done\n", tid);

	// //does a single thread have this issue?
	// for (uint64_t i =0; i< nvals; i++){

	// 	assert(vals[i] != 0);

	// 	my_vqf->insert(vals[i]);

	// }
	
}

__host__ std::chrono::duration<double> query_timing(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){

	auto start = std::chrono::high_resolution_clock::now();


	test_query_kernel<<<(32*nvals -1) / (32 * WARPS_PER_BLOCK) + 1, (32 * WARPS_PER_BLOCK)>>>(my_vqf, vals, nvals, misses);


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

  	return diff;

}


__host__ std::chrono::duration<double> full_query_timing(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){

	auto start = std::chrono::high_resolution_clock::now();


	test_full_query_kernel<<<(32*nvals -1) / (32 * WARPS_PER_BLOCK) + 1, (32 * WARPS_PER_BLOCK)>>>(my_vqf, vals, nvals, misses);


	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Full Queries per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();

  	return diff;
}


__host__ void sort_timing(optimized_vqf * my_vqf){


	auto start = std::chrono::high_resolution_clock::now();


	my_vqf->sort_and_check();

	cudaDeviceSynchronize();

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Sorted in " << diff.count() << " seconds\n";


  	return;


}

__global__ void check_hits(bool * hits, uint64_t * misses, uint64_t nitems){


	uint64_t tid = threadIdx.x + blockDim.x*blockIdx.x;

	if (tid >= nitems) return;

	if (!hits[tid]){

		atomicAdd((unsigned long long int *) misses, 1ULL);

	}
}

__host__ std::chrono::duration<double> bulk_query_timing(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){



	bool * hits;

	cudaMalloc((void **) & hits, nvals*sizeof(bool));

	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();


	

	my_vqf->bulk_query(vals, nvals, hits);

	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();



	//check hits

	check_hits<<<(nvals - 1)/ 1024 + 1, 1024>>>(hits, misses, nvals);

	cudaDeviceSynchronize();

	cudaFree(hits);

  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Bulk Queries per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();

  	return diff;
}


__host__ std::chrono::duration<double> sorted_bulk_query_timing(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){



	bool * hits;

	cudaMalloc((void **) & hits, nvals*sizeof(bool));

	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();


	

	my_vqf->sorted_bulk_query(vals, nvals, hits);

	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();



	//check hits

	check_hits<<<(nvals - 1)/ 1024 + 1, 1024>>>(hits, misses, nvals);

	cudaDeviceSynchronize();

	cudaFree(hits);

  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Sorted Bulk Queries per second: %f\n", nvals/diff.count());

  	printf("Misses %llu\n", misses[0]);  

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();

  	return diff;
}


__host__ std::chrono::duration<double> sorted_bulk_query_fp_timing(optimized_vqf* my_vqf, uint64_t * vals, uint64_t nvals, uint64_t * misses){



	bool * hits;

	cudaMalloc((void **) & hits, nvals*sizeof(bool));

	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();


	

	my_vqf->sorted_bulk_query(vals, nvals, hits);

	cudaDeviceSynchronize();
	//and insert

	auto end = std::chrono::high_resolution_clock::now();



	//check hits

	check_hits<<<(nvals - 1)/ 1024 + 1, 1024>>>(hits, misses, nvals);

	cudaDeviceSynchronize();

	cudaFree(hits);

  	std::chrono::duration<double> diff = end-start;


  	std::cout << "FP Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("FP Sorted Bulk Queries per second: %f\n", nvals/diff.count());

  	printf("Misses %llu, ratio: %f\n", misses[0], 1.0 * (nvals - misses[0])/nvals);  

  	cudaDeviceSynchronize();

  	misses[0] = 0;

  	cudaDeviceSynchronize();

  	return diff;
}



__host__ uint64_t * generate_data(uint64_t nitems){


	//malloc space

	uint64_t * vals = (uint64_t *) malloc(nitems * sizeof(uint64_t));


	//			   100,000,000
	uint64_t cap = 100000000ULL;

	for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

		uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


		RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(uint64_t));



		to_fill += togen;

		printf("Generated %llu/%llu\n", to_fill, nitems);

	}

	return vals;
}


__host__ uint64_t * load_main_data(uint64_t nitems){


	char main_location[] = "/global/cscratch1/sd/hunterm/vqf_data/main_data-32-data.txt";


	char * vals = (char * ) malloc(nitems * sizeof(uint64_t));

	//std::ifstream myfile(main_location);

	//std::string line;


	FILE * pFile;


	pFile = fopen(main_location, "rb");

	if (pFile == NULL) abort();

	size_t result;

	result = fread(vals, 1, nitems*sizeof(uint64_t), pFile);

	if (result != nitems*sizeof(uint64_t)) abort();



	// //current supported format is no spacing one endl for the file terminator.
	// if (myfile.is_open()){


	// 	getline(myfile, line);

	// 	strncpy(vals, line.c_str(), sizeof(uint64_t)*nitems);

	// 	myfile.close();
		

	// } else {

	// 	abort();
	// }


	return (uint64_t *) vals;


}

__host__ uint64_t * load_alt_data(uint64_t nitems){


	char main_location[] = "/global/cscratch1/sd/hunterm/vqf_data/fp_data-32-data.txt";


	char * vals = (char * ) malloc(nitems * sizeof(uint64_t));


	//std::ifstream myfile(main_location);

	//std::string line;


	FILE * pFile;


	pFile = fopen(main_location, "rb");

	if (pFile == NULL) abort();

	size_t result;

	result = fread(vals, 1, nitems*sizeof(uint64_t), pFile);

	if (result != nitems*sizeof(uint64_t)) abort();



	return (uint64_t *) vals;


}

int main(int argc, char** argv) {
	

	uint64_t nbits = atoi(argv[1]);

	uint64_t num_batches = atoi(argv[2]);

	double batch_percent = 1.0 / num_batches;


	uint64_t nitems = (1ULL << nbits) * .85;


	//add one? just to guarantee that the clip is correct
	uint64_t items_per_batch = 1.05*nitems * batch_percent;


	printf("Starting test with %d bits, %llu items inserted in %d batches of %d.\n", nbits, nitems, num_batches, items_per_batch);




	uint64_t * vals;
	uint64_t * dev_vals;

	uint64_t * other_vals;
	uint64_t * dev_other_vals;


	vals = load_main_data(nitems);


	uint64_t * fp_vals;

	//generate fp data to see comparison with true inserts
	fp_vals = load_alt_data(nitems);

	// vals = (uint64_t*) malloc(nitems*sizeof(vals[0]));

	// RAND_bytes((unsigned char *)vals, sizeof(*vals) * nitems);


	// other_vals = (uint64_t*) malloc(nitems*sizeof(other_vals[0]));

	// RAND_bytes((unsigned char *)other_vals, sizeof(*other_vals) * nitems);




	cudaMalloc((void ** )& dev_vals, items_per_batch*sizeof(vals[0]));

	//cudaMemcpy(dev_vals, vals, nitems * sizeof(vals[0]), cudaMemcpyHostToDevice);


	bool * inserts;


	cudaMalloc((void ** )& inserts, items_per_batch*sizeof(bool));

	cudaMemset(inserts, 0, items_per_batch*sizeof(bool));



	// cudaMalloc((void ** )& dev_other_vals, nitems*sizeof(other_vals[0]));

	// cudaMemcpy(dev_other_vals, other_vals, nitems * sizeof(other_vals[0]), cudaMemcpyHostToDevice);


	//allocate misses counter
	uint64_t * misses;
	cudaMallocManaged((void **)& misses, sizeof(uint64_t));

	misses[0] = 0;


	//change the way vqf is built to better suit test and use cases? TODO with active reconstruction for exact values / struct support
	optimized_vqf * my_vqf =  build_vqf(1ULL << nbits);


	std::chrono::duration<double>  * insert_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
	std::chrono::duration<double>  * query_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));
	std::chrono::duration<double>  * fp_diff = (std::chrono::duration<double>  *) malloc(num_batches*sizeof(std::chrono::duration<double>));

	uint64_t * batch_amount = (uint64_t *) malloc(num_batches*sizeof(uint64_t));





	printf("Setup done\n");

	//wipe_vals<<<nitems/32+1, 32>>>(dev_vals, nitems);


	cudaDeviceSynchronize();

	

	for (int batch = 0; batch< num_batches; batch++){

		//calculate size of segment

		printf("Batch %d:\n", batch);

		//runs from batch/num_batches*nitems to batch
		uint64_t start = batch*nitems/num_batches;
		uint64_t end = (batch+1)*nitems/num_batches;
		if (end > nitems) end = nitems;

		uint64_t items_to_insert = end-start;


		assert(items_to_insert < items_per_batch);

		batch_amount[batch] = items_to_insert;

		//prep dev_vals for this round
		cudaMemcpy(dev_vals, vals + start, items_to_insert*sizeof(vals[0]), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

		//launch inserts
		insert_diff[batch] += split_insert_timing(my_vqf, dev_vals, items_to_insert, misses);

		cudaDeviceSynchronize();

		cudaMemcpy(dev_vals, vals + start, items_to_insert*sizeof(vals[0]), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();


		//launch queries
		query_diff[batch] = sorted_bulk_query_timing(my_vqf, dev_vals, items_to_insert, misses);


		cudaDeviceSynchronize();

		cudaMemcpy(dev_vals, fp_vals + start, items_to_insert*sizeof(vals[0]), cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();


		//false queries
		fp_diff[batch] = sorted_bulk_query_fp_timing(my_vqf, dev_vals, items_to_insert, misses);


		cudaDeviceSynchronize();

		my_vqf->get_average_fill_block();

		cudaDeviceSynchronize();


		my_vqf->get_average_fill_team();


		cudaDeviceSynchronize();

		//keep some organized spacing
		printf("\n\n");

		fflush(stdout);

		cudaDeviceSynchronize();



	}

	std::chrono::duration<double> summed_insert_diff = std::chrono::nanoseconds::zero();

	for (int i =0; i < num_batches;i++){
		summed_insert_diff += insert_diff[i];
	}

	std::chrono::duration<double> summed_query_diff = std::chrono::nanoseconds::zero();

	for (int i =0; i < num_batches;i++){
		summed_query_diff += query_diff[i];
	}

	std::chrono::duration<double> summed_fp_diff = std::chrono::nanoseconds::zero();

	for (int i =0; i < num_batches;i++){
		summed_fp_diff += fp_diff[i];
	}



	//key_val_pair<uint64_t, uint64_t> test;
	printf("Tests Finished.\n");

	std::cout << "Queried " << nitems << " in " << summed_insert_diff.count() << " seconds\n";

	printf("Final speed: %f\n", nitems/summed_insert_diff.count());


	if (argc == 4){

		printf("Dumping into file\n");

		const char * dir = "batched_results/";

		char filename_insert[256];
		char filename_lookup[256];
		char filename_false_lookup[256];
		char filename_aggregate[256];

		const char * insert_op = "_insert_";

		snprintf(filename_insert, strlen(dir) + strlen(argv[3]) + strlen(insert_op) + strlen(argv[1]) + strlen(argv[2]) + 2, "%s%s%s%s_%s", dir, argv[3], insert_op, argv[1], argv[2]);

		const char * lookup_op = "_lookup_";

		snprintf(filename_lookup, strlen(dir) + strlen(argv[3]) + strlen(lookup_op) + strlen(argv[1]) + strlen(argv[2]) + 2, "%s%s%s%s_%s", dir, argv[3], lookup_op, argv[1], argv[2]);

		const char * fp_ops = "_fp_";

		snprintf(filename_false_lookup, strlen(dir) + strlen(argv[3]) + strlen(fp_ops) + strlen(argv[1]) + strlen(argv[2]) + 2, "%s%s%s%s_%s", dir, argv[3], fp_ops, argv[1], argv[2]);

		const char * agg_ops = "_aggregate_";

		snprintf(filename_aggregate, strlen(dir) + strlen(argv[3]) + strlen(agg_ops)+ strlen(argv[1]) + strlen(argv[2]) + 2, "%s%s%s%s_%s", dir, argv[3], agg_ops, argv[1], argv[2]);


		FILE *fp_insert = fopen(filename_insert, "w");
		FILE *fp_lookup = fopen(filename_lookup, "w");
		FILE *fp_false_lookup = fopen(filename_false_lookup, "w");
		FILE *fp_agg = fopen(filename_aggregate, "w");

		if (fp_insert == NULL) {
			printf("Can't open the data file %s\n", filename_insert);
			exit(1);
		}

		if (fp_lookup == NULL ) {
		    printf("Can't open the data file %s\n", filename_lookup);
			exit(1);
		}

		if (fp_false_lookup == NULL) {
			printf("Can't open the data file %s\n", filename_false_lookup);
			exit(1);
		}

		if (fp_agg == NULL) {
			printf("Can't open the data file %s\n", filename_aggregate);
			exit(1);
		}


		printf("Writing results to file: %s\n",  filename_insert);

		fprintf(fp_insert, "x_0 y_0\n");
		for (int i = 0; i < num_batches; i++){
			fprintf(fp_insert, "%d", i*100/num_batches);

			fprintf(fp_insert, " %f\n", batch_amount[i]/insert_diff[i].count());
		}
		printf("Insert performance written!\n");

		fclose(fp_insert);


		printf("Writing results to file: %s\n",  filename_lookup);

		fprintf(fp_lookup, "x_0 y_0\n");
		for (int i = 0; i < num_batches; i++){
			fprintf(fp_lookup, "%d", i*100/num_batches);

			fprintf(fp_lookup, " %f\n", batch_amount[i]/query_diff[i].count());
		}
		printf("lookup performance written!\n");

		fclose(fp_lookup);



		printf("Writing results to file: %s\n",  filename_false_lookup);

		fprintf(fp_false_lookup, "x_0 y_0\n");
		for (int i = 0; i < num_batches; i++){
			fprintf(fp_false_lookup, "%d", i*100/num_batches);

			fprintf(fp_false_lookup, " %f\n", batch_amount[i]/fp_diff[i].count());
		}
		printf("false_lookup performance written!\n");

		fclose(fp_false_lookup);


		printf("Writing results to file: %s\n",  filename_aggregate);

		//fprintf(fp_agg, "x_0 y_0\n");

		fprintf(fp_agg, "Aggregate inserts: %f\n", nitems/summed_insert_diff.count());
		fprintf(fp_agg, "Aggregate Queries: %f\n", nitems/summed_query_diff.count());
		fprintf(fp_agg, "Aggregate fp: %f\n", nitems/summed_fp_diff.count());



		printf("false_lookup performance written!\n");

		fclose(fp_false_lookup);



	}

	free(vals);

	free(fp_vals);

	cudaFree(dev_vals);

	cudaFree(misses);

	

	return 0;

}
