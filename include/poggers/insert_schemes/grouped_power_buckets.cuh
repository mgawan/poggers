#ifndef GROUPED_POWER_BUCKETS 
#define GROUPED_POWER_BUCKETS


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <assert.h>

#include <cooperative_groups.h>

#include <iostream>

//#include <poggers/hash_schemes/murmurhash.cuh>

namespace cg = cooperative_groups;


namespace poggers {

namespace insert_schemes {


//The insert schemes are in charge of device side inserts
//they handle memory allocations for the host-side table
// and most of the logic involved in inserting/querying
template <typename T, std::size_t num_buckets, std::size_t spacing>
struct __attribute__ ((__packed__)) bucket_wrapper{

	T internal_buckets[num_buckets]; 

	char spaces[spacing];

};


//Power of N Hashing
//Given a probing scheme with depth N, this insert strategy
//queries the fill of all N buckets

//TODO = separate bucket size from cuda size - use static assert to enforce correctness
// bucket_size/NUM_BUCKETS

//TODO - get godbolt to compile a cmake project? that would be nice.
template <typename Key, typename Val, std::size_t Partition_Size, std::size_t Bucket_Size, template <typename, typename, size_t, size_t> class Internal_Rep, std::size_t Max_Probes, template <typename, std::size_t> class Hasher, template<typename, std::size_t, template <typename, std::size_t> class, std::size_t> class Probing_Scheme, std::size_t Buckets_per_cache_line>

//template <typename Key, std::size_t Partition_Size, template <typename, std::size_t> class Hasher, std::size_t Max_Probes>
//template <typename Hasher1, typename Hasher2, std::size_t Max_Probes>
struct __attribute__ ((__packed__)) power_of_n_insert_grouped_buckets {


	using int_rep_type = Internal_Rep<Key, Val, Partition_Size, Bucket_Size>;

	//tag bits change based on the #of bytes allocated per block
private:

	//must be aligned to the cache width
	static_assert(sizeof(int_rep_type)*Buckets_per_cache_line <= 128);

	using rep_type = bucket_wrapper<int_rep_type, Buckets_per_cache_line, 128-sizeof(int_rep_type)*Buckets_per_cache_line>;

	rep_type * slots;

	const uint64_t num_buckets;
	const uint64_t seed;




public:



	//typedef key_type Hasher::key_type;
	//using key_type = Key;
	using probing_scheme_type = Probing_Scheme<Key,Partition_Size, Hasher, Max_Probes>;
	using my_type = power_of_n_insert_grouped_buckets<Key, Val, Partition_Size, Bucket_Size, Internal_Rep, Max_Probes, Hasher, Probing_Scheme, Buckets_per_cache_line>;



	//using partition_size = Hasher1::Partition_Size;

 
	
	//typedef key_val_pair<Key> Key;

	//init happens by a single thread on CPU/GPU
	//no cg needed

	//pull in hasher - need it's persistent storage

	//define default constructor so cuda doesn't yell
	__host__ __device__ power_of_n_insert_grouped_buckets(): num_buckets(0), seed(0) {};


	//only allowed to be defined on CPU
	__host__ power_of_n_insert_grouped_buckets(rep_type * ext_slots, uint64_t ext_nslots, uint64_t ext_seed): num_buckets(ext_nslots), seed(ext_seed){
		
		slots = ext_slots;
	}


	__host__ static my_type * generate_on_device(uint64_t ext_nslots, uint64_t ext_seed){

		rep_type * ext_slots;

		//min_buckets is the number of bucket objects required
		uint64_t min_buckets = (ext_nslots-1)/(Bucket_Size)+1;

		uint64_t min_cache_lines = (min_buckets-1)/Buckets_per_cache_line+1;

		
		//printf("Min buckets requested! %llu, grouped into %llu groups\n", min_buckets, min_cache_lines);



		//uint64_t true_nslots = min_buckets;

		//printf("Constructing table wtih %llu buckets, %llu slots\n", min_buckets, min_buckets*Bucket_Size);


		uint64_t bytes_used =  min_cache_lines*sizeof(rep_type);
		std::cout << "Using " << bytes_used << " bytes, " << 1.0*bytes_used/ext_nslots << " bytes per item, " << sizeof(int_rep_type) << " per bucket" << std::endl;
		//printf("Using %llu bytes, %llu bytes per item, %llu per bucket\n", min_cache_lines*sizeof(rep_type), 1.0*min_cache_lines*sizeof(rep_type)/(ext_nslots), sizeof(int_rep_type));


		cudaMalloc((void **)& ext_slots, min_cache_lines*sizeof(rep_type));
		cudaMemset(ext_slots, 0, min_cache_lines*sizeof(rep_type));

		my_type host_version (ext_slots, min_buckets, ext_seed);

		my_type * dev_version;

		cudaMalloc((void **)&dev_version, sizeof(my_type));

		cudaMemcpy(dev_version, &host_version, sizeof(my_type), cudaMemcpyHostToDevice);

		return dev_version;



	}

	__host__ static void free_on_device(my_type * dev_version){


		my_type host_version;

		cudaMemcpy(&host_version, dev_version, sizeof(my_type), cudaMemcpyDeviceToHost);

		cudaFree(host_version.slots);

		cudaFree(dev_version);

		return;

	}

	//Given a bucketID, attempt an insert
	// This simplifies the logic of the insert schemes
	// Hopefully without affecting performance?
	__device__ __inline__ bool insert_into_bucket(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val, uint64_t insert_slot){



       		//insert_slot = insert_slot*Bucket_Size;// + insert_tile.thread_rank();

       		//printf("checking_for_slot\n");

 			return slots[insert_slot / Buckets_per_cache_line].internal_buckets[insert_slot % Buckets_per_cache_line].insert(insert_tile, key, val);

	}

		__device__ __inline__ bool query_into_bucket(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val & ext_val, uint64_t insert_slot){


			//if (insert_tile.thread_rank() == 0) printf("In query!\n");

			return slots[insert_slot / Buckets_per_cache_line].internal_buckets[insert_slot % Buckets_per_cache_line].query(insert_tile, key,ext_val);

	}

	__device__ __inline__ int check_fill_bucket(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val, uint64_t insert_slot){


		return slots[insert_slot / Buckets_per_cache_line].internal_buckets[insert_slot % Buckets_per_cache_line].get_fill(insert_tile);





	}

	__device__ __inline__ bool remove_from_bucket(cg::thread_block_tile<Partition_Size> insert_tile, Key key, uint64_t insert_slot){


			//if (insert_tile.thread_rank() == 0) printf("In query!\n");



     		return slots[insert_slot / Buckets_per_cache_line].internal_buckets[insert_slot % Buckets_per_cache_line].remove(insert_tile, key);

	}

	__device__ __inline__ bool insert(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val val){

		//first step is to init probing scheme

		//if(insert_tile.thread_rank() == 0) printf("Inside of power of n insert\n");

		uint64_t buckets[Max_Probes];
		int fill [Max_Probes];


		probing_scheme_type insert_probing_scheme(seed);

		int i = 0;

		int min_fill = Bucket_Size;

		for (uint64_t insert_slot = insert_probing_scheme.begin(key); insert_slot != insert_probing_scheme.end(); insert_slot = insert_probing_scheme.next()){

       		insert_slot = insert_slot % num_buckets;

       		buckets[i] = insert_slot;


       		int current_fill = check_fill_bucket(insert_tile, key, val, insert_slot);

       		if (current_fill < Bucket_Size*.75){

       			if (insert_into_bucket(insert_tile, key, val , insert_slot)) return true;

       			//if we failed it must be full
       			current_fill = Bucket_Size;

       		}

       		fill[i] = current_fill;

       		if (fill[i] < min_fill) min_fill = fill[i];

       		i+=1;

       	}

       	// if (insert_tile.thread_rank() == 0){

       	// 	printf("Max Probes: %llu\n", Max_Probes);

       	// 	for (int i =0; i < Max_Probes; i++){

       	// 		printf("%llu\n", buckets[i]);
       	// 	}

       	// }

       	i = min_fill;

       	min_fill = Bucket_Size;

       	int count = 0;

       	while (i < Bucket_Size){

       		for (int j = 0; j < Max_Probes; j+=1){

       			if (fill[j] == i){

       				//double check me
       				//int bucket_to_try = insert_tile.shfl(j, __ffs(ballot_result)-1);

       				if (insert_into_bucket(insert_tile, key, val, buckets[j])){

       					// if (insert_tile.thread_rank() == 0){
       					// 	printf("Succeeded in bucket %d %llu\n", j, buckets[j]);
       					// }

       					return true;
       				}

       			}

       			if (fill[j] > i && fill[j] < min_fill){
       				min_fill = fill[j];
       			}


       		}

       		i = min_fill;

     	  	min_fill = Bucket_Size;

     	  	count +=1;

     	  	//if (count > Bucket_Size && insert_tile.thread_rank() == 0) printf("Stalling\n");


       	}

  //      	if (insert_tile.thread_rank() == 0){
		// 	printf("Failed... Current Fills\n");
	 //       		for (int i =0; i < Max_Probes; i++){

	 //       			printf("%llu: %d\n", buckets[i], fill[i]);

	 //       		}
		// }

     	return false;

	}

	__device__ __inline__ bool query(cg::thread_block_tile<Partition_Size> insert_tile, Key key, Val& ext_val){

		//first step is to init probing scheme

		//if (insert_tile.thread_rank() == 0) printf("Starting outer query!\n");


		probing_scheme_type insert_probing_scheme(seed);

		for (uint64_t insert_slot = insert_probing_scheme.begin(key); insert_slot != insert_probing_scheme.end(); insert_slot = insert_probing_scheme.next()){

       			
       	insert_slot = insert_slot % num_buckets;


       		

			if (query_into_bucket(insert_tile, key, ext_val, insert_slot)){

				//if (insert_tile.thread_rank() == 0) printf("Found in %llu!\n", insert_slot);
				return true;
			}
     	

		}

		//if (insert_tile.thread_rank() == 0) printf("Could not find %d\n", key);

		return false;


	}

	__device__ __inline__ bool remove(cg::thread_block_tile<Partition_Size> insert_tile, Key key){

		//first step is to init probing scheme

		//if (insert_tile.thread_rank() == 0) printf("Starting outer query!\n");


		probing_scheme_type insert_probing_scheme(seed);

		for (uint64_t insert_slot = insert_probing_scheme.begin(key); insert_slot != insert_probing_scheme.end(); insert_slot = insert_probing_scheme.next()){

       			
       		insert_slot = insert_slot % num_buckets;


       		

			if (remove_from_bucket(insert_tile, key, insert_slot)){

				//if (insert_tile.thread_rank() == 0) printf("Found in %llu!\n", insert_slot);
				return true;
			}
     	

		}

		//if (insert_tile.thread_rank() == 0) printf("Could not find %d\n", key);

		return false;


	}



};

template<std::size_t Buckets_per_cache_line> struct blocked_bucket_insert {
   template <typename Key, typename Val, std::size_t Partition_Size, std::size_t Bucket_Size, template <typename, typename, size_t, size_t> class Internal_Rep, std::size_t Max_Probes, template <typename, std::size_t> class Hasher, template<typename, std::size_t, template <typename, std::size_t> class, std::size_t> class Probing_Scheme>
   using representation = power_of_n_insert_grouped_buckets<Key, Val, Partition_Size, Bucket_Size, Internal_Rep, Max_Probes, Hasher, Probing_Scheme, Buckets_per_cache_line>;
};


//insert_schecmes
}


//poggers
}


#endif //GPU_BLOCK_