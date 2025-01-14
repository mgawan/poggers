/*
 * ============================================================================
 *
 *        Authors:  
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 *
 *        About:
 *          This file is for isolating a bug in the TCF delete marking
 *          - insert a few items, measure fill, delete, and then recalculate.
 *          
 *
 * ============================================================================
 */




//#include "include/templated_quad_table.cuh"
#include <poggers/metadata.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/probing_schemes/linear_probing.cuh>
#include <poggers/probing_schemes/double_hashing.cuh>
#include <poggers/probing_schemes/power_of_two.cuh>
#include <poggers/insert_schemes/single_slot_insert.cuh>
#include <poggers/insert_schemes/bucket_insert.cuh>
#include <poggers/insert_schemes/power_of_n.cuh>
#include <poggers/representations/key_val_pair.cuh>
#include <poggers/representations/shortened_key_val_pair.cuh>
#include <poggers/sizing/default_sizing.cuh>
#include <poggers/tables/base_table.cuh>
#include <poggers/insert_schemes/power_of_n_shortcut.cuh>

#include <poggers/sizing/variadic_sizing.cuh>

#include <poggers/representations/soa.cuh>
#include <poggers/insert_schemes/power_of_n_shortcut_buckets.cuh>

#include <poggers/tables/bucketed_table.cuh>

#include <poggers/metadata.cuh>
#include <poggers/hash_schemes/murmurhash.cuh>
#include <poggers/probing_schemes/double_hashing.cuh>
#include <poggers/probing_schemes/power_of_two.cuh>

// new container for 2-byte key val pairs
#include <poggers/representations/grouped_key_val_pair.cuh>

#include <poggers/representations/key_val_pair.cuh>
#include <poggers/representations/dynamic_container.cuh>

#include <poggers/sizing/default_sizing.cuh>

#include <poggers/insert_schemes/power_of_n_shortcut.cuh>

#include <poggers/insert_schemes/power_of_n_shortcut_buckets.cuh>

#include <poggers/representations/packed_bucket.cuh>

#include <poggers/insert_schemes/linear_insert_buckets.cuh>

#include <poggers/tables/bucketed_table.cuh>

#include <poggers/representations/grouped_storage_sub_bits.cuh>



#include <stdio.h>
#include <iostream>
#include <chrono>
#include <openssl/rand.h>

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



// using insert_type = poggers::insert_schemes::single_slot_insert<uint64_t, uint64_t, 8, 8, poggers::representations::key_val_pair, 5, poggers::hashers::murmurHasher, poggers::probing_schemes::doubleHasher>;

// using table_type = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 4, 4, poggers::insert_schemes::bucket_insert, 200, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
//      // poggers::representations::key_val_pair, 8>

//      //using forst_tier_table_type = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, poggers::insert_schemes::single_slot_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
    
// using second_tier_table_type = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::single_slot_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, table_type>;

// using inner_table = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

// using small_double_type = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, inner_table>;

// using p2_table = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair, 8, 16, poggers::insert_schemes::power_of_n_insert_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;

   
// using tier_one_iceberg = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 1, poggers::probing_schemes::linearProber, poggers::hashers::murmurHasher>;

// using tier_two_icerberg = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::power_of_n_insert_scheme, 2, poggers::probing_schemes::powerOfTwoHasher, poggers::hashers::murmurHasher>;

// using tier_three_iceberg = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::bucket_insert, 10, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


// using tier_two_icerberg_joined = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 8, poggers::insert_schemes::power_of_n_insert_scheme, 2, poggers::probing_schemes::powerOfTwoHasher, poggers::hashers::murmurHasher, true, tier_three_iceberg>;

// using iceberg_table = poggers::tables::static_table<uint64_t, uint64_t, poggers::representations::key_val_pair, 8, 64, poggers::insert_schemes::bucket_insert, 1, poggers::probing_schemes::linearProber, poggers::hashers::murmurHasher, true, tier_two_icerberg_joined>;


// using tiny_static_table_4 = poggers::tables::static_table<uint64_t, uint16_t, poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair, 4, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
// using tcf = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::shortened_key_val_wrapper<uint16_t>::key_val_pair, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, tiny_static_table_4>;

using tiny_static_table_4 = poggers::tables::static_table<uint64_t, uint16_t, poggers::representations::dynamic_container<poggers::representations::key_val_pair,uint16_t>::representation, 4, 4, poggers::insert_schemes::bucket_insert, 20, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;
using tcf = poggers::tables::static_table<uint64_t,uint16_t, poggers::representations::dynamic_container<poggers::representations::key_val_pair,uint16_t>::representation, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher, true, tiny_static_table_4>;


using del_backing_table = poggers::tables::bucketed_table<
    uint64_t, uint8_t,
    poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<
        poggers::representations::bit_grouped_container<10, 6>::representation, uint16_t>::representation>::representation,
    1, 8, poggers::insert_schemes::linear_insert_bucket_scheme, 20, poggers::probing_schemes::doubleHasher,
    poggers::hashers::murmurHasher>;



using del_TCF = poggers::tables::bucketed_table<
    uint64_t, uint8_t,
    poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<
        poggers::representations::bit_grouped_container<10, 6>::representation, uint16_t>::representation>::representation,
    1, 8, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::doubleHasher,
    poggers::hashers::murmurHasher, true, del_backing_table>;



using del_TCF_noback = poggers::tables::bucketed_table<
    uint64_t, uint8_t,
    poggers::representations::dynamic_bucket_container<poggers::representations::dynamic_container<
        poggers::representations::bit_grouped_container<10, 6>::representation, uint16_t>::representation>::representation,
    1, 8, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::doubleHasher,
    poggers::hashers::murmurHasher>;




// shortened_key_val_wrapper


//using double_buckets = poggers::tables::bucketed_table<uint64_t, uint64_t, poggers::representations::struct_of_arrays, 4, 16, poggers::insert_schemes::power_of_n_insert_shortcut_bucket_scheme, 2, poggers::probing_schemes::doubleHasher, poggers::hashers::murmurHasher>;


#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename T>
__host__ T * generate_data(uint64_t nitems){


   //malloc space

   T * vals = (T *) malloc(nitems * sizeof(T));


   //          100,000,000
   uint64_t cap = 100000000ULL;

   for (uint64_t to_fill = 0; to_fill < nitems; to_fill+=0){

      uint64_t togen = (nitems - to_fill > cap) ? cap : nitems - to_fill;


      RAND_bytes((unsigned char *) (vals + to_fill), togen * sizeof(T));



      to_fill += togen;

      //printf("Generated %llu/%llu\n", to_fill, nitems);

   }

   return vals;
}


template <typename Filter, typename Key, typename Val>
__global__ void print_tid_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals){


      auto tile = filter->get_my_tile();

      uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

      if (tid >= nvals) return;


      if (tile.thread_rank() == 0) printf("%llu: %d, %d\n", tid, keys[tid], vals[tid]);

}

template <typename Filter, typename Key, typename Val>
__global__ void speed_insert_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;




   if (!filter->insert(tile, keys[tid], vals[tid]) && tile.thread_rank() == 0){
      atomicAdd((unsigned long long int *) misses, 1ULL);
   } else{

      Val test_val = 0;
      assert(filter->query(tile, keys[tid], test_val));
   }

   //assert(filter->insert(tile, keys[tid], vals[tid]));


}

template <typename Filter, typename Key, typename Val>
__global__ void speed_insert_kernel_one_thread(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * misses){

   auto tile = filter->get_my_tile();

   uint64_t tid2 = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid2 > 0) return;


   for (uint64_t tid=0; tid < nvals; tid++){

      if (tid % 10 == 0 && tile.thread_rank() == 0) printf("%llu\n", tid);

      if (!filter->insert(tile, keys[tid], vals[tid]) && tile.thread_rank() == 0){

      atomicAdd((unsigned long long int *) misses, 1ULL);
    } else {

      filter->remove(tile, keys[tid]);
    }
    //else{

   //    Val test_val = 0;
   //    assert(filter->query(tile, keys[tid], test_val));
   // }


   }


   //assert(filter->insert(tile, keys[tid], vals[tid]));


}


template <typename Filter, typename Key, typename Val>
__global__ void speed_delete_kernel(Filter * filter, Key * keys, uint64_t nvals, uint64_t * del_misses, uint64_t * del_failures){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;

   if (!filter->remove(tile,keys[tid]) ){

      Val val;
      val+=0;

      filter->query(tile, keys[tid], val);
      filter->remove(tile, keys[tid]);

      filter->query(tile, keys[tid], val);

      if ( tile.thread_rank() == 0) atomicAdd((unsigned long long int *) del_misses, 1ULL);

   } else {

      Val val;
      //thank you compiler very cool
      val +=0 ;
      if (filter->query(tile,keys[tid], val) && tile.thread_rank() == 0 ){

         atomicAdd((unsigned long long int *) del_failures, 1ULL);

      }

   }
   //assert(filter->query(tile, keys[tid], val));


}


template <typename Filter, typename Key, typename Val>
__global__ void speed_delete_single_thread(Filter * filter, Key * keys, uint64_t nvals, uint64_t * del_misses, uint64_t * del_failures){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid != 0) return;


   for (uint64_t i=0; i < nvals; i++){

      if (!filter->remove(tile,keys[i]) ){

      Val val;
      val+=0;

      filter->query(tile, keys[i], val);
      filter->remove(tile, keys[i]);

      filter->query(tile, keys[i], val);

      if ( tile.thread_rank() == 0) atomicAdd((unsigned long long int *) del_misses, 1ULL);

   } else {

      Val val;
      //thank you compiler very cool
      val +=0 ;
      if (filter->query(tile,keys[i], val) && tile.thread_rank() == 0 ){

         atomicAdd((unsigned long long int *) del_failures, 1ULL);

      }

   }


   }
   
   //assert(filter->query(tile, keys[tid], val));


}



template <typename Filter, typename Key, typename Val>
__global__ void speed_query_kernel(Filter * filter, Key * keys, Val * vals, uint64_t nvals, uint64_t * query_misses, uint64_t * query_failures){

   auto tile = filter->get_my_tile();

   uint64_t tid = tile.meta_group_size()*blockIdx.x + tile.meta_group_rank();

   if (tid >= nvals) return;

   Val val = 0;
   val += 0;

   if (!filter->query(tile,keys[tid], val) && tile.thread_rank() == 0){
      atomicAdd((unsigned long long int *) query_misses, 1ULL);
   } else {

      if (val != vals[tid] && tile.thread_rank() == 0){
         atomicAdd((unsigned long long int *) query_failures, 1ULL);
      }

   }
   //assert(filter->query(tile, keys[tid], val));


}

template <typename Filter, typename Key, typename Val, typename Sizing_Type>
__host__ void test_speed(Sizing_Type * Initializer){

   uint64_t nitems = Initializer->total()*.9;

   Key * host_keys = generate_data<Key>(nitems);
   Val * host_vals = generate_data<Val>(nitems);

   Key * dev_keys;

   Val * dev_vals;

   cudaMalloc((void **)& dev_keys, nitems*sizeof(Key));
   cudaMalloc((void **)& dev_vals, nitems*sizeof(Val));

   cudaMemcpy(dev_keys, host_keys, nitems*sizeof(Key), cudaMemcpyHostToDevice);
   cudaMemcpy(dev_vals, host_vals, nitems*sizeof(Val), cudaMemcpyHostToDevice);


   uint64_t * misses;

   cudaMallocManaged((void **)& misses, sizeof(uint64_t)*3);
   cudaDeviceSynchronize();

   misses[0] = 0;
   misses[1] = 0;
   misses[2] = 0;

   //static seed for testing
   Filter * test_filter = Filter::generate_on_device(Initializer, 42);

   cudaDeviceSynchronize();

   //print_tid_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems);

   cudaDeviceSynchronize();

   auto insert_start = std::chrono::high_resolution_clock::now();

   //add function for configure parameters - should be called by ht and return dim3
   speed_insert_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems, misses);
   cudaDeviceSynchronize();
   auto insert_end = std::chrono::high_resolution_clock::now();


   cudaMemcpy(dev_keys, host_keys, nitems*sizeof(Key), cudaMemcpyHostToDevice);
   cudaMemcpy(dev_vals, host_vals, nitems*sizeof(Val), cudaMemcpyHostToDevice);


   cudaDeviceSynchronize();

   auto query_start = std::chrono::high_resolution_clock::now();

   speed_query_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems, &misses[1], &misses[2]);
   cudaDeviceSynchronize();
   auto query_end = std::chrono::high_resolution_clock::now();




   std::chrono::duration<double> insert_diff = insert_end-insert_start;
   std::chrono::duration<double> query_diff = query_end-query_start;


   cudaDeviceSynchronize();
   std::cout << "Inserted " << nitems << " in " << insert_diff.count() << " seconds\n";
   std::cout << "Queried " << nitems << " in " << query_diff.count() << " seconds\n";

   printf("Inserts/Queries: %f / %f\n", 1.0*nitems/insert_diff.count(), 1.0*nitems/query_diff.count());
   printf("%llu / %llu / %llu\n", misses[0], misses[1], misses[2]);

   cudaDeviceSynchronize();

   cudaFree(misses);

   cudaDeviceSynchronize();

   cudaFree(dev_keys);
   cudaFree(dev_vals);

   Filter::free_on_device(test_filter);

   free(host_keys);
   free(host_vals);

}


template <typename Filter, typename Key, typename Val>
__global__ void delete_insert_kernel(Filter * filter){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0)return;

   uint64_t key = 1;

   uint8_t  val = 0;

   auto tile = filter->get_my_tile();

   filter->insert(tile, key, val);

}

template <typename Filter, typename Key, typename Val>
__global__ void delete_delete_kernel(Filter * filter){

   uint64_t tid = threadIdx.x+blockIdx.x*blockDim.x;

   if (tid != 0)return;

   uint64_t key = 1;

   uint8_t  val = 0;

   auto tile = filter->get_my_tile();

   filter->remove(tile, key);

}

template <typename Filter, typename Key, typename Val, typename Sizing_Type>
__host__ void test_del_batched(Sizing_Type * Initializer){


   std::cout << "Starting test\n";

   uint64_t nitems = Initializer->total();




   cudaDeviceSynchronize();


   //static seed for testing
   Filter * test_filter = Filter::generate_on_device(Initializer, 42);

   cudaDeviceSynchronize();


   printf("Fill before %llu\n", test_filter->get_fill());


   cudaDeviceSynchronize();

   delete_insert_kernel<Filter, Key, Val><<<1,1>>>(test_filter);

   //print_tid_kernel<Filter, Key, Val><<<test_filter->get_num_blocks(nitems),test_filter->get_block_size(nitems)>>>(test_filter, dev_keys, dev_vals, nitems);

   cudaDeviceSynchronize();

   printf("Fill after insert %llu\n", test_filter->get_fill());


   cudaDeviceSynchronize();

   delete_delete_kernel<Filter, Key, Val><<<1,1>>>(test_filter);

   cudaDeviceSynchronize();

   printf("Fill after delete %llu\n", test_filter->get_fill());

   Filter::free_on_device(test_filter);


   //free pieces

   //time to output

}


// __host__ void test_p2(uint64_t nitems){

//    printf("size: %llu\n", nitems);
//    poggers::sizing::size_in_num_slots<1>half_split_20(nitems);
//    test_speed<p2_table, uint64_t, uint16_t>(&half_split_20);
// }


__host__ poggers::sizing::variadic_size generate_size(int nbits){

   uint64_t nslots = (1ULL << nbits);


   poggers::sizing::variadic_size internal_size(nslots, nslots/100);

   return internal_size;

}



int main(int argc, char** argv) {

   // poggers::sizing::size_in_num_slots<1> first_size_20(1ULL << 20);
   // printf("2^20\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_20);

   // poggers::sizing::size_in_num_slots<1> first_size_22(1ULL << 22);
   // printf("2^22\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_22);

   // poggers::sizing::size_in_num_slots<1> first_size_24(1ULL << 24);
   // printf("2^24\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_24);

   // poggers::sizing::size_in_num_slots<1> first_size_26(1ULL << 26);
   // printf("2^26\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_26);

   // poggers::sizing::size_in_num_slots<1> first_size_28(1ULL << 28);
   // printf("2^28\n");
   // test_speed<table_type, uint64_t, uint64_t>(&first_size_28);

   int nbits = 20;


   //build very tiny test
   poggers::sizing::variadic_size test_size_24 (24);

   // //printf("22 size: %llu\n", test_size_24.total());
   test_del_batched<del_TCF_noback, uint64_t, uint8_t>(&test_size_24);
   // test_speed_batched<tcqf, uint64_t, uint16_t>("results/test_24", generate_size(24), 20);
   // test_speed_batched<tcqf, uint64_t, uint16_t>("results/test_26", generate_size(26), 20);
   // test_speed_batched<tcqf, uint64_t, uint16_t>("results/test_28", generate_size(28), 20);
   // test_speed_batched<tcqf, uint64_t, uint16_t>("results/test_30", generate_size(30), 20);

   cudaDeviceSynchronize();


   // poggers::sizing::variadic_size test_size_24_tcf ((1ULL << nbits), (1ULL << nbits)/100);

   // //printf("22 size: %llu\n", test_size_24.total());
   // test_speed_batched<tcf, uint64_t, uint16_t>("results/test_32", &test_size_24_tcf, 20);



   poggers::sizing::size_in_num_slots<1> bucket_size (1ULL<<nbits);

   //test_speed_batched<double_buckets, uint64_t,uint64_t>("results/double_buckets", &bucket_size, 20);

   cudaDeviceSynchronize();

   // printf("alt table\n");

   // test_p2(6000);

   // test_p2(1ULL << 22);
   // test_p2(1ULL << 24);
   // test_p2(1ULL << 26);
   // test_p2(1ULL << 28);
   // test_p2(1ULL << 30);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_22);

   // poggers::sizing::size_in_num_slots<2>half_split_24(1ULL << 23, 1ULL << 23);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_24);

   // poggers::sizing::size_in_num_slots<2>half_split_26(1ULL << 25, 1ULL << 25);
   // test_speed<small_double_type, uint64_t, uint64_t>(&half_split_26);


   // printf("P2 tiny table\n");
   // poggers::sizing::size_in_num_slots<1>half_split_28(1ULL << 30);
   // test_speed<p2_table, uint64_t, uint16_t>(&half_split_28);


   //printf("Icerberg - Tier one\n");


   //poggers::sizing::size_in_num_slots<1> tier_one_iceberg_size(1ULL << 28);
   //test_speed<tier_one_iceberg, uint64_t, uint64_t>(&tier_one_iceberg_size);


   // printf("Icerberg - Tier two\n");

   // //this section is allocated 1/8th of the space as tier one
   // poggers::sizing::size_in_num_slots<1> tier_two_iceberg_size((1ULL << 28)/8);
   // test_speed<tier_two_icerberg, uint64_t, uint64_t>(&tier_two_iceberg_size);

   // printf("Icerberg - Tier three\n");

   // poggers::sizing::size_in_num_slots<1> tier_three_iceberg_size((1500));
   // test_speed<tier_three_iceberg, uint64_t, uint64_t>(&tier_three_iceberg_size);


   // printf("Icerberg - Joined\n");

   // poggers::sizing::size_in_num_slots<3> iceberg_size((1ULL << 28), (1ULL << 28)/8, 1500);
   // test_speed<iceberg_table, uint64_t, uint64_t>(&iceberg_size);

   del_TCF test;

	return 0;

}
