  #ifndef SUB_ALLOCATOR
#define SUB_ALLOCATOR


#include <cuda.h>
#include <cuda_runtime_api.h>

#include <poggers/allocators/free_list.cuh>
#include <poggers/allocators/aligned_stack.cuh>

#include "stdio.h"
#include "assert.h"

//#include "math.h"


#ifndef DEBUG_ASSERTS
#define DEBUG_ASSERTS 0
#endif

#ifndef DEBUG_PRINTS
#define DEBUG_PRINTS 0
#endif


//The sub allocator is a powerful tool for requesting memory in parallel
//several sub allocators exist in parallel within the main memory manager, 
// and assist threads with finding smaller allocations quickly
// threads are assigned a sub_allocator 


//a pointer list managing a set section o fdevice memory

__device__ int promote_to_log2(uint64_t size){

	float log_of_size = __log2f((float) size);

	return (int) log_of_size;

}

// __host__ const int host_promote_to_log2(uint64_t size){

// 	const float log_of_size = std::log2()

// }

namespace poggers {


namespace allocators { 


template <std::size_t bytes_per_substack, std::size_t maximum_p2>
struct sub_allocator {


	using stack_type = aligned_manager<bytes_per_substack, false>;
	using my_type = sub_allocator<bytes_per_substack, maximum_p2>;

	static_assert(maximum_p2 >= 3);


	stack_type * managers[maximum_p2-2];


	__device__ static my_type * init(header * heap){

		my_type * new_allocator = (my_type *) heap->malloc_aligned(sizeof(my_type), 16, 0);

		#if DEBUG_PRINTS
		printf("Booting manager with %llu stacks, expecting %llu bytes used\n", maximum_p2-2, sizeof(my_type)+bytes_per_substack*(maximum_p2-2));
		#endif

		for (int i = 0; i < maximum_p2-2; i++){

			#if DEBUG_PRINTS
			printf("Booting size %llu\n", 1ULL << (2+i));
			#endif
			new_allocator->managers[i] = stack_type::init_from_free_list(heap, 1ULL << (2+i));

			//set the pointer to themselves for a closed loop
			new_allocator->managers[i]->set_prev(new_allocator->managers[i]);

		}

		return new_allocator;

	}

	//relinquish this allocator entirely back to the free list
	//While this doesn't require that the allocator belongs to the heap it is being freed to
	//you should return it to the same free list
	__device__ static void free_allocator(header * heap, my_type * allocator_to_free){

		for (int i=0; i < maximum_p2-2; i++){


			stack_type * current_manager = allocator_to_free->managers[i];

			while (current_manager != nullptr){

				stack_type * next_manager = current_manager->get_next_atomic();

				stack_type::free_stack(heap, current_manager);
				current_manager = next_manager;

			}

			//stack_type::free_stack(heap, allocator_to_free->managers[i]);
			//heap->free(allocator_to_free->managers[i]);

		}

		heap->free(allocator_to_free);

	}

	__device__ bool swap_manager_atomic(int p2, stack_type * existing_manager, stack_type * new_manager){


		uint64_t result = atomicCAS((unsigned long long int *) managers + p2, (unsigned long long int) existing_manager, (unsigned long long int) new_manager);

		return (result == ((uint64_t) existing_manager));

	}


	__device__ void * malloc(uint64_t bytes_requested, header * heap){

		//always round up, then subtract two
		int p2_needed = promote_to_log2(bytes_requested-1)+1;

		#if DEBUG_ASSERTS
		assert(p2_needed < maximum_p2);
		#endif

		p2_needed = p2_needed-2;


		while (true){


		//grab the local manager
		//and attempt to scan
		//scan until either nullptr or successful malloc
		stack_type * local_manager = managers[p2_needed];


		while (local_manager != nullptr){

			void * malloced = local_manager->malloc();

			if (malloced == nullptr){
				//cycle
				local_manager = local_manager->get_next_atomic();


			} else {

				//malloced is correct
				//but everyone ahead of us is full - need to be moved


				//If there are no full nodes ahead of us, who cares?
				if (local_manager == managers[p2_needed]){
					return malloced;
				}

				//otherwise continue the swap operation
				//one thread grabs the swap head
				//all followers then return as normal.

				//we need

				//1) the end of the current list (managers[p2]->get_prev_atomic());
				//2) the current head 
				//3) the first dead node
				//4) the last dead node
				//5) the first live node (local manager)


				//going from A->B->...->C, want A->C->B->...

				//process
				//1) lock first valid node for transition
				//many threads might attempt this, so onky the main node may succeed
				//all others should return their malloced data and continue


				//This lock signifies competition between threads that have traversed + malloced

				if (!local_manager->request_lock()){
					//the local manager is already being worked on! I can safely leave
					return malloced;
				}


				//2) gather necessary variables for swap

				//Do we need to maintain that only one nullptr exists in the list? may be important to create a loop before unwinding
				//this will force threads already traversing to continue until the swap is done
				//I think this is safer so that's what we will do
				//otherwise multiple threads could attempt to add new stacks to the list, which is wasteful and potentially memory-leaky

				//do I need to load these atomically?
				stack_type * A = managers[p2_needed];


				//need to make sure no one else messes with the node structure while we allocate
				//mallocs happen uninterrupted, but no one else can claim the node being moved
				//and new new allocations can occur
				//this should be fine as by construction this node is still serving mallocs.
				A->stall_lock();


				stack_type * B = local_manager->get_prev_atomic();

				//C comes locked!
				stack_type * C = local_manager;


				//The previous of the head of the list is 
				stack_type * end_of_C = A->get_prev_atomic();


				//this variable has a short lifespan
				// It just saves us 1 atomicCAS to find the true end
				// this code is just in case the ptr main had to the end was updated by someone else
				// likely impossible but you never know ¯\_(ツ)_/¯
				{

					stack_type * C_trace = end_of_C->get_next_atomic();

					while (C_trace != nullptr){
						end_of_C = C_trace;
						C_trace = end_of_C->get_next_atomic();
					}


				}




				//end of C is now the last item in the list
				//do we need to lock?
				//I think not, anyone else doing this needs to lock main
				//must assert this precondition on appending new stacks as well
				//though that should be simple as they will attach themselves in main
				

				//swap out end of total list with ptr to the first dead node
				//this will create a loop that only this thread will be able to unwind

				//A->B->...->C->B

				//A is currently the start of B as all stacks before B are full
				//TODO - does this need to be atomic?
				end_of_C->set_next(A);
				A->set_prev(end_of_C);
				__threadfence();









				//3) swap curent main node with next free
				// this will push all full nodes out of the list so that only this thread has a ptr to them.
				//now anyone currently traversing will filter out of the loop

				//A->C->B->C->B

				//consider atomic swap here
				//atomicCAS((unsigned_long long int *) managers + p2_needed, managers[p2_needed], C);

				swap_manager_atomic(p2_needed, managers[p2_needed], C);
				//managers[p2_needed] = C;
				__threadfence();


				//4) swap current end of B with nullptr to reorganize end of list

				//A->C->B

				B->set_next(nullptr);

				C->set_prev(B);


				//we're done!
				C->unlock();
				A->unlock();

				//return malloced value!

				return malloced;

				}


			//not necessary handled by main if case
			//local_manager = local_manager->get_next_atomic();

			}


			//space to malloc a new one!

			//to prevent 30000 identical allocations, you must lock the head
			//This means that nodes can stall in this loop when other nodes are being shifted
			//only occurs iff 
			// 1) the node being moved is at the end of the list AND
			// 2) the node is filled as the movement happens.
			if (managers[p2_needed]->request_lock()){

			//I own the lock, append to this manager
				stack_type * new_stack = stack_type::init_from_free_list(heap, 1ULL << (2+p2_needed));

				if (new_stack != nullptr){

					//lock just in case someone else sees us mid update
					new_stack->stall_lock();

					//register the stack here!
					// hash_table->insert();

					new_stack->set_next(managers[p2_needed]);
					new_stack->set_prev(managers[p2_needed]->get_prev_atomic());

					managers[p2_needed]->set_prev(new_stack);

					__threadfence();

					void * new_malloc = new_stack->malloc();

					//managers[p2_needed]->unlock();

					swap_manager_atomic(p2_needed, managers[p2_needed], new_stack);
					//managers[p2_needed] = new_stack;
					__threadfence();

					new_stack->get_next_atomic()->unlock();
					new_stack->unlock();
					//local_manager->set_next(new_stack);

					//new manager is included, continue!
					return new_malloc;


				} else {	

					//throw an error, stack is full
					//printf("Error: Can't allocate new stack\n");
					//__trap();
					return nullptr;

				}

			} else {

				//think I need to reset local manager
				local_manager = managers[p2_needed];


			}


			//stack_type * new_stack = stack_type::init_from_free_list(heap, 1ULL << (2+i));




		}

		


	}

	__device__ void stack_free(void * address){
		stack_type::static_free(address);
	}

};

template <std::size_t x>
struct log2_template { enum { value = 1 + log2_template<x/2>::value }; };
  
template <> struct log2_template<1> { enum { value = 1 }; };


template <std::size_t bytes_per_substack, std::size_t max_size>
struct sub_allocator_wrapper {

	using sub_allocator_type = sub_allocator<bytes_per_substack, log2_template<max_size-1>::value+1>;

};


}

}


#endif //GPU_BLOCK_