/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__GPU__REDUCE
#define INQ__GPU__REDUCE

/*
 Copyright (C) 2019 Xavier Andrade

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.
  
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
  
 You should have received a copy of the GNU Lesser General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <inq_config.h>

#ifdef ENABLE_CUDA
#include <cuda.h>
#endif

#include <cstddef> // std::size_t
#include <cassert>

#include <gpu/run.hpp>
#include <math/array.hpp>

namespace inq {
namespace gpu {

#ifdef ENABLE_CUDA
template <class kernel_type, class array_type>
__global__ void reduce_kernel_1d(size_t size, kernel_type kernel, array_type odata) {

	extern __shared__ char shared_mem[];
	auto reduction_buffer = (typename array_type::element *) shared_mem;
	
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int ii = blockIdx.x*blockDim.x + threadIdx.x;

	if(ii < size){
		reduction_buffer[tid] = kernel(ii);
	} else {
		reduction_buffer[tid] = (typename array_type::element) 0.0;
	}

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x/2; s > 0; s >>= 1){
		if (tid < s) {
			reduction_buffer[tid] += reduction_buffer[tid + s];
		}
		__syncthreads();
	}
	
	// write result for this block to global mem
	if (tid == 0) odata[blockIdx.x] = reduction_buffer[0];

}
#endif

template <typename array_type>
struct array_access {
  array_type array;

  GPU_FUNCTION auto operator()(size_t ii) const {
    return array[ii];
  }
  
};

template <class kernel_type>
auto reduce(size_t size, kernel_type kernel) -> decltype(kernel(0)) {

  using type = decltype(kernel(0));
  
#ifndef ENABLE_CUDA

  type accumulator = 0.0;
  for(size_t ii = 0; ii < size; ii++){
    accumulator += kernel(ii);
  }
  return accumulator;

#else

	const int blocksize = 1024;

	unsigned nblock = (size + blocksize - 1)/blocksize;
	math::array<type, 1> result(nblock, 666.0);

  reduce_kernel_1d<<<nblock, blocksize, blocksize*sizeof(type)>>>(size, kernel, begin(result));	
  check_error(cudaGetLastError());
	
  if(nblock == 1) {
    cudaDeviceSynchronize();
    return result[0];
  } else {
    return reduce(nblock, array_access<decltype(begin(result))>{begin(result)});
  }
  
#endif
}

}
}

#ifdef INQ_GPU_REDUCE_UNIT_TEST
#undef INQ_GPU_REDUCE_UNIT_TEST

#include <catch2/catch.hpp>

struct ident {
  GPU_FUNCTION auto operator()(size_t ii){
    return double(ii);
  }
};

TEST_CASE("function gpu::reduce", "[gpu::reduce]") {
  
	using namespace inq;
	using namespace Catch::literals;

  const size_t maxsize = 387420490;

  
  for(size_t nn = 1; nn < maxsize; nn *= 3){
    CHECK(gpu::reduce(nn, ident{}) == (nn*(nn - 1.0)/2.0));    
  }
  
}

#endif
#endif

