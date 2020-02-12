/* -*- indent-tabs-mode: t -*- */

#ifndef OPERATIONS__SHIFT
#define OPERATIONS__SHIFT

/*
 Copyright (C) 2019 Xavier Andrade, Alfredo Correa.

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

#include <gpu/run.hpp>
#include <basis/field_set.hpp>
#include <cassert>

namespace operations {

	template <class array_1d, class field_set_type>
  void shift(const array_1d & factor, const field_set_type & shift, field_set_type & phi, double scale = 1.0){
    
    assert(size(factor) == phi.set_dist().local_size());

		//DATAOPERATIONS GPU::RUN 2D 
		gpu::run(phi.set_dist().local_size(), phi.basis().dist().local_size(),
						 [factorp = begin(factor), shiftp = begin(shift.matrix()), phip = begin(phi.matrix()), scale]
						 GPU_LAMBDA (auto ist, auto ipoint){
							 phip[ipoint][ist] += scale*(factorp[ist]*shiftp[ipoint][ist]);
						 });
  }
  
}

#ifdef UNIT_TEST
#include <catch2/catch.hpp>

TEST_CASE("function operations::shift", "[operations::shift]") {

	using namespace Catch::literals;
	const int npoint = 185193;
	const int nvec = 7;

	auto comm = boost::mpi3::environment::get_world_instance();
	
	boost::mpi3::cartesian_communicator<2> cart_comm(comm);

	auto basis_comm = cart_comm.axis(1);
	
	basis::trivial bas(npoint, basis_comm);
	
	SECTION("double"){
		
		basis::field_set<basis::trivial, double> aa(bas, nvec, cart_comm);
		basis::field_set<basis::trivial, double> bb(bas, nvec, cart_comm);

		math::array<double, 1> factor(aa.set_dist().local_size());
		
		for(int jj = 0; jj < aa.set_dist().local_size(); jj++){
			auto jjg = aa.set_dist().local_to_global(jj);
			for(int ii = 0; ii < bas.dist().local_size(); ii++){
				auto iig = bas.dist().local_to_global(ii);
				aa.matrix()[ii][jj] = 1.0 + 0.765*iig*jjg;
				bb.matrix()[ii][jj] = iig;
			}
			factor[jj] = 2.0*0.765*jjg;
		}

		operations::shift(factor, bb, aa, -0.5);
				
		for(int ii = 0; ii < bas.dist().local_size(); ii++){
			for(int jj = 0; jj < aa.set_dist().local_size(); jj++) REQUIRE(aa.matrix()[ii][jj] == Approx(1.0));
		}
	}
	
	SECTION("complex"){
		
		basis::field_set<basis::trivial, complex> aa(bas, nvec, cart_comm);
		basis::field_set<basis::trivial, complex> bb(bas, nvec, cart_comm);

		math::array<complex, 1> factor(aa.set_dist().local_size());
		
		for(int jj = 0; jj < aa.set_dist().local_size(); jj++){
			auto jjg = aa.set_dist().local_to_global(jj);
			for(int ii = 0; ii < bas.dist().local_size(); ii++){
				auto iig = bas.dist().local_to_global(ii);
				aa.matrix()[ii][jj] = complex(iig, 1.0 + 0.765*iig*jjg);
				bb.matrix()[ii][jj] = iig;
			}
			factor[jj] = complex(0.0, 2.0*0.765*jjg);
		}

		operations::shift(factor, bb, aa, -0.5);
				
		for(int ii = 0; ii < bas.dist().local_size(); ii++){
			auto iig = bas.dist().local_to_global(ii);
			for(int jj = 0; jj < aa.set_dist().local_size(); jj++) REQUIRE(real(aa.matrix()[ii][jj]) == Approx(iig));
			for(int jj = 0; jj < aa.set_dist().local_size(); jj++) REQUIRE(imag(aa.matrix()[ii][jj]) == Approx(1.0));
		}
	}	
	
	SECTION("mixed types"){
		
		basis::field_set<basis::trivial, complex> aa(bas, nvec, cart_comm);
		basis::field_set<basis::trivial, complex> bb(bas, nvec, cart_comm);

		math::array<double, 1> factor(aa.set_dist().local_size());
		
		for(int jj = 0; jj < aa.set_dist().local_size(); jj++){
			auto jjg = aa.set_dist().local_to_global(jj);
			for(int ii = 0; ii < bas.dist().local_size(); ii++){
				auto iig = bas.dist().local_to_global(ii);
				aa.matrix()[ii][jj] = complex(iig, 1.0 + 0.765*iig*jjg);
				bb.matrix()[ii][jj] = complex(0.0, iig);
			}
			factor[jj] = 2.0*0.765*jjg;
		}

		operations::shift(factor, bb, aa, -0.5);
				
		for(int ii = 0; ii < bas.dist().local_size(); ii++){
			for(int jj = 0; jj < aa.set_dist().local_size(); jj++) {
				auto iig = bas.dist().local_to_global(ii);
				REQUIRE(real(aa.matrix()[ii][jj]) == Approx(iig));
				REQUIRE(imag(aa.matrix()[ii][jj]) == Approx(1.0));
			}
		}
	}
	
}


#endif

#endif
