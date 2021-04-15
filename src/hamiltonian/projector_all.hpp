/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__PROJECTOR_ALL
#define INQ__HAMILTONIAN__PROJECTOR_ALL

/*
	Copyright (C) 2019-2020 Xavier Andrade, Alfredo A. Correa.

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

#include <pseudopod/spherical_harmonic.hpp>

#include <math/array.hpp>
#include <math/vector3.hpp>
#include <ions/unitcell.hpp>
#include <ions/periodic_replicas.hpp>
#include <basis/real_space.hpp>
#include <basis/spherical_grid.hpp>
#include <hamiltonian/atomic_potential.hpp>
#include <utils/profiling.hpp>
#include <utils/raw_pointer_cast.hpp>

namespace inq {
namespace hamiltonian {

class projector_all {

public:

	projector_all():
    max_sphere_size_(0),
    max_nproj_(0) {
  }
  
	template <typename ProjectorsType>
	projector_all(ProjectorsType const & projectors){

		CALI_CXX_MARK_FUNCTION;
    
		max_sphere_size_ = 0;
		max_nproj_ = 0;
		for(auto it = projectors.cbegin(); it != projectors.cend(); ++it) {
			max_sphere_size_ = std::max(max_sphere_size_, it->sphere_.size());
			max_nproj_ = std::max(max_nproj_, it->nproj_);			
		}

    coeff_ = decltype(coeff_)({projectors.size(), max_nproj_}, 0.0);

    auto iproj = 0;
    for(auto it = projectors.cbegin(); it != projectors.cend(); ++it) {
      coeff_[iproj]({0, it->nproj_}) = it->kb_coeff_;
      iproj++;
    }
    
  }
  
	template <typename ProjectorsType>
	math::array<complex, 3> project(ProjectorsType const & projectors, basis::field_set<basis::real_space, complex> const & phi) const {
    
		math::array<complex, 3> sphere_phi_all({projectors.size(), max_sphere_size_, phi.local_set_size()});
		math::array<complex, 3> projections_all({projectors.size(), max_nproj_, phi.local_set_size()});

		auto iproj = 0;
		for(auto it = projectors.cbegin(); it != projectors.cend(); ++it){
			auto sphere_phi = sphere_phi_all[iproj]({0, it->sphere_.size()});
			
			CALI_CXX_MARK_SCOPE("projector::gather");
				
			gpu::run(std::get<1>(sizes(sphere_phi)), it->sphere_.size(),
							 [sgr = begin(sphere_phi), gr = begin(phi.cubic()), sph = it->sphere_.ref()] GPU_LAMBDA (auto ist, auto ipoint){
								 sgr[ipoint][ist] = gr[sph.points(ipoint)[0]][sph.points(ipoint)[1]][sph.points(ipoint)[2]][ist];
								 });
			
			iproj++;
		}

		iproj = 0;
		for(auto it = projectors.cbegin(); it != projectors.cend(); ++it){
			auto sphere_phi = sphere_phi_all[iproj]({0, it->sphere_.size()});
			auto projections = projections_all[iproj]({0, it->nproj_});
			
			CALI_CXX_MARK_SCOPE("projector_gemm_1");
			namespace blas = boost::multi::blas;
			blas::real_doubled(projections) = blas::gemm(it->sphere_.volume_element(), it->matrix_, blas::real_doubled(sphere_phi));
				
			iproj++;
		}
		
		iproj = 0;
		for(auto it = projectors.cbegin(); it != projectors.cend(); ++it){
			auto projections = projections_all[iproj]({0, it->nproj_});
			
			CALI_CXX_MARK_SCOPE("projector_scal");
				
				//DATAOPERATIONS GPU::RUN 2D
				gpu::run(phi.local_set_size(), it->nproj_,
								 [proj = begin(projections), coe = begin(coeff_), iproj]
								 GPU_LAMBDA (auto ist, auto ilm){
									 proj[ilm][ist] = proj[ilm][ist]*coe[iproj][ilm];
								 });
				
			iproj++;
		}

		iproj = 0;
		for(auto it = projectors.cbegin(); it != projectors.cend(); ++it){
			CALI_CXX_MARK_SCOPE("projector_mpi_reduce");

			if(it->comm_.size() > 1){
				auto projections = +projections_all[iproj]({0, it->nproj_});
				it->comm_.all_reduce_in_place_n(raw_pointer_cast(projections.data_elements()), projections.num_elements(), std::plus<>{});
				projections_all[iproj]({0, it->nproj_}) = projections;
			}
			iproj++;
		}
		
		iproj = 0;
		for(auto it = projectors.cbegin(); it != projectors.cend(); ++it){
			auto sphere_phi = sphere_phi_all[iproj]({0, it->sphere_.size()});
			auto projections = projections_all[iproj]({0, it->nproj_});
			
			{
				CALI_CXX_MARK_SCOPE("projector_gemm_2");
				namespace blas = boost::multi::blas;
				blas::real_doubled(sphere_phi) = blas::gemm(1., blas::T(it->matrix_), blas::real_doubled(projections));
			}
			
			iproj++;
		}

		return sphere_phi_all;
			
	}

	////////////////////////////////////////////////////////////////////////////////////////////		

	template <typename ProjectorsType, typename SpherePhiType>
	void apply(ProjectorsType const & projectors, SpherePhiType & sphere_vnlphi, basis::field_set<basis::real_space, complex> & vnlphi) const {

		CALI_CXX_MARK_FUNCTION;
		
		auto iproj = 0;
		for(auto it = projectors.cbegin(); it != projectors.cend(); ++it){
			it->sphere_.scatter_add(sphere_vnlphi[iproj]({0, it->sphere_.size()}), vnlphi.cubic());
			iproj++;
		}
	}

private:
  
  long max_sphere_size_;
  int max_nproj_;
  math::array<double, 2> coeff_;
  
};
  
}
}

#ifdef INQ_HAMILTONIAN_PROJECTOR_ALL_UNIT_TEST
#undef INQ_HAMILTONIAN_PROJECTOR_ALL_UNIT_TEST

#include <config/path.hpp>
#include <ions/geometry.hpp>

#include <catch2/catch.hpp>

TEST_CASE("class hamiltonian::projector_all", "[hamiltonian::projector_all]") {
  
	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
  using math::vector3;

}

#endif

#endif

