/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__PROJECTOR
#define INQ__HAMILTONIAN__PROJECTOR

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
#include <ions/periodic_replicas.hpp>
#include <basis/double_grid.hpp>
#include <basis/real_space.hpp>
#include <basis/spherical_grid.hpp>
#include <hamiltonian/atomic_potential.hpp>
#include <utils/profiling.hpp>
#include <utils/raw_pointer_cast.hpp>

namespace inq {
namespace hamiltonian {

class projector {

#ifdef ENABLE_CUDA
public:
#endif
	
	void build(const basis::real_space & basis, atomic_potential::pseudopotential_type const & ps) {

		CALI_CXX_MARK_SCOPE("projector::build");
		
		int iproj_lm = 0;
		for(int iproj_l = 0; iproj_l < ps.num_projectors_l(); iproj_l++){
				
			int l = ps.projector_l(iproj_l);

			if(not basis.double_grid().enabled()) {
				
				// now construct the projector with the spherical harmonics
				gpu::run(sphere_.size(), 2*l + 1,
								 [mat = begin(matrix_),
									spline = ps.projector(iproj_l).cbegin(),
									sph = sphere_.ref(), l, iproj_lm,
									kb_ = begin(kb_coeff_),
									coe = ps.kb_coeff(iproj_l),
									metric = basis.cell().metric()] GPU_LAMBDA (auto ipoint, auto m) {
									 
									 if(ipoint == 0) kb_[iproj_lm + m] = coe;
									 mat[iproj_lm + m][ipoint] = spline.value(sph.distance(ipoint))*pseudo::math::spherical_harmonic(l, m - l, metric.to_cartesian(sph.point_pos(ipoint)));
								 });
				
			} else {

				CALI_CXX_MARK_SCOPE("projector::double_grid");
				
				gpu::run(sphere_.size(), 2*l + 1,
								 [mat = begin(matrix_), spline = ps.projector(iproj_l).cbegin(), sph = sphere_.ref(), l, iproj_lm, kb_ = begin(kb_coeff_), coe = ps.kb_coeff(iproj_l),
									dg = basis.double_grid().ref(), spac = basis.rspacing(), metric = basis.cell().metric()] GPU_LAMBDA (auto ipoint, auto m) {
									 
									 if(ipoint == 0) kb_[iproj_lm + m] = coe;
									 mat[iproj_lm + m][ipoint] = dg.value([spline, l, m] GPU_LAMBDA(auto pos) { return spline.value(length(pos))*pseudo::math::spherical_harmonic(l, m - l, pos);}, spac, metric.to_cartesian(sph.point_pos(ipoint)));
								 });
				
			}

			iproj_lm += 2*l + 1;
			
		}

		assert(iproj_lm == ps.num_projectors_lm());

	}
	
public:
	projector(const basis::real_space & basis, atomic_potential::pseudopotential_type const & ps, math::vector3<double> atom_position, int iatom):
		sphere_(basis, atom_position, ps.projector_radius()),
		nproj_(ps.num_projectors_lm()),
		matrix_({nproj_, sphere_.size()}),
		kb_coeff_(nproj_),
		comm_(sphere_.create_comm(basis.comm())),
		iatom_(iatom){

		build(basis, ps);

	}

	projector(projector const &) = delete;		

	auto empty() const {
		return nproj_ == 0 or sphere_.size() == 0;
	}

	template <typename OcType, typename PhiType, typename GPhiType>
	struct force_term {
		OcType oc;
		PhiType phi;
		GPhiType gphi;
		constexpr auto operator()(int ist, int ip) const {
			return -2.0*oc[ist]*real(phi[ip][ist]*conj(gphi[ip][ist]));
		}
	};
	
	int num_projectors() const {
		return nproj_;
	}
		
	auto kb_coeff(int iproj){
		return kb_coeff_[iproj];
	}

	auto iatom() const {
		return iatom_;
	}
	
	auto & sphere() const {
		return sphere_;
	}

	auto & matrix() const {
		return matrix_;
	}

	friend class projector_all;
	
private:

	basis::spherical_grid sphere_;
	int nproj_;
	math::array<double, 2> matrix_;
	math::array<double, 1> kb_coeff_;
	mutable parallel::communicator comm_;
	int iatom_;
    
};
  
}
}

#ifdef INQ_HAMILTONIAN_PROJECTOR_UNIT_TEST
#undef INQ_HAMILTONIAN_PROJECTOR_UNIT_TEST

#include <config/path.hpp>
#include <ions/geometry.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE("class hamiltonian::projector", "[hamiltonian::projector]") {
  
	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
  using math::vector3;
	
	pseudo::math::erf_range_separation const sep(0.625);

	auto comm = boost::mpi3::environment::get_world_instance();

	ions::geometry geo;
	systems::box box = systems::box::cubic(10.0_b).cutoff_energy(20.0_Ha);
  basis::real_space rs(box, comm);

	hamiltonian::atomic_potential::pseudopotential_type ps(config::path::unit_tests_data() + "N.upf", sep, rs.gcutoff());
	
	hamiltonian::projector proj(rs, ps, vector3<double>(0.0, 0.0, 0.0), 77);

	CHECK(proj.num_projectors() == 8);

	if(not proj.empty()){
		CHECK(proj.kb_coeff(0) ==  7.494508815_a);
		CHECK(proj.kb_coeff(1) ==  0.6363049519_a);
		CHECK(proj.kb_coeff(2) == -4.2939052122_a);
		CHECK(proj.kb_coeff(3) == -4.2939052122_a);
		CHECK(proj.kb_coeff(4) == -4.2939052122_a);
		CHECK(proj.kb_coeff(5) == -1.0069878791_a);
		CHECK(proj.kb_coeff(6) == -1.0069878791_a);
		CHECK(proj.kb_coeff(7) == -1.0069878791_a);
	}
	
	CHECK(proj.iatom() == 77);
	
}
#endif

#endif

