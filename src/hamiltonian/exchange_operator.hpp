/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__EXCHANGE_OPERATOR
#define INQ__HAMILTONIAN__EXCHANGE_OPERATOR

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

#include <basis/real_space.hpp>
#include <operations/overlap.hpp>
#include <operations/overlap_diagonal.hpp>
#include <operations/rotate.hpp>
#include <solvers/cholesky.hpp>
#include <solvers/poisson.hpp>
#include <states/orbital_set.hpp>

#include <optional>

namespace inq {
namespace hamiltonian {
  class exchange_operator {
		
  public:

		exchange_operator(const basis::real_space & basis, const int num_hf_orbitals, const double exchange_coefficient, bool const use_ace, boost::mpi3::cartesian_communicator<2> comm):
			exchange_coefficient_(exchange_coefficient),
			use_ace_(use_ace){

			if(exchange_coefficient_ != 0.0) hf_orbitals.emplace(basis, num_hf_orbitals, comm);	
			if(exchange_coefficient_ != 0.0) xi_.emplace(basis, num_hf_orbitals, std::move(comm));		
		}

		//////////////////////////////////////////////////////////////////////////////////

		template <class ElectronsType>
		double update(ElectronsType const & el){
			if(not enabled()) return 0.0;

			CALI_CXX_MARK_SCOPE("hf_update");

			assert(el.lot_size() == 1);			

			auto & phi = el.lot()[0];

			hf_occupations.reextent(phi.local_set_size());
			hf_occupations = el.occupations()[0];
			hf_orbitals->fields() = phi.fields();
			
			*xi_ = direct(phi, -1.0);

			auto exx_matrix = operations::overlap(*xi_, phi);

			double energy = -0.5*real(operations::sum_product(hf_occupations, exx_matrix.diagonal()));
			el.lot_states_comm_.all_reduce_in_place_n(&energy, 1, std::plus<>{});
			
			solvers::cholesky(exx_matrix.array());
			operations::rotate_trs(exx_matrix, *xi_);
			
			return energy;
		}

		//////////////////////////////////////////////////////////////////////////////////
		
		auto direct(const states::orbital_set<basis::real_space, complex> & phi, double scale = 1.0) const {
			states::orbital_set<basis::real_space, complex> exxphi(phi.skeleton());
			exxphi.fields() = 0.0;
			direct(phi, exxphi, scale);
			return exxphi;
		}

		//////////////////////////////////////////////////////////////////////////////////
		
		template <class BasisType, class HFType, class HFOccType, class PhiType, class ExxphiType>
		void block_exchange(double factor, BasisType const & basis, HFType const & hf, HFOccType const & hfocc, PhiType const & phi, ExxphiType & exxphi) const {

			auto nst = (~phi).size();
			auto nhf = (~hf).size();
			basis::field_set<basis::real_space, complex> rhoij(basis, nst);
			
			for(int jj = 0; jj < nhf; jj++){
				
				{ CALI_CXX_MARK_SCOPE("hartree_fock_exchange_gen_dens");
					gpu::run(nst, basis.local_size(),
									 [rho = begin(rhoij.matrix()), hfo = begin(hf), ph = begin(phi), jj] GPU_LAMBDA (auto ist, auto ipoint){ 
										 rho[ipoint][ist] = conj(hfo[ipoint][jj])*ph[ipoint][ist];
									 });
				}
				
				poisson_solver_.in_place(rhoij);
				
				{ CALI_CXX_MARK_SCOPE("hartree_fock_exchange_mul_pot");
					gpu::run(nst, basis.local_size(),
									 [pot = begin(rhoij.matrix()), hfo = begin(hf), exph = begin(exxphi), occ = begin(hfocc), jj, factor]
									 GPU_LAMBDA (auto ist, auto ipoint){
										 exph[ipoint][ist] += factor*occ[jj]*hfo[ipoint][jj]*pot[ipoint][ist];
									 });
				}
			}
		}

		//////////////////////////////////////////////////////////////////////////////////
		
		void direct(const states::orbital_set<basis::real_space, complex> & phi, states::orbital_set<basis::real_space, complex> & exxphi, double scale = 1.0) const {
			if(not enabled()) return;
			
			CALI_CXX_MARK_SCOPE("hartree_fock_exchange");
			
			double factor = -0.5*scale*exchange_coefficient_;

			if(not hf_orbitals->set_part().parallel()){
				block_exchange(factor, phi.basis(), hf_orbitals->matrix(), hf_occupations, phi.matrix(), exxphi.matrix());
			} else {

				auto mpi_type = boost::mpi3::detail::basic_datatype<complex>();
 
				math::array<complex, 2> rhfo({hf_orbitals->basis().local_size(), hf_orbitals->set_part().block_size()}, 0.0);
				rhfo(boost::multi::ALL, {0, hf_orbitals->set_part().local_size()}) = hf_orbitals->matrix();

				math::array<complex, 1> roccs(hf_orbitals->set_part().block_size(), 0.0);
				roccs({0, hf_orbitals->set_part().local_size()}) = hf_occupations;
				
				auto next_proc = phi.set_comm().rank() + 1;
				if(next_proc == phi.set_comm().size()) next_proc = 0;
				auto prev_proc = phi.set_comm().rank() - 1;
				if(prev_proc == -1) prev_proc = phi.set_comm().size() - 1;

				auto ipart = hf_orbitals->set_comm().rank();
				for(int istep = 0; istep < hf_orbitals->set_part().comm_size(); istep++){
					block_exchange(factor, phi.basis(), rhfo(boost::multi::ALL, {0, hf_orbitals->set_part().local_size(ipart)}), roccs, phi.matrix(), exxphi.matrix());

					if(istep == hf_orbitals->set_part().comm_size() - 1) break; //the last step we don't need to do communicate
					MPI_Sendrecv_replace(raw_pointer_cast(rhfo.data_elements()), rhfo.num_elements(), mpi_type, prev_proc, istep, next_proc, istep, hf_orbitals->set_comm().get(), MPI_STATUS_IGNORE);
					MPI_Sendrecv_replace(raw_pointer_cast(roccs.data_elements()), roccs.num_elements(), mpi_type, prev_proc, istep, next_proc, istep, hf_orbitals->set_comm().get(), MPI_STATUS_IGNORE);
			
					ipart++;
					if(ipart == hf_orbitals->set_comm().size()) ipart = 0;
				}

			}
		}

		//////////////////////////////////////////////////////////////////////////////////
		
		auto ace(const states::orbital_set<basis::real_space, complex> & phi) const {
			states::orbital_set<basis::real_space, complex> exxphi(phi.skeleton());
			exxphi.fields() = 0.0;
			ace(phi, exxphi);
			return exxphi;
		}

		//////////////////////////////////////////////////////////////////////////////////
		
		auto operator()(const states::orbital_set<basis::real_space, complex> & phi) const {
			states::orbital_set<basis::real_space, complex> exxphi(phi.skeleton());
			exxphi.fields() = 0.0;
			operator()(phi, exxphi);
			return exxphi;
		}

		//////////////////////////////////////////////////////////////////////////////////

		void operator()(const states::orbital_set<basis::real_space, complex> & phi, states::orbital_set<basis::real_space, complex> & exxphi) const {
			if(not enabled()) return;

			if(use_ace_) ace(phi, exxphi);
			else direct(phi, exxphi);
		}

		//////////////////////////////////////////////////////////////////////////////////
		
		void ace(const states::orbital_set<basis::real_space, complex> & phi, states::orbital_set<basis::real_space, complex> & exxphi) const {			
			if(not enabled()) return;
			namespace blas = boost::multi::blas;

			auto olap = operations::overlap(*xi_, phi);
			operations::rotate(olap, *xi_, exxphi, -1.0, 1.0);
		}

		//////////////////////////////////////////////////////////////////////////////////
		
		bool enabled() const {
			return hf_orbitals.has_value() or xi_.has_value();
		}

		//////////////////////////////////////////////////////////////////////////////////

	private:
		math::array<double, 1> hf_occupations;
		std::optional<states::orbital_set<basis::real_space, complex>> hf_orbitals;
		std::optional<states::orbital_set<basis::real_space, complex>> xi_;		
		solvers::poisson poisson_solver_;
		double exchange_coefficient_;
		bool use_ace_;
		
  };

}
}

#ifdef INQ_HAMILTONIAN_EXCHANGE_OPERATOR_UNIT_TEST
#undef INQ_HAMILTONIAN_EXCHANGE_OPERATOR_UNIT_TEST

#include <ions/unitcell.hpp>
#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

TEST_CASE("Class hamiltonian::exchange", "[hamiltonian::exchange]"){

	using namespace inq;
	using namespace Catch::literals;
  using math::vector3;
  /*
  auto ecut = 20.0_Ha;
  double ll = 10.0;
	*/
	/*
	ions::geometry geo;
  ions::UnitCell cell(vector3<double>(ll, 0.0, 0.0), vector3<double>(0.0, ll, 0.0), vector3<double>(0.0, 0.0, ll));
  basis::real_space rs(cell, input::basis::cutoff_energy(ecut));

	hamiltonian::atomic_potential pot(geo.num_atoms(), geo.atoms());
	
	states::ks_states st(states::ks_states::spin_config::UNPOLARIZED, 11.0);

  states::orbital_set<basis::real_space, complex> phi(rs, st.num_states());
	states::orbital_set<basis::real_space, complex> hphi(rs, st.num_states());
	
	hamiltonian::exchange<basis::real_space> ham(rs, cell, pot, geo, st.num_states(), 0.0);
	*/
}

#endif

#endif
