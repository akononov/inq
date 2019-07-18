/* -*- indent-tabs-mode: t; tab-width: 2 -*- */

#ifndef INQ__SYSTEM
#define INQ__SYSTEM

#include <ions/geometry.hpp>
#include <ions/unitcell.hpp>
#include <basis/real_space.hpp>
#include <hamiltonian/atomic_potential.hpp>
#include <states/ks_states.hpp>
#include <hamiltonian/ks_hamiltonian.hpp>
#include <basis/coefficients_set.hpp>
#include <operations/randomize.hpp>
#include <operations/overlap.hpp>
#include <operations/scal.hpp>
#include <operations/orthogonalization.hpp>
#include <solvers/steepest_descent.hpp>
#include <math/complex.hpp>

namespace inq {

  class system {

  public:
    
    system(const ions::geometry & geo_arg, const ions::UnitCell & arg_cell, const double & ecut):
      geo_(geo_arg),
      cell_(arg_cell),
      rs_(cell_, ecut),
      atomic_pot_(geo_.num_atoms(), geo_.atoms()),
      states_(states::ks_states::spin_config::UNPOLARIZED, atomic_pot_.num_electrons()),
      ham_(rs_, cell_, atomic_pot_, geo_),
      phi_(rs_, states_.num_states()){

      geo_.info(std::cout);
      cell_.info(std::cout);
      rs_.info(std::cout);  
      states_.info(std::cout);
      ham_.info(std::cout);

      operations::randomize(phi_);
    }

    void calculate_ground_state() {
      
      for(int ii = 0; ii < 2000; ii++){
	
	operations::scal_invsqrt(operations::overlap_diagonal(phi_), phi_);
	
	auto hphi = ham_(states_, phi_);
	
	auto overlap = operations::overlap_diagonal(hphi, phi_);
	
	std::cout << ii << '\t' << std::scientific << real(overlap[0]) << std::endl;
	
	solvers::steepest_descent(states_, ham_, phi_);
	
      }
    }

    auto calculate_energy() {
      
      operations::scal_invsqrt(operations::overlap_diagonal(phi_), phi_);
      
      auto hphi = ham_(states_, phi_);
      
      auto overlap = operations::overlap_diagonal(hphi, phi_);

      //DATAOPERATIONS
      double energy = 0.0;
      for(int ii = 0; ii < states_.num_states(); ii++) energy += real(overlap[ii]);
      return energy;
    }


    
  private:
    
    ions::geometry geo_;
    ions::UnitCell cell_;
    basis::real_space rs_;
    hamiltonian::atomic_potential atomic_pot_;
    states::ks_states states_;
    hamiltonian::ks_hamiltonian<basis::real_space> ham_;      
    basis::coefficients_set<basis::real_space, complex> phi_;
  

  };  
  
}

#endif

