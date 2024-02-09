/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__RESULT
#define INQ__INTERFACE__RESULT

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <ground_state/result.hpp>

namespace inq {
namespace interface {

struct {		

	std::string name() const {
		return "result";
	}

	std::string one_line() const {
		return "Get information about the result obtained by a ground-state calculation";
	}

	void help() const {
		
		std::cout << R""""(

The 'result' command
==================

This command queries the result obtained from a ground-state
calculation. Without arguments, `energy` prints all the energy values
calculated. The options allows you to query a specific energy
value. In this case only the value will be printed without any other
text, so it is suitable for easy parsing in scripting. The values are
returned in Hartree.

These are the available subcommands:

- `result energy`

  When no arguments are given, `energy` will print all the energy values available.

  Example: `inq result energy`.


- `result energy total`

  Returns the total energy of the calculation. This includes the ionic
  contribution.

  Example: `inq result energy total`.


- `result energy kinetic`

  The electronic kinetic energy.

  Example: `inq result energy kinetic`.


- `result energy eigenvalues`

  The sum of the eigenvalues, weighed by the occupations.

  Example: `inq result energy eigenvalues`.


- `result energy Hartree`

  The classical electrostatic interaction energy between electrons.

  Example: `inq result energy Hartree`.


- `result energy external`

  The energy of the interaction of the electrons with the local
  potential generated by the ions. This doesn't include the non-local
  pseudopotential part.

  Example: `inq result energy external`.


- `result energy non-local`

  The energy of the interaction of the electrons with the non-local
  part of the ionic pseudo-potentials.

  Example: `inq result energy non-local`.


- `result energy xc`

  The exchange and correlation energy from DFT semi-local
  functionals. It doesn't include the contribution from Hartree-Fock
  exchange (see `energy exact_exchange`).

  Example: `inq result energy xc`.


- `result energy nvxc`

  The energy of the interaction of the exchange and correlation
  potential and the density. This is different from the exchange and
  correlation energy.

  Example: `inq result energy nvxc`.


- `result energy exact-exchange`

  The Hartree-Fock exact-exchange energy. This is calculated for
  Hartree-Fock and hybrid functionals.

  Example: `inq result energy exact-exchange`.


- `result energy ion`

  The ion-ion interaction energy. This value is calculated taking into
  account the periodicity of the system.

  Example: `inq result energy ion`.


)"""";
	}

	void operator()() const {
		auto res = ground_state::result::load(".inq/default_result");
		if(input::environment::global().comm().root()) std::cout << res;
	}

	auto iterations() const {
		auto res = ground_state::result::load(".inq/default_result");
		return res.total_iter;
	}

	auto magnetization() const {
		auto res = ground_state::result::load(".inq/default_result");
		return res.magnetization;
	}
	
	void energy() const {
		auto ener = ground_state::result::load(".inq/default_result").energy;
		if(input::environment::global().comm().root()) std::cout << ener;
	}
	
  double energy_total() const{
    return ground_state::result::load(".inq/default_result").energy.total();
  }
	
  double energy_kinetic() const{
    return ground_state::result::load(".inq/default_result").energy.kinetic();
  }

  double energy_eigenvalues() const{
    return ground_state::result::load(".inq/default_result").energy.eigenvalues();
  }

  double energy_external() const{
    return ground_state::result::load(".inq/default_result").energy.external();
  }
  
  double energy_non_local() const{
    return ground_state::result::load(".inq/default_result").energy.non_local();
  }
  
  double energy_hartree() const{
    return ground_state::result::load(".inq/default_result").energy.hartree();
  }
  
  double energy_xc() const{
    return ground_state::result::load(".inq/default_result").energy.xc();
  }

  double energy_nvxc() const{
    return ground_state::result::load(".inq/default_result").energy.nvxc();
  }

  double energy_exact_exchange() const{
    return ground_state::result::load(".inq/default_result").energy.exact_exchange();
  }
  
  double energy_ion() const{
    return ground_state::result::load(".inq/default_result").energy.ion();
  }

	template <typename ArgsType>
	void command(ArgsType args, bool quiet) const {

		if(args.size() == 0){
			operator()();
			exit(0);
		}
		
		if(args.size() == 1 and args[0] == "iterations"){
			std::cout << iterations() << std::endl;
			exit(0);
		}

		if(args.size() == 1 and args[0] == "magnetization"){
			std::cout << magnetization() << std::endl;
			exit(0);
		}

		if(args.size() == 2 and args[0] == "magnetization"){
			auto idir = -1;
			if(args[1] == "1" or args[1] == "x") idir = 0;
			if(args[1] == "2" or args[1] == "y") idir = 1;
			if(args[1] == "3" or args[1] == "z") idir = 2;

			if(idir == -1) {
				if(input::environment::global().comm().root()) std::cerr << "Error: Invalid syntax in the 'result magnetization' command" << std::endl;
				exit(1);
			}
			
			assert(idir >= 0 and idir <= 2);
			
			if(input::environment::global().comm().root())  printf("%.6f\n", magnetization()[idir]);
			exit(0);
		}

		
		if(args[0] == "energy"){

			args.erase(args.begin());

			if(args.size() == 0) {
				energy();
				exit(0);
			}

			if(args.size() == 1 and args[0] == "total"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_total());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "kinetic"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_kinetic());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "eigenvalues"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_eigenvalues());
				exit(0);
			}
    
			if(args.size() == 1 and args[0] == "external"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_external());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "non-local"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_non_local());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "hartree"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_hartree());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "xc"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_xc());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "nvxc"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_nvxc());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "exact-exchange"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_exact_exchange());
				exit(0);
			}
        
			if(args.size() == 1 and args[0] == "ion"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_ion());
				exit(0);
			}
		}
      
		if(input::environment::global().comm().root()) std::cerr << "Error: Invalid syntax in the 'result' command" << std::endl;
		exit(1);
    
	}
	
} const result;

}
}
#endif

#ifdef INQ_INTERFACE_RESULT_UNIT_TEST
#undef INQ_INTERFACE_RESULT_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
