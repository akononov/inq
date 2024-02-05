/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__RUN
#define INQ__INTERFACE__RUN

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <ground_state/initial_guess.hpp>
#include <ground_state/calculate.hpp>

namespace inq {
namespace interface {

struct {

	std::string name() const {
		return "run";
	}

	std::string one_line() const {
		return "Runs the simulation.";
	}
	
	void help() const {
		
		std::cout << R""""(

The 'run' command
==================

This command runs the actual simulation. It requires all the
simulation parameters to be set before running.

Note that this is the expensive part in the execution on
inq. Depending on the system you are using you might want to execute
this in parallel or through a queuing system.

These are the options available:

- `ground-state`

   Runs a ground-state calculation with fixed ions.

)"""";
		
		exit(0);
	}

	void ground_state() const{
		auto ions = systems::ions::load(".inq/default_ions");
		systems::electrons electrons(ions, options::electrons::load(".inq/default_electrons_options"));
		
		if(not electrons.try_load(".inq/default_orbitals")){
			ground_state::initial_guess(ions, electrons);
		}
		auto result = ground_state::calculate(ions, electrons, options::theory::load(".inq/default_theory"));

		result.energy.save(input::environment::global().comm(), ".inq/default_energy");
		electrons.save(".inq/default_orbitals");
	}

	template <typename ArgsType>
	void command(ArgsType const & args, bool quiet) const {
		
		if(args.size() == 0) {
			std::cerr << "Error: Missing argument to the 'run' command" << std::endl;
			exit(1);
		}
		
		if(args.size() == 1 and args[0] == "ground-state") {
			ground_state();
			exit(0);
		}
		
		std::cerr << "Error: Invalid syntax in the 'run' command" << std::endl;
		exit(1);
	}
	
} const run;

}
}
#endif

#ifdef INQ_INTERFACE_RUN_UNIT_TEST
#undef INQ_INTERFACE_RUN_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
