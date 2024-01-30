/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE
#define INQ__INTERFACE

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <systems/ions.hpp>
#include <systems/electrons.hpp>
#include <ground_state/initial_guess.hpp>
#include <ground_state/calculate.hpp>

namespace inq {
namespace interface {

void clear(){
	if(input::environment::global().comm().root()) {
		std::filesystem::remove_all(".default_ions");
		std::filesystem::remove_all(".default_theory");
		std::filesystem::remove_all(".default_electrons_options");
		std::filesystem::remove_all(".default_orbitals");
	}
	input::environment::global().comm().barrier();
}

void cell_cubic(quantity<magnitude::length> const aa, int periodicity = 3){
  systems::ions ions(systems::cell::cubic(aa).periodicity(periodicity));
  ions.save(input::environment::global().comm(), ".default_ions");
}

void cell() {
  auto cell = systems::ions::load(".default_ions").cell();
  if(input::environment::global().comm().root()) std::cout << cell;
}

struct {

	void operator()(){
		auto ions = systems::ions::load(".default_ions");		
		if(input::environment::global().comm().root()) std::cout << ions;
	}

	static void add(input::species const & sp, vector3<quantity<magnitude::length>> const & pos){
		auto ions = systems::ions::load(".default_ions");
		ions.insert(sp, pos);
		ions.save(input::environment::global().comm(), ".default_ions");
	}

	static void clear(){
		auto ions = systems::ions::load(".default_ions");
		ions.clear();
		ions.save(input::environment::global().comm(), ".default_ions");
	}
	
} ions;
	
void electrons_extra_states(int nstates){
	auto el_opts = options::electrons::load(".default_electrons_options").extra_states(nstates);
	el_opts.save(input::environment::global().comm(), ".default_electrons_options");
}

void electrons_cutoff(quantity<magnitude::energy> ecut){
	auto el_opts = options::electrons::load(".default_electrons_options").cutoff(ecut);
	el_opts.save(input::environment::global().comm(), ".default_electrons_options");
}

void electrons_fourier_pseudo(){
	auto el_opts = options::electrons::load(".default_electrons_options").fourier_pseudo();
	el_opts.save(input::environment::global().comm(), ".default_electrons_options");
}

void theory_non_interacting(){
	auto theo = options::theory{}.non_interacting();
	theo.save(input::environment::global().comm(), ".default_theory");
}

auto run_ground_state(){
	auto ions = systems::ions::load(".default_ions");
	systems::electrons electrons(ions, options::electrons::load(".default_electrons_options"));

	if(not electrons.try_load(".default_orbitals")){
		ground_state::initial_guess(ions, electrons);
	}
	auto result = ground_state::calculate(ions, electrons, options::theory::load(".default_theory"));
	electrons.save(".default_orbitals");
	return result;
}

}
}
#endif

#ifdef INQ_INQ_INTERFACE_UNIT_TEST
#undef INQ_INQ_INTERFACE_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
