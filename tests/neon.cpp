/* -*- indent-tabs-mode: t -*- */

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq/inq.hpp>

int main(int argc, char ** argv){

	using namespace inq;
	using namespace inq::magnitude;	

	auto comm = input::environment::global().comm();

	utils::match energy_match(2.0e-5);

	interface::clear();
	
	//inq cell cubic 15.0 bohr finite
	interface::cell.cubic(15.0_b, 0);

	//inq cell
	interface::cell();

	//inq ions add Ne 0.0 0.0 0.0 bohr
	interface::ions.add("Ne", {0.0_b, 0.0_b, 0.0_b});

  //inq ions
	interface::ions();
	
	//inq electrons extra_states 3
	interface::electrons.extra_states(3);
	
	//inq electrons cutoff 30.0 Ha
	interface::electrons.cutoff(30.0_Ha);

	//inq theory non_interacting
	interface::theory.non_interacting();

	//REAL SPACE PSEUDO
	//inq run ground_state
	interface::run.ground_state();
	
	energy_match.check("total energy",     interface::energy.total()      , -61.861056649453);
	energy_match.check("kinetic energy",   interface::energy.kinetic()    ,  35.765610219604);
	energy_match.check("eigenvalues",      interface::energy.eigenvalues(), -61.861045337100);
	energy_match.check("external energy",  interface::energy.external()   , -79.509954154661);
	energy_match.check("non-local energy", interface::energy.non_local()  , -18.116701402044);
	energy_match.check("hartree",          interface::energy.hartree()    ,   0.000000000000);
	energy_match.check("xc",               interface::energy.xc()         ,   0.000000000000);
	energy_match.check("nvxc",             interface::energy.nvxc()       ,   0.000000000000);
	energy_match.check("ion-ion energy",   interface::energy.ion()        ,   0.000000000000);

	//FOURIER SPACE PSEUDO
	//inq electrons fourier_pseudo
	interface::electrons.fourier_pseudo();

	//inq run ground_state
	interface::run.ground_state();
	
	energy_match.check("total energy",     interface::energy.total()      , -61.861056649453);
	energy_match.check("kinetic energy",   interface::energy.kinetic()    ,  35.765555684056);
	energy_match.check("eigenvalues",      interface::energy.eigenvalues(), -61.861056649453);
	energy_match.check("external energy",  interface::energy.external()   , -79.509918897873);
	energy_match.check("non-local energy", interface::energy.non_local()  , -18.116693435635);
	energy_match.check("hartree",          interface::energy.hartree()    ,   0.000000000000);
	energy_match.check("xc",               interface::energy.xc()         ,   0.000000000000);
	energy_match.check("nvxc",             interface::energy.nvxc()       ,   0.000000000000);
	energy_match.check("ion-ion energy",   interface::energy.ion()        ,   0.000000000000);

	interface::energy();
	
	return energy_match.fail();
}
