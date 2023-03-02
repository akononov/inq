/* -*- indent-tabs-mode: t -*- */

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

#include <systems/ions.hpp>
#include <systems/electrons.hpp>
#include <utils/match.hpp>
#include <ground_state/initial_guess.hpp>
#include <ground_state/calculate.hpp>

#include <input/environment.hpp>

int main(int argc, char ** argv){

	using namespace inq;
	using namespace inq::magnitude;
			
	input::environment env(argc, argv);
	
	utils::match energy_match(3.0e-6);

	{
		auto box = systems::box::cubic(10.0_b).cutoff_energy(30.0_Ha);
		
		systems::ions ions(box);
		systems::electrons electrons(env.par(), ions, box, input::kpoints::grid({1, 1, 3}), input::config::temperature(300.0_K) | input::config::extra_states(2) | input::config::excess_charge(14.0));
		
		ground_state::initial_guess(ions, electrons);
		auto result = ground_state::calculate(ions, electrons, input::interaction::lda(), inq::input::scf::energy_tolerance(1e-8_Ha));
		
		energy_match.check("total energy",        result.energy.total()    , -0.567966887008);
		energy_match.check("kinetic energy",      result.energy.kinetic()  ,  2.485678229661);
		energy_match.check("eigenvalues",         result.energy.eigenvalues, -1.488504431382);
		energy_match.check("hartree",             result.energy.hartree    ,  0.000000000000);	
		energy_match.check("XC energy",           result.energy.xc         , -3.053646498826);
		energy_match.check("XC density integral", result.energy.nvxc       , -3.974185425356);
	}

	{
		auto a = 10.0_b;
		auto box = systems::box::lattice({a/sqrt(2.0), a/2.0, a/2.0}, {-a/sqrt(2), a/2.0, a/2.0}, {0.0_b, -a/sqrt(2.0), a/sqrt(2.0)}).cutoff_energy(30.0_Ha);
		
		systems::ions ions(box);
		systems::electrons electrons(env.par(), ions, box, input::config::temperature(300.0_K) | input::config::extra_states(2) | input::config::excess_charge(14.0), input::kpoints::grid({1, 1, 3}));
		
		ground_state::initial_guess(ions, electrons);
		auto result = ground_state::calculate(ions, electrons, input::interaction::lda(), inq::input::scf::energy_tolerance(1e-8_Ha));
		
		energy_match.check("total energy",        result.energy.total()    , -0.567967153667);
		energy_match.check("kinetic energy",      result.energy.kinetic()  ,  2.485678254306);
		energy_match.check("eigenvalues",         result.energy.eigenvalues, -1.488505190013);
		energy_match.check("hartree",             result.energy.hartree    ,  0.000000000000);	
		energy_match.check("XC energy",           result.energy.xc         , -3.053646207742);
		energy_match.check("XC density integral", result.energy.nvxc       , -3.974185043857);
	}

	{
		auto a = 10.0_b;
		auto box = systems::box::lattice({0.0_b, a/2.0, a/2.0}, {a/2.0, 0.0_b, a/2.0}, {a/2.0, a/2.0, 0.0_b}).cutoff_energy(30.0_Ha);
		
		systems::ions ions(box);
		systems::electrons electrons(env.par(), ions, box, input::config::temperature(300.0_K) | input::config::extra_states(2) | input::config::excess_charge(18.0), input::kpoints::grid({1, 1, 1}, false));
		
		ground_state::initial_guess(ions, electrons);
		auto result = ground_state::calculate(ions, electrons, input::interaction::lda(), inq::input::scf::energy_tolerance(1e-8_Ha));
		
		energy_match.check("total energy",        result.energy.total()    ,  3.023858131207);
		energy_match.check("kinetic energy",      result.energy.kinetic()  ,  9.474820254369);
		energy_match.check("eigenvalues",         result.energy.eigenvalues,  1.054657761130);
		energy_match.check("hartree",             result.energy.hartree    ,  0.000000000000);	
		energy_match.check("XC energy",           result.energy.xc         , -6.450962126449);
		energy_match.check("XC density integral", result.energy.nvxc       , -8.420162499813);
	}
		
	return energy_match.fail();
	
}
