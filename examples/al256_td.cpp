/* -*- indent-tabs-mode: t -*- */

/*
 Copyright (C) 2019 Xavier Andrade, Alfredo A. Correa

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
#include <config/path.hpp>
#include <input/atom.hpp>
#include <operations/io.hpp>
#include <utils/match.hpp>
#include <ground_state/initialize.hpp>
#include <ground_state/calculate.hpp>
#include <input/environment.hpp>
#include <input/parse_xyz.hpp>
#include <config/path.hpp>

#include <real_time/propagate.hpp>

#include<fstream>

int main(int argc, char ** argv){

	using namespace inq;
	using namespace inq::magnitude;	
	
	input::environment env(argc, argv);

	boost::mpi3::communicator comm_world = boost::mpi3::environment::get_world_instance();
	
	utils::match energy_match(4.0e-6);

	auto geo = input::parse_xyz(config::path::unit_tests_data() + "al256.xyz", 1.0_bohr);
	geo.emplace_back("H" | math::vector3<double>(0.00000, 1.91325, 1.91325));

	systems::ions ions(input::cell::cubic(4*7.6524459_b), geo);
	
	input::config conf;

	conf.excess_charge = -1;
	conf.extra_states = 64;
	conf.temperature = 300.0_K;
	
	systems::electrons electrons(comm_world, ions, input::basis::cutoff_energy(25.0_Ha), conf);
	
	inq::operations::io::load("al256_restart", electrons.phi_);

	auto dt = 0.055_atomictime;

	ions.geo().velocities()[ions.geo().num_atoms() - 1] = math::vector3<double>(0.1, 0.0, 0.0);

	auto ofs = std::ofstream{"al256_v0.1.dat"}; ofs<< "# distance (au), energy (au)\n";
	
	for(int ii = 0; ii < 1; ii++){
		auto propagation = real_time::propagate(
			ions, electrons, input::interaction::pbe(), 
			input::rt::num_steps(10) | input::rt::dt(dt), ions::propagator::impulsive{}
		);

		for(std::size_t i = 0; i != propagation.coordinates.size(); ++i){
			ofs << propagation.coordinates[i][ions.geo().num_atoms() - 1][0] <<'\t'<< propagation.energy[i] << std::endl;
		}
	}
	
	return energy_match.fail();
	
}
