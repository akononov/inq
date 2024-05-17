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

    auto & env = input::environment::global();

    auto ions = systems::ions::parse("POSCAR");
    auto functional = options::theory{}.lda();

    systems::electrons electrons(env.par(), ions, options::electrons{}.cutoff(500.0_eV).temperature(1.0_eV).extra_states(112));
    electrons.load("Al_restart");

	utils::match energy_match(3.0e-5);


    vector3<int> qi = {1,0,0};
//    auto qix = parallel::global_index(qi[0]);
//    auto qiy = parallel::global_index(qi[1]);
//    auto qiz = parallel::global_index(qi[2]);

    auto td = 0.01_fs;
    auto tw = 0.002_fs;
	auto kick = perturbations::ixs{0.001_eV * (1.0_fs/tw), qi, td, tw, "sin"};

   
    vector3<double, covariant> q = electrons.density_basis().reciprocal().point_op().gvector(qi[0], qi[1], qi[2]); 
    if(electrons.root()) {
        std::cout << "q vector (covariant): " << q << std::endl;
        std::cout << "q vector (cartesian): " << ions.cell().metric().to_cartesian(q) << std::endl;
    }
	
	auto const dt = 0.001_fs;
	long nsteps = 10000;
	
	gpu::array<double, 1> time(nsteps);
	gpu::array<complex, 1> nq(nsteps);
	gpu::array<double, 1> envt(nsteps);	
    gpu::array<double, 1> ne(nsteps);

	std::ofstream file;
	if(electrons.root()) {
        file.open("output.dat");
        file << "# time \t n(q) \t envelope(t) \t total electrons" << std::endl;
        file << std::fixed;
        file << std::setprecision(16);
    }
	
	auto output = [&](auto data){
	
        auto iter = data.iter();
	
		time[iter] = data.time();
        nq[iter] = data.density_q(qi);
		envt[iter] = data.ixs_envelope();
        ne[iter] = data.num_electrons();	
	
		if(data.root()) file << data.time() << '\t' << nq[iter] << '\t' << envt[iter] << '\t' << ne[iter] << std::endl;
		
	};
	
	real_time::propagate<>(ions, electrons, output, functional, options::real_time{}.num_steps(nsteps).dt(dt).etrs(), kick);
	
	return energy_match.fail();
	
}
