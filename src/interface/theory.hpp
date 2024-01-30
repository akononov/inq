/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__THEORY
#define INQ__INTERFACE__THEORY

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <options/theory.hpp>

namespace inq {
namespace interface {

struct {		

	std::string name() const {
		return "theory";
	}

	std::string one_line() const {
		return "Defines the theory used to represent the electrons-electron interaction.";
	}

	void operator()() const {
		auto theo = options::theory::load(".default_theory");
		std::cout << theo;
	}
	
	void non_interacting() const{
		auto theo = options::theory::load(".default_theory").non_interacting();
		theo.save(input::environment::global().comm(), ".default_theory");
	}

	void hartree() const{
		auto theo = options::theory::load(".default_theory").hartree();
		theo.save(input::environment::global().comm(), ".default_theory");
	}

	void hartree_fock() const{
		auto theo = options::theory::load(".default_theory").hartree_fock();
		theo.save(input::environment::global().comm(), ".default_theory");
	}
	
	template <typename ArgsType>
	void command(ArgsType const & args, bool quiet) const {

		if(args.size() == 0){
			operator()();
			
		} else if((args.size() == 1 and args[0] == "non_interacting") or (args.size() == 1 and args[0] == "non-interacting") or (args.size() == 2 and args[0] == "non" and args[1] == "interacting")){
			non_interacting();
			
		} else if( args.size() == 1 and args[0] == "hartree"){
			hartree();
			
		} else if((args.size() == 1 and args[0] == "hartree-fock") or (args.size() == 1 and args[0] == "hartree_fock") or (args.size() == 2 and args[0] == "hartree" and args[1] == "fock")){
			hartree_fock();
			
		} else {				
			std::cerr << "Invalid syntax in 'theory' command" << std::endl;
			exit(1);
		}

		if(not quiet) operator()();
		exit(0);
	}
	
} const theory;

}
}
#endif

#ifdef INQ_INTERFACE_THEORY_UNIT_TEST
#undef INQ_INTERFACE_THEORY_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
