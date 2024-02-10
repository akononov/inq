/* -*- indent-tabs-mode: t -*- */

#ifndef OPTIONS__REAL_TIME
#define OPTIONS__REAL_TIME

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <magnitude/time.hpp>
#include <utils/merge_optional.hpp>

#include <optional>
#include <cassert>


namespace inq {
namespace options {

class real_time {

public:

	enum class electron_propagator { ETRS, CRANK_NICOLSON };
	
	template<class OStream>
	friend OStream & operator<<(OStream & out, electron_propagator const & self){
		if(self == electron_propagator::ETRS)              out << "etrs";
		if(self == electron_propagator::CRANK_NICOLSON)    out << "crank-nicolson";
		return out;
	}

	template<class IStream>
	friend IStream & operator>>(IStream & in, electron_propagator & self){
		std::string readval;
		in >> readval;
		if(readval == "etrs"){
			self = electron_propagator::ETRS;
		} else if(readval == "crank-nicolson"){
			self = electron_propagator::CRANK_NICOLSON;
		} else {
			throw std::runtime_error("INQ error: Invalid propagation algorithm");
		}
		return in;
	}
	
private:

	std::optional<double> dt_;
	std::optional<int> num_steps_;
	std::optional<electron_propagator> prop_;

public:
	
	auto dt(quantity<magnitude::time> dt) const {
		real_time solver = *this;;
		solver.dt_ = dt.in_atomic_units();
		return solver;
	}

	auto dt() const {
		return dt_.value_or(0.01);
	}

	auto num_steps(double etol) const {
		real_time solver = *this;;
		solver.num_steps_ = etol;
		return solver;
	}
				
	auto num_steps() const {
		return num_steps_.value_or(100);
	}

	auto etrs() {
		real_time solver = *this;;
		solver.prop_ = electron_propagator::ETRS;
		return solver;
	}

	auto crank_nicolson() const {
		real_time solver = *this;;
		solver.prop_ = electron_propagator::CRANK_NICOLSON;
		return solver;
	}
	
	auto propagator() const {
		return prop_.value_or(electron_propagator::ETRS);
	}

	void save(parallel::communicator & comm, std::string const & dirname) const {
		auto error_message = "INQ error: Cannot save the options::real_time to directory '" + dirname + "'.";
		
		comm.barrier();

		auto exception_happened = true;
		if(comm.root()) {
			
			try { std::filesystem::create_directories(dirname); }
			catch(...) {
				comm.broadcast_value(exception_happened);
				throw std::runtime_error(error_message);
			}

			utils::save_optional(comm, dirname + "/time_step",     dt_,          error_message);
			utils::save_optional(comm, dirname + "/num_steps",     num_steps_,   error_message);
			utils::save_optional(comm, dirname + "/propagator",    prop_,        error_message);

			exception_happened = false;
			comm.broadcast_value(exception_happened);
			
		} else {
			comm.broadcast_value(exception_happened);
			if(exception_happened) throw std::runtime_error(error_message);
		}
		
		comm.barrier();
	}

	
	template<class OStream>
	friend OStream & operator<<(OStream & out, real_time const & self){
		
		using namespace magnitude;
		
		out << "Real-time:\n";
		
		out << "  time-step          = ";
		if(self.dt_.has_value()) {
			out << self.dt() << " atu | " << self.dt()/in_atomic_units(1.0_fs) << " fs";
		} else {
			out << "NOT SET *";
		}
		out << "\n";
		
		out << "  num-steps          = " << self.num_steps();
		if(not self.num_steps_.has_value()) out << " *";
		out << "\n";

		out << "\n  * default values" << std::endl;
		
		return out;
	}
	
};

}
}
#endif

#ifdef INQ_OPTIONS_REAL_TIME_UNIT_TEST
#undef INQ_OPTIONS_REAL_TIME_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

  using namespace inq;
  using namespace inq::magnitude;	
  using namespace Catch::literals;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	
	SECTION("Defaults"){

    options::real_time solver;

    CHECK(solver.dt() == 0.01_a);
    CHECK(solver.num_steps() == 100);
    CHECK(solver.propagator() == options::real_time::electron_propagator::ETRS);		

  }

  SECTION("Composition"){

    auto solver = options::real_time{}.num_steps(1000).dt(0.05_atomictime).crank_nicolson();
    
    CHECK(solver.num_steps() == 1000);
    CHECK(solver.dt() == 0.05_a);
		CHECK(solver.propagator() == options::real_time::electron_propagator::CRANK_NICOLSON);

		solver.save(comm, "save_real_time");

  }

}
#endif
