/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OPTIONS__ELECTRONS
#define INQ__OPTIONS__ELECTRONS

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <magnitude/energy.hpp>
#include <states/ks_states.hpp>
#include <utils/merge_optional.hpp>

#include <cassert>
#include <optional>

namespace inq {
namespace options {

class electrons {

	std::optional<int> extra_states_;
	std::optional<double> extra_electrons_;
	std::optional<double> temperature_;
	std::optional<states::ks_states::spin_config> spin_;
	std::optional<double> spacing_;
	std::optional<bool> double_grid_;	
	std::optional<double> density_factor_;
	std::optional<bool> spherical_grid_;
	std::optional<bool> fourier_pseudo_;
	std::optional<pseudo::set> pseudo_set_;

public:
	
	auto extra_states(int value){
		electrons conf = *this;
		conf.extra_states_ = value;
		return conf;
	}
	
	auto extra_states_val() const {
		return extra_states_.value_or(0);
	}

	auto extra_electrons(double value){
		electrons conf = *this;
		conf.extra_electrons_ = value;
		return conf;
	}

	auto extra_electrons_val() const {
		return extra_electrons_.value_or(0.0);
	}

	auto temperature(quantity<magnitude::energy> value){
		electrons conf = *this;
		conf.temperature_ = value.in_atomic_units();
		return conf;
	}

	auto temperature_val() const {
		return temperature_.value_or(0.0);
	}

	auto spin_unpolarized(){
		electrons conf = *this;
		conf.spin_ = states::ks_states::spin_config::UNPOLARIZED;
		return conf;
	}
	
	auto spin_polarized(){
		electrons conf = *this;
		conf.spin_ = states::ks_states::spin_config::POLARIZED;
		return conf;
	}

	auto spin_non_collinear(){
		electrons conf = *this;
		conf.spin_ = states::ks_states::spin_config::NON_COLLINEAR;
		return conf;
	}
	
	auto spin_val() const {
		return spin_.value_or(states::ks_states::spin_config::UNPOLARIZED);
	}
	
	auto num_spin_components_val() const {
		if(spin_val() == states::ks_states::spin_config::POLARIZED) return 2;
		return 1;
	}

	auto cutoff(quantity<magnitude::energy> arg_ecut){
		electrons conf = *this;
		conf.spacing_ = M_PI*sqrt(0.5/arg_ecut.in_atomic_units());
		return conf;		
	}

	auto spacing(quantity<magnitude::length> arg_spacing){
		electrons conf = *this;
		conf.spacing_ = arg_spacing.in_atomic_units();
		return conf;		
	}

	auto spacing_value() const {
		if(not spacing_.has_value()) throw std::runtime_error("Error: the cutoff energy or the spacing have not been set");
		return *spacing_;
	}

	auto double_grid(){
		electrons conf = *this;
		conf.double_grid_ = true;
		return conf;				
	}
	
	auto double_grid_value() const {
		return double_grid_.value_or(false);
	}

	auto density_factor(double arg_factor){
		electrons conf = *this;
		conf.density_factor_ = arg_factor;
		return conf;
	}

	auto density_factor_value() const {
		return density_factor_.value_or(1.0);
	}

	auto real_space_pseudo() const {
		electrons conf = *this;
		conf.fourier_pseudo_ = false;
		return conf;
	}

	auto fourier_pseudo() const {
		electrons conf = *this;
		conf.fourier_pseudo_ = true;
		return conf;
	}
		
	auto fourier_pseudo_value() const {
		return fourier_pseudo_.value_or(false);
	}

	auto pseudopotentials(pseudo::set && set){
		electrons conf = *this;
		conf.pseudo_set_.emplace(std::forward<pseudo::set>(set));
		return conf;
	}

	auto pseudopotentials_value() const{
		return pseudo_set_.value_or(pseudo::set::pseudodojo_pbe());
	}

	template <typename Type>
	static void save_value(parallel::communicator & comm, std::string const & filename, Type const & value, std::string const & error_message) {
		if(not value.has_value()) return;
		
		auto file = std::ofstream(filename);
		file.precision(25);
		
		if(not file) {
			auto exception_happened = true;
			comm.broadcast_value(exception_happened);
			throw std::runtime_error(error_message);
		}
		file << *value << std::endl;
	}

	void save(parallel::communicator & comm, std::string const & dirname) const {
		auto error_message = "INQ error: Cannot save the options::electrons to directory '" + dirname + "'.";
		
		auto exception_happened = true;
		if(comm.root()) {
			
			try { std::filesystem::create_directories(dirname); }
			catch(...) {
				comm.broadcast_value(exception_happened);
				throw std::runtime_error(error_message);
			}

			save_value(comm, dirname + "/extra_states", extra_states_, error_message);
			save_value(comm, dirname + "/extra_electrons", extra_electrons_, error_message);
			save_value(comm, dirname + "/temperature", temperature_, error_message);
			save_value(comm, dirname + "/spacing", spacing_, error_message);
			save_value(comm, dirname + "/double_grid", double_grid_, error_message);
			save_value(comm, dirname + "/density_factor", density_factor_, error_message);
			save_value(comm, dirname + "/spherical_grid", spherical_grid_, error_message);
			save_value(comm, dirname + "/fourier_pseudo", fourier_pseudo_, error_message);			

			//SPIN
			if(spin_.has_value()){
				auto file = std::ofstream(dirname + "/spin");
				
				if(not file) {
					auto exception_happened = true;
					comm.broadcast_value(exception_happened);
					throw std::runtime_error(error_message);
				}

				if(*spin_ == states::ks_states::spin_config::UNPOLARIZED){
					file << "unpolarized" << std::endl;
				} else if(*spin_ == states::ks_states::spin_config::POLARIZED){
					file << "polarized" << std::endl;
				} else if(*spin_ == states::ks_states::spin_config::NON_COLLINEAR){
					file << "non_collinear" << std::endl;
				}
			}

			//PSEUDO_SET
			if(pseudo_set_.has_value()){
				auto file = std::ofstream(dirname + "/pseudo_set");
				
				if(not file) {
					auto exception_happened = true;
					comm.broadcast_value(exception_happened);
					throw std::runtime_error(error_message);
				}

				file << pseudo_set_->path() << std::endl;
			}

			exception_happened = false;
			comm.broadcast_value(exception_happened);
			
		} else {
			comm.broadcast_value(exception_happened);
			if(exception_happened) throw std::runtime_error(error_message);
		}
		
		comm.barrier();
	}


	template <typename Type>
	static void load_value(std::string const & filename, std::optional<Type> & value) {
		auto file = std::ifstream(filename);
		if(file){
			Type readval;
			file >> readval;
			value = readval;
		}
	}
	
	static auto load(std::string const & dirname) {
		electrons opts;

		load_value(dirname + "/extra_states", opts.extra_states_);
		load_value(dirname + "/extra_electrons", opts.extra_electrons_);
		load_value(dirname + "/temperature", opts.temperature_);
		load_value(dirname + "/spacing", opts.spacing_);
		load_value(dirname + "/double_grid", opts.double_grid_);
		load_value(dirname + "/density_factor", opts.density_factor_);
		load_value(dirname + "/spherical_grid", opts.spherical_grid_);
		load_value(dirname + "/fourier_pseudo", opts.fourier_pseudo_);
		
		return opts;
	}
	
};

}
}
#endif

#ifdef INQ_OPTIONS_ELECTRONS_UNIT_TEST
#undef INQ_OPTIONS_ELECTRONS_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	using Catch::Approx;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	auto conf = options::electrons{}.spacing(23.1_b).extra_states(666).spin_non_collinear().pseudopotentials(pseudo::set::ccecp());

	CHECK(conf.extra_states_val() == 666);
	CHECK(conf.spacing_value() == 23.1_a);
	CHECK(conf.fourier_pseudo_value() == false);

	conf.save(comm, "options_electrons_save");
	auto read_conf = options::electrons::load("options_electrons_save");

	CHECK(read_conf.extra_states_val() == 666);
	CHECK(read_conf.spacing_value() == 23.1_a);
	CHECK(read_conf.fourier_pseudo_value() == false);	
}
#endif
