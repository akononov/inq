/* -*- indent-tabs-mode: t -*- */

//  Copyright (C) 2019-2020 Xavier Andrade, Alfredo A. Correa

#ifndef INQ__SYSTEMS__IONS
#define INQ__SYSTEMS__IONS

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <spglib.h>

#include <input/cif.hpp>
#include <input/species.hpp>
#include <ions/unit_cell.hpp>
#include <gpu/array.hpp>


namespace inq {
namespace systems {

class ions {

	inq::ions::unit_cell cell_;
	std::vector<input::species> atoms_;
	std::vector<vector3<double>> coordinates_;
	std::vector<vector3<double>> velocities_;	

	template <typename PositionType>
	void add_atom(input::species const & element, PositionType const & position){
		atoms_.push_back(element);
		coordinates_.push_back(in_atomic_units(position));
		velocities_.push_back(vector3<double>(0.0, 0.0, 0.0));					
	}

public:

	ions(inq::ions::unit_cell arg_cell_input):
		cell_(std::move(arg_cell_input)){
	}

	static ions parse(std::string filename) {

		std::string extension = filename.substr(filename.find_last_of(".") + 1);
		std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
		
		if(extension == "cif") {
			input::cif file(filename);
			ions parsed(file.cell());
			for(int ii = 0; ii < file.size(); ii++) parsed.insert_fractional(file.atoms()[ii], file.positions()[ii]);
			return parsed;
		}

		throw std::runtime_error("error: unsupported or unknwon file format '" + extension + "'.");
	}
	
	auto & atoms() const {
		return atoms_;
	}
	
	auto & coordinates() const {
		return coordinates_;
	}
	
	auto & coordinates() {
		return coordinates_;
	}

	auto & velocities() const {
		return velocities_;
	}
    
	auto & velocities() {
		return velocities_;
	}
	
	auto symmetry_string() const{

		assert(size() > 0);
		
		char symbol[11];
		
		std::vector<int> types(size());
		std::vector<double> positions(3*size());
		
		for(int iatom = 0; iatom < size(); iatom++){
			types[iatom] = atoms()[iatom].atomic_number();
			auto pos = cell_.metric().to_contravariant(cell_.position_in_cell(coordinates()[iatom]));
			positions[3*iatom + 0] = pos[0];
			positions[3*iatom + 1] = pos[1];
			positions[3*iatom + 2] = pos[2];
		}

		double amat[9];
		amat[0] = cell_.lattice(0)[0];
		amat[1] = cell_.lattice(0)[1];
		amat[2] = cell_.lattice(0)[2];
		amat[3] = cell_.lattice(1)[0];
		amat[4] = cell_.lattice(1)[1];
		amat[5] = cell_.lattice(1)[2];
		amat[6] = cell_.lattice(2)[0];
		amat[7] = cell_.lattice(2)[1];
		amat[8] = cell_.lattice(2)[2];
		
		auto symnum = spg_get_international(symbol, reinterpret_cast<double (*)[3]>(amat), reinterpret_cast<double (*)[3]>(positions.data()), types.data(), size(), 1e-4);
		return symbol + std::string(" (number ") + std::to_string(symnum) + std::string(")");
	}
	
	auto & cell() const {
		return cell_;
	}

	auto insert(input::species const & sp, vector3<quantity<magnitude::length>> const & pos){
		add_atom(sp, pos);
	}

	template <class ContainerType>
	auto insert(ContainerType const & container){
		for(auto atom : container) add_atom(atom.species(), atom.position());
	}

	void insert_fractional(input::species const & sp, vector3<double, contravariant> const & pos){
		add_atom(sp, cell_.metric().to_cartesian(pos));
	}
	
	int size() const {
		return (long) coordinates_.size();
	}
	
	template <class output_stream>
	void info(output_stream & out) const {
		out << "GEOMETRY:" << std::endl;
		out << "  Number of atoms = " << size() << std::endl;
		out << std::endl;
	}
	
	template<class OStream>
	friend OStream& operator<<(OStream& os, ions const& self){
		self.info(os);
		return os;
	}

	class atom {
		
		input::species species_;
		vector3<double> position_;
		
	public:
		
		atom(const input::species & arg_spec, const vector3<double> & arg_position):
			species_(arg_spec),
			position_(arg_position){
		}
		
		const auto & species() const {
			return species_;
		}
		
		const auto & position() const {
			return position_;
		}
		
	};

};

}
}
#endif

#ifdef INQ_SYSTEMS_IONS_UNIT_TEST
#undef INQ_SYSTEMS_IONS_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;

	SECTION("Create empty and add an atom"){
		
		auto dcc = 1.42_A;
		auto aa = sqrt(3)*dcc;
		auto lz = 10.0_b;
		systems::ions ions(ions::unit_cell::lattice(aa*vector3{1.0, 0.0, 0.0}, aa*vector3{-1.0/2.0, sqrt(3.0)/2.0, 0.0}, {0.0_b, 0.0_b, lz}).periodicity(2));
		
		CHECK(ions.cell().periodicity() == 2);
		
    CHECK(ions.size() == 0);

    ions.insert("Xe", {1000.0_b, -200.0_b, 6.0_b});

    CHECK(ions.size() == 1);
    CHECK(ions.atoms()[0].atomic_number() == 54);
    CHECK(ions.atoms()[0] == input::species(54));
    CHECK(ions.atoms()[0].charge() == -54.0_a);
    CHECK(ions.atoms()[0].mass() == 239333.5935636_a);
    CHECK(ions.coordinates()[0][0] == 1000.0_a);
    CHECK(ions.coordinates()[0][1] == -200.0_a);
    CHECK(ions.coordinates()[0][2] == 6.0_a);
		CHECK(ions.velocities()[0][0] == 0.0_a);
    CHECK(ions.velocities()[0][1] == 0.0_a);
    CHECK(ions.velocities()[0][2] == 0.0_a);
		
    ions.coordinates()[0][0] += 8;  
    
    CHECK(ions.coordinates()[0][0] == 1008.0_a);
		CHECK(ions.velocities().size() == ions.coordinates().size());
		
  }
	
	SECTION("Read an xyz file"){
		
		systems::ions ions(ions::unit_cell::cubic(66.6_A).finite());

    ions.insert(input::parse_xyz(config::path::unit_tests_data() + "benzene.xyz"));
		
    CHECK(ions.size() == 12);
    
    CHECK(ions.atoms()[2] == "C");
    CHECK(ions.atoms()[2].charge() == -6.0_a);
    CHECK(ions.atoms()[2].mass() == 21892.1617296_a);
    CHECK(ions.coordinates()[2][0] == 2.2846788549_a);
    CHECK(ions.coordinates()[2][1] == -1.3190288178_a);
    CHECK(ions.coordinates()[2][2] == 0.0_a);

    CHECK(ions.atoms()[11] == "H");
    CHECK(ions.atoms()[11].charge() == -1.0_a);
    CHECK(ions.atoms()[11].mass() == 1837.17994584_a);
    CHECK(ions.coordinates()[11][0] == -4.0572419367_a);
    CHECK(ions.coordinates()[11][1] == 2.343260364_a);
    CHECK(ions.coordinates()[11][2] == 0.0_a);
		CHECK(ions.velocities()[11][0] == 0.0_a);
    CHECK(ions.velocities()[11][1] == 0.0_a);
    CHECK(ions.velocities()[11][2] == 0.0_a);

		CHECK(ions.velocities().size() == ions.coordinates().size());
		
    ions.insert("Cl", {-3.0_b, 4.0_b, 5.0_b});

    CHECK(ions.size() == 13);
    CHECK(ions.atoms()[12].atomic_number() == 17);
    CHECK(ions.atoms()[12] == input::species(17));
    CHECK(ions.atoms()[12].charge() == -17.0_a);
    CHECK(ions.atoms()[12].mass() == 64614.105771_a);
    CHECK(ions.coordinates()[12][0] == -3.0_a);
    CHECK(ions.coordinates()[12][1] == 4.0_a);
    CHECK(ions.coordinates()[12][2] == 5.0_a);
		CHECK(ions.velocities()[12][0] == 0.0_a);
    CHECK(ions.velocities()[12][1] == 0.0_a);
    CHECK(ions.velocities()[12][2] == 0.0_a);

		CHECK(ions.velocities().size() == ions.coordinates().size());
		
  }

	SECTION("CIF - Al"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "Al.cif");

		CHECK(ions.cell().lattice(0)[0] == 7.634890386_a);
		CHECK(ions.cell().lattice(0)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(0)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[1] == 7.634890386_a);
		CHECK(ions.cell().lattice(1)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[2] == 7.634890386_a);		
		
		CHECK(ions.size() == 4);

		CHECK(ions.atoms()[0] == "Al");
		CHECK(ions.atoms()[1] == "Al");
		CHECK(ions.atoms()[2] == "Al");		
		CHECK(ions.atoms()[3] == "Al");

		CHECK(ions.coordinates()[0][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[0][2] == Approx(0.0).margin(1e-12));
		
		CHECK(ions.coordinates()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[1][1] == -3.817445193_a);
		CHECK(ions.coordinates()[1][2] == -3.817445193_a);

		CHECK(ions.coordinates()[2][0] == -3.817445193_a);
		CHECK(ions.coordinates()[2][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[2][2] == -3.817445193_a);

		CHECK(ions.coordinates()[3][0] == -3.817445193_a);
		CHECK(ions.coordinates()[3][1] == -3.817445193_a);
		CHECK(ions.coordinates()[3][2] == Approx(0.0).margin(1e-12));
	}
	
	SECTION("CIF - SrTiO3"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "9002806.cif");

		CHECK(ions.cell().lattice(0)[0] == 10.4316661532_a);
		CHECK(ions.cell().lattice(0)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(0)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[1] == 10.4316661532_a);
		CHECK(ions.cell().lattice(1)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[2] == 14.7525249371_a);		
		
		CHECK(ions.size() == 20);

		//Conversions for units and cell convention to compare with obabel
		//generated coordinates
		//
		//  1.38005 ->  2.6079165383
		//  1.95168 ->  3.6881312343
		//  2.76010 -> -5.2158330766
		//  3.90335 -> -7.3762624686
		//  4.14015 -> -2.6079165383
		//  5.85503 -> -3.6881312343
		
		//Sr         0.00000        0.00000        1.95168
		CHECK(ions.atoms()[0] == "Sr");
		CHECK(ions.coordinates()[0][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[0][2] == 3.6881312343_a);

		//Sr         0.00000        0.00000        5.85503
		CHECK(ions.atoms()[1] == "Sr");
		CHECK(ions.coordinates()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[1][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[1][2] == -3.6881312343_a);

		//Sr         2.76010        2.76010        5.85503
		CHECK(ions.atoms()[2] == "Sr");
		CHECK(ions.coordinates()[2][0] == -5.2158330766_a);
		CHECK(ions.coordinates()[2][1] == -5.2158330766_a);
		CHECK(ions.coordinates()[2][2] == -3.6881312343_a);

		//Sr         2.76010        2.76010        1.95167
		CHECK(ions.atoms()[3] == "Sr");
		CHECK(ions.coordinates()[3][0] == -5.2158330766_a);
		CHECK(ions.coordinates()[3][1] == -5.2158330766_a);
		CHECK(ions.coordinates()[3][2] ==  3.6881312343_a);

		//Ti         2.76010        0.00000        0.00000
		CHECK(ions.atoms()[4] == "Ti");
		CHECK(ions.coordinates()[4][0] == -5.2158330766_a);
		CHECK(ions.coordinates()[4][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[4][2] == Approx(0.0).margin(1e-12));

		//Ti         2.76010        0.00000        3.90335
		CHECK(ions.atoms()[5] == "Ti");
		CHECK(ions.coordinates()[5][0] == -5.2158330766_a);
		CHECK(ions.coordinates()[5][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[5][2] == -7.3762624686_a);

		//Ti         0.00000        2.76010        3.90335
		CHECK(ions.atoms()[6] == "Ti");
		CHECK(ions.coordinates()[6][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[6][1] == -5.2158330766_a);
		CHECK(ions.coordinates()[6][2] == -7.3762624686_a);

		//Ti         0.00000        2.76010        0.00000
		CHECK(ions.atoms()[7] == "Ti");
		CHECK(ions.coordinates()[7][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[7][1] == -5.2158330766_a);
		CHECK(ions.coordinates()[7][2] == Approx(0.0).margin(1e-12));

		//O          0.00000        2.76010        1.95168
		CHECK(ions.atoms()[8] == "O");
		CHECK(ions.coordinates()[8][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[8][1] == -5.2158330766_a);
		CHECK(ions.coordinates()[8][2] == 3.6881312343_a);

		//O          0.00000        2.76010        5.85503
		CHECK(ions.atoms()[9] == "O");
		CHECK(ions.coordinates()[9][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[9][1] == -5.2158330766_a);
		CHECK(ions.coordinates()[9][2] == -3.6881312343_a);

		//O          2.76010        0.00000        5.85503
		CHECK(ions.atoms()[10] == "O");
		CHECK(ions.coordinates()[10][0] == -5.2158330766_a);
		CHECK(ions.coordinates()[10][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[10][2] == -3.6881312343_a);

		//O          2.76010        0.00000        1.95167
		CHECK(ions.atoms()[11] == "O");
		CHECK(ions.coordinates()[11][0] == -5.2158330766_a);
		CHECK(ions.coordinates()[11][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[11][2] ==  3.6881312343_a);

		//O          4.14015        1.38005        0.00000
		CHECK(ions.atoms()[12] == "O");
		CHECK(ions.coordinates()[12][0] == -2.6079165383_a);
		CHECK(ions.coordinates()[12][1] ==  2.6079165383_a);
		CHECK(ions.coordinates()[12][2] == Approx(0.0).margin(1e-12));

		//4.14015        1.38005        3.90335
		CHECK(ions.atoms()[13] == "O");
		CHECK(ions.coordinates()[13][0] == -2.6079165383_a);
		CHECK(ions.coordinates()[13][1] ==  2.6079165383_a);
		CHECK(ions.coordinates()[13][2] == -7.3762624686_a);

		//O          1.38005        4.14015        3.90335
		CHECK(ions.atoms()[14] == "O");
		CHECK(ions.coordinates()[14][0] ==  2.6079165383_a);
		CHECK(ions.coordinates()[14][1] == -2.6079165383_a);
		CHECK(ions.coordinates()[14][2] == -7.3762624686_a);

		//O          1.38005        1.38005        3.90335
		CHECK(ions.atoms()[15] == "O");
		CHECK(ions.coordinates()[15][0] ==  2.6079165383_a);
		CHECK(ions.coordinates()[15][1] ==  2.6079165383_a);
		CHECK(ions.coordinates()[15][2] == -7.3762624686_a);

		//O          4.14015        4.14015        3.90335
		CHECK(ions.atoms()[16] == "O");
		CHECK(ions.coordinates()[16][0] == -2.6079165383_a);
		CHECK(ions.coordinates()[16][1] == -2.6079165383_a);
		CHECK(ions.coordinates()[16][2] == -7.3762624686_a);

		//O          4.14015        4.14015        0.00000
		CHECK(ions.atoms()[17] == "O");
		CHECK(ions.coordinates()[17][0] == -2.6079165383_a);
		CHECK(ions.coordinates()[17][1] == -2.6079165383_a);
		CHECK(ions.coordinates()[17][2] == Approx(0.0).margin(1e-12));

		//O          1.38005        1.38005        0.00000
		CHECK(ions.atoms()[18] == "O");
		CHECK(ions.coordinates()[18][0] ==  2.6079165383_a);
		CHECK(ions.coordinates()[18][1] ==  2.6079165383_a);
		CHECK(ions.coordinates()[18][2] == Approx(0.0).margin(1e-12));

		//O          1.38005        4.14015        0.00000
		CHECK(ions.atoms()[19] == "O");
		CHECK(ions.coordinates()[19][0] ==  2.6079165383_a);
		CHECK(ions.coordinates()[19][1] == -2.6079165383_a);
		CHECK(ions.coordinates()[19][2] == Approx(0.0).margin(1e-12));
		
	}

	SECTION("CIF - Ca2PI symmetrized"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "Ca2PI_symm.cif");

		CHECK(ions.cell().lattice(0)[0] == 8.1469916149_a);
		CHECK(ions.cell().lattice(0)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(0)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[0] == -4.0734958074_a);
		CHECK(ions.cell().lattice(1)[1] ==  7.0555017029_a);
		CHECK(ions.cell().lattice(1)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[2] == 42.0773092856_a);		
		
		CHECK(ions.size() == 12);
		
		CHECK(ions.atoms()[0] == "Ca");
		CHECK(ions.coordinates()[0][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[0][2] == 9.6727962369_a);
		
		CHECK(ions.atoms()[1] == "Ca");
		CHECK(ions.coordinates()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[1][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[1][2] == -9.6727962369_a);

		CHECK(ions.atoms()[2] == "Ca");
		CHECK(ions.coordinates()[2][0] ==  -4.0734958074_a);
		CHECK(ions.coordinates()[2][1] ==   2.3518339010_a);
		CHECK(ions.coordinates()[2][2] == -18.3787432869_a);
		
		CHECK(ions.atoms()[3] == "Ca");
		CHECK(ions.coordinates()[3][0] ==  -4.0734958074_a);
		CHECK(ions.coordinates()[3][1] ==   2.3518339010_a);
		CHECK(ions.coordinates()[3][2] ==   4.3529735250_a);
		
		CHECK(ions.atoms()[4] == "Ca");
		CHECK(ions.coordinates()[4][0] ==   4.0734958074_a);
		CHECK(ions.coordinates()[4][1] ==  -2.3518339010_a);
		CHECK(ions.coordinates()[4][2] ==  -4.3529735250_a);
		
		CHECK(ions.atoms()[5] == "Ca");
		CHECK(ions.coordinates()[5][0] ==   4.0734958074_a);
		CHECK(ions.coordinates()[5][1] ==  -2.3518339010_a);
		CHECK(ions.coordinates()[5][2] ==  18.3787432869_a);

		CHECK(ions.atoms()[6] == "P");
		CHECK(ions.coordinates()[6][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[6][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[6][2] == -21.0386546428_a);

		CHECK(ions.atoms()[7] == "P");
		CHECK(ions.coordinates()[7][0] == -4.0734958074_a);
		CHECK(ions.coordinates()[7][1] ==  2.3518339010_a);
		CHECK(ions.coordinates()[7][2] == -7.0128848809_a);

		CHECK(ions.atoms()[8] == "P");
		CHECK(ions.coordinates()[8][0] ==  4.0734958074_a);
		CHECK(ions.coordinates()[8][1] == -2.3518339010_a);
		CHECK(ions.coordinates()[8][2] ==  7.0128848809_a);

		CHECK(ions.atoms()[9] == "I");
		CHECK(ions.coordinates()[9][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[9][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[9][2] == Approx(0.0).margin(1e-12));

		CHECK(ions.atoms()[10] == "I");
		CHECK(ions.coordinates()[10][0] == -4.0734958074_a);
		CHECK(ions.coordinates()[10][1] ==  2.3518339010_a);
		CHECK(ions.coordinates()[10][2] == 14.0257697619_a);

		CHECK(ions.atoms()[11] == "I");
		CHECK(ions.coordinates()[11][0] ==   4.0734958074_a);
		CHECK(ions.coordinates()[11][1] ==  -2.3518339010_a);
		CHECK(ions.coordinates()[11][2] == -14.0257697619_a);
		
	}
	
	SECTION("CIF - Ca2PI not symmetrized"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "Ca2PI.cif");

		CHECK(ions.cell().lattice(0)[0] == 8.1469916149_a);
		CHECK(ions.cell().lattice(0)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(0)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[0] == -4.0734958074_a);
		CHECK(ions.cell().lattice(1)[1] ==  7.0555017029_a);
		CHECK(ions.cell().lattice(1)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[2] == 42.0773092856_a);		
		
		CHECK(ions.size() == 12);

		//the order her is to match the symmetrized test above
		CHECK(ions.atoms()[1] == "Ca");
		CHECK(ions.coordinates()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[1][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[1][2] == 9.6727962369_a);

		CHECK(ions.atoms()[4] == "Ca");
		CHECK(ions.coordinates()[4][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[4][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[4][2] == -9.6727962369_a);

		CHECK(ions.atoms()[3] == "Ca");
		CHECK(ions.coordinates()[3][0] ==  -4.0734958074_a);
		CHECK(ions.coordinates()[3][1] ==   2.3518339010_a);
		CHECK(ions.coordinates()[3][2] == -18.3787432869_a);

		CHECK(ions.atoms()[0] == "Ca");
		CHECK(ions.coordinates()[0][0] ==  -4.0734958074_a);
		CHECK(ions.coordinates()[0][1] ==   2.3518339010_a);
		CHECK(ions.coordinates()[0][2] ==   4.3529735250_a);

		CHECK(ions.atoms()[5] == "Ca");
		CHECK(ions.coordinates()[5][0] ==   4.0734958074_a);
		CHECK(ions.coordinates()[5][1] ==  -2.3518339010_a);
		CHECK(ions.coordinates()[5][2] ==  -4.3529735250_a);

		CHECK(ions.atoms()[2] == "Ca");
		CHECK(ions.coordinates()[2][0] ==   4.0734958074_a);
		CHECK(ions.coordinates()[2][1] ==  -2.3518339010_a);
		CHECK(ions.coordinates()[2][2] ==  18.3787432869_a);

		CHECK(ions.atoms()[7] == "P");
		CHECK(ions.coordinates()[7][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[7][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[7][2] == -21.0386546428_a);

		CHECK(ions.atoms()[8] == "P");
		CHECK(ions.coordinates()[8][0] == -4.0734958074_a);
		CHECK(ions.coordinates()[8][1] ==  2.3518339010_a);
		CHECK(ions.coordinates()[8][2] == -7.0128848809_a);
		
		CHECK(ions.atoms()[6] == "P");
		CHECK(ions.coordinates()[6][0] ==  4.0734958074_a);
		CHECK(ions.coordinates()[6][1] == -2.3518339010_a);
		CHECK(ions.coordinates()[6][2] ==  7.0128848809_a);

		CHECK(ions.atoms()[9] == "I");
		CHECK(ions.coordinates()[9][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[9][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[9][2] == Approx(0.0).margin(1e-12));

		CHECK(ions.atoms()[10] == "I");
		CHECK(ions.coordinates()[10][0] == -4.0734958074_a);
		CHECK(ions.coordinates()[10][1] ==  2.3518339010_a);
		CHECK(ions.coordinates()[10][2] == 14.0257697619_a);

		CHECK(ions.atoms()[11] == "I");
		CHECK(ions.coordinates()[11][0] ==   4.0734958074_a);
		CHECK(ions.coordinates()[11][1] ==  -2.3518339010_a);
		CHECK(ions.coordinates()[11][2] == -14.0257697619_a);
		
	}
	
	SECTION("CIF - Na"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "Na.cif");

		//These lattice vectors match openbabel
		CHECK(ions.cell().lattice(0)[0] == 17.7976863062_a);
		CHECK(ions.cell().lattice(0)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(0)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[0] == 16.3923048366_a);
		CHECK(ions.cell().lattice(1)[1] == 6.9318092872_a);
		CHECK(ions.cell().lattice(1)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[0] == 16.3923048366_a);
		CHECK(ions.cell().lattice(2)[1] ==  3.3234384423_a);
		CHECK(ions.cell().lattice(2)[2] ==  6.0831518898_a);		
		
		CHECK(ions.size() == 3);
		
		CHECK(ions.atoms()[0] == "Na");
		CHECK(ions.coordinates()[0][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.coordinates()[0][2] == Approx(0.0).margin(1e-12));

		CHECK(ions.atoms()[1] == "Na");
		CHECK(ions.coordinates()[1][0] == 11.2403978126_a);
		CHECK(ions.coordinates()[1][1] ==  2.2789211505_a);
		CHECK(ions.coordinates()[1][2] ==  1.3517980130_a);

		CHECK(ions.atoms()[2] == "Na");
		CHECK(ions.coordinates()[2][0] == -11.2403978126_a);
		CHECK(ions.coordinates()[2][1] ==  -2.2789211505_a);
		CHECK(ions.coordinates()[2][2] ==  -1.3517980130_a);
	}
	
}
#endif
