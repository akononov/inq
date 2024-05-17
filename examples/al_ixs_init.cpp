
#include <inq/inq.hpp>

int main(int argc, char ** argv){

	using namespace inq;
	using namespace inq::magnitude;

    auto & env = input::environment::global();

    auto ions = systems::ions::parse("POSCAR");

    systems::electrons electrons(env.par().states().domains(), ions, options::electrons{}.cutoff(500.0_eV).temperature(1.0_eV).extra_states(112));	

	auto functional = options::theory{}.lda();
	
	ground_state::initial_guess(ions, electrons);

	auto result = ground_state::calculate(ions, electrons, functional, inq::options::ground_state{}.energy_tolerance(1e-8_Ha));
	electrons.save("Al_restart");

    std::cout << "total energy = " << result.energy.total() << std::endl;
}
