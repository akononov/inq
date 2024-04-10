/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__PERTURBATIONS__IXS
#define INQ__PERTURBATIONS__IXS

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa, Yifan Yao
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

#include <math/vector3.hpp>
#include <magnitude/energy.hpp>
#include <perturbations/none.hpp>

namespace inq {
namespace perturbations {

class ixs : public perturbations::none {

public:
	ixs(quantity<magnitude::energy> amplitude, inq::vector3<double, covariant> q, quantity<magnitude::time> tdelay, quantity<magnitude::time> twidth):
		amplitude_(amplitude.in_atomic_units()),
        q_(q),
		tdelay_(tdelay.in_atomic_units()),
		twidth_(twidth.in_atomic_units())
	{}

	auto has_potential() const {
		return true;
	}
	
	template<typename PotentialType>
	void potential(const double time, PotentialType & potential) const {
		auto Vixs = [q = q_, tdelay = tdelay_, twidth = twidth_, amplitude = amplitude_, t = time](inq::vector3<double, contravariant> rr) {
		    return amplitude/sqrt(2.0*M_PI)/twidth * exp( pow((t-tdelay)/(2.0*twidth),2) ) * complex(cos(q.dot(rr)), sin(q.dot(rr)));
		};

		gpu::run(potential.basis().local_sizes()[2], potential.basis().local_sizes()[1], potential.basis().local_sizes()[0],
						 [point_op = potential.basis().point_op(), vk = begin(potential.cubic()), Vixs] GPU_LAMBDA (auto iz, auto iy, auto ix) {
							 auto rr = point_op.rvector(ix, iy, iz);
							 vk[ix][iy][iz] += Vixs(rr);
						 });
	}

private:
	double amplitude_;
	double tdelay_;
	double twidth_;
    inq::vector3<double, covariant> q_;
};

}
}
#endif

#ifdef INQ_PERTURBATIONS_IXS_UNIT_TEST
#undef INQ_PERTURBATIONS_IXS_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

using namespace inq;
using namespace Catch::literals;
using namespace magnitude;

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
    perturbations::ixs nop(1.0_Ha, {0,0,1}, 0.3_fs, 0.1_fs);
    CHECK(not nop.has_uniform_electric_field());
}
#endif
