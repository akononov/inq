/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OBSERVABLES__SPECTRUM
#define INQ__OBSERVABLES__SPECTRUM

/*
 Copyright (C) 2019 Xavier Andrade, Alfredo Correa.

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

#include <inq_config.h>

#include <magnitude/energy.hpp>
#include <math/array.hpp>
#include <math/complex.hpp>

namespace inq {
namespace observables {

template <typename TimeType, typename TimeSeriesType, typename RetElementType = decltype(exp(complex{0.0, 1.0})*TimeSeriesType{}[0])>
math::array<RetElementType, 1> spectrum(quantity<magnitude::energy> maxw, quantity<magnitude::energy> dw, TimeType const & time, TimeSeriesType const & time_series) {

	CALI_CXX_MARK_FUNCTION;

  assert(time.size() == time_series.size());

  long ntime = time.size();
  long nfreq = maxw/dw + 1;
  
  math::array<RetElementType, 1> freq_series(nfreq);

  assert(freq_series.size() == nfreq);

  gpu::run(nfreq,
           [fse = begin(freq_series), tim = begin(time), tse = begin(time_series), ntime, dw] GPU_LAMBDA (auto ifreq){
             
             double ww = dw.in_atomic_units()*ifreq;

             RetElementType sum = 0.5*(tim[1] - tim[0])*tse[0]*exp(complex{0.0, 1.0}*0.0);
             for(long itime = 1; itime < ntime - 1; itime++){
               assert(tim[itime] > tim[itime - 1]);

               auto fract = tim[itime]/tim[ntime - 1];
               auto damp_factor = 1.0 - 3.0*fract*fract + 2.0*fract*fract*fract;
               
               sum += 0.5*damp_factor*(tim[itime + 1] - tim[itime - 1])*exp(complex{0.0, 1.0}*ww*tim[itime])*tse[itime];
             }

             //the damp function is zero for the last value
             //sum += 0.5*(tim[ntime - 1] - tim[ntime - 2])*exp(complex{0.0, 1.0}*ww*tim[ntime - 1])*tse[ntime - 1];
             
             fse[ifreq] = sum;
           }
           );

  return freq_series;
}


}
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#ifdef INQ_OBSERVABLES_SPECTRUM_UNIT_TEST
#undef INQ_OBSERVABLES_SPECTRUM_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <ions/unit_cell.hpp>
#include <fstream>

using namespace inq;
using namespace magnitude;
using namespace Catch::literals;

TEST_CASE("observables::spectrum", "[observables::spectrum]") {

  int ntime = 1000;
  double dtime = 0.1;

  math::array<double, 1> time(ntime);
  math::array<double, 1> tseries(ntime);
  math::array<math::vector3<double>, 1> vtseries(ntime);
  
  double freq1 = 10.0;
  double freq2 = 6.39;
  double amp1 = 2.0;
  double amp2 = -1.5;
  
  for(int itime = 0; itime < ntime; itime++){
    time[itime] = dtime*itime;
    auto func = amp1*cos(time[itime]*freq1) + amp2*sin(time[itime]*freq2);
    tseries[itime] = func;
    vtseries[itime] = math::vector3<double>{-1.0*func, 20.0*func, -0.3*func};
  }

  auto maxw = 20.0_Ha;
  auto dw = 0.1_Ha;
  
  auto fseries = observables::spectrum(maxw, dw, time, tseries);
  auto vfseries = observables::spectrum(maxw, dw, time, vtseries);

  auto nfreq = fseries.size();

  CHECK(nfreq == 201);

  CHECK(real(fseries[12]) == -0.2352749195_a);
  CHECK(imag(fseries[12]) == -0.0264556811_a);
  CHECK(real(vfseries[12][0]) == 0.2352749195_a);
  CHECK(imag(vfseries[12][0]) == 0.0264556811_a);
  CHECK(real(vfseries[12][1]) == -4.7054983903_a);
  CHECK(imag(vfseries[12][1]) == -0.5291136216_a);
  CHECK(real(vfseries[12][2]) == 0.0705824759_a);
  CHECK(imag(vfseries[12][2]) == 0.0079367043_a);
  
  CHECK(real(fseries[100]) == 50.1204711636_a);
  CHECK(imag(fseries[100]) == 0.0321104817_a);
  CHECK(real(vfseries[100][0]) == -50.1204711636_a);
  CHECK(imag(vfseries[100][0]) == -0.0321104817_a);
  CHECK(real(vfseries[100][1]) == 1002.4094232726_a);
  CHECK(imag(vfseries[100][1]) == 0.6422096339_a);
  CHECK(real(vfseries[100][2]) == -15.0361413491_a);
  CHECK(imag(vfseries[100][2]) == -0.0096331445_a);

  std::ofstream file("spectrum.dat");
  
  for(int ifreq = 0; ifreq < nfreq; ifreq++){
    file << ifreq*dw.in_atomic_units() << '\t' << real(fseries[ifreq]) << '\t' << imag(fseries[ifreq]) << '\t' << real(vfseries[ifreq]) << '\t' << real(vfseries[ifreq]) << std::endl;
  }
  
}

#endif
#endif
