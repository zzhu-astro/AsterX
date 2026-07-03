#ifndef ASTERX_RAD_EOS_UTILS_HXX
#define ASTERX_RAD_EOS_UTILS_HXX

#include <array>
#include <cassert>
#include <cmath>
#include <type_traits>

#include <cctk.h>
#include <cctk_Parameters.h>
#include <util_Table.h>

#include <loop_device.hxx>

#include "setup_eos.hxx"

namespace AsterX {

inline CCTK_REAL leakage_radeos_ramp(const CCTK_REAL cctk_time) {
  if (!CCTK_IsThornActive("LeakageBaseX"))
    CCTK_ERROR("Rad_idealgas requires active LeakageBaseX.");

  int smooth_type;
  const void *const smooth_p =
      CCTK_ParameterGet("smooth_radeos", "LeakageBaseX", &smooth_type);
  if (smooth_p == nullptr || smooth_type != PARAMETER_BOOLEAN)
    CCTK_ERROR("Could not read LeakageBaseX::smooth_radeos.");

  const CCTK_INT smooth_radeos =
      *static_cast<const CCTK_INT *>(smooth_p);
  if (!smooth_radeos)
    return CCTK_REAL(1.0);

  int t_type;
  const void *const t_p =
      CCTK_ParameterGet("t_leakage", "LeakageBaseX", &t_type);
  if (t_p == nullptr || t_type != PARAMETER_REAL)
    CCTK_ERROR("Could not read LeakageBaseX::t_leakage.");

  const CCTK_REAL t_leakage = *static_cast<const CCTK_REAL *>(t_p);
  return fmin(fmax(cctk_time / (t_leakage + CCTK_REAL(1.0e-15)),
                   CCTK_REAL(0.0)),
              CCTK_REAL(1.0));
}

template <typename EOSType>
inline CCTK_REAL optional_radeos_ramp(const CCTK_REAL cctk_time) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return leakage_radeos_ramp(cctk_time);
  } else {
    return CCTK_REAL(1.0);
  }
}

inline std::array<int, Loop::dim> group_indextype(const int gi) {
  assert(gi >= 0);
  const int tags = CCTK_GroupTagsTableI(gi);
  assert(tags >= 0);
  std::array<CCTK_INT, Loop::dim> index;
  int iret = Util_TableGetIntArray(tags, Loop::dim, index.data(), "index");
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    const int centering = CCTK_GroupCenteringTableI(gi);
    assert(centering >= 0);
    iret =
        Util_TableGetIntArray(centering, Loop::dim, index.data(), "centering");
  }
  if (iret == UTIL_ERROR_TABLE_NO_SUCH_KEY) {
    index = {0, 0, 0};
  } else if (iret >= 0) {
    assert(iret == Loop::dim);
  } else {
    assert(0);
  }

  std::array<int, Loop::dim> indextype;
  for (int d = 0; d < Loop::dim; ++d)
    indextype[d] = index[d];
  return indextype;
}

// Bundle of the three LeakageBaseX optical-depth grid functions.
struct leakage_optd_gfs {
  Loop::GF3D2<const CCTK_REAL> tot, emit, abs;
};

inline leakage_optd_gfs leakage_optd_gf(const cGH *cctkGH) {
  if (!CCTK_IsThornActive("LeakageBaseX"))
    CCTK_ERROR("Rad_idealgas requires active LeakageBaseX.");
  const int gi = CCTK_GroupIndex("LeakageBaseX::optical_depth");
  const int vt = CCTK_VarIndex("LeakageBaseX::optd_tot");
  const int ve = CCTK_VarIndex("LeakageBaseX::optd_emit");
  const int va = CCTK_VarIndex("LeakageBaseX::optd_abs");
  if (gi < 0 || vt < 0 || ve < 0 || va < 0)
    CCTK_ERROR("Could not find LeakageBaseX::optd_{tot,emit,abs} for "
               "Rad_idealgas.");
  const Loop::GF3D2layout layout(cctkGH, group_indextype(gi));
  return {Loop::GF3D2<const CCTK_REAL>(
              layout, static_cast<const CCTK_REAL *>(
                          CCTK_VarDataPtrI(cctkGH, 0, vt))),
          Loop::GF3D2<const CCTK_REAL>(
              layout, static_cast<const CCTK_REAL *>(
                          CCTK_VarDataPtrI(cctkGH, 0, ve))),
          Loop::GF3D2<const CCTK_REAL>(
              layout, static_cast<const CCTK_REAL *>(
                          CCTK_VarDataPtrI(cctkGH, 0, va)))};
}

template <typename EOSType, typename T>
inline leakage_optd_gfs
optional_leakage_optd_gf(const cGH *cctkGH,
                         const Loop::GF3D2<T> &reference) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return leakage_optd_gf(cctkGH);
  } else {
    const Loop::GF3D2<const CCTK_REAL> null_gf(reference.layout, nullptr);
    return {null_gf, null_gf, null_gf};
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline EOSX::optical_depths
local_optd(const leakage_optd_gfs &od,
           const Loop::vect<int, Loop::dim> &I) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return EOSX::optical_depths{od.tot(I), od.emit(I), od.abs(I)};
  } else {
    return EOSX::optical_depths{0.0, 0.0, 0.0};
  }
}

// Convert (local optical depths, time ramp) into the scalar radiation
// prefactor. This is the ONLY place optical depths feed the EOS; all
// thermodynamic dispatch helpers below take the finished rad_pref scalar.
template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_rad_prefactor(const EOSType *eos, const EOSX::optical_depths &od,
                  const CCTK_REAL rad_ramp) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->rad_prefactor(od, rad_ramp);
  } else {
    return CCTK_REAL(0.0);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_press_from_rho_temp(const EOSType *eos, const CCTK_REAL rho,
                        const CCTK_REAL temp, const CCTK_REAL ye,
                        const CCTK_REAL rad_pref) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->press_from_rho_temp_ye_pref(rho, temp, ye, rad_pref);
  } else {
    return eos->press_from_rho_temp_ye(rho, temp, ye);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_eps_from_rho_temp(const EOSType *eos, const CCTK_REAL rho,
                      const CCTK_REAL temp, const CCTK_REAL ye,
                      const CCTK_REAL rad_pref) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->eps_from_rho_temp_ye_pref(rho, temp, ye, rad_pref);
  } else {
    return eos->eps_from_rho_temp_ye(rho, temp, ye);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_eps_from_rho_press(const EOSType *eos, const CCTK_REAL rho,
                       const CCTK_REAL press, const CCTK_REAL ye,
                       const CCTK_REAL rad_pref) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->eps_from_rho_press_ye_pref(rho, press, ye, rad_pref);
  } else {
    return eos->eps_from_rho_press_ye(rho, press, ye);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_temp_from_rho_eps(const EOSType *eos, const CCTK_REAL rho, CCTK_REAL &eps,
                      const CCTK_REAL ye, const CCTK_REAL rad_pref) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->temp_from_rho_eps_ye_pref(rho, eps, ye, rad_pref);
  } else {
    return eos->temp_from_rho_eps_ye(rho, eps, ye);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_press_from_rho_eps(const EOSType *eos, const CCTK_REAL rho, CCTK_REAL &eps,
                       const CCTK_REAL ye, const CCTK_REAL rad_pref) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->press_from_rho_eps_ye_pref(rho, eps, ye, rad_pref);
  } else {
    return eos->press_from_rho_eps_ye(rho, eps, ye);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_kappa_from_rho_eps(const EOSType *eos, const CCTK_REAL rho, CCTK_REAL &eps,
                       const CCTK_REAL ye, const CCTK_REAL rad_pref) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->kappa_from_rho_eps_ye_pref(rho, eps, ye, rad_pref);
  } else {
    return eos->kappa_from_rho_eps_ye(rho, eps, ye);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_csnd_from_rho_temp(const EOSType *eos, const CCTK_REAL rho,
                       const CCTK_REAL temp, const CCTK_REAL ye,
                       const CCTK_REAL rad_pref) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->csnd_from_rho_temp_ye_pref(rho, temp, ye, rad_pref);
  } else {
    return eos->csnd_from_rho_temp_ye(rho, temp, ye);
  }
}

} // namespace AsterX

#endif
