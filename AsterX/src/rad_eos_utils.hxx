#ifndef ASTERX_RAD_EOS_UTILS_HXX
#define ASTERX_RAD_EOS_UTILS_HXX

#include <array>
#include <cassert>
#include <type_traits>

#include <cctk.h>
#include <util_Table.h>

#include <loop_device.hxx>

#include "setup_eos.hxx"

namespace AsterX {

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

inline Loop::GF3D2<const CCTK_REAL> leakage_optd_gf(const cGH *cctkGH) {
  if (!CCTK_IsThornActive("LeakageBaseX"))
    CCTK_ERROR("Rad_idealgas requires active LeakageBaseX.");
  const int vi = CCTK_VarIndex("LeakageBaseX::optd");
  const int gi = CCTK_GroupIndex("LeakageBaseX::optical_depth");
  if (vi < 0 || gi < 0)
    CCTK_ERROR("Could not find LeakageBaseX::optd for Rad_idealgas.");
  const Loop::GF3D2layout layout(cctkGH, group_indextype(gi));
  return Loop::GF3D2<const CCTK_REAL>(
      layout, static_cast<const CCTK_REAL *>(CCTK_VarDataPtrI(cctkGH, 0, vi)));
}

template <typename EOSType>
inline Loop::GF3D2<const CCTK_REAL>
optional_leakage_optd_gf(const cGH *cctkGH,
                         const Loop::GF3D2<CCTK_REAL> &reference) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return leakage_optd_gf(cctkGH);
  } else {
    return Loop::GF3D2<const CCTK_REAL>(reference.layout, nullptr);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
local_optd(const Loop::GF3D2<const CCTK_REAL> &optd,
           const Loop::vect<int, Loop::dim> &I) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return optd(I);
  } else {
    return CCTK_REAL(0.0);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_press_from_rho_temp(const EOSType *eos, const CCTK_REAL rho,
                        const CCTK_REAL temp, const CCTK_REAL ye,
                        const CCTK_REAL optd) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->press_from_rho_temp_ye_tau(rho, temp, ye, optd);
  } else {
    return eos->press_from_rho_temp_ye(rho, temp, ye);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_eps_from_rho_temp(const EOSType *eos, const CCTK_REAL rho,
                      const CCTK_REAL temp, const CCTK_REAL ye,
                      const CCTK_REAL optd) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->eps_from_rho_temp_ye_tau(rho, temp, ye, optd);
  } else {
    return eos->eps_from_rho_temp_ye(rho, temp, ye);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_eps_from_rho_press(const EOSType *eos, const CCTK_REAL rho,
                       const CCTK_REAL press, const CCTK_REAL ye,
                       const CCTK_REAL optd) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->eps_from_rho_press_ye_tau(rho, press, ye, optd);
  } else {
    return eos->eps_from_rho_press_ye(rho, press, ye);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_temp_from_rho_eps(const EOSType *eos, const CCTK_REAL rho, CCTK_REAL &eps,
                      const CCTK_REAL ye, const CCTK_REAL optd) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->temp_from_rho_eps_ye_tau(rho, eps, ye, optd);
  } else {
    return eos->temp_from_rho_eps_ye(rho, eps, ye);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_press_from_rho_eps(const EOSType *eos, const CCTK_REAL rho, CCTK_REAL &eps,
                       const CCTK_REAL ye, const CCTK_REAL optd) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->press_from_rho_eps_ye_tau(rho, eps, ye, optd);
  } else {
    return eos->press_from_rho_eps_ye(rho, eps, ye);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_kappa_from_rho_eps(const EOSType *eos, const CCTK_REAL rho, CCTK_REAL &eps,
                       const CCTK_REAL ye, const CCTK_REAL optd) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->kappa_from_rho_eps_ye_tau(rho, eps, ye, optd);
  } else {
    return eos->kappa_from_rho_eps_ye(rho, eps, ye);
  }
}

template <typename EOSType>
CCTK_DEVICE CCTK_HOST inline CCTK_REAL
eos_csnd_from_rho_temp(const EOSType *eos, const CCTK_REAL rho,
                       const CCTK_REAL temp, const CCTK_REAL ye,
                       const CCTK_REAL optd) {
  if constexpr (std::is_same_v<EOSType, EOSX::eos_3p_rad_idealgas>) {
    return eos->csnd_from_rho_temp_ye_tau(rho, temp, ye, optd);
  } else {
    return eos->csnd_from_rho_temp_ye(rho, temp, ye);
  }
}

} // namespace AsterX

#endif
