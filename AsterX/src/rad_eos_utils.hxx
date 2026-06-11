#ifndef ASTERX_RAD_EOS_UTILS_HXX
#define ASTERX_RAD_EOS_UTILS_HXX

#include <array>
#include <cassert>

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

} // namespace AsterX

#endif
