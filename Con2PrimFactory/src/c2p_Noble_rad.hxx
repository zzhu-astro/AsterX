#ifndef C2P_NOBLE_RAD_HXX
#define C2P_NOBLE_RAD_HXX

#include "c2p_2DNoble.hxx"

namespace Con2PrimFactory {

class c2p_Noble_rad : public c2p_2DNoble {
public:
  template <typename EOSType>
  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline c2p_Noble_rad(
      const EOSType *eos_3p, const atmosphere &atm, CCTK_INT maxIter,
      CCTK_REAL tol, CCTK_REAL alp_thresh_in, CCTK_REAL vwlim, CCTK_REAL B_lim,
      CCTK_REAL rho_BH_in, CCTK_REAL eps_BH_in, CCTK_REAL vwlim_BH_in,
      CCTK_REAL sigma_max_in, CCTK_REAL inv_beta_max_in, bool ye_len,
      bool use_z, bool use_temperature, bool use_pressure_atmo,
      bool soft_root_conv, CCTK_REAL soft_root_width_factor_in)
      : c2p_2DNoble(eos_3p, atm, maxIter, tol, alp_thresh_in, vwlim, B_lim,
                    rho_BH_in, eps_BH_in, vwlim_BH_in, sigma_max_in,
                    inv_beta_max_in, ye_len, use_z, use_temperature,
                    use_pressure_atmo, soft_root_conv,
                    soft_root_width_factor_in) {}

  template <typename EOSType>
  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
  solve(const EOSType *eos_3p, prim_vars &pv, prim_vars &pv_seeds,
        cons_vars &cv, const CCTK_REAL alp, const vec<CCTK_REAL, 3> &beta,
        const smat<CCTK_REAL, 3> &glo, const CCTK_REAL tau_opt,
        c2p_report &rep) const;

  template <typename EOSType, bool limiting>
  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
  bh_interior_tau(const EOSType *eos_3p, prim_vars &pv, cons_vars &cv,
                  const smat<CCTK_REAL, 3> &glo,
                  const CCTK_REAL tau_opt) const;

private:
  template <typename EOSType>
  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
  prims_floors_and_ceilings_tau(const EOSType *eos_3p, prim_vars &pv,
                                const cons_vars &cv, const CCTK_REAL alp,
                                const vec<CCTK_REAL, 3> &beta,
                                const smat<CCTK_REAL, 3> &glo,
                                const CCTK_REAL tau_opt,
                                const CCTK_REAL rad_pref,
                                c2p_report &rep) const;

  template <typename EOSType>
  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
  residual(const EOSType *eos_3p, const cons_vars &cv, const CCTK_REAL Ssq,
           const CCTK_REAL Bsq, const CCTK_REAL BiSi,
           const CCTK_REAL rad_pref, const CCTK_REAL x[3],
           CCTK_REAL r[3]) const {
    const CCTK_REAL Zloc = x[0];
    const CCTK_REAL Vsq = x[1];
    const CCTK_REAL temp = x[2];
    const CCTK_REAL invZ = CCTK_REAL(1.0) / Zloc;
    const CCTK_REAL q = sqrt(fmax(CCTK_REAL(1.0) - Vsq, CCTK_REAL(1.0e-14)));
    const CCTK_REAL rho = cv.dens * q;
    const CCTK_REAL erad = eos_3p->rad_energy_density_pref(temp, rad_pref);
    const CCTK_REAL press = rho * temp + erad / CCTK_REAL(3.0);
    const CCTK_REAL q2 = CCTK_REAL(1.0) - Vsq;

    r[0] = Ssq - Vsq * (Bsq + Zloc) * (Bsq + Zloc) +
           BiSi * BiSi * invZ * invZ * (Bsq + CCTK_REAL(2.0) * Zloc);
    r[1] = cv.tau + cv.dens - CCTK_REAL(0.5) * Bsq *
                                      (CCTK_REAL(1.0) + Vsq) +
           CCTK_REAL(0.5) * BiSi * BiSi * invZ * invZ - Zloc + press;
    r[2] = Zloc * q2 / rho - CCTK_REAL(1.0) -
           eos_3p->gamma * temp / eos_3p->gm1 -
           CCTK_REAL(4.0) * erad / (CCTK_REAL(3.0) * rho);
  }

  template <typename EOSType>
  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
  jacobian(const EOSType *eos_3p, const cons_vars &cv, const CCTK_REAL Bsq,
           const CCTK_REAL BiSi, const CCTK_REAL rad_pref,
           const CCTK_REAL x[3], CCTK_REAL J[3][3]) const {
    const CCTK_REAL Zloc = x[0];
    const CCTK_REAL Vsq = x[1];
    const CCTK_REAL temp = x[2];
    const CCTK_REAL Z2 = Zloc * Zloc;
    const CCTK_REAL Z3 = Z2 * Zloc;
    const CCTK_REAL BpZ = Bsq + Zloc;
    const CCTK_REAL C = BiSi * BiSi;
    const CCTK_REAL q = sqrt(fmax(CCTK_REAL(1.0) - Vsq, CCTK_REAL(1.0e-14)));
    const CCTK_REAL rho = cv.dens * q;
    const CCTK_REAL drho_dvsq =
        -cv.dens / (CCTK_REAL(2.0) * q);
    const CCTK_REAL temp2 = temp * temp;
    const CCTK_REAL temp3 = temp2 * temp;
    const CCTK_REAL temp4 = temp2 * temp2;
    const CCTK_REAL erad = rad_pref * temp4;
    const CCTK_REAL q2 = CCTK_REAL(1.0) - Vsq;

    const CCTK_REAL dP_dvsq = temp * drho_dvsq;
    const CCTK_REAL dP_dtemp =
        rho + CCTK_REAL(4.0) * rad_pref * temp3 / CCTK_REAL(3.0);

    J[0][0] = -CCTK_REAL(2.0) * Vsq * BpZ -
              CCTK_REAL(2.0) * C * BpZ / Z3;
    J[0][1] = -BpZ * BpZ;
    J[0][2] = CCTK_REAL(0.0);

    J[1][0] = -C / Z3 - CCTK_REAL(1.0);
    J[1][1] = -CCTK_REAL(0.5) * Bsq + dP_dvsq;
    J[1][2] = dP_dtemp;

    J[2][0] = q2 / rho;
    J[2][1] =
        -Zloc / (CCTK_REAL(2.0) * rho) +
        CCTK_REAL(4.0) * erad * drho_dvsq /
            (CCTK_REAL(3.0) * rho * rho);
    J[2][2] = -eos_3p->gamma / eos_3p->gm1 -
              CCTK_REAL(16.0) * rad_pref * temp3 /
                  (CCTK_REAL(3.0) * rho);
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline bool
  solve_linear3(CCTK_REAL A[3][3], CCTK_REAL b[3], CCTK_REAL x[3]) const {
    for (int col = 0; col < 3; ++col) {
      int piv = col;
      CCTK_REAL piv_abs = fabs(A[col][col]);
      for (int row = col + 1; row < 3; ++row) {
        const CCTK_REAL a = fabs(A[row][col]);
        if (a > piv_abs) {
          piv = row;
          piv_abs = a;
        }
      }
      if (piv_abs <= CCTK_REAL(1.0e-300))
        return false;
      if (piv != col) {
        for (int j = col; j < 3; ++j) {
          const CCTK_REAL tmp = A[col][j];
          A[col][j] = A[piv][j];
          A[piv][j] = tmp;
        }
        const CCTK_REAL tmp = b[col];
        b[col] = b[piv];
        b[piv] = tmp;
      }
      const CCTK_REAL inv_piv = CCTK_REAL(1.0) / A[col][col];
      for (int row = col + 1; row < 3; ++row) {
        const CCTK_REAL fac = A[row][col] * inv_piv;
        for (int j = col; j < 3; ++j)
          A[row][j] -= fac * A[col][j];
        b[row] -= fac * b[col];
      }
    }
    for (int row = 2; row >= 0; --row) {
      CCTK_REAL sum = b[row];
      for (int j = row + 1; j < 3; ++j)
        sum -= A[row][j] * x[j];
      x[row] = sum / A[row][row];
    }
    return true;
  }

  template <typename EOSType>
  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
  WZT2Prim(CCTK_REAL Z_Sol, CCTK_REAL vsq_Sol, CCTK_REAL temp_Sol,
           CCTK_REAL Bsq, CCTK_REAL BiSi, const EOSType *eos_3p, prim_vars &pv,
           const cons_vars &cv, const smat<CCTK_REAL, 3> &gup,
           const smat<CCTK_REAL, 3> &glo, const CCTK_REAL rad_pref) const {
    const CCTK_REAL W_Sol = CCTK_REAL(1.0) / sqrt(CCTK_REAL(1.0) - vsq_Sol);

    pv.rho = cv.dens / W_Sol;

    if (use_zprim) {
      CCTK_REAL zx = W_Sol *
                     (gup(X, X) * cv.mom(X) + gup(X, Y) * cv.mom(Y) +
                      gup(X, Z) * cv.mom(Z)) /
                     (Z_Sol + Bsq);
      zx += W_Sol * BiSi * cv.dBvec(X) / (Z_Sol * (Z_Sol + Bsq));

      CCTK_REAL zy = W_Sol *
                     (gup(X, Y) * cv.mom(X) + gup(Y, Y) * cv.mom(Y) +
                      gup(Y, Z) * cv.mom(Z)) /
                     (Z_Sol + Bsq);
      zy += W_Sol * BiSi * cv.dBvec(Y) / (Z_Sol * (Z_Sol + Bsq));

      CCTK_REAL zz = W_Sol *
                     (gup(X, Z) * cv.mom(X) + gup(Y, Z) * cv.mom(Y) +
                      gup(Z, Z) * cv.mom(Z)) /
                     (Z_Sol + Bsq);
      zz += W_Sol * BiSi * cv.dBvec(Z) / (Z_Sol * (Z_Sol + Bsq));

      const CCTK_REAL zx_down =
          glo(X, X) * zx + glo(X, Y) * zy + glo(X, Z) * zz;
      const CCTK_REAL zy_down =
          glo(X, Y) * zx + glo(Y, Y) * zy + glo(Y, Z) * zz;
      const CCTK_REAL zz_down =
          glo(X, Z) * zx + glo(Y, Z) * zy + glo(Z, Z) * zz;
      const CCTK_REAL Zsq = zx * zx_down + zy * zy_down + zz * zz_down;
      const CCTK_REAL SafeLor = sqrt(CCTK_REAL(1.0) + Zsq);

      pv.vel(X) = zx / SafeLor;
      pv.vel(Y) = zy / SafeLor;
      pv.vel(Z) = zz / SafeLor;
      pv.w_lor = SafeLor;
    } else {
      pv.vel(X) = (gup(X, X) * cv.mom(X) + gup(X, Y) * cv.mom(Y) +
                   gup(X, Z) * cv.mom(Z)) /
                  (Z_Sol + Bsq);
      pv.vel(X) += BiSi * cv.dBvec(X) / (Z_Sol * (Z_Sol + Bsq));

      pv.vel(Y) = (gup(X, Y) * cv.mom(X) + gup(Y, Y) * cv.mom(Y) +
                   gup(Y, Z) * cv.mom(Z)) /
                  (Z_Sol + Bsq);
      pv.vel(Y) += BiSi * cv.dBvec(Y) / (Z_Sol * (Z_Sol + Bsq));

      pv.vel(Z) = (gup(X, Z) * cv.mom(X) + gup(Y, Z) * cv.mom(Y) +
                   gup(Z, Z) * cv.mom(Z)) /
                  (Z_Sol + Bsq);
      pv.vel(Z) += BiSi * cv.dBvec(Z) / (Z_Sol * (Z_Sol + Bsq));

      pv.w_lor = W_Sol;
    }

    pv.Ye = cv.DYe / cv.dens;
    pv.temperature = temp_Sol;
    pv.eps =
        eos_3p->eps_from_rho_temp_ye_pref(pv.rho, pv.temperature, pv.Ye,
                                           rad_pref);
    pv.press = eos_3p->press_from_rho_temp_ye_pref(pv.rho, pv.temperature,
                                                   pv.Ye, rad_pref);
    pv.entropy =
        eos_3p->kappa_from_rho_temp_ye_tau(pv.rho, pv.temperature, pv.Ye,
                                           tau_opt);
    pv.Bvec = cv.dBvec;

    const vec<CCTK_REAL, 3> Elow = calc_cross_product(pv.Bvec, pv.vel);
    pv.E = calc_contraction(gup, Elow);
  }
};

template <typename EOSType>
CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
c2p_Noble_rad::prims_floors_and_ceilings_tau(
    const EOSType *eos_3p, prim_vars &pv, const cons_vars &cv,
    const CCTK_REAL alp, const vec<CCTK_REAL, 3> &beta,
    const smat<CCTK_REAL, 3> &glo, const CCTK_REAL tau_opt,
    const CCTK_REAL rad_pref, c2p_report &rep) const {

  bool recomp_eps_press_entropy = false;

  const CCTK_REAL rho_h_fluid_old = pv.rho + pv.rho * pv.eps + pv.press;

  if (pv.Ye < eos_3p->rgye.min) {
    pv.Ye = eos_3p->rgye.min;
    rep.adjust_cons = true;
    recomp_eps_press_entropy = true;
  }

  if (pv.Ye > eos_3p->rgye.max) {
    pv.Ye = eos_3p->rgye.max;
    rep.adjust_cons = true;
    recomp_eps_press_entropy = true;
  }

  vec<CCTK_REAL, 3> v_low = calc_contraction(glo, pv.vel);
  CCTK_REAL vsq_Sol = calc_contraction(v_low, pv.vel);
  CCTK_REAL sol_v = sqrt(vsq_Sol);

  if (sol_v > v_lim) {
    pv.rho = cv.dens / w_lim;
    pv.vel *= v_lim / sol_v;
    pv.w_lor = w_lim;
    rep.adjust_cons = true;

    if (use_temp) {
      recomp_eps_press_entropy = true;
    } else {
      recomp_eps_press_entropy = false;
      pv.eps =
          eos_3p->eps_from_rho_press_ye_pref(pv.rho, pv.press, pv.Ye,
                                              rad_pref);
      pv.temperature =
          eos_3p->temp_from_rho_eps_ye_pref(pv.rho, pv.eps, pv.Ye, rad_pref);
      pv.entropy =
          eos_3p->kappa_from_rho_temp_ye_tau(pv.rho, pv.temperature, pv.Ye,
                                             tau_opt);
    }
  }

  if (pv.rho > eos_3p->rgrho.max) {
    pv.rho = eos_3p->rgrho.max;
    rep.adjust_cons = true;

    if (use_temp) {
      recomp_eps_press_entropy = true;
    } else {
      recomp_eps_press_entropy = false;
      pv.eps =
          eos_3p->eps_from_rho_press_ye_pref(pv.rho, pv.press, pv.Ye,
                                              rad_pref);
      pv.temperature =
          eos_3p->temp_from_rho_eps_ye_pref(pv.rho, pv.eps, pv.Ye, rad_pref);
      pv.entropy =
          eos_3p->kappa_from_rho_temp_ye_tau(pv.rho, pv.temperature, pv.Ye,
                                             tau_opt);
    }
  }

  if (pv.temperature > eos_3p->rgtemp.max) {
    pv.temperature = eos_3p->rgtemp.max;
    recomp_eps_press_entropy = true;
    rep.adjust_cons = true;
  }

  if (use_press_atmo) {
    if (pv.press < atmo.press_atmo) {
      pv.press = atmo.press_atmo;
      pv.eps =
          eos_3p->eps_from_rho_press_ye_pref(pv.rho, pv.press, pv.Ye,
                                              rad_pref);
      pv.temperature =
          eos_3p->temp_from_rho_eps_ye_pref(pv.rho, pv.eps, pv.Ye, rad_pref);
      pv.entropy =
          eos_3p->kappa_from_rho_temp_ye_tau(pv.rho, pv.temperature, pv.Ye,
                                             tau_opt);
      recomp_eps_press_entropy = false;
      rep.adjust_cons = true;
    }
  } else {
    if (pv.temperature < atmo.temp_atmo) {
      pv.temperature = atmo.temp_atmo;
      recomp_eps_press_entropy = true;
      rep.adjust_cons = true;
    }
  }

  if (recomp_eps_press_entropy) {
    pv.eps =
        eos_3p->eps_from_rho_temp_ye_pref(pv.rho, pv.temperature, pv.Ye,
                                          rad_pref);
    pv.press =
        eos_3p->press_from_rho_temp_ye_pref(pv.rho, pv.temperature, pv.Ye,
                                            rad_pref);
    pv.entropy =
        eos_3p->kappa_from_rho_temp_ye_tau(pv.rho, pv.temperature, pv.Ye,
                                           tau_opt);
    recomp_eps_press_entropy = false;
  }

  v_low = calc_contraction(glo, pv.vel);
  const vec<CCTK_REAL, 3> B_low = calc_contraction(glo, pv.Bvec);

  const CCTK_REAL Bdotv = calc_contraction(pv.Bvec, v_low);
  const CCTK_REAL alp_b0 = pv.w_lor * Bdotv;

  const CCTK_REAL B2 = calc_contraction(pv.Bvec, B_low);
  const CCTK_REAL bsq = (B2 + alp_b0 * alp_b0) / (pv.w_lor * pv.w_lor);

  bool mag_ceiling = false;

  if (bsq > sigma_max * pv.rho) {
    pv.rho = bsq / sigma_max;
    mag_ceiling = true;
  }

  if (bsq > CCTK_REAL(2.0) * inv_beta_max * pv.press) {
    pv.press = CCTK_REAL(0.5) * bsq / inv_beta_max;
    mag_ceiling = true;
  }

  if (mag_ceiling) {

    rep.adjust_cons = true;

    if (use_temp) {
      pv.temperature =
          eos_3p->temp_from_rho_press_ye_pref(pv.rho, pv.press, pv.Ye,
                                              rad_pref);
      pv.eps =
          eos_3p->eps_from_rho_temp_ye_pref(pv.rho, pv.temperature, pv.Ye,
                                            rad_pref);
      pv.entropy =
          eos_3p->kappa_from_rho_temp_ye_tau(pv.rho, pv.temperature, pv.Ye,
                                             tau_opt);
    } else {
      pv.eps =
          eos_3p->eps_from_rho_press_ye_pref(pv.rho, pv.press, pv.Ye,
                                              rad_pref);
      pv.temperature =
          eos_3p->temp_from_rho_eps_ye_pref(pv.rho, pv.eps, pv.Ye, rad_pref);
      pv.entropy =
          eos_3p->kappa_from_rho_temp_ye_tau(pv.rho, pv.temperature, pv.Ye,
                                             tau_opt);
    }

    const CCTK_REAL B = fmax(sqrt(B2), CCTK_REAL(1.0e-64));

    const CCTK_REAL ut = pv.w_lor / alp;

    const CCTK_REAL u1 = pv.w_lor * (pv.vel(0) - beta(0) / alp);
    const CCTK_REAL u2 = pv.w_lor * (pv.vel(1) - beta(1) / alp);
    const CCTK_REAL u3 = pv.w_lor * (pv.vel(2) - beta(2) / alp);

    const CCTK_REAL v_par_old = pv.w_lor * Bdotv / B / ut;

    const CCTK_REAL ut_perp =
        CCTK_REAL(1.0) /
        sqrt(CCTK_REAL(1.0) / (ut * ut) + v_par_old * v_par_old);

    const CCTK_REAL u1_perp =
        ut_perp * (u1 / ut - v_par_old * pv.Bvec(0) / B);
    const CCTK_REAL u2_perp =
        ut_perp * (u2 / ut - v_par_old * pv.Bvec(1) / B);
    const CCTK_REAL u3_perp =
        ut_perp * (u3 / ut - v_par_old * pv.Bvec(2) / B);

    const CCTK_REAL BdotQ = pv.w_lor * rho_h_fluid_old * Bdotv * ut;

    const CCTK_REAL rho_h_fluid_new = pv.rho + pv.rho * pv.eps + pv.press;

    const CCTK_REAL xx =
        CCTK_REAL(2.0) * BdotQ / (B * rho_h_fluid_new * ut_perp);

    const CCTK_REAL v_par_new =
        xx / (CCTK_REAL(1.0) + sqrt(CCTK_REAL(1.0) + xx * xx)) / ut_perp;

    const CCTK_REAL v1_new = v_par_new * pv.Bvec(0) / B + u1_perp / ut_perp;
    const CCTK_REAL v2_new = v_par_new * pv.Bvec(1) / B + u2_perp / ut_perp;
    const CCTK_REAL v3_new = v_par_new * pv.Bvec(2) / B + u3_perp / ut_perp;

    pv.vel(0) = (v1_new + beta(0)) / alp;
    pv.vel(1) = (v2_new + beta(1)) / alp;
    pv.vel(2) = (v3_new + beta(2)) / alp;

    v_low = calc_contraction(glo, pv.vel);
    vsq_Sol = calc_contraction(v_low, pv.vel);
    sol_v = sqrt(vsq_Sol);

    if (sol_v > v_lim) {
      pv.vel *= v_lim / sol_v;
      pv.w_lor = w_lim;
    } else {
      pv.w_lor = CCTK_REAL(1.0) / sqrt(CCTK_REAL(1.0) - vsq_Sol);
    }
  }
}

template <typename EOSType, bool limiting>
CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
c2p_Noble_rad::bh_interior_tau(const EOSType *eos_3p, prim_vars &pv,
                               cons_vars &cv,
                               const smat<CCTK_REAL, 3> &glo,
                               const CCTK_REAL tau_opt) const {
  const CCTK_REAL wlim_BH = sqrt(CCTK_REAL(1.0) + vwlim_BH * vwlim_BH);
  const CCTK_REAL vlim_BH = vwlim_BH / wlim_BH;

  bool recomp_flag = false;

  if constexpr (limiting) {

    if (pv.rho > rho_BH) {
      pv.rho = rho_BH;
      recomp_flag = true;
    };

    if (pv.eps > eps_BH) {
      pv.eps = eps_BH;
      recomp_flag = true;
    };

    const CCTK_REAL sol_v = sqrt((pv.w_lor * pv.w_lor - CCTK_REAL(1.0))) /
                            pv.w_lor;
    if (sol_v > vlim_BH) {
      pv.vel *= vlim_BH / sol_v;
      pv.w_lor = wlim_BH;
      recomp_flag = true;
    };

    if (recomp_flag) {
      pv.temperature =
          eos_3p->temp_from_rho_eps_ye_tau(pv.rho, pv.eps, pv.Ye, tau_opt);
      pv.press =
          eos_3p->press_from_rho_eps_ye_tau(pv.rho, pv.eps, pv.Ye, tau_opt);
      pv.entropy =
          eos_3p->kappa_from_rho_eps_ye_tau(pv.rho, pv.eps, pv.Ye, tau_opt);

      cv.from_prim(pv, glo);
    };

  } else {

    pv.rho = rho_BH;
    pv.eps = eps_BH;
    pv.Ye = atmo.ye_atmo;

    pv.temperature =
        eos_3p->temp_from_rho_eps_ye_tau(pv.rho, pv.eps, pv.Ye, tau_opt);
    pv.press =
        eos_3p->press_from_rho_eps_ye_tau(pv.rho, pv.eps, pv.Ye, tau_opt);
    pv.entropy =
        eos_3p->kappa_from_rho_eps_ye_tau(pv.rho, pv.eps, pv.Ye, tau_opt);

    const CCTK_REAL spatial_detg = calc_det(glo);
    const smat<CCTK_REAL, 3> gup = calc_inv(glo, spatial_detg);

    const CCTK_REAL Z_loc =
        (pv.rho * (CCTK_REAL(1.0) + pv.eps) + pv.press) * wlim_BH * wlim_BH;

    const vec<CCTK_REAL, 3> B_low = calc_contraction(glo, pv.Bvec);
    const CCTK_REAL Bsq = calc_contraction(pv.Bvec, B_low);

    vec<CCTK_REAL, 3> mom_low = cv.mom / sqrt(spatial_detg);
    vec<CCTK_REAL, 3> mom_up = calc_contraction(gup, mom_low);
    const CCTK_REAL Ssq_old = calc_contraction(mom_low, mom_up);
    const CCTK_REAL S_old = sqrt(Ssq_old) + CCTK_REAL(1.0e-50);

    const CCTK_REAL BiSi_old = calc_contraction(mom_low, pv.Bvec);
    const CCTK_REAL BiEsi = BiSi_old / S_old;

    const CCTK_REAL Ssq_new =
        ((Z_loc + Bsq) * (Z_loc + Bsq) * vlim_BH * vlim_BH) /
        (CCTK_REAL(1.0) +
         BiEsi * BiEsi * (CCTK_REAL(2.0) * Z_loc + Bsq) / (Z_loc * Z_loc));
    const CCTK_REAL S_new = sqrt(Ssq_new);

    pv.vel(X) = mom_up(X) * S_new / S_old / (Z_loc + Bsq);
    pv.vel(X) += BiEsi * S_new * pv.Bvec(X) / (Z_loc * (Z_loc + Bsq));

    pv.vel(Y) = mom_up(Y) * S_new / S_old / (Z_loc + Bsq);
    pv.vel(Y) += BiEsi * S_new * pv.Bvec(Y) / (Z_loc * (Z_loc + Bsq));

    pv.vel(Z) = mom_up(Z) * S_new / S_old / (Z_loc + Bsq);
    pv.vel(Z) += BiEsi * S_new * pv.Bvec(Z) / (Z_loc * (Z_loc + Bsq));

    pv.w_lor = wlim_BH;

    cv.from_prim(pv, glo);
  };
}

template <typename EOSType>
CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
c2p_Noble_rad::solve(const EOSType *eos_3p, prim_vars &pv,
                     prim_vars &pv_seeds, cons_vars &cv, const CCTK_REAL alp,
                     const vec<CCTK_REAL, 3> &beta,
                     const smat<CCTK_REAL, 3> &glo,
                     const CCTK_REAL tau_opt, c2p_report &rep) const {
  rep.iters = 0;
  rep.adjust_cons = false;
  rep.set_atmo = false;
  rep.soft_root_conv = false;
  rep.status = c2p_report::SUCCESS;

  const CCTK_REAL spatial_detg = calc_det(glo);
  const CCTK_REAL sqrt_detg = sqrt(spatial_detg);

  const bool minor1{glo(X, X) > CCTK_REAL(0.0)};
  const bool minor2{glo(X, X) * glo(Y, Y) - glo(X, Y) * glo(X, Y) >
                    CCTK_REAL(0.0)};
  const bool minor3{spatial_detg > CCTK_REAL(0.0)};
  if (!(minor1 && minor2 && minor3)) {
    rep.set_invalid_detg(sqrt_detg);
    set_to_nan(pv, cv);
    return;
  }

  const smat<CCTK_REAL, 3> gup = calc_inv(glo, spatial_detg);
  const cons_vars cv_const = cv;

  cv.dens /= sqrt_detg;
  cv.tau /= sqrt_detg;
  cv.mom /= sqrt_detg;
  cv.dBvec /= sqrt_detg;
  cv.DYe /= sqrt_detg;
  cv.DEnt /= sqrt_detg;

  pv_seeds.Bvec = cv.dBvec;

  const CCTK_REAL Ssq = get_Ssq_Exact(cv.mom, gup);
  const CCTK_REAL Bsq = get_Bsq_Exact(pv_seeds.Bvec, glo);
  const CCTK_REAL BiSi = get_BiSi_Exact(pv_seeds.Bvec, cv.mom);
  const CCTK_REAL rad_pref = eos_3p->rad_prefactor(tau_opt);

  vec<CCTK_REAL, 3> w_vsq_bsq;
  if (use_zprim) {
    const vec<CCTK_REAL, 3> zvec = pv_seeds.vel * pv_seeds.w_lor;
    w_vsq_bsq = getZ_WLorentz_vsq_bsq_Seeds(pv_seeds.Bvec, zvec, glo);
  } else {
    w_vsq_bsq = get_WLorentz_vsq_bsq_Seeds(pv_seeds.Bvec, pv_seeds.vel, glo);
  }
  pv_seeds.w_lor = w_vsq_bsq(0);
  CCTK_REAL vsq_seed = w_vsq_bsq(1);

  if ((!isfinite(cv.dens)) || (!isfinite(Ssq)) || (!isfinite(Bsq)) ||
      (!isfinite(BiSi)) || (!isfinite(cv.DYe)) || (!isfinite(cv.DEnt))) {
    rep.set_nans_in_cons(cv.dens, Ssq, Bsq, BiSi, cv.DYe);
    set_to_nan(pv, cv);
    return;
  }

  if (Bsq > Bsq_lim) {
    rep.set_B_limit(Bsq);
    set_to_nan(pv, cv);
    return;
  }

  pv_seeds.rho = cv.dens / pv_seeds.w_lor;
  CCTK_REAL temp_seed = pv_seeds.temperature;
  if ((!isfinite(temp_seed)) || temp_seed <= CCTK_REAL(0.0)) {
    temp_seed = fmax(eos_3p->gm1 * fmax(pv_seeds.eps, atmo.eps_atmo),
                     eos_3p->rgtemp.min);
  }
  pv_seeds.press = eos_3p->press_from_rho_temp_ye_pref(
      pv_seeds.rho, temp_seed, pv_seeds.Ye, rad_pref);

  const CCTK_REAL Z_Seed =
      get_Z_Seed(pv_seeds.rho,
                 eos_3p->eps_from_rho_temp_ye_pref(pv_seeds.rho, temp_seed,
                                                    pv_seeds.Ye, rad_pref),
                 pv_seeds.press, pv_seeds.w_lor);

  constexpr CCTK_REAL dv = CCTK_REAL(1.0) - CCTK_REAL(1.0e-10);
  CCTK_REAL x[3] = {fmax(fabs(Z_Seed), Zmin), fmin(fmax(vsq_seed, CCTK_REAL(0.0)), dv),
                    fmax(temp_seed, CCTK_REAL(1.0e-300))};
  CCTK_REAL errx = CCTK_REAL(1.0);
  CCTK_REAL f = CCTK_REAL(0.0);
  CCTK_REAL df = CCTK_REAL(0.0);

  CCTK_INT k = 1;
  for (; k <= maxIterations; ++k) {
    CCTK_REAL r[3];
    residual(eos_3p, cv, Ssq, Bsq, BiSi, rad_pref, x, r);
    f = CCTK_REAL(0.5) * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    df = -CCTK_REAL(2.0) * f;

    CCTK_REAL J[3][3];
    jacobian(eos_3p, cv, Bsq, BiSi, rad_pref, x, J);

    CCTK_REAL rhs[3] = {-r[0], -r[1], -r[2]};
    CCTK_REAL dx[3] = {CCTK_REAL(0.0), CCTK_REAL(0.0), CCTK_REAL(0.0)};
    if (!solve_linear3(J, rhs, dx)) {
      rep.set_root_bracket();
      cv = cv_const;
      return;
    }

    errx = CCTK_REAL(0.0);
    for (int i = 0; i < 3; ++i)
      errx = fmax(errx, fabs(dx[i]) / fmax(fabs(x[i]), CCTK_REAL(1.0)));

    x[0] += dx[0];
    x[1] += dx[1];
    x[2] += dx[2];

    x[0] = fmax(x[0], Zmin);
    x[1] = fmin(fmax(x[1], CCTK_REAL(0.0)), dv);
    x[2] = fmax(x[2], CCTK_REAL(1.0e-300));

    if (errx <= tolerance)
      break;
  }

  rep.iters = k;
  if (errx > tolerance) {
    bool accept_soft = false;
    if (soft_root_convergence && isfinite(errx)) {
      accept_soft = errx <= soft_root_width_factor * tolerance;
    }
    if (!accept_soft) {
      rep.set_root_conv();
      cv = cv_const;
      return;
    }
    rep.set_soft_root_conv();
  }

  if ((!isfinite(f)) || (!isfinite(df))) {
    rep.set_root_bracket();
    cv = cv_const;
    return;
  }

  WZT2Prim(x[0], x[1], x[2], Bsq, BiSi, eos_3p, pv, cv, gup, glo, rad_pref);

  if (pv.rho <= CCTK_REAL(0.0)) {
    rep.set_range_rho(cv.dens, pv.rho);
    cv = cv_const;
    return;
  }

  if (pv.temperature <= CCTK_REAL(0.0) || pv.eps <= CCTK_REAL(0.0)) {
    rep.set_range_eps(pv.eps);
    cv = cv_const;
    return;
  }

  if (pv.rho < atmo.rho_cut) {
    rep.set_atmo_set();
    atmo.set(pv, cv, glo);
    return;
  }

  prims_floors_and_ceilings_tau(eos_3p, pv, cv, alp, beta, glo, tau_opt,
                                rad_pref, rep);

  if (rep.adjust_cons) {
    cv.from_prim(pv, glo);
    cv.dBvec = cv_const.dBvec;
  } else {
    cv = cv_const;
    cv.DEnt = cv.dens * pv.entropy;
  }
}

} // namespace Con2PrimFactory

#endif
