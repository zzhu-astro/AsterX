#ifndef EOS_3P_RAD_IDEALGAS_HXX
#define EOS_3P_RAD_IDEALGAS_HXX

#include <algorithm>
#include <cmath>

#include "../eos_3p.hxx"

namespace EOSX {

class eos_3p_rad_idealgas : public eos_3p {
public:
  CCTK_REAL gamma, gm1, inv_gamma, temp_over_eps;
  CCTK_REAL arad_code, rad_factor;
  range rgeps;

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
  init(CCTK_REAL gamma_, CCTK_REAL arad_code_, CCTK_REAL rad_factor_,
       range &rgeps_, const range &rgrho_, const range &rgye_) {
    gamma = gamma_;
    gm1 = gamma_ - 1.0;
    inv_gamma = 1.0 / gamma_;
    temp_over_eps = gm1;
    arad_code = arad_code_;
    rad_factor = rad_factor_;
    rgeps = rgeps_;
    if (gamma <= 1.0) {
      printf("EOS_RadIdealGas: initialized with gamma <= 1.\n");
      assert(false);
    }
    set_range_rho(rgrho_);
    set_range_ye(rgye_);
    set_range_temp(range(temp_over_eps * rgeps.min, temp_over_eps * rgeps.max));
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  rad_prefactor(const CCTK_REAL tau_opt) const {
    const CCTK_REAL tau_pos = fmax(CCTK_REAL(0.0), tau_opt);
    return rad_factor * tau_pos / (tau_pos + CCTK_REAL(1.0)) * arad_code;
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  rad_energy_density(const CCTK_REAL temp, const CCTK_REAL tau_opt) const {
    const CCTK_REAL t2 = temp * temp;
    return rad_prefactor(tau_opt) * t2 * t2;
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  press_from_rho_temp_ye_tau(const CCTK_REAL rho, const CCTK_REAL temp,
                             const CCTK_REAL ye,
                             const CCTK_REAL tau_opt) const {
    return rho * temp + rad_energy_density(temp, tau_opt) / CCTK_REAL(3.0);
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  eps_from_rho_temp_ye_tau(const CCTK_REAL rho, const CCTK_REAL temp,
                           const CCTK_REAL ye,
                           const CCTK_REAL tau_opt) const {
    return temp / gm1 + rad_energy_density(temp, tau_opt) / rho;
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  enthalpy_from_rho_temp_ye_tau(const CCTK_REAL rho, const CCTK_REAL temp,
                                const CCTK_REAL ye,
                                const CCTK_REAL tau_opt) const {
    const CCTK_REAL erad = rad_energy_density(temp, tau_opt);
    return CCTK_REAL(1.0) + temp / gm1 + temp +
           CCTK_REAL(4.0) * erad / (CCTK_REAL(3.0) * rho);
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  csnd_from_rho_temp_ye_tau(const CCTK_REAL rho, const CCTK_REAL temp,
                            const CCTK_REAL ye,
                            const CCTK_REAL tau_opt) const {
    const CCTK_REAL temp_pos = fmax(temp, CCTK_REAL(1.0e-300));
    const CCTK_REAL pref = rad_prefactor(tau_opt);
    const CCTK_REAL t2 = temp_pos * temp_pos;
    const CCTK_REAL t3 = t2 * temp_pos;
    const CCTK_REAL h = enthalpy_from_rho_temp_ye_tau(rho, temp_pos, ye, tau_opt);
    const CCTK_REAL dPdt = rho + CCTK_REAL(4.0) * pref * t3 / CCTK_REAL(3.0);
    const CCTK_REAL dtdrho =
        (CCTK_REAL(1.0) +
         CCTK_REAL(4.0) * pref * t3 / (CCTK_REAL(3.0) * rho)) /
        (CCTK_REAL(4.0) * pref * t2 + rho / (gm1 * temp_pos));
    const CCTK_REAL dpdrho = temp_pos + dPdt * dtdrho;
    const CCTK_REAL cs2 = dpdrho / h;
    return sqrt(fmin(fmax(cs2, CCTK_REAL(0.0)), CCTK_REAL(0.999999)));
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  entropy_from_rho_temp_ye_tau(const CCTK_REAL rho, const CCTK_REAL temp,
                               const CCTK_REAL ye,
                               const CCTK_REAL tau_opt) const {
    const CCTK_REAL eps_gas = temp / gm1;
    const CCTK_REAL pref = rad_prefactor(tau_opt);
    return log(eps_gas * pow(rho, -gm1)) +
           gm1 * CCTK_REAL(4.0) * pref * temp * temp * temp /
               (CCTK_REAL(3.0) * rho);
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  temp_from_rho_eps_ye_tau(const CCTK_REAL rho, CCTK_REAL &eps,
                           const CCTK_REAL ye,
                           const CCTK_REAL tau_opt) const {
    const CCTK_REAL pref = rad_prefactor(tau_opt);
    const CCTK_REAL temp_gas = gm1 * eps;
    const CCTK_REAL temp_rad = pow(eps * rho / pref, CCTK_REAL(0.25));
    CCTK_REAL temp = fmin(temp_gas, temp_rad);

    for (int n = 0; n < 20; ++n) {
      const CCTK_REAL t2 = temp * temp;
      const CCTK_REAL t3 = t2 * temp;
      const CCTK_REAL t4 = t2 * t2;
      const CCTK_REAL f = temp / gm1 + pref * t4 / rho - eps;
      const CCTK_REAL df = CCTK_REAL(1.0) / gm1 +
                           CCTK_REAL(4.0) * pref * t3 / rho;
      const CCTK_REAL dtemp = f / df;
      temp = fmax(temp - dtemp, CCTK_REAL(0.0));
      if (fabs(dtemp) <= CCTK_REAL(1.0e-12) * fmax(temp, CCTK_REAL(1.0)))
        break;
    }
    return temp;
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  temp_from_rho_press_ye_tau(const CCTK_REAL rho, CCTK_REAL &press,
                             const CCTK_REAL ye,
                             const CCTK_REAL tau_opt) const {
    const CCTK_REAL pref = rad_prefactor(tau_opt);
    const CCTK_REAL temp_gas = press / fmax(rho, CCTK_REAL(1.0e-300));
    const CCTK_REAL temp_rad = pow(CCTK_REAL(3.0) * press / pref, CCTK_REAL(0.25));
    CCTK_REAL temp = fmin(temp_gas, temp_rad);
    for (int n = 0; n < 20; ++n) {
      const CCTK_REAL t2 = temp * temp;
      const CCTK_REAL t3 = t2 * temp;
      const CCTK_REAL t4 = t2 * t2;
      const CCTK_REAL f = rho * temp + pref * t4 / CCTK_REAL(3.0) - press;
      const CCTK_REAL df =
          rho + CCTK_REAL(4.0) * pref * t3 / CCTK_REAL(3.0);
      const CCTK_REAL dtemp = f / df;
      temp = fmax(temp - dtemp, CCTK_REAL(0.0));
      if (fabs(dtemp) <= CCTK_REAL(1.0e-12) * fmax(temp, CCTK_REAL(1.0)))
        break;
    }
    return temp;
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  eps_from_rho_press_ye_tau(const CCTK_REAL rho, const CCTK_REAL press,
                            const CCTK_REAL ye,
                            const CCTK_REAL tau_opt) const {
    CCTK_REAL press_tmp = press;
    const CCTK_REAL temp =
        temp_from_rho_press_ye_tau(rho, press_tmp, ye, tau_opt);
    return eps_from_rho_temp_ye_tau(rho, temp, ye, tau_opt);
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  press_from_rho_eps_ye_tau(const CCTK_REAL rho, CCTK_REAL &eps,
                            const CCTK_REAL ye,
                            const CCTK_REAL tau_opt) const {
    const CCTK_REAL temp = temp_from_rho_eps_ye_tau(rho, eps, ye, tau_opt);
    return press_from_rho_temp_ye_tau(rho, temp, ye, tau_opt);
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  kappa_from_rho_eps_ye_tau(const CCTK_REAL rho, CCTK_REAL &eps,
                            const CCTK_REAL ye,
                            const CCTK_REAL tau_opt) const {
    const CCTK_REAL temp = temp_from_rho_eps_ye_tau(rho, eps, ye, tau_opt);
    return kappa_from_rho_temp_ye_tau(rho, temp, ye, tau_opt);
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  press_from_rho_eps_ye(const CCTK_REAL rho, CCTK_REAL &eps,
                        const CCTK_REAL ye) const {
    return press_from_rho_eps_ye_tau(rho, eps, ye, CCTK_REAL(0.0));
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  eps_from_rho_press_ye(const CCTK_REAL rho, const CCTK_REAL press,
                        const CCTK_REAL ye) const {
    return eps_from_rho_press_ye_tau(rho, press, ye, CCTK_REAL(0.0));
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  csnd_from_rho_eps_ye(const CCTK_REAL rho, CCTK_REAL &eps,
                       const CCTK_REAL ye) const {
    const CCTK_REAL temp = temp_from_rho_eps_ye(rho, eps, ye);
    return csnd_from_rho_temp_ye(rho, temp, ye);
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  csnd_from_rho_temp_ye(const CCTK_REAL rho, const CCTK_REAL temp,
                        const CCTK_REAL ye) const {
    return csnd_from_rho_temp_ye_tau(rho, temp, ye, CCTK_REAL(0.0));
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  temp_from_rho_eps_ye(const CCTK_REAL rho, CCTK_REAL &eps,
                       const CCTK_REAL ye) const {
    return temp_from_rho_eps_ye_tau(rho, eps, ye, CCTK_REAL(0.0));
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  temp_from_rho_press_ye(const CCTK_REAL rho, CCTK_REAL &press,
                         const CCTK_REAL ye) const {
    return temp_from_rho_press_ye_tau(rho, press, ye, CCTK_REAL(0.0));
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  eps_from_rho_temp_ye(const CCTK_REAL rho, const CCTK_REAL temp,
                       const CCTK_REAL ye) const {
    return eps_from_rho_temp_ye_tau(rho, temp, ye, CCTK_REAL(0.0));
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  press_from_rho_temp_ye(const CCTK_REAL rho, const CCTK_REAL temp,
                         const CCTK_REAL ye) const {
    return press_from_rho_temp_ye_tau(rho, temp, ye, CCTK_REAL(0.0));
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  entropy_from_rho_temp_ye(const CCTK_REAL rho, const CCTK_REAL temp,
                           const CCTK_REAL ye) const {
    return entropy_from_rho_temp_ye_tau(rho, temp, ye, CCTK_REAL(0.0));
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  entropy_from_rho_eps_ye(const CCTK_REAL rho, const CCTK_REAL eps,
                          const CCTK_REAL ye) const {
    CCTK_REAL eps_tmp = eps;
    const CCTK_REAL temp =
        temp_from_rho_eps_ye_tau(rho, eps_tmp, ye, CCTK_REAL(0.0));
    return entropy_from_rho_temp_ye_tau(rho, temp, ye, CCTK_REAL(0.0));
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  press_from_rho_kappa_ye(const CCTK_REAL rho, const CCTK_REAL kappa,
                          const CCTK_REAL ye) const {
    return kappa * pow(rho, gamma);
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  eps_from_rho_kappa_ye(const CCTK_REAL rho, const CCTK_REAL kappa,
                        const CCTK_REAL ye) const {
    return kappa * pow(rho, gamma - CCTK_REAL(1.0)) / gm1;
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  kappa_from_rho_temp_ye_tau(const CCTK_REAL rho, const CCTK_REAL temp,
                             const CCTK_REAL ye,
                             const CCTK_REAL tau_opt) const {
    return temp * pow(rho, CCTK_REAL(1.0) - gamma);
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline CCTK_REAL
  kappa_from_rho_eps_ye(const CCTK_REAL rho, CCTK_REAL &eps,
                        const CCTK_REAL ye) const {
    return kappa_from_rho_eps_ye_tau(rho, eps, ye, CCTK_REAL(0.0));
  }

  CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline range
  range_eps_from_rho_ye(const CCTK_REAL rho, const CCTK_REAL ye) const {
    return rgeps;
  }

  CCTK_HOST CCTK_DEVICE inline CCTK_REAL
  mu_lepton_from_rho_temp_ye(const CCTK_REAL rho, const CCTK_REAL temp,
                             const CCTK_REAL ye) const {
    return CCTK_REAL(0.0);
  }
};

} // namespace EOSX

#endif
