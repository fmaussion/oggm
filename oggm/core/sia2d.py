import numpy as np
from numpy import ix_

from oggm import cfg, utils
from oggm.core.distributed import Model2D, filter_ice_border
from oggm.cfg import G, SEC_IN_DAY


class Upstream2D(Model2D):
    """Distributed shallow ice model.

    Very slow.
    """

    def __init__(self, bed_topo, init_ice_thick=None, dx=None,
                 mb_model=None, y0=0., glen_a=None, mb_elev_feedback='annual',
                 cfl=0.124, max_dt=31 * SEC_IN_DAY,
                 ice_thick_filter=filter_ice_border, mb_filter=None):
        """Create a new 2D model from gridded data.

        Parameters
        ----------
        bed_topo : 2d array
            the topography
        init_ice_thick : 2d array (optional)
            the initial ice thickness (default is zero everywhere)
        dx : float
            map resolution (m)
        dy : float
            map resolution (m)
        mb_model : oggm.core.massbalance model
            the mass balance model to use for the simulation
        y0 : int
            the starting year
        glen_a : float
            Glen's flow law parameter A
        mb_elev_feedback : str (default: 'annual')
            when to update the mass balance model ('annual', 'monthly', or
            'always')
        cfl : float (default:0.124)
            forward time stepping stability criteria. Default is just beyond
            R. Hindmarsh's idea of 1/2(n+1).
        max_dt : int (default: 31 days)
            maximum allow time step (in seconds). Useful because otherwise the
            automatic time step can be quite ambitious.
        ice_thick_filter : func
            function to apply to the ice thickness *after* each time step.
            See filter_ice_border for an example. Set to None for doing nothing
        mb_filter : ndarray
            2d array indicating the mask where positive mb is allowed
            (useful to allow only specific glaciers to grow)
        """
        super(Upstream2D, self).__init__(bed_topo,
                                         init_ice_thick=init_ice_thick,
                                         dx=dx, mb_model=mb_model, y0=y0,
                                         glen_a=glen_a,
                                         mb_elev_feedback=mb_elev_feedback,
                                         ice_thick_filter=ice_thick_filter,
                                         mb_filter=mb_filter)

        # We introduce Gamma to shorten the equations
        self.rho = cfg.PARAMS['ice_density']
        self.glen_n = cfg.PARAMS['glen_n']
        self.gamma = (2. * self.glen_a * (self.rho * G) ** self.glen_n
                      / (self.glen_n + 2))

        # forward time stepping stability criteria
        # default is just beyond R. Hindmarsh's idea of 1/2(n+1)
        self.cfl = cfl
        self.max_dt = max_dt

        # extend into 2D
        self.Lx = 0.5 * (self.nx - 1) * self.dx
        self.Ly = 0.5 * (self.ny - 1) * self.dy

        # Some indices
        self.k = np.arange(0, self.ny)
        self.kp = np.hstack([np.arange(1, self.ny), self.ny - 1])
        self.km = np.hstack([0, np.arange(0, self.ny - 1)])
        self.l = np.arange(0, self.nx)  # flake8: noqa E741
        self.lp = np.hstack([np.arange(1, self.nx), self.nx - 1])
        self.lm = np.hstack([0, np.arange(0, self.nx - 1)])
        self.H_upstream_up = np.zeros((self.ny, self.nx))
        self.H_upstream_dn = np.zeros((self.ny, self.nx))

        # Easy optimisation
        self._ixklp = ix_(self.k, self.lp)
        self._ixkl = ix_(self.k, self.l)
        self._ixklm = ix_(self.k, self.lm)
        self._ixkpl = ix_(self.kp, self.l)
        self._ixkml = ix_(self.km, self.l)
        self._ixkplp = ix_(self.kp, self.lp)
        self._ixkplm = ix_(self.kp, self.lm)
        self._ixkmlm = ix_(self.km, self.lm)
        self._ixkmlp = ix_(self.km, self.lp)

    def diffusion_upstream_2d(self):
        # Built upon the Eq. (62) with the term in y in the diffusivity.
        # It differs from diffusion_Upstream_2D_V1 only for the definition of
        # "s_grad", l282-283 & l305-306 in V1 and l355-356 & l379-380 in V2)

        H = self.ice_thick
        S = self.surface_h
        N = self.glen_n

        # Optim
        S_ixklp = S[self._ixklp]
        S_ixkl = S[self._ixkl]
        S_ixklm = S[self._ixklm]
        S_ixkml = S[self._ixkml]
        S_ixkpl = S[self._ixkpl]
        S_ixkplp = S[self._ixkplp]
        S_ixkplm = S[self._ixkplm]
        S_ixkmlm = S[self._ixkmlm]
        S_ixkmlp = S[self._ixkmlp]

        Hl = H[self._ixkl]
        Hlp = H[self._ixklp]
        Hlm = H[self._ixklm]
        Hk = Hl
        Hkp = H[self._ixkpl]
        Hkm = H[self._ixkml]

        # --- all the l components

        # applying Eq. (61) to the scheme
        H_l_up = 0.5 * (Hlp + Hl)
        H_l_dn = 0.5 * (Hl + Hlm)

        H_l_upstream_up = self.H_upstream_up
        gt = S_ixklp > S_ixkl
        H_l_upstream_up[gt] = Hlp[gt]
        H_l_upstream_up[~gt] = Hl[~gt]

        H_l_upstream_dn = self.H_upstream_dn
        gt = S_ixkl > S_ixklm
        H_l_upstream_dn[gt] = Hl[gt]
        H_l_upstream_dn[~gt] = Hlm[~gt]

        # applying Eq. (62) to the scheme
        S_diff = S_ixkpl - S_ixkml
        S_lpdiff = S_ixklp - S_ixkl
        S_lmdiff = S_ixkl - S_ixklm
        s_l_grad_up = (((S_diff + S_ixkplp - S_ixkmlp)
                        ** 2. / (4 * self.dx) ** 2.) +
                       (S_lpdiff ** 2. / self.dy ** 2.)) ** ((N - 1.) / 2.)
        s_l_grad_dn = (((S_diff + S_ixkplm - S_ixkmlm)
                        ** 2. / (4 * self.dx) ** 2.) +
                       (S_lmdiff ** 2. / self.dy ** 2.)) ** ((N - 1.) / 2.)

        D_l_up = self.gamma * H_l_up ** (N + 1) * H_l_upstream_up * s_l_grad_up
        D_l_dn = self.gamma * H_l_dn ** (N + 1) * H_l_upstream_dn * s_l_grad_dn

        # --- all the k components

        # applying Eq. (61) to the scheme
        H_k_up = 0.5 * (Hkp + Hl)
        H_k_dn = 0.5 * (Hl + Hkm)

        H_k_upstream_up = self.H_upstream_up
        gt = S_ixkpl > S_ixkl
        H_k_upstream_up[gt] = Hkp[gt]
        H_k_upstream_up[~gt] = Hk[~gt]

        H_k_upstream_dn = self.H_upstream_dn
        gt = S_ixkl > S_ixkml
        H_k_upstream_dn[gt] = Hk[gt]
        H_k_upstream_dn[~gt] = Hkm[~gt]

        # applying Eq. (62) to the scheme
        S_diff = S_ixklp - S_ixklm
        S_kpdiff = S_ixkpl - S_ixkl
        S_kmdiff = S_ixkl - S_ixkml
        s_k_grad_up = (((S_diff + S_ixkplp - S_ixkplm)
                        ** 2. / (4 * self.dy) ** 2.) +
                       (S_kpdiff ** 2. / self.dx ** 2.)) ** ((N - 1.) / 2.)
        s_k_grad_dn = (((S_diff + S_ixkmlp - S_ixkmlm)
                        ** 2. / (4 * self.dy) ** 2.) +
                       (S_kmdiff ** 2. / self.dx ** 2.)) ** ((N - 1.) / 2.)

        D_k_up = self.gamma * H_k_up ** (N + 1) * H_k_upstream_up * s_k_grad_up
        D_k_dn = self.gamma * H_k_dn ** (N + 1) * H_k_upstream_dn * s_k_grad_dn

        # --- Check the cfl condition
        divisor = max(max(np.max(np.abs(D_k_up)), np.max(np.abs(D_k_dn))),
                      max(np.max(np.abs(D_l_up)), np.max(np.abs(D_l_dn))))
        if divisor == 0:
            dt_cfl = self.max_dt
        else:
            dt_cfl = (self.cfl * min(self.dx ** 2., self.dy ** 2.) / divisor)

        # --- Calculate Final diffusion term
        div_k = (D_k_up * S_kpdiff / self.dy -
                 D_k_dn * S_kmdiff / self.dy) / self.dy
        div_l = (D_l_up * S_lpdiff / self.dx -
                 D_l_dn * S_lmdiff / self.dx) / self.dx

        return div_l + div_k, dt_cfl

    def step(self, dt):
        """Advance one step."""

        div_q, dt_cfl = self.diffusion_upstream_2d()

        dt_use = utils.clip_scalar(np.min([dt_cfl, dt]), 0, self.max_dt)

        self.ice_thick = utils.clip_min(self.surface_h +
                                        (self.get_mb() + div_q) * dt_use -
                                        self.bed_topo,
                                        0)

        # Next step
        self.t += dt_use
        return dt
