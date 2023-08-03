import numpy as np
import xarray as xr
import os

from oggm import cfg, utils
from oggm.cfg import SEC_IN_YEAR


def filter_ice_border(ice_thick):
    """Sets the ice thickness at the border of the domain to zero."""
    ice_thick[0, :] = 0
    ice_thick[-1, :] = 0
    ice_thick[:, 0] = 0
    ice_thick[:, -1] = 0
    return ice_thick


class Model2D(object):
    """Interface to a distributed model"""

    def __init__(self, bed_topo, init_ice_thick=None, dx=None, dy=None,
                 mb_model=None, y0=0., glen_a=None, mb_elev_feedback='annual',
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
        ice_thick_filter : func
            function to apply to the ice thickness *after* each time step.
            See filter_ice_border for an example. Set to None for doing nothing.
        mb_filter : ndarray
            2d array indicating the mask where positive mb is allowed
            (useful to allow only specific glaciers to grow)
        """

        # Mass balance
        self.mb_elev_feedback = mb_elev_feedback
        self.mb_model = mb_model
        self.mb_filter = mb_filter

        # Defaults
        if glen_a is None:
            glen_a = cfg.PARAMS['glen_a']
        self.glen_a = glen_a

        if dy is None:
            dy = dx

        self.dx = dx
        self.dy = dy
        self.dxdy = dx * dy

        self.y0 = None
        self.t = None
        self.reset_y0(y0)

        self.ice_thick_filter = ice_thick_filter

        # Data
        self.bed_topo = bed_topo
        self.ice_thick = None
        self.reset_ice_thick(init_ice_thick)
        self.ny, self.nx = bed_topo.shape

    @property
    def mb_model(self):
        return self._mb_model

    @mb_model.setter
    def mb_model(self, value):
        # We need a setter because the MB func is stored as an attr too
        _mb_call = None
        if value:
            if self.mb_elev_feedback in ['always', 'monthly']:
                _mb_call = value.get_monthly_mb
            elif self.mb_elev_feedback in ['annual', 'never']:
                _mb_call = value.get_annual_mb
            else:
                raise ValueError('mb_elev_feedback not understood')
        self._mb_model = value
        self._mb_call = _mb_call
        self._mb_current_date = None
        self._mb_current_out = dict()
        self._mb_current_heights = dict()

    def reset_y0(self, y0):
        """Reset the initial model time"""
        self.y0 = y0
        self.t = 0

    def reset_ice_thick(self, ice_thick=None):
        """Reset the ice thickness"""
        if ice_thick is None:
            ice_thick = self.bed_topo * 0.
        self.ice_thick = ice_thick.copy()

    @property
    def yr(self):
        return self.y0 + self.t / SEC_IN_YEAR

    @property
    def area_m2(self):
        return np.sum(self.ice_thick > 0) * self.dxdy

    @property
    def volume_m3(self):
        return np.sum(self.ice_thick * self.dxdy)

    @property
    def volume_km3(self):
        return self.volume_m3 * 1e-9

    @property
    def area_km2(self):
        return self.area_m2 * 1e-6

    @property
    def surface_h(self):
        return self.bed_topo + self.ice_thick

    def get_mb(self, year=None):
        """Get the mass balance at the requested height and time.

        Optimized so that no mb model call is necessary at each step.
        """

        if year is None:
            year = self.yr

        # Do we have to optimise?
        if self.mb_elev_feedback == 'always':
            _mb = self._mb_call(self.surface_h.flatten(), year=year)
            _mb = _mb.reshape((self.ny, self.nx))
            if self.mb_filter is not None:
                _mb[~ self.mb_filter & (_mb > 0)] = 0
            return _mb

        date = utils.floatyear_to_date(year)
        if self.mb_elev_feedback == 'annual':
            # ignore month changes
            date = (date[0], date[0])

        if self._mb_current_date != date or (self._mb_current_out is None):
            # We need to reset all
            self._mb_current_date = date
            _mb = self._mb_call(self.surface_h.flatten(), year=year)
            _mb = _mb.reshape((self.ny, self.nx))
            if self.mb_filter is not None:
                _mb[~ self.mb_filter & (_mb > 0)] = 0
            self._mb_current_out = _mb

        return self._mb_current_out

    def step(self, dt):
        """Advance one step."""
        raise NotImplementedError

    def run_until(self, y1, stop_if_border=False):
        """Run until a selected year."""

        t = (y1 - self.y0) * SEC_IN_YEAR
        while self.t < t:
            self.step(t - self.t)
            if stop_if_border:
                if (np.any(self.ice_thick[0, :] > 10) or
                        np.any(self.ice_thick[-1, :] > 10) or
                        np.any(self.ice_thick[:, 0] > 10) or
                        np.any(self.ice_thick[:, -1] > 10)):
                    raise RuntimeError('Glacier exceeds boundaries')
            if self.ice_thick_filter is not None:
                self.ice_thick = self.ice_thick_filter(self.ice_thick)

        if np.any(~np.isfinite(self.ice_thick)):
            raise FloatingPointError('NaN in numerical solution.')

    def run_until_equilibrium(self, rate=0.001, ystep=5, max_ite=200):
        """Run until an equilibrium is reached (can take a while)."""

        ite = 0
        was_close_zero = 0
        t_rate = 1
        while (t_rate > rate) and (ite <= max_ite) and (was_close_zero < 5):
            ite += 1
            v_bef = self.volume_m3
            self.run_until(self.yr + ystep)
            v_af = self.volume_m3
            if np.isclose(v_bef, 0., atol=1):
                t_rate = 1
                was_close_zero += 1
            else:
                t_rate = np.abs(v_af - v_bef) / v_bef
        if ite > max_ite:
            raise RuntimeError('Did not find equilibrium.')

    def run_until_and_store(self, ye, step=2, run_path=None, grid=None,
                            print_stdout=False, stop_if_border=False):
        """Run until a selected year and store the output in a NetCDF file."""

        yrs = np.arange(np.floor(self.yr), np.floor(ye) + 1, step)
        out_thick = np.zeros((len(yrs), self.ny, self.nx))
        for i, yr in enumerate(yrs):
            if print_stdout and (yr / 10) == int(yr / 10):
                print('{}: year {} of {}, '
                      'max thick {:.1f}m'.format(print_stdout,
                                                 int(yr),
                                                 int(ye),
                                                 self.ice_thick.max()),
                      end='\r', flush=True)
            self.run_until(yr, stop_if_border=stop_if_border)
            out_thick[i, :, :] = self.ice_thick

        run_ds = grid.to_dataset() if grid else xr.Dataset()
        run_ds['ice_thickness'] = xr.DataArray(out_thick,
                                               dims=['time', 'y', 'x'],
                                               coords={'time': yrs})

        run_ds['bed_topo'] = xr.DataArray(self.bed_topo,
                                          dims=['y', 'x'])

        # write output?
        if run_path is not None:
            if os.path.exists(run_path):
                os.remove(run_path)
            run_ds.to_netcdf(run_path)

        return run_ds