"""
Module for calculating parameters relevant for Julian, 1994 tremor model
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import hilbert
# from tqdm import tqdm

# Set some defaut values
gravity_default = 3.711
depth_default = 2.e3
pratio_default = 1.01
mu_default = 7.e9
rho_default = 2.7e3
A_default = 0.
L_default = 200.
k_default = 1.e9
h0_equil_frac = 0.95 # The assumed fraction of equilibrium h0 used

# First a couple private utility functions
def _vectorfield(w, t, p): # Private utility function for integration
    """
    Defines the differential equations for the tremor system

    Arguments:
      w : vector of the state variables
          w : [v, h, u] fluid velocity, wall position, wall velocity
          t : time
          p : vector of the parameters
              [rho, eta, p1, p2, L, M, A, k, h0]
    """
    v, h, u = w
    rho, eta, p1, p2, L, M, A, k, h0 = p

    omega0 = L*(0.5*(p1 + p2) - 0.5*rho*v*v) - k*(h-h0)
    omega1 = A + L*L*L/(12*h)*((12*eta/(h*h)) - 0.5*rho*u/h)
    omega2 = M + rho*L*L*L/(12*h)
    f = [1./rho*((p1 - p2)/L - 12*eta/(h*h)*v),
         u,
         1./omega2*(omega0 - omega1*u)]
    return f

def _wtcoef(t, t1, t2, t3, t4):
    """
    Function to calculate cosine taper

    returns weight coefficient between 0 and 1

    cosine taper from 0 to 1 t1 < t < t2
    1 for t2 < t < t3
    cosine taper from 1 to 0 t3 < t < t4
    0 for t < t1 or t > t2
    """

    if t3 > t4:
        raise ValueError('wtcoef: t3>t4')
    if t1 > t2:
        raise ValueError('wtcoef: t1>t2')

    if (t >= t2) and (t <= t3):
        wt = 1.0
    elif (t >= t4) or (t <= t1):
        wt = 0.0
    elif (t > t3) and (t < t4):
        wt = 0.5 * (1.0 + math.cos(math.pi * (t - t3)/(t4 - t3)))
    elif (t > t1) and (t < t2):
        wt = 0.5 * (1.0 + math.cos(math.pi * (t - t2)/(t2 - t1)))
    else:
        print(t, t1, t2, t3, t4)
        raise ValueError('wtcoef: this should be impossible')
    return wt

class TremorModel(object):
    """
    An object to set parameters for a tremor model and derive properties.  
    If values are not specified, default values set in this module are assumed.

    Expected inputs:
    depth - Depth to constriction in meters 
    pratio - Percent overpressure of lower reservoir.  Upper reservoir is
             assumed hydrostatic
    mu - shear modulus of wall material
    rho - density of fluid
    A - anelastic damping of wall material
    g - gravity of the planet
    k - elastic spring constant of wall
    width - horizontal dimenstion of constriction

    Note: only one of k or width should be defined as k is uniquely defined by
          mu, L, and width.  If both are set, only k will be used.
    """
    def __init__(self, depth=None, pratio=None, mu=None, rho=None,
                 A=None, L=None, g=None, k=None, width=None, h0=None, p1=None,
                 p2=None, M=None):
        if depth is not None:
            self.depth = depth
        else:
            self.depth = depth_default
        if p2 is not None:
            self.p2 = p2
            if p1 is not None:
                self.p1 = p1
                self.pratio = self.p2/self.p1 # Ignores any set pratio value
            elif pratio is not None:
                self.pratio = pratio
                self.p1 = self.p2*self.pratio
            else:
                self.pratio = pratio_default
                self.p1 = self.p2*self.pratio
        elif p1 is not None: # p2 is None but p1 is set
            print("Warning: p1 cannot be set without p2. Using defaults")
            self.p2 = None
            self.p1 = None
            if pratio is not None:
                self.pratio = pratio
            else:
                self.pratio = pratio_default
        else: # Both p1 and p2 are None
            self.p1 = None
            self.p2 = None
            if pratio is not None:
                self.pratio = pratio
            else:
                self.pratio = pratio_default
        if mu is not None:
            self.mu = mu
        else:
            self.mu = mu_default
        if rho is not None:
            self.rho = rho
        else:
            self.rho = rho_default
        if A is not None:
            self.A = A
        else:
            self.A = A_default
        if L is not None:
            self.L = L
        else:
            self.L = L_default
        if g is not None:
            self.g = g
        else:
            self.g = gravity_default
        if k is not None:
            self.k = k
            if width is not None:
                print("Warning: Cannot set both k and width.  Only using k.")
            self.calc_width()
        else:
            if width is not None:
                self.width = width
                self.calc_k()
            else:
                self.k = k_default
                self.calc_width()
        # If these are None, will be set in calc_derived
        self.h0 = h0 
        self.M = M

    def calc_width(self):
        """ 
        Determine constriction width from L, mu, k
        """
        self.width = self.L * self.mu/self.k

    def calc_k(self):
        """ 
        Determine wall constant k from  L, mu, width
        """
        self.k = self.L * self.mu/self.width
    
    def calc_derived(self):
        """
        Fill in other necessary derived parameters once input parameters
        are set
        """
        if self.p2 is None:
            self.p2 = self.rho * self.g * self.depth
        if self.p1 is None:
            self.p1 = self.p2 * self.pratio
        self.dp = self.p1 - self.p2
        if self.M is None:
            self.M = 0.5 * self.rho * self.L * self.L
        if self.h0 is None:
            self.h0 = -0.5*h0_equil_frac*(self.p1 + self.p2)*self.L/self.k
        self.aspect = self.width/self.L # May allow this to be set
        self.Vp = math.sqrt(3.*self.mu/self.rho) # Assumes Poisson solid

    def set_eta(self, eta):
        """
        Set an array of one or more viscosity values
        """
        if hasattr(eta, '__len__'):
            self.eta = np.array(eta)
        else:
            self.eta = np.array([eta])

    def _calc_hs(self):
        """
        For a given value or array of viscosity, eta, calculate steady crack
        opening hs

        Note, currently assumes there will be one real, positive root found,
        but this should be checked in the future
        """
        try:
            hs = np.zeros_like(self.eta)
        except NameError:
            print("Error: Set eta before calculating hs")
            raise
        except:
            raise
        B = 0.
        C = 0.
        D = self.k
        E = -1. * (self.k*self.h0 + 0.5*(self.p1 + self.p2)*self.L)
        
        # For each eta, solve quartic and pick real, positive root
        for i, val in enumerate(self.eta): # Try to make more efficient?
            A = self.rho * self.dp * self.dp / (288. * val * val * self.L)
            roots = np.roots([A, B, C, D, E])
            hs[i] = roots.real[np.logical_and(abs(roots.imag)<1e-5,
                                              roots.real > 0)][0]
        self.hs = hs

    # The following functions need documentation and error catching
    def _calc_vs(self):
        self.vs = self.hs*self.hs/(12.*self.eta)

    def _calc_a(self):
        self.a = self.eta*self.L*self.L*self.L/(self.hs*self.hs*self.hs)

    def _calc_m(self):
        self.m = self.rho*self.L*self.L*self.L/(12.*self.hs)

    def calc_R(self):
        self._calc_hs()
        self._calc_vs()
        self._calc_a()
        self._calc_m()
        self.r1 = self.rho*self.L*self.vs*self.vs/(self.k*self.hs)
        self.r2 = (self.A + self.a)*self.m/((self.M + self.m)*self.a)
        self.r3 = self.a*(self.A + self.a)/(self.k*self.m)
        self.R = (1.0 + 2.*self.r1)/((1.+self.r2)*(1.+self.r3))

    def calc_f(self): # oscillation frequency
        omega = np.sqrt(self.k*(1. + self.r3)/(self.M + self.m))
        self.f = omega/(2.*math.pi)
        # self.f = np.sqrt(self.k*(1. + self.r3)/(self.M + self.m))
        
    def generate_tremor(self, duration, dt, w0):
        """
        Integrate the initial value problem to generate a time series of tremor
        
        Inputs:
        duration : length of output time series in seconds
        dt : sampling of output time series in seconds
        w0 : [vi, hi, ui] initial condition of model run, where v is fluid
             velocity, h is wall position, and u is wall velocity

        Returns:
        Tuple (t, wsol)
            t = array of time samples
            wsol = array with dimensions n_eta, n_t, 3 containing the fluid
                   velocity, wall position, and wall velocity
        """
        # Create the time samples
        t0 = 0.
        t = np.arange(t0, duration, dt)

        # Loop over eta values
        wsol = []
        for val in self.eta:
            # Pack parameters
            p = [self.rho, val, self.p1, self.p2, self.L, self.M, self.A,
                 self.k, self.h0]

            ivp_out = solve_ivp(fun=lambda t, w: _vectorfield(w, t, p),
                                t_span=(0, duration), y0=w0, method='RK45',
                                t_eval=t)
            wsol.append(np.transpose(ivp_out['y']))

        wsol = np.array(wsol)
        self.wsol = (t, wsol) #include with object
        return (t, wsol)
            
    def get_duration(self, taper=None, threshold=None):
        """
        Function to determine when wall velocity u drops to 1/e of original 
        amplitude

        Uses wsol calculated in generate_tremor
        """
        if taper is None:
            taper = 0.05
        if threshold is None: # Defaults to threshold of 5% of max envelope
            threshold = 0.05
        tarray = self.wsol[0]
        t_taper = taper*(tarray[-1] - tarray[0])
        wsol = self.wsol[1]
        dims = wsol.shape
        u = wsol[:,:,2]
        t1 = tarray[0]
        t2 = t1 + t_taper
        t3 = tarray[-1] - t_taper
        t4 = tarray[-1]
        wt = np.zeros_like(tarray)
        for i, t in enumerate(tarray):
            wt[i] = _wtcoef(t, t1, t2, t3, t4)
        for i in range(dims[0]): # Looping over etas
            u[i,:] = np.multiply(wt, u[i,:])
        # Loop over etas
        duration = np.zeros(dims[0])
        for i in range(dims[0]):
            # Compute envelope
            analytic = hilbert(u[i, :])
            envelope = np.abs(analytic)
            th = threshold*envelope.max()
            # Find index of last value that exceeds th
            ind = np.where(envelope > th)[0][-1]
            duration[i] = tarray[ind] - tarray[0]
            
        return duration
