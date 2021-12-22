 # -*- coding: utf-8 -*-
"""
file: sim_toolbox.py
author: Ardavan Farahvash (MIT)

description: 
Functions and Classes for doing 1D Time-Dependent QM Simulations
"""

import numpy as np
from numpy.lib.scimath import sqrt

################################### Constants and Conversion Factors
global hbar, nm_to_bohr, eV_to_hartree, fs_to_au

nm_to_bohr = 18.897259885789
eV_to_hartree = 0.0367493000
fs_to_au =  41.341374575751
hbar = 1.0 

################################### Solver Classes

class TISE(object):
    """
    Time-Independent Schrodinger Equation Solvers
    """
    def __init__(self, m = 1.0, hbar = 1.0):
        """
        Define Planck's Constant and Mass of Particle
        """
        self.m = m
        self.hbar = hbar
    
    def particle_in_box(self, x_arr, n_solve):
        """
        Particle in a box wavefunctions.
        """
        n = np.arange(1,n_solve+1)
        L = x_arr.max() - x_arr.min()
        x_c = x_arr.mean()
        
        E = (n**2 * np.pi**2 * self.hbar**2)/(2 * self.m * L**2)
        psi = np.sqrt(2/L) * np.sin(n[None,:] *np.pi/L *(x_arr[:,None] - 
                                                         x_c + L/2))
        return E, psi
        
    def matrix_numerov(self, x_arr, V_arr, n_solve):
        """
        Matrix Numerov Method.
    
        Parameters
        ----------
        x_arr : Numpy Array.
            1D array of grid points
        V_arr : Numpy Array.
            1D array of potential at each grid point.
        n_solve : Int.
            Number of states to solve for.
    
        Returns
        -------
        E : Numpy Array.
            Energy eigenvalues.
        psi_arr : Numpy Array.
            Normalized Wavefunctions.
        """
        dx_arr = x_arr[1:] - x_arr[0:-1]
    
        if np.all(np.abs(dx_arr - dx_arr[0]) > 1e-10):
            raise ValueError("Error: X-spacing is not uniform")
    
        if len(x_arr) != len(V_arr):
            raise ValueError("x_arr and V_arr are not the same length")
            
        if len(x_arr) <= n_solve-2:
            raise ValueError("Number of Desired solutions exceeds number of gridpoints")
    
        dx = dx_arr[0]
        N_x = len(x_arr)
        
        # Parameter Matrices
        A = np.zeros((N_x-2, N_x-2))
        B = np.zeros((N_x-2, N_x-2))
    
        # Unitless potential
        g = -(2 * self.m / self.hbar**2) * V_arr
    
        for i in range(1, len(x_arr)-1):
            # Calculate Parameters
            alpha = (1 + 1.0/12 * (dx**2) * g[i+1])
            beta = -2 * (1 - 5.0/12 * (dx**2) * g[i])
            gamma = (1 + 1.0/12 * (dx**2) * g[i-1])
    
            # Build Matrix
            j = i - 1
            if i == 1:
                A[j, j] = beta
                A[j, j+1] = alpha
                B[j, j] = 10
                B[j, j+1] = 1
    
            elif i == len(x_arr)-2:
                A[j, j-1] = gamma
                A[j, j] = beta
                B[j, j-1] = 1
                B[j, j] = 10
    
            else:
                A[j, j-1] = gamma
                A[j, j] = beta
                A[j, j+1] = alpha
                B[j, j-1] = 1
                B[j, j] = 10
                B[j, j+1] = 1
    
        # Solve for Eigenvalues and eigenvectors
        eige, eigv = np.linalg.eigh(np.linalg.solve(B, A))
        
        indx_sort = np.argsort(-eige)
        eige = eige[indx_sort]
        eigv = eigv[:, indx_sort]
        eige = eige[0:n_solve]
        eigv = eigv[:,0:n_solve]
        
        # Calculate Energies from eigenvalues
        E = eige * (-(12/dx**2) * (self.hbar**2/(2*self.m)))
    
        # Calcualate/Normalize Wavefunctions from eigenvectors
        psi_arr = np.zeros((N_x, n_solve))
    
        psi_arr[1:-1, 0:n_solve] = eigv[:, 0:n_solve]
        N = np.trapz(np.abs(psi_arr)**2, x_arr, axis=0)
        psi_arr = psi_arr/sqrt(N)
    
        return E, psi_arr

class TDSE(object):
    """
    Time-Independent Schrodinger Equation Solvers
    """
    def __init__(self, hbar = 1.0):
        """
        Define Planck's Constant and Mass of Particle
        """
        self.hbar = hbar
    
    def prop_constant(self, t_arr, x_arr, psi_H, E, psi_0):
        """
        Time propogation for system with a time-independent Hamiltonian.
    
        Parameters
        ----------
        t_arr : Numpy Array.
            Array of timepoints. (Nt)
        x_arr : Numpy Array.
            Array of gridpoints. (Ngrid)
        psi_H : Numpy Array. 
            Eigenstates of Hamiltonian. (Ngrid, Nstates)
        E : Numpy Array.
            Eigenenergies of Hamiltonian. (Nstates)
        psi_0 : Numpy Array.
            Initial Wavefunction. (Ngrid)
    
        Returns
        -------
        psi_t : Numpy Array. 
            Wavefunction at time t. (Nt,Ngrid)
        """
        # Expand initial wavefunction in coefficients of reference Hamiltonian
        c_0 = np.trapz((psi_0[:, None]*np.conj(psi_H)).T, x_arr)
        c_0 = c_0/np.linalg.norm(c_0)
    
        # Propogate Wavefunction
        exp_E = np.exp(-1j*np.outer(E, t_arr)/self.hbar)
        psi_t = np.einsum("i,xi,it->tx", c_0, psi_H, exp_E)
    
        return psi_t
        
    def prop_spectral_rk4(self, t_arr, x_arr, V_kn_t, E_H, psi_H, psi_0):
        """
        RK4 - Time propogation for system with a time-dependent Hamiltonian.
        Uses spectral (matrix multiplication) method.
    
        Parameters
        ----------
        t_arr : Numpy Array.
            Array of timepoints. (Nt)
        x_arr : Numpy Array.
            Array of gridpoints. (Ngrid)
        V_kn : Numpy array - (Nt*2,Nstates,Nstates)
            Matrix elements of perturbing potential at each half-timestep.
        E_H : Numpy array
            Eigenenergies of Reference Hamiltonian.(Nstates)
        psi_H : Numpy array 
            Eigenstates of Reference Hamiltonian. (Ngrid, Nstates)
        psi_0 : Numpy Array.
            Initial Wavefunction. (Ngrid)
            
        Returns
        -------
        psi_t : Numpy Array. 
            Wavefunction at time t. (Nt,Ngrid)s
        """
        
        nt = np.size(t_arr)
        nstates = np.size(E_H)
    
        #Time Derivative 
        def tderiv(t, c, H_kn):
            deriv = -1j/self.hbar * np.einsum("kn,n->k", H_kn, c)
            return deriv
        
        # Expand initial wavefunction in coefficients of references Hamiltonian
        c_0 = np.trapz((psi_0[:, None]*np.conj(psi_H)).T, x_arr)
        c_0 = c_0/np.linalg.norm(c_0)
        c_t = [c_0]
        
        # Calculate time-dependent Hamiltonian
        E_diag = np.zeros((1,nstates,nstates),dtype=np.complex128)
        E_diag[0, np.diag_indices(nstates)[0],np.diag_indices(nstates)[1]] = E_H
        H_kn = V_kn_t + E_diag
        
        # Propogate Wavefunction
        for i in range(nt-1):
            # Index For keeping track of half steps
            j = 2*i
            t = t_arr[i]
            dt = t_arr[i+1] - t_arr[i]
            
            # rk4 steps
            k_1 = tderiv(t, c_t[i], H_kn[j])
            k_2 = tderiv(t + dt/2, c_t[i] + dt*k_1/2, H_kn[j+1])
            k_3 = tderiv(t + dt/2, c_t[i] + dt*k_2/2, H_kn[j+1])
            k_4 = tderiv(t + dt  , c_t[i] + dt*k_3  , H_kn[j+2])
            c_new = c_t[i] + (dt/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)
            
            # ensure normalization and append
            #c_new = c_new/np.linalg.norm(c_new)
            c_t.append(c_new)
    
        c_t = np.array(c_t)
        psi_t = np.einsum("ti,xi->tx", c_t, psi_H)
    
        return psi_t, c_t
    
    #TO-DO: implement split operator
    def prop_split_operator():
        pass
        
################################### Matrix Element Calculators

def calc_mel_Vtx(x_arr, V_tx, psi_H):
    """
    Calculate Matrix Elements of General Time/Space Dependent Potential V(t,x).

    Parameters
    ----------
    x_arr : Numpy array - (nx)
        Grid of points.
    V_tx : Numpy array - (nt,nx)
        Potential Energy.
    psi_H : Numpy array - (nx,nstates)
        Eigenstates of Reference Hamiltonian.

    Returns
    -------
    V_kn : Numpy array - (nt,nstates,nstates)
        Matrix elements.

    """
    nt = np.size(V_tx,axis=0)
    #nx = np.size(V_tx, axis=1)
    nstates = np.size(psi_H, axis=1)
    
    V_kn = np.zeros((nt,nstates,nstates),dtype=np.complex128)
    for k in range(nstates):
        for n in range(k, nstates):
            int_f = V_tx.T * psi_H[:, k, None] * np.conj(psi_H[:, n, None])
            V_kn[:, k, n] = np.trapz(int_f, x_arr, axis=0)
            V_kn[:, n, k] = np.conj(V_kn[:, k, n])
    return V_kn

def calc_mel_sepV(x_arr, V_x, V_t, psi_H):
    """
    Calculate Matrix Elements of Seperate Time/Space Dependent Potential
    V(x,t) = V(x) * V(t)
 
    Parameters
    ----------
    x_arr : Numpy array - (nx)
        Grid of points.
    V_x : Numpy array - (nx)
        Space part of Potential.
    V_t : Numpy array - (nt)
        Time part of Potential.
    psi_H : Numpy array - (nx,nstates)
        Eigenstates of Reference Hamiltonian.

    Returns
    -------
    V_kn : Numpy array - (nt,nstates,nstates)
        Matrix elements.

    """
    nstates = np.size(psi_H, axis=1)
    
    V_kn = np.zeros((nstates,nstates),dtype=np.complex128)
    for k in range(nstates):
        for n in range(k, nstates):
            int_f = V_x * psi_H[:, k] * np.conj(psi_H[:, n])
            V_kn[k, n] = np.trapz(int_f, x_arr)
            V_kn[n, k] = np.conj(V_kn[k, n])
    V_kn_t = (V_kn[:,:,None] * V_t).T
    return V_kn_t

def calc_mel_dipole(x_arr, E_omega, A_t, E_H, psi_H, q = 1):
    """
    Calculate matrix elements of radiation under the dipole approximation.
    V_kn(t) =  -i * q * A(t) *  (E_kn/hbar*omega) * <k | r | n>
    
    Parameters
    ----------
    x_arr : Numpy array - (nx)
        Grid of points.
    E_omega : Float.
        Energy of incident light, E = hbar * omega
    A_t : Numpy array - (nt,nx)
        Time dependence of radiation field.
    E_H : Numpy array - (nstates)
        Eigenenergies of Reference Hamiltonian.
    psi_H : Numpy array - (nx,nstates)
        Eigenstates of Reference Hamiltonian.
    q : Float.
        Charge of particle, default is 1.

    Returns
    -------
    V_kn : Numpy array - (nt,nstates,nstates)
        Matrix elements.

    """
    # Energy Difference Matrix
    E_kn = E_H[:,None] - E_H
    
    # Matrix Elements of position operator
    V_tx = calc_mel_sepV(x_arr, x_arr, A_t, psi_H)
    
    # Matrix elements of Pulse under dipole approximation
    V_kn = -1j * q * (E_kn[None,:,:]/E_omega) * V_tx 
    
    return V_kn

################################### Field Envelope Functions

def create_Vconstant(t_arr, EField):
    """
    Create Constant Potential
    """
    
    V_t = EField * np.ones(np.size(t_arr))
    
    return(V_t)

def create_Vsin(t_arr, EField, E_omega):
    """
    Create Time-Dependent Potential for continuous sin wave.
    """
    
    V_t = EField * np.sin(E_omega/hbar * t_arr)
    
    return(V_t)

def create_Vsin_finite(t_arr, EField, E_omega, tmax):
    """
    Create Time-Dependent Potential for finite sine wave.
    """
    nt = len(t_arr)
    indx_tmax = np.argmin(np.abs(t_arr - tmax))

    V_t = np.zeros(nt)
    V_t = EField * np.sin(E_omega/hbar * t_arr[0:indx_tmax])
    
    return(V_t)

def create_Vpulse(t_arr, EField, E_omega, tpulse):
    """
    Create Time-Dependent Potential for a pulse wave.
    """
    
    wave = np.sin(E_omega/hbar * t_arr)
    envelope = np.sin(np.pi * t_arr/tpulse)**2
    
    V_t = EField * wave * envelope
    return(V_t)

def create_Vpulse_finite(t_arr, EField, E_omega, tpulse, tmax):
    """
    Create Time-Dependent Potential for finite pulse wave.
    """
    nt = len(t_arr)
    indx_tmax = np.argmin(np.abs(t_arr - tmax))
    
    wave = np.sin(E_omega/hbar * t_arr[0:indx_tmax])
    envelope = np.sin(np.pi * t_arr[0:indx_tmax]/tpulse)**2
    
    V_t = np.zeros(nt)
    V_t[0:indx_tmax] = EField * wave * envelope
    return(V_t)

if __name__ == "__main__":
    pass