import numpy
from numpy import zeros, arange, diff, real, sqrt, exp, pi, mean, angle, unwrap
from numpy.fft import fft, ifft, fftshift
from numpy.random import randn, seed

def simulate_scintillation(U, mu0, p1, p2, rho_over_veff, freqs, N, dt=0.01, SEED=None):
    '''
    ----------------------------------------------------------------------------
    Given compact normalized phase screen parameters, computes phase screen PSD
    and generates random phase screen for each frequency in `freqs`.  The wave
    is then propagated to ground using parabolic wave approximation.
    
    Inputs:
    `U` -- universal strength parameter (0.1-0.3 weak; 0.3-0.6 moderate; 0.6-1.0
        moderate-strong; 1.0- strong)
    `mu0` -- normalized scale breakpoint
    `p1` -- normailzed inner-scale
    `p2` -- normalized outer-scale
    `freqs` -- iterable set of frequencies for which to run simulation
    `N` -- size of simulation time-series
    `dt` -- sampling period of simulation time series
    `SEED` -- random number generator seed.  Unused if `None` (default).
    
    Returns:
    `psi` -- shape (M, N) array containing the complex received signal time 
        series (axis 1) for each frequency (axis 0)
    `phase_screens` -- shape (M, N) array containing the real-valued phase
        screen (axis 1) for each frequency (axis 0)
    '''
    freq_fit = freqs[0]
    M = len(freqs)
    phase_screens = zeros((M, N))       # equivalent ionosphere phase screens
    psi = zeros((M, N), dtype=complex)  # received fields

    # generate complex random number for PSD generation
    if SEED is not None:
        seed(SEED)
    # `/ sqrt(2)` not needed since we take real part afterwards
    X = randn(N) + randn(N) * 1j

    mu = 2 * pi * rho_over_veff * (arange(N) - N // 2) / (N * dt)
    dmu = diff(mu)[0]
    Cpp = U if mu0 >= 1 else U / mu0**(p2 - p1)

    # generate SDF(mu)
    SDF0 = Cpp * (abs(mu)**(-p1) * (abs(mu) <= mu0) + mu0**(p2 - p1) * abs(mu)**(-p2) * (abs(mu) > mu0))
    SDF0[N // 2] = 0  # set DC to zero
    SDF = SDF0 * dmu / (2 * pi)

    # generate phase screen realization
    phase_screens[0, :] = real(fftshift(fft(fftshift(sqrt(SDF) * X))))

    for i in range(1, M):
        f1, f2 = freq_fit, freqs[i]
        Cpp_2 = Cpp * (f1 / f2)**(2 + (p1 - 1) / 2)
        mu0_2 = mu0 * sqrt(f1 / f2)
        rho_over_veff_2 = rho_over_veff * sqrt(f1 / f2)
    #     U_2 = Cpp_2 if mu0_2 >= 1 else Cpp_2 * mu0_2**(p2 - p1)
        mu_2 = mu * rho_over_veff_2 / rho_over_veff
        dmu_2 = diff(mu_2)[0]
        SDF0 = Cpp_2 * (abs(mu_2)**(-p1) * (abs(mu_2) <= mu0_2) + mu0_2**(p2 - p1) * abs(mu_2)**(-p2) * (abs(mu_2) > mu0_2))
        SDF0[N // 2] = 0  # set DC to zero
        SDF = SDF0 * dmu_2 / (2 * pi)
        phase_screens[i, :] = real(fftshift(fft(fftshift(sqrt(SDF) * X))))

    # propagate to receiver
    for i in range(M):
        f1, f2 = freq_fit, freqs[i]
        rho_over_veff_2 = rho_over_veff * sqrt(f1 / f2)
        mu_2 = mu * rho_over_veff_2 / rho_over_veff
        prop_factor = fftshift(exp(-1j * abs(mu_2)**2 / 2))
        psi0 = exp(1j * phase_screens[i, :])
        psi_hat = fft(psi0)
        psi[i, :] = ifft(psi_hat * prop_factor)

    return psi, phase_screens



