import numpy as np
import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission, LandingSequence
import ast2000tools.utils as utils
import ast2000tools.constants as const
from ast2000tools.shortcuts import SpaceMissionShortcuts
import scipy.constants as scs
from numba import njit
import itertools
from tqdm import trange

"""
EGEN KODE: Anton Brekke
"""

seed = utils.get_seed('antonabr')
system = SolarSystem(seed)

# Funksjon som gir overflatetemperatur på planet
def surface_T(R_star, T_star, r_dist):
    # Returns in Kelvin
    return (T_star**4 / 4)**(1/4) * (R_star / r_dist)**(1/2)

# Generellse konstanter
planet_mass = system.masses[6] * const.m_sun
planet_radius = system.radii[6] * 1000
R_star = system.star_radius * 1000          # m
gamma = 1.4
k = const.k_B
mh = const.m_p
# g = 9.81  # antatt er konstant (husk å bytte til g for planeten vi befinner oss på)
T_star = 7985       # K

# Masser til vanlige gasser i atmosfærer:
m_o2 = 2 * 16*mh
m_h2o = 2*mh + 16*mh
m_co2 = 12*mh + 2 * 16*mh
m_ch4 = 12*mh + 4*mh
m_co = 12*mh + 16*mh
m_n2o = 2 * 14*mh + 16*mh
# print('\n')
# print(f'm_o2: {m_o2}')
# print(f'm_h2o: {m_h2o}')
# print(f'm_co2: {m_co2}')
# print(f'm_ch4: {m_ch4}')
# print(f'm_co: {m_co}')
# print(f'm_n2o: {m_n2o}')
# print('\n')

mass_array = np.array([m_o2, m_h2o, m_co2, m_ch4, m_co, m_n2o])
def optimize_mu(mass_array, option='minimize'):
    if option == 'minimize':
        opt = lambda x: np.min(x)
    if option == 'maximize':
        opt = lambda x: np.max(x)
    combinations_mass = np.array([*itertools.combinations_with_replacement(mass_array, mass_array.size)])
    # print(combinations_mass)
    N, M = combinations_mass.shape
    mu_comb = np.zeros((N))
    for i in trange(len(mu_comb), desc='Calculating mu'):
        mu_comb[i] = np.sum(combinations_mass[i]) / (M*mh)
    # print('')
    where_min = np.where(mu_comb == opt(mu_comb))
    # print(f'Total combinations: {len(mu_comb)}')
    # print(where_min)
    # print(mu_comb[where_min])
    # print(combinations_mass[where_min])
    return combinations_mass[where_min], mu_comb[where_min]
mass_comb, min_mu = optimize_mu(mass_array, 'minimize')

n = 0
# print(f'Amount of combinations to choose: {len(min_mu)}\n')
# print(mass_comb[n])         # Kombinasjon av masse
mu = min_mu[n]            # Komposisjon av atmosfære
r_dist = 3.64 * const.AU            # Middel-avstand
g = planet_mass * const.G / (planet_radius)**2  # antatt er konstant (husk å bytte til g for planeten vi befinner oss på)
rho0 = system.atmospheric_densities[6]
T0 = surface_T(R_star, T_star, r_dist)

# Adiabatiske konstanter
K0 = mu * mh / k
A = T0 * (rho0 / K0)**(1 - gamma)
# Enten eller for C (kan løses på 3 måter)
# C = (gamma / (1-gamma)) * (rho0*T0 / K0)**((gamma - 1) / gamma)
C = T0 * A**(-1/gamma) * (gamma / (1-gamma))

r_shift_adiabatic_isoterm = T0 / (2*K0*g) * (gamma / (1 - gamma)) - C*A**(1/gamma) / (K0*g)     # r der det skifter mellom adiaabtisk og isoterm (T_adiabatic = T0/2)

# Likninger løst analytisk
def T_adiabatic(r):
    return A**(1/gamma) * (1 - gamma) / gamma * (K0*A**(-1/gamma)*g*r + C)

def rho_adiabatic(r):
    return A**(-1/gamma)*K0 * (((1 - gamma) / gamma) * (K0*A**(-1/gamma)*g*r + C))**(1 / (gamma - 1))

def P_adiabatic(r):
    return (((1 - gamma) / gamma) * (K0*A**(-1/gamma)*g*r + C))**(gamma / (gamma - 1))

rho_shift_adiabatic_isoterm = rho_adiabatic(r_shift_adiabatic_isoterm)          # rho der det skifter mellom adiabatisk og isoterm
temp_shift_adiabatic_isoterm = T_adiabatic(r_shift_adiabatic_isoterm)          # temp der det skifter mellom adiabatisk og isoterm
P_shift_adiabatic_isoterm = P_adiabatic(r_shift_adiabatic_isoterm)          # trykk der det skifter mellom adiabatisk og isoterm


# Isoterme konstanter
K1 = k*T0 / (2*mu*g*mh)
C0 = rho_shift_adiabatic_isoterm*np.exp(1/K1 * r_shift_adiabatic_isoterm)           # Løser for C0 i isoterm likning, gadd ikke analytisk

def get_rho(r):
    if isinstance(r, (int,float)):
        if r <= r_shift_adiabatic_isoterm:
            rho = rho_adiabatic(r)
        else:
            rho = C0 * np.exp(-1/K1*r)
        return rho
    else:
        adiabatic = np.where(r <= r_shift_adiabatic_isoterm)
        isoterm = np.where(r > r_shift_adiabatic_isoterm)

        adiabatic_rho = rho_adiabatic(r[adiabatic])
        isoterm_rho = C0 * np.exp(-1/K1*r[isoterm])

        rho = np.concatenate((adiabatic_rho, isoterm_rho))
        return rho

def get_T(r):
    if isinstance(r, (int,float)):
        if r <= r_shift_adiabatic_isoterm:
            temp = T_adiabatic(r)
        else:
            temp = T0 / 2
        return temp
    else:
        adiabatic = np.where(r <= r_shift_adiabatic_isoterm)
        isoterm = np.where(r > r_shift_adiabatic_isoterm)

        adiabatic_temp = T_adiabatic(r[adiabatic])
        isoterm_temp = T0 / 2 * np.ones_like(r[isoterm])

        temp = np.concatenate((adiabatic_temp, isoterm_temp))
        return temp

def get_P(r):
    if isinstance(r, (int,float)):
        if r <= r_shift_adiabatic_isoterm:
            P = P_adiabatic(r)
        else:
            P = K1*g*get_rho(r)
        return P
    else:
        adiabatic = np.where(r <= r_shift_adiabatic_isoterm)
        isoterm = np.where(r > r_shift_adiabatic_isoterm)

        adiabatic_P = P_adiabatic(r[adiabatic])
        isoterm_P = K1*g*get_rho(r[isoterm])

        P = temp = np.concatenate((adiabatic_P, isoterm_P))
        return P

if __name__ == '__main__':
    # printer konstanter for å ha kontroll på størrelsene
    print(f'Radius stjerne [m]: {R_star}')
    print(f'Overflate temperatur planet [K]: {T0}')
    print(f'A: {A}')
    print(f'C: {C}')
    print(f'C0: {C0}')
    print(f'gamma: {gamma}')
    print(f'K0: {K0}')
    print(f'K1: {K1}')
    print(f'rho0 [kg/m^3]: {rho0}')
    print(f'g [m/s^2]: {g}')
    print(f'mu: {mu}')
    print(f'm_H [kg]: {mh}')
    print(f'r_shift_adiabatic_isoterm [m]: {r_shift_adiabatic_isoterm}')
    print(f'rho_shift_adiabatic_isoterm [m]: {rho_shift_adiabatic_isoterm}')
    print(f'temp_shift_adiabatic_isoterm [K]: {temp_shift_adiabatic_isoterm}')
    print(f'P_shift_adiabatic_isoterm [Pa]: {P_shift_adiabatic_isoterm}')

    # Plotter alle sammen
    r = np.linspace(0, int(1e5), int(1e5))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(r, get_rho(r), color='r', label=r'$\rho\;[kg/m^3]$')
    ax2.plot(r, get_T(r), color='tab:orange', label='T [K]')
    ax3.plot(r, get_P(r), color='tab:blue', label='P [Pa]')

    ax3.set_xlabel('r [m]', fontsize=12, weight='bold')
    ax1.set_ylabel(r'$\rho\;[kg/m^3]$', fontsize=12, weight='bold')
    ax2.set_ylabel('T [K]', fontsize=12, weight='bold')
    ax3.set_ylabel('P [Pa]', fontsize=12, weight='bold')

    fig.tight_layout()
    ax1.legend(); ax2.legend(); ax3.legend()
    plt.show()

    # Plot rho
    plt.plot(r, get_rho(r), color='r', label=r'$\rho$ [kg/m^3]')
    plt.xlabel('r [m]', fontsize=12, weight='bold')
    plt.ylabel(r'$\rho\;[kg/m^3]$', fontsize=12, weight='bold')
    plt.legend()
    plt.show()

    # Plot temp
    plt.plot(r, get_T(r), color='tab:orange', label='T [K]')
    plt.xlabel('r [m]', fontsize=12, weight='bold')
    plt.ylabel('T [K]', fontsize=12, weight='bold')
    plt.legend()
    plt.show()

    # Plot P
    plt.plot(r, get_P(r), color='tab:blue', label='P [Pa]')
    plt.xlabel('r [m]', fontsize=12, weight='bold')
    plt.ylabel('P [Pa]', fontsize=12, weight='bold')
    plt.legend()
    plt.show()
