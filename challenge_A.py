'''
SNARVEI OG EGEN KODE
'''

import numpy as np
import matplotlib.pyplot as plt
from ast2000tools.solar_system import SolarSystem
from ast2000tools.space_mission import SpaceMission, LandingSequence
import ast2000tools.utils as utils
import ast2000tools.constants as const
from ast2000tools.shortcuts import SpaceMissionShortcuts
from challenge_C_part_6 import *
from numba import njit

seed = utils.get_seed('antonabr')
system = SolarSystem(seed)
mission = SpaceMission.load('mission_after_launch1.pickle')
print('Look, I am still in space:', mission.rocket_launched)

'''Shortcut begin'''

code_stable_orbit = 95927
code_orientation = 43160
system = SolarSystem(seed)

shortcut = SpaceMissionShortcuts(mission, [code_stable_orbit, code_orientation])

# Orientation software shortcut
pos, vel, angle = shortcut.get_orientation_data()
print("Position after launch:", pos)
print("Velocity after launch:", vel)

#Verifying orientation with shortcut data
mission.verify_manual_orientation(pos, vel, angle)

# Initialize interplanetary travel instance
travel = mission.begin_interplanetary_travel()

# Shortcut to make the landing sequence class start with a stable orbit
shortcut.place_spacecraft_in_stable_orbit(2, 200e3, 0, 6)

# Initializing landing sequence class instance
landing = mission.begin_landing_sequence()

# Calling landing sequece oreint function
t, pos, vel = landing.orient()

print("We are at:")
print("Time :", t)
print("pos :", pos)
print("vel :", vel)

'''Shortcut end'''
print('')
print('--------------------------')
print('')

'''
EGEN KODE
'''

# fall_time = 14420
#
# landing.fall_until_time(fall_time)

initial_time, pos, vel = landing.orient()

A = mission.lander_area     # cross section area of the lander in m^2
R = system.radii[6] * 1000 # m
m_planet = system.masses[6] * const.m_sun # kg
G = const.G # gravitational constant
g = G * m_planet / R**2 # gravitational acceleration on our planet 3.174917721996738 m/s^2
m_lander = mission.lander_mass # kg
m_craft = mission.spacecraft_mass # kg
T = system.rotational_periods[6] * 3600 * 24

def F_drag(position, velocity, C_d=1):

    x, y = position
    vx, vy = velocity

    r = np.sqrt(x**2 + y**2)
    r_theta = np.arctan(y / x)

    vr = x * vx + y * vy / np.sqrt(x**2 + y**2)
    v_theta = r * ((x * vy - vx * y) / (x**2 + y**2))

    omega = 2 * np.pi / T

    v = np.array([vr, v_theta])
    w = np.array([0, omega * r])
    v_drag = v + w

    F_d = - 1/2 * get_rho(r - R) * C_d * A * v_drag * np.linalg.norm(v_drag)

    F_d_r = F_d[0]
    F_d_theta = F_d[1]
    F_d_x = F_d_r * np.cos(r_theta) - F_d_theta * np.sin(r_theta)
    F_d_y = F_d_r * np.sin(r_theta) + F_d_theta * np.cos(r_theta)
    F_d_cartesian = np.array([F_d_x, F_d_y])

    return F_d_cartesian

def trajectory_lander(initial_time, initial_velocity, initial_position, simulation_time):

    dt = 0.001
    time_steps = int(np.ceil(simulation_time / dt))

    position = np.zeros((2, time_steps))
    velocity = np.zeros((2, time_steps))
    t = np.zeros(time_steps)

    position[:,0] = initial_position[:2]
    velocity[:,0] = initial_velocity[:2]

    for i in range(time_steps-1):

        r_vec = position[:,i]
        r_norm = np.linalg.norm(r_vec)
        unit_r = r_vec / r_norm

        F_G = -m_craft * g * unit_r

        F_d = F_drag(position[:,i], velocity[:,i])
        F_tot = F_G + F_d

        a = F_tot / m_craft

        velocity[:,i+1] = velocity[:,i] + a * dt
        position[:,i+1] = position[:,i] + velocity[:,i+1] * dt
        t[i+1] = t[i] + dt

    return t, position, velocity

t, r, v = trajectory_lander(initial_time, vel, pos, 600)

plt.subplot(211)
plt.title('Position plot')
plt.plot(r[0,:], r[1,:], label='$\vec{r}$')
plt.legend()

plt.subplot(212)
plt.title('$r_y$ as function of time')
plt.plot(t, r[1,:], label='$r_y$')
plt.legend()
plt.show()
