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
shortcut.place_spacecraft_in_stable_orbit(2, 100e3, 0, 6)

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

t, pos, vel = landing.orient()

A = mission.lander_area     # cross section area of the lander in m^2
R = system.radii[6] * 1000 # m
m_planet = system.masses[6] * const.m_sun # kg
G = const.G # gravitational constant
g = G * m_planet / R**2 # gravitational acceleration on our planet 3.174917721996738 m/s^2
m_lander = mission.lander_mass # kg
m_craft = mission.spacecraft_mass # kg

def air_resistance(position, velocity, C_d=1):

    vr, v_theta = velocity
    r, r_theta = position

    T = system.rotational_periods[6]
    omega = 2 * np.pi / T

    v_drag = np.array([-vr, (omega * r) - v_theta])

    F_drag = 0.5 * get_rho(r - R) * C_d * A * v_drag**2

    return F_drag


def trajectory_landing(initial_time, initial_position, initial_velocity, simulation_time):
    x, y, z = initial_position
    vx, vy, vz = initial_velocity

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(y / x)

    v = np.sqrt(vx**2 + vy**2)
    vr = (x*vx + y*vy) / np.sqrt(x**2 + y**2)
    v_theta = np.sqrt(v**2 - vr**2)

    dt = 0.001 # s
    time_steps = int(np.ceil(simulation_time / dt))
    t = np.zeros(time_steps)

    x = np.zeros((2, time_steps))
    v = np.zeros((2, time_steps))

    x[:,0] = np.array([r, theta])
    v[:,0] = np.array([vr, v_theta])

    F_G = - m_craft * g * np.array([1, 0])

    for i in range(time_steps-1):

        F_drag = air_resistance(x[:,i], v[:,i])
        print(np.linalg.norm(F_drag))
        F_tot = F_G + F_drag
        a = F_tot / m_lander

        v[:,i+1] = v[:,i] + a * dt
        x[:,i+1] = x[:,i] + v[:,i+1] * dt
        t[i+1] = t[i] + dt
    plt.plot(t, F_drag)

    return t, x, v


t, r, v = trajectory_landing(t, pos, vel, 600)
plt.plot(r[0,:], r[1,:])
plt.show()
