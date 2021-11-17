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
shortcut.place_spacecraft_in_stable_orbit(2, 0, 0, 6)

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

    r, r_theta = position
    vr, v_theta = velocity
    omega = 2 * np.pi / T

    v = np.array([vr, v_theta])
    w = np.array([0, omega * r])
    v_drag = v + w

    F_d = - 1/2 * get_rho(r - R) * C_d * A * v_drag * np.linalg.norm(v_drag)

    return F_d

def trajectory_lander(initial_time, initial_velocity, initial_position, simulation_time):

    dt = 0.001
    time_steps = int(np.ceil(simulation_time / dt))

    x, y, z = initial_position
    vx, vy, vz = initial_velocity

    r = np.sqrt(x**2 + y**2)
    r_theta = np.arctan(y / x)
    print(r)

    vr = x * vx + y * vy / np.sqrt(x**2 + y**2)
    v_theta = r * (x * vy - vx * y / (x**2 + y**2))

    position = np.zeros((2, time_steps))
    velocity = np.zeros((2, time_steps))
    t = np.zeros(time_steps)

    position[:,0] = np.array([r, r_theta])
    velocity[:,0] = np.array([vr, v_theta])

    F_G = np.array([-m_craft * g, 0])

    for i in range(time_steps-1):

        F_d = F_drag(position[:,i], velocity[:,i])
        F_tot = F_G + F_d

        a = F_tot / m_craft

        velocity[:,i+1] = velocity[:,i] + a * dt
        position[:,i+1] = position[:,i] + velocity[:,i+1] * dt
        t[i+1] = t[i] + dt

    # rx = position[0,:] * np.cos(position[1,:])
    # ry = position[0,:] * np.sin(position[1,:])
    # r_cart = np.array([rx, ry])
    #
    # vx = velocity[0,:] * np.cos(position[1,:]) - position[0,:]**2 * velocity[1,:] * np.sin(position[1,:])
    # vy = velocity[0,:] * np.sin(position[1,:]) + position[0,:]**2 * velocity[1,:] * np.cos(position[1,:])
    # v_cart = np.array([vx, vy])

    return t, position, velocity #t, r_cart, v_cart

t, r, v = trajectory_lander(initial_time, vel, pos, 300)


# fig = plt.figure()
# ax = fig.add_subplot(projection='polar')
# ax.plot(r[0,:].T, r[1,:].T)
# plt.show()
