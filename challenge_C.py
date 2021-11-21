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
from numba import njit
from time import time

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
shortcut.place_spacecraft_in_stable_orbit(2, 250e3, 0, 6)

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
omega_0 = vel[1]    # m/s
r_0 = pos[0]    # m
T_0_craft = 2 * np.pi * r_0 / omega_0   # s

R = system.radii[6] * 1e3   # radius of planet in m

height_0 = np.linalg.norm(pos) - R
landing_spot = pos - np.array([height_0, 0, 0])

# print(landing_spot)
'''
[1861743.59425878       0.               0.        ]
'''

# print(T_0_craft)
'''
5812.267294706657
'''
quarter = 900 # s
delta_v = vel + np.array([vel[1], -vel[1], 0])

landing.look_in_direction_of_planet()

landing.adjust_parachute_area(35)   # testing value m^2
landing.adjust_landing_thruster(1500, 200)   # testing value N
landing.launch_lander(delta_v)
landing.fall(2*quarter)

# landing.start_video()
landing.deploy_parachute()
landing.fall(2*quarter)

# landing.finish_video()

t, pos, vel = landing.orient()

print('')
print(f'Height above surface at time t={t} s:', np.linalg.norm(pos) - R, 'm')

print('Spacecraft mass:', mission.spacecraft_mass, 'kg')
print('Lander mass:', mission.lander_mass, 'kg')
print('Planet mass:', system.masses[6] * const.m_sun, 'kg')
