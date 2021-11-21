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
shortcut.place_spacecraft_in_stable_orbit(2, 500e3, 0, 6)

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

A_lander = mission.lander_area     # cross section area of the lander in m^2
A_craft = mission.spacecraft_area # cross section area of the craft in m^2
A_parachute = 0
A_lander_parachute = A_lander + A_parachute
R = system.radii[6] * 1000 # m
m_planet = system.masses[6] * const.m_sun # kg
G = const.G # gravitational constant
g_1 = G * m_planet / R**2 # gravitational acceleration on our planet 3.174917721996738 m/s^2
m_lander = mission.lander_mass # kg
m_craft = mission.spacecraft_mass # kg
T = system.rotational_periods[6] * 3600 * 24

@njit
def F_drag(position, velocity, area, C_d=1):

    x, y = position
    vx, vy = velocity

    r = np.sqrt(x**2 + y**2)
    r_theta = np.arctan(y / x)

    vr = x * vx + y * vy / np.sqrt(x**2 + y**2)
    v_theta = r * ((x * vy - vx * y) / (x**2 + y**2))

    omega = 2 * np.pi / T

    v = np.array([vr, v_theta])
    w = np.array([0, omega * r])
    v_drag = v - w

    r_test = r - R

    if r_test <= r_shift_adiabatic_isoterm:
        rho = A**(-1/gamma)*K0 * (((1 - gamma) / gamma) * (K0*A**(-1/gamma)*g_1*r_test + C))**(1 / (gamma - 1))
    else:
        rho = C0 * np.exp(-1/K1*r_test)

    F_d = - 1/2 * rho * C_d * area * v_drag * np.linalg.norm(v_drag)

    F_d_r = F_d[0]
    F_d_theta = F_d[1]
    F_d_x = F_d_r * np.cos(r_theta) - F_d_theta * np.sin(r_theta)
    F_d_y = F_d_r * np.sin(r_theta) + F_d_theta * np.cos(r_theta)
    F_d_cartesian = np.array([F_d_x, F_d_y])
    v_drag_cartesian = np.array([v_drag[0] * np.cos(r_theta) - v_drag[1] * np.sin(r_theta), v_drag[0] * np.sin(r_theta) + v_drag[1] * np.cos(r_theta)])

    return F_d_cartesian, v_drag_cartesian, rho

@njit
def trajectory_lander(initial_time, initial_velocity, initial_position, simulation_time):

    dt = 0.01
    time_steps = int(np.ceil(simulation_time / dt))

    position = np.zeros((2, time_steps))
    velocity = np.zeros((2, time_steps))
    t = np.linspace(initial_time, initial_time + simulation_time, time_steps)

    position[:,0] = initial_position[:2]
    velocity[:,0] = initial_velocity[:2]

    F_gravity = np.zeros((2, time_steps))
    F_drag_array = np.zeros((2, time_steps))
    v_drag_array = np.zeros((2, time_steps))
    rho_array = np.zeros(time_steps)

    for i in range(time_steps-1):

        r_vec = position[:,i]
        r_norm = np.linalg.norm(r_vec)
        unit_r = r_vec / r_norm

        g_acceleration = G * m_planet / r_norm**2

        F_d, v_drag, rho = F_drag(position[:,i], velocity[:,i], A_craft)
        F_G = -m_craft * g_acceleration * unit_r
        rho_array[i] = rho
        # if t[i] > 5:
        #     F_G = -m_lander * g_acceleration * unit_r
        #     if t[i] > 15:
        #         F_d = F_drag(position[:,i], velocity[:,i], A_lander_parachute)
        #     else:
        #         F_d = F_drag(position[:,i], velocity[:,i], A_lander)
        # else:
        #     F_G = -m_craft * g_acceleration * unit_r
        #     F_d = F_drag(position[:,i], velocity[:,i], A_craft)

        F_gravity[:,i] = F_G
        F_drag_array[:,i] = F_d
        v_drag_array[:,i] = v_drag
        F_tot = F_G + F_d

        a = F_tot / m_craft

        velocity[:,i+1] = velocity[:,i] + a * dt
        position[:,i+1] = position[:,i] + velocity[:,i+1] * dt

        if np.logical_and(abs(position[0,i]) < 2, abs(position[0,i]) > 0) == True:
            print('90 degrees reached, position:', position[:,i])

        if np.linalg.norm(position[:,i+1]) <= R:
            print('Ground hit after', t[i+1], 's. Good luck.')
            return t[:i], position[:,:i], velocity[:,:i], F_gravity[:,:i], F_drag_array[:,:i], v_drag_array[:,:i], rho_array[:i]

        if np.linalg.norm(position[:,i+1]) > 1e7:
            print('Craft is headed to deep space after', t[i+1], 's. See you later. Maybe. Good luck.')
            return t[:i], position[:,:i], velocity[:,:i], F_gravity[:,:i], F_drag_array[:,:i], v_drag_array[:,:i], rho_array[:i]

    return t, position, velocity, F_gravity, F_drag_array, v_drag_array, rho_array

quarter = 900 # 15 minutes = 900 s

t1 = time()
t, r, v, F_g, F_d, v_drag, rho = trajectory_lander(initial_time, vel, pos, 4*quarter)
t2 = time()
print('Simulation took', t2-t1, 's to complete.')

# F_tot = F_g + F_d
# fig, (ax7, ax8, ax9) = plt.subplots(3, 1, sharex=True)
# ax7.set_title('Forces plot, gravity OFF')
# ax7.plot(t[:-1], F_g[:,:-1].T)
# ax7.legend(['F_g_x', 'F_g_y'])
# ax7.set_ylabel('[N]')
#
# ax8.plot(t[:-1], F_d[:,:-1].T)
# ax8.legend(['F_d_x', 'F_d_y'])
# ax8.set_ylabel('[N]')
#
# ax9.plot(t[:-1], F_tot[:,:-1].T)
# ax9.legend(['F_tot_x', 'F_tot_y'])
# ax9.set_ylabel('[N]')
# ax9.set_xlabel('Time [s]')



# plt.plot(t, rho, label='rho')
# plt.xlabel('Time [s]')
# plt.legend()

# testF_d = [F_d[:,i]/np.linalg.norm(F_d[:,i]) for i in range(11)]
# test_v_drag = [v_drag[:,i]/np.linalg.norm(v_drag[:,i]) for i in range(11)]
#
# print('F_d:', F_d[:,:10].T)
# print('v:', v_drag[:,:10].T)
# print('F_d/|F_d|:')
# for i in range(len(testF_d)):
#     print(testF_d[i])
# print('v/|v|:')
# for i in range(len(test_v_drag)):
#     print(test_v_drag[i])

# height = np.linspace(0,200e3)
# density = get_rho(height)

# plt.plot(height, density, label='Density')
# plt.xlabel('Height [m]')
# plt.legend()
# plt.show()

# plt.figure()
#
# theta_planet = np.linspace(0, 2*np.pi, 101)
# x_planet = R * np.cos(theta_planet)
# y_planet = R * np.sin(theta_planet)
#
# plt.title('Simulation of landing')
# plt.plot(r[0,:], r[1,:], 'r', label='Position')
# plt.plot(0,0, 'ko', label='Center')
# plt.plot(x_planet, y_planet, 'b', lw=0.5, label='Surface')
# plt.axis('equal')
# plt.legend()
#
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
#
# ax1.plot(t[:-1], np.linalg.norm(v[:,:-1], axis=0), label='|v|')
# ax1.legend()
#
# ax2.plot(t, np.linalg.norm(r, axis=0) - R, 'r', label='|r|')
# ax2.legend()
#
# ax3.plot(t[:-1], np.linalg.norm(F_g[:,:-1], axis=0), label='Gravity')
# ax3.plot(t[:-1], np.linalg.norm(F_d[:,:-1], axis=0), label='Drag')
# ax3.set_xlabel('Time [s]')
# ax3.legend()
#
# fig2, (ax4, ax5) = plt.subplots(2, 1, sharex=True)
#
# ax4.plot(t[:-1], F_g[:,:-1].T)
# ax4.legend(['F_g_x', 'F_g_y'])
#
# ax5.plot(t[:-1], F_d[:,:-1].T)
# ax5.legend(['F_d_x', 'F_d_y'])
# ax5.set_xlabel('Time [s]')


plt.show()
