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
shortcut.place_spacecraft_in_stable_orbit(2, 400e3, 0, 6)

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

'''Challenge A'''

initial_time, pos, vel = landing.orient()

A_lander = mission.lander_area     # cross section area of the lander in m^2
A_craft = mission.spacecraft_area # cross section area of the craft in m^2
A_parachute = 70 # m^2
A_lander_parachute = A_lander + A_parachute
R = system.radii[6] * 1000 # m
m_planet = system.masses[6] * const.m_sun # kg
G = const.G # gravitational constant
g_1 = G * m_planet / R**2 # gravitational acceleration on our planet 3.174917721996738 m/s^2
m_lander = mission.lander_mass # kg
m_craft = mission.spacecraft_mass # kg
T = system.rotational_periods[6] * 3600 * 24 # s

# print('')
# print('Specs for craft and lander:')
# print('')
# print('\t\t Spacecraft \t Lander')
# print(f'Area [m^2]: \t {A_craft} \t\t {A_lander}')
# print(f'Mass [kg]: \t {m_craft} \t {m_lander}')
# print('')
'''
Specs for craft and lander:

                 Spacecraft      Lander
Area [m^2]:      16.0            0.3
Mass [kg]:       1100.0          90.0
'''

def terminal_velocity(m, position, area, C_d=1, v_safe=None):
    ''' Function to calculate terminal velocity or parachute area'''

    r = np.linalg.norm(position)
    g = G * m_planet / r**2
    r_test = r - R

    if r_test <= r_shift_adiabatic_isoterm:
        rho = A**(-1/gamma)*K0 * (((1 - gamma) / gamma) * (K0*A**(-1/gamma)*g_1*r_test + C))**(1 / (gamma - 1))
    else:
        rho = C0 * np.exp(-1/K1*r_test)

    v_terminal = np.sqrt(2 * m * g / (rho * C_d * area))

    if v_safe == 3:
        parachute_area = 2 * m * g / (get_rho(0) * C_d * v_safe**2)
        return parachute_area
    else:
        parachute_area = 2 * m * g / (rho * C_d * v_terminal**2)

    return v_terminal, parachute_area

# print('')
# print(f'Estimated parachute area: {terminal_velocity(m_lander, [R,0,0], A_lander, 1, 3)} m^2')
'''
Estimated parachute area: 60.585680617431905 m^2
'''

'''Challenge B'''

@njit
def F_drag(position, velocity, area, C_d=1):
    '''
    This function takes cartesian coordinates x and y and
    transforms them into polar coordinates. The calculations
    are done, and then the coords. are being transformed back
    into cartesian coordinates.
    '''
    x, y = position
    vx, vy = velocity

    r = np.sqrt(x**2 + y**2)
    r_theta = np.arctan(y / x)

    # here im using the values for r and r_theta and derive these
    # to get v_r and v_theta
    vr = (x * vx + y * vy) / np.sqrt(x**2 + y**2)
    v_theta = r * ((x * vy - vx * y) / (x**2 + y**2))

    omega = 2 * np.pi / T   # angular velocity of the planet

    v = np.array([vr, v_theta])
    w = np.array([0, omega * r])
    v_drag = v - w      # crafts velocity relative to the atmosphere

    r_test = r - R      # since the formula for density has 0 as the planet surface, we test for the height above the surface

    if r_test <= r_shift_adiabatic_isoterm:
        rho = A**(-1/gamma)*K0 * (((1 - gamma) / gamma) * (K0*A**(-1/gamma)*g_1*r_test + C))**(1 / (gamma - 1))
    else:
        rho = C0 * np.exp(-1/K1*r_test)

    F_d = - 1/2 * rho * C_d * area * v_drag * np.linalg.norm(v_drag)

    F_d_r = F_d[0]
    F_d_theta = F_d[1]
    # transformation back to cartesian
    F_d_x = F_d_r * np.cos(r_theta) - F_d_theta * np.sin(r_theta)
    F_d_y = F_d_r * np.sin(r_theta) + F_d_theta * np.cos(r_theta)
    F_d_cartesian = np.array([F_d_x, F_d_y])

    return F_d_cartesian

@njit
def trajectory_lander(initial_time, initial_velocity, initial_position, simulation_time):

    dt = 0.01
    time_steps = int(np.ceil(simulation_time / dt))

    position = np.zeros((2, time_steps))
    velocity = np.zeros((2, time_steps))
    total_drag_preassure = np.zeros(time_steps)
    drag_force = np.zeros((2, time_steps))
    t = np.linspace(initial_time, initial_time + simulation_time, time_steps)

    position[:,0] = initial_position[:2]
    velocity[:,0] = initial_velocity[:2]

    print('----------------------------------------')
    print('LANDING SIMULATION STARTED')
    print('')
    print('Boosted with dv=', 0.8*position[:,0], 'at time t=', initial_time, 's.')

    for i in range(time_steps-1):

        r_vec = position[:,i]
        r_norm = np.linalg.norm(r_vec)
        unit_r = r_vec / r_norm

        g_acceleration = G * m_planet / r_norm**2

        if t[i] > 450:
            if t[i] < 450 + dt:     # launches lander with dv
                velocity[:,i] *= 0.25
                print('')
                print('Lander launched with dv= 0.25*vel=', velocity[:,i] / 4, 'm/s at t=', int(t[i]), 's.')

            F_G = -m_lander * g_acceleration * unit_r

            if t[i] > 2500:
                if t[i] < 2500 + dt:    # opens parachute
                    print('')
                    print('Opened parachute at t=', int(t[i]), 's at height', int(np.linalg.norm(position[:,i]) - R), 'm.')

                F_d = F_drag(position[:,i], velocity[:,i], A_lander_parachute)
                drag_force[:,i+1] = F_d
                total_drag_preassure[i+1] = np.linalg.norm(F_d) / A_lander

                if np.linalg.norm(F_d / A_lander) >= 1e7:
                    print('')
                    print('FATAL ERROR!')
                    print('Total drag preassure on the parachute is', np.linalg.norm(F_d / A_lander), 'Pa, but can not exceed 10 000 000 Pa.')
                    return t[:i], position[:,:i], velocity[:,:i], total_drag_preassure[:i], drag_force[:,:i]
            else:
                F_d = F_drag(position[:,i], velocity[:,i], A_lander)
                if np.linalg.norm(F_d) >= 25e4:
                    print('')
                    print('FATAL ERROR!')
                    print('Drag force on lander is', np.linalg.norm(F_d), 'N, but can not exceed 250 000 N.')
                    return t[:i], position[:,:i], velocity[:,:i], total_drag_preassure[:i], drag_force[:,:i]
                total_drag_preassure[i+1] = np.linalg.norm(F_d) / A_lander
        else:
            F_G = -m_craft * g_acceleration * unit_r
            F_d = F_drag(position[:,i], velocity[:,i], A_craft)

        F_tot = F_G + F_d

        a = F_tot / m_craft

        # Euler-Cromer
        velocity[:,i+1] = velocity[:,i] + a * dt
        position[:,i+1] = position[:,i] + velocity[:,i+1] * dt

        if np.linalg.norm(position[:,i+1]) <= R:
            print('')
            print('Ground hit after', t[i+1], 's. Good luck.')
            print('')
            return t[:i], position[:,:i], velocity[:,:i], total_drag_preassure[:i], drag_force[:,:i]

    return t, position, velocity, total_drag_preassure, drag_force

quarter = 900 # 15 minutes = 900 s

t1 = time()
t, r, v, drag_preassure, F_drag = trajectory_lander(initial_time, 0.8*vel, pos, 25*quarter)
t2 = time()
print('Simulation took', t2-t1, 's to complete.')
# print(f'Height above ground after t={t[-1]} s: {np.linalg.norm(r[:,-1]) - R} m.')

r_radial = np.sqrt(r[0,:]**2 + r[1,:]**2)
r_theta = np.arctan(r[1,:] / r[0,:])

vr = (r[0,:] * v[0,:] + r[1,:] * v[1,:]) / np.sqrt(r[0,:]**2 + r[1,:]**2)
v_theta = r_radial * ((r[0,:] * v[1,:] - v[0,:] * r[1,:]) / (r[0,:]**2 + r[1,:]**2))

print('')
if abs(vr[-1]) < 3:
    print(f'Soft landing performed, with final radial velocity v_r={vr[-1]} m/s.')

else:
    print(f'Hard landing performed, with final radial velocity v_r={vr[-1]} m/s. Hope you braced.')

plt.figure()

plt.title('Radial velocity')
plt.plot(t, vr, label='v_r', color='k')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
# plt.savefig('radial_velocity.png')

plt.figure()

theta_planet = np.linspace(0, 2*np.pi, 1001)
x_planet = R * np.cos(theta_planet)
y_planet = R * np.sin(theta_planet)

plt.title('Simulation of landing')
plt.plot(r[0,:], r[1,:], 'r', label='Position')
plt.plot(0,0, 'ko', label='Center')
plt.plot(x_planet, y_planet, 'b', lw=0.5, label='Surface')
plt.axis('equal')
plt.legend()
# plt.savefig('simulation_of_landing.png')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.set_title('Drag preassure on lander')
ax1.plot(t, drag_preassure, label='P')
ax1.legend()
ax1.set_ylabel('Preassure [Pa]')

ax2.set_title('Force on parachute')
ax2.plot(t, np.linalg.norm(F_drag, axis=0), label='Drag')
ax2.legend()
ax2.set_ylabel('Force [N]')
ax2.set_xlabel('Time [s]')
# plt.savefig('preassure_and_forces.png')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

ax1.set_title('Height above surface')
ax1.plot(t, np.linalg.norm(r, axis=0) - R, label='h', color='r')
ax1.legend()
ax1.set_ylabel('Height [m]')

ax2.set_title('Velocity')
ax2.plot(t, v.T)
ax2.legend(['vx', 'vy'])
ax2.set_ylabel('Velocity [m/s]')

ax3.set_title('Absolute velocity')
ax3.plot(t, np.linalg.norm(v, axis=0), label='|v|', color='k')
ax3.legend()
ax3.set_ylabel('Velocity [m/s]')
ax3.set_xlabel('Time [s]')
# plt.savefig('three_plots.png')

plt.show()

'''
----------------------------------------
LANDING SIMULATION STARTED

Boosted with dv= [1809394.87540424       0.        ] at time t= 0 s.

Lander launched with dv= 0.25*vel= [-60.58733923  99.20488693] m/s at t= 450 s.

Opened parachute at t= 2500 s at height 40147 m.

Ground hit after 14512.786450127312 s. Good luck.

Simulation took 20.354193925857544 s to complete.

Soft landing performed, with final radial velocity v_r=-2.784942675738035 m/s.
'''

'''Challenge D'''

# landing_spot_time = 33412.2
landing_spot_initial_coords = np.array([887581.40969006, 2080375.27433583, 0])

'''The following function is copied from Part 6.'''
def landing_site_coordinates(coords, time_elapsed):
    '''
    Function to calculate new coordinates.
    To get from cartesian to spherical coords:

    x = rho sin(theta) cos(phi)
    y = rho sin(theta) sin(phi)
    z = rho cos(theta)

    where

        0 <   rho     <= infinity
        0 <=  theta   <= pi
        0 <=  phi     <= 2pi
    '''

    x, y, z = coords

    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan(y / x)
    theta = np.arccos(z / rho)

    omega = 2 * np.pi / T

    phi_new = phi + omega * time_elapsed
    rho_new = R

    new_coords = np.array([rho_new, theta, phi_new])

    return new_coords

landing.look_in_direction_of_motion()
# landing.start_video()
landing.adjust_parachute_area(70)
landing.boost(-0.8*vel)
landing.fall(450)

t, pos, vel = landing.orient()

landing.launch_lander(0.25*vel)
landing.fall_until_time(1200)
landing.deploy_parachute()
landing.fall_until_time(1400)
# landing.finish_video()
landing.fall_until_time(1561)

landing_site = landing_site_coordinates(landing_spot_initial_coords, 2377)

print('Landing site coordinates as we calculated:')
print(f'\t theta = {landing_site[1] * 180 / np.pi} deg')
print(f'\t phi = {landing_site[2] * 180 / np.pi:.5f} deg')

'''
Performed automatic orientation:
Time: 0 s
Position: (2.26174e+06, 0, 0) m
Velocity: (0, 2205.84, 0) m/s
Camera pointing towards direction of motion.
Parachute area: 70 m^2
Spacecraft boosted with delta-v (-0, -1764.67, -0) m/s.
Spacecraft fell for 450 s.
Performed automatic orientation:
Time: 450 s
Position: (2.03688e+06, 191635, 0) m
Velocity: (-1033.03, 392.673, 0) m/s
Landing module launched at time 450 s with delta-v (-258.257, 98.1683, 0) m/s.
Lander fell until time 1200 s.
Parachute with area 70 m^2 deployed at time 1200 s.
Lander fell until time 1400 s.
Lander reached the surface at time 1560.72 s.
Successfully landed on planet 6 at time 1560.72 s with velocity 2.79098 m/s. Well done!
*** Achievement unlocked: Touchdown! ***
Landing site coordinates recorded:
  theta = 90 deg
  phi = 7.09578 deg
Lander rested on surface until time 1561 s.
Landing site coordinates as we calculated:
         theta = 90.0 deg
         phi = 67.78303 deg
'''
