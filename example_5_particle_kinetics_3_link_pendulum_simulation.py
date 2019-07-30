########## example_5_particle_kinetics_3_link_pendulum_simulation.py ##########

'''

This script creates a simulation for a triple 2D pendulum (no rod friction) by harcoding in the EOM data from:
example_5_particle_kinetics_3_link_pendulum_create_eom.py

'''

##############
# dependencies
##############

from numpy import arange, linspace, sin, cos, pi, rad2deg, deg2rad
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

###########
# main data
###########

##############################
# initialize inital conditions
##############################

# define the parameters for simulation
m1, m2, m3 = 1.0, 1.0, 1.0 # [kg] set the masses of each pendulum 'mass'
L1, L2, L3 = 1.0, 1.0, 1.0 # [m] set the length of the each pendulum rod
g = 9.81 # [m/s^2] set the gravitational force constant

# set the initial state position vector
q0_1, q0_2, q0_3 = deg2rad(180.0), deg2rad(90.0), deg2rad(270.0)
q_init = [q0_1, q0_2, q0_3]

# set the initial state velocity vector
qd0_1, qd0_2, qd0_3 = 0, 0, 0
qd_init = [qd0_1, qd0_2, qd0_3]

# set the initial full state vector for the pendulum system:
state0 = [q_init, qd_init] # create a list of lists for the initial state vector
state0 = [item for sublist in state0 for item in sublist] # flatten the full state vector into a single list

# define time parameters
T = 150.0 # set the length of time of simulation
dt = 0.05 # determine the timestep for simulation
t = linspace(0.0, T, T/dt) # determine the time vector for simulation
t_span = [0, T] # determine the time span list for solving the ODE

# set the initial figure number
fgn = 1

#################################################################################################################
# create the state vector state = [pendulum positions [rad, rad, rad], pendulum velocities [rad/s, rad/s, rad/s]]
#################################################################################################################

def create_triple_pendulum_stated(t, state):

	# initialize q_dot parameters & implement the dynamic equations of motion
	qd1 = state[3] # qd1 (angular velocity state of pendulum 1) is equal to q[3] (angular velocity state of pendulum 1)
	qd2 = state[4] # qd2 (angular velocity state of pendulum 2) is equal to q[4] (angular velocity state of pendulum 2)
	qd3 = state[5] # qd3 (angular velocity state of pendulum 3) is equal to q[5] (angular velocity state of pendulum 3)
	qdd1 = -(L1*L3*m3*((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2])) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))*(-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3)))*(-g*L3*m3*sin(state[2]) + L1*L3*m3*(sin(state[0])*cos(state[2]) - sin(state[2])*cos(state[0]))*state[3]**2 + L2*L3*m3*(sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[1]))*state[4]**2)/(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(-L1**2*L3**2*m3**2*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))**2 + L3**2*m3*(L1**2*m3 + L1**2*m3 + L1**2*m3)) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2) + ((((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(-L1**2*L3**2*m3**2*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))**2 + L3**2*m3*(L1**2*m3 + L1**2*m3 + L1**2*m3)) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2)*((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (-L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) - L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2) - (-L1*L3*m3*((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2])) - (-L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) - L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))*(-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3)))*(L1*L3*m3*((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2])) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))*(-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))))*(-g*L1*m3*sin(state[0]) - g*L1*m3*sin(state[0]) - g*L1*m3*sin(state[0]) + L1*L2*m3*(-sin(state[0])*cos(state[1]) + sin(state[1])*cos(state[0]))*state[4]**2 + L1*L2*m3*(-sin(state[0])*cos(state[1]) + sin(state[1])*cos(state[0]))*state[4]**2 + L1*L3*m3*(-sin(state[0])*cos(state[2]) + sin(state[2])*cos(state[0]))*state[5]**2)/(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(-L1**2*L3**2*m3**2*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))**2 + L3**2*m3*(L1**2*m3 + L1**2*m3 + L1**2*m3)) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2)*(L1**2*m3 + L1**2*m3 + L1**2*m3)) + (-(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(-L1**2*L3**2*m3**2*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))**2 + L3**2*m3*(L1**2*m3 + L1**2*m3 + L1**2*m3)) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2)*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))*(L1**2*m3 + L1**2*m3 + L1**2*m3) + (L1*L3*m3*((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2])) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))*(-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3)))*(-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))*(L1**2*m3 + L1**2*m3 + L1**2*m3))*(-g*L2*m3*sin(state[1]) - g*L2*m3*sin(state[1]) + L1*L2*m3*(sin(state[0])*cos(state[1]) - sin(state[1])*cos(state[0]))*state[3]**2 + L1*L2*m3*(sin(state[0])*cos(state[1]) - sin(state[1])*cos(state[0]))*state[3]**2 + L2*L3*m3*(-sin(state[1])*cos(state[2]) + sin(state[2])*cos(state[1]))*state[5]**2)/(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(-L1**2*L3**2*m3**2*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))**2 + L3**2*m3*(L1**2*m3 + L1**2*m3 + L1**2*m3)) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2)*(L1**2*m3 + L1**2*m3 + L1**2*m3))
	qdd2 = -(-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))*(L1**2*m3 + L1**2*m3 + L1**2*m3)*(-g*L3*m3*sin(state[2]) + L1*L3*m3*(sin(state[0])*cos(state[2]) - sin(state[2])*cos(state[0]))*state[3]**2 + L2*L3*m3*(sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[1]))*state[4]**2)/(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(-L1**2*L3**2*m3**2*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))**2 + L3**2*m3*(L1**2*m3 + L1**2*m3 + L1**2*m3)) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2) + ((((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(-L1**2*L3**2*m3**2*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))**2 + L3**2*m3*(L1**2*m3 + L1**2*m3 + L1**2*m3)) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2)*(-L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) - L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) - (-L1*L3*m3*((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2])) - (-L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) - L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))*(-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3)))*(-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3)))*(-g*L1*m3*sin(state[0]) - g*L1*m3*sin(state[0]) - g*L1*m3*sin(state[0]) + L1*L2*m3*(-sin(state[0])*cos(state[1]) + sin(state[1])*cos(state[0]))*state[4]**2 + L1*L2*m3*(-sin(state[0])*cos(state[1]) + sin(state[1])*cos(state[0]))*state[4]**2 + L1*L3*m3*(-sin(state[0])*cos(state[2]) + sin(state[2])*cos(state[0]))*state[5]**2)/(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(-L1**2*L3**2*m3**2*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))**2 + L3**2*m3*(L1**2*m3 + L1**2*m3 + L1**2*m3)) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2)) + ((((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(-L1**2*L3**2*m3**2*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))**2 + L3**2*m3*(L1**2*m3 + L1**2*m3 + L1**2*m3)) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2)*(L1**2*m3 + L1**2*m3 + L1**2*m3) + (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2*(L1**2*m3 + L1**2*m3 + L1**2*m3))*(-g*L2*m3*sin(state[1]) - g*L2*m3*sin(state[1]) + L1*L2*m3*(sin(state[0])*cos(state[1]) - sin(state[1])*cos(state[0]))*state[3]**2 + L1*L2*m3*(sin(state[0])*cos(state[1]) - sin(state[1])*cos(state[0]))*state[3]**2 + L2*L3*m3*(-sin(state[1])*cos(state[2]) + sin(state[2])*cos(state[1]))*state[5]**2)/(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(-L1**2*L3**2*m3**2*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))**2 + L3**2*m3*(L1**2*m3 + L1**2*m3 + L1**2*m3)) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2))
	qdd3 = ((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(L1**2*m3 + L1**2*m3 + L1**2*m3)*(-g*L3*m3*sin(state[2]) + L1*L3*m3*(sin(state[0])*cos(state[2]) - sin(state[2])*cos(state[0]))*state[3]**2 + L2*L3*m3*(sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[1]))*state[4]**2)/(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(-L1**2*L3**2*m3**2*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))**2 + L3**2*m3*(L1**2*m3 + L1**2*m3 + L1**2*m3)) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2) + (-L1*L3*m3*((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2])) - (-L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) - L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))*(-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3)))*(-g*L1*m3*sin(state[0]) - g*L1*m3*sin(state[0]) - g*L1*m3*sin(state[0]) + L1*L2*m3*(-sin(state[0])*cos(state[1]) + sin(state[1])*cos(state[0]))*state[4]**2 + L1*L2*m3*(-sin(state[0])*cos(state[1]) + sin(state[1])*cos(state[0]))*state[4]**2 + L1*L3*m3*(-sin(state[0])*cos(state[2]) + sin(state[2])*cos(state[0]))*state[5]**2)/(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(-L1**2*L3**2*m3**2*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))**2 + L3**2*m3*(L1**2*m3 + L1**2*m3 + L1**2*m3)) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))*(L1**2*m3 + L1**2*m3 + L1**2*m3)*(-g*L2*m3*sin(state[1]) - g*L2*m3*sin(state[1]) + L1*L2*m3*(sin(state[0])*cos(state[1]) - sin(state[1])*cos(state[0]))*state[3]**2 + L1*L2*m3*(sin(state[0])*cos(state[1]) - sin(state[1])*cos(state[0]))*state[3]**2 + L2*L3*m3*(-sin(state[1])*cos(state[2]) + sin(state[2])*cos(state[1]))*state[5]**2)/(((L2**2*m3 + L2**2*m3)*(L1**2*m3 + L1**2*m3 + L1**2*m3) - (L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])))**2)*(-L1**2*L3**2*m3**2*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))**2 + L3**2*m3*(L1**2*m3 + L1**2*m3 + L1**2*m3)) - (-L1*L3*m3*(sin(state[0])*sin(state[2]) + cos(state[0])*cos(state[2]))*(L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1])) + L1*L2*m3*(sin(state[0])*sin(state[1]) + cos(state[0])*cos(state[1]))) + L2*L3*m3*(sin(state[1])*sin(state[2]) + cos(state[1])*cos(state[2]))*(L1**2*m3 + L1**2*m3 + L1**2*m3))**2)
	stated = [qd1, qd2, qd3, qdd1, qdd2, qdd3] # stated = [mass angular vel [rad/s], mass angular accel [rad/s^2]] for each pendulum
	return stated

# solve dq/dt for q(t) where q(t) = [angular position; angular velocity]
state = solve_ivp(create_triple_pendulum_stated, t_span, state0, t_eval = t) # integrate stated [pendulum angular velocities, pendulum angular accelerations] for state [pendulum angular positions, pendulum angular velocities]
[q1, q2, q3, qd1, qd2, qd3] = [state.y[0, :], state.y[1, :], state.y[2, :], state.y[3, :], state.y[4, :], state.y[5, :]] # extract states from the state vector
state = [q1, q2, q3, qd1, qd2, qd3] # rearrange states back into a state vector

#############################################
# plot the time responses of the state vector 
#############################################

# plot the time responses of the state vector
fig = plt.figure(num = ('Figure '+ str(fgn) + ': Time Responses for the State Vector')) # create a blank figure and change its title
plt.plot(t, q1, label = 'Angular position θ(t) [rad] for pendulum 1') # plot the angular position q1(t)
plt.plot(t, qd1, label = 'Angular velocity θ_dot(t) [rad/sec] for pendulum 1') # plot the angular velocity qd1(t)
plt.plot(t, q2, label = 'Angular position θ(t) [rad] for pendulum 2') # plot the angular position q2(t)
plt.plot(t, qd2, label = 'Angular velocity θ_dot(t) [rad/sec] for pendulum 2') # plot the angular velocity qd2(t)
plt.plot(t, q3, label = 'Angular position θ(t) [rad] for pendulum 3') # plot the angular position q3(t)
plt.plot(t, qd3, label = 'Angular velocity θ_dot(t) [rad/sec] for pendulum 3') # plot the angular velocity qd3(t)
plt.grid() # turn on the plot grid
plt.title('Time Responses for the State Vector') # create a plot title
plt.xlabel('Time(s)') # label the x axis
plt.ylabel('Angular Disp.(rad) and Angular Vel.(rad/s)') # label the y axis
plt.legend() # create a plot legend
fgn += 1 # incrememnt the figure number
plt.show() # show the current figure

################################################
# animate the time responses of the state vector
################################################

# define the x and y position of pendulum 1
x1 = L1*sin(q1) # x position
y1 = -L1*cos(q1) # y position

# determine the x and y position of pendulum 2
x2 = L2*sin(q2) + x1 # x position
y2 = -L2*cos(q2) + y1 # y position

# determine the x and y position of pendulum 3
x3 = L3*sin(q3) + x2 # x position
y3 = -L3*cos(q3) + y2 # y position

# initalize the animation figure
fig = plt.figure(num = ('Figure '+ str(fgn) + ': Animation of the Time Responses for the State Vector')) # create a blank figure and change the figure name
ax = fig.add_subplot(111, xlim=(-3, 3), ylim=(-3, 3)) # create a blank plot on the figure with certain axes limits
ax.grid() # turn on the plot grid
line, = ax.plot([], [], 'o-', lw=2) # initialize the line to be animated
time_template = 'Elapsed Time = %.1fs' # initialized the time counter on the plot
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes) # set the position of the time counter on the plot

# create an animation function to create the actual animation
def animate(i):
    xpos = [0, x1[i], x2[i], x3[i]] # the system is anchored at the origin, and is updated to x positions according to the state vector q
    ypos = [0, y1[i], y2[i], y3[i]] # the system is anchored at the origin, and is updated to y positions according to the state vector q
    line.set_data(xpos, ypos) # unpack position data into out initialized animation line
    time_text.set_text(time_template % (i*dt)) # update the time counter
    return line, time_text

# call the animation function
animation = animation.FuncAnimation(fig, animate, interval=16.67) # write the animation function to the animation figure, set a 60Hz update interval
plt.show() # show the current figure

