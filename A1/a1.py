"""
ENGG1001 Assignment 1
Semester 2, 2022
"""
import math
from a1_support import *


# Fill these in with your details

__author__ = "Vu Khanh Vy Ho"
__email__ = "vy.ho18032002@gmail.com"
__date__ = "7/12/2022"


# Write your functions here

def specify_inputs():
    length_of_runway = float(input("Input the length of the runway (km): "))
    slope_of_runway = float(input("Input the slope of the runway (rad): "))
    mass_of_the_plane = float(input("Input the mass of the plane (in kg): "))
    engine_thrust = float(input("Input the engine thrust (in N): "))
    reference_area = float(input("Input the reference area (in m^2): "))
    lift_off_velocity = float(input("Input the lift-off velocity (m/s): "))
    drag_coefficient = float(input("Input the drag coefficient: "))
    air_density = float(input("Input the air density (in kg/m^3): "))
    initial_velocity = float(input("Input the initial velocity (v_0) at start of runway (in m/s): "))
    position = float(input("Input the position (x_0) at the start of the runway (in m): "))
    time_increment = float(input("Input the time increment (in secs): "))
    runway_attributes = (length_of_runway, slope_of_runway)
    plane_attributes = (mass_of_the_plane, engine_thrust, reference_area, lift_off_velocity, drag_coefficient)
    inputs = (air_density, initial_velocity, position, time_increment)
    return runway_attributes, plane_attributes, inputs
def calculate_acceleration(plane_attributes, density, velocity, slope):
    F_gravity = plane_attributes[0]*ACC_DUE_TO_GRAV*math.sin(slope)
    F_drag = 0.5*density*velocity*velocity*plane_attributes[2]*plane_attributes[-1]
    final = plane_attributes[1] + F_gravity - F_drag
    return (1/plane_attributes[0]) * final
def calculate_motion(runway_attributes, plane_attributes, inputs):
    velocities = []
    velocities.append(inputs[1])
    positions = []
    positions.append(inputs[2])
    suitability = False
    counter = 0
    while velocities[counter] < plane_attributes[3]:
        a = calculate_acceleration(plane_attributes, inputs[0], velocities[counter], runway_attributes[1])
        v = velocities[counter] + a*inputs[3]
        p = positions[counter] + velocities[counter] * inputs[3] + 0.5*a*inputs[3]*inputs[3]
        velocities.append(round(v,3))
        positions.append(round(p,3))
        counter += 1
    if positions[counter] < (positions[counter]/(runway_attributes[0])) * 100:
        suitability = True
    return velocities, positions, suitability
def print_table(runway_test_data, plane_test_data, inputs, start_column, end_column):
    num = end_column - start_column + 2
    print("#" * (19 * num + num + 1))
    line_1 = f"{'#':^1} {'Plane number':^17} {'#':^1}"
    for i in range (start_column, end_column + 1):
        line_1 += str(f"{' Runway ' + str(i) + ' distance':^18} {'#':^1}")
    print(line_1)
    print("#" * (19 * num + num + 1))
    for i in range(0,len(runway_test_data)):
        line = f"{'#':^1} {str(i):^17} {'#':^1}"
        counter = 0
        while counter < len(runway_test_data):
            velocities, positions, suitability = calculate_motion(runway_test_data[counter], plane_test_data[i],inputs)
            if suitability is True: 
                suitability = "T"
            else:
                suitability = "F"
            line += f"{str(positions[len(positions) - 1]):^16} {str(suitability):^1} {'#':^1}"
            counter += 1
        print(line)
