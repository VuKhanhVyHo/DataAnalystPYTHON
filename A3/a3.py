"""
Assignment 3
Semester 2, 2022
ENGG1001
"""

# NOTE: Do not import any other libraries!
import math
import numpy as np
import matplotlib.pyplot as plt
import random as rand

from numpy import ndarray


# Replace these <strings> with your name, student number and email address.
__author__ = "Ho Vu Khanh Vy, 47597868"
__email__ = "vy.ho18032002@gmail.com"

def create_animation(pos, ax):
    """ Create an animation of vehicles on the roadway 
    Parameters:
        pos: array of vehicle positions
        ax: axis object 
    """    

    Nveh = pos.shape[0]
    Nstep = pos.shape[1]
    l_road = np.nanmax(pos) 
    
    for ts in range(0, Nstep):
        ax.cla()
        ax.plot([-3, -3], [0, l_road], c="black")
        ax.plot([3, 3], [0, l_road], c="black")
        ax.set_xlim(-50, 50)
        for i in range(0, Nveh):
            if np.isnan(pos[i, ts]) == 0:
                cur_pos = pos[i, ts]
                ax.plot([-1, 1, 1, -1, -1], [cur_pos-4, cur_pos-4, cur_pos, cur_pos, cur_pos-4], c="blue")

        
        plt.text(-40, l_road/2, 'time='+ f"{ts/10}"+'s')
        plt.xticks([])
        plt.ylabel('space [m]')
        plt.draw()
        
        plt.pause(0.001)


# COMPUTE AND RETURN PARAMETERS OF VEHICLE I IN THE NEXT TIME STEP
def vehicle_update(x,v,a,front_x, front_v, dt, ts):
    """
    Parameters:
        x: position of vehicle i [m]
        v: speed of vehivle i [m/s]
        a: acceleration of vehicle i [m/s2]
        front_x: position of the vehicle in front, i-1 [m]
        front_v: speed of the vehicle in front, i-1 [m/s]
        dt: time step [s]
        ts: time of simulation [s]
    Returns:
        x: position of vehicle i in the next time step [m]
        v: speed of vehicle i in the next time step [m.s]
        a: acceleration of vehicle i in the next time step [m/s2]
    """
    l = 6.0 # vehicle length [m]
    s0 = 4.0 # max desired spacing [m]
    T = 1.0 # reaction time [s]
    v_max = 19.44 # maximum desired speed [m/s]
    a_max = 1.50 # maximum acceleration [m/s2]
    b_max = 4.10 # maximum comfortable deceleration [m/s2]

    if (v + a*dt) < 0:
        x -= 1/2 * (v**2/a)
        v = 0.0

    else:
        x += v*dt + (a*(dt)**2)/2
        v += a*dt

    if np.isnan(front_x) == True:
        if ts >= 30 and ts <60:
            a = -b_max * (v/v_max)
        else:
            a = a_max * (1 - (v/v_max)**4)
            
    else:
        delta_v = v - front_v
        s_i = front_x - x - l
        s_desired = s0 + v*T + v*delta_v/(2*np.sqrt(a_max*b_max))
        a = a_max * (1 - (v/v_max)**4 - (s_desired/s_i)**2)
    
    return (x, v, a)

#COMPUTE AND UPDATES THE POSITION, VELOCITY AND ACCELERATION OF ALL VEHICLES ON THE ROADWAY SECTION
def road_update(vehs_x, vehs_v, vehs_a, dt, ts, l_road):
    """
    Parameters:
        vehs_x: array of vehivle positions [m] - (Nveh x 1)
        vehs_v: array of vehicle speeds [m/s] - (Nveh x 1)
        vehs_a: array of vehicle accelerations [m/s2] - (Nveh x 1)
        dt: time step [s]
        ts: time of simulation [s]
        l_road: roadway length[m]
    Returns:
        none
    """
    for i in range(0, len(vehs_x)):
        if i == 0:
            front_x = np.nan
            front_v = np.nan
           
        else:
           front_x = vehs_x[i-1]
           front_v = vehs_v[i-1]                                        
        vehs_x[i], vehs_v[i], vehs_a[i] = vehicle_update(vehs_x[i],
                                                         vehs_v[i],
                                                         vehs_a[i],
                                                         front_x,
                                                         front_v,
                                                         dt, ts)
        
        if vehs_x[i] > l_road:
            vehs_x[i] = np.nan
            vehs_v[i] = np.nan
            vehs_a[i] = np.nan

#COMPUTES AND RETURNS POSITIONS AND VELOCITIES OF ALL VEHI
def simulation(dt, Tsim, Nveh, hway, l_road):
    """
    Parameters:
        dt: time step [s]
        Tsim: total simulation time [s]
        Nveh: total number of vehicles [veh]
        hway: time gap between vehicle entries [number of time steps]
        l_road: roadway length [m]
    Returns:
        pos: array of updated vehicle positions at all time steps [m] - (Nveh x Nstep)
        speed: array of updated vehivle speeds at all time steps [m/s] - (Nveh x Nstep)
    """
    v_max = 19.44 # maximum desired speed [m/s]
    Nstep = int(Tsim / dt)
    pos = np.full((Nveh, Nstep), np.nan)
    speed = np.full((Nveh, Nstep), np.nan)
    a = np.full((Nveh, Nstep), np.nan)
    counter = 0
    for i, t in enumerate(np.arange(0, Tsim, dt)):
        if i != 0:
            pos[:,i] = pos[:, i-1]
            speed[:,i] = speed[:, i-1]
            a[:,i] = a[:, i-1]
        if i % hway == 0 and counter < Nveh:
            pos[counter,i] = 0
            speed[counter,i] = v_max
            a[counter,i] = 0
            counter += 1 
        road_update(pos[:, i], speed[:, i], a[:, i], dt, dt*i, l_road)
    return pos, speed

#PLOTTING THE VEHICLE POSITIONS OVER TIME
def plot_positions(pos: np.array):
    """
    Parameters:
    pos: array of updated vehicle positions at all time steps [m] - (Nveh x Nstep)
    """
    fig, axs = plt.subplots()
    axs.plot(pos.T)
    axs.set(xlabel = "time [step]", 
            ylabel = "position [m]")
    fig.tight_layout()
    plt.show()

#REPRESENT THE VEHICLE
class Vehicle:
    def __init__(self, vehid, l, s0, T, v_max, a_max, b_max):
        """
        Parameters:
            l: vehicle length [m]
            s0: max desired spacing [m]
            T: reaction time [s]
            v_max: maximum desired speed [m/s]
            a_max: maximum acceleration[m/s2]
            b_max: maximum comfortable deceleration [m/s2]
            vehid: vehicle id based on the order of entry, 
                e.g., the id of the first vehicle is 0, the second is 1, etc. 
        Returns:
            none
        """
        self.vehid = vehid
        self.l = l 
        self.s0 = s0
        self.T = T
        self.v_max = v_max + rand.uniform(-3,3)
        self.a_max = a_max + rand.uniform(-0.5, 0.5)
        self.v = self.v_max
        self.a = 0
        self.x = 0
        self.b_max = b_max
        self.stoppingZone = False
    
    def update(self, front, dt):
        """
        Parameters:
            front (vehicle): vehicle in front,
            dt: time step [s]
        Returns:
            none
        """
        if self.v + self.a * dt < 0:
            self.x = self.x - 1/2 * ((self.v**2)/(self.a))
            self.v = 0.0

        self.x = self.x + self.v*dt + self.a*(dt ** 2)/2
        self.v = self.v + self.a*dt
        if front is None:
            if (self.stoppingZone is True):
                self.a = -self.b_max*(self.v/self.v_max)
            else:
                s_desired = 0
        else:
            si= front - self.x - self.l
            s_desired = self.s0 + self.v*self.T +(self.v * (self.v - front))/(2*np.sqrt(self.a_max * self.b_max))
            s_desired /= si
            self.a = self.a_max*(1-(self.v/self.v_max)**4 - (s_desired)**2)
        

    def stop(self):
        """
        Returns: 
            none
        """
        self.stoppingZone = True

    def unstop(self):
        """
        Returns:
            none
        """
        self.stoppingZone = False
    
    def get_pos(self):
        """
        Returns:
            (float): position of the vehicle
        """
        return self.x
    
    def get_speed(self):
        """
        Returns:
            (float): speed of the vehicle
        """
        return self.v
    
    def get_length(self):
        """
        Returns:
            (float): length of the vehicle
        """
        return self.l
    
    def get_id(self):
        """
        Returns:
            (int): id of the vehicle
        """
        return self.vehid
    
class Road:
    def __init__(self, length):
        """
        Parameters:
            length: length of roadway section [m]
        Returns:
            none
        """
        self.l = length
        self.listOfVehicle = []
    
    def update(self, dt):
        """
        Parameters:
            dt: time step [s]
        Returns:
            none
        """ 
        for i in range(0,len(self.listOfVehicle)):
            if len(self.listOfVehicle == 0):
                front = None
            else:
                front = self.listOfVehicle[i-1]
                
            self.listOfVehicle[i].update(front, dt)
    
        if len(self.listOfVehicle) > 0:
            if self.listOfVehicle[0].get_pos() > self.l:
                self._vehicles.remove(self.listOfVehicle[0])
    
    def add_vehicle(self, veh):
        """
        Parameters:
            veh: vehicle object
        Returns: none
        """
        self.listOfVehicle.append(veh)
    
class Simulation:
    def __init__(self, dt,Tsim, Nveh, hway, l_road, l, s0, T, v_max, a_max, b_max):
        """
        Parameters:
            dt: time step [s]
            Tsim: total simulation time [s]
            Nveh: total number of vehivles [veh]
            hway: time gap between vehicle entries [number of time steps]
            l_road: roadway length [m]
            l: vehicle length [m]
            s0: max desired spacing [m]
            T: reaction time [s]
            v_max: maximum desired speed [m/s]
            a_max: maximum acceleration [m/s2]
            b_max: maximum comfortable decleration [m/s2]
        Returns:
            none
        """
        self.dt = dt
        self.Tsim = Tsim
        self.Nveh = Nveh
        self.hway = hway
        self.l_road = l_road
        self.l = l
        self.s0 = s0
        self.T = T
        self.v_max = v_max
        self.a_max = a_max
        self.b_max = b_max 
    
    def run(self):
        """
        Parameters:
            none
        Returns:
            pos (ndarray): array of updated vehicle positions at all time steps [m] - (Nveh x Tstep)
            spee (ndarray): array of updated vehicle speeds at all time steps [m/s] - (Nveh x Tstep)
        """
        Nstep = int(self.Tsim / self.dt)
        pos = np.full((self.Nveh, Nstep), np.nan)
        speed = np.full((self.Nveh, Nstep), np.nan)
        a = np.full((self.Nveh, Nstep), np.nan)
        counter = 0
        for i, t in enumerate(np.arange(0, self.Tsim, self.dt)):
            if i != 0:
                pos[:,i] = pos[:, i-1]
                speed[:,i] = speed[:, i-1]
                a[:,i] = a[:, i-1]
            if i % self.hway == 0 and counter < self.Nveh:
                pos[counter,i] = 0
                speed[counter,i] = self.v_max
                a[counter,i] = 0
                counter += 1 
            road_update(pos[:, i], speed[:, i], a[:, i], self.dt, self.dt*i, self.l_road)
        return pos, speed

def main() -> None:
    """Entry point to interaction"""
    print("Implement your solution and run this file")
    x, v, a = vehicle_update(300,19.44,0,np.nan,np.nan,0.1,30)
    print(str(x) + ' ' + str(v) + ' ' + str(a))
    vehs_x = np.array([115.0, 85.0, 45.0])
    vehs_v = np.array([19.44, 18.0, 16.0])
    vehs_a = np.array([0, 0.5, 1])
    dt = 0.1
    ts = 100
    l_road = 200
    road_update(vehs_x, vehs_v, vehs_a, dt, ts, l_road)
    vehs_x = np.array([199.0, 85.0, 45.0])
    vehs_v = np.array([19.44, 18.0, 16.0])
    vehs_a = np.array([0, 0.5, 1])
    dt = 0.1
    ts = 100
    l_road = 200
    road_update(vehs_x, vehs_v, vehs_a, dt, ts, l_road)
    dt = 0.1
    Tsim = 120 
    Nveh = 10 
    hway = 40 
    l_road = 1000
    pos, speed = simulation(dt, Tsim, Nveh, hway, l_road)
    plot_positions(pos)
    fig, ax = plt.subplots()
    create_animation(pos, ax)



if __name__ == "__main__":
    main()
