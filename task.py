import numpy as np
from physics_sim import PhysicsSim 

def distance_reward(d, dmax=300):
    """
    distance based reward function
    d: distance to the target
    dmax: maximum distance (ref: PhysicsSim's env_bounds)
    """
    # decrease from 1 to 0 within dmax, 
    # slope gets sharper when d approaches 0.
    reward = 1 - (d / dmax)**0.4
    return reward


def time_reward(t, tmax=5):
    """
    time based reward function
    t: time spent
    tmax: maximum runtime (ref: PhysicsSim's runtime)
    """
    # linearly decrease from 1 to 0 within tmax
    reward = 1 - t / tmax
    return reward


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, obs='pos', init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            obs: 'pos', 'pos_v', 'pos_v_ang'
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # make an sim object
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.obs = obs
        self.action_repeat = 3
        
        # task's observation size
        if self.obs == 'pos':
            s_len = 6 # position [x, y, z, phi, theta, psi]
        elif self.obs == 'pos_v':
            s_len = 9 # position + velocity [vx, vy, vz] 
        elif self.obs == 'pos_v_ang': 
            s_len = 12 # position + velocity + angular velocity [vphi, vtheta, vpsi]
        self.state_size = self.action_repeat * s_len
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4 # 4 roter speeds
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
    def get_reward(self):
        """
        Reward function for task "takeoff"
        """
        ## difference vector 
        delta = self.sim.pose[:3] - self.target_pos 
        
        ## Euclidean distance to the target
        dtotal = np.sqrt(np.dot(delta, delta)) 
        
        ## abs distance in z (height) to the target
        dz = np.abs(delta[2])  
        
        ## reward dtotal (300 m is the env bound of the sim)
        reward = distance_reward(dtotal, dmax=np.sqrt(300**2 + 300**2 + 300**2)) \
                * time_reward(self.sim.time, tmax=self.sim.runtime)
        
        ## success condition: height 
        if dz < 1:  
            self.sim.done = True # end the episode. (no further thing to learn if target is reached.)
            reward += 100   # an extra big reward. (to avoid whirling around the target for collecting positive reward)
       
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        stmp = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            if self.obs == 'pos':
                s = np.array(self.sim.pose) # shape (6,)
            elif self.obs == 'pos_v':
                s = np.concatenate((self.sim.pose, self.sim.v)) # shape (9,)
            elif self.obs == 'pos_v_ang':
                s = np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v)) # shape (12,) 
            stmp.append(s) 
        next_state = np.concatenate(stmp)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        if self.obs == 'pos':
            s = np.array(self.sim.pose) # shape (6,)
        elif self.obs =='pos_v':
            s = np.concatenate((self.sim.pose, self.sim.v)) # shape (9,)
        elif self.obs == 'pos_v_ang':
            s = np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v)) # shape (12,) 
        state = np.concatenate([s] * self.action_repeat)
        return state