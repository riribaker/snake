import numpy as np
from numpy.core.fromnumeric import argmin
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        # At step t, the agent is in current state s and chooses an optimal action a using the learned values of Q(s,a)
        # iterate through possible actions, find max Q, in case of tie: RIGHT > LEFT > DOWN > UP (so choose current over previous)
        
        # Determine reward from current environment
        reward = -0.1
        if points > self.points: reward = 1
        if dead: reward = -1
        
        self.points = points
        s_prime = self.generate_state(environment)          # get state representation of environment at t+1 (current)
        
        # at t = 1: Training -> exploration factored into determining best action to take
        if(self.s != None and self.a != None and self._train):
            
            self.N[self.s][self.a] +=1      # update N-table for this (previous) state
            
            Qmaxaction = self.choose_action(s_prime, False)
            Qmax = self.Q[s_prime][Qmaxaction]
            alpha = self.C / (self.C + self.N[self.s][self.a])   # learning rate
            newQ = self.Q[self.s][self.a] + alpha *(reward + (self.gamma*Qmax) - self.Q[self.s][self.a])
            self.Q[self.s][self.a] = newQ  # update Q-table
            
        if(self._train):   
            astar = self.choose_action(s_prime, True)
        # when done with training or t = 0, choose best action based on max Q value
        else:
            astar = self.choose_action(s_prime,False)
        
        if dead:
            self.reset()
            return astar
        
        self.s = s_prime
        self.a = astar
        return astar
    
    def choose_action(self, s_prime, explore):
        # tie breaker = RIGHT > LEFT > DOWN > UP
        astar = -1
        maxQ =  float('-inf') #self.Q[s_prime][astar]
        
        if not explore:
            for i in self.actions:
                Q = self.Q[s_prime][i]
                if Q >= maxQ and i >= astar:
                    astar = i
                    maxQ = Q
            return astar
        
        # with exploring
        for i in self.actions:
            N = self.N[s_prime][i]
            Q = self.Q[s_prime][i]
            if N < self.Ne : Q = 1
            if Q >= maxQ and i >= astar: 
                astar = i
                maxQ = Q
        return astar
        
                 
        
    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        # Environment = snake_head_x, snake_head_y, snake_body, food_x, food_y]
        # State = ( food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, 
        #           adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        snake_head_x, snake_head_y, snake_body,food_x,food_y = environment
        
        # Find direction of food relative to snakes head -> food_dir_x, food_dir_y
        food_dir_x = 0 if food_x == snake_head_x else 1     # 0 if same coord
        food_dir_y = 0  if food_y == snake_head_y else 1    # 0 if same coord
        if(food_x > snake_head_x): food_dir_x = 2         # stays 1 if food to left of head, 2 for right
        if(food_y > snake_head_y): food_dir_y = 2         # stays 1 if food to top of head,  2 for bottom
        
        # Determine if adjoining wall next to snake head
        adjoining_wall_x = 1 if snake_head_x == utils.WALL_SIZE else 0                                      # 1 if wall to left
        adjoining_wall_y = 1 if snake_head_y == utils.WALL_SIZE else 0                                      # 1 if wall above
        if(snake_head_x + utils.GRID_SIZE == utils.DISPLAY_SIZE - utils.WALL_SIZE): adjoining_wall_x = 2    # 2 if wall to right
        if(snake_head_y + utils.GRID_SIZE == utils.DISPLAY_SIZE - utils.WALL_SIZE): adjoining_wall_y = 2    # 2 if wall below
        
        # Determine if snakes body in spaces next to head
        adjoining_body_top = 1 if (snake_head_x,snake_head_y-utils.GRID_SIZE) in snake_body else 0          # adjoining top square
        adjoining_body_bottom = 1 if (snake_head_x,snake_head_y+utils.GRID_SIZE) in snake_body else 0       # adjoining bottom square
        adjoining_body_left = 1 if (snake_head_x-utils.GRID_SIZE,snake_head_y) in snake_body else 0         # adjoining left square
        adjoining_body_right = 1 if (snake_head_x+utils.GRID_SIZE,snake_head_y) in snake_body else 0        # adjoining right square
        
        s = (food_dir_x,food_dir_y,adjoining_wall_x,adjoining_wall_y,adjoining_body_top,adjoining_body_bottom,adjoining_body_left,adjoining_body_right)
        return s