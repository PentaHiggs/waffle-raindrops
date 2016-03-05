import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from python import namedtuple


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        print "LearningAgent.__init__ run"
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        
        self.actions = [None, 'forward', 'left', 'right']
        self.stateTuple = namedtuple('stateTuple', 
                        ['light','oncoming','right','left','next_waypoint','deadline_approaching'])
        self.stateActionTuple = namedtuple('stateActionTuple',
                        ['state', 'action'])
                        
        self.Q = {}
        self.s_a = None
        self.reward = None
        
    def reset(self, destination=None):
        """ Resets the LearningAgent.  Also called after __init__ by the environment on start """
        self.planner.route_to(destination)
        self.s_a = None
        self.reward = None
        
    def update(self, t):
        # Q function learning parameters
        defaultVal = 10.
        epsilon = .1
        learningRate = 1./t
        discountFactor = 1./2.
        
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        state = stateTuple( 'light'                 = inputs['light'],
                            'oncoming'              = inputs['oncoming'],
                            'right'                 = inputs['right'],
                            'left'                  = inputs['left'],
                            'next_waypoint'         = self.next_waypoint,
                            'deadline_approaching'  = (deadline < 5),
                            )
        
   
        # TODO: Select action according to your policy
        action = None
        
        # Calculate optimal action given current knowledge of Q function
        optimalAction = None
        qVal = [10., 10., 10., 10.]
        for i, action in enumerate(self.actions):
            s_a = self.stateActionTuple(state, action)
            if self.Q.has_key(s_a):
                qVal[i] = self.Q[s_a]
            else:
                self.Q[s_a] = defaultVal
        maxVal = max(qVal)
        optimalAction = self.actions[ random.choice([for i,j in enumerate(qVal) if j==maxVal]) ]
        
        # Choose whether to act optimally or act randomly
        if random.random() <= epsilon:
            action = self.actions[random.randint(0,3)]
        else:
            action = optimalAction
            
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # Update Q function
        optimal_s_a = self.stateActionTuple(state, optimalAction)
        if not self.s_a == None: # if this isn't the first time we run update(self, t)
            self.Q[s_a] = (1-learningRate) * self.Q[s_a] + \
                learningRate * (self.reward + discountFactor * (self.Q[optimal_s_a]) 
                
        # Update the old state variables
        self.s_a = s_a
        self.reward = reward
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, waypoint =  {}".format(deadline, inputs, action, reward, self.next_waypoint)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
