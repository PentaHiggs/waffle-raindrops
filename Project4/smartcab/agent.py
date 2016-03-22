import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, defaultQ=3., epsilon=.06, discountFactor=.5, learningRateMultiplier=5.):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        print "LearningAgent.__init__ run"
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # Defining tuples to be used later in the Q function
        self.stateTuple = namedtuple('stateTuple', 
                        ['light','oncoming','right','left','next_waypoint'])
        self.stateActionTuple = namedtuple('stateActionTuple',
                        ['state', 'action'])      
        
        # Set __init__ arguments to class variables  
        self.defaultQ = defaultQ
        self.epsilon = epsilon
        self.discountFactor = discountFactor
        self.learningRateMultiplier = learningRateMultiplier
        
        # Initialize other variables                  
        self.Q = {}
        self.timesVisited = {}
        self.s_a = None
        self.oldReward = 0
        self.netReward = 0
        
        self.debug = False
        
    def reset(self, destination=None):
        """ Resets the LearningAgent.  Also called after __init__ by the environment on start """
        self.planner.route_to(destination)
        self.s_a = None
        self.oldReward = 0
        print "Self reward = {}".format(self.netReward)
        self.netReward = 0
        if self.debug == True:
            print "Positive Rewards = {} out of {}".format(self.positiveRewards, self.totalRewards)
        self.positiveRewards = 0
        self.totalRewards = 0
        
    def update(self, t):
        actions = [None, 'forward', 'left', 'right']
        
        # Used to print diagonstic output later, useful to know which bad moves were random bad moves.
        isRandom = False
        
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        state = self.stateTuple( light                   = inputs['light'],
                                 oncoming                = inputs['oncoming'],
                                 right                   = inputs['right'],
                                 left                    = inputs['left'],
                                 next_waypoint           = self.next_waypoint,
                                 #deadline_approaching    = (deadline < 5),
                                )
        
        # Update Q function
        def findOptimalAction(Q, state):
            """ Finds an optimal action given the Q function and current state.  If more than
            one action is optimal, then randomly chooses from the set of optimal actions.  If
            the Q function hasn't been intialized at that state, action pair yet, initializes
            it at that pair with value defaultQ.
            """
            #Initialize the q values for the four action-state pairs to default value
            qValues = [self.defaultQ for i in range(4)]
            for i, action in enumerate(actions):
                s_a = self.stateActionTuple(state, action)
                if Q.has_key(s_a):
                    qValues[i] = Q[s_a]
                else:
                    Q[s_a] = self.defaultQ
            bestQ = max(qValues)
            # Choose randomly from the list of best actions.     
            return actions[ random.choice([i for i,qValue in enumerate(qValues) if qValue==bestQ]) ]
        
        optimal_s_a = self.stateActionTuple(state, findOptimalAction(self.Q, state))
        if not self.s_a == None: # if this isn't the first time we run update(self, t)
            learningRate = self.learningRateMultiplier/(self.timesVisited[self.s_a]+1)
            self.Q[self.s_a] = (1-learningRate) * self.Q[self.s_a] + \
                learningRate * (self.oldReward + self.discountFactor * (self.Q[optimal_s_a]))  
           
        # TODO: Select action according to your policy
        action = None
        optimalAction = findOptimalAction(self.Q, state)
        
        if random.random() <= self.epsilon:
            action = actions[random.randint(0,3)]
            isRandom = True
        else:
            action = optimalAction
            
        # Execute action and get reward
        reward = self.env.act(self, action)
         
        # Update the old state variables
        self.s_a = self.stateActionTuple(state, action)
        
        if self.timesVisited.has_key(self.s_a):
            self.timesVisited[self.s_a] += 1
        else:
            self.timesVisited[self.s_a] =1
            
        self.oldReward = reward
        self.netReward += reward
        
        self.positiveRewards += 1 if (reward > 0) else 0
        self.totalRewards += 1
        
        if self.debug:
            #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, random = {}, \waypoint =  {}".format(deadline, inputs, action, reward, isRandom, self.next_waypoint)
            pass
    
    def returnPerformanceMetrics(self):
        return {'steps': self.totalRewards, 'posRewardSteps': self.positiveRewards} 

    
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()

    Qs = [1,2,3,6,8]
    gammas = [1/2., 1/4., 3/4.]
    epsilons = [.05, .1, .15]
    lrms = [.2, 1., 5.]
    
    # We're going to save trial results to a file to avoid having to search through terminal
    saveFile = open("results.txt", "a")
    import datetime
    saveFile.write("--Running at {}--".format(str(datetime.datetime.now)))
    
    from itertools import product
    for i,j,k,l in product(Qs, gammas, epsilons, lrms):
        valuesDict = {"defaultQ":i, "discountFactor":j, "epsilon":k, "learningRateMultiplier":l}
        a = e.create_agent(LearningAgent, **valuesDict)
        e.set_primary_agent(a, enforce_deadline=True)
        sim = Simulator(e, update_delay=0)
        sim.run(n_trials=100)
        a.debug = False
        
        # Record the results of 10 runs
        perf = {'steps':[], 'posRewardSteps':[]}
        for m in xrange(10):
            sim = Simulator(e, update_delay=0)
            sim.run(n_trials=1)
            metrics = a.returnPerformanceMetrics()
            for key, value in metrics.items():
                perf[key].append(value)
        for key, item in valuesDict.items():
            saveFile.write("{} | ".format(item))
        saveFile.write("| {}, {} \n".format(sum(perf['steps'])/10., sum(perf['posRewardSteps'])/10.))
    saveFile.close()
    return        
       

if __name__ == '__main__':
    run()
