# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
from game import Actions
import random, util
from operator import add

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print "New position", newPos
        # print "new Food", newFood
        # print "new Ghost States", newGhostStates
        # print "newScaredTimes", newScaredTimes

        foodDistance = 0
        closestFood = 10000000
        for x in range(newFood.width):
          for y in range(newFood.height):
            if newFood[x][y]:
              distance = manhattanDistance(newPos, (x, y))
              foodDistance += distance
              if distance < closestFood:
                closestFood = distance
        ghostDistance = 0
        for ghost in newGhostStates:
          ghostDistance += manhattanDistance(newPos, ghost.getPosition())
        if ghostDistance < 2:
          return -100000000000
        if foodDistance < 2:
          return 100000000000
        return (- foodDistance - 10*closestFood**2 - 10/(ghostDistance)**2 + successorGameState.getScore()**3) 

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        def findValue(state, depth):
          if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          if depth == self.depth*state.getNumAgents():
            return self.evaluationFunction(state)
          if depth % state.getNumAgents() == 0:
            return maxValue(state, depth, depth % state.getNumAgents())
          else:
            return minValue(state, depth, depth % state.getNumAgents())

        def maxValue(state, depth, agentIndex):
          return reduce(max, map(lambda x: findValue(state.generateSuccessor(agentIndex, x), depth+1), state.getLegalActions(agentIndex)))

        def minValue(state, depth, agentIndex):
          return reduce(min, map(lambda x: findValue(state.generateSuccessor(agentIndex, x), depth+1), state.getLegalActions(agentIndex)))

        maxV = -float('inf')
        minimaxAction = 0
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(0):
          value = findValue(gameState.generateSuccessor(0, action), 1) 
          if value > maxV:
            minimaxAction = action
            maxV = value
        return minimaxAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alphas = [-float('inf')]
        for _ in range(1, gameState.getNumAgents()):
          alphas.append(float('inf'))

        def findValue(state, depth, alphas):
          if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          if depth == self.depth*state.getNumAgents():
            return self.evaluationFunction(state)
          if depth % state.getNumAgents() == 0:
            return maxValue(state, depth, depth % state.getNumAgents(), alphas)
          else:
            return minValue(state, depth, depth % state.getNumAgents(), alphas)

        def maxValue(state, depth, agentIndex, alphas):
          alphas = alphas[:]
          v = -float('inf')
          for action in state.getLegalActions(agentIndex):
            v = max(v, findValue(state.generateSuccessor(agentIndex, action), depth+1, alphas))
            if v > min(alphas[1:]):
            # if v > alphas[agentIndex]:
              return v
            alphas[agentIndex] = max(alphas[agentIndex], v)
          return v

        def minValue(state, depth, agentIndex, alphas):
          alphas = alphas[:]
          v = float('inf')
          for action in state.getLegalActions(agentIndex):
            v = min(v, findValue(state.generateSuccessor(agentIndex, action), depth+1, alphas))
            if v < alphas[0]:
              return v
            alphas[agentIndex] = min(alphas[agentIndex], v)
          return v
      
        maxV = -float('inf')
        minimaxAction = 0
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(0):
          value = findValue(gameState.generateSuccessor(0, action), 1, alphas) 
          if value > maxV:
            minimaxAction = action
            maxV = value
          alphas[0] = max(alphas[0], maxV)
        return minimaxAction
        
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        
        def findValue(state, depth):
          if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
          if depth == self.depth*state.getNumAgents():
            return self.evaluationFunction(state)
          if depth % state.getNumAgents() == 0:
            return maxValue(state, depth, depth % state.getNumAgents())
          else:
            return minValue(state, depth, depth % state.getNumAgents())

        def maxValue(state, depth, agentIndex):
          return reduce(max, map(lambda x: findValue(state.generateSuccessor(agentIndex, x), depth+1), state.getLegalActions(agentIndex)))

        def minValue(state, depth, agentIndex):
          actions = state.getLegalActions(agentIndex)
          return reduce(add, map(lambda x: findValue(state.generateSuccessor(agentIndex, x), depth+1), actions)) / len(actions)

        maxV = -float('inf')
        minimaxAction = 0
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        for action in gameState.getLegalActions(0):
          value = findValue(gameState.generateSuccessor(0, action), 1) 
          if value > maxV:
            minimaxAction = action
            maxV = value
        return minimaxAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: linear combination of various factors plus some negative
          factors which make sure that pacman does not get eaten often."
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = currentGameState.getScore()
    capsules = currentGameState.getCapsules()

    ghostDistance = 0
    for ghost in newGhostStates:
      ghostDistance += manhattanDistance(newPos, ghost.getPosition())

    capsuleDistance = 0
    if len(capsules) > 0 and max(newScaredTimes) > 25:
      if ghostDistance < 2:
        return -1000000000
      else:
        closestCapsule = 10000
        for capsule in capsules:
          capsuleDistance += mazeDistance(capsule, newPos, currentGameState)
          if capsuleDistance < closestCapsule:
            closestCapsule = capsuleDistance
    else:
      capsuleDistance = 10000000000000000

    foodDistance = 0
    closestFood = (1234, 5678)
    for x in range(newFood.width):
      for y in range(newFood.height):
        if newFood[x][y]:
          distance = manhattanDistance(newPos, (x, y))
          foodDistance += distance
          if distance < manhattanDistance(closestFood, newPos):
            closestFood = (x, y)
    if closestFood != (1234, 5678):
      closestFood = mazeDistance(closestFood, newPos, currentGameState)
      
    if ghostDistance < 2:
      return -100000000000
    elif foodDistance == 0:
      return 100000000 * score
    if foodDistance == 2:
      return 1000000 * score
    elif foodDistance == 1:
      return 10000000 * score

    value = 0
    value += - foodDistance
    value += - 10*closestFood**2
    value += - 10/ghostDistance**2
    value += score**3
    value += 100000000 / (1 + capsuleDistance)
    return value




class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()

class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test,
    successor function and cost function.  This search problem can be
    used to find paths to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class Node():
  """
  A container storing the current state of a node, the list 
  of  directions that need to be followed from the start state to
  get to the current state and the specific problem in which the
  node will be used.
  """
  def __init__(self, state, path, cost=0, heuristic=0, problem=None):
    self.state = state
    self.path = path
    self.cost = cost
    self.heuristic = heuristic
    self.problem = problem

  def __str__(self):
    string = "Current State: "
    string += __str__(self.state)
    string += "\n"
    string == "Path: " + self.path + "\n"
    return string

  def getSuccessors(self, heuristicFunction=None):
    children = [] 
    for successor in self.problem.getSuccessors(self.state):
      state = successor[0]
      path = list(self.path)
      path.append(successor[1])
      cost = self.cost + successor[2]
      if heuristicFunction:
        heuristic = heuristicFunction(state, self.problem)
      else:
        heuristic = 0
      node = Node(state, path, cost, heuristic, self.problem)
      children.append(node)
    return children

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    
    closed = set()
    fringe = util.Queue()

    startNode = Node(problem.getStartState(), [], 0, 0, problem)
    fringe.push(startNode)

    while True:
      if fringe.isEmpty():
        return False
      node = fringe.pop()
      if problem.isGoalState(node.state):
        return node.path
      if node.state not in closed:
        closed.add(node.state)
        for childNode in node.getSuccessors():
          fringe.push(childNode)

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built.  The gameState can be any game state -- Pacman's position
    in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(breadthFirstSearch(prob))







# Abbreviation
better = betterEvaluationFunction
bfs = breadthFirstSearch

