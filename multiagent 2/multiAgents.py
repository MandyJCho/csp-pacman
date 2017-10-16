# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

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
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFoodList = (currentGameState.getFood()).asList()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    # total points of the successor state
    totalPts = 0

    # sort the food list by the manhattan distance from pacman
    # to the food in decreasing order
    oldFoodList.sort(lambda f1, f2: ManhDistCmp(newPos, f1, f2))

    # let foodPts be the distance from successor's position of pacman
    # to closest food in the current state
    foodPts = manhattanDistance(newPos, oldFoodList[0])

    # if pacman eats a food in the successor state, increase
    # the total points by 2; otherwise, increase the total
    # points by reciprocal of foodPts
    if foodPts == 0:
      totalPts += 2
    else:
      totalPts += 1.0 / foodPts

    # store ghosts' positions in a list
    ghostsPos = []
    for ghost in newGhostStates:
      ghostsPos.append(ghost.getPosition())

    ghostPts = 0
    if len(ghostsPos) != 0:
      # sort the ghost list by the manhattan distance from pacman
      # to the ghost in decreasing order
      ghostsPos.sort(lambda g1, g2: ManhDistCmp(newPos, g1, g2))
      
      # return a very low value if pacman is caught by
      # a ghost in the successor state, -999 in this case
      if manhattanDistance(newPos, ghostsPos[0]) == 0:
        return -999

      # otherwise, the successor state get a value of -3 times reciprocal
      # of distance to the closest ghost as points for ghost part
      else:
        ghostPts = -3.0 / manhattanDistance(newPos, ghostsPos[0])

    # add ghostPts to the total points
    totalPts += ghostPts

    # decrease the total points by 1 if the action is STOP
    if action == Directions.STOP:
      totalPts -= 1  

    return totalPts

def ManhDistCmp(pos, p1, p2):
  """
    This is a comparator for sorting list of positions respected
    to pacman's position.
  """
  diff = manhattanDistance(pos, p1) - manhattanDistance(pos, p2)

  if diff < 0:
    return -1
  elif diff > 0:
    return 1
  else:
    return 0
  

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
    numAgents = gameState.getNumAgents()
    totalDepth = self.depth * numAgents  # 1 ply means each agent moves one time

    # remove STOP from pacman's action list if it exists in the list
    actions = gameState.getLegalActions(0)
    if Directions.STOP in actions:
        actions.remove(Directions.STOP)

    # get a list pacman's successor states
    newStates = []
    for action in actions:
      newStates.append(gameState.generateSuccessor(0, action))

    # get a list of values of pacman's successor states
    vals = []
    for nextState in newStates:
      vals.append(self.MaxMinValue(nextState, 1, numAgents, totalDepth - 1))

    # find the largest value(s) the pacman can get
    # among all the successor states
    maxVal = max(vals)
    bestIndices = [idx for idx in range(len(vals)) if vals[idx] == maxVal]
    # return the action that will let pacman get
    # the largest value; randomly pick one action if
    # there are multiple actions with the greatest value
    chosenIdx = random.choice(bestIndices)
    return actions[chosenIdx]

  def MaxMinValue(self, gameState, agentIdx, numAgents, depth):
    """
      This function calculate greatest(smallest) value pacman(ghost)
      can get among all the successor states.
    """
    # return the value of current state using evaluationFunction
    # if the current state is a terminal state (win/lose) or if
    # the function hits the specified depth
    if gameState.isWin() or gameState.isLose() or depth == 0:
      return self.evaluationFunction(gameState)
    print agentIdx
    actions = gameState.getLegalActions(agentIdx)
    # remove STOP from actions list if the agent is pacman
    if agentIdx == 0:
      if Directions.STOP in actions:
        actions.remove(Directions.STOP)
  
    # get a list of successor states
    newStates = []
    for action in actions:
      newStates.append(gameState.generateSuccessor(agentIdx, action))
    
    # evalue the successor states by recursively calling this function
    # until it is terminal state or depth is 0
    vals = []
    for nextState in newStates:
      vals.append(self.MaxMinValue(nextState, (agentIdx + 1) % numAgents, numAgents, depth - 1))

    # if the agent is pacman, return the maximum value;
    # otherwise, return the minimum value
    if agentIdx == 0:
      return max(vals)
    else:
      return min(vals)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    # We used this global variable to compare with the number of recurssive calls of MinimaxAgent
    # callsCount = 0

    def getAction(self, gameState):

        """
          Returns the minimax action using self.depth and self.evaluationFunction
       """
        "*** YOUR CODE HERE ***"

        # AlphaBetaAgent.callsCount = 0
        # If the game state is lost or win then just return the evaluation value
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        # This part is almost the same with MinimaxAgent other than two additional parameters of alpha and beta values
        # by recursion calls in MiniMax tree to make a decision
        # alpha = the best choice of the highest values we have found so far along the path for MAX.
        #         (initializes -9999999)
        # beta = the best choice of the lowest values we have found so far along the path for MIN.
        #         (initializes +9999999)

        finalDirection = self.maxValueAlphaBeta(gameState, self.depth * gameState.getNumAgents(), gameState.getNumAgents(), 0, -9999999, 9999999)
        # print("Number of Calls: ", AlphaBetaAgent.callsCount)
        return finalDirection[1]

    # The helper function to remove STOP action to improve efficiency
    def removeStop(self, legalMoves):
        stopDirection = Directions.STOP
        if stopDirection in legalMoves:
            legalMoves.remove(stopDirection)

    # The maxValue method for PacMan whose agent index is zero
    def maxValueAlphaBeta(self, gameState, depth, numOfAgents, agentIndex, alpha, beta):
        # AlphaBetaAgent.callsCount += 1
        legalMoves = gameState.getLegalActions(agentIndex)
        self.removeStop(legalMoves)

        # The next states, either passed in to getAction or generated via GameState.generateSuccessor
        nextStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
        if depth == 0 or len(nextStates) == 0:
            return (self.evaluationFunction(gameState), Directions.STOP)

        maxValue = -9999999
        action = legalMoves[0]
        moveIndex = 0
        chooseMax = []

        # There are only one case to call minValue from Pacman perspective
        for nextState in nextStates:
            res = self.minValueAlphaBeta(nextState, depth-1, numOfAgents-2, numOfAgents, (agentIndex+1)%numOfAgents, alpha, beta)
            chooseMax.append(res)
            if res > maxValue:
                maxValue = res
                action = legalMoves[moveIndex]
                # Check "cut-off" condition
                if res > alpha: # If alpha is smaller than the current Max value
                    alpha = res # Then replace alpha with the current Max value
                if alpha >= beta: # When alpha value is larger than beta value
                    return (alpha, action)  # beta cut-off
            moveIndex = moveIndex + 1

        return (maxValue, action)

    # The minValue method for each ghost whose index is agentIndex
    def minValueAlphaBeta(self, gameState, depth, numOfMins, numOfAgents, agentIndex, alpha, beta):
        # AlphaBetaAgent.callsCount += 1

        legalMoves = gameState.getLegalActions(agentIndex)
        self.removeStop(legalMoves)

        # The next states, either passed in to getAction or generated via GameState.generateSuccessor
        nextStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
        chooseMin = []

        # Considering three cases:
        # Case 1. If there is no more depth or next states then return evaluation value
        if depth == 0 or len(nextStates) == 0:
            return self.evaluationFunction(gameState)

        # Case 2. If all ghosts are evaluated then call maxValue for Pacman agent
        if numOfMins == 0:
            for nextState in nextStates:
                res =  self.maxValueAlphaBeta(nextState, depth-1,  numOfAgents, (agentIndex+1)%numOfAgents, alpha, beta)
                chooseMin.append(res[0])
                # Check "cut-off" condition
                if min(chooseMin) < beta:  # If beta is larger than the current Min value
                    beta = min(chooseMin) # Then replace beta with the current Min value
                if alpha >= beta: # When alpha value is larger than beta value
                    return alpha  #alpha cut-off

        # Case 3. If there are more ghosts left then call minValue for Ghost agent
        else:
            for nextState in nextStates:
                res = self.minValueAlphaBeta(nextState, depth-1, numOfMins-1, numOfAgents, (agentIndex+1)%numOfAgents, alpha, beta)
                chooseMin.append(res)
                # Check "cut-off" condition
                if min(chooseMin) < beta: # If beta is larger than the current Min value
                    beta = min(chooseMin) # Then replace beta with the current Min value
                if alpha >= beta: # When alpha value is larger than beta value
                    return alpha  #alpha cut-off

        return min(chooseMin)


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
    "*** YOUR CODE HERE ***"
    numAgents = gameState.getNumAgents()
    totalDepth = self.depth * numAgents  # 1 ply means each agent moves one time

    # remove STOP from pacman's action list if it exists in the list
    actions = gameState.getLegalActions(0)
    if Directions.STOP in actions:
        actions.remove(Directions.STOP)

    # get a list pacman's successor states
    newStates = []
    for action in actions:
      newStates.append(gameState.generateSuccessor(0, action))

    # get a list of values of pacman's successor states
    vals = []
    for nextState in newStates:
      vals.append(self.ExpectimaxValue(nextState, 1, numAgents, totalDepth - 1))

    # find the largest value(s) the pacman can get
    # among all the successor states
    maxVal = max(vals)
    bestIndices = [idx for idx in range(len(vals)) if vals[idx] == maxVal]

    # return the action that will let pacman get
    # the largest value; randomly pick one action if
    # there are multiple actions with the greatest value
    chosenIdx = random.choice(bestIndices)
    return actions[chosenIdx]

  def ExpectimaxValue(self, gameState, agentIdx, numAgents, depth):
    # return the value of current state using evaluationFunction
    # if the current state is a terminal state (win/lose) or if
    # the function hits the specified depth
    if gameState.isWin() or gameState.isLose() or depth == 0:
      return self.evaluationFunction(gameState)

    actions = gameState.getLegalActions(agentIdx)
    # remove STOP from actions list if the agent is pacman
    if agentIdx == 0:
      if Directions.STOP in actions:
        actions.remove(Directions.STOP)

    # get a list of successor states
    newStates = []
    for action in actions:
      newStates.append(gameState.generateSuccessor(agentIdx, action))

    # evalue the successor states by recursively calling this function
    # until it is terminal state or depth is 0
    vals = []
    for nextState in newStates:
      vals.append(self.ExpectimaxValue(nextState, (agentIdx + 1) % numAgents, numAgents, depth - 1))

    # if the agent is pacman, return the maximum value;
    # otherwise, return the expectation according to
    # how the ghosts act (assume the ghost has equal chance
    # to choose each action among all the legal actions)
    if agentIdx == 0:
      return max(vals)
    else:
      return sum(vals) / len(actions)

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    I used current score, number of food, distance to the closest food and
    distance to the closest ghost to calculate the value of the current state.

    The value of the current state is total points (totalPts) in my function.

    First of all, total points are based on current score.
    
    Then I took number of food into account. Minus 1 point on
    total points for each food left in the current state.

    After that, I calculate the distance from pacman to the
    closest food/ghost and combine them by multiplying distance to
    cloest food by square of reciprocal of distance to the closest
    food

    Finally, I combined all the value I got to produce value of the
    current state as following way:
    total points of the current state =
      current score +
      distance to closest ghost / (distance to the closest food)^2 -
      number of food left

    For special cases such as no food/ghost in current state, please
    refer to comment in the method.

    Bottleneck: This evaluation function has a low probability of
    getting over 1000 points when pacman wins.
  """
  "*** YOUR CODE HERE ***"
  pos = currentGameState.getPacmanPosition()
  foodList = (currentGameState.getFood()).asList()
  ghostStates = currentGameState.getGhostStates()

  totalPts = currentGameState.getScore()

  # return the current score if there is no food left
  if len(foodList) == 0:
    return totalPts

  # -1 point for each food left in the current state that
  # encourages pacman to eat food
  foodNumPts = -1 * currentGameState.getNumFood()

  # sort the food position in ascending order of manhattan distance
  # to the pacman's current position
  foodList.sort(lambda f1, f2: ManhDistCmp(pos, f1, f2))

  # define food distance point as reciprocal of closest food
  # to the pacman's current position
  foodDistPts = 1.0 / manhattanDistance(pos, foodList[0])

  # get a list of ghost distance to the pacman's current position
  ghostsDistList = []
  for ghostState in ghostStates:
    ghostsDistList.append(manhattanDistance(pos, ghostState.getPosition()))

  # return current score plus all the food points if there is
  # no ghost in the current state (might not happen in this project)
  if len(ghostsDistList) == 0:
    totalPts += (foodDistPts ** 2) + foodNumPts
    return totalPts
  
  ghostsDistList.sort()

  # total points of the current state:
  # current score +
  # distance to closest ghost / (distance to the closest food)^2 -
  # number of food left
  totalPts += ghostsDistList[0] * (foodDistPts ** 2) + foodNumPts

  return totalPts

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


