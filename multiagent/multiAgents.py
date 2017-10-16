# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    def getAction(self, gameState):
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action, gameState.getFood()) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action, newFood):
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        badGhosts = dict()

        # get the food coordinates
        foodCoords = []
        for xInd, x in enumerate(newFood):
            for yInd, y in enumerate(x):
                if newFood[xInd][yInd]:
                    foodCoords.append((xInd, yInd))
        min = sys.maxsize

        # compute values for the successor
        for food in foodCoords:
            manDis = manhattanDistance(newPos, food)
            if manDis < min:
                min = manDis

        if min is 0: min = 1

        # get dangerous ghosts
        for ind, ghost in enumerate(newGhostStates):
            if newScaredTimes[ind] is 0:
                badGhosts[ghost] = newScaredTimes[ind]

        # add a multiplier if the spot is dangerous
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            # the coord should only differ by 1 if the ghost is too close
            if (abs(newPos[0] - ghostPos[0]) + abs(newPos[1] - ghostPos[1])) < 2 and badGhosts.has_key(ghost):
                min *= -1

        return float(sys.maxsize) / min


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
    def getMiniMaxValue(self, gameState, agentInd, depth):
        # evaluate state when max depth reached or end of game
        if depth is 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # get next game states
        nextStates = [gameState.generateSuccessor(agentInd, action)
                      for action in gameState.getLegalActions(agentInd)]

        depth -= 1
        numAgents = gameState.getNumAgents()
        # get max for pacman
        if agentInd is 0:
            return max([self.getMiniMaxValue(state, 1, depth) for state in nextStates])

        # min layer for each ghost
        return min([self.getMiniMaxValue(state, (agentInd + 1) % numAgents, depth)
                    for state in nextStates])

    def getAction(self, gameState):
        # get initial actions
        actions = gameState.getLegalActions(0)
        nextStates = [gameState.generateSuccessor(0, action) for action in actions]

        # get max values for each action
        depth = self.depth * gameState.getNumAgents() - 1
        values = [self.getMiniMaxValue(state, 1, depth) for state in nextStates]

        # get max action
        action = actions[0]
        max = -sys.maxsize
        for i in range(0, len(values)):
             if values[i] > max:
                max = values[i]
                action = actions[i]

        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    def getAlphaBeta(self, gameState, agentInd, a, b, depth):
        # evaluate state when max depth reached or end of game
        if depth is 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # get max for pacman
        depth -= 1

        if agentInd is 0:
            v = -sys.maxsize
            for action in gameState.getLegalActions(agentInd):
                state = gameState.generateSuccessor(agentInd, action)
                v = max(self.getAlphaBeta(state, 1, a, b, depth), v)
                if v >= b:
                    break
                a = max(a, v)
            return v
        else:
            v = sys.maxsize
            agentInd = (agentInd + 1) % gameState.getNumAgents()
            for action in gameState.getLegalActions(agentInd):
                state = gameState.generateSuccessor(agentInd, action)
                v = min(self.getAlphaBeta(state, agentInd, a, b, depth), v)
                if v <= a:
                    break
                b = min(b, v)
            return v

    def getAction(self, gameState):
        # get initial actions
        actions = gameState.getLegalActions(0)
        nextStates = [gameState.generateSuccessor(0, action) for action in actions]

        # get max values for each action
        depth = self.depth * gameState.getNumAgents() - 1
        a = -sys.maxsize
        b = sys.maxsize
        values = [self.getAlphaBeta(state, 1, a, b, depth) for state in nextStates]

        # get max action
        action = actions[0]
        max = -sys.maxsize
        for i in range(0, len(values)):
             if values[i] > max:
                max = values[i]
                action = actions[i]

        return action


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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

