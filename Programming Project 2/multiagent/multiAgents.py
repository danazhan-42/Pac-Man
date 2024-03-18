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


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        newFoodList = newFood.asList()
        # find the nearest food
        nearestFood = float('inf')
        if len(newFoodList) > 0:
            nearestFood = min([manhattanDistance(newPos, food)
                               for food in newFoodList])
        # find the nearest ghost and avoid it
        nearestGhost = min([manhattanDistance(newPos, ghost.getPosition())
                            for ghost in newGhostStates])
        if nearestGhost < 2:
            return -float('inf')

        return successorGameState.getScore() + 2.0 / nearestFood - 1.0 / nearestGhost


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        maxValue, bestAct = self.max_value(gameState, 0)
        return bestAct

    # helper method to implement getAction
    # returns the max value and the best action for the max player (pacman)
    def max_value(self, gameState, depth):

        actions = gameState.getLegalActions(0)

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState), None)

        maxValue = -(float("inf"))

        bestAct = None
        for action in actions:
            # check the next layer of the tree with values of each action
            value = self.min_value(
                gameState.generateSuccessor(0, action), 1, depth)[0]
            if (value > maxValue):
                # if the value is bigger than the max_value, update the max_value and the best action
                maxValue, bestAct = value, action
        return (maxValue, bestAct)

    # helper method to implement getAction
    # returns the min value and the best action for the min player (ghost)
    def min_value(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)

        if len(actions) == 0:
            return (self.evaluationFunction(gameState), None)

        minValue = float("inf")  # initialize min_value to infinity
        bestAct = None
        for action in actions:
            # if the agent is the last ghost, we need to check the next layer of the tree with pacman
            if (agentIndex == gameState.getNumAgents() - 1):
                value = self.max_value(gameState.generateSuccessor(
                    agentIndex, action), depth + 1)[0]
            else:
                value = self.min_value(gameState.generateSuccessor(
                    agentIndex, action), agentIndex + 1, depth)[0]

            if (value < minValue):  # if the value is smaller than the min_value, update the min_value and the best action
                minValue, bestAct = value, action

        return (minValue, bestAct)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = - float("inf")
        beta = float("inf")

        maxValue, bestAct = self.max_value(gameState, 0, alpha, beta)
        return bestAct

    # helper method to implement getAction
    # returns the max value and the best action for the max player (pacman)
    def max_value(self, gameState, depth, alpha, beta):

        actions = gameState.getLegalActions(0)

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState), None)
        # initialize max_value to infinity
        maxValue = -(float("inf"))

        bestAct = None
        for action in actions:
            value = self.min_value(gameState.generateSuccessor(0, action), 1, depth, alpha, beta)[
                0]  # check the next layer of the tree with values of each action
            if (value > maxValue):
                # if the value is bigger than the max_value, update the max_value and the best action
                maxValue, bestAct = value, action
            if maxValue > beta:  # if the max_value is bigger than beta, return the max_value and the best action
                return (maxValue, bestAct)
            alpha = max(alpha, maxValue)  # update alpha
        return (maxValue, bestAct)

    # helper method to implement getAction
    # returns the min value and the best action for the min player (ghost)
    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)

        if len(actions) == 0:
            return (self.evaluationFunction(gameState), None)

        minValue = float("inf")  # initialize min_value to infinity
        bestAct = None
        for action in actions:
            # if the agent is the last ghost, we need to check the next layer of the tree with pacman
            if (agentIndex == gameState.getNumAgents() - 1):
                value = self.max_value(gameState.generateSuccessor(
                    agentIndex, action), depth + 1, alpha, beta)[0]
            else:
                value = self.min_value(gameState.generateSuccessor(
                    agentIndex, action), agentIndex + 1, depth, alpha, beta)[0]

            if (value < minValue):  # if the value is smaller than the min_value, update the min_value and the best action
                minValue, bestAct = value, action
            if minValue < alpha:
                return (minValue, bestAct)
            beta = min(beta, minValue)

        return (minValue, bestAct)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        maxValue, bestAct = self.max_value(gameState, 0)
        return bestAct

    # helper method to implement getAction
    # max_value method is identical to the one in minimax alogrithm
    # returns the max value and the best action for the max player (pacman)

    def max_value(self, gameState, depth):

        actions = gameState.getLegalActions(0)

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return (self.evaluationFunction(gameState), None)

        # initialize max_value to infinity
        maxValue = -(float("inf"))

        bestAct = None
        for action in actions:
            # check the next layer of the tree with values of each action
            value = self.min_value(
                gameState.generateSuccessor(0, action), 1, depth)[0]
            if (value > maxValue):
                # if the value is bigger than the max_value, update the max_value and the best action
                maxValue, bestAct = value, action
        # return the max_value and the best action
        return (maxValue, bestAct)

    # helper method to implement getAction
    # returns the min value and the best action for the min player (ghost)
    # in expectimax, the min_value method is different from the one in minimax
    # incorporate the probability of each action into the min_value
    def min_value(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)

        if len(actions) == 0:
            return (self.evaluationFunction(gameState), None)

        minValue = 0  # initialize min_value to 0
        bestAct = None
        for action in actions:
            # if the agent is the last ghost, we need to check the next layer of the tree with pacman
            if (agentIndex == gameState.getNumAgents() - 1):
                value = self.max_value(gameState.generateSuccessor(
                    agentIndex, action), depth + 1)[0]
            else:
                value = self.min_value(gameState.generateSuccessor(
                    agentIndex, action), agentIndex + 1, depth)[0]

            # calculate the probability of each action
            prob = 1/len(actions)
            minValue += prob * value  # update the min_value

        return (minValue, bestAct)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: the evaluation function takes into account the following factors:
    - the score of the current game state
    - the distance to the nearest food
    - the number of nearby ghosts
    - the number of scared ghosts
    - the number of left foods and capsules
    The nearer the food, the fewer the nearby ghosts, the fewer the left foods and capsules, and the more the scared ghosts, the better the game state.
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return -float("inf")

    newPos = currentGameState.getPacmanPosition()
    newFoodList = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    numOfLeftFoodAndCapsule = len(
        newFoodList) + len(currentGameState.getCapsules())

    if len(newFoodList) > 0:
        nearestFood = min([manhattanDistance(newPos, food)
                           for food in newFoodList])

    numOfNearbyGhosts = sum([(manhattanDistance(newPos, ghost.getPosition()) < 2)
                             for ghost in newGhostStates])
    numOfScaredGhosts = sum([(ghost.scaredTimer != 0)
                             for ghost in newGhostStates])
    # The nearer the food, the fewer the nearby ghosts, the fewer the left foods and capsules,
    # and the more the scared ghosts, the better the game state.
    # use the 0.01 to avoid the division by zero
    total = currentGameState.getScore() + 2 / nearestFood + 1 / (numOfNearbyGhosts+0.01) + \
        1 / (numOfLeftFoodAndCapsule+0.01) - 1 / (numOfScaredGhosts + 0.01)

    return total


# Abbreviation
better = betterEvaluationFunction
