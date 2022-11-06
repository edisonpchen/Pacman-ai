import math
from pacai.agents.capture.capture import CaptureAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
import logging
import random
import time
from pacai.util import probability
from pacai.util import util


def createTeam(
    firstIndex,
    secondIndex,
    isRed,
    first="pacai.agents.capture.dummy.DummyAgent",
    second="pacai.agents.capture.dummy.DummyAgent",
    **kwargs
):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = OffensiveAgent
    secondAgent = DefensiveAgent

    return [
        firstAgent(firstIndex, **kwargs),
        secondAgent(secondIndex, **kwargs),
    ]


class OffensiveAgent(ReflexCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.stalemateCounter = 0
        self.stalemateDistance = 3
        self.lastAction = None

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        features["successorScore"] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # Compute distance to nearest capsule
        capsuleList = self.getCapsules(successor)
        oldCapsuleList = self.getCapsules(gameState)

        # Get own position
        myPos = successor.getAgentState(self.index).getPosition()

        # # Add capsule as food
        # foodList.extend(capsuleList)

        # reward PacMan for getting close to and eating a capsule
        if oldCapsuleList:
            minDist = min(
                [self.getMazeDistance(myPos, capsule) for capsule in oldCapsuleList]
            )
            features["eatCapsule"] = 1 / (minDist + 0.1)
        if len(capsuleList) < len(oldCapsuleList):
            features["ateCapsule"] = 10
        # if capsuleList:
        #     myPos = successor.getAgentState(self.index).getPosition()
        #     minDistance = min([self.getMazeDistance(myPos, cap) for cap in capsuleList])
        #     features['distanceToCapsule'] = minDistance

        # if (len(foodList) > 0):
        #     if capsuleList:
        #         myPos = successor.getAgentState(self.index).getPosition()
        #         minDistance_f = min([self.getMazeDistance(myPos, cap) for cap in capsuleList])
        #         minDistance_c = min([self.getMazeDistance(myPos, food) for food in foodList])
        #         if minDistance_f > minDistance_c:
        #             features['distanceToFood'] = minDistance_f
        #             features['distanceToCapsule'] = -1000000
        #         else:
        #             features['distanceToFood'] = -100000000000
        #             features['distanceToCapsule'] = minDistance_c

        # compute dist to enemy
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

        # # This should always be True, but better safe than sorry.
        # if (len(foodList) > 0):
        #     myPos = successor.getAgentState(self.index).getPosition()
        #     minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        #     features['distanceToFood'] = 1/(minDistance + 0.1)

        # This should always be True, but better safe than sorry.
        if len(foodList) > 0:
            # get my position
            myPos = successor.getAgentState(self.index).getPosition()
            foodDists = []
            for food in foodList:
                # calculate the distance from the current food the enemy
                distToClosestEnemy = min(
                    [self.getMazeDistance(e.getPosition(), food) for e in enemies],
                    default=None,
                )
                # calculate the dist from the current food to our position
                foodDist = self.getMazeDistance(myPos, food)
                # if the enemy is closer to the food than we are, devalue it
                if distToClosestEnemy > 2 or len(foodList) <= 2:
                    foodDists.append(foodDist)
            minDistance = min(foodDists, default=0)
            features["distanceToFood"] = 1 / (minDistance + 0.1)
        defenders = [a for a in enemies if a.isGhost() and a.getPosition() is not None]
        if len(defenders) > 0:
            # dists = [distance.manhattan(myPos, a.getPosition()) for a in defenders]
            dists = []
            closestGhost = None
            currentMin = 10000
            for a in defenders:
                distance = self.getMazeDistance(myPos, a.getPosition())
                dists.append(distance)
                # get the ghost object that is closest to Pacman
                if distance < currentMin:
                    currentMin = distance
                    closestGhost = a
            distanceToEnemy = 10
            minDist = min(dists)
            # if ghost is within 5 distance of pacman
            if dists and minDist < 5 and successor.getAgentState(self.index).isPacman():
                distanceToEnemy = minDist
            # if ghosts are scared then go towards them
            # THIS IS NOT WORKING PACMAN STILL RUNS AWAY FROM SCARED GHOSTS
            # current solution is just prioritize food
            if closestGhost.isScared():
                distanceToEnemy *= -0.0
            features["distanceToEnemy"] = -1 / (distanceToEnemy + 0.1)
        # Discourage stopping
        if action == Directions.STOP:
            features["stop"] = 1
        if (
            self.stalemateCounter >= 10
            and self.getDistToMiddle(successor) <= self.stalemateDistance
        ):
            features["staleMate"] = self.stalemateCounter
        else:
            features["staleMate"] = 0
        return features

    def getWeights(self, gameState, action):
        return {
            "successorScore": 100,
            "distanceToFood": 1,
            "distanceToEnemy": 1,
            "stop": -200,
            "distanceToCapsule": 0,
            "eatCapsule": 1,
            "ateCapsule": 100,
            "staleMate": -1,
        }

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        actions = gameState.getLegalActions(self.index)

        # increment stalemate counter

        if self.getDistToMiddle(gameState) <= self.stalemateDistance:
            self.stalemateCounter = self.stalemateCounter + 1
        else:
            self.stalemateCounter = 0
        start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        logging.debug(
            "evaluate() time for agent %d: %.4f" % (self.index, time.time() - start)
        )

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        ##################################################
        ##################################################
        ##################################################

        # if self.lastAction in bestActions:
        #     bestActions.remove(self.lastAction)
        # nextAction =random.choice(bestActions)
        # self.lastAction = nextAction
        # return nextAction
        return random.choice(bestActions)

    def getDistToMiddle(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        middle = int(gameState.getInitialLayout().width / 2)
        return abs(myPos[0] - middle)


class DefensiveAgent(ReflexCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.stalemateCounter = 0
        self.stalemateDistance = 3

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        if (
            not myState.isScared()
        ):  # and myPos[0]>20:#:gameState.getInitialLayout().width/2: # and not already on the other side

            # Computes whether we're on defense (1) or offense (0).
            features["onDefense"] = 1
            if myState.isPacman():
                features["onDefense"] = 0
            # Computes distance to invaders we can see.
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            invaders = [
                a for a in enemies if a.isPacman() and a.getPosition() is not None
            ]
            features["numInvaders"] = len(invaders)

            if len(invaders) > 0:
                dists = [
                    (self.getMazeDistance(myPos, a.getPosition()), a) for a in invaders
                ]
                features["invaderDistance"], closestInvader = min(dists)
            # when there are no invaders go the middle of the layout
            else:
                red = gameState.getRedTeamIndices() == self.getTeam(gameState)
                if red:
                    modifier = 2.3
                else:
                    modifier = 1.7
                middle = [
                    int(gameState.getInitialLayout().width / modifier),
                    int(gameState.getInitialLayout().height / 2),
                ]
                while gameState.hasWall(middle[0], middle[1]):
                    if red:
                        middle[0] = middle[0] - 1
                    else:
                        middle[0] = middle[0] + 1
                midDist = self.getMazeDistance(myPos, tuple(middle))
                features["waitMiddle"] = midDist
            if action == Directions.STOP:
                features["stop"] = 1
            rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
            if action == rev:
                features["reverse"] = 1
            # calculates invaderFoodDistance which is the distance to the
            # food closest to the closest invader
            if len(invaders) > 0:
                foodList = self.getFoodYouAreDefending(successor).asList()

                # This should always be True, but better safe than sorry.
                if len(foodList) > 0:
                    myPos = successor.getAgentState(self.index).getPosition()
                    invaderPos = closestInvader.getPosition()
                    closestFood = min(
                        [
                            (self.getMazeDistance(invaderPos, food), food)
                            for food in foodList
                        ]
                    )
                    # print(self.getMazeDistance(myPos, closestFood[1]))
                    features["invaderFoodDistance"] = self.getMazeDistance(
                        myPos, closestFood[1]
                    )
            # supposedly makes the scared defender run from enemy pacman
            # NOT WELL TESTED
            features["runWhileScared"] = 0
            if myState.isScared():
                dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                invaderDist = min(dists)
                features["runWhileScared"] = invaderDist
        else:  # if agent is scared
            features = OffensiveAgent.getFeatures(self, gameState, action)
        return features

    def getWeights(self, gameState, action):

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        if not myState.isScared():
            return {
                "numInvaders": -1000,
                "onDefense": 100,
                "invaderDistance": -10,
                "stop": -100,
                "reverse": -2,
                "invaderFoodDistance": 0,
                "waitMiddle": -100,
                "runWhileScared": 100,
            }
        else:
            return OffensiveAgent.getWeights(self, gameState, action)

    def getAction(self, gameState):
        return super().getAction(gameState)


class GeneralAgent(CaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def chooseAction(self, gameState):
        return None


class OffensiveQLearningAgent(OffensiveAgent):
    """
    List of Class values:
        self.index              : Index of current agent
        self.red                : Whether or not you're on the red team
        self.agentsOnTeam       : Agent objects controlling you and your teammates
        self.distance           : Maze distance calculator
        self.observationHistory : A history of observations
        self.timeForComputing   : Time to spend each turn on computing maze distances

    Notes:
        This probably could not be used for two agents
    """

    def __init__(
        self, index, alpha=0.005, epsilon=0.05, gamma=1, numTraining=10, **kwargs
    ):
        """
        Args:
            alpha: The learning rate.
            epsilon: The exploration rate.
            gamma: The discount factor.
            numTraining: The number of training episodes.
        """
        super().__init__(index, **kwargs)

        self.lastAction = None
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discountRate = float(gamma)
        self.numTraining = int(numTraining)
        self.weights = {}

    def chooseAction(self, gameState):

        self.update(gameState)

        if probability.flipCoin(self.epsilon):
            action = random.choice(gameState.getLegalActions(self.index))
        else:
            action = self.getPolicy(gameState)
        self.lastAction = action
        return action

    def getPolicy(self, gameState):
        max_action = Directions.STOP
        max_q_val = 0
        for action in gameState.getLegalActions(self.index):
            qVal = self.getQValue(gameState, action)
            if qVal > max_q_val or max_action is None:
                max_q_val = qVal
                max_action = action
        return max_action

    def getQValue(self, state, action):
        qVal = 0
        features = self.getFeatures(state, action)
        for feature in features:
            featureValue = features[feature]
            weight = self.weights.get(feature, 1)
            qVal += weight * featureValue
        return qVal

    def update(self, gameState):
        lastAction = self.lastAction
        if not lastAction:
            return
        lastState = self.getPreviousObservation()
        reward = self.getReward(gameState)
        discount = self.discountRate
        qPrimeVal = self.getValue(gameState)
        qVal = self.getQValue(lastState, lastAction)

        correction = (reward + discount * qPrimeVal) - qVal

        features = self.getFeatures(lastState, lastAction)
        for feature in features:
            oldWeight = self.getWeight(feature)
            newWeight = oldWeight + self.alpha * correction * features[feature]
            self.setWeight(feature, newWeight)

    # Calculates the reward we got for the previous action
    def getReward(self, gameState):
        """----- Notes -----

        This will take some thought:
            In P3 the reward is the change in score after we took the last action.

            We can't use this approach here because the score is affected by our opponents and the
            other agent on our team.

            Some ideas for how to calculate reward for offensive agents:
                - Use the change in amount of food (i.e. add a reward if we eat a food)
                - Add reward if we eat a power up

            I'm not sure how we will calculate rewards for invaders because there is no function
            to tell if we ate an invader

        """
        # TODO: rewrite this function

        prevObv = self.getPreviousObservation()
        reward = (
            len(self.getFood(prevObv).asList())
            - len(self.getFood(gameState).asList())
            + len(self.getCapsules(prevObv))
            - len(self.getCapsules(gameState))
        )
        if reward > 0:
            print("reward: " + str(reward))
        return reward

    def getWeight(self, feature):
        if feature in self.weights:
            self.weights.get(feature, 1)
        initialWeights = {
            "successorScore": 100,
            "distanceToFood": -1,
            "distanceToEnemy": 100,
            "stop": -200,
            "distanceToCapsule": 0,
            "eatCapsule": -2,
            "ateCapsule": 100,
        }
        return initialWeights.get(feature, 1)

    def setWeight(self, feature, value):
        if feature in self.weights and self.weights.get(feature, 0) != value:
            print(str(feature) + ": " + str(self.weights[feature]))
        self.weights[feature] = value

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getValue(self, gameState):
        actions = []
        for legalAction in gameState.getLegalActions(self.index):
            actions.append(self.getQValue(gameState, legalAction))
        if actions:
            return max(actions)
        else:
            return 0.0


class OffensiveMiniMaxAgent(OffensiveAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

        self.weights = {}

    def generateAllSuccessorStates(self, gameState, index):
        legalActions = gameState.getLegalActions(index)
        rtn = [
            (gameState.generateSuccessor(index, action), action)
            for action in legalActions
        ]
        for x in rtn:
            if x[1] == "Stop":
                rtn.remove(x)
        return rtn

    def chooseAction(self, gameState):
        listOfAgents = []
        listOfAgents.append(self.index)
        listOfAgents.extend(self.getOpponents(gameState))
        # listOfAgents = [listOfAgents.pop(listOfAgents.index(self.index))] + listOfAgents

        rtn = self.getActionRecur(gameState, 0, listOfAgents)[1]
        return rtn

    def getActionRecur(self, state, level, listOfAgents):
        depth = 1
        numAgents = len(listOfAgents)
        agentIndex = listOfAgents[level % numAgents]

        # Return if depth reached
        if level >= (depth * numAgents):
            return (self.getValue(state), None)
        successorStates = self.generateAllSuccessorStates(state, agentIndex)

        # Return if action reached
        if len(successorStates) == 0:
            return (self.getValue(state), None)
        if agentIndex in self.getOpponents(state):
            minEval = math.inf
            for successor in successorStates:
                successorEval = self.getActionRecur(
                    successor[0], level + 1, listOfAgents
                )[0]
                if minEval > successorEval:
                    minEval = successorEval
                    minAction = successor[1]
            return (minEval, minAction)
        else:
            maxEval = -math.inf
            for successor in successorStates:
                successorEval = self.getActionRecur(
                    successor[0], level + 1, listOfAgents
                )[0]
                if maxEval < successorEval:
                    maxEval = successorEval
                    maxAction = successor[1]
            return (maxEval, maxAction)

    """ This is an implementation of alpha beta pruning
    def getActionRecurAB(self, state, level, alpha, beta, listOfAgents):
        depth = 1
        numAgents = state.getNumAgents()
        agentIndex = listOfAgents[level % numAgents]

        if level >= (depth * numAgents):
            return (self.getValue(state), None)

        successorStates = self.generateAllSuccessorStates(state, agentIndex)
        if len(successorStates) == 0:
            return (self.getValue(state), None)

        if agentIndex in self.getOpponents(state):
            minEval = math.inf
            for successor in successorStates:
                successorEval = self.getActionRecurAB(successor[0], \
                    level + 1, alpha, beta, listOfAgents)[0]
                if minEval > successorEval:
                    minEval = successorEval
                    minState = successor
                beta = min(beta, minEval)
                if beta <= alpha:
                    break
            return (minEval, minState[1])
        else:
            maxEval = -math.inf
            for successor in successorStates:
                successorEval = self.getActionRecurAB(successor[0], \
                    level + 1, alpha, beta, listOfAgents)[0]
                if maxEval < successorEval:
                    maxEval = successorEval
                    maxState = successor
                alpha = max(alpha, maxEval)
                tmp = self.getValue(successor[0])
                print(successorEval)
                print(tmp)
                if beta <= alpha:
                    break
            return (maxEval, maxState[1])
    """

    def getValue(self, gameState):
        actions = []
        legalActions = gameState.getLegalActions(self.index)
        for legalAction in legalActions:
            actions.append(self.evaluate(gameState, legalAction))
        if actions:
            return max(actions)
        else:
            return 0.0

    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        stateEval = sum(features[feature] * weights[feature] for feature in features)

        return stateEval
