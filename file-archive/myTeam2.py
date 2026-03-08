from pacai.util import probability
from pacai.agents.capture.capture import CaptureAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
from pacai.core.actions import Actions
from math import exp
from random import choice
from time import time
from collections import Counter
from pacai.student.search import aStarSearch

def createTeam(firstIndex, secondIndex, isRed):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """
    # change training to false when submitting !!!!!!!!!!!!!!!!!!!!
    return [
        OffensiveQLearningAgent(firstIndex, training=False),
        AggressiveDefensiveAgent(secondIndex),
    ]

class AggressiveDefensiveAgent(ReflexCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = {
            'numInvaders': 0,
            'invaderDistance': 0,
            'defenseDistance': 0,
            'stop': 0
        }

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = tuple(map(int, myState.getPosition()))  # Convert to integer grid position

        # Find visible opponents
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]

        # Prioritize stopping invaders
        features['numInvaders'] = len(invaders)
        if invaders:
            invaderDistances = [(self.getMazeDistance(myPos, tuple(map(int, a.getPosition()))), a) for a in invaders]
            invaderDistances.sort()
            closestInvaderDist, closestInvader = invaderDistances[0]
            features['invaderDistance'] = closestInvaderDist
        
        # If no visible invaders, move towards a high-priority defense area near the border
        else:
            defensePos = self.getValidDefensePosition(gameState)  # Use the new function to get a valid position
            if defensePos:
                features['defenseDistance'] = self.getMazeDistance(myPos, defensePos)

        # Avoid stopping
        if action == Directions.STOP:
            features['stop'] = 1
        
        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,  # High priority to stopping invaders
            'invaderDistance': -20,  # Chase invaders aggressively
            'defenseDistance': -10,  # Move towards the best defensive position
            'stop': -100  # Penalize stopping
        }

    def getValidDefensePosition(self, gameState):
        """
        Finds a smart defensive position near the border based on enemy movements.
        - The agent will NOT cross into enemy territory.
        - It prioritizes defending food that is closest to the enemy's offense.
        - It avoids unnecessary movement near the midline.
        - If no enemy is seen, it stays deeper on our side to block entry.
        """
        foodGrid = self.getFoodYouAreDefending(gameState)
        foodList = foodGrid.asList()

        if not foodList:
            return None  # No food left to defend

        # Get game info
        walls = gameState.getWalls()
        width, height = walls.getWidth(), walls.getHeight()
        halfWidth = width // 2  # Red team defends left side, Blue team defends right side

        # Defensive boundary correction:
        # - If RED, stay at `halfWidth - 2` (just before midline)
        # - If BLUE, stay at `halfWidth` (right side border)
        defenseX = halfWidth - 2 if self.red else halfWidth + 1

        # Identify enemy Pacman agents (offense)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        enemyPacmen = [e for e in enemies if e.isPacman() and e.getPosition() is not None]

        # If no enemy Pacmen are visible, stay deeper in defense
        if not enemyPacmen:
            return (defenseX, height // 2)  # Stay slightly back instead of hovering midline

        # Determine the enemy's closest food target (but only if it's on our side)
        closestFood = None
        minDist = float('inf')
        enemyClosest = None

        for enemy in enemyPacmen:
            enemyPos = enemy.getPosition()

            for food in foodList:
                if (self.red and food[0] >= halfWidth) or (not self.red and food[0] < halfWidth):
                    continue  # Skip food that isn't on our side

                dist = self.getMazeDistance(enemyPos, food)
                if dist < minDist:
                    minDist = dist
                    closestFood = food
                    enemyClosest = enemyPos  # Track enemy’s closest position

        if closestFood is None:
            return (defenseX, height // 2)  # No valid food found, stay back in defense

        # Defensive agent must **stay on its side**
        bestY = closestFood[1]  # Y-position matches enemy's likely target

        # Ensure position is inside valid bounds
        bestY = max(1, min(bestY, height - 2))

        # Ensure position is not a wall
        if walls[defenseX][bestY]:
            # Find closest non-wall position
            for dy in range(-2, 3):  # Check nearby vertical tiles
                if 1 <= bestY + dy < height - 1 and not walls[defenseX][bestY + dy]:
                    return (defenseX, bestY + dy)
            return None  # No valid position found

        # **Extra safety check:** If enemy is too close, don't move too aggressively
        if enemyClosest and self.getMazeDistance(enemyClosest, (defenseX, bestY)) <= 2:
            return (halfWidth - 3 if self.red else halfWidth + 2, height // 2)  # Move deeper instead of staying vulnerable

        return (defenseX, bestY)

class OffensiveQLearningAgent(CaptureAgent):
    """
    A hybrid agent that uses A* for pathfinding and
    approximate Q-learning for high-level decision making.
    """
    
    def __init__(self, index, **kwargs):
        super().__init__(index)
        # Training mode flag
        self.trainingMode = kwargs.get('training', True)
        
        if self.trainingMode:
            self.epsilon = 0.3
            self.alpha = 0.1
            # self.astarReliance = 1.0
        else:
            # Q-Learning parameters during evaluation
            self.epsilon = 0.05
            self.alpha = 0.01
            # self.astarReliance = 0.7
        
        self.gamma = 0.9   # Discount factor
        
        # Initialize weights
        self.weights = self.getInitialWeights()
        
        # Tracking variables
        self.lastState = None
        self.lastAction = None
        self.foodEaten = 0
        self.episodeRewards = 0
        self.start = None
        # Print mode for debugging
        print(f"Training mode: {self.trainingMode}")

    def registerInitialState(self, gameState):
        """
        This method is called before any moves are made.
        """
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        # Debugging
        # redIndices = gameState.getRedTeamIndices()
        # blueIndices = gameState.getBlueTeamIndices()
        # print(f"Red team indices: {redIndices}")
        # print(f"Blue team indices: {blueIndices}")
        
    def getInitialWeights(self):
        """
        Returns initial weights for Q-learning.
        """
        return {
            "bias": 15.542316214952754,
            "distanceToFood": -5.02478564377559547,
            "distanceToGhost": -4.90487651065788,
            "foodDensity": -12.188541651495376,
            "scaredGhostProximity": 2.2472124527704262,
            "powerCapsuleDistance": 0.8725559931786301,
            "ghostScaredTime": -6.421185170226166,
            "avoidDeadEnds": -1.07177988473965,
            "stop": -4.61239106362926
        }
          
    def sanitizeWeights(self):
        """
        Sanitize weights to remove NaN values.
        """
        for feature in self.weights:
            # Check if weight is NaN
            if self.weights[feature] != self.weights[feature]:
                print(f"Warning: NaN detected in weight for {feature}, resetting to initial value")
                # Reset to initial value from getInitialWeights
                initialWeights = self.getInitialWeights()
                if feature in initialWeights:
                    self.weights[feature] = initialWeights[feature]
                else:
                    self.weights[feature] = 0.0
            
    def updateParametersAfterEpisode(self):
        """
        Gradually decay parameters.
        Only called during training.
        """
        if not self.trainingMode:
            print("ERROR: Not in training mode")
            return

        self.epsilon = max(0.05, self.epsilon * 0.995)  # Decay epsilon
        self.alpha = max(0.01, self.alpha * 0.995)  # Decay alpha
    
    def chooseAction(self, gameState):
        """
        Main method that gets called on each turn to choose an action.
        """
        start_time = time()
        # self.moveCounter += 1
        
        # Only update weights during training
        if self.trainingMode and self.lastState is not None:
            self.updateWeights(gameState)
            
        # Get list of legal actions
        actions = gameState.getLegalActions(self.index)
        
        # Get state information
        nearbyGhosts = self.getNearbyGhosts(gameState)
        
        # TERRITORY-BASED DECISION MAKING
        # 1. In home territory: Use A* to get to enemy territory
        # 2. In enemy territory with no nearby threats: Use A* for efficient food collection
        # 3. In enemy territory with nearby threats: Use Q-learning for tactical decisions
    
        astar_action = None
        if (len(nearbyGhosts) == 0 and time() - start_time < 0.7):
            # if random() < self.astarReliance:
            pathToFood = self.findPathToFood(gameState)
            if pathToFood and len(pathToFood) > 0:
                astar_action = pathToFood[0]
        
        # Choose action using epsilon-greedy policy by default
        action = astar_action if astar_action else (
            choice(actions) if probability.flipCoin(self.epsilon)
            else self.getBestAction(gameState, actions)
        )
        
        ghostPositions = [(g.getPosition(), d) for g, d in nearbyGhosts]
        if astar_action:
            print(f"A* to {action}, ghosts:{ghostPositions}")
        else:
            print(f"Q-learning to {action}, ghosts: {ghostPositions}")

        # Store state and action for next update
        self.lastState = gameState
        self.lastAction = action
                
        if action is None:
            actions = gameState.getLegalActions(self.index)
            if actions:
                action = choice(actions)
            else:
                action = Directions.STOP
        
        return action
        
    def getBestAction(self, gameState, actions):
        """
        Compute the action with the highest Q-value.
        """
        bestValue = float('-inf')
        bestAction = actions[0]
        
        for action in actions:
            value = self.getQValue(gameState, action)
            if value > bestValue:
                bestValue = value
                bestAction = action
                
        return bestAction
        
    def getQValue(self, gameState, action):
        """
        Calculate Q-value using features and weights.
        """
        features = self.getFeatures(gameState, action)
        qValue = 0
        
        for feature, value in features.items():
            qValue += self.weights.get(feature, 0) * value
            
        return qValue
    
    def getFeatures(self, gameState, action):
        """
        Extract features for Q-function approximation.
        """
        features = Counter()
        successor = self.getSuccessor(gameState, action)
        
        # Bias feature
        features['bias'] = 1.0
        
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        isPacman = myState.isPacman()
        
        # Stop feature
        features['stop'] = 1.0 if action == Directions.STOP else 0.0
        
        # Food features
        foodList = self.getFood(successor).asList()
        if len(foodList) > 0:
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance / 30.0
            
            # Food density - count food in nearby area (within 5 steps)
            nearbyFood = sum(1 for food in foodList if self.getMazeDistance(myPos, food) <= 5)
            features['foodDensity'] = nearbyFood / 5.0  # Normalize
        
        # Ghost features
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts = [g for g in enemies if not g.isPacman() and g.getPosition() is not None]
        
        if len(ghosts) > 0:
            ghostPositions = [g.getPosition() for g in ghosts]
            distances = [self.getMazeDistance(myPos, pos) for pos in ghostPositions]
            minGhostDistance = min(distances)
            
            # If ghost is too close and we're pacman, heavily penalize
            if isPacman:
                features['distanceToGhost'] = exp(-minGhostDistance / 5.0)
            else:
                features['distanceToGhost'] = 0.0
            
            # Scared ghost features
            scaredGhosts = [g for g in ghosts if g.getScaredTimer() > 0]
            if scaredGhosts:
                dist = [self.getMazeDistance(myPos, g.getPosition()) for g in scaredGhosts]
                minScaredDist = min(dist)
                features['scaredGhostProximity'] = 1.0 / (minScaredDist + 1)
                features['ghostScaredTime'] = max([g.getScaredTimer() for g in scaredGhosts]) / 40.0
        
        # Power capsule features
        capsules = self.getCapsules(successor)
        if len(capsules) > 0:
            minCapDist = min([self.getMazeDistance(myPos, cap) for cap in capsules])
            features['powerCapsuleDistance'] = minCapDist / 20.0  # Normalize
            
        # Detect dead ends (spaces with only 1 exit)
        wallGrid = gameState.getWalls()
        x, y = int(myPos[0]), int(myPos[1])
        exits = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if not wallGrid[x + dx][y + dy]:
                exits += 1
                
        if isPacman:
            penalty = max(0.0, 1.0 - (exits - 1) * 0.25)
            features['avoidDeadEnds'] = penalty
            
        return features
    
    def getSuccessor(self, gameState, action):
        """
        Find the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor
    
    def updateWeights(self, currentState):
        """
        Update weights based on transition from last state.
        Only use during training, not during actual gameplay.
        """
        if self.lastState is None or self.lastAction is None:
            return
        
        # Calculate reward
        reward = self.getReward(self.lastState, currentState)
        
        # Get current Q-value
        currentQ = self.getQValue(self.lastState, self.lastAction)
        
        # Get best next action's Q-value
        nextActions = currentState.getLegalActions(self.index)
        if len(nextActions) == 0:
            maxQ = 0
        else:
            maxQ = max([self.getQValue(currentState, a) for a in nextActions])
        
        # Calculate difference
        diff = (reward + self.gamma * maxQ) - currentQ
        
        # Prevent extreme values
        if abs(diff) > 100:
            diff = 100 * (1 if diff > 0 else -1)
        
        # Update weights with clipping
        features = self.getFeatures(self.lastState, self.lastAction)
        MAX_WEIGHT_DELTA = 5.0  # Maximum allowed weight change per update
        
        for feature in features:
            # Skip update if feature value is extreme or NaN
            if features[feature] != features[feature] or abs(features[feature]) > 1000:
                continue
                
            # Calculate the proposed weight change (delta)
            delta = self.alpha * diff * features[feature]
            
            # Clip update magnitude to prevent large jumps
            if abs(delta) > MAX_WEIGHT_DELTA:
                delta = MAX_WEIGHT_DELTA * (1 if delta > 0 else -1)
            
            # Apply the clipped update
            new_weight = self.weights.get(feature, 0) + delta
            
            # Check for NaN before updating
            if new_weight == new_weight:  # Not NaN
                self.weights[feature] = new_weight

    def getReward(self, lastState, currentState):
        """
        Calculate reward signal for Q-learning.
        """
        # Base reward on score change
        reward = self.getScore(currentState) - self.getScore(lastState)
        
        # Check if we ate food
        oldFood = self.getFood(lastState).asList()
        currentFood = self.getFood(currentState).asList()
        
        if len(oldFood) > len(currentFood):
            reward += 1.0  # Reward for eating food
            
        # Penalty for dying
        lastPos = lastState.getAgentPosition(self.index)
        currentPos = currentState.getAgentPosition(self.index)

        if (lastPos != currentPos and
            lastState.getAgentState(self.index).isPacman() and
            self.getMazeDistance(lastPos, currentPos) > 5 and
            currentPos == self.start):
        
            reward -= 5.0
        
        reward -= 0.01  # Small penalty for each step
        
        return reward
    
    def getNearbyGhosts(self, gameState):
        """
        Return a list of ghost positions within a certain distance.
        """
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [
            e for e in enemies if not e.isPacman()
            and e.getPosition() is not None and e.getScaredTimer() <= 0
        ]
        
        myPos = gameState.getAgentPosition(self.index)
        nearby = []
        
        for ghost in ghosts:
            if ghost.getPosition() is None:
                continue
            
            distance = self.getMazeDistance(myPos, ghost.getPosition())
            if distance <= 5:
                nearby.append((ghost, distance))
                
        return nearby
    
    def findPathToFood(self, gameState):
        """
        Use A* search to find path to closest food.
        """
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [
            g for g in enemies
            if not g.isPacman() and g.getPosition() is not None and g.getScaredTimer() <= 0]
        ghostPositions = [g.getPosition() for g in ghosts]
        # Define the problem components for A* search

        def foodHeuristic(state, problem):
            # Distance to closest food as heuristic
            foodGrid = self.getFood(gameState)
            position = state
            
            if foodGrid.count() == 0:
                return 0
                
            # Find distance to closest food
            foodList = foodGrid.asList()
            minDistance = min([self.getMazeDistance(position, food) for food in foodList])
            return minDistance
            
        class FoodSearchProblem:
            def __init__(self, gameState, agent, ghostPositions):
                self.start = gameState.getAgentPosition(agent.index)
                self.food = agent.getFood(gameState)
                self.walls = gameState.getWalls()
                self.agent = agent
                # self.gameState = gameState
                self.ghostPositions = ghostPositions
                self.costFn = self.ghostAwareCost
                
            def ghostAwareCost(self, position):
                baseCost = 1
                if not self.ghostPositions:
                    return baseCost
                
                ghostDistances = [self.agent.getMazeDistance(position, ghostPos)
                                  for ghostPos in self.ghostPositions]
                minGhostDistance = min(ghostDistances) if ghostDistances else 999
                
                if minGhostDistance <= 3:
                    return baseCost + 1000 * exp(-minGhostDistance)
                elif minGhostDistance <= 5:
                    return baseCost + 50 / (minGhostDistance + 1)
                else:
                    return baseCost
                
            def getStartState(self):
                return self.start
                
            def startingState(self):
                return self.getStartState()
                
            def isGoal(self, state):
                return self.isGoalState(state)
                
            def successorStates(self, state):
                return self.getSuccessors(state)
                
            def isGoalState(self, state):
                x, y = state
                return self.food[x][y]
                
            def getSuccessors(self, state):
                successors = []
                for direction in [
                    Directions.NORTH,
                    Directions.SOUTH,
                    Directions.EAST,
                    Directions.WEST
                ]:
                    x, y = state
                    dx, dy = Actions.directionToVector(direction)
                    nextx, nexty = int(x + dx), int(y + dy)
                    
                    if not self.walls[nextx][nexty]:
                        nextState = (nextx, nexty)
                        cost = self.costFn(nextState)
                        successors.append((nextState, direction, cost))
                return successors
                
            def getCostOfActions(self, actions):
                if actions is None:
                    return 999999
                x, y = self.getStartState()
                cost = 0
                for action in actions:
                    x, y = self.getNextState(x, y, action)
                    cost += self.costFn((x, y))
                return cost
                
            def getNextState(self, x, y, action):
                dx, dy = Actions.directionToVector(action)
                return (x + dx, y + dy)
        
        # Create the search problem and run A*
        problem = FoodSearchProblem(gameState, self, ghostPositions)
        path = aStarSearch(problem, foodHeuristic)
        return path