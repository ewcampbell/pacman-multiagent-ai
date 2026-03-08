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
            invaderDistances.sort(key=lambda x: x[0])  # Correctly sort by distance
            closestInvaderDist, _ = invaderDistances[0]  # We only need the distance, not the AgentState
            features['invaderDistance'] = closestInvaderDist

        # If no visible invaders, defend power pellets first, then fallback to food
        else:
            defensePos = self.getValidDefensePosition(gameState)
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
            'defenseDistance': -15,  # Defend power pellets and food
            'stop': -100  # Penalize stopping
        }

    def getValidDefensePosition(self, gameState):
        """
        Finds a smart defensive position near the border while ensuring it is valid in random maps.
        - The agent will NOT cross into enemy territory.
        - It prioritizes defending power pellets first.
        - If no pellets, it defends food closest to the enemy's offense.
        - If no enemy is seen, it stays deeper on our side to block entry.
        - It ensures the chosen position is valid on RANDOM maps.
        """
        walls = gameState.getWalls()
        width, height = walls.getWidth(), walls.getHeight()
        halfWidth = width // 2  # Red team defends left side, Blue team defends right side

        # Defensive boundary correction:
        defenseX = halfWidth - 2 if self.red else halfWidth + 1

        # Ensure defenseX is walkable
        if walls[defenseX][height // 2]:  # If the default position is a wall, adjust it
            for dy in range(-3, 4):  # Check nearby y-positions
                if not walls[defenseX][height // 2 + dy]:
                    defenseX = (defenseX, height // 2 + dy)
                    break
            else:
                return None  # No valid defensive spot found

        # Get power pellets on our side
        pellets = [p for p in self.getCapsulesYouAreDefending(gameState) if not walls[p[0]][p[1]]]

        # Prioritize defending power pellets if they exist
        if pellets:
            try:
                closestPellet = min(pellets, key=lambda p: self.getMazeDistance(defenseX, p))
                return self.getSafeDefensePosition(gameState, *closestPellet)
            except Exception:
                pass  # If the pellet is unreachable, fallback to food defense

        # Get food positions on our side
        foodGrid = self.getFoodYouAreDefending(gameState)
        foodList = [f for f in foodGrid.asList() if not walls[f[0]][f[1]]]  # Ensure food isn't inside a wall

        if not foodList:
            return self.getSafeDefensePosition(gameState, defenseX, height // 2)  # No food left, fallback to default

        # Identify enemy Pacman agents (offense)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        enemyPacmen = [e for e in enemies if e.isPacman() and e.getPosition() is not None]

        # If no enemy Pacmen are visible, stay deeper in defense
        if not enemyPacmen:
            return self.getSafeDefensePosition(gameState, defenseX, height // 2)

        # Determine the enemy's closest food target (but only if it's on our side)
        closestFood = None
        minDist = float('inf')

        for enemy in enemyPacmen:
            enemyPos = enemy.getPosition()

            for food in foodList:
                if (self.red and food[0] >= halfWidth) or (not self.red and food[0] < halfWidth):
                    continue  # Skip food that isn't on our side

                dist = self.getMazeDistance(enemyPos, food)
                if dist < minDist:
                    minDist = dist
                    closestFood = food

        if closestFood is None:
            return self.getSafeDefensePosition(gameState, defenseX, height // 2)  # No valid food found, stay back

        # Defensive agent must **stay on its side**
        bestY = closestFood[1]

        # Ensure position is inside valid bounds
        bestY = max(1, min(bestY, height - 2))

        return self.getSafeDefensePosition(gameState, defenseX, bestY)

    def getSafeDefensePosition(self, gameState, x, y):
        """
        Ensures the selected defensive position is a valid and walkable tile in random maps.
        """
        walls = gameState.getWalls()
        if not walls[x][y]:  # If it's already a valid walkable tile, return it
            return (x, y)

        # If it's a wall, find the closest open space
        for dx in range(-1, 2):  # Check x offsets
            for dy in range(-2, 3):  # Check y offsets
                newX, newY = x + dx, y + dy
                if 1 <= newX < walls.getWidth() - 1 and 1 <= newY < walls.getHeight() - 1:
                    if not walls[newX][newY]:
                        return (newX, newY)  # Return first valid walkable tile

        return (x, y)  # Fallback (this should rarely happen)