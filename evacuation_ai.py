import heapq
import matplotlib.pyplot as plt
import numpy as np

# =============================
# GRID DEFINITION
# 0 = free path
# 1 = blocked (fire, debris)
# 'S' = start
# 'E' = exit
# =============================

grid = [
 ['S',0,0,0,0,0,0,0,0,0,0,0],
 [1,1,1,1,0,1,1,1,1,1,1,0],
 [0,0,0,0,0,0,0,0,0,0,1,0],
 [0,1,1,1,1,1,1,1,1,0,1,0],
 [0,1,0,0,0,0,0,0,1,0,0,0],
 [0,1,0,1,1,1,1,0,1,1,1,0],
 [0,0,0,1,0,0,0,0,0,0,0,0],
 [1,1,0,1,0,1,1,1,1,1,1,0],
 [0,0,0,0,0,0,0,0,0,0,1,0],
 [0,1,1,1,1,1,1,1,1,0,1,0],
 [0,0,0,0,0,0,0,0,0,0,0,0],
 [1,1,1,1,1,1,1,1,1,1,0,'E']
]





# =============================
# HEURISTIC FUNCTION
# =============================

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# =============================
# FIND START / EXIT POSITION
# =============================

def find_position(grid, value):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == value:
                return (i, j)
    return None

# =============================
# A* SEARCH ALGORITHM
# =============================

def astar(grid):
    start = find_position(grid, 'S')
    goal = find_position(grid, 'E')

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue

                temp_g = g_score[current] + 1

                if neighbor not in g_score or temp_g < g_score[neighbor]:
                    g_score[neighbor] = temp_g
                    f_score = temp_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current

    return None

# =============================
# VISUALIZATION FUNCTION
# =============================

def visualize(grid, path):
    visual = np.zeros((len(grid), len(grid[0]), 3))

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                visual[i][j] = [0, 0, 0]       # blocked - black
            else:
                visual[i][j] = [1, 1, 1]       # free - white

    for (x, y) in path:
        visual[x][y] = [1, 0, 0]               # path - red

    plt.imshow(visual)
    plt.title("Evacuation Route (Red Path)")
    plt.axis('off')
    plt.show()

# =============================
# RUN AI PLANNER
# =============================

path = astar(grid)

if path:
    print("Evacuation Path:", path)
    visualize(grid, path)
else:
    print("No safe evacuation path found")
