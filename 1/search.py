from heapq import heappush, heappop

# build graph
graph = {}
with open('edges.txt', 'r') as f:
    for line in f.readlines():
        startNode, endNode, weight = map(float, line.split())
        if not startNode in graph: graph[startNode] = {}
        if not endNode in graph: graph[endNode] = {}
        graph[startNode][endNode] = weight
        graph[endNode][startNode] = weight

# get the heuristic data
heur = {}
with open('heuristic.txt', 'r') as f:
    for line in f.readlines():
        node, dist = map(float,line.split(" "))
        heur[node] = dist

# format output
def prettyPrint(path, nodesVisited):
    print("Num nodes visited: " + str(nodesVisited))
    print("Num of nodes on path: " + str(len(path)))
    dist = 0
    for i in range(0, len(path) - 2):
        dist += graph[path[i]][path[i+1]]
    print("Distance (km): " + str(dist)) 


def bfs(startNode, endNode):
    queue = [(startNode, [startNode])]
    seen = set()
    numOfNodesVisited = 0
    while len(queue) > 0:
        numOfNodesVisited += 1
        current, path = queue.pop(0)
        for neighbor in graph[current]:
            if neighbor == endNode:
                path.append(endNode)
                return (path, numOfNodesVisited)
            if not neighbor in seen:
                newPath = [*path, neighbor]
                seen.add(neighbor)
                queue.append((neighbor, newPath))
    return False

def ucs(startNode, endNode):
    queue = [(0, [startNode])]
    seen = set()
    numOfNodesVisited = 0
    while len(queue) > 0:
        numOfNodesVisited += 1
        weight, path = heappop(queue)
        checking = path[len(path) - 1]
        for neighbor in graph[checking]:
            if neighbor == endNode:
                path.append(neighbor)
                return (path, numOfNodesVisited)
            if not neighbor in seen:
                newWeight = weight + graph[checking][neighbor]
                newPath = [*path, neighbor]
                seen.add(neighbor)
                heappush(queue, (newWeight, newPath))
    return False

def astar(startNode, endNode):
    queue = [(heur[startNode],[startNode])]
    seen = set()
    numOfNodesVisited = 0
    while len(queue) > 0:
        numOfNodesVisited += 1
        score, path = heappop(queue)
        current = path[len(path) - 1]
        score -= heur[current] # remove the last heuristic
        for neighbor in graph[current]:
            if neighbor == endNode:
                path.append(neighbor)
                return (path, numOfNodesVisited)
            if not neighbor in seen:
                # remove the last heuristic
                newScore = score + heur[neighbor] + graph[current][neighbor]
                newPath = [*path, neighbor]
                heappush(queue, (newScore, newPath))
                seen.add(neighbor)
    return False

print("\nBFS:")
prettyPrint(*bfs(1025817038, 105012740))
print("\nUCS:")
prettyPrint(*ucs(1025817038, 105012740))
print("\nA*:")
prettyPrint(*astar(1025817038, 105012740))
