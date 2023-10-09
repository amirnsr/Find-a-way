from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from utils import Shape, Node, Graph

def read_image(loc):
  im = Image.open(loc).convert('L')
  im.resize((100, 100))
  return np.asarray(im.resize((100, 100)))

def detect_nodes(image):
  k = np.zeros((100,100))

  rep=[]

  row, col = np.where(image>200)

  for i in range(len(row)):
    if k[row[i]][col[i]] > 0:
      continue
    M = Shape((row[i], col[i]), image)
    for j in M.clas:
      k[j] = 255
    k[int(M.center[0])][int(M.center[1])] = 10
    rep.append((int(M.center[0]),int(M.center[1]),0))

  row, col = np.where(image==0)

  for i in range(len(row)):
    if k[row[i]][col[i]] > 0:
      continue
    M = Shape((row[i], col[i]), image)
    for j in M.clas:
      k[j] = 50
    k[int(M.center[0])][int(M.center[1])] = 10
    rep.append((int(M.center[0]),int(M.center[1]),1))

  row, col = np.where(image==127)

  for i in range(len(row)):
    if k[row[i]][col[i]] > 0:
      continue
    M = Shape((row[i], col[i]), image)
    if len(M.clas) < 40:
        continue
    for j in M.clas:
      k[j] = 150
    k[int(M.center[0])][int(M.center[1])] = 10
    rep.append((int(M.center[0]),int(M.center[1]),2))

  rep.sort(key=lambda x: (x[0],x[1]))
  rep = np.array(rep).reshape(8,6,3)

  k= np.zeros((8,6))

  for i in range(k.shape[0]):
    for j in range(k.shape[1]):
      k[i][j] = rep[i][j][2]

  return k

def dfs(graph):
  # Create a stack to store the nodes to visit.
  stack = [[nodes[7][4], 0, 0]]

  while(len(stack)):
    stack[-1][2] += 1
    cur, level, visited = stack[-1]
    if visited > 1 or graph.is_connected==False:
      stack.pop()
      cur.reset()
      continue
    if cur.order != -1:
      continue
    cur.order = level
    cur.visited = visited
    for i in cur.edges:
      if i.order != -1:
        continue
      stack.append([i, level+1, -1])
    if graph.terminal():
      break


if __name__ == "__main__":
  # Read the image.
  image = read_image(r"maps/game.jpg")
  
  # Detect all of the nodes in the image.
  k = detect_nodes(image)
  
  # Create a graph from the nodes.
  graph = Graph()

  for i in range(k.shape[0]):
    for j in range(k.shape[1]):
      N = Node(k[i][j], i, j)
      graph.add_node(N)

  nodes = np.array(graph.nodes).reshape(k.shape)

  # Add edges to the graph between connected nodes.
  for i in range(k.shape[0]):
    for j in range(k.shape[1]):
      for n_i in range(max(0,i-1), min(k.shape[0],i+2)):
        for n_j in range(max(0,j-1), min(k.shape[1],j+2)):
          if abs(n_i-i) + abs(n_j-j) != 1:
            continue
          if k[n_i][n_j] != 0:
            continue
          graph.add_edge(nodes[i][j], nodes[n_i][n_j])

  graph.set_adj_matrix()
  
  dfs(graph)

  for step in graph.solution():
    print(step)