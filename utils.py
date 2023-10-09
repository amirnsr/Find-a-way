import numpy as np
from numpy.linalg import matrix_power

class Shape():
  def __init__(self, rep, img):
    self.rep = rep
    self.img = img
    self.clas = self.extrapolate(self.rep)
    self.center = np.average(self.clas, axis=0)
    self.center[0] = int(self.center[0])
    self.center[1] = int(self.center[1])
    if self.center[0]%8<5:
      self.center[0] = self.center[0] - self.center[0]%8
    else:
      self.center[0] = self.center[0] + 8 - self.center[0]%8

  def extrapolate(self, rep):
    ret = []
    tosee = [rep]
    while(len(tosee)):
      cur = tosee.pop()
      #print(len(ret))
      ret.append(cur)
      adj = self.FindAdj(cur)
      for enum in enumerate(adj):
        if enum[1] in set(ret) or enum[1] in set(tosee):
          continue
        if abs(int(self.img[enum[1]]) - int(self.img[self.rep])) < 20:
          #print(enum[1], self.img[enum[1]])
          #time.sleep(1)
          tosee.append(enum[1])
    return ret

  def FindAdj(self, cell):
    ret=[]
    for i in range(max(0,cell[0]-1), min(101,cell[0]+2)):
      for j in range(max(0,cell[1]-1), min(101,cell[1]+2)):
        ret.append((i,j))
    return ret

class Node():
  def __init__(self, num, coord1, coord2):
    self.start = (num==2)
    self.block = (num==1)
    self.order = -1
    self.edges = []
    self.coord = (coord1, coord2)
    self.visited = 0
    self.index = -1

  def reset(self):
    self.order = -1

  def cut(self): # whether the node is a cute node
    neighb = [i for i in self.edges if i.order==-1]
    if len(neighb) != 2:
      return False
    if neighb[0].coord[0] == neighb[1].coord[0] or neighb[0].coord[1] == neighb[1].coord[1]:
      return True
    return False

class Graph():
  def __init__(self, nodes=[]):
    self.nodes = []
    self.sol = []
    self.visited_nodes = set()

  def add_node(self, node):
    self.nodes.append(node)
    node.index = len(self.nodes) - 1

  def add_edge(self, node1, node2):
    node1.edges.append(node2)

  def remove_node(self, node):
    self.nodes.remove(node)
    for i in self.nodes:
      for j in i.edges:
        try:
          i.remove(node)
        except:
          continue

  def is_connected(self):
    M = matrix_power(self.adj_matrix,1)
    N = [i.index for i in self.nodes if i.block==False and i.visited==0]
    for i in range(M.shape[0]):
      M[i][i]=1

    M = M[N,:][:,N]

    M = matrix_power(M,len(N))
    for i in range(M.shape[0]):
      if len(M[i][M[i]!=0]) == len(N):
        return True
    return False
    """visited = set()
    queue = [(seed,0)]
    while(len(queue)):
      cur, level = queue.pop(0)
      cur.visited = True
      visited.add(cur)
      for i in cur.edges:
        if i.visited:
          continue
        queue.append((i,level+1))
      if(len(visited) == 35):
        return True
    if(len(visited) == 35):
        return True
    #print(len(visited))
    return False"""

  def set_adj_matrix(self):
    adj = np.zeros((len(self.nodes),len(self.nodes)), dtype=int)
    for i in range(len(self.nodes)):
      for j in self.nodes[i].edges:
        adj[i][j.index] = 1
        adj[j.index][i] = 1

    self.adj_matrix = adj


  def terminal(self):
    return ~np.any(np.array([i.order for i in [j for j in self.nodes if j.block!=True]])==-1)

  def solution(self):
    if not self.terminal:
      return False
    ret=[]
    self.sol = sorted([(i.order,i) for i in self.nodes if i.block==False],key=lambda x: x[0])
    for i in range(1, len(self.sol)):
      if self.sol[i][1].coord[0] == self.sol[i-1][1].coord[0] + 1:
        ret.append('Down')
      elif self.sol[i][1].coord[0] == self.sol[i-1][1].coord[0] - 1:
        ret.append('Up')
      elif self.sol[i][1].coord[1] == self.sol[i-1][1].coord[1] + 1:
        ret.append('Right')
      elif self.sol[i][1].coord[1] == self.sol[i-1][1].coord[1] - 1:
        ret.append('Left')
    return ret
