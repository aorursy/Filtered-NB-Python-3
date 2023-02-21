#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import pandas
import matplotlib.pyplot as plt




filename = '../input/cities/cities.csv'
df = pandas.read_csv(filename) # import 100% data set
df.tail()




nb_cities = max(df.CityId)
print("Number of cities to visit : ", nb_cities)




def sieve_of_eratosthenes(n):
    primes = [True for i in range(n+1)] # Start assuming all numbers are primes
    primes[0] = False # 0 is not a prime
    primes[1] = False # 1 is not a prime
    for i in range(2,int(np.sqrt(n)) + 1):
        if primes[i]:
            k = 2
            while i*k <= n:
                primes[i*k] = False
                k += 1
    return(primes)
prime_cities = sieve_of_eratosthenes(max(df.CityId))




sum(prime_cities)




df['prime'] = prime_cities
df.head()




fig, ax = plt.subplots(figsize = (10,10))
ax.scatter(x = df.X, y = df.Y, alpha = 0.5, s = 1)
ax.set_title('Cities chart. North Pole, prime and non prime cities', fontsize = 16)
ax.scatter(x = df.X[0], y = df.Y[0], c = 'r', s =12)
ax.scatter(x = df[df.prime].X, y = df[df.prime].Y, s = 1, c = 'purple', alpha = 0.3)
ax.annotate('North Pole', (df.X[0], df.Y[0]),fontsize =12)
ax.set_axis_off()




def total_distance(dfcity,path):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        next_city = city_num
        total_distance = total_distance +             np.sqrt(pow((df.X[city_num] - df.X[prev_city]),2) + pow((df.Y[city_num] - df.Y[prev_city]),2)) *             (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return total_distance
dumbest_path = list(df.CityId[:].append(pandas.Series([0])))
print('Total distance with the dumbest path is '+ "{:,}".format(total_distance(df,dumbest_path)))




# Generic tree node class 
class TreeNode(object): 
    def __init__(self, val): 
        self.val = val 
        self.left = None
        self.right = None
        self.height = 1
  
  
# Insert operation 
class AVL_Tree(object): 
  
    # Recursive function to insert key in  
    # subtree rooted with node and returns 
    # new root of subtree. 
    def insert(self, root, key): 
      
        # Perform normal BST 
        if not root: 
            return TreeNode(key) 
        elif key < root.val: 
            root.left = self.insert(root.left, key) 
        else: 
            root.right = self.insert(root.right, key) 
  
        # Update the height of the  
        # ancestor node 
        root.height = 1 + max(self.getHeight(root.left), 
                           self.getHeight(root.right)) 
  
        # Step 3 - Get the balance factor 
        balance = self.getBalance(root) 
  
        # Step 4 - If the node is unbalanced,  
        # then try out the 4 cases 
        # Case 1 - Left Left 
        if balance > 1 and key < root.left.val: 
            return self.rightRotate(root) 
  
        # Case 2 - Right Right 
        if balance < -1 and key > root.right.val: 
            return self.leftRotate(root) 
  
        # Case 3 - Left Right 
        if balance > 1 and key > root.left.val: 
            root.left = self.leftRotate(root.left) 
            return self.rightRotate(root) 
  
        # Case 4 - Right Left 
        if balance < -1 and key < root.right.val: 
            root.right = self.rightRotate(root.right) 
            return self.leftRotate(root) 
  
        return root 
  
    def leftRotate(self, z): 
  
        y = z.right 
        T2 = y.left 
  
        # Perform rotation 
        y.left = z 
        z.right = T2 
  
        # Update heights 
        z.height = 1 + max(self.getHeight(z.left), 
                         self.getHeight(z.right)) 
        y.height = 1 + max(self.getHeight(y.left), 
                         self.getHeight(y.right)) 
  
        # Return the new root 
        return y 
  
    def rightRotate(self, z): 
  
        y = z.left 
        T3 = y.right 
  
        # Perform rotation 
        y.right = z 
        z.left = T3 
  
        # Update heights 
        z.height = 1 + max(self.getHeight(z.left), 
                        self.getHeight(z.right)) 
        y.height = 1 + max(self.getHeight(y.left), 
                        self.getHeight(y.right)) 
  
        # Return the new root 
        return y 
  
    def getHeight(self, root): 
        if not root: 
            return 0
  
        return root.height 
  
    def getBalance(self, root): 
        if not root: 
            return 0
  
        return self.getHeight(root.left) - self.getHeight(root.right) 
  
    def preOrder(self, root): 
  
        if not root: 
            return
  
        print("{0} ".format(root.val)) 
        self.preOrder(root.left) 
        self.preOrder(root.right) 
    
    def preOrderList(self, root, l): 
        if not root:
            return 
        l.append(root.val)
        self.preOrderList(root.left,l) 
        self.preOrderList(root.right,l) 
  
  
# Driver program to test above function 
myTree = AVL_Tree() 
root = None
for x in range(max(df.CityId)+1):
        root = myTree.insert(root, x)




some_list = []
myTree.preOrderList(root,some_list)
print('Total distance with the AVL tree with grid city path is '+ "{:,}".format(total_distance(df,some_list)))




def quickSort(alist):
   quickSortHelper(alist,0,len(alist)-1)

def quickSortHelper(alist,first,last):
   if first<last:

       splitpoint = partition(alist,first,last)

       quickSortHelper(alist,first,splitpoint-1)
       quickSortHelper(alist,splitpoint+1,last)


def partition(alist,first,last):
   pivotvalue = sortx_path[alist[first]]

   leftmark = first+1
   rightmark = last

   done = False
   while not done:

       while leftmark <= rightmark and sortx_path[alist[leftmark]] <= pivotvalue:
           leftmark = leftmark + 1

       while sortx_path[alist[rightmark]] >= pivotvalue and rightmark >= leftmark:
           rightmark = rightmark -1

       if rightmark < leftmark:
           done = True
       else:
           temp = alist[leftmark]
           alist[leftmark] = alist[rightmark]
           alist[rightmark] = temp

   temp = alist[first]
   alist[first] = alist[rightmark]
   alist[rightmark] = temp


   return rightmark

alist=[]
for x in range(1,max(df.CityId)+1):
        alist.append(x)
sortx_path=[]
for i in range (max(df.CityId)+1):
    sortx_path.append(df["X"][i])
quickSort(alist)
print('Total distance with the quick sort city path is '+ "{:,}".format(total_distance(df,alist)))




def nearest_neighbour():
    ids = df.CityId.values[1:]
    coordinates = np.array([df.X.values, df.Y.values]).T[1:]
    path = [0,]
    while len(ids) > 0:
        last_x, last_y = df.X[path[-1]], df.Y[path[-1]]
        dist = ((coordinates - np.array([last_x, last_y]))**2).sum(-1)
        nearest_index = dist.argmin()
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        coordinates = np.delete(coordinates, nearest_index, axis=0)
    path.append(0)
    return path

nnpath = nearest_neighbour()

print('Total distance with path sorted using Nearest Neighbour algorithm '+  "is {:,}".format(total_distance(df,nnpath)))




def distance(c1, c2):# the purpose of creating a separate function is to calculate the total distance easily
     return np.sqrt(pow((c1[1] - c2[1]),2) + pow((c1[2] - c2[2]),2))




SmallSize=df[:50]
class Vertex:
    def __init__(self,key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self,nbr):
        return self.connectedTo[nbr]




class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self,f,t,cost=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())




def createGraph(data):
    g = Graph()
    for i in range (len(data)):
        g.addVertex(data.values[i][0])
    return g




def createGraphWithEdges(data):
    g = Graph() # init graph
    for i in range(len(data) - 1): # loop through data, get distance between each vertex and others
        for j in range(i + 1, len(data)): # this loop use to reduce duplicate edge and time
            # append vertices and edge
            g.addEdge(data.values[i][0],data.values[j][0],distance(list(data.values[i]),list(data.values[j])))
    return g




sampleg = createGraphWithEdges(SmallSize)
def naiveePath(graph, path, base):
    if len(path) < graph.numVertices:
        min_dis = 100000000
        min_id = -1
        path.append(base)
        for i in range(graph.numVertices):
            if i in path:
                pass
            elif base < i:
                compare_dis = sampleg.getVertex(base).getWeight(sampleg.getVertex(i))
                if compare_dis < min_dis:
                    min_dis = compare_dis
                    min_id = i
            else:
                compare_dis = sampleg.getVertex(i).getWeight(sampleg.getVertex(base))
                if compare_dis < min_dis:
                    min_dis = compare_dis
                    min_id = i
        base = min_id
        naiveePath(graph, path, base)
    else:
        path.append(0)
        return path
first_path = []
naiveePath(sampleg, first_path, 0)
print("The total distance with using graphs is {}".format(total_distance(SmallSize,first_path)))  




def submission():
    dict = {'Path': alist}  
    df = pandas.DataFrame(dict) 
    #write data from dataframe to csv file
    df.to_csv('sample_submission.csv',index=False)
submission()

