"""
Sia Yi Bin
33363129
FIT2004 Assignment 1
"""

# ==========
# Q1

def function1(fitmons):
    """
    This function recursively finding the optimal combination of fitmons to fuse together 
    to get the highest cuteness score. 

    Precondition: The fitmons have not been fused together.
    Postcondition: The optimal combination of fitmons have been found.

    Input:
        fitmons: A list of fitmons, where each fitmon is represented by a list of 3 elements.
        dp: A dictionary to store the memoization of the cuteness score of the fitmons.
    Return:
        dp: A dictionary containing the optimal combination of current stage of fitmons.
    Time complexity: 
        Best case analysis: O(N^2), where N is the number of fitmons. The function will go through all the fitmons and
                            for each fitmon, it will depends on the value of left and right affinity to go through all 
                            the other fitmons to find the optimal combination. Best case when the input list is length 2,
                            return after 1 fuse.
        Worst case analysis: O(N^2), where N is the number of fitmons. Same with best case, the function need to go through
                            all the fitmons and for each fitmon, it will find the optimal combination.
    Space complexity: 
        Input space analysis: O(N), where N is the number of fitmons in the input list. 
        Aux space analysis: O(N), N is the number of fitmons, the dp dictionary will store the optimal combination of each stage of fitmons.
    """
    n = len(fitmons)
    dp = [[0] * n for _ in range(n)]
    
    # Base case
    for i in range(n):
        dp[i][i] = fitmons[i][1]  # cuteness score of single fitmon

    for length in range(2, n + 1):  # length of the subarray
        for i in range(n - length + 1):
            j = i + length - 1
            # Consider all possible fusions within this range
            for k in range(i, j):
                # Fusion happens between fitmons[k] and fitmons[k + 1]
                # Calculate affinity for fusion
                fused_affinity = min(fitmons[k][2], fitmons[k + 1][2])
                if fused_affinity > 0:
                    # Calculate cuteness for fusion
                    fused_cuteness = dp[i][k] + dp[k + 1][j]
                    dp[i][j] = max(dp[i][j], fused_cuteness)

    return dp[0][n - 1]

def fuse(fitmons):
  """
    This function call function1 to find the optimal combination of fitmons to fuse together
    and return the highest cuteness score.

    Precondition: Same as function1
    Postcondition: Same as function1

    Input:
        fitmons: A list of fitmons, where each fitmon is represented by a list of 3 elements.
    Return:
        An integer representing the highest cuteness score.

    Time complexity: 
        Best case analysis: Same as function1, O(N^2), where N is the number of fitmons.
        Worst case analysis: Same as function1, O(N^2), where N is the number of fitmons.
    Space complexity: 
        Input space analysis: O(N), where N is the number of fitmons in the input list.
        Aux space analysis: O(N), same as function1
    """
  return function1(fitmons)


# ==========
# Q2
import math
class TreeMap:
    """
    A class representing a tree map, core function escape() is used to find the optimal route from start to end.
    """

    def __init__(self, roads, solulus):
        """
        This init function initialize the graph with the given roads and solulus.

        Precondition: The graph has not been initialised.
        Postcondition: Stored a list of vertices and their corresponding edges.

        Input:
            roads: A list of tuples representing the roads between vertices with .
            solulus: A list of tuples representing the solulus.
        Return:
            None

        Time complexity: 
            Best case analysis: O(E), where E is number of roads, we loop through edges to create vertices.
            Worst case analysis: Still O(E), where E is number of roads.
        Space complexity: 
            Input space analysis: O(N+M), where N is the number of tuple in the roads and M is the number of tuple in the solulus.
            Aux space analysis: O(V), where L is number of tree, list of solulus is also near O(V), 
                                but after eliminate constant O(2V) is still O(V).
        """
        #location = vertices
        tree = 0

        for road in roads:
            tree = max(tree, road[0], road[1])

        self.vertices = [None] * (tree + 1)
        
        for i in range(tree + 1):
            self.vertices[i] = Vertex(i)
        #roads = edges
        for edge in roads:
            u = self.vertices[edge[0]]
            v = self.vertices[edge[1]]
            w = edge[2]
            
            self.vertices[u.id].edges.append(Edge(u,v,w))
        
        #adding in solulus 
        self.solulus = solulus
        
        #reverse graph
        self.reverse_vertices = [None] * (tree + 1)
        for i in range(tree + 1):
            self.reverse_vertices[i] = Vertex(i)

        for edge in roads:
            v = self.reverse_vertices[edge[0]]
            u = self.reverse_vertices[edge[1]]
            w = edge[2]
            self.reverse_vertices[u.id].edges.append(Edge(u,v,w))
        
    def escape(self, start, exits):
        """
        This function uses Dijkstra's algorithm to find the shortest path from the start vertex to the 
        most optimal solulu and then to the exit vertex.

        Precondition: Visited vertex, its corresponding distance and previous vertex are finalised.
        Postcondition: All vertices, their distance, previous vertex are finalised.

        Input:
            start: An integer representing the starting vertex
            exits: An integer list representing the exit vertices
        Return:
            (total_time, route):Returns a tuple contains with the fastest escape time and corresponding route. 
            total_time: The escape time represented by an integer.
            route: The escape route represented by a list of vertices.

        Time complexity: 
            Best case analysis: O(ElogV), E is the number of edges, V is the number of vertices. No matter what we need to go through 
                                all vertices and all edges before adding them to the MinHeap, total causes O(E)*O(logV) = O(ElogV).
                                The rest of the operations are calculate the fastest escape, O(V).
            Worst case analysis: O(ElogV), E is the number of edges, V is the number of vertices. Overall same with best case, worst when
                                reconstruct the route going through all edges, O(E), overall still O(ElogV).

        Space complexity: 
            Input space analysis: O(M), where M is the number of item in the exits list, start is just an integer. 
            Aux space analysis: O(V+E), where V is the number of vertices and E is the number of edges. O(E) space is used for the result tuple
                                and O(V) space is used for the MinHeap. Create 2 dijkstra graph cause O(2V+2E). 
                                Eliminate the constant, O(V+E) is the space complexity.
        """
        total_time = 0    
        route = []
        dummy_vertex = Vertex(len(self.reverse_vertices))
        self.reverse_vertices.append(dummy_vertex)
        self.reset()
        # add edges from dummy vertex to all exits with 0 weight
        for exit in exits:
            self.reverse_vertices[dummy_vertex.id].edges.append(Edge(dummy_vertex, self.reverse_vertices[exit], 0))
        
        # dijkstra for common graph
        origin_point = self.vertices[start]
        origin_point.distance = 0
        origin_point.discovered = True
        list_discovered = MinHeap(len(self.vertices))
        list_discovered.add(origin_point.distance, origin_point)
        while len(list_discovered) > 0:
            min_value = list_discovered.get_min()
            u = min_value[1] 
            u.visited = True
            for edge in u.edges:
                v = edge.v       
                if v.discovered == False: #means distance still infinity
                    v.distance = u.distance + edge.w
                    v.previous = u
                    v.discovered = True #v added to the queue
                    list_discovered.add(v.distance, v)
                elif v.visited == False and v.distance > u.distance + edge.w:
                    v.distance = u.distance + edge.w
                    v.previous = u
                    list_discovered.heapify(v.distance, v)
       
        # dijkstra for reverse graph
        end_point = self.reverse_vertices[len(self.reverse_vertices)-1]
        origin_point2 = end_point # Reverse start point = general end point
        origin_point2.distance = 0
        origin_point2.discovered = True
        list_discovered = MinHeap(len(self.reverse_vertices))
        list_discovered.add(origin_point2.distance, origin_point2)
        while len(list_discovered) > 0:
            min_value = list_discovered.get_min()
            u = min_value[1]
            u.visited = True
            for edge in u.edges:
                v = edge.v
                if v.discovered == False:
                    v.distance = u.distance + edge.w
                    v.previous = u
                    v.discovered = True
                    list_discovered.add(v.distance, v)
                elif v.visited == False and v.distance > u.distance + edge.w:
                    v.distance = u.distance + edge.w
                    v.previous = u
                    list_discovered.heapify(v.distance, v)
        
        fastest_escape = math.inf
        start_solulu = None
        end_solulu = None 
        # Find the fastest escape time and the corresponding solulu
        for solulu in self.solulus:
            pre_tp_v = self.vertices[solulu[0]]
            post_tp_v = self.reverse_vertices[solulu[2]]  
            # startpoint to each solulu + weight of solulu + solulu to endpoint
            if pre_tp_v.distance + post_tp_v.distance + solulu[1] < fastest_escape:         
                fastest_escape = pre_tp_v.distance + post_tp_v.distance + solulu[1]           
                start_solulu = pre_tp_v
                end_solulu = post_tp_v      
        total_time = fastest_escape
        # Reconstruct the route by backtracking
        if start_solulu == None or end_solulu == None:
            return None
        current_vertex = start_solulu
        while current_vertex != origin_point:
            route.append(current_vertex.id)
            current_vertex = current_vertex.previous
        route.append(current_vertex.id)
        route.reverse()
        if start_solulu.id != end_solulu.id:
            route.append(end_solulu.id)
        current_vertex = end_solulu 
        while current_vertex.previous != end_point:
            current_vertex = current_vertex.previous   
            route.append(current_vertex.id)

        return (total_time, route)
    
    def reset(self):
        """
        Reset general graph and reverse graph to original state.
        Precondition: The graph has been initialised.
        Postcondition: The graph has been reset to its original state.
        Input: Vertices in the graph
        Time complexity: O(V), V is the number of vertices in the graph. 
                        We need to go through all the vertices to reset them.
                        best and worst case analysis is the same.
        Input space complexity: O(V), V is the number of vertices in the graph.
        Auxiliary space complexity: O(1)
        """
        for v in self.vertices:
            v.distance = math.inf
            v.visited = False
            v.discovered = False
            v.previous = None

        for v in self.reverse_vertices:
            v.distance = math.inf
            v.visited = False
            v.discovered = False
            v.previous = None

class Edge:
    """
    A class represent an edge between two vertices in a graph
    """
    def __init__(self, u, v, w):
        """
        Constructor for the Edge class.
        Input:
             u: The starting vertex of the edge.
             v: The ending vertex of the edge.
             w: The weight of the edge.
        """
        self.u = u
        self.v = v
        self.w = w

class Vertex:
    """
    A class representing a vertex in a graph.
    """
    def __init__(self, id):
        """
        Constructor for the Vertex class.
        input: The id(an integer) of the vertex.
        """
        self.id = id
        self.edges = []
        self.distance = math.inf
        self.visited = False
        self.discovered = False
        self.previous = None

class MinHeap:
    """
    A class representing a MinHeap.
    Refer by the max heap given from FIT1008 assignment.
    https://edstem.org/au/courses/12108/lessons/42810/slides/296785
    """
    def __init__(self, vertices):
        """
        Constructor for the MinHeap class.
        Input: The number of vertices in the graph.
        """
        self.the_array = [None]
        self.length = 0
        self.index = [0] * vertices
        for i in range(vertices):
            self.index[i] = i #index of list = vertex, value = current index in the MinHeap

    def __len__(self):
        """
        Returns the length of the MinHeap.
        """
        return self.length

    def is_empty(self):
        """
        Returns True if the MinHeap is empty, False otherwise.
        """
        return self.length == 0
    
    def rise(self, k):
        """
        Rise element at index k to its correct position
        Precondition: 1 <= k <= self.length
        """
        item = self.the_array[k] 
        current_index = item[1].id
        while k > 1 and item[0] < self.the_array[k // 2][0]:
            parent_index = self.the_array[k // 2][1].id #the index of the parent
            self.the_array[k] = self.the_array[k // 2]
            self.index[parent_index] = k
            k = k // 2
        self.the_array[k] = item
        self.index[current_index] = k

    def add(self, key, item):
        """
        Swaps elements while rising
        """
        self.length += 1
        self.the_array += [None]
        self.the_array[self.length] = (key, item)
        self.rise(self.length)

    def smallest_child(self, k):
        """
        Returns the index of k's child with greatest value.
        Precondition: 1 <= k <= self.length // 2
        """
        if 2*k == self.length or self.the_array[2*k][0] < self.the_array[2*k+1][0]:
            return 2*k
        else:
            return 2*k+1
        
    def sink(self, k):
        """ 
        Make the element at index k sink to the correct position.

        Input:
            k: An integer representing the index of the element to be sunk.

        Precondition: 1 <= k <= self.length
        Postcondition: The element at index k has been sunk to its correct position.

        Time complexity: O(log V), where V is the number of vertices in the graph.
        """
        item = self.the_array[k]
        current_index = item[1].id
        while 2*k <= self.length and item[0] > self.the_array[self.smallest_child(k)][0]:
            child_index = self.the_array[self.smallest_child(k)][1].id
            self.the_array[k] = self.the_array[self.smallest_child(k)]
            self.index[child_index] = k
            k = self.smallest_child(k)
    
        self.the_array[k] = item
        self.index[current_index] = k

    def get_min(self):
        """ 
        Remove (and return) the minimum element from the heap. 
        """
        if self.length == 0:
            raise IndexError("get_min from empty heap")
        fastest_escape = self.the_array[1]
        self.length -= 1
        if self.length > 0:
            index = self.the_array[self.length+1][1].id
            self.the_array[1] = self.the_array[self.length+1]
            self.index[index] = 1
            self.sink(1)
        self.the_array.pop()
        return fastest_escape
    
    def heapify(self, key, item):
        """
        Updates the item in the heap with the key and performs a rise.
        Input:
            key: An integer representing the key of the item.
            item: An object representing the item to be updated.
        """
        index = self.index[item.id]
        self.the_array[index] = (key,self.the_array[index][1])
        self.rise(index)
# ==========
# Main Run
# The following below should be deleted before your submission, but you can use it for testing etc... I am leaving it as an example...

# Q1
# if __name__ == "__main__":
    #Example
    q1output = fuse([[0, 29, 0.9], [0.9, 91, 0.8], [0.8, 48, 0]])
    print(q1output)

# Q2
if __name__ == "__main__":
    # Example 1
    # The roads represented as a list of tuples
    roads = [(0,1,4), (1,2,2), (2,3,3), (3,4,1), (1,5,2),
    (5,6,5), (6,3,2), (6,4,3), (1,7,4), (7,8,2),
    (8,7,2), (7,3,2), (8,0,11), (4,3,1), (4,8,10)]
    # The solulus represented as a list of tuples
    solulus = [(5,10,0), (6,1,6), (7,5,7), (0,5,2), (8,4,8)]
    # Creating a TreeMap object based on the given roads
    myforest = TreeMap(roads, solulus)
    
    #Example 1.1
    start = 1
    exits = [7, 2, 4]
    q2output = myforest.escape(start, exits)
    print(q2output)