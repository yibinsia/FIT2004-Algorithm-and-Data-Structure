"""
Sia Yi Bin
33363129
Assignment 2
version: 3.11.9
"""
# ==========
# Q1
class TrieNode:
    """
    Class to represent a Node in a Trie.
    """
    def __init__(self, size=5, data=None):
        """
        This initializes a Node of a Trie.

        Precondition: A Node hasn't been initialized yet.
        Postcondition: A Node containing a list of its children is created, and a payload is stored within it.

        Input:
            size: An integer indicating the size of the maximum number of children a Node can have.
            data: The data to be stored on the node.

        Return:
            None

        Time complexity: 
            Best case analysis: O(1), initializing the TrieNode takes constant time.
            Worst case analysis: O(1), initializing the TrieNode takes constant time.
        Space complexity: 
            Input space analysis: O(1), since input size is constant.
            Aux space analysis: O(N) where N is the size provided via input.
        """
        self.data = data
        self.link = [None] * size
        self.indices = []

class Trie:
    """
    Class to represent a Trie, with iterative approach insert function.
    This class is used to store all the suffixes of a genome sequence in a Trie.
    This implementation is refering to the live programming session of Dr. Lim Wern Han 
    with some modifications by my own.
    """
    def __init__(self):
        """
        This initializes a Trie.

        Precondition:
        Postcondition: A Trie with a root node is created.

        Input:
            None

        Return:
            None

        Time complexity: 
            Best case analysis: O(1), creating the root node with constant size array and a empty list takes constant time.
            Worst case analysis: O(1), creating the root node with constant size array and a empty list takes constant time.
        Space complexity: 
            Input space analysis: O(1), since there is no input.
            Aux space analysis: O(1), creating the constant size array for the root node and a empty list takes constant space.
        """
        self.root = TrieNode()
        self.substrings = []

    def insert(self, key, start_index, substring):
        """
        This function inserts a key into the Trie and tracks valid substrings.

        Precondition: The key has not been inserted into the Trie yet.
        Postcondition: The key is inserted into the Trie and the list substring are appended.

        Input:
            key: A string of the key to be inserted.
            start_index: An integer representing the starting index of the key in the genome.
            substring: A list to collect all substrings.

        Return:
            None

        Time complexity: 
            Best case analysis: O(N), where N is the length of the key,
                                The function will iterate over each character of the key once,the plus operation and 
                                appending string to the list are O(1) operation. The best case occurs when the node for
                                the key is all ready in the trie, no need to create a new node.

            Worst case analysis: O(N), where N is the length of the key, since the function always iterates over the key,
                                the worst case is when the key is not in the trie, and the function needs to create a new node
                                for each character of the key.
        Space complexity: 
            Input space analysis: O(N), where N is the length of the key, the input substring is just an empty list.
            Aux space analysis: O(L), where L is the length of the substring list, additional space is used to store the substrings.
                                
        """
        current = self.root
        data = ""

        for i in range(len(key)):
            char = key[i]
            index = ord(char) - ord('A') + 1
            if current.link[index] is None:
                current.link[index] = TrieNode()
            current = current.link[index]
            data += char
            current.indices.append(start_index)
            substring.append(data)

    def search(self, key):
        """
        This function search for the key in the Trie.
        Precondition: The key is not empty.
        Postcondition: The key is found in the Trie.
        Input:
            key: A string representing the key to be searched.
        Return:
            data: The data stored in the node where the key is found.
        Time complexity:
            Best case analysis: O(1), the best case occurs when the first character of the key 
                                is not found in the trie.
            Worst case analysis: O(M), where M is the length of the key, the worst case occurs 
                                when the entire key is found in the trie.
        Space complexity:
            Input space analysis: O(M), where M is the length of the key.
            Aux space analysis: O(1), the space complexity is constant, because only a pointer 
                                'current' is used.
        """
        current = self.root
        for char in key:
            index = ord(char) - ord('A') + 1
            if not current.link[index]:
                raise Exception("Key not found")
            current = current.link[index]
        return current.indices

class OrfFinder:
    def __init__(self, genome):
        """
        This initializes the OrfFinder with a given genome sequence.

        Precondition: The Trie hasn't been initialized yet.
        Postcondition: The Trie is initialized with the given genome sequence and all 
                        suffixes are inserted into the Trie.

        Input:
            genome: A string representing the genome sequence.

        Return:
            None

        Time complexity: 
            Best case analysis: The best case is same with the worst case, because the 
                                construction of the suffix trie always insert all the suffixes 
                                of the genome sequence. The insert dominates the time complexity.
                                The insertion causing the time complexity to be O(N^2), where N is
                                the length of the genome sequence, which is also the longest suffix.
                                O(N) inserting the suffixes, and O(N) for each suffix, O(N * N) = O(N^2).
            Worst case analysis: The worst case is same with the best case, O(N^2), the insert construction 
                                of the suffix trie always insert all the suffixes of the genome sequence.
        Space complexity: 
            Input space analysis: O(N), where N is the length of the genome.
            Aux space analysis: O(M^2), where M is the length of the genome sequence, constructing the suffix trie 
                                cause the space complexity to be O(M^2), because the suffix trie stores M suffixes,
                                which means having M leaves, O(M), and the longest suffix has a O(M) height, 
                                O(M * M) = O(M^2).
        """
        self.genome = genome
        self.trie = Trie()
        
        for i in range(len(self.genome)):
            substr = self.genome[i:]
            self.trie.insert(substr, i, self.trie.substrings)

    def find(self, start, end):
        """
        This function finds all valid substrings that start with the given start string and end with the given end string.

        Precondition: The Trie is initialized and all suffixes are inserted.
        Postcondition: All valid substrings are found and returned.

        Input:
            start: A string representing the start sequence.
            end: A string representing the end sequence.

        Return:
            valid_substrings: A list of all valid substrings.

        Time complexity: 
            Best case analysis: O(M*(T + U)), where M is the length of the substring list, T is the length of the start string, 
                                U is the length of the end string, this case occurs when there is no valid substring found in 
                                the list of substrings, which returns an empty list.
            Worst case analysis: O(M*(T + U) + V), where M is the length of the substring list, T is the length of the start string, 
                                U is the length of the end string, and V is the number of characters in the output list. 
                                This case occurs when all the substrings are valid, and the output list contains all the substrings. 
        Space complexity: 
            Input space analysis: O(T + U), where T is the length of the start string, U is the length of the end string,
                                as we are only using the input start and end strings and a pointer to the precomputed list.
            Aux space analysis: O(V), where V is the number of characters in the list valid_substrings.
        """
        valid_substrings = []
            
        for s in self.trie.substrings:
            if s.startswith(start) and s.endswith(end) and len(s) >= len(start) + len(end):
                valid_substrings.append(s)

        return valid_substrings
    
# ==========
# Q2
import math
class FlowNetwork:
    """
    Class to represent a Flow Network.
    """
    def __init__(self, preferences, officers_per_org, min_shifts, max_shifts):
        """
        Initializes the FlowNetwork with the given parameters, set up the nodes by 4 types: Source, Officer, CompanyShiftDay, Sink.
        Set up the edges based on the preferences and officers_per_org.

        Precondition: The FlowNetwork hasn't been initialized yet.

        Postcondition: A FlowNetwork is created with nodes and edges set up according to the given parameters.

        Input:
            preferences (a list of lists): Shift preferences for each security officer.
            officers_per_org (a list of lists): Number of officers required for each shift per company.
            min_shifts (int): Minimum number of shifts that an officer must work.
            max_shifts (int): Maximum number of shifts that an officer can work.

        Return:
            None

        Time complexity:
            Best case analysis: O(NM), where N is the total number of nodes, M is the total number of edges.
                                The best is same with the worst case, since the function always initializes all the nodes 
                                and creates all edges based on the preferences and requirements for each company.
            Worst case analysis: O(NM), where N is the total number of nodes, M is the total number of edges.
                                The same steps are performed with the best case.
        Space complexity:
            Input space analysis: O(N + M + 3 + 30) = O(N + M), where N is the length of preferences and M is length of officers_per_org.
            Aux space analysis: O(V), where V is the total number of vertices, the aux space complexity is dominated by the number of nodes
                                created in the flow network, this is due to nodes creation and edges creation, which requires V space and V edges,
                                O(V + V) = O(V).
        """
        self.preferences = preferences
        self.officers_per_org = officers_per_org
        self.min_shifts = min_shifts
        self.max_shifts = max_shifts
        self.no_of_officers = len(preferences)
        self.no_of_companys = len(officers_per_org)
        self.shifts = 3
        self.days = 30
        
        self.vertices = []
        self.vertices.append(Node("s", "Source")) 

        # Add in Officers nodes
        for i in range(self.no_of_officers):
            self.vertices.append(Node(i, "Officer"))

        # Add in Company-Shift-Day nodes
        for i in range(self.no_of_companys * self.shifts * self.days):
            self.vertices.append(Node(i, "CompanyShiftDay"))

        self.vertices.append(Node("t", "Sink")) 

        self.source = self.vertices[0]
        self.sink = self.vertices[-1]

        self.create_edges()
        
    def create_edges(self):
        """
        Creates the edges between the nodes with capacity based on the preferences and requirements.

        Precondition: The edges between each nodes haven't been created yet.
        Postcondition: Edges are added to all of the vertices.

        Input: 
            None
        Return: 
            None

        Time complexity:
            Best case analysis: O(NM), where N is the number of officers and M is the number of companies.
                                For the first for loop since the shifts (3) and days (30) are constant, 
                                total multiple up to 90 (3 * 30), so the time complexity is O(90NM), approximately O(NM).
            Worst case analysis: O(NM), where N is the number of officers and M is the number of companies.
                                The worst case is same with the best case, since the function always iterates over all the officers
                                and all the companies.
        Space complexity:
            Input space analysis: O(1), no input is used.
            Aux space analysis: O(V), where V is the number of vertices. V edges are created to connect the nodes at most.
        """
        # Create edges from officers to company-shift-day nodes based on preferences
        for i in range(len(self.preferences)):
            officer = self.preferences[i]
            for shift in range(self.shifts):
                if officer[shift] == 1: # If the officer prefers the shift create an edge
                    for day in range(self.days):
                        for company in range(self.no_of_companys):
                            index = 1 + self.no_of_officers + (company * self.shifts * self.days) + (shift * self.days) + day
                            self.vertices[i + 1].edges.append(Edge(self.vertices[i + 1], self.vertices[index], capacity=1))

        # Create edges from company-shift-day nodes to sink
        for company in range(self.no_of_companys):
            for shift in range(self.shifts):
                for day in range(self.days):
                    index = 1 + self.no_of_officers + (company * self.shifts * self.days) + (shift * self.days) + day
                    self.vertices[index].edges.append(Edge(self.vertices[index], self.sink, capacity = self.officers_per_org[company][shift]))
                    
        # Create edges from source to officers
        for i in range(self.no_of_officers):
            self.source.edges.append(Edge(self.source, self.vertices[i+1], capacity = self.max_shifts))

    def bfs(self, source, sink, bottleneck):
        """
        Performs a breadth-first search (BFS) to find an augmenting path in the flow network.
        
        Precondition: The FlowNetwork object must be initialized and edges must be created.
        Postcondition: An augmenting path is found if it exists and the flow value will be returned.
                        Otherwise, 0 is returned.

        Input:
            source (Node): The source in the flow network.
            sink (Node): The sink in the flow network.
            bottleneck (int): An infinite value to be used as the initial flow value.

        Return:
            flow: If an augmenting path is found, the flow value is returned, otherwise 0.

        Time complexity:
            Best case analysis: O(V + E), where V is the number of vertices and E is the number of edges.
                                The best and worst case are the same, since the function always iterates over all the vertices
                                and all the edges.
            Worst case analysis: O(V + E), where V is the number of vertices and E is the number of edges.
                                The same steps are performed with the best case.
        Space complexity:
            Input space analysis: O(1), only the source and sink nodes are used as input.
            Aux space analysis: O(V), where V is the number of vertices. The queue stores the vertices to be visited.
        """
        queue = []
        queue.append((source, bottleneck))
        source.visited = True
        while queue:
            u, flow = queue.pop(0)
            for edge in u.edges:
                residual = edge.capacity - edge.flow
                if not edge.v.visited and residual > 0:
                    edge.v.visited = True
                    edge.v.previous = edge
                    new_flow = min(flow, residual)
                    if edge.v == sink:
                        return new_flow
                    queue.append((edge.v, new_flow))
        return 0

    def min_cut(self, source, sink):
        """
        Finding the min cut to get the max flow in the flow network using the Ford-Fulkerson algorithm.
        Keeps finding augmenting paths to update the flow using BFS until no more augmenting paths can be found.

        Precondition: The maximum flow hasn't been calculated yet.
        Postcondition: Maximum flow returned.

        Input:
            source (Node): The source in the flow network.
            sink (Node): The sink in the flow network.

        Return:
            int: The maximum flow value.

        Time complexity:
            Best case analysis: O(V + E), where V is the number of vertices and E is the number of edges.
                                The best case occurs when bfs immediately found no augmenting path and break.
            Worst case analysis: O(VE^2), where V is the number of vertices and E is the number of edges.
                                The worst case occurs when the function iterates E times, because 
                                the flow is increased by small amount each time, the max flow is slowly reached.
                                Each bfs call is O(V + E), and the number of bfs calls is O(E), O(E * (V + E)) = O(VE^2).
        Space complexity:
            Input space analysis: O(1), as the function takes two nodes as input.
            Aux space analysis: O(V), where V is the number of vertices for the BFS queue and path tracking.
        """
        flow = 0
        while True:
            # Reset the visited flag for all vertices before next bfs is called
            for v in self.vertices:
                v.visited = False
            augment_flow = self.bfs(source, sink, math.inf)
            if augment_flow == 0: 
                break
            vertex = sink
            while vertex != source:
                edge = vertex.previous
                edge.flow += augment_flow
                edge.reverse.flow -= augment_flow
                vertex = edge.u
            flow += augment_flow
        return flow

    def adjust_constraint_demand(self):
        """
        Removes constraints and adjusts demands in the flow network.

        Precondition: The FlowNetwork has been initialized and edges has been created, the 
                        constraints and demands haven't been removed.
        Postcondition: Constraints are removed, and demands are adjusted for each vertex.

        Time complexity:
            Best case analysis: O(V + E), where V is the number of vertices and E is the number of edges,
                                the best case is same with worst case, since the function always iterates 
                                over all the edges of all vertices to adjust the constraints and demands.
            Worst case analysis: O(V + E), where V is the number of vertices and E is the number of edges.
                                The same steps are performed with the best case.
        Space complexity:
            Input space analysis: O(1), no input is used.
            Aux space analysis: O(1), no additional space is required.
        """
        for v in self.vertices:
            for e in v.edges:
                e.capacity -= e.constraint
                e.u.demand += e.constraint
                e.v.demand -= e.constraint
                e.constraint = 0
            if v.demand < 0:
                source_demand = v.demand * -1
                self.source.edges.append(Edge(self.source, v, constraint = 0, flow = 0, capacity = source_demand))
                v.demand = 0
            elif v.demand > 0:
                sink_demand = v.demand
                v.edges.append(Edge(v, self.sink, constraint = 0, flow = 0, capacity = sink_demand))
                v.demand = 0

class Node:
    """
    Class to represent a Node in the FlowNetwork.
    """
    def __init__(self, id, type, demand=0):
        """
        Initializes a Node and setting their id, type, and demand.

        Precondition: The Node hasn't been initialized yet.
        Postcondition: A Node object is created with the given attributes.

        Input:
            id (str or int): Unique identifier for the node along with type.
            type (str): Type of node (e.g., "Source", "Officer", "CompanyShiftDay", "Sink").
            demand (int): Demand value for the node (default is 0).

        Time complexity:
            Best case analysis: O(1). Initializing a node and setting attributes cause constant time operations.
            Worst case analysis: O(1). Same with best case, the time complexity is constant.

        Space complexity:
            Input space analysis: O(1). The id, type, and demand are just either a string or an integer.
            Aux space analysis: O(1). As no additional space required.
        """
        self.id = id
        self.type = type
        self.demand = demand
        self.previous = None
        self.visited = False
        self.edges = []
        

class Edge:
    """
    Class to represent an Edge in the FlowNetwork.
    """
    def __init__(self, u, v, constraint=0, flow=0, capacity=0, reversed=False):
        """
        Initializes an Edge connect with two nodes with the given flow, constraint, and capacity.

        Precondition: An Edge hasn't been initialized yet.
        Postcondition: An Edge object is created with the given attributes.

        Input:
            u (Node): Starting node of the edge.
            v (Node): Ending node of the edge.
            flow (int): Flow value for the edge (default set to 0).
            constraint (int): Constraint value for the edge (default set to 0).
            capacity (int): Capacity value for the edge (default set to 0).
            reversed (bool): Indicates if the edge is a reverse edge (default set to False).

        Time complexity:
            Best case analysis: O(1). Initializing an edge and setting attributes cause constant time operations.
            Worst case analysis: O(1). The same steps are performed in the worst case, so the time complexity is also constant.

        Space complexity:
            Input space analysis: O(1). The input space is constant, they are just integers or booleans.
            Aux space analysis: O(1). As no additional space is required.
        """
        self.u = u
        self.v = v
        self.constraint = constraint
        self.flow = flow
        self.capacity = capacity
        self.reversed = reversed
        if not reversed:
            reverse_edge = Edge(v, u, -flow, reversed=True)
            self.reverse = reverse_edge
            reverse_edge.reverse = self
            v.edges.append(reverse_edge)

def allocate(preferences, officers_per_org, min_shifts, max_shifts):
    """
    Allocates each security officers to each companies based on security officer's preferences and company's demand.

    Precondition: The allocation list hasn't been created yet.

    Postcondition: A nested list representing the allocation of officers to companies for each shift and day is returned.

    Input:
        preferences (a list of lists): Each list inside contains values 0 or 1 indicating shift preferences.
        officers_per_org (a list of lists): Each list inside contains the number of officers required for each 
                                            shift per day.
        min_shifts (int): Minimum number of shifts that an officer must work.
        max_shifts (int): Maximum number of shifts that an officer can work.

    Return:
        allocation (list of lists): A nested list representing the allocation of officers to companies for each shift and day.
                                    Returns None if no feasible allocation is found.

    Time complexity:
        Best case analysis: Initialize the FlowNetwork takes O(NM) time, where N is the number of officers and M is the number of companies.
                            Adjust the constraint demand takes O(V + E) time, where V is the number of vertices and E is the number of edges.
                            The min_cut function takes O(V + E) time in the best case, which the BFS immediately found no augmenting path.
                            To calculate the total demand takes O(N) time, where N is the number of officers. 
                            The best case occurs when the function immediately found no feasible allocation and return none,
                            Total time complexity is O(NM + V + E + N) = O(NM).

        Worst case analysis: Same steps are performed with the best case until the min_cut calling and has the worst case time complexity.
                            The min_cut function takes O(VE^2) time in the worst case, where V is the number of vertices and E is the number of edges.
                            Then the allocation is found and returned, which takes O(NM) time to construct the allocation list, where N is the number 
                            of officers and M is the number of companies, the total time complexity is O(NM + V + E + VE^2 + NM) = O(VE^2).

    Space complexity:
        Input space analysis: O(N + M), where N is the length of preferences, M is the length of officers_per_org, 
                                min_shifts, and max_shifts are just integers.

        Aux space analysis: Creating the FlowNetwork takes O(V + E) space, where V is the number of vertices, E is the number of edges.
                            The allocation list takes O(NM) space, where N is the number of officers and M is the number of companies.
                            The total space complexity is O(V + E + NM).
    """
    network = FlowNetwork(preferences, officers_per_org, min_shifts, max_shifts)
    network.adjust_constraint_demand()
    maxflow = network.min_cut(network.source, network.sink)
    
    total_demand = 0
    for shift in officers_per_org:
        total_demand += sum(shift)
    total_demand *= 30
    
    if maxflow != total_demand:
        return None # Not feasible

    allocation = []
    for _ in range(len(preferences)):
        company_allocation = []
        for _ in range(len(officers_per_org)):
            day_allocation = []
            for _ in range(30):
                shift_allocation = [0] * 3
                day_allocation.append(shift_allocation)
            company_allocation.append(day_allocation)
        allocation.append(company_allocation)

    # Allocate the officers to the companies
    for i in range(len(preferences)):
        for edge in network.vertices[i+1].edges:
            if edge.flow > 0 and not edge.reversed:
                company_shift_day_id = edge.v.id - 1 - len(preferences)
                company_id = company_shift_day_id // (3 * 30)
                shift_day_id = company_shift_day_id % (3 * 30)
                shift_id = shift_day_id // 30
                day_id = shift_day_id % 30
                allocation[i][company_id][day_id][shift_id] = 1

    return allocation
# ==========    