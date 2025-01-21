# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #print(problem)
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST

    done = 0 # 0 if the algo still goes, 1 if found, -1 if not found

    queue = util.Stack()
    way_q = util.Stack()
    # Stack : LIFO
    #   push : self.list.append(item)
    #   pop : self.list.pop()
    #   isEmpty 


    ret_path = [] # path to returned

    start = problem.getStartState()
    
    if problem.isGoalState(start): return ret_path

    track_node = util.Stack()
    
    track_node.push([])

    next_exp = start
    
    expanded = []

    cur_long = track_node.pop()
    while not problem.isGoalState(next_exp):
        expanded.append(next_exp)
        succ = problem.getSuccessors(next_exp)

        for dir in range(len(succ)):
            new_ext = cur_long.copy()
            nbor = succ[dir] # neighbor : (state, action, cost)
            new_ext = [nbor]+new_ext
            track_node.push(new_ext)
        
        try_node = track_node.pop()
        try_exp = try_node[0][0]
        while try_exp in expanded:
            try_node = track_node.pop()
            try_exp = try_node[0][0]
        next_exp = try_exp
        cur_long = try_node
    
    for node in cur_long:
        ret_path = [node[1]]+ret_path

    print("len", len(ret_path))
    return ret_path
    """
    start = problem.getStartState()

    queue.push([start]) #track where to expand
    way_q.push([])  #track path

    expanded = []

    ret_path = []

    def do_dfs(problem: SearchProblem)->List[Directions]:


        return []
    while not queue.isEmpty() and done == 0:
        tr_cur = queue.pop()
        tr_path = way_q.pop()


        #print("cur : ",tr_cur)
        succ = problem.getSuccessors(tr_cur[0])
        #print("succ : ", succ)
        for i in range(len(succ)):
            cur_exp = succ[i][0]
            dir = succ[i][1]
            
            if cur_exp in expanded:
                continue
            expanded.append(cur_exp)
            #print("tr_cur : ",tr_cur)
            cp_ex = tr_cur.copy()
            cp_path = tr_path.copy()

            cp_path.append(dir)
            cp_ex = [cur_exp] + cp_ex
            if problem.isGoalState(cur_exp):
                ret_path = cp_path
                done = 1
                #print("Done")
            queue.push(cp_ex)
            way_q.push(cp_path)
            
    #print("Ret : ",ret_path)
    print("len : ", len(ret_path))
    return ret_path



    """




    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    """
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    stack : LIFO
    push : append
    pop : pop
    
    """

    ret_path = [] # path to return

    cur_level = util.Queue()

    start = (problem.getStartState(),"",0)

    if problem.isGoalState(start[0]): return ret_path

    cur_level.push([start])
    done = 0    
    expanded = [start[0]]
    #print("expanded : ", expanded)
    store_path = []
    while done == 0:
        cur_ext = cur_level.pop()
        cur_state = cur_ext[0]
        #expanded.append(cur_state[0])
        #check_goal = problem.isGoalState(cur_state[0])
        #print("goal : {}, cur_state  : {}".format(check_goal,cur_state))

        check_goal = problem.isGoalState(cur_state[0])
        
        if check_goal:   
            done = 1
            store_path = store_path + cur_ext
            continue
        succ = problem.getSuccessors(cur_state[0])
        for dir in range(len(succ)):
            nbor = succ[dir]
            try_exp = nbor[0]
            if try_exp in expanded:
                #print("YES, expanded : \n",expanded)
                continue
            else:
                expanded.append(try_exp)
                #print("NO\n")
            new_ext = cur_ext.copy()
            new_ext = [nbor] +new_ext
            #print(new_ext)
            cur_level.push(new_ext)
        #print("not yet : ", cur_ext)
    #print(cur_ext)
    for node in cur_ext[:-1]:
        ret_path = [node[1]]+ret_path
    return ret_path
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    ret_path = [] # path to return
    prio_dict = {}
    prio_q = util.PriorityQueue()

    init_node = problem.getStartState()
    start = (init_node, "")

    first = (0,[start])

    prio_dict[init_node] = 0
    prio_q.push(first,0) # the elements is (cost : prio_val, path) 
                        # elements in path is (node, action_to_node)
    
    done =0
    while done ==0:
        #print("hh", prio_q.heap)
        cur = prio_q.pop()
        #nexp = cur[0]
        #print("cur", cur)
        cost = cur[0]
        #print("cost", cost)
        exp = cur[1][0]
        #print("exp", exp)

        if(problem.isGoalState(exp[0])):
            done = 1
            ret_path = cur    
            continue
        elif exp[0] in prio_dict and cost > prio_dict[exp[0]]: 
            continue
        succ = problem.getSuccessors(exp[0])
    
        for nbor in succ:
            #print("nbor", nbor)
            nb_node = nbor[0]
            #print("nb_node",nb_node)
            new_cost = cost + nbor[2]
            if nb_node not in prio_dict or new_cost <= prio_dict[nb_node]:
                prio_dict[nb_node] = new_cost
                path_cp = cur[1].copy()
                #print("p1", path_cp)
                new_node = (nb_node, nbor[1])
                nnew = [new_node] + path_cp
                path_cp = [new_cost,nnew] 
                #print("p_cp", path_cp)
                prio_q.push(path_cp,new_cost)
    ret_act = []
    #print("r_p", ret_path)
    for node in ret_path[1][:-1]:
        ret_act = [node[1]]+ret_act
    #print("len : ",len(ret_act))
    return ret_act

    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    ret_path = [] # path to return
    prio_dict = {}
    prio_q = util.PriorityQueue()

    init_node = problem.getStartState()
    start = (init_node, "")

    first = (0,[start])
    print("init : ",init_node)
    prio_dict[init_node] = 0
    prio_q.push(first,0) # the elements is (cost : prio_val, path) 
                        # elements in path is (node, action_to_node)
    
    done =0
    while done ==0:
        #print("hh", prio_q.heap)
        cur = prio_q.pop()
        #nexp = cur[0]
        #print("cur", cur)
        cost = cur[0]
        #print("cost", cost)
        exp = cur[1][0]
        #print("exp", exp)

        if(problem.isGoalState(exp[0])):
            done = 1
            ret_path = cur    
            continue
        elif exp[0] in prio_dict and cost > prio_dict[exp[0]]: 
            continue
        succ = problem.getSuccessors(exp[0])
    
        for nbor in succ:
            #print("nbor", nbor)
            nb_node = nbor[0]
            #print("nb_node",nb_node)
            new_cost = cost + nbor[2] 
            heur = heuristic(nb_node,problem)
            new_f = new_cost + heur
            if nb_node not in prio_dict or new_f < prio_dict[nb_node]:
                prio_dict[nb_node] = new_f
                path_cp = cur[1].copy()
                #print("p1", path_cp)
                new_node = (nb_node, nbor[1])
                nnew = [new_node] + path_cp
                path_cp = [new_cost,nnew] 
                #print("p_cp", path_cp)
                prio_q.push(path_cp,new_f)
    ret_act = []
    #print("r_p", ret_path)
    for node in ret_path[1][:-1]:
        ret_act = [node[1]]+ret_act
    #print("len : ",len(ret_act))
    return ret_act


    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


""""
store : bfs
done = 0 # 0 if the algo still goes, 1 if found, -1 if not found

    queue = util.Stack()
    way_q = util.Stack()
    # Stack : LIFO
    #   push : self.list.append(item)
    #   pop : self.list.pop()
    #   isEmpty 
    start = problem.getStartState()

    queue.push([start]) #track where to expand
    way_q.push([])  #track path

    expanded = []

    ret_path = []

    while not queue.isEmpty() and done == 0:
        tr_cur = queue.pop()
        tr_path = way_q.pop()
        if problem.isGoalState(tr_cur[0]):
                ret_path = cp_path
                done = 1
                continue
        #print("cur : ",tr_cur)
        succ = problem.getSuccessors(tr_cur[0])
        print("succ : ", succ)
        for i in range(len(succ)):
            cur_exp = succ[i][0]
            dir = succ[i][1]
            
            print("cur_exp", cur_exp)
            if cur_exp in expanded:
                continue
            expanded.append(cur_exp)
            #print("tr_cur : ",tr_cur)
            cp_ex = tr_cur.copy()
            cp_path = tr_path.copy()

            cp_path.append(dir)
            cp_ex = [cur_exp] + cp_ex
            if problem.isGoalState(cur_exp):
                ret_path = cp_path
                done = 1
                print("Done")
            queue.push(cp_ex)
            way_q.push(cp_path)
        
    #print("Ret : ",ret_path)
    print("len : ", len(ret_path))
    return ret_path

"""


""""
Store : dfs

ret_path = [] # path to returned

    start = problem.getStartState()
    
    if problem.isGoalState(start): return ret_path

    track_node = util.Stack()
    
    track_node.push([])

    next_exp = start
    
    expanded = []

    cur_long = track_node.pop()
    while not problem.isGoalState(next_exp):
        expanded.append(next_exp)
        succ = problem.getSuccessors(next_exp)

        for dir in range(len(succ)):
            new_ext = cur_long.copy()
            nbor = succ[dir] # neighbor : (state, action, cost)
            new_ext = [nbor]+new_ext
            track_node.push(new_ext)
        
        try_node = track_node.pop()
        try_exp = try_node[0][0]
        while try_exp in expanded:
            try_node = track_node.pop()
            try_exp = try_node[0][0]
        next_exp = try_exp
        cur_long = try_node
    
    for node in cur_long:
        ret_path = [node[1]]+ret_path

    print("len", len(ret_path))
    return ret_path
 
"""