import heapq
import copy 

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    # final result is in here
    distance = 0

    # turns the state into a 2d matrix and compute mahattan distance
    for i in range(len(from_state)):
        for j in range(len(to_state)):
            # don't count dist for 0
            if from_state[i] != 0 and to_state[j] != 0 and to_state[j] == from_state[i]:
                x1, y1 = idx_to_2d(i)
                x2, y2 = idx_to_2d(j)
                manhattan_dis = abs(x1-x2) + abs(y1-y2)
                distance += manhattan_dis
    return distance

def idx_to_2d(num):
    """
    Helper function that compute a relative 2d coordinate (3x3) for an index in a list
    param: num an index in the correct
    return: x and y coordinate 
    """
    x = num // 3
    y = num % 3
    return x, y

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = [] # result variable is here
    
    # loop through the state and compute all new states
    for i in range(len(state)):
        if state[i] == 0:
            continue
        # if current index is on column 1, then possible locationto move: up, right, down
        if i % 3 == 0:
            if (i - 3) >= 0:
                if state[i - 3] == 0:
                    succ_states.append(get_state(i, i-3, state))
            if (i + 1) < len(state):
                if state[i + 1] == 0:
                    succ_states.append(get_state(i, i+1, state))
            if (i+3) < len(state):
                if state[i + 3] == 0:
                    succ_states.append(get_state(i, i+3, state))
        # if current index is on column2, then possible location to move: left, right, up, down
        elif i % 3 == 1:
            if (i -3) >= 0:
                if state[i-3] == 0:
                    succ_states.append(get_state(i, i-3, state))
            if (i - 1) >= 0:
                if state[i-1] == 0:
                    succ_states.append(get_state(i, i-1, state))
            if (i+1) < len(state):
                if state[i+1] == 0:
                    succ_states.append(get_state(i, i+1, state))
            if (i + 3) < len(state):
                if state[i+3] == 0:
                    succ_states.append(get_state(i, i+3, state))
        # if current index is on column3, then possible location to move: up, left, down
        elif i % 3 == 2:
            if (i - 3) >= 0:
                if state[i - 3] == 0:
                    succ_states.append(get_state(i, i-3, state))
            if (i - 1) >= 0:
                if state[i - 1] == 0:
                    succ_states.append(get_state(i, i-1, state))
            if (i+3) < len(state):
                if state[i + 3] == 0:
                    succ_states.append(get_state(i, i+3, state))
    return sorted(succ_states)

def get_state(idx0, idxnew, state):
    """
    Helper function to get new state given two index to switch
    param: idx0 - index to set to 0, idxnew- index where idx 0 's old value should go to. state: list of num
    """
    # not distort original state, use copy
    tmp_state = copy.deepcopy(state) 
    tmp_state[idx0] = 0
    tmp_state[idxnew] = state[idx0]
    return tmp_state

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    # priority queue
    pq = []
    # track parent index
    step = 0 
    # add start state to que
    heapq.heappush(pq, (step + get_manhattan_distance(state,goal_state), state, (step,get_manhattan_distance(state, goal_state),-1)))
    # track all visited heapq items and also state visited. Also track maximum queue length
    paths = [] 
    state_visited = [] 
    max_qlength = 0 

    # loop while we have item in pq, stop if we found the target
    while len(pq) > 0:
        # keep track of maximum queue value
        if len(pq) >= max_qlength:
            max_qlength = len(pq)
        # current state
        curr = heapq.heappop(pq)
        paths.append(curr)
        curr_state = curr[1]
        # add curr state to close
        if curr_state not in state_visited:
            state_visited.append(curr_state)
        # stop loop if we already found goal state
        if curr_state == goal_state:
            break
        
        # span on all possible next state, add to q if not already in close
        possible_states = get_succ(curr_state)
        for st in possible_states:
            if st in state_visited:
                continue
            heapq.heappush(pq,(paths[step][2][0] + 1 + get_manhattan_distance(st, goal_state), st, (paths[step][2][0]+1, get_manhattan_distance(st, goal_state), step)))
        step += 1

    # construct the path, using the parent index and the paths list we created
    result = [paths[-1]]
    i = paths[-1][2][2]
    while i != -1:
        result.append(paths[i])
        i = paths[i][2][2]
    result.reverse() # reverse the order, as it is constructed backward

    # print the result
    for idx in range(len(result)):
        print(f'{result[idx][1]} h={result[idx][2][1]} moves: {result[idx][2][0]}')
    print(f'Max queue length: {max_qlength}')
    
if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    #print_succ([2,5,1,4,0,6,7,0,3])
    #print()

    #print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    #print()
    #print(get_manhattan_distance([2,5,1,4,3,6,7,0,0], [1, 2, 3, 4, 5, 6, 7, 0, 0]))

    #solve([2,5,1,4,0,6,7,0,3])
    #print()
    solve([4,3,0,5,1,6,7,2,0])
    #solve([1,2,3,4,5,6,7,0,0])
"""
test = [[4, 6, 2, 5, 0, 0, 3, 7, 1], \
[3, 4, 0, 5, 1, 6, 2, 0, 7],\
[5, 6, 0, 7, 1, 0, 2, 4, 3],\
[5, 3, 1, 7, 6, 0, 4, 0, 2],\
[1, 3, 5, 2, 0, 0, 6, 4, 7],\
[4, 3, 2, 7, 1, 0, 0, 5, 6],\
[3, 4, 6, 0, 2, 1, 7, 0, 5],\
[4, 5, 6, 0, 7, 0, 1, 2, 3],\
[5, 2, 7, 1, 3, 0, 6, 0, 4],\
[3, 2, 4, 0, 7, 0, 1, 5, 6],\
[3, 2, 0, 6, 4, 1, 5, 0, 7],\
[3, 5, 0, 7, 2, 4, 1, 6, 0],\
[7, 1, 0, 4, 3, 6, 2, 0, 5],\
[7, 5, 6, 1, 4, 0, 3, 2, 0],\
[3, 7, 6, 4, 1, 0, 2, 0, 5]]

for case in test:
    solve(case)
"""
