import math
from typing import Tuple, List, Optional  # Support for type hints (https://docs.python.org/3/library/typing.html)
from operator import itemgetter
Node = Tuple[int, int, int, List[int]]

G_PUZZLE_TYPE = 0
G_PUZZLE_SIDE_LENGTH = 0
G_PUZZLE_GOAL_STATE = 0


class NPuzzle(object):


    def __init__(self, initial_state: List) -> None:
        """
        Constructor of an N-Puzzle object. State of N-Puzzle is represented as a 1-D list. For example:

            1 2 3
            4 8 0  ==>  [1, 2, 3, 4, 8, 0, 7, 6, 5]
            7 6 5

        The initial_state must be a valid size list containing all integers from zero to (size of list -1).
        """
        self.puzzle_type = None  # puzzle type, e.g. 8 for 8-puzzle
        self.puzzle_side_length = None  # e.g. 3 for 8-puzzle
        self.initial_state = None  # initial state of the puzzle
        self.current_state = None  # current state of the puzzle
        self.visited_states = []  # visited states

        # check if the initial state corresponds to a valid puzzle
        self.valid_state = self._is_valid_state(initial_state)

        global G_PUZZLE_GOAL_STATE
        self.goal_state = G_PUZZLE_GOAL_STATE = self._get_goal_state()


    def _is_valid_state(self, state: List) -> bool:
        """
        Check whether the list of integers representing the (initial) state of the Puzzle is valid:
        """
        puzzle_size = len(state)  # size of the puzzle

        # checking condition a) above
        sqrt_puzzle_size = int(math.sqrt(puzzle_size))  # sqrt function returns float so typecasting to int
        if puzzle_size != sqrt_puzzle_size * sqrt_puzzle_size:
            print("[_is_valid_state] ERROR: Non valid size of initial state for N-Puzzle!")
            return False

        global G_PUZZLE_TYPE, G_PUZZLE_SIDE_LENGTH
        self.initial_state = self.current_state = state
        self.puzzle_type = G_PUZZLE_TYPE = puzzle_size - 1
        self.puzzle_side_length = G_PUZZLE_SIDE_LENGTH = sqrt_puzzle_size

        return True


    def move_zero(self, state, direction: str) -> Optional[List[int]]:
        """
        Moves the element zero in the requested direction (if possible) and returns the new state.
        """
        if direction == "u":
            if self._is_zero_movable(state, "u"):
                index = state.index(0)
                state[index - self.puzzle_side_length], state[index] = state[index], state[index - self.puzzle_side_length]
        if direction == "d":
            if self._is_zero_movable(state, "d"):
                index = state.index(0)
                state[index + self.puzzle_side_length], state[index] = state[index], state[index + self.puzzle_side_length]
        if direction == "l":
            if self._is_zero_movable(state, "l"):
                index = state.index(0)
                state[index - 1], state[index] = state[index], state[index - 1]
        if direction == "r":
            if self._is_zero_movable(state, "r"):
                index = state.index(0)
                state[index + 1], state[index] = state[index], state[index + 1]
        return state


    def _is_zero_movable(self, state, direction : str) -> bool:
        """
        Checks if the element zero is movable in the requested direction for the provided state.
        """
        is_movable = False
        if direction == "u":
            index = state.index(0)
            is_movable = True if index >= self.puzzle_side_length else False
        elif direction == "d":
            index = state.index(0)
            is_movable = True if index < self.puzzle_type + 1 - self.puzzle_side_length else False
        elif direction == "l":
            index = state.index(0)
            is_movable = False if index % self.puzzle_side_length == 0 else True
        elif direction == "r":
            index = state.index(0)
            is_movable = False if index % self.puzzle_side_length == self.puzzle_side_length - 1 else True
        else:
            print("[_is_zero_movable] WARNING: Unrecognized direction option. Allowable directions are \"u\" for up, "
                  "\"d\" for down, \"l\" for left and \"r\" for right. Returned false.")
        return is_movable


    def _get_goal_state(self) -> List:
        """
        Creates fixed goal state.
        """
        goal_state = []
        for i in range(self.puzzle_type):
            goal_state.append(i+1)
        goal_state.append(0)
        return goal_state


    def is_goal_state(self, state) -> bool:
        """
        Check if goal state
        """
        self.current_state = state
        self.visited_states.append(state)
        return state == self.goal_state


    def is_visited_state(self, state) -> bool:
        """
        Check is state has been visited
        """
        return state in self.visited_states


    def print_current_state(self) -> None:
        self.print_state(self.current_state)


    def print_goal_state(self):
        self.print_state(self.goal_state)


    def print_state(self, state):
        print("*" * self.puzzle_side_length)
        for i in range(0, self.puzzle_type+1, self.puzzle_side_length):
            print("".join(str(j) for j in state[i:i+self.puzzle_side_length]))
        print("*" * self.puzzle_side_length)


def make_node(state: List, h: int=0, g: int=0, priority: int=0) -> Node:
    """
    Creates a node. A node is just a tuple of (priority, h, g, current_state)
    """
    return (priority, h, g, state)


class PriorityQueue(object):

    def __init__(self):
        self.elements = []
        self.max_queue_size = 0

    def max_size(self):
        return self.max_queue_size

    def empty(self):
        return len(self.elements) == 0

    def put(self, node: Node):
        self.elements.append(node)
        self.elements.sort(key=itemgetter(0))
        self.max_queue_size = self.max_queue_size if self.max_queue_size > len(self.elements) else len(self.elements)

    def get(self):
        return self.elements.pop(0)


def expand(node, problem) -> PriorityQueue:
    """
    Expands a node.
    """
    new_nodes = PriorityQueue()
    state_zero_up = problem.move_zero(node[3][:], "u")
    state_zero_down = problem.move_zero(node[3][:], "d")
    state_zero_left = problem.move_zero(node[3][:], "l")
    state_zero_right = problem.move_zero(node[3][:], "r")
    if state_zero_up and not problem.is_visited_state(state_zero_up):
        new_node = make_node(state_zero_up, 0, node[2] + 1, 0)
        new_nodes.put(new_node)
    if state_zero_down and not problem.is_visited_state(state_zero_down):
        new_node = make_node(state_zero_down, 0, node[2] + 1, 0)
        new_nodes.put(new_node)
    if state_zero_left and not problem.is_visited_state(state_zero_left):
        new_node = make_node(state_zero_left, 0, node[2] + 1, 0)
        new_nodes.put(new_node)
    if state_zero_right and not problem.is_visited_state(state_zero_right):
        new_node = make_node(state_zero_right, 0, node[2] + 1, 0)
        new_nodes.put(new_node)
    return new_nodes


def uniform_cost_search(nodes: PriorityQueue, new_nodes: PriorityQueue) -> None:
    """
    Uniform Cost Search (UCS).
    """
    while not new_nodes.empty():
        node = new_nodes.get()
        nodes.put(node)


def number_misplaced_tiles(state):
    """
    Calculates misplace tiles.
    """
    global G_PUZZLE_TYPE, G_PUZZLE_GOAL_STATE
    count = 0
    for i in range(G_PUZZLE_TYPE):
        if i + 1 != state[i]:
            count += 1
    return count


def misplaced_tile_heuristic(nodes, new_nodes):
    """
    Queueing function for Misplaced Tile Heuristic.
    """
    while not new_nodes.empty():
        node = new_nodes.get()
        mth_node = make_node(node[3], number_misplaced_tiles(node[3]), node[2], number_misplaced_tiles(node[3]) + node[2])
        nodes.put(mth_node)


def manhattan_distance(state):
    """
    Calculates the Manhattan Distance.
    """
    count = 0
    for i in range(G_PUZZLE_TYPE):
        index = state.index(i + 1)
        row_diff = abs((i / G_PUZZLE_SIDE_LENGTH) - (index / G_PUZZLE_SIDE_LENGTH))
        col_diff = abs((i % G_PUZZLE_SIDE_LENGTH) - (index % G_PUZZLE_SIDE_LENGTH))
        count += (row_diff + col_diff)
    index = state.index(0)
    row_diff = abs((G_PUZZLE_TYPE / G_PUZZLE_SIDE_LENGTH) - (index / G_PUZZLE_SIDE_LENGTH))
    col_diff = abs((G_PUZZLE_TYPE % G_PUZZLE_SIDE_LENGTH) - (index % G_PUZZLE_SIDE_LENGTH))
    count += (row_diff + col_diff)
    return int(count)


def manhattan_distance_heuristic(nodes, new_nodes):
    """
    Queueing function for Misplaced Tile Heuristic.
    """
    while not new_nodes.empty():
        node = new_nodes.get()
        mdh_node = make_node(node[3], manhattan_distance(node[3]), node[2], manhattan_distance(node[3]) + node[2])
        nodes.put(mdh_node)


def general_search(problem, queueing_func) -> Optional[Node]:
    """
    General search algorithm implementation.
    """
    nodes = PriorityQueue()
    initial_node = make_node(problem.initial_state)
    nodes.put(initial_node)
    while not nodes.empty():
        node = nodes.get()
        state, g, h = node[3], node[2], node[1]
        if g or h:
            print("The best state to expand with g(n) = {} and h(n) = {} is...".format(g, h))
        problem.print_state(state)
        if problem.is_goal_state(state):
            print("Reached Goal State !!!")
            print("To solve this problem the search algorithm expanded a total of {} nodes".format(len(problem.visited_states)-1))
            print("The maximum number of nodes in the queue was {}".format(nodes.max_size()))
            print("The depth of the goal node was {}".format(g))
            return node
        print("Expanding this state...")
        queueing_func(nodes, expand(node, problem))
    return None


if __name__ == "__main__":
    print("Welcome to the N-Puzzle solver!")
    print("Enter a valid type of N-Puzzle (e.g. 8 for 8-puzzle):")
    puzzle_size = int(input()) + 1
    print("Type \"1\" to use a default puzzle, or \"2\" to enter your own puzzle:")
    puzzle_choice = int(input())
    state = []
    if puzzle_choice == 1:
        state = [x for x in range(puzzle_size)]
    elif puzzle_choice == 2:
        sqrt_puzzle_size = int(math.sqrt(puzzle_size))  # sqrt function returns float so typecasting to int
        if puzzle_size != sqrt_puzzle_size * sqrt_puzzle_size:
            print("Non valid size of N-Puzzle! Must be a perfect square. Exiting...")
        for i in range(sqrt_puzzle_size):
            print("Enter elements for row {}".format(i + 1))
            state.extend([int(x) for x in input().split()])
    else:
        print("Unrecognized puzzle option! Exiting..")
        exit(0)
    problem = NPuzzle(state)
    print("Initial state:")
    problem.print_current_state()
    print("Goal state:")
    problem.print_goal_state()
    print("Select an algorithm:\n1. Uniform Cost Search\n2. A* with the Misplaced Tile heuristic.\n3. A* with the Manhattan distance heuristic.")
    method_choice = int(input())
    if method_choice == 1:
        general_search(problem, uniform_cost_search)
    elif method_choice == 2:
        general_search(problem, misplaced_tile_heuristic)
    elif method_choice == 3:
        general_search(problem, manhattan_distance_heuristic)
    else:
        print("Unrecognized method option! Exiting..")
        exit(0)
