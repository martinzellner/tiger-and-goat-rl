import gymnasium as gym
from gymnasium import spaces
import numpy as np
        
def position_to_index(position):
    return tuple(np.array(divmod(position, 5)))

def index_to_position(index):
    row, col = index
    return row * 5 + col

def decode_move_index(x):
    if x <= 25:
        return None, x
    else:
        return (x - 25) // 25, (x - 25) % 25 

def move_index(start_pos, end_pos):
    return start_pos * 25 + end_pos + 25

class TigerGoatEnv(gym.Env):
    def __init__(self):
        super(TigerGoatEnv, self).__init__()

        self.board_size = (5, 5)
        self.max_goats = 21
        self.goats_to_place = self.max_goats
        self.goats_captured = 0
        self.regular_moves = 50

        # adjacency matrix representing the game board
        self.adj = np.array([[0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
                            1., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
                            1., 0., 0., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0.,
                            1., 1., 1., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0.,
                            0., 0., 1., 0., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0.,
                            0., 0., 1., 1., 0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                            1., 0., 0., 0., 1., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1.,
                            0., 1., 0., 0., 1., 1., 1., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                            1., 0., 1., 0., 0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.,
                            0., 1., 0., 1., 0., 0., 1., 1., 1.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                            0., 0., 1., 0., 0., 0., 0., 0., 1.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                            1., 0., 0., 0., 0., 1., 0., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            1., 0., 0., 0., 1., 0., 1., 0., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            1., 1., 1., 0., 0., 1., 0., 1., 0.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 1., 0., 0., 0., 1., 0., 1.],
                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 1., 1., 0., 0., 0., 1., 0.]])

        self.action_space = spaces.Discrete(649)

        # 0: empty
        # 1: goats
        # 2: tigers
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.board_size[0], self.board_size[1], 1), dtype=np.int32)
 
        self.reset()

    def reset(self):
        self.tigers, self.goats = self.initial_board()
        self.turn = 0  # 0: goat, 1: tiger

        self.goats_to_place = self.max_goats
        self.goats_captured = 0
        self.regular_moves = 50

        return self._get_observation()

    def step(self, action):
        if self.turn == 0:
            if self.goats_to_place > 0:
                start_pos, dst_pos = decode_move_index(action)

                # update board
                self.goats[position_to_index(dst_pos)] = 1

                self.goats_to_place -= 1
            else:
                start_pos, dst_pos = decode_move_index(action)

                # update board
                self.goats[position_to_index(start_pos)] = 0
                self.goats[position_to_index(dst_pos)] = 1

                self.regular_moves -= 1
        else:
            goat_capturing_moves = { move: goat_pos for tiger_pos in self.get_tiger_positions() for move, goat_pos in self.valid_tiger_moves(tiger_pos)}
            start_pos, end_pos = decode_move_index(action)

            # update board
            self.tigers[position_to_index(start_pos)] = 0
            self.tigers[position_to_index(end_pos)] = 1

            # if a goat was captured, the goat needs to be removed
            if goat_capturing_moves.get(end_pos):
                self.goats[position_to_index(goat_capturing_moves.get(end_pos))] = 0
                self.goats_captured += 1

                if self.goats_captured >= 5:
                    return self._get_observation(), -1, True, {}
        
        self.turn = (self.turn + 1) % 2

        # check winning conditions
        if not self.valid_tiger_moves_exist():
            print("Goat Wins!")
        if self.goats_captured >= 5:
            print("Tiger Wins!")
            
        done = self.goats_captured >= 5 or self.regular_moves <= 0 or not self.valid_tiger_moves_exist()
        reward = -1 if self.goats_captured >= 5 else 0

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """ get the current state of the board (including both goats and tigers) """
        return np.array(self.goats + 2 * self.tigers).reshape(self.board_size[0], self.board_size[1], 1)

    def render(self):
        """ render the current state """
        print(f"{'Tiger' if self.turn else 'Goat'} is playing:")
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                if self.tigers[i, j] == 1:
                    print("T", end=" ")
                elif self.goats[i, j] == 1:
                    print("G", end=" ")
                else:
                    print(".", end=" ")
            print()
        print(f"Goats to place: {self.goats_to_place}, Goats captured: {self.goats_captured}, Regular moves: {self.regular_moves}")
        print()

    def close(self):
        pass

    def initial_board(self):
        """ initialize the board with 4 tigers in the corners. """
        goats = np.zeros((5, 5))
        
        tigers = np.zeros((5,5))
        tigers[0, 0] = 1
        tigers[0, -1] = 1
        tigers[-1, 0] = 1
        tigers[-1, -1] = 1

        return tigers, goats


    def valid_goat_placement_moves(self):
        """ calculate valid goat placement moves (game phase where not all goats are placed yet. """
        occupied_spots = self.goats + self.tigers
        return list(map(index_to_position, (zip(*np.where(occupied_spots == 0)))))

    
    def valid_goat_moves(self, position):        
        # get valid moves based on board constraints
        valid_board_moves = self.get_valid_board_moves(position)
    
        occupied = self.goats + self.tigers
        # regular moves: if the destination position is not occupied by a tiger or a goat
        regular_moves = filter(lambda dst: occupied[tuple(position_to_index(dst))] == 0, valid_board_moves)
        
        return regular_moves

    def valid_tiger_moves(self, position):
        """ calculate valid moves for the tiger """

        # get valid moves based on board constraints
        valid_board_moves = self.get_valid_board_moves(position)
    
        occupied = self.goats + self.tigers

        valid_moves = set()
        for dst in valid_board_moves:
            # regular moves: if the destination position is not occupied by a tiger or a goat
            if occupied[tuple(position_to_index(dst))] == 0:
                valid_moves.add((dst, None))
                
            # jumps: if the tiger can jump over a goat and capture it
            jump_dst = self.possible_jump(position, dst)
            if jump_dst is not None:
                valid_moves.add((jump_dst, dst))

        return valid_moves  

    def possible_jump(self, start, destination):
        """
            Check for a tiger if it can perform a jump. It can jump if:
            - it moves along a straight line
            - the direct neighbor is a goat
            - the destination field of the jump is empty

        """
        
        start_idx = np.array(position_to_index(start))
        dst_idx = np.array(position_to_index(destination))

        # movement direction as a vector
        moving_direction = start_idx - dst_idx
        
        # jump destination if we continue moving in the moving direction
        jump_dst = dst_idx - moving_direction

        # check for the board limits
        if not ((0 <= jump_dst[0] < 5) and (0 <= jump_dst[1] < 5)):
            return None
        
        occupied = self.goats + self.tigers

        # check that goat is present and that the jump destination position is empty
        if self.goats[tuple(dst_idx)] == 1 and occupied[tuple(jump_dst)] == 0:
            return index_to_position(jump_dst)
        else:
            return None    

    def get_goat_positions(self):
        """ returns the positions of all goats """
        return [index_to_position(idx) for idx in zip(*self.goats.nonzero())]

    def get_tiger_positions(self):
        """ returns the positions of all tigers """
        return [index_to_position(idx) for idx in zip(*self.tigers.nonzero())]

    def valid_tiger_moves_exist(self):
        """ checks if there is valid moves for the tiger (winning condition for the goats) """
        tiger_positions = self.get_tiger_positions()
        for pos in tiger_positions:
            if self.valid_tiger_moves(pos):
                return True
        return False
        
    def get_valid_board_moves(self, position):
        """ returns all possible movement destinations based on the board structure. This does not respect state or jumps. """
        return set(self.adj[position].nonzero()[0].tolist())

    def get_valid_moves(self):
        """ returns a mask of the action vector, allowing only valid moves. """
        if self.turn == 0:
            if self.goats_to_place > 0:
                valid_moves = self.valid_goat_placement_moves()
            else:
                goat_positions = self.get_goat_positions()
                valid_moves = [move_index(goat_pos, move) for goat_pos in goat_positions for move in self.valid_goat_moves(goat_pos)]
        else:
            tiger_positions = self.get_tiger_positions()
            valid_moves = [move_index(tiger_pos, move) for tiger_pos in tiger_positions for move, goat_captured in self.valid_tiger_moves(tiger_pos)]

        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for move in valid_moves:
            mask[move] = 1
        return mask

# Register the environment with Gym
gym.envs.registration.register(
    id='TigerGoat-v0',
    entry_point='tiger_goat_env:TigerGoatEnv'
)
