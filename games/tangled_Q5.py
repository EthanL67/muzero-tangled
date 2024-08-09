import random

from utils import *
import datetime
import pathlib

import numpy as np
import torch
import networkx as nx
from qubobrute.core import *
from qubobrute.simulated_annealing import simulate_annealing_gpu
from pyqubo import Spin

from .abstract_game import AbstractGame

args = dotdict({
    'game_variant': 'tangled_Q5',
})

class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (3, 32, 33)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(3*80+32))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 83  # Maximum number of moves if game is not finished before
        self.num_simulations = 500  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 32  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 50  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 83  # Number of game moves to keep for every batch element
        self.td_steps = 83  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Tangled_Q5()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                act = int(
                    input(
                        f"Available actions for the player {self.to_play()}: {self.legal_actions()}. Make selection: "
                    )
                )

                if act in self.legal_actions():
                    break
            except:
                pass
            print("Wrong input, try again")
        return act

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        if action_number < 3*self.env.e:
            edge = action_number // 3 + 1
            col = action_number % 3 - 1
            return f"Play edge {edge}, color {col} (action {action_number})"
        else:
            vertex = action_number - 3*self.env.e
            return f"Play vertex {vertex} (action {action_number})"


class Tangled_Q5:
    def __init__(self):
        if args.game_variant == "tangled_K4":
            self.G = nx.complete_graph(4)
        elif args.game_variant == "tangled_P":
            self.G = nx.petersen_graph()
        elif args.game_variant == "tangled_Q5":
            self.G = nx.hypercube_graph(5)
            self.G = nx.convert_node_labels_to_integers(self.G)
        else:
            self.G = nx.complete_graph(3)

        self.v = int(nx.number_of_nodes(self.G))
        self.edges = np.array(self.G.edges, dtype=int)
        self.e = len(self.edges)
        self.adj_matrix = np.array(nx.adjacency_matrix(self.G).todense(), dtype=int)

        self.board = np.zeros((2, self.v, self.v + 1), dtype="int32")
        self.board[1, :, :-1] = self.adj_matrix
        self.board[1, :, -1] = np.ones(self.v, dtype="int32")
        self.player = 1

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = np.zeros((2, self.v, self.v + 1), dtype="int32")
        self.board[1, :, :-1] = self.adj_matrix
        self.board[1, :, -1] = np.ones(self.v, dtype="int32")
        self.player = 1
        return self.get_observation()

    def step(self, action):
        # row = action // (self.v + 1)
        # col = action % (self.v + 1)
        # self.board[row, col] = self.player
        #
        # win = self.have_winner()
        # done = win or len(self.legal_actions()) == 0
        #
        # reward = 1 if win else 0
        #
        # self.player *= -1

        # edge is played
        if action < self.e * 3:
            idx = action // 3
            color = action % 3 - 1
            # print("edge action idx: ", idx, ", color: ", color)self.board_data.

            edge_index = self.edges[idx]

            self.board[0, edge_index[0], edge_index[1]] = color
            self.board[0, edge_index[1], edge_index[0]] = color
            self.board[1, edge_index[0], edge_index[1]] = 0
            self.board[1, edge_index[1], edge_index[0]] = 0

        # vertex is played
        else:
            action -= 3 * self.e
            # print("vertex action: ", action)
            self.board[0, action, -1] = self.player
            self.board[1, action, -1] = 0

        if len(self.legal_actions()) == 0:
            win = self.have_winner()
            reward = 1 if win else 0
            done = True
        else:
            done = False
            reward = 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = np.where(self.board[0, :, :] == 1, 1, 0)
        board_player2 = np.where(self.board[0, :, :] == -1, 1, 0)
        board_to_play = np.full((self.v, self.v + 1), self.player)
        return np.array([board_player1, board_player2, board_to_play], dtype="int32")

    def legal_actions(self):
        # legal = []
        # for i in range(9):
        #     row = i // 3
        #     col = i % 3
        #     if self.board[row, col] == 0:
        #         legal.append(i)

        legal = []

        # If all edges except one have been filled, and the player has not selected a vertex, we must select a vertex
        if np.sum(self.board[1, :, :-1]) > 2 or self.player in self.board[0, :, -1]:
            # Check open edges
            for i in range(self.e):
                edge_index = self.edges[i]
                if self.board[1, edge_index[0], edge_index[1]] == 1:
                    legal.append(i * 3)
                    legal.append(i * 3 + 1)
                    legal.append(i * 3 + 2)

        # If the player has already selected a vertex, they may not select another
        if self.player not in self.board[0, :, -1]:
            # Check open vertices
            for i in range(self.v):
                if self.board[1, i, -1] == 1:
                    legal.append(3*self.e + i)

        return legal

    def have_winner(self):
        # # Horizontal and vertical checks
        # for i in range(3):
        #     if (self.board[i, :] == self.player * np.ones(3, dtype="int32")).all():
        #         return True
        #     if (self.board[:, i] == self.player * np.ones(3, dtype="int32")).all():
        #         return True
        #
        # # Diagonal checks
        # if (
        #     self.board[0, 0] == self.player
        #     and self.board[1, 1] == self.player
        #     and self.board[2, 2] == self.player
        # ):
        #     return True
        # if (
        #     self.board[2, 0] == self.player
        #     and self.board[1, 1] == self.player
        #     and self.board[0, 2] == self.player
        # ):
        #     return True
        #
        # return False

        score = self.calculateScore()

        if score * self.player > 0:
                return True    # current player won

        return False

    def calculateScore(self):
        J = self.board[0, :, :-1]
        v = self.board[0, :, -1]

        if np.all(J == 0):
            return 0

        # Define binary variables
        spins = [Spin(f'spin_{i}') for i in range(self.v)]

        # Construct the Hamiltonian
        H = 0.5 * np.sum(J * np.outer(spins, spins))

        # Compile the model to a binary quadratic model (BQM)
        model = H.compile()
        qubo, offset = model.to_qubo(index_label=True)

        if len(qubo) == 0:
            return 0

        # Determine the shape of the array (assuming you have all the indices)
        max_row = max(index[0] for index in qubo.keys()) + 1
        max_col = max(index[1] for index in qubo.keys()) + 1

        # Initialize the 2D NumPy array with zeros
        q = np.zeros((max_row, max_col))

        # Fill the array with the values from the dictionary
        for index, value in qubo.items():
            q[index] = value

        energies, solutions = simulate_annealing_gpu(q, offset, n_iter=1000, n_samples=10000, temperature=1.0,
                                                     cooling_rate=0.99)

        # Find the minimum energy
        min_energy = energies.min()

        # Find all indices with the minimum energy
        min_indices = np.where(energies == min_energy)[0]

        # Create a set to store unique solutions
        unique_solutions = set()

        for index in min_indices:
            # Convert the solution to a tuple to make it hashable
            solution_tuple = tuple(solutions[index])
            if solution_tuple not in unique_solutions:
                unique_solutions.add(solution_tuple)

        # assign an equal probability of finding each of the ground states
        prob = 1 / len(unique_solutions)

        # Convert the list of lists to a 2D NumPy array
        unique_solutions_np = np.array([list(tup) for tup in unique_solutions])

        C = np.corrcoef(unique_solutions_np, rowvar=False)

        if type(C) is np.ndarray:
            scores = np.sum(C, axis=1) - 1
            score = np.dot(scores, v)
        else:
            score = 0

        if np.isnan(score):
            score = 0

        return score

    def expert_action(self):
        return random.choice(self.legal_actions())

    def render(self):
        print(self.board[::-1])
