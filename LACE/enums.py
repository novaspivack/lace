# =========== START of enums.py ===========
from __future__ import annotations
import ast
import random
import setproctitle
import dataclasses
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from typing import Dict, List, Tuple, Optional, Union, TypeVar, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
plt.ioff()
from enum import Enum, auto
from dataclasses import dataclass, field
import numpy as np
import warnings
import cProfile
import pstats


warnings.filterwarnings('ignore', category=UserWarning)
_current_log_file: Optional[str] = None
# Type aliases for improved type hints
NodeIndex = int
GridArray = npt.NDArray[np.float64]
NeighborIndices = npt.NDArray[np.int64]
Coordinates = Tuple[float, ...]
StateVarType = TypeVar('StateVarType', bound=Union[bool, int, float])
_global_grid_boundary: str = 'bounded'
_CREATING_DEFAULT_LIBRARY = False



################################################
#                       ENUMS                  #
################################################


class NeighborhoodType(Enum):
    """Defines types of neighborhood relationships"""
    VON_NEUMANN = auto()  # 4 neighbors in 2D (N,S,E,W), 6 neighbors in 3D (N,S,E,W,Up,Down)
    MOORE = auto()        # 8 neighbors in 2D (N,S,E,W,NE,NW,SE,SW), 26 neighbors in 3D (all adjacent cells)
    HEX = auto()         # 6 neighbors in 2D (hexagonal grid), not valid for 3D
    HEX_PRISM = auto()   # 12 neighbors in 3D (6 in hexagonal plane + 3 above + 3 below), only valid for 3D

class Dimension(Enum):
    """Enum for dimension types"""
    TWO_D = auto()
    THREE_D = auto()

class CoordinateSystemType(Enum):
    """Enum for coordinate system types"""
    CARTESIAN = auto()
    HEXAGONAL = auto()

class StateType(Enum):
    """Defines supported state types"""
    BINARY = auto()   # Represents 0.0 or 1.0
    INTEGER = auto()  # Represents discrete integer steps (e.g., degree)
    REAL = auto()     # Represents continuous float values (e.g., 0.0-1.0 or -1.0-1.0)

class EdgeInitialization(Enum):
    """Enum for different edge initialization methods"""
    NONE = auto()    # No initial edges
    RANDOM = auto()  # Random edges based on probability
    FULL = auto()    # All neighboring nodes connected
    DISTANCE = auto() # Edges based on distance threshold
    NEAREST = auto() # Connect to nearest N neighbors
    SIMILARITY = auto() # Connect based on state similarity

class ShapeType(Enum):
    SQUARE = auto()
    CIRCLE = auto()
    LINE = auto()
    CUBE = auto()
    POLYGON = auto()
    CUSTOM = auto()
    SPHERE = auto() 
    TRIANGLE = auto()


# TODO: Fully integrate all of these tiebreker types into the app - ensure that this refactored sections works in the code as well
class TieBreaker(Enum):
    HIGHER_STATE = auto()
    LOWER_STATE = auto()
    MORE_CONNECTIONS = auto()
    FEWER_CONNECTIONS = auto()
    HIGHER_STATE_MORE_NEIGHBORS = auto()
    LOWER_STATE_FEWER_NEIGHBORS = auto()
    HIGHER_STATE_FEWER_NEIGHBORS = auto()
    LOWER_STATE_MORE_NEIGHBORS = auto()
    RANDOM = auto()
    AGREEMENT = auto()
    # --- ADDED in Round 4 ---
    SAME_CONNECTIONS = auto()           # Keep edge if degrees are equal
    MUTUAL_CONNECTIONS_EXIST = auto()   # Keep edge if nodes share a common neighbor
    NO_MUTUAL_CONNECTIONS_EXIST = auto() # Keep edge if nodes do NOT share a common neighbor
    # ---

    @staticmethod
    def resolve(node1_state, node2_state, node1_edges, node2_edges, tiebreaker_type):
        if tiebreaker_type == TieBreaker.HIGHER_STATE:
            return node1_state > node2_state
        elif tiebreaker_type == TieBreaker.LOWER_STATE:
            return node1_state < node2_state
        elif tiebreaker_type == TieBreaker.MORE_CONNECTIONS:
            return len(node1_edges) > len(node2_edges)
        elif tiebreaker_type == TieBreaker.FEWER_CONNECTIONS:
            return len(node1_edges) < len(node2_edges)
        elif tiebreaker_type == TieBreaker.HIGHER_STATE_MORE_NEIGHBORS:
            if node1_state > node2_state:
                return True
            elif node1_state == node2_state:
                return len(node1_edges) > len(node2_edges)
            return False
        elif tiebreaker_type == TieBreaker.LOWER_STATE_FEWER_NEIGHBORS:
            if node1_state < node2_state:
                return True
            elif node1_state == node2_state:
                return len(node1_edges) < len(node2_edges)
            return False
        elif tiebreaker_type == TieBreaker.HIGHER_STATE_FEWER_NEIGHBORS:
            if node1_state > node2_state:
                return True
            elif node1_state == node2_state:
                return len(node1_edges) < len(node2_edges)
            return False
        elif tiebreaker_type == TieBreaker.LOWER_STATE_MORE_NEIGHBORS:
            if node1_state < node2_state:
                return True
            elif node1_state == node2_state:
                return len(node1_edges) > len(node2_edges)
            return False
        elif tiebreaker_type == TieBreaker.RANDOM:
            return random.choice([True, False])
        elif tiebreaker_type == TieBreaker.AGREEMENT: # Keep edges resulting from main logic
            return True # Assume edge exists due to prior agreement
        # --- ADDED in Round 4 (Placeholder logic for static method) ---
        elif tiebreaker_type == TieBreaker.SAME_CONNECTIONS:
            return len(node1_edges) == len(node2_edges)
        elif tiebreaker_type == TieBreaker.MUTUAL_CONNECTIONS_EXIST:
            # Placeholder: Requires actual neighbor comparison logic here
            return True # Assume mutual connection for placeholder
        elif tiebreaker_type == TieBreaker.NO_MUTUAL_CONNECTIONS_EXIST:
            # Placeholder: Requires actual neighbor comparison logic here
            return False # Assume no mutual connection for placeholder
        # ---
        else:
            raise ValueError(f"Invalid tiebreaker type: {tiebreaker_type}")


# =========== END of enums.py ===========