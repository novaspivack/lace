import hashlib
import json
import ast
import platform
import shutil
import sys
import types
import uuid
import psutil
import copy
from datetime import datetime
from math import log
import inspect
import dataclasses
from pickle import TRUE
import dill
from tkinter import simpledialog
from turtle import Turtle
import matplotlib
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import MaxNLocator, MultipleLocator
from typing import Dict, List, Set, Tuple, Optional, Type, Union, Any, Callable, cast, TypeVar, NamedTuple
import numpy as np
import numpy.typing as npt
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
plt.ioff()
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D as Axes3DType
import matplotlib.markers as markers
from matplotlib.colors import Normalize
from matplotlib.markers import MarkerStyle 
from matplotlib.colors import to_rgba
import networkx as nx # type: ignore
from contextlib import contextmanager
import asyncio
import cProfile
import pstats
from enum import Enum, auto
from dataclasses import dataclass, asdict, field, fields
from collections import defaultdict
from _collections_abc import Mapping
from abc import ABC, abstractmethod
import logging
from itertools import combinations, product
import random
import os
from datetime import datetime
import time
import traceback
from functools import wraps, lru_cache
from numba import jit, njit, prange, cuda
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, shared_memory
import setproctitle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from queue import Full, Empty
from typing_extensions import Protocol
from scipy.spatial import cKDTree # type: ignore
import warnings
import re 



warnings.filterwarnings('ignore', category=UserWarning)
_current_log_file: Optional[str] = None
# Type aliases for improved type hints
NodeIndex = int
GridArray = npt.NDArray[np.float64]
NeighborIndices = npt.NDArray[np.int64]
Coordinates = Tuple[float, ...]
StateVarType = TypeVar('StateVarType', bound=Union[bool, int, float])

class NeighborhoodType(Enum):
    """Defines types of neighborhood relationships"""
    VON_NEUMANN = auto()  # 4 neighbors in 2D (N,S,E,W), 6 neighbors in 3D (N,S,E,W,Up,Down)
    MOORE = auto()        # 8 neighbors in 2D (N,S,E,W,NE,NW,SE,SW), 26 neighbors in 3D (all adjacent cells)
    HEX = auto()         # 6 neighbors in 2D (hexagonal grid), not valid for 3D
    HEX_PRISM = auto()   # 12 neighbors in 3D (6 in hexagonal plane + 3 above + 3 below), only valid for 3D

class Dimension(Enum):
    """Defines supported dimension types"""
    TWO_D = auto()
    THREE_D = auto()

class StateType(Enum):
    """Defines supported state types"""
    BOOLEAN = auto()
    INTEGER = auto()
    REAL = auto()

class EdgeInitialization(Enum):
    """Enum for different edge initialization methods"""
    NONE = auto()    # No initial edges
    RANDOM = auto()  # Random edges based on probability
    FULL = auto()    # All neighboring nodes connected
    DISTANCE = auto() # Edges based on distance threshold
    NEAREST = auto() # Connect to nearest N neighbors
    SIMILARITY = auto() # Connect based on state similarity
    
class GlobalSettings:
    
    class Defaults:
        DEFAULT_RULE = "Majority Rule"
        NUM_SIMILAR_RULES = 3 # what does this do and are we using it?
        
        # Default rule author info
        DEFAULT_AUTHOR = "Nova Spivack"
        DEFAULT_URL = "https://novaspivack.com/network_automata"
        DEFAULT_EMAIL = "novaspivackrelay @ gmail . com" # written this way so that sourcecode in github wont attract spam

    ENABLE_TIEBREAKERS: bool = True  # Default to tiebreakers enabled; tiebreakers are rules that are used to break ties when nodes disagree about an edge
   
    class Simulation:
        
        ADMIN_MODE = bool = TRUE # allows editing of default rules with app author contact info added by default

        DIMENSION_TYPE: Dimension = Dimension.TWO_D
        
        NEIGHBORHOOD_TYPE = NeighborhoodType.MOORE  # Set default neighborhood type
        # NEIGHBORHOOD_TYPE options:
        # - NeighborhoodType.VON_NEUMANN: 4 neighbors in 2D (N,S,E,W), 6 neighbors in 3D (N,S,E,W,Up,Down)
        # - NeighborhoodType.MOORE: 8 neighbors in 2D (N,S,E,W,NE,NW,SE,SW), 26 neighbors in 3D (all adjacent cells)
        # - NeighborhoodType.HEX: 6 neighbors in 2D (hexagonal grid), not valid for 3D
        # - NeighborhoodType.HEX_PRISM: 12 neighbors in 3D (6 in hexagonal plane + 3 above + 3 below), only valid for 3D
        
        GRID_SIZE_2D = 25  # Size per dimension for 2D; 100x100 = 10,000 nodes
        GRID_SIZE_3D = 10   # Size per dimension for 3D; 10x10x10 = 1,000 nodes
        
        @classmethod
        def set_grid_size(cls, size: int, dimension_type: Dimension) -> None:
            """Sets the grid size for specified dimension type"""
            if 10 <= size <= 100:  # Enforce size limits
                if dimension_type == Dimension.TWO_D:
                    cls.GRID_SIZE_2D = size
                else:
                    cls.GRID_SIZE_3D = size
                logger.info(f"Grid size updated for {dimension_type.name}: {size}")
            else:
                raise ValueError("Grid size must be between 10 and 100")
        
        @classmethod
        def get_grid_dimensions(cls) -> Tuple[int, ...]:
            """Gets grid dimensions based on dimension type"""
            if cls.DIMENSION_TYPE == Dimension.TWO_D:
                return (cls.GRID_SIZE_2D, cls.GRID_SIZE_2D)
            else:
                return (cls.GRID_SIZE_3D, cls.GRID_SIZE_3D, cls.GRID_SIZE_3D)
        
        @classmethod
        def get_current_grid_size(cls) -> int:
            """Gets the current grid size based on dimension type"""
            return cls.GRID_SIZE_2D if cls.DIMENSION_TYPE == Dimension.TWO_D else cls.GRID_SIZE_3D
        
        NUM_STEPS = 100  # Number of simulation steps to run
        
        INITIAL_NODE_DENSITY = 0.55
        INITIAL_EDGE_DENSITY = 0.5
        
        # Milliseconds between simulation steps - lower values make the simulation run faster
        STEP_DELAY = 50 # Initial step delay value
        MIN_STEP_DELAY = 10
        MAX_STEP_DELAY = 100
        TARGET_FPS = 30
        STEP_DELAY_ADJUSTMENT_RATE = 0.1  # How quickly to adjust (10% per frame)
        
        # Initialize class variable
        _current_step_delay: int = STEP_DELAY

        @classmethod
        def reset_step_delay(cls):
            """Reset step delay to default value"""
            cls._current_step_delay = cls.STEP_DELAY

        @classmethod
        def adjust_step_delay(cls, current_frame_time: float) -> int:
            """Dynamically adjust step delay to maintain target FPS"""
            try:
                if not hasattr(cls, '_current_step_delay'):
                    cls._current_step_delay = cls.STEP_DELAY

                target_frame_time = 1.0 / cls.TARGET_FPS
                adjustment = cls.STEP_DELAY_ADJUSTMENT_RATE
                
                if current_frame_time > target_frame_time:
                    # Too slow - increase delay less aggressively
                    cls._current_step_delay = min(
                        cls.MAX_STEP_DELAY,
                        int(cls._current_step_delay * (1.0 + adjustment))
                    )
                else:
                    # Too fast - decrease delay more aggressively
                    cls._current_step_delay = max(
                        cls.MIN_STEP_DELAY,
                        int(cls._current_step_delay * (1.0 - adjustment))
                    )
                
                return cls._current_step_delay
                
            except Exception as e:
                logger.error(f"Error adjusting step delay: {e}")
                return cls.STEP_DELAY  # Return default on error
                
        # Minimum number of generations to run before checking for stability
        # Prevents premature stability detection during initial pattern formation
        MIN_GENERATIONS = 50
        
        # Number of recent generations to consider when checking for stability
        # Larger window means more stable detection but slower response
        STABILITY_WINDOW = 20
        
        # Maximum allowed variation in activity over stability window
        # Lower values require more stable patterns before stopping
        STABILITY_THRESHOLD = 0.01

        
        LOG_LEVEL = "DEBUG"
        
        # Size of chunks for parallel processing - larger chunks mean fewer
        # communication overhead but less even distribution of work
        CHUNK_SIZE = 1000
        
        ###### $$ PERFORMANNCE SETTINGS ######
        
        # Number of parallel processes to use - leaves one CPU core free
        # for system and GUI operations
        NUM_PROCESSES = 1 # max(1, mp.cpu_count() - 1)
        
        # Enable GPU acceleration if CUDA is available
        USE_GPU = cuda.is_available()
        
        # Size of LRU cache for computational results
        # Larger cache means more memory use but faster repeat calculations
        CACHE_SIZE = 1024
        
        # Size of spatial partitioning cells for neighbor lookups
        # Smaller cells mean more precise partitioning but more memory use
        SPATIAL_HASH_CELL_SIZE = 5.0
        
    class Visualization:
        EDGE_SCALE = 15.0 # this controls the zoom
        NODE_SIZE = 0.6
        NODE_SHAPE = 'sphere'
        NODE_SPACING = 0.0  # Default to no extra spacing
        MAX_NODE_SPACING = 2.0  # Maximum allowed spacing multiplier
        NODE_OPACITY = 0.8
        RESOLUTION = 20
        EDGE_WIDTH = 1.0
        EDGE_OPACITY = 0.8
        NODE_VISIBILITY_THRESHOLD = 0.0
        SHOW_INACTIVE_NODES = False
        SHOW_INACTIVE_EDGES = False
        HIGHLIGHT_DURATION = 1
        ROTATION_SPEED = 0.5
        
        ZOOM_FACTOR = 2.0
        ANIMATION_INTERVAL = 50  # Reduced for better performance
        ANIMATION_BLIT = True    # Enable blitting
        ANIMATION_CACHE = True   # Enable caching
        ANIMATION_REPEAT = False
        
        WINDOW_SIZE = (1200, 1350)  # Main application window 
        RULE_EDITOR_WINDOW_SIZE = (1350, 1000)  # Rule Editor window
        RULE_EDITOR_PARAM_WIDTH = 950  # Fixed width for the parameter column
        RULE_EDITOR_METADATA_WIDTH = 400 # Width for the metadata column
        RULE_EDITOR_BOTTOM_PADDING = 20  # Padding at bottom of parameter area
        RULE_EDITOR_HEADING_FONT_SIZE = 16 # Font size for column headings
        RULE_EDITOR_FIELD_FONT_SIZE = 15 # Font size for field names
        RULE_EDITOR_FONT_SIZE = 13  # Font size for parameter descriptions
        RULE_EDITOR_FIELD_ENTRY_WIDTH = 80  # Width of entry fields in metadata column
  
        FIGURE_SIZE = (12, 10)
        CONTROL_HEIGHT = 80
        WINDOW_POSITION = None
        WINDOW_PADDING = 20
        CONTROL_PADDING = 10
        MOUSE_ROTATION_SENSITIVITY = 0.5
        MOUSE_ZOOM_SENSITIVITY = 0.1
        
        # New Visualization Performance Settings
        USE_HARDWARE_ACCELERATION = True
        MAX_VISIBLE_NODES = 10000  # Limit for performance
        DYNAMIC_RESOLUTION = True   # Adjust resolution based on performance
        
        @classmethod
        def set_node_spacing(cls, spacing: float):
            """Set node spacing within allowed range"""
            cls.NODE_SPACING = max(0.0, min(spacing, cls.MAX_NODE_SPACING))
        
    class Colors:
        NODE_INACTIVE = '#ffffff'  # White 
        NODE_ACTIVE = '#f77b4' # Blue
        NODE_HIGHLIGHT = '#8B0000'  # Dark Red
        EDGE_NORMAL = '#2ca02c'
        EDGE_HIGHLIGHT = '#8B0000'  # Dark Red
        BACKGROUND = '#ffffff'  # White background for main area
        CONTROL_BACKGROUND = '#f0f0f0'  # Light grey for control panel
        CONTROL_BUTTON = '#e0e0e0'  # Slightly darker grey for buttons
        
    class Cache:
        ENABLE_CACHING = True
        METRIC_CACHE_SIZE = 1000
        NEIGHBOR_CACHE_SIZE = 1000
        STATE_CACHE_SIZE = 1000
        CACHE_INVALIDATION_THRESHOLD = 0.1  # 10% change triggers invalidation
        
    class SpatialHash:
        MIN_CELLS_PER_DIMENSION = 10
        MAX_CELLS_PER_DIMENSION = 1000
        TARGET_CELLS_PER_NODE = 8  # Aim for 8 nodes per cell on average
        MEMORY_THRESHOLD = 0.8  # Use up to 80% of available memory
        ADAPTATION_INTERVAL = 100  # Steps between adaptations
        MIN_CELL_SIZE = 1.0
        MAX_CELL_SIZE = 100.0
        PERFORMANCE_WINDOW = 50  # Steps to average performance over
        TARGET_QUERY_TIME = 0.001  # Target time for neighbor queries (seconds)
        
        # Auto-tuning parameters
        ENABLE_AUTO_TUNING = True
        GROWTH_FACTOR = 1.2
        SHRINK_FACTOR = 0.8
        MIN_ADAPTATION_THRESHOLD = 0.1  # Minimum performance change to trigger adaptation
        

# Enhanced logging setup with performance monitoring
class PerformanceLogger:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        
    def log_metric(self, name: str, value: float):
        self.metrics[name].append(value)
        
    def get_average(self, name: str) -> float:
        values = self.metrics[name]
        if values:
            return float(sum(values) / len(values))
        return 0.0
        
    def clear(self):
        self.metrics.clear()

perf_logger = PerformanceLogger()

class FindFontFilter(logging.Filter):
    def filter(self, record):
        return "findfont" not in record.getMessage() and not record.name.startswith('matplotlib')


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging with single log file"""
    try:
        # Check if this is a worker process
        if mp.current_process().name != 'MainProcess':
            # Worker processes should not create log files
            logger = logging.getLogger(__name__)
            if not logger.handlers:
                logger.addHandler(logging.NullHandler())
            return logger

        # Get the root logger
        root_logger = logging.getLogger()
        
        # Check if logging is already initialized in main process
        if root_logger.handlers:
            return logging.getLogger(__name__)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_level = getattr(logging, GlobalSettings.Simulation.LOG_LEVEL, logging.INFO)

        # Create single formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'
        )

        # Single file handler for all logs
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'cellular_automata_{timestamp}.log')
        )
        file_handler.setFormatter(formatter)
        
        # Console handler for INFO and above
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        # Configure root logger
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        # Configure matplotlib and numba loggers
        for logger_name in ['matplotlib', 'numba']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            find_font_filter = FindFontFilter()
            for handler in logger.handlers:
                handler.addFilter(find_font_filter)

        logger = logging.getLogger(__name__)
        logger.debug("Logging initialized with single log file: " + 
                    os.path.join(log_dir, f'cellular_automata_{timestamp}.log'))
        return logger

    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise
             
# Directory setup with enhanced error handling
APP_DIR = "GraphCA"
SUBDIRS = {
    'logs': 'logs',
    'saves': 'saves',
    'rules_backups': 'rules_backups', # Renamed directory
    'data': 'data',
    'cache': 'cache',  
    'profiles': 'profiles' 
}

def setup_directories() -> Tuple[dict, str]:
    try:
        base_path = os.path.join(os.getcwd(), APP_DIR)
        os.makedirs(base_path, exist_ok=True)
        
        paths = {}
        for key, subdir in SUBDIRS.items():
            path = os.path.join(base_path, subdir)
            os.makedirs(path, exist_ok=True)
            paths[key] = path
            
        return paths, base_path
    except Exception as e:
        print(f"Fatal error in directory setup: {str(e)}")
        raise SystemExit(1)

# Initialize directory structure and logger
APP_PATHS, BASE_PATH = setup_directories()
logger = setup_logging(APP_PATHS['logs'])

def timer_decorator(func):
    """Decorator to measure execution time of methods"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} took {execution_time:.4f} seconds")
        return result
    return wrapper

def log_errors(func):
    """Decorator to catch and log errors with context"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error in {func.__name__}: {str(e)}\n"
                f"Args: {args}, Kwargs: {kwargs}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise
    return wrapper

# Optimized decorators and computational functions
@njit(cache=True, parallel=True)
def compute_node_updates_vectorized(node_states: npt.NDArray[np.float64],
                                  neighbor_indices: npt.NDArray[np.int64]) -> npt.NDArray:
    """Vectorized computation of node state updates using Numba with parallel processing"""
    num_nodes = len(node_states)
    new_states = np.zeros(num_nodes, dtype=np.float64)

    for i in prange(num_nodes):
        neighbors = neighbor_indices[i]
        valid_neighbors = neighbors[neighbors != -1]
        if len(valid_neighbors) > 0:
            active_neighbors = np.sum(node_states[valid_neighbors] > 0)
            new_states[i] = 1.0 if active_neighbors > len(valid_neighbors) / 2 else 0.0

    return new_states

def _get_hex_vertices(self, center_x: float, center_y: float, size: float) -> np.ndarray:
    """Calculate vertices for a hexagon"""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 points
    x = center_x + size * np.cos(angles)
    y = center_y + size * np.sin(angles)
    return np.column_stack([x, y])

def _get_hex_prism_vertices(self, 
                        center_x: float, 
                        center_y: float, 
                        center_z: float,
                        size: float,
                        height: float) -> List[np.ndarray]:
    """Calculate vertices for a hexagonal prism"""
    # Get bottom and top hexagon vertices
    bottom = self._get_hex_vertices(center_x, center_y, size)
    top = self._get_hex_vertices(center_x, center_y, size)
    
    # Create faces
    faces = []
    
    # Add top and bottom faces
    faces.append(np.column_stack([bottom, np.full(6, center_z)]))
    faces.append(np.column_stack([top, np.full(6, center_z + height)]))
    
    # Add side faces
    for i in range(6):
        j = (i + 1) % 6
        face = np.array([
            [bottom[i, 0], bottom[i, 1], center_z],
            [bottom[j, 0], bottom[j, 1], center_z],
            [top[j, 0], top[j, 1], center_z + height],
            [top[i, 0], top[i, 1], center_z + height]
        ])
        faces.append(face)
        
    return faces

def get_field_metadata(param_class: Type, field_name: str) -> Optional[Dict[str, Any]]:
    """Extract metadata for a dataclass field"""
    try:
        # Try to get the metadata from class source
        source = inspect.getsource(param_class)
        lines = source.split('\n')
        
        # Find the field definition
        for i, line in enumerate(lines):
            if f"{field_name}:" in line and "=" in line:
                # Found the field definition
                # Extract metadata from the field
                parts = line.split("=", 1)
                if len(parts) > 1:
                    metadata_str = parts[1].strip()
                    if metadata_str.startswith("field(default="):
                        # Extract metadata from field() call
                        try:
                            tree = ast.parse(line)
                            assign = tree.body[0]
                            value = cast(ast.Assign, assign).value
                            
                            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'field':
                                for keyword in value.keywords:
                                    if keyword.arg == 'metadata':
                                        # Extract metadata dictionary
                                        metadata_dict = ast.literal_eval(ast.unparse(keyword.value))
                                        return metadata_dict
                        except Exception as e:
                            logger.error(f"Error parsing metadata for {field_name}: {e}")
                            logger.error(traceback.format_exc())
                            return None
                break
    except Exception as e:
        logger.error(f"Error getting metadata for {field_name}: {e}")
        logger.error(traceback.format_exc())
    
    return None

class NeighborhoodData:
    """Manages neighborhood relationships and states for the grid"""
    
    def __init__(self, dimensions: Tuple[int, ...], max_neighbors: int):
        self.dimensions = dimensions
        self.total_nodes = np.prod(dimensions)
        self.max_neighbors = max_neighbors
        
        # Core data structures
        self.states = np.zeros(self.total_nodes, dtype=np.float64)
        self.neighbor_indices = np.full((self.total_nodes, max_neighbors), -1, dtype=np.int64)
        self.edge_matrix = np.zeros((self.total_nodes, self.total_nodes), dtype=np.bool_) 
        
        # Performance tracking
        self.perf_stats = {
            'updates': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def update_state(self, node_idx: int, state: float):
        """Update state for a node using flat index"""
        # Direct update using flat index
        self.states[node_idx] = state
        self.perf_stats['updates'] += 1

    def update_edges(self, node_idx: int, new_edges: Set[int]):
        """Update edges for a node more efficiently"""
        try:
            # Clear existing edges using vectorized operations
            self.edge_matrix[node_idx] = False
            self.edge_matrix[:, node_idx] = False
            
            # Set new edges in batch
            if new_edges:
                edge_indices = np.array(list(new_edges))
                self.edge_matrix[node_idx, edge_indices] = True
                self.edge_matrix[edge_indices, node_idx] = True
                
        except Exception as e:
            logger.error(f"Error updating edges: {str(e)}")
            raise
            
    def update_edges_from_matrix(self, edge_updates: np.ndarray):
        """Update edges from a boolean matrix"""
        # Convert edge updates from neighbor format to full matrix format
        full_edge_matrix = np.zeros_like(self.edge_matrix)
        for i in range(len(edge_updates)):
            valid_neighbors = self.neighbor_indices[i] != -1
            neighbor_indices = self.neighbor_indices[i][valid_neighbors]
            for j, neighbor_idx in enumerate(neighbor_indices):
                if j < len(edge_updates[i]) and edge_updates[i][j]:
                    full_edge_matrix[i, neighbor_idx] = True
                    full_edge_matrix[neighbor_idx, i] = True  # Symmetric
                    
        # Update edge matrix
        np.logical_or(self.edge_matrix, full_edge_matrix, out=self.edge_matrix)

    def get_active_neighbor_indices(self, node_idx: int, states: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Get indices of active neighbors for a given node"""
        neighbors = self.get_neighbor_indices(node_idx)
        
        # Filter neighbors to only include valid indices within the states array
        valid_neighbors = neighbors[(0 <= neighbors) & (neighbors < len(states))]
        
        # Filter valid neighbors to only include active neighbors
        active_neighbors = valid_neighbors[states[valid_neighbors] > GlobalSettings.Visualization.NODE_VISIBILITY_THRESHOLD]
        
        return active_neighbors
    
    def get_active_edges(self, node_idx: int, states: npt.NDArray[np.float64]) -> Set[int]:
        """Get set of active edges for a given node"""
        neighbors = self.get_neighbor_indices(node_idx)
        active_edges = {n for n in neighbors if 0 <= n < len(states) and self.edge_matrix[node_idx, n] and states[n] > GlobalSettings.Visualization.NODE_VISIBILITY_THRESHOLD}
        return active_edges
    
    def get_neighbor_indices(self, node_idx: int) -> np.ndarray:
        """Get valid neighbor indices for a node"""
        return self.neighbor_indices[node_idx] [self.neighbor_indices[node_idx] != -1]
        
    def get_metric(self, node_idx: int, metric_name: str) -> float:
        """Get metric value for a node"""
        if metric_name == 'edge_density':
            neighbors = self.get_neighbor_indices(node_idx)
            if len(neighbors) < 2:
                return 0.0
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            actual_edges = np.sum(self.edge_matrix[neighbors][:, neighbors]) / 2
            return float(actual_edges / possible_edges) if possible_edges > 0 else 0.0
            
        raise ValueError(f"Unknown metric: {metric_name}")

    @njit(cache=True)
    def _calculate_neighbor_angles(self, 
                                 node_idx: int, 
                                 dimensions: Tuple[int, ...],
                                 node_positions: np.ndarray) -> np.ndarray:
        """Calculate angles of neighbors relative to a central node"""
        
        # Get coordinates of central node
        center_coords = _unravel_index(node_idx, dimensions)
        
        # Initialize array to store angles
        angles = np.full(self.max_neighbors, np.nan, dtype=np.float64)
        
        # Get neighbor indices
        neighbor_indices = self.neighbor_indices[node_idx]
        
        # Calculate angles for each valid neighbor
        for i in range(self.max_neighbors):
            if neighbor_indices[i] != -1:
                neighbor_coords = _unravel_index(neighbor_indices[i], dimensions)
                
                # Calculate angle using arctangent
                delta_x = neighbor_coords[1] - center_coords[1]
                delta_y = neighbor_coords[0] - center_coords[0]
                angle = np.arctan2(delta_y, delta_x)
                
                angles[i] = angle
                
        return angles

    def get_neighbor_angles(self, node_idx: int, node_positions: np.ndarray) -> Dict[int, float]:
        """Get angles of valid neighbors relative to a central node"""
        angles = self._calculate_neighbor_angles(node_idx, self.dimensions, node_positions)
        
        # Create dictionary mapping neighbor index to angle
        neighbor_angles = {}
        neighbor_indices = self.neighbor_indices[node_idx]
        
        for i in range(self.max_neighbors):
            if neighbor_indices[i] != -1 and not np.isnan(angles[i]):
                neighbor_angles[neighbor_indices[i]] = angles[i]
                
        return neighbor_angles
        
    def clear(self):
        """Clear all data"""
        self.states.fill(0)
        self.edge_matrix.fill(False)
        self.perf_stats['updates'] = 0
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_updates': self.perf_stats['updates'],
            'cache_hits': self.perf_stats['cache_hits'],
            'cache_misses': self.perf_stats['cache_misses'],
            'memory_usage': (
                self.states.nbytes +
                self.neighbor_indices.nbytes +
                self.edge_matrix.nbytes
            )
        }

class PerformanceMonitor:
    """Monitors and tracks performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        self.active_measurements: Set[str] = set()
        self.frame_times: List[float] = []
        self.memory_usage: List[float] = []
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 30  # seconds
        
    def update_frame_time(self, frame_time: float):
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 100:
            self.frame_times = self.frame_times[-100:]
            
    def get_average_frame_time(self) -> float:
        if not self.frame_times:
            return 0.0
        return float(np.mean(self.frame_times))

    def cleanup(self):
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self.frame_times = self.frame_times[-100:]
            self.memory_usage = self.memory_usage[-100:]
            self.last_cleanup_time = current_time
            
    @contextmanager
    def measure(self, name: str):
        """Context manager for measuring execution time"""
        try:
            self.start_measurement(name)
            yield
        finally:
            self.end_measurement(name)
            
    def start_measurement(self, name: str):
        """Start measuring a named operation"""
        if name in self.active_measurements:
            logger.warning(f"Measurement '{name}' already active")
            return
        self.start_times[name] = time.time()
        self.active_measurements.add(name)
        
    def end_measurement(self, name: str):
        """End measuring a named operation"""
        if name not in self.active_measurements:
            logger.warning(f"Measurement '{name}' not active")
            return
        duration = time.time() - self.start_times[name]
        self.metrics[name].append(duration)
        self.active_measurements.remove(name)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        for name, measurements in self.metrics.items():
            if measurements:
                stats[name] = {
                    'avg': np.mean(measurements[-100:]),  # Last 100 measurements
                    'min': np.min(measurements),
                    'max': np.max(measurements),
                    'count': len(measurements)
                }
        return stats
        
    def reset(self):
        """Reset all measurements"""
        self.metrics.clear()
        self.start_times.clear()
        self.active_measurements.clear()
         
class SimulationStats:
    """Tracks simulation statistics"""
    
    def __init__(self):
        self.stats: Dict[str, List[float]] = defaultdict(list)
        self.current_stats: Dict[str, float] = {}
        
    def update(self, **kwargs):
        """Update statistics with new values"""
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.stats[key].append(float(value))
                self.current_stats[key] = float(value)
            else:
                logger.warning(f"Skipping non-numeric stat: {key} with value: {value}")
            
    def get_recent_activity(self, window: int = 100) -> np.ndarray:
        """Get recent activity levels"""
        if 'active_ratio' in self.stats:
            return np.array(self.stats['active_ratio'][-window:])
        return np.array([])
        
    def get_current(self) -> Dict[str, float]:
        """Get current statistics"""
        return self.current_stats.copy()
        
    def reset(self):
        """Reset all statistics"""
        self.stats.clear()
        self.current_stats.clear()
        
class SpatialHashGrid:
    """Adaptive spatial partitioning for neighbor lookups with dynamic cell size adjustment"""
    
    def __init__(self, dimensions: Tuple[int, ...], initial_cell_size: Optional[float] = None):
        self.dimensions = dimensions
        self.total_volume = np.prod(dimensions)
        self.num_nodes = 0
        
        # Performance monitoring
        self.query_times: List[float] = []
        self.memory_usage: List[float] = []
        self.steps_since_adaptation = 0
        
        # Initialize cell size
        if initial_cell_size is None:
            self.cell_size = self._calculate_initial_cell_size()
        else:
            self.cell_size = initial_cell_size
            
        # Initialize data structures
        self.cells: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
        self.node_positions: Dict[int, np.ndarray] = {}
        self.kdtree: Optional[cKDTree] = None
        
        # Performance metrics
        self.last_adaptation_time = time.time()
        self.performance_metrics = {
            'query_times': [],
            'cells_per_query': [],
            'memory_usage': [],
            'load_balance': []
        }
        
    def _calculate_initial_cell_size(self) -> float:
        """Calculate optimal initial cell size based on dimension and target density"""
        avg_dimension = float(np.mean(self.dimensions))  # Explicit cast to float
        target_cells = max(
            GlobalSettings.SpatialHash.MIN_CELLS_PER_DIMENSION,
            min(
                GlobalSettings.SpatialHash.MAX_CELLS_PER_DIMENSION,
                avg_dimension / GlobalSettings.SpatialHash.TARGET_CELLS_PER_NODE
            )
        )
        return float(max(  # Explicit cast to float
            GlobalSettings.SpatialHash.MIN_CELL_SIZE,
            min(
                GlobalSettings.SpatialHash.MAX_CELL_SIZE,
                avg_dimension / target_cells
            )
        ))
        
    def _get_available_memory(self) -> float:
        """Get available system memory in bytes"""
        return psutil.virtual_memory().available
        
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage of the spatial hash grid"""
        # Estimate memory usage of core data structures
        positions_memory = len(self.node_positions) * self.dimensions[0] * 8  # 8 bytes per float64
        cells_memory = sum(len(cell) * 4 for cell in self.cells.values())  # 4 bytes per int
        kdtree_memory = len(self.node_positions) * self.dimensions[0] * 8 if self.kdtree else 0
        
        return positions_memory + cells_memory + kdtree_memory
        
    def _calculate_load_balance(self) -> float:
        """Calculate load balance factor (0-1, higher is better)"""
        if not self.cells:
            return 1.0
            
        cell_sizes = [len(cell) for cell in self.cells.values()]
        if not cell_sizes:
            return 1.0
            
        avg_size = np.mean(cell_sizes)
        if avg_size == 0:
            return 1.0
            
        variance = np.var(cell_sizes)
        return 1.0 / (1.0 + np.sqrt(variance) / avg_size)
            
    def _should_adapt(self) -> bool:
        """Determine if grid should adapt based on performance metrics"""
        if not GlobalSettings.SpatialHash.ENABLE_AUTO_TUNING:
            return False
            
        if self.steps_since_adaptation < GlobalSettings.SpatialHash.ADAPTATION_INTERVAL:
            return False
            
        if len(self.performance_metrics['query_times']) < GlobalSettings.SpatialHash.PERFORMANCE_WINDOW:
            return False
            
        # Check performance metrics
        avg_query_time = float(np.mean(self.performance_metrics['query_times'][-GlobalSettings.SpatialHash.PERFORMANCE_WINDOW:]))
        load_balance = self._calculate_load_balance()
        memory_usage = self._estimate_memory_usage()
        available_memory = self._get_available_memory()
        
        # Decision criteria
        performance_threshold = abs(avg_query_time - GlobalSettings.SpatialHash.TARGET_QUERY_TIME) / GlobalSettings.SpatialHash.TARGET_QUERY_TIME
        memory_threshold = memory_usage / available_memory
        
        # Cast the boolean expression to Python bool
        return bool(
            performance_threshold > GlobalSettings.SpatialHash.MIN_ADAPTATION_THRESHOLD or
            memory_threshold > GlobalSettings.SpatialHash.MEMORY_THRESHOLD or
            load_balance < 0.5
        )
        
    def adapt_grid(self):
        """Dynamically adapt grid based on performance metrics"""
        if not self._should_adapt():
            return
            
        avg_query_time = np.mean(self.performance_metrics['query_times'][-GlobalSettings.SpatialHash.PERFORMANCE_WINDOW:])
        load_balance = self._calculate_load_balance()
        memory_usage = self._estimate_memory_usage()
        available_memory = self._get_available_memory()
        
        # Determine adaptation strategy
        if avg_query_time > GlobalSettings.SpatialHash.TARGET_QUERY_TIME:
            # Queries too slow - increase cell size to reduce number of cells
            if memory_usage < available_memory * GlobalSettings.SpatialHash.MEMORY_THRESHOLD:
                self.cell_size *= GlobalSettings.SpatialHash.GROWTH_FACTOR
        elif load_balance < 0.5:
            # Poor load balance - decrease cell size
            if memory_usage < available_memory * GlobalSettings.SpatialHash.MEMORY_THRESHOLD:
                self.cell_size *= GlobalSettings.SpatialHash.SHRINK_FACTOR
        elif memory_usage > available_memory * GlobalSettings.SpatialHash.MEMORY_THRESHOLD:
            # Too much memory usage - increase cell size
            self.cell_size *= GlobalSettings.SpatialHash.GROWTH_FACTOR
            
        # Enforce bounds
        self.cell_size = np.clip(
            self.cell_size,
            GlobalSettings.SpatialHash.MIN_CELL_SIZE,
            GlobalSettings.SpatialHash.MAX_CELL_SIZE
        )
        
        # Rebuild grid with new cell size
        self._rebuild_grid()
        
        # Reset adaptation timer
        self.steps_since_adaptation = 0
        self.last_adaptation_time = time.time()
        
    def _rebuild_grid(self):
        """Rebuild the entire grid with current cell size"""
        old_positions = self.node_positions.copy()
        self.clear()
        
        # Reinsert all nodes
        for node_idx, position in old_positions.items():
            self.update_node(node_idx, position)
            
    @njit(cache=True)
    def _get_cell_coords(self, position: np.ndarray) -> Tuple[int, ...]:
        """Convert position to cell coordinates"""
        return tuple(int(p // self.cell_size) for p in position)
        
    def update_node(self, node_idx: int, position: np.ndarray):
        """Update node position in spatial hash grid"""
        start_time = time.time()
        
        if node_idx in self.node_positions:
            old_cell = self._get_cell_coords(self.node_positions[node_idx])
            self.cells[old_cell].remove(node_idx)
            
        self.node_positions[node_idx] = position
        new_cell = self._get_cell_coords(position)
        self.cells[new_cell].add(node_idx)
        self.kdtree = None  # Invalidate KD-tree
        
        # Update performance metrics
        self.performance_metrics['query_times'].append(time.time() - start_time)
        self.performance_metrics['cells_per_query'].append(len(self.cells))
        self.performance_metrics['memory_usage'].append(self._estimate_memory_usage())
        self.performance_metrics['load_balance'].append(self._calculate_load_balance())
        
        self.steps_since_adaptation += 1
        self.num_nodes = len(self.node_positions)
        
        # Check if adaptation is needed
        if self._should_adapt():
            self.adapt_grid()
            
    def get_nearby_nodes(self, position: np.ndarray, radius: float) -> Set[int]:
        """Get nodes within radius of position using spatial hashing"""
        start_time = time.time()
        
        if self.kdtree is None and len(self.node_positions) > 1000:  # Only use KD-tree for large datasets
            positions = np.array(list(self.node_positions.values()))
            self.kdtree = cKDTree(positions)
            if self.kdtree is not None:
                nearby_indices = set(self.kdtree.query_ball_point(position, radius))
            else:
                nearby_indices = set()
        else:
            # Use grid-based lookup for smaller datasets or when KD-tree is invalid
            cell_coords = self._get_cell_coords(position)
            search_radius = int(np.ceil(radius / self.cell_size))
            nearby_indices = set()
            
            # Search neighboring cells
            ranges = [range(max(0, c-search_radius), c+search_radius+1) 
                     for c in cell_coords]
            for neighbor_cell in product(*ranges):
                if neighbor_cell in self.cells:
                    nearby_indices.update(self.cells[neighbor_cell])
                    
            # Filter by actual distance
            nearby_indices = {idx for idx in nearby_indices 
                            if np.linalg.norm(self.node_positions[idx] - position) <= radius}
                            
        query_time = time.time() - start_time
        self.performance_metrics['query_times'].append(query_time)
        
        return nearby_indices
        
    def clear(self):
        """Clear all stored data"""
        self.cells.clear()
        self.node_positions.clear()
        self.kdtree = None
        self.performance_metrics = {
            'query_times': [],
            'cells_per_query': [],
            'memory_usage': [],
            'load_balance': []
        }
        self.steps_since_adaptation = 0
        self.num_nodes = 0
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            'cell_size': self.cell_size,
            'num_cells': len(self.cells),
            'num_nodes': self.num_nodes,
            'avg_query_time': np.mean(self.performance_metrics['query_times'][-100:]) if self.performance_metrics['query_times'] else 0,
            'load_balance': self._calculate_load_balance(),
            'memory_usage': self._estimate_memory_usage(),
            'adaptations': self.steps_since_adaptation
        }


@njit(cache=True)
def _unravel_index(idx: int, dimensions: Tuple[int, ...]) -> np.ndarray:
    """Convert flat index to coordinates"""
    coords = np.zeros(len(dimensions), dtype=np.int64)
    for i in range(len(dimensions)-1, -1, -1):
        coords[i] = idx % dimensions[i]
        idx //= dimensions[i]
    return coords

@njit(cache=True)
def _ravel_multi_index(coords: np.ndarray, dimensions: Tuple[int, ...]) -> int:
    """Convert coordinates to flat index"""
    idx = 0
    multiplier = 1
    for i in range(len(dimensions)-1, -1, -1):
        idx += coords[i] * multiplier
        multiplier *= dimensions[i]
    return idx

@njit(cache=True)
def _get_moore_neighbors(idx: int, dimensions: Tuple[int, ...]) -> npt.NDArray[np.int64]:
    """Calculate Moore neighborhood indices"""
    coords = _unravel_index(idx, dimensions)
    ndim = len(dimensions)
    
    if ndim == 2:
        # Preallocate NumPy array for neighbors
        neighbors = np.empty(8, dtype=np.int64)
        num_neighbors = 0
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                    
                new_coords = np.array([coords[0] + dx, coords[1] + dy], dtype=np.int64)
                
                if (new_coords[0] >= 0 and new_coords[0] < dimensions[0] and 
                    new_coords[1] >= 0 and new_coords[1] < dimensions[1]):
                    neighbors[num_neighbors] = _ravel_multi_index(new_coords, dimensions)
                    num_neighbors += 1
        
        # Resize the array to the actual number of neighbors
        return neighbors[:num_neighbors]
    
    elif ndim == 3:
        # Preallocate NumPy array for neighbors
        neighbors = np.empty(26, dtype=np.int64)
        num_neighbors = 0
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                        
                    new_coords = np.array([coords[0] + dx, coords[1] + dy, coords[2] + dz], dtype=np.int64)
                    
                    if (new_coords[0] >= 0 and new_coords[0] < dimensions[0] and 
                        new_coords[1] >= 0 and new_coords[1] < dimensions[1] and
                        new_coords[2] >= 0 and new_coords[2] < dimensions[2]):
                        neighbors[num_neighbors] = _ravel_multi_index(new_coords, dimensions)
                        num_neighbors += 1
        
        # Resize the array to the actual number of neighbors
        return neighbors[:num_neighbors]
    
    return np.empty(0, dtype=np.int64)  # Return empty array with correct type

@njit(cache=True)
def _get_von_neumann_neighbors(idx: int, dimensions: Tuple[int, ...]) -> npt.NDArray[np.int64]:
    """Calculate von Neumann neighborhood indices"""
    coords = _unravel_index(idx, dimensions)
    ndim = len(dimensions)
    
    # Preallocate NumPy array for neighbors
    if ndim == 2:
        neighbors = np.empty(4, dtype=np.int64)
    else:
        neighbors = np.empty(6, dtype=np.int64)
    num_neighbors = 0
    
    for dim in range(ndim):
        for offset in [-1, 1]:
            new_coords = coords.copy()
            new_coords[dim] += offset
            
            valid = True
            for i in range(ndim):
                if new_coords[i] < 0 or new_coords[i] >= dimensions[i]:
                    valid = False
                    break
                    
            if valid:
                neighbors[num_neighbors] = _ravel_multi_index(new_coords, dimensions)
                num_neighbors += 1
                
    return neighbors[:num_neighbors] if num_neighbors > 0 else np.empty(0, dtype=np.int64)

@njit(cache=True)
def _get_hex_neighbors(idx: int, dimensions: Tuple[int, ...]) -> npt.NDArray[np.int64]:
    """Calculate hexagonal neighborhood indices (2D only)"""
    if len(dimensions) != 2:
        return np.empty(0, dtype=np.int64)
    
    # Suppose dimensions = (n_rows, n_cols)
    coords = _unravel_index(idx, dimensions)
    row, col = coords[0], coords[1]
    n_rows, n_cols = dimensions

    # Then if we are using an "odd row" offset layout:
    is_odd_row = (row % 2 == 1)

    # Offsets for HEX (odd-r horizontal layout, pointy-topped)
    if is_odd_row:
        offsets = np.array([
            [-1,  0],  # NW
            [ 0, -1],  # N
            [ 1,  0],  # NE
            [-1,  1],  # SW
            [ 0,  1],  # S
            [ 1,  1]   # SE
        ], dtype=np.int64)
    else:
        offsets = np.array([
            [-1, -1],  # NW
            [ 0, -1],  # N
            [ 1, -1],  # NE
            [-1,  0],  # SW
            [ 0,  1],  # S
            [ 1,  0]   # SE
        ], dtype=np.int64)

    # Preallocate NumPy array for neighbors
    neighbors = np.empty(6, dtype=np.int64)
    num_neighbors = 0

    for offset in offsets:
        new_r = row + offset[0]
        new_c = col + offset[1]
        if 0 <= new_r < n_rows and 0 <= new_c < n_cols:
            neighbor_idx = _ravel_multi_index(np.array([new_r, new_c]), dimensions)
            if neighbor_idx != idx:
                neighbors[num_neighbors] = neighbor_idx
                num_neighbors += 1
                
    return neighbors[:num_neighbors] if num_neighbors > 0 else np.empty(0, dtype=np.int64)

@njit(cache=True)
def _get_hex_prism_neighbors(idx: int, dimensions: Tuple[int, int, int]) -> npt.NDArray[np.int64]:
    """Calculate hexagonal prism neighborhood indices (3D only)"""
    if len(dimensions) != 3:
        return np.empty(0, dtype=np.int64)
            
    coords = _unravel_index(idx, dimensions)
    x, y, z = coords[0], coords[1], coords[2]
    width, height, depth = dimensions
    
    # Preallocate NumPy array for neighbors
    neighbors = np.empty(12, dtype=np.int64)
    num_neighbors = 0
    
    # Get in-plane hex neighbors (same as 2D hex grid)
    is_odd_row = y % 2 == 1
    
    # Create all offsets as NumPy arrays
    if is_odd_row:
        planar_offsets = np.array([
            [-1, -1, 0],  # Northwest
            [0, -1, 0],   # Northeast
            [-1, 0, 0],   # West
            [1, 0, 0],    # East
            [-1, 1, 0],   # Southwest
            [0, 1, 0],    # Southeast
        ], dtype=np.int64)
        
        vertical_offsets = np.array([
            [0, 0, 1],     # Directly above
            [-1, 0, 1],    # Above-left
            [0, -1, 1],    # Above-right
            [0, 0, -1],    # Directly below
            [-1, 1, -1],   # Below-left
            [0, 1, -1],    # Below-right
        ], dtype=np.int64)
    else:
        planar_offsets = np.array([
            [0, -1, 0],    # Northwest
            [1, -1, 0],    # Northeast
            [-1, 0, 0],    # West
            [1, 0, 0],     # East
            [0, 1, 0],     # Southwest
            [1, 1, 0],     # Southeast
        ], dtype=np.int64)
        
        vertical_offsets = np.array([
            [0, 0, 1],     # Directly above
            [0, -1, 1],    # Above-left
            [1, 0, 1],     # Above-right
            [0, 0, -1],    # Directly below
            [0, 1, -1],    # Below-left
            [1, 1, -1],    # Below-right
        ], dtype=np.int64)
    
    # Combine offsets
    all_offsets = np.vstack((planar_offsets, vertical_offsets))
    
    # Check each potential neighbor
    for offset in all_offsets:
        new_x = x + offset[0]
        new_y = y + offset[1]
        new_z = z + offset[2]
        
        # Validate bounds
        if (0 <= new_x < width and 
            0 <= new_y < height and 
            0 <= new_z < depth):
            neighbor_idx = _ravel_multi_index(np.array([new_x, new_y, new_z]), dimensions)
            neighbors[num_neighbors] = neighbor_idx
            num_neighbors += 1
    
    return neighbors[:num_neighbors] if num_neighbors > 0 else np.empty(0, dtype=np.int64)

class Grid:
    """Optimized grid implementation using NumPy arrays and parallel processing"""
                            
    def __init__(self, dimensions, neighborhood_type, dimension_type, rule, unique_id=None):
        """Initialize grid with specified dimensions and neighborhood type"""
        try:
            # Set rule and dimensions
            self.rule = rule
            self.dimensions = dimensions
            self.neighborhood_type = neighborhood_type
            self.dimension_type = dimension_type
            self.unique_id = unique_id
            self._unique_id = unique_id  # Store also as _unique_id for compatibility
            
            # Create grid array
            total_cells = np.prod(dimensions)
            self.grid_array = np.full(dimensions, -1.0, dtype=np.float64)
            
            # Store total nodes count
            self.total_nodes = total_cells
            
            # Store max neighbors based on neighborhood type
            if neighborhood_type == NeighborhoodType.VON_NEUMANN:
                self.max_neighbors = 4 if dimension_type == Dimension.TWO_D else 6
            elif neighborhood_type == NeighborhoodType.MOORE:
                self.max_neighbors = 8 if dimension_type == Dimension.TWO_D else 26
            elif neighborhood_type == NeighborhoodType.HEX:
                self.max_neighbors = 6
            elif neighborhood_type == NeighborhoodType.HEX_PRISM:
                self.max_neighbors = 14
            else:
                self.max_neighbors = 8  # Default
            
            # Create neighborhood data - check if it needs 2 or 3 arguments
            try:
                # Try with 3 arguments first
                self.neighborhood_data = NeighborhoodData(total_cells, self.max_neighbors)
            except TypeError:
                # Fall back to 2 arguments if needed
                logger.debug(f"NeighborhoodData.__init__ requires 2 arguments, falling back")
                self.neighborhood_data = NeighborhoodData(total_cells, self.max_neighbors)
            
            # Initialize performance monitoring
            self.grid_stats = {
                'avg_update_time': 0.0,
                'last_update_time': 0.0,
                'update_times': [],
                'avg_step_time': 0.0,
                'last_step_time': 0.0,
                'step_times': []
            }
            
            # Initialize shared memory vars
            self.shared_mem = None
            self.shared_array = None
            
            # Add update lock for thread safety
            self._update_lock = threading.Lock()
            
            # Set flags
            self._is_edge_initialized = False
            
            # Initialization time tracking
            self.initialization_time = 0.0
            
            # Add a flag to track if shared memory has been unlinked
            self._shared_memory_unlinked = False
            # Add a flag to track if process pool has been shut down
            self._shutdown_complete = False
            
        except Exception as e:
            logger.error(f"Error initializing grid: {e}")
            raise
                    
    def _initialize_process_pool(self):
        """Initialize the process pool for parallel processing"""
        try:
            if self.process_pool is None:
                if GlobalSettings.Simulation.NUM_PROCESSES > 1:
                    self.process_pool = ProcessPoolExecutor(
                        max_workers=GlobalSettings.Simulation.NUM_PROCESSES,
                        initializer=Grid._process_initializer)
                    logger.info(f"Initialized process pool with {GlobalSettings.Simulation.NUM_PROCESSES} workers")
                else:
                    self.process_pool = ThreadPoolExecutor(max_workers=1)
                    logger.info("Initialized single-threaded executor")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize process pool: {e}")
            # Fallback to single threaded
            try:
                self.process_pool = ThreadPoolExecutor(max_workers=1)
                logger.info("Falling back to single-threaded executor after error")
                return True
            except Exception as e2:
                logger.error(f"Failed to initialize even the fallback thread pool: {e2}")
                return False

    def setup_shared_memory(self):
        """Setup shared memory for grid data"""
        try:
            # Use the unique ID stored in the Grid instance
            shared_mem_name = f'/{self._unique_id}' # Shorten the name
            self._shared_mem_name = shared_mem_name # Store shared memory name
            
            # Calculate the size of the shared memory block
            array_size = self.grid_array.nbytes
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Attempt to create the shared memory block
                    self.shared_mem = shared_memory.SharedMemory(name=shared_mem_name, create=True, size=array_size)
                    logger.info(f"Created shared memory block: {shared_mem_name} of size {array_size} bytes")
                    break  # Exit the retry loop if successful
                except FileExistsError:
                    logger.warning(f"Shared memory block already exists, attempting to connect: {shared_mem_name}")
                    try:
                        # Attempt to connect to the existing shared memory block
                        self.shared_mem = shared_memory.SharedMemory(name=shared_mem_name, create=False, size=array_size)
                        logger.info(f"Successfully connected to existing shared memory: {shared_mem_name}")
                        
                        # Create a NumPy array backed by shared memory
                        self.shared_array = np.ndarray(self.grid_array.shape, dtype=self.grid_array.dtype, buffer=self.shared_mem.buf)
                        
                        # Verify data integrity
                        if self.shared_array.shape != self.grid_array.shape or self.shared_array.dtype != self.grid_array.dtype:
                            logger.warning(f"Incompatible shared memory block found, attempting to recreate")
                            self.shared_mem.close()
                            self.shared_mem.unlink()
                            self.shared_mem = None
                            self.shared_array = None
                            continue  # Retry creation
                        else:
                            logger.info("Shared memory data integrity verified")
                            break  # Exit the retry loop if successful
                            
                    except Exception as e:
                        logger.error(f"Error connecting to existing shared memory: {e}")
                        # Attempt to unlink and retry
                        try:
                            if self.shared_mem is not None:
                                self.shared_mem.close()
                                self.shared_mem.unlink()
                                self.shared_mem = None
                                self.shared_array = None
                                logger.info("Existing shared memory block unlinked, retrying creation")
                        except Exception as unlink_error:
                            logger.error(f"Error unlinking shared memory: {unlink_error}")
                        continue  # Retry creation
                else:
                    break # Exit the retry loop if successful
            else:
                logger.error(f"Failed to create or connect to shared memory after {max_retries} attempts")
                raise Exception("Failed to create or connect to shared memory")
            
            # Create a NumPy array backed by shared memory
            self.shared_array = np.ndarray(self.grid_array.shape, dtype=self.grid_array.dtype, buffer=self.shared_mem.buf)
            
            # Copy the data to shared memory
            np.copyto(self.shared_array, self.grid_array)
            
            logger.debug(f"Grid data copied to shared memory: {shared_mem_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup shared memory: {e}")
            raise
                            
    @staticmethod
    def _init_worker():
        """Initialize worker process"""
        try:
            worker_id = mp.current_process().name
            print(f"Worker {worker_id} initializing")
            root_logger = logging.getLogger()
            
            # Log existing handlers before clearing
            print(f"Worker {worker_id} handlers before clearing: {[str(h) for h in root_logger.handlers]}")
            
            # Disable logging in worker processes
            root_logger.handlers = []
            root_logger.addHandler(logging.NullHandler())
            
            # Log final handler state
            print(f"Worker {worker_id} final handlers: {[str(h) for h in root_logger.handlers]}")
            
        except Exception as e:
            print(f"Error in worker initialization: {e}")
        
    def cleanup(self):
        """Clean up shared resources"""
        try:
            # Clean up shared memory
            if self.shared_mem is not None and not self._shared_memory_unlinked:
                shared_mem_name = self._shared_mem_name  # Use stored shared memory name
                logger.info(f"Attempting to clean up shared memory: {shared_mem_name}")
                try:
                    self.shared_mem.close()
                    self.shared_mem.unlink()
                    logger.info(f"Shared memory {shared_mem_name} unlinked successfully")
                    self._shared_memory_unlinked = True  # Set the flag
                except Exception as e:
                    logger.error(f"Error cleaning up shared memory: {e}")
                finally:
                    self.shared_mem = None
                    self.shared_array = None
            else:
                if self._shared_memory_unlinked:
                    logger.info("Shared memory already unlinked, skipping cleanup")
                else:
                    logger.info("No shared memory to clean up, skipping cleanup")

            # Clean up process pool
            if self.process_pool is not None:
                try:
                    self.process_pool.shutdown(wait=True, cancel_futures=True)
                    logger.info("Process pool shut down successfully")
                except Exception as e:
                    logger.error(f"Error shutting down process pool: {e}")
                finally:
                    self.process_pool = None
                    self._shutdown_complete = True # Set shutdown complete flag

        except Exception as e:
            logger.error(f"Error in Grid cleanup: {e}")

            # Clean up process pool
            if self.process_pool is not None:
                try:
                    self.process_pool.shutdown(wait=True, cancel_futures=True)
                    logger.info("Process pool shut down successfully")
                except Exception as e:
                    logger.error(f"Error shutting down process pool: {e}")
                finally:
                    self.process_pool = None
                    self._shutdown_complete = True # Set shutdown complete flag

    def initialize_edges(self, initialization_type=None):
        """Initialize edge connections between nodes based on rule parameters"""
        if initialization_type is None:
            initialization_type = self.rule.get_param('edge_initialization', 'NONE')
            
        logger.debug(f"Grid.initialize_edges: Initializing with type: {initialization_type}")
        
        try:
            # Convert string to enum if it's a string
            if isinstance(initialization_type, str):
                initialization_type = EdgeInitialization[initialization_type]
            elif not isinstance(initialization_type, EdgeInitialization):
                logger.warning(f"Unknown edge initialization type: {initialization_type}, defaulting to NONE")
                initialization_type = EdgeInitialization.NONE
        except KeyError:
            logger.warning(f"Unknown edge initialization type: {initialization_type}, defaulting to NONE")
            initialization_type = EdgeInitialization.NONE
        
        if initialization_type == EdgeInitialization.NONE:
            # No initial edges
            return
            
        if initialization_type == EdgeInitialization.RANDOM:
            # Initialize with random connections
            connect_probability = self.rule.get_param('connect_probability', 0.1)
            
            # Only connect active nodes
            active_indices = np.where(self.grid_array.ravel() > 0)[0]
            
            for i in active_indices:
                # Get potential neighbors
                neighbors = self.neighborhood_data.get_neighbor_indices(i)
                active_neighbors = [n for n in neighbors if n in active_indices]
                
                for neighbor in active_neighbors:
                    if np.random.random() < connect_probability:
                        self.neighborhood_data.edge_matrix[i, neighbor] = True
                        self.neighborhood_data.edge_matrix[neighbor, i] = True
            
            logger.debug(f"Initialized {np.sum(self.neighborhood_data.edge_matrix) // 2} random edges")
            
        elif initialization_type == EdgeInitialization.FULL:
            # Connect all neighbors
            active_indices = np.where(self.grid_array.ravel() > 0)[0]
            
            for i in active_indices:
                # Get potential neighbors
                neighbors = self.neighborhood_data.get_neighbor_indices(i)
                active_neighbors = [n for n in neighbors if n in active_indices]
                
                for neighbor in active_neighbors:
                    self.neighborhood_data.edge_matrix[i, neighbor] = True
                    self.neighborhood_data.edge_matrix[neighbor, i] = True
                    
            logger.debug(f"Initialized {np.sum(self.neighborhood_data.edge_matrix) // 2} edges (full neighbors)")
        
        elif initialization_type == EdgeInitialization.DISTANCE:
            # Connect based on distance threshold
            distance_threshold = self.rule.get_param('distance_threshold', 1.5)
            active_indices = np.where(self.grid_array.ravel() > 0)[0]
            
            # Get coordinates for all active nodes
            coords = {}
            for i in active_indices:
                coords[i] = tuple(_unravel_index(i, self.dimensions))
            
            # Connect nodes within distance threshold
            for i in active_indices:
                i_coord = coords[i]
                for j in active_indices:
                    if i != j:
                        j_coord = coords[j]
                        # Calculate Euclidean distance
                        distance = np.sqrt(np.sum((np.array(i_coord) - np.array(j_coord))**2))
                        if distance <= distance_threshold:
                            self.neighborhood_data.edge_matrix[i, j] = True
                            self.neighborhood_data.edge_matrix[j, i] = True
            
            logger.debug(f"Initialized {np.sum(self.neighborhood_data.edge_matrix) // 2} edges based on distance threshold")
        
        elif initialization_type == EdgeInitialization.NEAREST:
            # Connect to nearest N neighbors
            n_nearest = self.rule.get_param('n_nearest', 3)
            active_indices = np.where(self.grid_array.ravel() > 0)[0]
            
            # Get coordinates for all active nodes
            coords = {}
            for i in active_indices:
                coords[i] = tuple(_unravel_index(i, self.dimensions))
            
            # For each active node, find nearest neighbors
            for i in active_indices:
                i_coord = coords[i]
                distances = []
                
                for j in active_indices:
                    if i != j:
                        j_coord = coords[j]
                        # Calculate Euclidean distance
                        distance = np.sqrt(np.sum((np.array(i_coord) - np.array(j_coord))**2))
                        distances.append((j, distance))
                
                # Sort by distance and take nearest n_nearest
                distances.sort(key=lambda x: x[1])
                nearest = [j for j, _ in distances[:n_nearest]]
                
                # Connect to nearest neighbors
                for j in nearest:
                    self.neighborhood_data.edge_matrix[i, j] = True
                    self.neighborhood_data.edge_matrix[j, i] = True
            
            logger.debug(f"Initialized {np.sum(self.neighborhood_data.edge_matrix) // 2} edges based on nearest neighbors")
        
        elif initialization_type == EdgeInitialization.SIMILARITY:
            # Connect based on state similarity
            similarity_threshold = self.rule.get_param('similarity_threshold', 0.8)
            active_indices = np.where(self.grid_array.ravel() > 0)[0]
            
            for i in active_indices:
                i_state = self.grid_array.ravel()[i]
                for j in active_indices:
                    if i != j:
                        j_state = self.grid_array.ravel()[j]
                        # Check if states are similar enough (within threshold percentage)
                        similarity = min(float(i_state), float(j_state)) / max(float(i_state), float(j_state))
                        if similarity >= similarity_threshold:
                            self.neighborhood_data.edge_matrix[i, j] = True
                            self.neighborhood_data.edge_matrix[j, i] = True
            
            logger.debug(f"Initialized {np.sum(self.neighborhood_data.edge_matrix) // 2} edges based on state similarity")
                            
    def cleanup_ghost_edges(self):
        """Clean up ghost edges in the edge matrix"""
        try:
            logger.debug("Starting ghost edge cleanup")
            
            # Get indices of all active nodes using vectorized operation
            active_indices = np.where(self.grid_array.ravel() > GlobalSettings.Visualization.NODE_VISIBILITY_THRESHOLD)[0]
            
            # Create a set of active indices for faster lookup
            active_set = set(active_indices)
            
            # Iterate through all possible edges
            total_nodes = np.prod(self.dimensions)
            cleaned_edge_count = 0
            
            for i in range(total_nodes):
                for j in range(i + 1, total_nodes):
                    # Check if an edge exists but either node is inactive
                    if self.neighborhood_data.edge_matrix[i, j] and (i not in active_set or j not in active_set):
                        # Remove the edge
                        self.neighborhood_data.edge_matrix[i, j] = False
                        self.neighborhood_data.edge_matrix[j, i] = False
                        cleaned_edge_count += 1
            
            logger.debug(f"Cleaned up {cleaned_edge_count} ghost edges")
            
        except Exception as e:
            logger.error(f"Error cleaning up ghost edges: {str(e)}\nTraceback:\n{traceback.format_exc()}")
            raise
            
    def verify_edge_matrix(self):
        """Verify that the edge matrix is symmetric"""
        try:
            edge_matrix = self.neighborhood_data.edge_matrix
            is_symmetric = np.array_equal(edge_matrix, edge_matrix.T)
            
            if not is_symmetric:
                logger.warning("Edge matrix is not symmetric")
                
                # Attempt to fix the matrix
                self.neighborhood_data.edge_matrix = np.logical_or(edge_matrix, edge_matrix.T).astype(bool)
                logger.info("Edge matrix has been made symmetric")
                
                is_symmetric = np.array_equal(self.neighborhood_data.edge_matrix, self.neighborhood_data.edge_matrix.T)
                if not is_symmetric:
                    logger.error("Failed to make edge matrix symmetric")
                    return False
            
            # Verify that all diagonal elements are False
            diagonal = np.diag(edge_matrix)
            if np.any(diagonal):
                logger.warning("Diagonal elements of edge matrix are not all False")
                np.fill_diagonal(self.neighborhood_data.edge_matrix, False)
                logger.info("Diagonal elements of edge matrix set to False")
                
            return True
        except Exception as e:
            logger.error(f"Error verifying edge matrix: {e}")
            return False
                
    def __del__(self):
        """Cleanup resources"""
        if self.shared_mem:
            try:
                self.shared_mem.close()
                self.shared_mem.unlink()
            except Exception as e:
                logger.error(f"Error cleaning up shared memory: {e}")

        if self.process_pool:
            try:
                self.process_pool.shutdown(wait=True)
            except Exception as e:
                logger.error(f"Error shutting down process pool: {e}")

    def set_rule(self, rule: 'Rule'):
        """Update the rule instance used by the grid"""
        try:
            # Store the new rule instance
            self.rule = rule
            
            # Log the rule change and its parameters
            logger.info(f"Grid updated to use rule: {rule.name}")
            logger.debug(f"Grid rule parameters: {rule.params}")
            
            # Invalidate any cached rule data
            if hasattr(self.rule, 'invalidate_cache'):
                self.rule.invalidate_cache()
                
        except Exception as e:
            logger.error(f"Error updating grid rule: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't reraise - handle gracefully
            return False
            
        return True
                
    def _initialize_neighbors(self):
        """Initialize neighbor indices for all nodes, handling empty cells"""
        total_nodes = np.prod(self.dimensions)
        max_neighbors = self._calculate_max_neighbors()
        
        # Initialize neighbor array with -1 (invalid neighbor)
        neighbor_indices = np.full((total_nodes, max_neighbors), -1, dtype=np.int64)
        
        try:
            # Calculate neighbors based on neighborhood type
            for idx in range(total_nodes):
                neighbors = []
                
                if self.neighborhood_type == NeighborhoodType.VON_NEUMANN:
                    neighbors = _get_von_neumann_neighbors(idx, self.dimensions)
                elif self.neighborhood_type == NeighborhoodType.MOORE:
                    neighbors = _get_moore_neighbors(idx, self.dimensions)
                elif self.neighborhood_type == NeighborhoodType.HEX:
                    if self.dimension_type != Dimension.TWO_D:
                        raise ValueError("HEX neighborhood type is only valid for 2D")
                    neighbors = _get_hex_neighbors(idx, self.dimensions)
                elif self.neighborhood_type == NeighborhoodType.HEX_PRISM:
                    if self.dimension_type != Dimension.THREE_D:
                        raise ValueError("HEX_PRISM neighborhood type is only valid for 3D")
                    if len(self.dimensions) != 3:
                        raise ValueError("HEX_PRISM requires 3D dimensions")
                    neighbors = _get_hex_prism_neighbors(idx, cast(Tuple[int, int, int], self.dimensions))
                    
                # Store valid neighbors
                for i, neighbor in enumerate(neighbors):
                    if i < max_neighbors:
                        neighbor_indices[idx, i] = neighbor
                            
                # Add debug logging
                # logger.debug(f"Node {idx} neighbors: {neighbors}") # REMOVED
                
            logger.debug(f"Initialized neighbors for {self.neighborhood_type.name} in {self.dimension_type.name}")
            return neighbor_indices
                
        except Exception as e:
            logger.error(f"Error initializing neighbors: {str(e)}")
            raise
        
    def _calculate_max_neighbors(self) -> int:
        """Calculate maximum possible neighbors based on neighborhood type"""
        if self.dimension_type == Dimension.TWO_D:
            if self.neighborhood_type == NeighborhoodType.VON_NEUMANN:
                return 4
            elif self.neighborhood_type == NeighborhoodType.MOORE:
                return 8
            elif self.neighborhood_type == NeighborhoodType.HEX:
                return 6
            else:
                raise ValueError(f"Invalid neighborhood type {self.neighborhood_type} for 2D")
        else:  # THREE_D
            if self.neighborhood_type == NeighborhoodType.VON_NEUMANN:
                return 6
            elif self.neighborhood_type == NeighborhoodType.MOORE:
                return 26
            elif self.neighborhood_type == NeighborhoodType.HEX_PRISM:
                return 12  # 6 in plane + 3 above + 3 below
            else:
                raise ValueError(f"Invalid neighborhood type {self.neighborhood_type} for 3D")
                                        
    def _chunk_grid(self) -> List[Tuple[int, int, int, int]]:
        """Chunk grid into smaller pieces for parallel processing with halo"""
        total_nodes = len(self.grid_array.ravel())
        num_processes = GlobalSettings.Simulation.NUM_PROCESSES
        chunk_size = max(1000, total_nodes // (num_processes * 4))
        
        # Get dependency depth from rule (how many hops it needs to see)
        dependency_depth = getattr(self.rule, 'dependency_depth', 1)
        
        # Base halo size calculation
        base_halo_size = max(1, self.rule.required_halo)
        
        # Calculate final halo size based on neighborhood type and dependency depth
        if self.neighborhood_type == NeighborhoodType.MOORE:
            # MOORE needs larger halos because of diagonals
            halo_size = max(2, base_halo_size) * dependency_depth
        elif self.neighborhood_type == NeighborhoodType.HEX_PRISM:
            # HEX_PRISM also needs larger halos in 3D
            halo_size = max(2, base_halo_size) * dependency_depth
        else:
            # VON_NEUMANN and HEX can use standard halo
            halo_size = base_halo_size * dependency_depth

        chunks = []
        for i in range(0, total_nodes, chunk_size):
            start = i
            end = min(i + chunk_size, total_nodes)

            # Calculate halo start and end indices with dependency depth considered
            halo_start = max(0, start - halo_size)
            halo_end = min(total_nodes, end + halo_size)

            chunks.append((start, end, halo_start, halo_end))

        logger.debug(f"Created {len(chunks)} chunks with halo size {halo_size} for rule with dependency depth {dependency_depth}")
        return chunks
        
    @staticmethod
    def _process_initializer():
        """Initialize process for parallel execution"""
        # Silence warnings in worker processes
        np.seterr(all='ignore')
        
        # Set process name for better debugging
        # Optional: Install setproctitle package for better process names
        try:
            setproctitle.setproctitle("NetworkCA Worker")
        except ImportError:
            pass
            
    @staticmethod
    def _process_chunk(neighborhood_data: 'NeighborhoodData',
                    start_idx: int, end_idx: int, shared_mem_name: str,
                    grid_shape: Tuple[int, ...], max_neighbors: int,
                    rule: 'Rule', rule_params: Dict[str, Any],
                    dimension_type: Dimension,
                    neighborhood_type: NeighborhoodType,
                    dimensions: Tuple[int, ...],
                    halo_start: int, halo_end: int,
                    result_queue: Queue) -> Tuple[npt.NDArray[np.float64], List[Set[int]]]:
        """Process a chunk of nodes in parallel"""
        shared_mem = None
        chunk_size = end_idx - start_idx
        
        try:
            # Access shared memory
            if shared_mem_name:
                shared_mem = shared_memory.SharedMemory(name=shared_mem_name)
                grid_array = np.ndarray(grid_shape, dtype=np.float64, buffer=shared_mem.buf)
            else:
                grid_array = np.zeros(grid_shape, dtype=np.float64)
                logger.debug("Using zero-filled NumPy array instead of shared memory")

            # Initialize states from grid array, including halo region
            states = grid_array.ravel()
            
            # Create a copy of the edge matrix for this chunk
            edge_matrix = neighborhood_data.edge_matrix.copy()

            # Process chunk
            new_states = np.zeros(chunk_size, dtype=np.float64)
            proposed_edges: List[Set[int]] = [set() for _ in range(chunk_size)]

            # Process each node in chunk
            for i in range(chunk_size):
                node_idx = start_idx + i
                if node_idx >= len(states):
                    continue

                try:
                    # Get current state
                    current_state = states[node_idx]
                    
                    # Skip empty nodes but preserve their states
                    if current_state < 0:
                        new_states[i] = current_state
                        proposed_edges[i] = set()  # No edges for empty nodes
                        continue

                    # Compute new state
                    new_state = rule.compute_state_update(
                        node_idx, 
                        neighborhood_data,
                        dimension_type=dimension_type
                    )
                    
                    # Store new state
                    new_states[i] = new_state

                    # Handle edge updates based on new state
                    if new_state > 0:  # Only active nodes propose edges
                        # For active nodes, compute new edges to other active nodes
                        new_edges = rule.compute_edge_updates(
                            node_idx, 
                            neighborhood_data,
                            dimension_type=dimension_type
                        )
                        
                        # Only propose edges to active nodes
                        active_edges = {n for n in new_edges if 0 <= n < len(states) and states[n] > 0}
                        proposed_edges[i] = active_edges
                    else:
                        # Inactive nodes don't have any edges
                        proposed_edges[i] = set()

                except Exception as node_e:
                    logger.error(f"Error processing individual node {node_idx}: {node_e}")
                    logger.error(traceback.format_exc())
                    # Continue with next node
                    new_states[i] = states[node_idx]  # Preserve existing state
                    proposed_edges[i] = set()

            if shared_mem is not None:
                shared_mem.close()
            return new_states, proposed_edges

        except ValueError as ve:
            logger.error(f"Value error in _process_chunk: {ve}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return np.zeros(chunk_size, dtype=np.float64), [set() for _ in range(chunk_size)]
        except IndexError as ie:
            logger.error(f"Index error in _process_chunk: {ie}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return np.zeros(chunk_size, dtype=np.float64), [set() for _ in range(chunk_size)]
        except TypeError as te:
            logger.error(f"Type error in _process_chunk: {te}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return np.zeros(chunk_size, dtype=np.float64), [set() for _ in range(chunk_size)]
        except MemoryError as me:
            logger.error(f"Memory error in _process_chunk: {me}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return np.zeros(chunk_size, dtype=np.float64), [set() for _ in range(chunk_size)]
        except Exception as e:
            logger.error(f"Error in chunk processing: {e}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return np.zeros(chunk_size, dtype=np.float64), [set() for _ in range(chunk_size)]
        
    @staticmethod
    def _process_states_only(neighborhood_data: 'NeighborhoodData',
                      start_idx: int, end_idx: int, shared_mem_name: str,
                      grid_shape: Tuple[int, ...], max_neighbors: int,
                      rule: 'Rule', rule_params: Dict[str, Any],
                      dimension_type: Dimension,
                      neighborhood_type: NeighborhoodType,
                      dimensions: Tuple[int, ...],
                      halo_start: int, halo_end: int,
                      result_queue: Queue) -> npt.NDArray[np.float64]:
        """Process only the states for a chunk of nodes in parallel"""
        shared_mem = None
        chunk_size = end_idx - start_idx
        
        try:
            # Access shared memory
            if shared_mem_name:
                shared_mem = shared_memory.SharedMemory(name=shared_mem_name)
                grid_array = np.ndarray(grid_shape, dtype=np.float64, buffer=shared_mem.buf)
            else:
                grid_array = np.zeros(grid_shape, dtype=np.float64)
                logger.debug("Using zero-filled NumPy array instead of shared memory")

            # Initialize states from grid array, including halo region
            states = grid_array.ravel()
            
            # Process chunk
            new_states = np.zeros(chunk_size, dtype=np.float64)

            # Process each node in chunk
            for i in range(chunk_size):
                node_idx = start_idx + i
                if node_idx >= len(states):
                    continue

                try:
                    # Get current state
                    current_state = states[node_idx]
                    
                    # Skip empty nodes but preserve their states
                    if current_state < 0:
                        new_states[i] = current_state
                        continue

                    # Compute new state only
                    new_state = rule.compute_state_update(
                        node_idx, 
                        neighborhood_data,
                        dimension_type=dimension_type
                    )
                    
                    # Store new state
                    new_states[i] = new_state

                except Exception as node_e:
                    logger.error(f"Error processing node state {node_idx}: {node_e}")
                    logger.error(traceback.format_exc())
                    # Continue with next node
                    new_states[i] = states[node_idx]  # Preserve existing state

            if shared_mem is not None:
                shared_mem.close()
            return new_states

        except ValueError as ve:
            logger.error(f"Value error in _process_states_only: {ve}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return np.zeros(chunk_size, dtype=np.float64)
        except IndexError as ie:
            logger.error(f"Index error in _process_states_only: {ie}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return np.zeros(chunk_size, dtype=np.float64)
        except TypeError as te:
            logger.error(f"Type error in _process_states_only: {te}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return np.zeros(chunk_size, dtype=np.float64)
        except MemoryError as me:
            logger.error(f"Memory error in _process_states_only: {me}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return np.zeros(chunk_size, dtype=np.float64)
        except Exception as e:
            logger.error(f"Error in chunk state processing: {e}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return np.zeros(chunk_size, dtype=np.float64)
    
    @staticmethod
    def _process_edges_only(neighborhood_data: 'NeighborhoodData',
                        start_idx: int, end_idx: int, shared_mem_name: str,
                        grid_shape: Tuple[int, ...], max_neighbors: int,
                        rule: 'Rule', rule_params: Dict[str, Any],
                        dimension_type: Dimension,
                        neighborhood_type: NeighborhoodType,
                        dimensions: Tuple[int, ...],
                        halo_start: int, halo_end: int,
                        result_queue: Queue) -> List[Set[int]]:
        """Process only the edges for a chunk of nodes in parallel"""
        shared_mem = None
        chunk_size = end_idx - start_idx
        
        try:
            # Access shared memory
            if shared_mem_name:
                shared_mem = shared_memory.SharedMemory(name=shared_mem_name)
                grid_array = np.ndarray(grid_shape, dtype=np.float64, buffer=shared_mem.buf)
            else:
                grid_array = np.zeros(grid_shape, dtype=np.float64)
                logger.debug("Using zero-filled NumPy array instead of shared memory")

            # Initialize states from grid array, including halo region
            states = grid_array.ravel()
            
            # Process chunk
            proposed_edges: List[Set[int]] = [set() for _ in range(chunk_size)]

            # Process each node in chunk
            for i in range(chunk_size):
                node_idx = start_idx + i
                if node_idx >= len(states):
                    continue

                try:
                    # Get current state
                    current_state = states[node_idx]
                    
                    # Skip inactive nodes
                    if current_state <= 0:
                        proposed_edges[i] = set()
                        continue

                    # Compute edge updates for active nodes only
                    new_edges = rule.compute_edge_updates(
                        node_idx, 
                        neighborhood_data,
                        dimension_type=dimension_type
                    )
                    
                    # Only propose edges to active nodes
                    active_edges = {n for n in new_edges if 0 <= n < len(states) and states[n] > 0}
                    proposed_edges[i] = active_edges

                except Exception as node_e:
                    logger.error(f"Error processing node edges {node_idx}: {node_e}")
                    logger.error(traceback.format_exc())
                    # Continue with next node
                    proposed_edges[i] = set()

            if shared_mem is not None:
                shared_mem.close()
            return proposed_edges

        except ValueError as ve:
            logger.error(f"Value error in _process_edges_only: {ve}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return [set() for _ in range(chunk_size)]
        except IndexError as ie:
            logger.error(f"Index error in _process_edges_only: {ie}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return [set() for _ in range(chunk_size)]
        except TypeError as te:
            logger.error(f"Type error in _process_edges_only: {te}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return [set() for _ in range(chunk_size)]
        except MemoryError as me:
            logger.error(f"Memory error in _process_edges_only: {me}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return [set() for _ in range(chunk_size)]
        except Exception as e:
            logger.error(f"Error in chunk edge processing: {e}")
            logger.error(traceback.format_exc())
            if shared_mem is not None:
                shared_mem.close()
            return [set() for _ in range(chunk_size)]
        
    def _update_grid_sequential(self):
        """Sequential (non-chunked) grid update for extremely complex rules"""
        try:
            logger.debug("=============== SEQUENTIAL GRID UPDATE ===============")
            
            # Get current state
            grid_array = self.grid_array
            
            # Log state before processing
            active_before = np.sum(grid_array.ravel() > 0)
            edge_count_before = np.sum(self.neighborhood_data.edge_matrix) // 2
            logger.debug(f"Grid state before update: Active nodes = {active_before}, Edges = {edge_count_before}")
            
            # Store original states for comparison
            original_states = np.ascontiguousarray(grid_array.ravel().copy())
            
            # Initialize arrays for new states and edge proposals
            total_nodes = self.total_nodes
            new_states = np.zeros_like(original_states)
            proposed_edges = [set() for _ in range(total_nodes)]
            
            # First phase: compute all new states
            for node_idx in range(total_nodes):
                # Get current state
                current_state = original_states[node_idx]
                
                # Skip empty nodes but preserve their states
                if current_state < 0:
                    new_states[node_idx] = current_state
                    continue
                    
                # Compute new state
                new_state = self.rule.compute_state_update(
                    node_idx,
                    self.neighborhood_data,
                    dimension_type=self.dimension_type
                )
                
                # Store new state
                new_states[node_idx] = new_state
            
            # Apply new states before computing edges
            with self._update_lock:
                self.grid_array = new_states.reshape(self.grid_array.shape)
                self.neighborhood_data.states = self.grid_array.ravel()
            
            # Second phase: compute edge updates with new states
            active_nodes = np.where(new_states > 0)[0]
            for node_idx in active_nodes:
                # Compute edge updates for active nodes
                new_edges = self.rule.compute_edge_updates(
                    node_idx,
                    self.neighborhood_data,
                    dimension_type=self.dimension_type
                )
                
                # Only propose edges to active nodes
                active_edges = {n for n in new_edges if n in active_nodes}
                proposed_edges[node_idx] = active_edges
            
            # Create proposed edge matrix
            proposed_edge_matrix = np.zeros_like(self.neighborhood_data.edge_matrix)
            
            # Collect proposed edges between active nodes
            for i in active_nodes:
                if proposed_edges[i] is not None:
                    for n in proposed_edges[i]:
                        proposed_edge_matrix[i, n] = True
            
            # Apply tie-breaker logic
            tiebreaker_type = TieBreaker[self.rule.get_param('tiebreaker_type', 'RANDOM')].value
            enable_tiebreakers = GlobalSettings.ENABLE_TIEBREAKERS
            final_edge_matrix = Grid._apply_edge_tiebreakers(
                new_states,
                proposed_edge_matrix,
                self.neighborhood_data.neighbor_indices,
                tiebreaker_type,
                enable_tiebreakers
            )
            
            # Update edge matrix
            with self._update_lock:
                old_edge_count = np.sum(self.neighborhood_data.edge_matrix) // 2
                self.neighborhood_data.edge_matrix = final_edge_matrix.copy()
                new_edge_count = np.sum(self.neighborhood_data.edge_matrix) // 2
            
            # Log final state
            active_after = np.sum(self.grid_array > 0)
            logger.debug(f"Sequential grid update complete - Active nodes: {active_before} -> {active_after}")
            logger.debug(f"Edge count: {edge_count_before} -> {new_edge_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in sequential grid update: {e}")
            logger.error(traceback.format_exc())
            raise
                                                                                                                                                                                                                                                                                    
    def update_grid_parallel(self):
        """Update grid state using parallel processing with multi-phase support for advanced rules"""
        try:
            # Ensure process pool exists
            if self.process_pool is None or getattr(self.process_pool, '_broken', False):
                if GlobalSettings.Simulation.NUM_PROCESSES > 1:
                    try:
                        self.process_pool = ProcessPoolExecutor(
                            max_workers=GlobalSettings.Simulation.NUM_PROCESSES,
                            initializer=Grid._process_initializer
                        )
                        logger.info(f"Recreated process pool with {GlobalSettings.Simulation.NUM_PROCESSES} workers")
                    except Exception as e:
                        logger.error(f"Failed to create process pool: {e}")
                        self.process_pool = ThreadPoolExecutor(max_workers=1)
                        logger.info("Falling back to single-threaded executor after error")
                else:
                    self.process_pool = ThreadPoolExecutor(max_workers=1)
                    logger.info("Recreated single-threaded executor")

            # Check if rule requires non-chunked processing
            if getattr(self.rule, 'disable_chunking', False):
                logger.info(f"Rule {self.rule.name} requires non-chunked processing")
                return self._update_grid_sequential()

            # Get dependency depth from rule
            dependency_depth = getattr(self.rule, 'dependency_depth', 1)
            needs_multi_phase = dependency_depth > 1

            chunks = self._chunk_grid()
            futures = []
            
            logger.debug("=============== GRID UPDATE (grid.update_grid_parallel) ===============")
        
            # Get current state
            grid_array = self.grid_array
            
            # Log state before processing
            active_before = np.sum(grid_array.ravel() > 0)
            edge_count_before = np.sum(self.neighborhood_data.edge_matrix) // 2
            logger.debug(f"Grid state before update: Active nodes = {active_before}, Edges = {edge_count_before}")
            logger.debug(f"Processing with rule: {self.rule.name}")  
            logger.debug(f"Rule parameters: {self.rule.params}")

            # Store original states for comparison
            original_states = np.ascontiguousarray(grid_array.ravel().copy())

            # Determine processing approach based on rule complexity
            if needs_multi_phase:
                logger.debug(f"Using multi-phase processing for rule with dependency depth {dependency_depth}")
                
                # Phase 1: Compute new states only
                logger.debug("Phase 1: Computing states...")
                new_states = self._compute_states_phase(original_states, chunks)
                
                # Apply new states immediately
                with self._update_lock:
                    self.grid_array = new_states.reshape(self.grid_array.shape)
                    self.neighborhood_data.states = self.grid_array.ravel()
                    self.state_updated = True
                    logger.debug(f"Phase 1 complete: {np.sum(new_states != original_states)} nodes changed state")
                
                # Phase 2: Compute edge updates with updated states
                logger.debug("Phase 2: Computing edge updates...")
                proposed_edges = self._compute_edges_phase(new_states, chunks)
                logger.debug("Phase 2 complete: Edge proposals collected")
                
            else:
                # Use existing method for simple rules
                new_states = np.zeros_like(original_states)
                proposed_edges = [set() for _ in range(self.total_nodes)]
                
                # Process results from each chunk
                for start, end, halo_start, halo_end in chunks:
                    # Ensure process pool exists before submitting task
                    if self.process_pool is None:
                        try:
                            self._initialize_process_pool()
                            if self.process_pool is None:
                                logger.error("Failed to initialize process pool, cannot continue")
                                return False
                        except Exception as e:
                            logger.error(f"Error initializing process pool: {e}")
                            return False
                    
                    try:
                        future = self.process_pool.submit(
                            Grid._process_chunk,
                            self.neighborhood_data,
                            int(start),
                            int(end),
                            self._shared_mem_name if self._shared_mem_name is not None else "",
                            self.neighborhood_data.states.shape,
                            self.max_neighbors,
                            self.rule,
                            self.rule.params,
                            self.dimension_type,
                            self.neighborhood_type,
                            self.dimensions,
                            halo_start,
                            halo_end,
                            Queue()
                        )
                        futures.append((start, end, future))
                    except Exception as e:
                        logger.error(f"Error submitting task for chunk {start}:{end}: {e}")
                        continue
                
                # Process results as they complete
                for start, end, future in futures:
                    try:
                        chunk_state, chunk_edge_updates = future.result(timeout=5.0)
                        
                        # Apply chunk results to new_states array
                        new_states[start:end] = chunk_state
                        
                        # Collect proposed edges
                        if chunk_edge_updates is not None:
                            for i, edges in enumerate(chunk_edge_updates):
                                node_idx = start + i
                                if 0 <= node_idx < len(proposed_edges):
                                    if edges is not None:
                                        proposed_edges[node_idx].update(edges)
                                else:
                                    logger.error(f"Node index {node_idx} out of range for proposed_edges")
                        else:
                            logger.warning(f"Received None for chunk_edge_updates from chunk {start}:{end}")
                        
                        logger.debug(f"Processed chunk {start}:{end}, states changed: {np.sum(chunk_state != original_states[start:end])}")
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {start}:{end}: {e}")
                        continue
                        
                # Update states if not in multi-phase mode
                with self._update_lock:
                    if not np.array_equal(new_states, original_states):
                        self.grid_array = new_states.reshape(self.grid_array.shape)
                        self.neighborhood_data.states = self.grid_array.ravel()
                        self.state_updated = True
                        logger.debug(f"Grid state updated, {np.sum(new_states != original_states)} nodes changed")

            # Create proposed edge matrix
            proposed_edge_matrix = np.zeros_like(self.neighborhood_data.edge_matrix)
            
            # Only process active nodes
            active_nodes = np.where(new_states > 0)[0]
            logger.debug(f"Active nodes after state update: {len(active_nodes)}")

            # Ensure proposed_edges exists and is properly initialized
            if proposed_edges is None:
                logger.warning("proposed_edges was None, initializing empty sets")
                proposed_edges = [set() for _ in range(self.total_nodes)]

            # Initialize all None entries at once
            for i in range(len(proposed_edges)):
                if proposed_edges[i] is None:
                    proposed_edges[i] = set()

            # Collect all proposed edges between active nodes
            for i in active_nodes:
                # Check if the index is valid
                if i < 0 or i >= len(proposed_edges):
                    logger.error(f"Active node index {i} out of range for proposed_edges of length {len(proposed_edges)}")
                    continue
                    
                # Now safely collect the edges (since we know proposed_edges[i] is not None)
                for n in proposed_edges[i]:
                    if n in active_nodes:  # Only connect active nodes
                        proposed_edge_matrix[i, n] = True

            # Apply tie-breaker logic
            tiebreaker_type = TieBreaker[self.rule.get_param('tiebreaker_type', 'RANDOM')].value
            enable_tiebreakers = GlobalSettings.ENABLE_TIEBREAKERS
            final_edge_matrix = Grid._apply_edge_tiebreakers(
                new_states,
                proposed_edge_matrix,
                self.neighborhood_data.neighbor_indices,
                tiebreaker_type,
                enable_tiebreakers
            )

            # Count edges after tiebreaker
            tiebreaker_edges = np.sum(final_edge_matrix) // 2
            logger.debug(f"Edges after tiebreaker: {tiebreaker_edges}")

            # Update edge matrix
            with self._update_lock:
                old_edge_count = np.sum(self.neighborhood_data.edge_matrix) // 2
                self.neighborhood_data.edge_matrix = final_edge_matrix.copy()
                new_edge_count = np.sum(self.neighborhood_data.edge_matrix) // 2
                logger.debug(f"Edge count changed from {old_edge_count} to {new_edge_count}")

                # Final verification - no inactive nodes should have edges
                inactive_mask = new_states <= 0
                for i in np.where(inactive_mask)[0]:
                    if np.any(self.neighborhood_data.edge_matrix[i, :]) or np.any(self.neighborhood_data.edge_matrix[:, i]):
                        logger.error(f"Inactive node {i} still has edges after update! Cleaning up...")
                        # Force cleanup
                        self.neighborhood_data.edge_matrix[i, :] = False
                        self.neighborhood_data.edge_matrix[:, i] = False

            # Log final state
            active_after = np.sum(self.grid_array > 0)
            logger.debug(f"Grid update complete - Active nodes: {active_before} -> {active_after}")
            logger.debug(f"Edge count: {edge_count_before} -> {new_edge_count}")

            return True

        except Exception as e:
            logger.error(f"Error in grid update: {e}")
            logger.error(traceback.format_exc())
            raise
            
    def _compute_states_phase(self, original_states, chunks):
        """First phase: compute only states with original edge information"""
        new_states = np.zeros_like(original_states)
        futures = []
        
        # Submit state computation jobs
        for start, end, halo_start, halo_end in chunks:
            # Ensure process pool exists with safer initialization
            if self.process_pool is None:
                try:
                    self._initialize_process_pool()
                    if self.process_pool is None:  # Still None after initialization?
                        logger.error("Failed to initialize process pool, using fallback single-threaded executor")
                        self.process_pool = ThreadPoolExecutor(max_workers=1)
                except Exception as e:
                    logger.error(f"Error initializing process pool: {e}")
                    self.process_pool = ThreadPoolExecutor(max_workers=1)
            
            future = self.process_pool.submit(
                Grid._process_states_only,
                self.neighborhood_data,
                int(start),
                int(end),
                self._shared_mem_name if self._shared_mem_name is not None else "",
                self.neighborhood_data.states.shape,
                self.max_neighbors,
                self.rule,
                self.rule.params,
                self.dimension_type,
                self.neighborhood_type,
                self.dimensions,
                halo_start,
                halo_end,
                Queue()
            )
            futures.append((start, end, future))
        
        # Process results as they complete
        for start, end, future in futures:
            try:
                chunk_state = future.result(timeout=5.0)
                
                # Apply chunk results to new_states array
                new_states[start:end] = chunk_state
                
                logger.debug(f"Processed states chunk {start}:{end}, states changed: {np.sum(chunk_state != original_states[start:end])}")
                
            except Exception as e:
                logger.error(f"Error processing states chunk {start}:{end}: {e}")
                continue
        
        return new_states

    def _compute_edges_phase(self, new_states, chunks):
        """Second phase: compute edges with updated state information"""
        proposed_edges = [set() for _ in range(self.total_nodes)]
        futures = []
        
        # Submit edge computation jobs
        for start, end, halo_start, halo_end in chunks:
            # Ensure process pool exists with safer initialization
            if self.process_pool is None:
                try:
                    self._initialize_process_pool()
                    if self.process_pool is None:  # Still None after initialization?
                        logger.error("Failed to initialize process pool, using fallback single-threaded executor")
                        self.process_pool = ThreadPoolExecutor(max_workers=1)
                except Exception as e:
                    logger.error(f"Error initializing process pool: {e}")
                    self.process_pool = ThreadPoolExecutor(max_workers=1)
            
            future = self.process_pool.submit(
                Grid._process_edges_only,
                self.neighborhood_data,
                int(start),
                int(end),
                self._shared_mem_name if self._shared_mem_name is not None else "",
                self.neighborhood_data.states.shape,
                self.max_neighbors,
                self.rule,
                self.rule.params,
                self.dimension_type,
                self.neighborhood_type,
                self.dimensions,
                halo_start,
                halo_end,
                Queue()
            )
            futures.append((start, end, future))
        
        # Process results as they complete
        for start, end, future in futures:
            try:
                chunk_edge_updates = future.result(timeout=5.0)
                
                # Validate and collect proposed edges
                if chunk_edge_updates is not None:
                    for i, edges in enumerate(chunk_edge_updates):
                        node_idx = start + i
                        if edges is not None:
                            proposed_edges[node_idx].update(edges)
                else:
                    logger.warning(f"Received None for chunk_edge_updates from chunk {start}:{end}")
                
                logger.debug(f"Processed edges chunk {start}:{end}")
                
            except Exception as e:
                logger.error(f"Error processing edges chunk {start}:{end}: {e}")
                continue
        
        return proposed_edges
                             
    @staticmethod
    @njit(cache=True)
    def _apply_edge_tiebreakers(grid_array: npt.NDArray[np.float64],
                            proposed_edge_matrix: npt.NDArray[np.bool_],
                            neighbor_indices: npt.NDArray[np.int64],
                            tiebreaker_type: int,
                            enable_tiebreakers: bool) -> npt.NDArray[np.bool_]:
        """Apply tie-breaker logic to resolve edge conflicts"""
        num_nodes = grid_array.shape[0]
        final_edge_matrix = np.copy(proposed_edge_matrix)
        
        # First, filter out inactive nodes - they should have no edges
        active_mask = grid_array > 0  # Using 0 as the threshold for active nodes
        
        # Remove all edges involving inactive nodes
        for i in range(num_nodes):
            if not active_mask[i]:
                final_edge_matrix[i, :] = False
                final_edge_matrix[:, i] = False
        
        if enable_tiebreakers:
            for i in range(num_nodes):
                # Skip inactive nodes
                if not active_mask[i]:
                    continue
                    
                for j in range(i + 1, num_nodes):
                    # Skip inactive nodes
                    if not active_mask[j]:
                        continue
                        
                    # Check if both nodes agree on connection state
                    if final_edge_matrix[i, j] == final_edge_matrix[j, i]:
                        # They agree - respect their decision
                        # Ensure symmetry (technically redundant, but for safety)
                        final_edge_matrix[j, i] = final_edge_matrix[i, j]
                        continue
                    
                    # We have a disagreement - one node wants to connect but the other doesn't
                    # This is where tie-breaking logic applies
                    node1_state = grid_array[i]
                    node2_state = grid_array[j]
                    
                    # Get valid neighbors for both nodes
                    node1_neighbors = neighbor_indices[i]
                    node1_neighbors = node1_neighbors[node1_neighbors != -1]
                    node2_neighbors = neighbor_indices[j]
                    node2_neighbors = node2_neighbors[node2_neighbors != -1]
                    
                    # Filter to only count active neighbors
                    active_neighbors1 = 0
                    active_neighbors2 = 0
                    
                    for n_idx in range(len(node1_neighbors)):
                        n = node1_neighbors[n_idx]
                        if 0 <= n < len(grid_array) and active_mask[n]:
                            active_neighbors1 += 1
                            
                    for n_idx in range(len(node2_neighbors)):
                        n = node2_neighbors[n_idx]
                        if 0 <= n < len(grid_array) and active_mask[n]:
                            active_neighbors2 += 1
                    
                    # Apply tiebreaker logic based on specified type
                    if tiebreaker_type == 1:  # TieBreaker.HIGHER_STATE
                        node1_wins = node1_state > node2_state
                    elif tiebreaker_type == 2:  # TieBreaker.LOWER_STATE
                        node1_wins = node1_state < node2_state
                    elif tiebreaker_type == 3:  # TieBreaker.MORE_CONNECTIONS
                        node1_wins = active_neighbors1 > active_neighbors2
                    elif tiebreaker_type == 4:  # TieBreaker.FEWER_CONNECTIONS
                        node1_wins = active_neighbors1 < active_neighbors2
                    elif tiebreaker_type == 5:  # TieBreaker.HIGHER_STATE_MORE_NEIGHBORS
                        if node1_state > node2_state:
                            node1_wins = True
                        elif node1_state == node2_state:
                            node1_wins = active_neighbors1 > active_neighbors2
                        else:
                            node1_wins = False
                    elif tiebreaker_type == 6:  # TieBreaker.LOWER_STATE_FEWER_NEIGHBORS
                        if node1_state < node2_state:
                            node1_wins = True
                        elif node1_state == node2_state:
                            node1_wins = active_neighbors1 < active_neighbors2
                        else:
                            node1_wins = False
                    elif tiebreaker_type == 7:  # TieBreaker.HIGHER_STATE_FEWER_NEIGHBORS
                        if node1_state > node2_state:
                            node1_wins = True
                        elif node1_state == node2_state:
                            node1_wins = active_neighbors1 < active_neighbors2
                        else:
                            node1_wins = False
                    elif tiebreaker_type == 8:  # TieBreaker.LOWER_STATE_MORE_NEIGHBORS
                        if node1_state < node2_state:
                            node1_wins = True
                        elif node1_state == node2_state:
                            node1_wins = active_neighbors1 > active_neighbors2
                        else:
                            node1_wins = False
                    elif tiebreaker_type == 9:  # TieBreaker.AGREEMENT
                        # If they disagree, remove the edge
                        final_edge_matrix[i, j] = False
                        final_edge_matrix[j, i] = False
                        continue
                    else:  # TieBreaker.RANDOM or invalid
                        node1_wins = np.random.rand() > 0.5
                    
                    # Apply the outcome of the tiebreaker
                    if node1_wins:
                        # Node i wins - use its decision
                        decision = final_edge_matrix[i, j]
                        final_edge_matrix[i, j] = decision
                        final_edge_matrix[j, i] = decision
                    else:
                        # Node j wins - use its decision
                        decision = final_edge_matrix[j, i]
                        final_edge_matrix[i, j] = decision
                        final_edge_matrix[j, i] = decision
        
        # Final check: ensure all edges involving inactive nodes are removed
        # This is needed even with the initial filter, as a node might have become inactive during rule processing
        for i in range(num_nodes):
            if not active_mask[i]:
                final_edge_matrix[i, :] = False
                final_edge_matrix[:, i] = False
        
        # Final symmetrization to ensure matrix consistency
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if final_edge_matrix[i, j] != final_edge_matrix[j, i]:
                    # This should not happen, but just in case
                    final_edge_matrix[i, j] = final_edge_matrix[j, i] = (final_edge_matrix[i, j] and final_edge_matrix[j, i])
        
        return final_edge_matrix
                                                                                                                                                                                                                                     
    def get_node_state(self, coords: Tuple[int, ...]) -> float:
        """Get state of node at coordinates"""
        return self.grid_array[coords]
        
    def set_node_state(self, idx, state):
        """Set the state of a node by index"""
        # Update grid array
        if isinstance(idx, (list, tuple, np.ndarray)):
            # Convert multi-dimensional index to flat index
            flat_idx = np.ravel_multi_index(tuple(idx), self.dimensions)
        else:
            flat_idx = idx
            
        if flat_idx < 0 or flat_idx >= self.total_nodes:
            logger.error(f"Invalid node index: {flat_idx}, must be in range [0, {self.total_nodes-1}]")
            return False
            
        self.grid_array.ravel()[flat_idx] = state
        self.neighborhood_data.states[flat_idx] = state
        
        # If setting to inactive or empty, remove all edges
        if state <= 0:
            self.neighborhood_data.edge_matrix[flat_idx, :] = False
            self.neighborhood_data.edge_matrix[:, flat_idx] = False
            
        return True
    
    def get_performance_stats(self):
        """Get performance statistics for the grid and rule"""
        stats = {
            'grid_size': self.total_nodes,
            'active_nodes': np.sum(self.grid_array > 0),
            'edge_count': np.sum(self.neighborhood_data.edge_matrix) // 2,
            'rule': self.rule.name
        }
        
        # Add rule-specific stats
        rule_stats = getattr(self.rule, 'perf_stats', {})
        if rule_stats:
            compute_times = rule_stats.get('compute_times', [])
            if compute_times:
                stats['avg_compute_time'] = sum(compute_times) / len(compute_times)
                stats['max_compute_time'] = max(compute_times)
                stats['min_compute_time'] = min(compute_times)
            
            stats['cache_hits'] = rule_stats.get('cache_hits', 0)
            stats['cache_misses'] = rule_stats.get('cache_misses', 0)
        
        return stats
                                                                 
class RuleMetrics(Protocol):
    """Protocol defining the interface for rule metrics"""
    
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Compute metric value"""
        ...

class BaseMetrics:
    """Base metrics implementation that follows RuleMetrics protocol"""
    
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Default compute implementation"""
        return 0.0

    @staticmethod
    @njit(cache=True)
    def active_neighbor_ratio(states: npt.NDArray[np.float64],
                            neighbor_indices: npt.NDArray[np.int64],
                            edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Ratio of active neighbors to total neighbors"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        return float(np.sum(states[valid_neighbors] > 0) / len(valid_neighbors))
        
    @staticmethod
    @njit(cache=True)
    def edge_density(states: npt.NDArray[np.float64],
                    neighbor_indices: npt.NDArray[np.int64],
                    edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Local edge density in neighborhood"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) < 2:
            return 0.0
        
        possible_edges = len(valid_neighbors) * (len(valid_neighbors) - 1) / 2
        if possible_edges == 0:
            return 0.0
            
        actual_edges = 0.0
        for i in range(len(valid_neighbors)):
            for j in range(i + 1, len(valid_neighbors)):
                if edge_matrix[valid_neighbors[i], valid_neighbors[j]]:
                    actual_edges += 1.0
                    
        return float(actual_edges / possible_edges)

    @staticmethod
    def average_neighbor_degree(states: npt.NDArray[np.float64],
                            neighbor_indices: npt.NDArray[np.int64],
                            edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Average degree (number of connections) of a node's neighbors"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        total_neighbor_degrees = 0
        for neighbor in valid_neighbors:
            total_neighbor_degrees += np.sum(edge_matrix[neighbor])
        
        return float(total_neighbor_degrees / len(valid_neighbors))

    @staticmethod
    def assortativity(states: npt.NDArray[np.float64],
                    neighbor_indices: npt.NDArray[np.int64],
                    edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Measures the assortativity coefficient of the local network"""
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) < 2:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the original edge matrix
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if edge_matrix[subgraph_nodes[i], subgraph_nodes[j]]:
                    G.add_edge(i, j)
        
        # Calculate assortativity
        try:
            return float(nx.degree_assortativity_coefficient(G))
        except Exception:
            return 0.0  # Handle disconnected graphs

    @staticmethod
    def betweenness_centrality(states: npt.NDArray[np.float64],
                            neighbor_indices: npt.NDArray[np.int64],
                            edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Measures the betweenness centrality of a node in the local network"""
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the original edge matrix
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if edge_matrix[subgraph_nodes[i], subgraph_nodes[j]]:
                    G.add_edge(i, j)
        
        # Calculate betweenness centrality
        try:
            centrality = nx.betweenness_centrality(G)
            return float(centrality[0])
        except Exception:
            return 0.0  # Handle disconnected graphs

    @staticmethod
    def closeness_centrality(states: npt.NDArray[np.float64],
                            neighbor_indices: npt.NDArray[np.int64],
                            edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Measures the closeness centrality of a node in the local network"""
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the original edge matrix
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if edge_matrix[subgraph_nodes[i], subgraph_nodes[j]]:
                    G.add_edge(i, j)
        
        # Calculate closeness centrality
        try:
            centrality = nx.closeness_centrality(G, u=0)
            return float(centrality[0])
        except Exception:
            return 0.0  # Handle disconnected graphs

    @staticmethod
    def eigenvector_centrality(states: npt.NDArray[np.float64],
                            neighbor_indices: npt.NDArray[np.int64],
                            edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Measures the eigenvector centrality of a node in the local network"""
        
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the original edge matrix
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if edge_matrix[subgraph_nodes[i], subgraph_nodes[j]]:
                    G.add_edge(i, j)
        
        # Calculate eigenvector centrality
        try:
            centrality = nx.eigenvector_centrality(G)[0]
            return float(centrality)
        except Exception:
            return 0.0  # Handle disconnected graphs

    @staticmethod
    def graph_laplacian_energy(states: npt.NDArray[np.float64],
                            neighbor_indices: npt.NDArray[np.int64],
                            edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Measures the graph laplacian energy of the local network"""
        
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the original edge matrix
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if edge_matrix[subgraph_nodes[i], subgraph_nodes[j]]:
                    G.add_edge(i, j)
        
        # Calculate graph laplacian energy
        try:
            laplacian = nx.laplacian_matrix(G).asfptype()
            eigenvalues = np.linalg.eigvalsh(laplacian.toarray())
            energy = np.sum(np.abs(eigenvalues))
            return float(energy)
        except Exception:
            return 0.0  # Handle disconnected graphs

    @staticmethod
    @njit(cache=True)
    def variance_neighbor_state(states: npt.NDArray[np.float64],
                            neighbor_indices: npt.NDArray[np.int64],
                            edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Variance of neighbor states"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        neighbor_states = states[valid_neighbors]
        mean = np.mean(neighbor_states)
        squared_diff_sum = 0.0
        for state in neighbor_states:
            squared_diff_sum += (state - mean) ** 2
        return float(squared_diff_sum / len(neighbor_states))
        
    @staticmethod
    @njit(cache=True)
    def median_neighbor_state(states: npt.NDArray[np.float64],
                            neighbor_indices: npt.NDArray[np.int64],
                            edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Median of neighbor states"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        neighbor_states = states[valid_neighbors]
        # Sort states manually for Numba compatibility
        sorted_states = np.sort(neighbor_states)
        mid = len(sorted_states) // 2
        if len(sorted_states) % 2 == 0:
            return float((sorted_states[mid-1] + sorted_states[mid]) / 2)
        return float(sorted_states[mid])

    @staticmethod
    @njit(cache=True)
    def mode_neighbor_state(states: npt.NDArray[np.float64],
                        neighbor_indices: npt.NDArray[np.int64],
                        edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Mode of neighbor states using bincount"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        neighbor_states = states[valid_neighbors]
        # Convert states to integers for bincount
        state_ints = (neighbor_states * 100).astype(np.int64)  # Scale for precision
        counts = np.bincount(state_ints)
        mode_idx = np.argmax(counts)
        return float(mode_idx) / 100.0  # Convert back to original scale
        
    @staticmethod
    @njit(cache=True)
    def state_entropy(states: npt.NDArray[np.float64],
                    neighbor_indices: npt.NDArray[np.int64],
                    edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Information entropy of neighbor states using bincount"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        neighbor_states = states[valid_neighbors]
        # Convert states to integers for bincount
        state_ints = (neighbor_states * 100).astype(np.int64)  # Scale for precision
        counts = np.bincount(state_ints)
        # Remove zero counts
        counts = counts[counts > 0]
        # Calculate probabilities
        probabilities = counts.astype(np.float64) / len(neighbor_states)
        # Calculate entropy manually
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        return float(entropy)

    @staticmethod
    @njit(cache=True)
    def clustering_coefficient(states: npt.NDArray[np.float64],
                            neighbor_indices: npt.NDArray[np.int64],
                            edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Local clustering coefficient"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) < 3:
            return 0.0
        
        triangles = 0
        possible_triangles = len(valid_neighbors) * (len(valid_neighbors) - 1) * (len(valid_neighbors) - 2) / 6
        
        if possible_triangles == 0:
            return 0.0
            
        for i in range(len(valid_neighbors)):
            for j in range(i + 1, len(valid_neighbors)):
                for k in range(j + 1, len(valid_neighbors)):
                    if (edge_matrix[valid_neighbors[i], valid_neighbors[j]] and
                        edge_matrix[valid_neighbors[j], valid_neighbors[k]] and
                        edge_matrix[valid_neighbors[k], valid_neighbors[i]]):
                        triangles += 1
                        
        return float(triangles / possible_triangles)

class StateEntropy(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        return BaseMetrics.state_entropy(states, neighbor_indices, edge_matrix)

class ClusteringCoefficient(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        return BaseMetrics.clustering_coefficient(states, neighbor_indices, edge_matrix)
    
class ActiveNeighborRatio(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Ratio of active neighbors to total neighbors"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        return float(np.sum(states[valid_neighbors] > 0) / len(valid_neighbors))

class EdgeDensity(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Local edge density in neighborhood"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) < 2:
            return 0.0
        possible_edges = len(valid_neighbors) * (len(valid_neighbors) - 1) / 2
        actual_edges = float(np.sum(edge_matrix[valid_neighbors][:, valid_neighbors])) / 2
        return float(actual_edges / possible_edges) if possible_edges > 0 else 0.0

class AverageNeighborDegree(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Average degree (number of connections) of a node's neighbors"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        total_neighbor_degrees = 0
        for neighbor in valid_neighbors:
            total_neighbor_degrees += np.sum(edge_matrix[neighbor])
        
        return float(total_neighbor_degrees / len(valid_neighbors))
    
class VarianceNeighborState(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        return BaseMetrics.variance_neighbor_state(states, neighbor_indices, edge_matrix)

class ModeNeighborState(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        return BaseMetrics.mode_neighbor_state(states, neighbor_indices, edge_matrix)

class NodeEdgeRatio(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Compute Node Edge Ratio, ignoring ghost edges"""
        neighborhood_data = NeighborhoodData(states.shape, len(neighbor_indices))
        neighborhood_data.neighbor_indices = neighbor_indices.reshape(1, -1)
        neighborhood_data.edge_matrix = edge_matrix
        
        valid_neighbors = neighborhood_data.get_neighbor_indices(0)
        active_edges = neighborhood_data.get_active_edges(0, states)
        
        if len(valid_neighbors) == 0:
            return 0.0
        
        edge_count = len(active_edges)
        return float(len(valid_neighbors) / edge_count) if edge_count > 0 else float('inf')

class LocalEdgeDensity(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Compute Local Edge Density, ignoring ghost edges"""
        neighborhood_data = NeighborhoodData(states.shape, len(neighbor_indices))
        neighborhood_data.neighbor_indices = neighbor_indices.reshape(1, -1)
        neighborhood_data.edge_matrix = edge_matrix
        
        valid_neighbors = neighborhood_data.get_neighbor_indices(0)
        if len(valid_neighbors) < 2:
            return 0.0
        
        active_neighbors = neighborhood_data.get_active_neighbor_indices(0, states)
        
        local_edges = 0
        for i in range(len(active_neighbors)):
            for j in range(i + 1, len(active_neighbors)):
                if edge_matrix[active_neighbors[i], active_neighbors[j]]:
                    local_edges += 1
        
        possible_edges = len(active_neighbors) * (len(active_neighbors) - 1) / 2
        return float(local_edges / possible_edges) if possible_edges > 0 else 0.0

class EdgeDivisibility(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        edge_count = np.sum(edge_matrix[valid_neighbors])
        # Return 1.0 if edge count is divisible by target (e.g., 3 or 4), 0.0 otherwise
        # Default to checking divisibility by 3 (for triangular patterns)
        return 1.0 if edge_count % 3 == 0 else 0.0

class MedianNeighborState(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Compute Edge Divisibility, ignoring ghost edges"""
        neighborhood_data = NeighborhoodData(states.shape, len(neighbor_indices))
        neighborhood_data.neighbor_indices = neighbor_indices.reshape(1, -1)
        neighborhood_data.edge_matrix = edge_matrix
        
        valid_neighbors = neighborhood_data.get_neighbor_indices(0)
        active_edges = neighborhood_data.get_active_edges(0, states)
        
        edge_count = len(active_edges)
        # Return 1.0 if edge count is divisible by target (e.g., 3 or 4), 0.0 otherwise
        # Default to checking divisibility by 3 (for triangular patterns)
        return 1.0 if edge_count % 3 == 0 else 0.0

@dataclass
class RuleMetadata:
    """Metadata for rules in the library"""
    name: str
    type: str                  
    position: int              
    category: str              
    author: str               
    url: str                  
    email: str                
    date_created: str         
    date_modified: str        
    version: str              
    description: str          
    tags: List[str]           
    dimension_compatibility: List[str]
    neighborhood_compatibility: List[str]
    parent_rule: Optional[str]
    rating: Optional[int] = None  # New field for rating
    notes: Optional[str] = None   # New field for notes
    controller: Optional['SimulationController'] = None # Add controller attribute
    neighborhood_type: Optional[NeighborhoodType] = None # ADDED
    
class DefaultRuleMetadata:
    """Default metadata values for rules"""
    
    @staticmethod
    def get_default_metadata(rule_type: str, rule_name: str, category: str, description: str, dimension_compatibility: Optional[List[str]] = None, neighborhood_compatibility: Optional[List[str]] = None, rating: Optional[int] = None, notes: Optional[str] = None) -> RuleMetadata:
        """Create default metadata for a rule"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        return RuleMetadata(
            name=rule_name,
            type=rule_type,
            position=1,  # Default rules are always position 1
            category=category,
            author="Nova Spivack",
            url="https://novaspivack.com/network_automata",
            email="novaspivackrelay @ gmail . com",
            date_created=current_date,
            date_modified=current_date,
            version="1.0",
            description=description,
            tags=[category, rule_type, "default"],
            dimension_compatibility=dimension_compatibility or ["TWO_D", "THREE_D"],  # Can be overridden for specific rules
            neighborhood_compatibility=neighborhood_compatibility or [],
            parent_rule=None,
            rating=rating,
            notes=notes
        )
        
class Rule(ABC):
    """Abstract base class for cellular automaton rules"""

    name: str
    metadata: 'RuleMetadata'

    def __init__(self, metadata: 'RuleMetadata'):
        self.metadata = metadata
        self.name = metadata.name
        self.required_metrics: List[Any] = []
        self.metric_cache: Dict[Any, Any] = {}
        self.cache_generation = 0
        self._params: Dict[str, Any] = {}  # Make private
        self.required_halo: int = 1
        self.dependency_depth: int = 1  # Default: only immediate neighbors
        self.disable_chunking: bool = False  # For extremely complex rules

        # Performance monitoring
        self.perf_stats: Dict[str, Any] = {
            'compute_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Add initial_conditions to the PARAMETER_METADATA
        if not hasattr(self, 'PARAMETER_METADATA'):
            self.PARAMETER_METADATA = {}
        
        if 'initial_conditions' not in self.PARAMETER_METADATA:
            self.PARAMETER_METADATA['initial_conditions'] = {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            }
        
    @abstractmethod
    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData, node_positions: Optional[np.ndarray] = None, dimension_type: Optional[Dimension] = None) -> float: # Verify this signature
        """Compute state update for a single node. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData, node_positions: Optional[np.ndarray] = None, dimension_type: Optional[Dimension] = None) -> Set[int]: # Verify this signature
        """Compute edge updates for a single node. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def get_metric(self,
                  metric: Type['RuleMetrics'],
                  node_idx: int,
                  neighborhood_data: 'NeighborhoodData') -> float:
        """Get cached metric value or compute if not available"""
        cache_key = (node_idx, metric.__name__)
        
        if cache_key in self.metric_cache:
            self.perf_stats['cache_hits'] += 1
            return self.metric_cache[cache_key]
            
        self.perf_stats['cache_misses'] += 1
        value = metric.compute(
            neighborhood_data.states,
            neighborhood_data.neighbor_indices[node_idx],
            neighborhood_data.edge_matrix
        )
        self.metric_cache[cache_key] = value
        return value
        
    def invalidate_cache(self):
        """Invalidate metric cache"""
        self.metric_cache.clear()
        self.cache_generation += 1
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'name': self.name,
            'avg_compute_time': np.mean(self.perf_stats['compute_times'] [-100:])
                               if self.perf_stats['compute_times'] else 0,
            'cache_hits': self.perf_stats['cache_hits'],
            'cache_misses': self.perf_stats['cache_misses'],
            'cache_size': len(self.metric_cache),
            'cache_generation': self.cache_generation
        }

    @property
    def params(self) -> Dict[str, Any]:
        """Get parameters dictionary"""
        return self._params

    @params.setter
    def params(self, new_params: Dict[str, Any]):
        """Set parameters with validation"""
        validated_params = {}
        for name, value in new_params.items():
            if self._validate_parameter(name, value):
                validated_params[name] = value
            else:
                logger.warning(f"Invalid parameter value for {name}: {value}")
        self._params = validated_params

    def update_parameter(self, name: str, value: Any) -> bool:
        """Update a single parameter with validation"""
        if self._validate_parameter(name, value):
            self._params[name] = value
            return True
        logger.warning(f"Invalid parameter value for {name}: {value}")
        return False

    def get_param(self, name: str, default: Any = None) -> Any:
        """Get parameter value with validation"""
        value = self._params.get(name, default)
        if value is not None and not self._validate_parameter(name, value):
            logger.warning(f"Invalid parameter value for {name}: {value}, using default")
            return default
        return value
        
    def _validate_parameter(self, name: str, value: Any) -> bool:
        """Validate a parameter value"""
        try:
            if name.startswith('use_') and name.endswith('_density') or name.startswith('use_'):
                # Handle boolean parameters
                if isinstance(value, bool):
                    return True
                # Also allow string representations of booleans
                if isinstance(value, str):
                    return value.lower() in ['true', 'false', '1', '0', 'yes', 'no', 'on', 'off']
                return False
                
            if name.startswith('min_') and name.endswith('_neighbors'):
                return isinstance(value, int) and 0 <= value <= self._get_max_neighbors_for_validation()
            elif name.startswith('max_') and name.endswith('_neighbors'):
                return isinstance(value, int) and 0 <= value <= self._get_max_neighbors_for_validation()
            elif name.endswith('_rate') or name.endswith('_density'):
                return isinstance(value, float) and 0.0 <= value <= 1.0
            elif name == 'tiebreaker_type':
                return value in TieBreaker.__members__
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation error for {name}: {e}")
            return False
        
    def _get_max_neighbors_for_validation(self) -> int:
        """Helper method to get the maximum number of neighbors for validation"""
        if GlobalSettings.Simulation.DIMENSION_TYPE == Dimension.TWO_D:
            if GlobalSettings.Simulation.NEIGHBORHOOD_TYPE == NeighborhoodType.VON_NEUMANN:
                return 4
            elif GlobalSettings.Simulation.NEIGHBORHOOD_TYPE == NeighborhoodType.MOORE:
                return 8
            elif GlobalSettings.Simulation.NEIGHBORHOOD_TYPE == NeighborhoodType.HEX:
                return 6
            else:
                return 8  # Default for safety
        else:  # Dimension.THREE_D
            if GlobalSettings.Simulation.NEIGHBORHOOD_TYPE == NeighborhoodType.VON_NEUMANN:
                return 6
            elif GlobalSettings.Simulation.NEIGHBORHOOD_TYPE == NeighborhoodType.MOORE:
                return 26
            elif GlobalSettings.Simulation.NEIGHBORHOOD_TYPE == NeighborhoodType.HEX_PRISM:
                return 12
            else:
                return 26  # Default for safety

class RuleVariation:
    """Holds parameters for a specific dimension and neighborhood combination"""
    def __init__(self, 
                 dimension: Optional[Dimension] = None,  # None means "Any"
                 neighborhood: Optional[NeighborhoodType] = None,  # None means "Any"
                 **params):
        self.dimension = dimension
        self.neighborhood = neighborhood
        self.params = params

class RuleParameters:
    """Container for rule parameters with dimension and neighborhood variations"""
    
    def __init__(self, **kwargs):
        self._variations: List[RuleVariation] = []
        self._default_params = {}
        self._validators = {}
        self._descriptions = {}
        
        # If no explicit variations provided, treat kwargs as default parameters
        if 'variations' not in kwargs:
            self._default_params = kwargs
        else:
            self._variations = kwargs['variations']
            if 'default' in kwargs:
                self._default_params = kwargs['default']
    
    def add_variation(self, variation: RuleVariation):
        """Add a new parameter variation"""
        self._variations.append(variation)
    
    def get_parameters(self, dimension: Dimension, neighborhood: NeighborhoodType) -> Dict[str, Any]:
        """Get parameters for specific dimension and neighborhood"""
        # Look for exact match first
        for var in self._variations:
            if var.dimension == dimension and var.neighborhood == neighborhood:
                return var.params
        
        # Look for dimension match with any neighborhood
        for var in self._variations:
            if var.dimension == dimension and var.neighborhood is None:
                return var.params
                
        # Look for neighborhood match with any dimension
        for var in self._variations:
            if var.dimension is None and var.neighborhood == neighborhood:
                return var.params
        
        # Return default parameters if no specific variation found
        return self._default_params
    
    def add_parameter(self, 
                     name: str, 
                     value: Any, 
                     validator: Optional[Callable[[Any], bool]] = None,
                     description: str = ""):
        """Add a parameter with optional validation"""
        if validator is not None and not validator(value):
            raise ValueError(f"Invalid value for parameter {name}: {value}")
            
        self._default_params[name] = value
        self._validators[name] = validator
        self._descriptions[name] = description
        
    def __getattr__(self, name: str) -> Any:
        """Get parameter value using current dimension and neighborhood"""
        if name in self._default_params:
            # Get current dimension and neighborhood from GlobalSettings
            current_dim = GlobalSettings.Simulation.DIMENSION_TYPE
            current_neighborhood = GlobalSettings.Simulation.NEIGHBORHOOD_TYPE
            
            # Get parameters for current configuration
            params = self.get_parameters(current_dim, current_neighborhood)
            return params.get(name, self._default_params[name])
            
        raise AttributeError(f"Parameter {name} not found")
        
    def __setattr__(self, name: str, value: Any):
        """Set parameter value with validation"""
        if name.startswith('_'):
            super().__setattr__(name, value)
            return
            
        if name not in self._default_params:
            raise AttributeError(f"Parameter {name} not found")
            
        if self._validators[name] is not None and not self._validators[name](value):
            raise ValueError(f"Invalid value for parameter {name}: {value}")
            
        self._default_params[name] = value
        
    def get_description(self, name: str) -> str:
        """Get parameter description"""
        return self._descriptions.get(name, "No description available")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary"""
        return {
            'default': self._default_params.copy(),
            'variations': [
                {
                    'dimension': var.dimension.name if var.dimension else None,
                    'neighborhood': var.neighborhood.name if var.neighborhood else None,
                    'params': var.params.copy()
                }
                for var in self._variations
            ]
        }

class TieBreaker(Enum):
    HIGHER_STATE = auto()
    LOWER_STATE = auto()
    MORE_CONNECTIONS = auto()
    FEWER_CONNECTIONS = auto()
    HIGHER_STATE_MORE_NEIGHBORS = auto()
    LOWER_STATE_FEWER_NEIGHBORS = auto()
    HIGHER_STATE_FEWER_NEIGHBORS = auto()
    LOWER_STATE_MORE_NEIGHBORS = auto()
    RANDOM = auto()  # Add a random option
    AGREEMENT = auto() # ADDED: New tie-breaker strategy

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
        elif tiebreaker_type == TieBreaker.AGREEMENT: # ADDED: New tie-breaker strategy
            return False # Always remove edge if they disagree
        else:
            raise ValueError(f"Invalid tiebreaker type: {tiebreaker_type}")
               
################################################
#            RULE LIBRARY HELPERS              #
################################################

@dataclass
class SymmetryData:
    """Container for symmetry calculation results"""
    score: float
    axes: np.ndarray
    rotation_score: float
    axis_alignments: Dict[int, float]
    symmetric_pairs: List[Tuple[int, int]]

@njit(cache=True)
def calculate_vectors_and_angles(center: np.ndarray, 
                               points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate vectors, angles, and distances from center to points"""
    vectors = points - center
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    distances = np.linalg.norm(vectors, axis=1)
    return vectors, angles, distances

@njit(cache=True)
def calculate_rotation_score(center: np.ndarray,
                           neighbor_positions: np.ndarray,
                           neighbor_states: np.ndarray,
                           rotational_order: int,
                           rotation_tolerance: float,
                           symmetry_tolerance: float) -> float:
    """Calculate rotational symmetry score"""
    if len(neighbor_positions) == 0:
        return 0.0
        
    vectors, angles, distances = calculate_vectors_and_angles(center, neighbor_positions)
    
    angle_step = 2*np.pi / rotational_order
    rotational_matches = 0
    total_checks = 0
    
    for i, angle in enumerate(angles):
        for rot in range(1, rotational_order):
            target_angle = angle + rot * angle_step
            if target_angle >= 2*np.pi:
                target_angle -= 2*np.pi
                
            angle_diffs = np.abs(angles - target_angle)
            angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
            closest_idx = np.argmin(angle_diffs)
            
            if (angle_diffs[closest_idx] <= rotation_tolerance and
                abs(distances[i] - distances[closest_idx]) <= symmetry_tolerance and
                abs(neighbor_states[i] - neighbor_states[closest_idx]) <= symmetry_tolerance):
                rotational_matches += 1
            total_checks += 1
    
    return rotational_matches / total_checks if total_checks > 0 else 0.0

@njit(cache=True)
def _calculate_symmetry_score(center: np.ndarray,
                           neighbor_positions: np.ndarray,
                           neighbor_states: np.ndarray,
                           symmetry_tolerance: float,
                           axis_angle_tolerance: float) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Calculate symmetry score and identify symmetric pairs
    
    Returns:
        Tuple of (symmetry_score, list_of_symmetric_pairs)
    """
    if len(neighbor_positions) == 0:
        return 0.0, []
        
    vectors, angles, distances = calculate_vectors_and_angles(center, neighbor_positions)
    
    symmetric_count = 0
    total_pairs = 0
    symmetric_pairs = []
    
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            angle_diff = abs(angles[i] - angles[j])
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            
            distance_match = abs(distances[i] - distances[j]) <= symmetry_tolerance
            state_match = abs(neighbor_states[i] - neighbor_states[j]) <= symmetry_tolerance
            
            if (abs(angle_diff - np.pi) <= axis_angle_tolerance and 
                distance_match and state_match):
                symmetric_count += 1
                symmetric_pairs.append((i, j))
            total_pairs += 1
    
    symmetry_score = symmetric_count / total_pairs if total_pairs > 0 else 0.0
    return symmetry_score, symmetric_pairs

def calculate_symmetry_score(center: np.ndarray,
                           neighbor_positions: np.ndarray,
                           neighbor_states: np.ndarray,
                           symmetry_tolerance: float,
                           axis_angle_tolerance: float) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Calculate symmetry score and identify symmetric pairs
    
    Returns:
        Tuple of (symmetry_score, list_of_symmetric_pairs)
    """
    return _calculate_symmetry_score(center, neighbor_positions, neighbor_states, symmetry_tolerance, axis_angle_tolerance)

@njit(cache=True)
def _calculate_axis_alignment(center: np.ndarray,
                           point: np.ndarray,
                           axes: np.ndarray) -> float:
    """Calculate alignment of point with symmetry axes"""
    if len(axes) == 0:
        return 0.0
        
    vector = point - center
    angle = np.arctan2(vector[1], vector[0])
    
    angle_diffs = np.abs(angle - axes)
    angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
    min_diff = np.min(angle_diffs)
    
    # Convert to alignment score (1 = perfect alignment, 0 = maximum misalignment)
    return 1.0 - (min_diff / (np.pi/2))

def calculate_axis_alignment(center: np.ndarray,
                           point: np.ndarray,
                           axes: np.ndarray) -> float:
    """Calculate alignment of point with symmetry axes"""
    return _calculate_axis_alignment(center, point, axes)

@dataclass
class HierarchyData:
    """Container for hierarchy calculation results"""
    level: int                      # Current level in hierarchy
    level_neighbors: Dict[int, List[int]]  # Neighbors by level {level: [node_indices]}
    influence_up: float             # Influence from lower levels
    influence_down: float           # Influence from higher levels
    level_density: float            # Density of connections within level
    cross_level_density: float      # Density of connections between levels

@njit(cache=True)
def _calculate_hierarchy_level(node_idx: int,
                            neighbor_indices: np.ndarray,
                            edge_matrix: np.ndarray,
                            neighbor_levels: np.ndarray,
                            max_level: int) -> int:
    """
    Calculate hierarchical level for a node based on neighbor levels
    
    Args:
        node_idx: Index of node to calculate level for
        neighbor_indices: Array of neighbor indices
        edge_matrix: Edge matrix
        neighbor_levels: Array of known neighbor levels
        max_level: Maximum allowed hierarchy level
    
    Returns:
        Calculated level for the node
    """
    if len(neighbor_indices) == 0:
        return 0
        
    # Get levels of connected neighbors
    connected_neighbors = neighbor_indices[edge_matrix[node_idx, neighbor_indices] > 0]
    
    if not len(connected_neighbors):
        return 0
        
    # Calculate level based on neighbors
    avg_neighbor_level = np.mean(neighbor_levels)
    max_neighbor_level = np.max(neighbor_levels)
    min_neighbor_level = np.min(neighbor_levels)
    
    # Determine level based on connection patterns
    if len(neighbor_levels) > 2:
        # Node connects multiple levels - likely a bridge node
        return min(int((max_neighbor_level + min_neighbor_level) / 2), max_level)
    else:
        # Node follows standard hierarchy
        return min(int(avg_neighbor_level + 0.5), max_level)
    
@njit(cache=True)
def _calculate_level_influence(node_idx: int,
                            node_level: int,
                            neighbor_indices: np.ndarray,
                            edge_matrix: np.ndarray,
                            neighbor_levels: np.ndarray,
                            neighbor_states: np.ndarray) -> Tuple[float, float]:
    """
    Calculate influence from higher and lower levels
    
    Returns:
        Tuple of (influence_up, influence_down)
    """
    if len(neighbor_indices) == 0:
        return 0.0, 0.0
        
    influence_up = 0.0
    influence_down = 0.0
    total_neighbors = 0
    
    for i in range(len(neighbor_indices)):
        n = neighbor_indices[i]
        if edge_matrix[node_idx, n]:
            neighbor_level = neighbor_levels[i]
            if neighbor_level > node_level:
                influence_down += neighbor_states[i]
            elif neighbor_level < node_level:
                influence_up += neighbor_states[i]
            total_neighbors += 1
                
    if total_neighbors == 0:
        return 0.0, 0.0
        
    return (float(influence_up / total_neighbors), float(influence_down / total_neighbors))

def calculate_level_influence(node_idx: int,
                            node_level: int,
                            neighborhood_data: NeighborhoodData,
                            current_levels: Dict[int, int]) -> Tuple[float, float]:
    """
    Calculate influence from higher and lower levels
    
    Returns:
        Tuple of (influence_up, influence_down)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0
        
    # Get levels of connected neighbors
    neighbor_levels = np.array([current_levels.get(n, 0) for n in neighbors], dtype=np.int64)
    neighbor_states = neighborhood_data.states[neighbors]
    
    influence_up, influence_down = _calculate_level_influence(
        node_idx,
        node_level,
        neighbors,
        neighborhood_data.edge_matrix,
        neighbor_levels,
        neighbor_states
    )
    
    return influence_up, influence_down

def analyze_hierarchy(node_idx: int,
                     neighborhood_data: NeighborhoodData,
                     current_levels: Dict[int, int],
                     max_level: int) -> HierarchyData:
    """
    Comprehensive hierarchy analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        current_levels: Dictionary of current node levels
        max_level: Maximum allowed hierarchy level
    
    Returns:
        HierarchyData containing hierarchy metrics
    """
    # Calculate node's level
    neighbor_levels = np.array([current_levels.get(n, 0) for n in range(len(neighborhood_data.states))], dtype=np.int64)
    level = _calculate_hierarchy_level(node_idx, neighborhood_data.get_neighbor_indices(node_idx), neighborhood_data.edge_matrix, neighbor_levels, max_level)
    
    # Get neighbors by level
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    level_neighbors = defaultdict(list)
    for n in neighbors:
        if neighborhood_data.edge_matrix[node_idx, n] > 0:
            n_level = current_levels.get(n, 0)
            level_neighbors[n_level].append(n)
            
    # Calculate influences
    influence_up, influence_down = calculate_level_influence(
        node_idx, level, neighborhood_data, current_levels
    )
    
    # Calculate density metrics
    same_level_edges = sum(1 for n in level_neighbors[level]
                          if neighborhood_data.edge_matrix[node_idx, n] > 0)
    level_density = (same_level_edges / len(level_neighbors[level]) 
                    if level_neighbors[level] else 0.0)
    
    cross_edges = sum(1 for l in level_neighbors if l != level
                     for n in level_neighbors[l]
                     if neighborhood_data.edge_matrix[node_idx, n] > 0)
    cross_level_density = (cross_edges / (len(neighbors) - len(level_neighbors[level]))
                          if len(neighbors) > len(level_neighbors[level]) else 0.0)
    
    return HierarchyData(
        level=level,
        level_neighbors=dict(level_neighbors),
        influence_up=influence_up,
        influence_down=influence_down,
        level_density=level_density,
        cross_level_density=cross_level_density
    )

@njit(cache=True)
def calculate_symmetry_axes(center: np.ndarray, 
                          neighbor_positions: np.ndarray,
                          num_axes: int) -> np.ndarray:
    """Calculate symmetry axes based on neighbor positions"""
    if len(neighbor_positions) == 0:
        return np.array([])
        
    # Calculate angles to all neighbors
    vectors = neighbor_positions - center
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    # Sort angles
    angles = np.sort(angles)
    
    # Calculate angle differences
    angle_diffs = np.diff(angles)
    angle_diffs = np.append(angle_diffs, 2*np.pi - (angles[-1] - angles[0]))
    
    # Find most evenly spaced angles
    best_axes = []
    min_variance = float('inf')
    
    # Try different starting angles
    for start_angle in angles:
        # Generate candidate axes
        axis_angles = np.array([
            start_angle + i * (2*np.pi / num_axes)
            for i in range(num_axes)
        ])
        
        # Calculate variance in neighbor distances to axes
        variances = []
        for axis in axis_angles:
            # Calculate distances from neighbors to this axis
            angle_diffs = np.abs(angles - axis)
            angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
            variances.append(np.var(angle_diffs))
            
        total_variance = np.mean(variances)
        
        if total_variance < min_variance:
            min_variance = total_variance
            best_axes = axis_angles
            
    return np.array(best_axes)

def analyze_symmetry(center: np.ndarray,
                    neighbor_positions: np.ndarray,
                    neighbor_states: np.ndarray,
                    params: RuleParameters) -> SymmetryData:
    """
    Comprehensive symmetry analysis returning all relevant metrics
    
    Args:
        center: Position of central node
        neighbor_positions: Positions of neighboring nodes
        neighbor_states: States of neighboring nodes
        params: Rule parameters containing symmetry-related settings
    
    Returns:
        SymmetryData containing all symmetry metrics
    """
    # Calculate basic symmetry score and pairs
    symmetry_score, symmetric_pairs = calculate_symmetry_score(
        center, neighbor_positions, neighbor_states,
        params.symmetry_tolerance, params.axis_angle_tolerance
    )
    
    # Calculate rotational symmetry
    rotation_score = calculate_rotation_score(
        center, neighbor_positions, neighbor_states,
        params.rotational_order, params.rotation_tolerance,
        params.symmetry_tolerance
    )
    
    # Calculate symmetry axes
    axes = calculate_symmetry_axes(
        center, neighbor_positions, params.num_symmetry_axes
    )
    
    # Calculate axis alignments for each neighbor
    axis_alignments = {}
    for i in range(len(neighbor_positions)):
        alignment = calculate_axis_alignment(center, neighbor_positions[i], axes)
        axis_alignments[i] = alignment
    
    return SymmetryData(
        score=symmetry_score,
        axes=axes,
        rotation_score=rotation_score,
        axis_alignments=axis_alignments,
        symmetric_pairs=symmetric_pairs
    )

@dataclass
class ResonanceData:
    """Container for resonance/wave pattern calculation results"""
    phase: float                    # Current phase (0-2π)
    frequency: float                # Current frequency
    amplitude: float                # Current amplitude
    phase_coherence: float         # Measure of phase alignment with neighbors
    frequency_match: float         # Measure of frequency match with neighbors
    standing_wave_strength: float  # Measure of standing wave pattern
    wave_nodes: List[int]          # Nodes participating in wave pattern
    resonance_score: float         # Overall resonance score

@njit(cache=True)
def calculate_phase_metrics(node_idx: int,
                          node_phase: float,
                          neighborhood_data: NeighborhoodData,
                          node_phases: Dict[int, float]) -> Tuple[float, float]:
    """
    Calculate phase coherence and frequency match with neighbors
    
    Returns:
        Tuple of (phase_coherence, frequency_match)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0
        
    # Calculate phase differences
    phase_diffs = []
    for n in neighbors:
        if n in node_phases:
            diff = abs(node_phase - node_phases[n])
            # Normalize to [-π, π]
            if diff > np.pi:
                diff = 2*np.pi - diff
            phase_diffs.append(diff)
            
    if not phase_diffs:
        return 0.0, 0.0
        
    # Phase coherence (1 = perfect alignment, 0 = random)
    phase_coherence = 1.0 - (np.mean(phase_diffs) / np.pi)
    
    # Frequency match (based on phase difference stability)
    freq_diffs = np.diff(sorted(phase_diffs))
    frequency_match = 1.0 - (np.std(freq_diffs) if len(freq_diffs) > 0 else 0.0)
    
    return float(phase_coherence), float(frequency_match)

@njit(cache=True)
def calculate_standing_wave_pattern(node_idx: int,
                                  node_positions: np.ndarray,
                                  neighborhood_data: NeighborhoodData,
                                  node_phases: Dict[int, float],
                                  wavelength: float) -> Tuple[float, List[int]]:
    """
    Calculate standing wave pattern strength and participating nodes
    
    Returns:
        Tuple of (standing_wave_strength, wave_node_indices)
    """
    if node_positions is None:
        return 0.0, []
        
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, []
        
    center_pos = node_positions[node_idx]
    wave_nodes = []
    phase_positions = []
    
    # Collect phases and positions
    for n in neighbors:
        if n in node_phases:
            pos = node_positions[n]
            dist = np.linalg.norm(pos - center_pos)
            if dist > 0:  # Avoid division by zero
                expected_phase = 2 * np.pi * (dist / wavelength)
                actual_phase = node_phases[n]
                # Compare expected vs actual phase
                phase_diff = abs(expected_phase - actual_phase) % (2 * np.pi)
                if phase_diff < np.pi/4:  # Node follows standing wave pattern
                    wave_nodes.append(n)
                    phase_positions.append(phase_diff)
                    
    if not phase_positions:
        return 0.0, []
        
    # Calculate standing wave strength
    strength = 1.0 - (np.mean(phase_positions) / (np.pi/2))
    return float(strength), wave_nodes

def analyze_resonance(node_idx: int,
                     node_positions: Optional[np.ndarray],
                     neighborhood_data: NeighborhoodData,
                     node_phases: Dict[int, float],
                     node_frequencies: Dict[int, float],
                     wavelength: float) -> ResonanceData:
    """
    Comprehensive resonance analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        node_positions: Array of node positions
        neighborhood_data: Network neighborhood data
        node_phases: Dictionary of current node phases
        node_frequencies: Dictionary of current node frequencies
        wavelength: Current wavelength for standing wave calculation
    
    Returns:
        ResonanceData containing all resonance metrics
    """
    # Get current node's phase and frequency
    phase = node_phases.get(node_idx, 0.0)
    frequency = node_frequencies.get(node_idx, 0.0)
    
    # Calculate phase metrics
    phase_coherence, frequency_match = calculate_phase_metrics(
        node_idx, phase, neighborhood_data, node_phases
    )
    
    # Calculate standing wave pattern if positions available
    if node_positions is not None:
        standing_wave_strength, wave_nodes = calculate_standing_wave_pattern(
            node_idx, node_positions, neighborhood_data, node_phases, wavelength
        )
    else:
        standing_wave_strength, wave_nodes = 0.0, []
    
    # Calculate amplitude based on neighbor states
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    amplitude = np.mean(neighborhood_data.states[neighbors]) if len(neighbors) > 0 else 0.0
    
    # Calculate overall resonance score
    resonance_score = (phase_coherence + frequency_match + standing_wave_strength) / 3.0
    
    return ResonanceData(
        phase=phase,
        frequency=frequency,
        amplitude=float(amplitude),
        phase_coherence=phase_coherence,
        frequency_match=frequency_match,
        standing_wave_strength=standing_wave_strength,
        wave_nodes=wave_nodes,
        resonance_score=resonance_score
    )

@dataclass
class ModuleData:
    """Container for module analysis results"""
    module_id: int                      # Current module identifier
    module_nodes: Set[int]              # Nodes in same module
    module_density: float               # Internal connection density
    external_density: float             # External connection density
    module_role: str                    # Role within module ('core', 'boundary', 'bridge')
    module_stability: float             # Measure of module stability
    inter_module_connections: Dict[int, Set[int]]  # Connections to other modules {module_id: {node_ids}}
    specialization_score: float         # Measure of module specialization

@njit(cache=True)
def calculate_module_metrics(node_idx: int,
                           neighborhood_data: NeighborhoodData,
                           module_assignments: Dict[int, int]) -> Tuple[float, float]:
    """
    Calculate internal and external connection densities for a node's module
    
    Returns:
        Tuple of (internal_density, external_density)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0
        
    current_module = module_assignments.get(node_idx, -1)
    if current_module == -1:
        return 0.0, 0.0
        
    internal_connections = 0
    external_connections = 0
    internal_possible = 0
    external_possible = 0
    
    for n in neighbors:
        n_module = module_assignments.get(n, -1)
        if n_module == current_module:
            internal_possible += 1
            if neighborhood_data.edge_matrix[node_idx, n]:
                internal_connections += 1
        else:
            external_possible += 1
            if neighborhood_data.edge_matrix[node_idx, n]:
                external_connections += 1
                
    internal_density = (internal_connections / internal_possible 
                       if internal_possible > 0 else 0.0)
    external_density = (external_connections / external_possible 
                       if external_possible > 0 else 0.0)
    
    return float(internal_density), float(external_density)

@njit(cache=True)
def determine_module_role(node_idx: int,
                         neighborhood_data: NeighborhoodData,
                         module_assignments: Dict[int, int],
                         internal_density: float,
                         external_density: float) -> str:
    """
    Determine node's role within its module
    
    Returns:
        Role classification ('core', 'boundary', 'bridge')
    """
    if internal_density > 0.7 and external_density < 0.3:
        return 'core'
    elif external_density > 0.5:
        return 'bridge'
    else:
        return 'boundary'

@njit(cache=True)
def _calculate_module_stability(node_idx: int,
                               neighbor_indices: np.ndarray,
                               edge_matrix: np.ndarray,
                               module_assignments: np.ndarray,
                               states: np.ndarray,
                               current_module: int) -> float:
    """
    Calculate stability of node's module based on state patterns
    
    Returns:
        Stability score (0-1)
    """
    if len(neighbor_indices) == 0:
        return 0.0
        
    # Get module neighbors
    module_neighbors = neighbor_indices[module_assignments[neighbor_indices] == current_module]
    
    if not len(module_neighbors):
        return 0.0
        
    # Calculate state pattern stability
    state_differences = np.abs(states[node_idx] - states[module_neighbors])
    
    return float(1.0 - (np.mean(state_differences) if len(state_differences) > 0 else 1.0))

def calculate_module_stability(node_idx: int,
                             neighborhood_data: NeighborhoodData,
                             module_assignments: Dict[int, int],
                             states: np.ndarray) -> float:
    """
    Calculate stability of node's module based on state patterns
    
    Returns:
        Stability score (0-1)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0
        
    current_module = module_assignments.get(node_idx, -1)
    if current_module == -1:
        return 0.0
    
    # Convert module assignments to numpy array
    module_assignments_array = np.array([module_assignments.get(n, -1) for n in range(len(states))], dtype=np.int64)
    
    return _calculate_module_stability(
        node_idx,
        neighbors,
        neighborhood_data.edge_matrix,
        module_assignments_array,
        states,
        current_module
    )
    
def analyze_module(node_idx: int,
                  neighborhood_data: NeighborhoodData,
                  module_assignments: Dict[int, int],
                  specialization_scores: Dict[int, float]) -> ModuleData:
    """
    Comprehensive module analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        module_assignments: Dictionary of current module assignments
        specialization_scores: Dictionary of module specialization scores
    
    Returns:
        ModuleData containing all module metrics
    """
    current_module = module_assignments.get(node_idx, -1)
    
    # Get basic metrics
    internal_density, external_density = calculate_module_metrics(
        node_idx, neighborhood_data, module_assignments
    )
    
    # Determine role
    role = determine_module_role(
        node_idx, neighborhood_data, module_assignments,
        internal_density, external_density
    )
    
    # Calculate stability
    stability = calculate_module_stability(
        node_idx, neighborhood_data, module_assignments,
        neighborhood_data.states
    )
    
    # Get module nodes and inter-module connections
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    module_nodes = {n for n in neighbors 
                   if module_assignments.get(n, -1) == current_module}
    
    inter_module_connections = defaultdict(set)
    for n in neighbors:
        n_module = module_assignments.get(n, -1)
        if n_module != current_module and n_module != -1:
            if neighborhood_data.edge_matrix[node_idx, n]:
                inter_module_connections[n_module].add(n)
    
    # Get specialization score
    specialization = specialization_scores.get(current_module, 0.0)
    
    return ModuleData(
        module_id=current_module,
        module_nodes=module_nodes,
        module_density=internal_density,
        external_density=external_density,
        module_role=role,
        module_stability=stability,
        inter_module_connections=dict(inter_module_connections),
        specialization_score=specialization
    )

@dataclass
class MemoryPattern:
    """Represents a stored pattern in the network"""
    pattern_id: int
    node_states: Dict[int, float]
    node_connections: Set[Tuple[int, int]]
    last_seen: int
    frequency: int
    stability: float
    influence: float

@dataclass
class MemoryData:
    """Container for memory analysis results"""
    matching_patterns: List[Tuple[MemoryPattern, float]]  # (pattern, match_score)
    best_match_score: float
    pattern_influence: float
    reconstruction_score: float
    memory_stability: float
    pattern_completion: float

@njit(cache=True)
def _calculate_pattern_match(current_states: np.ndarray,
                          current_connections: np.ndarray,
                          pattern_states: Dict[int, float],
                          pattern_connections: List[Tuple[int, int]],
                          neighborhood: np.ndarray) -> float:
    """
    Calculate how well current state matches a stored pattern
    
    Returns:
        Match score (0-1)
    """
    if len(neighborhood) == 0:
        return 0.0
        
    # Calculate state match
    state_matches = 0
    total_states = 0
    for idx in neighborhood:
        if idx in pattern_states:
            state_matches += 1 - abs(current_states[idx] - pattern_states[idx])
            total_states += 1
            
    state_score = state_matches / total_states if total_states > 0 else 0.0
    
    # Calculate connection match
    connection_matches = 0
    total_connections = 0
    for i in range(len(neighborhood)):
        for j in range(i + 1, len(neighborhood)):
            idx_i = neighborhood[i]
            idx_j = neighborhood[j]
            has_connection = current_connections[idx_i, idx_j] > 0
            should_have_connection = (idx_i, idx_j) in pattern_connections
            if has_connection == should_have_connection:
                connection_matches += 1
            total_connections += 1
                
    connection_score = connection_matches / total_connections if total_connections > 0 else 0.0
    
    # Combine scores
    return float(0.6 * state_score + 0.4 * connection_score)

def calculate_pattern_match(current_states: np.ndarray,
                          current_connections: np.ndarray,
                          pattern_states: Dict[int, float],
                          pattern_connections: Set[Tuple[int, int]],
                          neighborhood: np.ndarray) -> float:
    """
    Calculate how well current state matches a stored pattern
    
    Returns:
        Match score (0-1)
    """
    # Convert pattern_connections to a list of tuples
    pattern_connections_list = list(pattern_connections)
    
    return _calculate_pattern_match(
        current_states,
        current_connections,
        pattern_states,
        pattern_connections_list,
        neighborhood
    )
    
@njit(cache=True)
def calculate_pattern_completion(current_states: np.ndarray,
                               pattern_states: Dict[int, float],
                               neighborhood: np.ndarray) -> float:
    """
    Calculate pattern completion percentage
    
    Returns:
        Completion score (0-1)
    """
    if len(neighborhood) == 0:
        return 0.0
        
    matching_nodes = 0
    pattern_nodes = 0
    
    for idx in neighborhood:
        if idx in pattern_states:
            pattern_nodes += 1
            if abs(current_states[idx] - pattern_states[idx]) < 0.1:
                matching_nodes += 1
                
    return float(matching_nodes / pattern_nodes if pattern_nodes > 0 else 0.0)

def calculate_memory_stability(patterns: List[MemoryPattern],
                             current_step: int,
                             decay_rate: float) -> Dict[int, float]:
    """
    Calculate stability scores for stored patterns
    
    Returns:
        Dictionary of pattern_id to stability score
    """
    stability_scores = {}
    
    for pattern in patterns:
        # Calculate time-based decay
        time_factor = np.exp(-decay_rate * (current_step - pattern.last_seen))
        # Combine with frequency and stored stability
        stability = (0.4 * time_factor + 
                    0.3 * min(pattern.frequency / 10.0, 1.0) +
                    0.3 * pattern.stability)
        stability_scores[pattern.pattern_id] = stability
        
    return stability_scores

def analyze_memory(node_idx: int,
                  neighborhood_data: NeighborhoodData,
                  patterns: List[MemoryPattern],
                  current_step: int,
                  decay_rate: float) -> MemoryData:
    """
    Comprehensive memory analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        patterns: List of stored memory patterns
        current_step: Current simulation step
        decay_rate: Rate of memory decay
    
    Returns:
        MemoryData containing all memory metrics
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    
    # Calculate pattern matches
    matching_patterns = []
    best_match_score = 0.0
    
    for pattern in patterns:
        match_score = calculate_pattern_match(
            neighborhood_data.states,
            neighborhood_data.edge_matrix,
            pattern.node_states,
            pattern.node_connections,
            neighbors
        )
        if match_score > 0:
            matching_patterns.append((pattern, match_score))
            best_match_score = max(best_match_score, match_score)
    
    # Sort by match score
    matching_patterns.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate stability scores
    stability_scores = calculate_memory_stability(patterns, current_step, decay_rate)
    
    # Calculate pattern influence
    pattern_influence = 0.0
    if matching_patterns:
        # Weight influence by match score and pattern stability
        for pattern, match_score in matching_patterns:
            stability = stability_scores[pattern.pattern_id]
            pattern_influence += match_score * stability * pattern.influence
        pattern_influence /= len(matching_patterns)
    
    # Calculate reconstruction score
    reconstruction_score = 0.0
    if matching_patterns:
        best_pattern = matching_patterns[0][0]
        reconstruction_score = calculate_pattern_completion(
            neighborhood_data.states,
            best_pattern.node_states,
            neighbors
        )
    
    # Calculate overall memory stability
    memory_stability = float(np.mean(list(stability_scores.values()))) if stability_scores else 0.0
    
    # Calculate pattern completion tendency
    pattern_completion = reconstruction_score if matching_patterns else 0.0
    
    return MemoryData(
        matching_patterns=matching_patterns,
        best_match_score=best_match_score,
        pattern_influence=pattern_influence,
        reconstruction_score=reconstruction_score,
        memory_stability=memory_stability,
        pattern_completion=pattern_completion
    )

@dataclass
class FlowData:
    """Container for flow analysis results"""
    flow_direction: np.ndarray          # Primary direction of flow
    flow_strength: float                # Strength of flow
    bottleneck_score: float            # Measure of flow bottleneck (0-1, 1=severe)
    path_redundancy: float             # Measure of alternative paths
    capacity_utilization: float        # Current flow capacity usage
    upstream_pressure: float           # Pressure from upstream nodes
    downstream_pressure: float         # Pressure from downstream nodes
    flow_stability: float             # Measure of flow stability
    critical_paths: Set[Tuple[int, int]]  # Critical path connections

@njit(cache=True)
def _calculate_flow_direction(node_idx: int,
                           node_positions: np.ndarray,
                           neighbor_indices: np.ndarray,
                           edge_matrix: np.ndarray,
                           neighbor_states: np.ndarray,
                           flow_bias: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate primary flow direction and strength
    
    Returns:
        Tuple of (direction_vector, flow_strength)
    """
    if len(neighbor_indices) == 0:
        return np.zeros(3), 0.0
        
    # Calculate weighted direction based on connections and states
    flow_vector = np.zeros(3)
    total_weight = 0.0
    
    for n in neighbor_indices:
        if edge_matrix[node_idx, n]:
            direction = node_positions[n] - node_positions[node_idx]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
                # Weight by state and alignment with global bias
                weight = (neighbor_states[n] * 
                         (1.0 + np.dot(direction, flow_bias)))
                flow_vector += direction * weight
                total_weight += weight
    
    if total_weight > 0:
        flow_vector = flow_vector / total_weight
        
    flow_strength = np.linalg.norm(flow_vector)
    if flow_strength > 0:
        flow_vector = flow_vector / flow_strength
        
    return flow_vector, float(flow_strength)

def calculate_flow_direction(node_idx: int,
                           node_positions: np.ndarray,
                           neighborhood_data: NeighborhoodData,
                           flow_bias: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate primary flow direction and strength
    
    Returns:
        Tuple of (direction_vector, flow_strength)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    return _calculate_flow_direction(
        node_idx,
        node_positions,
        neighbors,
        neighborhood_data.edge_matrix,
        neighborhood_data.states,
        flow_bias
    )
    
@njit(cache=True)
def _calculate_bottleneck(node_idx: int,
                        neighbor_indices: np.ndarray,
                        edge_matrix: np.ndarray,
                        flow_direction: np.ndarray,
                        node_positions: np.ndarray) -> Tuple[float, float]:
    """
    Calculate bottleneck severity and path redundancy
    
    Returns:
        Tuple of (bottleneck_score, redundancy_score)
    """
    if len(neighbor_indices) == 0:
        return 1.0, 0.0
        
    # Count upstream and downstream connections
    upstream_count = 0
    downstream_count = 0
    
    for n in neighbor_indices:
        if edge_matrix[node_idx, n]:
            direction = node_positions[n] - node_positions[node_idx]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
                alignment = np.dot(direction, flow_direction)
                if alignment > 0.1:  # Downstream
                    downstream_count += 1
                elif alignment < -0.1:  # Upstream
                    upstream_count += 1
    
    # Calculate bottleneck score
    min_connections = min(upstream_count, downstream_count)
    max_connections = max(upstream_count, downstream_count)
    bottleneck = 1.0 - (min_connections / max_connections if max_connections > 0 else 0.0)
    
    # Calculate path redundancy
    redundancy = min_connections / len(neighbor_indices) if len(neighbor_indices) > 0 else 0.0
    
    return float(bottleneck), float(redundancy)

def calculate_bottleneck(node_idx: int,
                        neighborhood_data: NeighborhoodData,
                        flow_direction: np.ndarray,
                        node_positions: np.ndarray) -> Tuple[float, float]:
    """
    Calculate bottleneck severity and path redundancy
    
    Returns:
        Tuple of (bottleneck_score, redundancy_score)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    return _calculate_bottleneck(
        node_idx,
        neighbors,
        neighborhood_data.edge_matrix,
        flow_direction,
        node_positions
    )
    
@njit(cache=True)
def calculate_flow_pressure(node_idx: int,
                          neighborhood_data: NeighborhoodData,
                          flow_direction: np.ndarray,
                          node_positions: np.ndarray) -> Tuple[float, float]:
    """
    Calculate upstream and downstream pressure
    
    Returns:
        Tuple of (upstream_pressure, downstream_pressure)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0
        
    upstream_pressure = 0.0
    downstream_pressure = 0.0
    upstream_count = 0
    downstream_count = 0
    
    for n in neighbors:
        if neighborhood_data.edge_matrix[node_idx, n]:
            direction = node_positions[n] - node_positions[node_idx]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
                alignment = np.dot(direction, flow_direction)
                if alignment > 0.1:  # Downstream
                    downstream_pressure += neighborhood_data.states[n]
                    downstream_count += 1
                elif alignment < -0.1:  # Upstream
                    upstream_pressure += neighborhood_data.states[n]
                    upstream_count += 1
    
    upstream_pressure = (upstream_pressure / upstream_count 
                        if upstream_count > 0 else 0.0)
    downstream_pressure = (downstream_pressure / downstream_count 
                         if downstream_count > 0 else 0.0)
    
    return float(upstream_pressure), float(downstream_pressure)

def analyze_flow(node_idx: int,
                neighborhood_data: NeighborhoodData,
                node_positions: np.ndarray,
                flow_bias: np.ndarray,
                capacity_threshold: float) -> FlowData:
    """
    Comprehensive flow analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        node_positions: Array of node positions
        flow_bias: Global flow direction bias
        capacity_threshold: Threshold for capacity utilization
    
    Returns:
        FlowData containing all flow metrics
    """
    # Calculate primary flow direction and strength
    flow_direction, flow_strength = calculate_flow_direction(
        node_idx, node_positions, neighborhood_data, flow_bias
    )
    
    # Calculate bottleneck and redundancy
    bottleneck_score, path_redundancy = calculate_bottleneck(
        node_idx, neighborhood_data, flow_direction, node_positions
    )
    
    # Calculate pressures
    upstream_pressure, downstream_pressure = calculate_flow_pressure(
        node_idx, neighborhood_data, flow_direction, node_positions
    )
    
    # Calculate capacity utilization
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    active_connections = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])
    capacity_utilization = (active_connections / len(neighbors) 
                          if len(neighbors) > 0 else 0.0)
    
    # Calculate flow stability
    flow_stability = (flow_strength * (1.0 - bottleneck_score) * 
                     path_redundancy * (1.0 - abs(capacity_utilization - 0.5)))
    
    # Identify critical paths
    critical_paths = set()
    for n in neighbors:
        if neighborhood_data.edge_matrix[node_idx, n]:
            direction = node_positions[n] - node_positions[node_idx]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
                if (abs(np.dot(direction, flow_direction)) > 0.8 and
                    bottleneck_score > 0.5):
                    critical_paths.add((node_idx, n))
    
    return FlowData(
        flow_direction=flow_direction,
        flow_strength=flow_strength,
        bottleneck_score=bottleneck_score,
        path_redundancy=path_redundancy,
        capacity_utilization=capacity_utilization,
        upstream_pressure=upstream_pressure,
        downstream_pressure=downstream_pressure,
        flow_stability=flow_stability,
        critical_paths=critical_paths
    )
       
@dataclass
class Territory:
    """Represents a controlled territory in the network"""
    territory_id: int
    controller_id: int
    member_nodes: Set[int]
    boundary_nodes: Set[int]
    resource_level: float
    strength: float
    alliances: Set[int]  # Set of allied territory IDs

@dataclass
class CompetitionData:
    """Container for competition analysis results"""
    territory: Optional[Territory]     # Current territory
    territory_position: str            # Position in territory ('core', 'boundary', 'contested')
    resource_access: float            # Access to resources (0-1)
    competitive_pressure: float       # Pressure from competitors (0-1)
    alliance_strength: float          # Strength from alliances (0-1)
    territorial_stability: float      # Stability of territory (0-1)
    expansion_potential: float        # Potential for expansion (0-1)
    defense_strength: float          # Defensive capability (0-1)
    contested_neighbors: Set[int]     # Neighboring nodes under competition

@njit(cache=True)
def _calculate_territory_metrics(node_idx: int,
                              neighbor_indices: np.ndarray,
                              territory_assignments: np.ndarray,
                              resource_distribution: np.ndarray,
                              current_territory: int) -> Tuple[float, float, int]:
    """
    Calculate territory-related metrics for a node
    
    Returns:
        Tuple of (resource_access, competitive_pressure, position)
    """
    if len(neighbor_indices) == 0:
        return 0.0, 0.0, 0
        
    # Calculate resource access
    resource_access = resource_distribution[node_idx]
    different_territory_count = 0
    
    for n in neighbor_indices:
        if territory_assignments[n] == current_territory:
            resource_access += resource_distribution[n] * 0.5
        else:
            different_territory_count += 1
            
    # Calculate competitive pressure
    competitive_pressure = different_territory_count / len(neighbor_indices)
    
    # Determine position
    if competitive_pressure == 0:
        position = 1  # core
    elif competitive_pressure > 0.5:
        position = 2  # contested
    else:
        position = 3  # boundary
        
    return float(resource_access), float(competitive_pressure), int(position)

def calculate_territory_metrics(node_idx: int,
                              neighborhood_data: NeighborhoodData,
                              territory_assignments: Dict[int, int],
                              resource_distribution: np.ndarray) -> Tuple[float, float, str]:
    """
    Calculate territory-related metrics for a node
    
    Returns:
        Tuple of (resource_access, competitive_pressure, position)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0, 'isolated'
        
    current_territory = territory_assignments.get(node_idx, -1)
    if current_territory == -1:
        return 0.0, 1.0, 'contested'
        
    # Convert territory assignments to numpy array
    territory_assignments_array = np.array([territory_assignments.get(n, -1) for n in range(len(neighborhood_data.states))], dtype=np.int64)
    
    resource_access, competitive_pressure, position_code = _calculate_territory_metrics(
        node_idx,
        neighbors,
        territory_assignments_array,
        resource_distribution,
        current_territory
    )
    
    # Convert position code to string
    if position_code == 1:
        position = 'core'
    elif position_code == 2:
        position = 'contested'
    else:
        position = 'boundary'
        
    return resource_access, competitive_pressure, position

@njit(cache=True)
def _calculate_alliance_metrics(node_idx: int,
                             neighbor_indices: np.ndarray,
                             territory_assignments: np.ndarray,
                             alliances: np.ndarray,
                             states: np.ndarray,
                             current_territory: int) -> Tuple[float, float]:
    """
    Calculate alliance-related metrics
    
    Returns:
        Tuple of (alliance_strength, defense_strength)
    """
    if len(neighbor_indices) == 0:
        return 0.0, 0.0
        
    # Calculate alliance strength
    allied_neighbors = 0
    friendly_strength = 0.0
    total_strength = 0.0
    
    for n in neighbor_indices:
        n_territory = territory_assignments[n]
        if n_territory == current_territory or n_territory in alliances:
            allied_neighbors += 1
            friendly_strength += states[n]
        total_strength += states[n]
            
    alliance_strength = allied_neighbors / len(neighbor_indices)
    defense_strength = friendly_strength / total_strength if total_strength > 0 else 0.0
    
    return float(alliance_strength), float(defense_strength)

def calculate_alliance_metrics(node_idx: int,
                             neighborhood_data: NeighborhoodData,
                             territory_assignments: Dict[int, int],
                             alliances: Dict[int, Set[int]]) -> Tuple[float, float]:
    """
    Calculate alliance-related metrics
    
    Returns:
        Tuple of (alliance_strength, defense_strength)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0
        
    current_territory = territory_assignments.get(node_idx, -1)
    if current_territory == -1:
        return 0.0, 0.0
        
    # Convert territory assignments and alliances to numpy arrays
    territory_assignments_array = np.array([territory_assignments.get(n, -1) for n in range(len(neighborhood_data.states))], dtype=np.int64)
    alliances_set = alliances.get(current_territory, set())
    alliances_array = np.array(list(alliances_set), dtype=np.int64)
    
    alliance_strength, defense_strength = _calculate_alliance_metrics(
        node_idx,
        neighbors,
        territory_assignments_array,
        alliances_array,
        neighborhood_data.states,
        current_territory
    )
    
    return alliance_strength, defense_strength

def calculate_expansion_potential(node_idx: int,
                                neighborhood_data: NeighborhoodData,
                                territory_assignments: Dict[int, int],
                                resource_distribution: np.ndarray,
                                territory_strengths: Dict[int, float]) -> float:
    """
    Calculate potential for territory expansion
    
    Returns:
        Expansion potential score (0-1)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0
        
    current_territory = territory_assignments.get(node_idx, -1)
    if current_territory == -1:
        return 0.0
        
    current_strength = territory_strengths.get(current_territory, 0.0)
    
    # Evaluate expansion opportunities
    expansion_score = 0.0
    opportunities = 0
    
    for n in neighbors:
        n_territory = territory_assignments.get(n, -1)
        if n_territory != current_territory:
            # Consider resource value and relative strength
            n_strength = territory_strengths.get(n_territory, 0.0)
            strength_ratio = current_strength / (n_strength + 0.1)  # Avoid division by zero
            resource_value = resource_distribution[n]
            
            opportunity_score = resource_value * strength_ratio
            expansion_score += opportunity_score
            opportunities += 1
            
    return float(expansion_score / opportunities if opportunities > 0 else 0.0)

def analyze_competition(node_idx: int,
                       neighborhood_data: NeighborhoodData,
                       territories: Dict[int, Territory],
                       territory_assignments: Dict[int, int],
                       resource_distribution: np.ndarray) -> CompetitionData:
    """
    Comprehensive competition analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        territories: Dictionary of territory objects
        territory_assignments: Dictionary of node territory assignments
        resource_distribution: Array of resource values for nodes
    
    Returns:
        CompetitionData containing all competition metrics
    """
    current_territory_id = territory_assignments.get(node_idx, -1)
    current_territory = territories.get(current_territory_id, None)
    
    # Calculate basic metrics
    resource_access, competitive_pressure, position = calculate_territory_metrics(
        node_idx, neighborhood_data, territory_assignments, resource_distribution
    )
    
    # Calculate alliance metrics
    alliance_strength, defense_strength = calculate_alliance_metrics(
        node_idx, neighborhood_data, territory_assignments,
        {t.territory_id: t.alliances for t in territories.values()}
    )
    
    # Calculate territory strengths
    territory_strengths = {
        t.territory_id: t.strength for t in territories.values()
    }
    
    # Calculate expansion potential
    expansion_potential = calculate_expansion_potential(
        node_idx, neighborhood_data, territory_assignments,
        resource_distribution, territory_strengths
    )
    
    # Calculate territorial stability
    if current_territory:
        stability = (defense_strength * (1.0 - competitive_pressure) * 
                    (0.5 + 0.5 * alliance_strength))
    else:
        stability = 0.0
    
    # Identify contested neighbors
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    contested_neighbors = {
        n for n in neighbors
        if territory_assignments.get(n, -1) != current_territory_id
    }
    
    return CompetitionData(
        territory=current_territory,
        territory_position=position,
        resource_access=resource_access,
        competitive_pressure=competitive_pressure,
        alliance_strength=alliance_strength,
        territorial_stability=stability,
        expansion_potential=expansion_potential,
        defense_strength=defense_strength,
        contested_neighbors=contested_neighbors
    )
    
@dataclass
class FractalPattern:
    """Represents a fractal pattern in the network"""
    pattern_id: int
    scale: float                    # Current scale of pattern
    center_node: int               # Central node of pattern
    member_nodes: Set[int]         # Nodes in pattern
    symmetry_points: List[np.ndarray]  # Key symmetry points
    recursion_depth: int           # Current depth of recursion
    similarity_score: float        # Self-similarity measure

@dataclass
class FractalData:
    """Container for fractal analysis results"""
    current_pattern: Optional[FractalPattern]  # Current fractal pattern
    pattern_role: str              # Role in pattern ('center', 'branch', 'terminal')
    scale_position: float          # Position in current scale (0-1)
    self_similarity: float         # Local self-similarity measure
    fractal_dimension: float       # Local fractal dimension
    branching_score: float         # Measure of branching completeness
    scale_transition_potential: float  # Potential for scale transition
    pattern_stability: float       # Stability of fractal pattern
    recursive_patterns: List[FractalPattern]  # Nested patterns

@njit(cache=True)
def _calculate_fractal_metrics(node_idx: int,
                            neighbor_indices: np.ndarray,
                            node_positions: np.ndarray,
                            reference_scale: float) -> Tuple[float, float, int]:
    """
    Calculate basic fractal metrics for a node
    
    Returns:
        Tuple of (self_similarity, fractal_dimension, pattern_role)
    """
    if len(neighbor_indices) == 0:
        return 0.0, 1.0, 2  # terminal
        
    # Calculate distances to neighbors
    distances = np.zeros(len(neighbor_indices))
    for i, n in enumerate(neighbor_indices):
        diff = node_positions[n] - node_positions[node_idx]
        distances[i] = np.sqrt(np.sum(diff * diff))
    
    # Calculate scale ratios
    scale_ratios = distances / reference_scale
    
    # Calculate self-similarity through distance distribution
    unique_ratios = np.unique(np.round(scale_ratios, decimals=2))
    self_similarity = len(unique_ratios) / len(distances) if len(distances) > 0 else 0.0
    
    # Estimate local fractal dimension
    if len(distances) > 1:
        log_counts = np.log(np.arange(1, len(distances) + 1))
        log_distances = np.log(np.sort(distances))
        fractal_dim = np.abs(np.polyfit(log_distances, log_counts, 1)[0])
    else:
        fractal_dim = 1.0
    
    # Determine pattern role
    if len(neighbor_indices) >= 3 and self_similarity > 0.7:
        role = 0  # center
    elif len(neighbor_indices) == 1:
        role = 2  # terminal
    else:
        role = 1  # branch
        
    return float(self_similarity), float(fractal_dim), int(role)

def calculate_fractal_metrics(node_idx: int,
                            neighborhood_data: NeighborhoodData,
                            node_positions: np.ndarray,
                            reference_scale: float) -> Tuple[float, float, str]:
    """
    Calculate basic fractal metrics for a node
    
    Returns:
        Tuple of (self_similarity, fractal_dimension, pattern_role)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    self_similarity, fractal_dim, role_code = _calculate_fractal_metrics(
        node_idx,
        neighbors,
        node_positions,
        reference_scale
    )
    
    # Convert role code to string
    if role_code == 0:
        role = 'center'
    elif role_code == 1:
        role = 'branch'
    else:
        role = 'terminal'
        
    return self_similarity, fractal_dim, role

@njit(cache=True)
def _calculate_branching_metrics(node_idx: int,
                              neighbor_indices: np.ndarray,
                              node_positions: np.ndarray,
                              target_branching: int) -> Tuple[float, float]:
    """
    Calculate branching-related metrics
    
    Returns:
        Tuple of (branching_score, scale_transition_potential)
    """
    if len(neighbor_indices) == 0:
        return 0.0, 0.0
        
    # Calculate angular distribution of neighbors
    angles = np.zeros(len(neighbor_indices))
    for i, n in enumerate(neighbor_indices):
        diff = node_positions[n] - node_positions[node_idx]
        angles[i] = np.arctan2(diff[1], diff[0])
    
    # Sort angles and calculate angular gaps
    angles = np.sort(angles)
    gaps = np.diff(angles)
    gaps = np.append(gaps, 2*np.pi - (angles[-1] - angles[0]))
    
    # Calculate branching score based on angular distribution
    ideal_gap = 2*np.pi / target_branching
    gap_scores = 1.0 - np.abs(gaps - ideal_gap) / np.pi
    branching_score = float(np.mean(gap_scores))
    
    # Calculate scale transition potential
    if len(neighbor_indices) == target_branching:
        scale_potential = branching_score
    else:
        scale_potential = branching_score * (len(neighbor_indices) / target_branching)
        
    return float(branching_score), float(scale_potential)

def calculate_branching_metrics(node_idx: int,
                              neighborhood_data: NeighborhoodData,
                              node_positions: np.ndarray,
                              target_branching: int) -> Tuple[float, float]:
    """
    Calculate branching-related metrics
    
    Returns:
        Tuple of (branching_score, scale_transition_potential)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    return _calculate_branching_metrics(
        node_idx,
        neighbors,
        node_positions,
        target_branching
    )
    
def calculate_pattern_stability(node_idx: int,
                              neighborhood_data: NeighborhoodData,
                              pattern: Optional[FractalPattern],
                              self_similarity: float,
                              branching_score: float) -> float:
    """
    Calculate stability of fractal pattern
    
    Returns:
        Stability score (0-1)
    """
    if pattern is None:
        return 0.0
        
    # Consider multiple stability factors
    stability_factors = [
        self_similarity,                    # Local self-similarity
        branching_score,                    # Branching completeness
        pattern.similarity_score,           # Pattern-wide similarity
        1.0 - (pattern.recursion_depth / 5.0)  # Depth penalty
    ]
    
    # Calculate member state consistency
    if pattern.member_nodes:
        states = [neighborhood_data.states[n] for n in pattern.member_nodes]
        state_consistency = 1.0 - np.std(states) if states else 0.0
        stability_factors.append(float(state_consistency))
    
    return float(np.mean(stability_factors))

def analyze_fractal(node_idx: int,
                   neighborhood_data: NeighborhoodData,
                   node_positions: np.ndarray,
                   patterns: Dict[int, FractalPattern],
                   reference_scale: float,
                   target_branching: int) -> FractalData:
    """
    Comprehensive fractal analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        node_positions: Array of node positions
        patterns: Dictionary of existing fractal patterns
        reference_scale: Base scale for pattern measurement
        target_branching: Target number of branches
    
    Returns:
        FractalData containing all fractal metrics
    """
    # Calculate basic metrics
    self_similarity, fractal_dimension, pattern_role = calculate_fractal_metrics(
        node_idx, neighborhood_data, node_positions, reference_scale
    )
    
    # Calculate branching metrics
    branching_score, scale_transition_potential = calculate_branching_metrics(
        node_idx, neighborhood_data, node_positions, target_branching
    )
    
    # Find current pattern
    current_pattern = None
    for pattern in patterns.values():
        if node_idx in pattern.member_nodes:
            current_pattern = pattern
            break
    
    # Calculate scale position
    if current_pattern:
        center_pos = node_positions[current_pattern.center_node]
        node_pos = node_positions[node_idx]
        distance = np.linalg.norm(node_pos - center_pos)
        scale_position = distance / (current_pattern.scale * reference_scale)
    else:
        scale_position = 0.0
    
    # Calculate pattern stability
    pattern_stability = calculate_pattern_stability(
        node_idx, neighborhood_data, current_pattern,
        self_similarity, branching_score
    )
    
    # Find recursive patterns
    recursive_patterns = []
    if current_pattern:
        for pattern in patterns.values():
            if (pattern.recursion_depth > current_pattern.recursion_depth and
                pattern.center_node in current_pattern.member_nodes):
                recursive_patterns.append(pattern)
    
    return FractalData(
        current_pattern=current_pattern,
        pattern_role=pattern_role,
        scale_position=float(scale_position),
        self_similarity=self_similarity,
        fractal_dimension=fractal_dimension,
        branching_score=branching_score,
        scale_transition_potential=scale_transition_potential,
        pattern_stability=pattern_stability,
        recursive_patterns=recursive_patterns
    )

@dataclass
class Organism:
    """Represents a living organism in the network"""
    organism_id: int
    cells: Set[int]                # Node indices that make up the organism
    age: int                       # Age in simulation steps
    energy: float                  # Current energy level
    genome: Dict[str, float]       # Genetic parameters
    generation: int                # Generational distance from origin
    parent_id: Optional[int]       # ID of parent organism
    offspring: Set[int]            # IDs of child organisms
    phenotype: str                 # Current expressed behavior type
    fitness: float                 # Current fitness score

@dataclass
class LifeData:
    """Container for artificial life analysis results"""
    organism: Optional[Organism]    # Current organism
    cell_role: str                 # Role in organism ('core', 'membrane', 'specialized')
    energy_balance: float          # Current energy input/output balance
    metabolic_rate: float          # Rate of energy consumption
    reproduction_potential: float   # Potential for reproduction
    mutation_rate: float           # Current mutation rate
    adaptation_score: float        # Measure of environmental adaptation
    survival_score: float          # Overall survival probability
    interaction_partners: Set[int]  # Organisms interacting with this cell

@njit(cache=True)
def _calculate_metabolic_metrics(node_idx: int,
                              neighbor_indices: np.ndarray,
                              edge_matrix: np.ndarray,
                              energy_distribution: np.ndarray,
                              base_metabolism: float) -> Tuple[float, float]:
    """
    Calculate metabolism-related metrics
    
    Returns:
        Tuple of (energy_balance, metabolic_rate)
    """
    if len(neighbor_indices) == 0:
        return 0.0, base_metabolism
        
    # Calculate energy intake from environment
    energy_intake = energy_distribution[node_idx]
    
    # Calculate energy exchange with neighbors
    energy_exchange = 0.0
    for n in neighbor_indices:
        if edge_matrix[node_idx, n]:
            # Energy flows from high to low concentration
            energy_diff = energy_distribution[n] - energy_distribution[node_idx]
            energy_exchange += energy_diff * 0.1  # Exchange rate
            
    # Calculate metabolic rate based on activity
    metabolic_rate = base_metabolism * (1.0 + 0.1 * len(neighbor_indices))
    
    # Calculate energy balance
    energy_balance = energy_intake + energy_exchange - metabolic_rate
    
    return float(energy_balance), float(metabolic_rate)

def calculate_metabolic_metrics(node_idx: int,
                              neighborhood_data: NeighborhoodData,
                              energy_distribution: np.ndarray,
                              base_metabolism: float) -> Tuple[float, float]:
    """
    Calculate metabolism-related metrics
    
    Returns:
        Tuple of (energy_balance, metabolic_rate)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    return _calculate_metabolic_metrics(
        node_idx,
        neighbors,
        neighborhood_data.edge_matrix,
        energy_distribution,
        base_metabolism
    )
    
@njit(cache=True)
def _calculate_reproduction_metrics(node_idx: int,
                                 neighbor_indices: np.ndarray,
                                 edge_matrix: np.ndarray,
                                 organism_assignments: np.ndarray,
                                 energy_levels: np.ndarray,
                                 reproduction_threshold: float,
                                 current_organism: int) -> Tuple[float, float]:
    """
    Calculate reproduction-related metrics
    
    Returns:
        Tuple of (reproduction_potential, mutation_rate)
    """
    if len(neighbor_indices) == 0:
        return 0.0, 0.0
        
    # Calculate available space for reproduction
    empty_neighbors = np.sum(organism_assignments[neighbor_indices] == -1)
    space_factor = empty_neighbors / len(neighbor_indices)
    
    # Calculate energy-based reproduction potential
    energy_level = energy_levels[current_organism]
    energy_factor = max(0.0, min(1.0, energy_level / reproduction_threshold))
    
    # Calculate base mutation rate inversely proportional to fitness
    base_mutation_rate = 0.1
    mutation_rate = base_mutation_rate * (1.0 - energy_factor)
    
    # Calculate final reproduction potential
    reproduction_potential = space_factor * energy_factor
    
    return float(reproduction_potential), float(mutation_rate)

def calculate_reproduction_metrics(node_idx: int,
                                 neighborhood_data: NeighborhoodData,
                                 organism_assignments: Dict[int, int],
                                 energy_levels: Dict[int, float],
                                 reproduction_threshold: float) -> Tuple[float, float]:
    """
    Calculate reproduction-related metrics
    
    Returns:
        Tuple of (reproduction_potential, mutation_rate)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0
        
    current_organism = organism_assignments.get(node_idx, -1)
    if current_organism == -1:
        return 0.0, 0.0
        
    # Convert organism assignments and energy levels to numpy arrays
    organism_assignments_array = np.array([organism_assignments.get(n, -1) for n in range(len(neighborhood_data.states))], dtype=np.int64)
    energy_levels_array = np.array(list(energy_levels.values()), dtype=np.float64)
    
    reproduction_potential, mutation_rate = _calculate_reproduction_metrics(
        node_idx,
        neighbors,
        neighborhood_data.edge_matrix,
        organism_assignments_array,
        energy_levels_array,
        reproduction_threshold,
        current_organism
    )
    
    return reproduction_potential, mutation_rate

def calculate_adaptation_score(node_idx: int,
                             neighborhood_data: NeighborhoodData,
                             organism: Optional[Organism],
                             environment_factors: Dict[str, float]) -> float:
    """
    Calculate environmental adaptation score
    
    Returns:
        Adaptation score (0-1)
    """
    if organism is None:
        return 0.0
        
    # Consider multiple adaptation factors
    adaptation_factors = []
    
    # Genome-environment match
    for factor, value in environment_factors.items():
        if factor in organism.genome:
            match_score = 1.0 - abs(organism.genome[factor] - value)
            adaptation_factors.append(match_score)
    
    # Age-based adaptation
    age_factor = min(1.0, organism.age / 100.0)  # Normalize age
    adaptation_factors.append(age_factor)
    
    # Size-based adaptation
    size_factor = min(1.0, len(organism.cells) / 10.0)  # Normalize size
    adaptation_factors.append(size_factor)
    
    # Generation-based adaptation
    generation_factor = min(1.0, organism.generation / 10.0)  # Normalize generation
    adaptation_factors.append(generation_factor)
    
    return float(np.mean(adaptation_factors) if adaptation_factors else 0.0)

def analyze_life(node_idx: int,
                neighborhood_data: NeighborhoodData,
                organisms: Dict[int, Organism],
                organism_assignments: Dict[int, int],
                energy_distribution: np.ndarray,
                environment_factors: Dict[str, float],
                base_metabolism: float,
                reproduction_threshold: float) -> LifeData:
    """
    Comprehensive artificial life analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        organisms: Dictionary of living organisms
        organism_assignments: Dictionary of node organism assignments
        energy_distribution: Array of environmental energy values
        environment_factors: Dictionary of environmental factors
        base_metabolism: Base metabolic rate
        reproduction_threshold: Energy threshold for reproduction
    
    Returns:
        LifeData containing all life metrics
    """
    # Get current organism
    organism_id = organism_assignments.get(node_idx, -1)
    current_organism = organisms.get(organism_id, None)
    
    # Calculate metabolic metrics
    energy_balance, metabolic_rate = calculate_metabolic_metrics(
        node_idx, neighborhood_data, energy_distribution, base_metabolism
    )
    
    # Calculate reproduction metrics
    reproduction_potential, mutation_rate = calculate_reproduction_metrics(
        node_idx, neighborhood_data, organism_assignments,
        {org.organism_id: org.energy for org in organisms.values()},
        reproduction_threshold
    )
    
    # Calculate adaptation score
    adaptation_score = calculate_adaptation_score(
        node_idx, neighborhood_data, current_organism, environment_factors
    )
    
    # Determine cell role
    if current_organism:
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        foreign_neighbors = sum(1 for n in neighbors 
                              if organism_assignments.get(n, -1) != organism_id)
        if foreign_neighbors == 0:
            cell_role = 'core'
        elif foreign_neighbors > len(neighbors) / 2:
            cell_role = 'membrane'
        else:
            cell_role = 'specialized'
    else:
        cell_role = 'none'
    
    # Calculate survival score
    survival_factors = [
        max(0.0, energy_balance),
        1.0 - metabolic_rate / (base_metabolism * 2),
        adaptation_score
    ]
    survival_score = float(np.mean(survival_factors))
    
    # Identify interaction partners
    interaction_partners = set()
    if current_organism:
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        for n in neighbors:
            n_organism = organism_assignments.get(n, -1)
            if n_organism != -1 and n_organism != organism_id:
                interaction_partners.add(n_organism)
    
    return LifeData(
        organism=current_organism,
        cell_role=cell_role,
        energy_balance=energy_balance,
        metabolic_rate=metabolic_rate,
        reproduction_potential=reproduction_potential,
        mutation_rate=mutation_rate,
        adaptation_score=adaptation_score,
        survival_score=survival_score,
        interaction_partners=interaction_partners
    )

class Assortativity(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Measures the assortativity coefficient of the local network"""
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) < 2:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the original edge matrix
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if edge_matrix[subgraph_nodes[i], subgraph_nodes[j]]:
                    G.add_edge(i, j)
        
        # Calculate assortativity
        try:
            return float(nx.degree_assortativity_coefficient(G))
        except Exception:
            return 0.0  # Handle disconnected graphs

class BetweennessCentrality(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Measures the betweenness centrality of a node in the local network"""
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the original edge matrix
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if edge_matrix[subgraph_nodes[i], subgraph_nodes[j]]:
                    G.add_edge(i, j)
        
        # Calculate betweenness centrality
        try:
            centrality = nx.betweenness_centrality(G)
            return float(centrality[0])
        except Exception:
            return 0.0  # Handle disconnected graphs

class ClosenessCentrality(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Measures the closeness centrality of a node in the local network"""
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the original edge matrix
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if edge_matrix[subgraph_nodes[i], subgraph_nodes[j]]:
                    G.add_edge(i, j)
        
        # Calculate closeness centrality
        try:
            centrality = nx.closeness_centrality(G, u=0)
            return float(centrality[0])
        except Exception:
            return 0.0  # Handle disconnected graphs

class EigenvectorCentrality(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Measures the eigenvector centrality of a node in the local network"""
        
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the original edge matrix
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if edge_matrix[subgraph_nodes[i], subgraph_nodes[j]]:
                    G.add_edge(i, j)
        
        # Calculate eigenvector centrality
        try:
            centrality = nx.eigenvector_centrality(G)[0]
            return float(centrality)
        except Exception:
            return 0.0  # Handle disconnected graphs

class GraphLaplacianEnergy(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                edge_matrix: npt.NDArray[np.bool_]) -> float:
        """Measures the graph laplacian energy of the local network"""
        
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the original edge matrix
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if edge_matrix[subgraph_nodes[i], subgraph_nodes[j]]:
                    G.add_edge(i, j)
        
        # Calculate graph laplacian energy
        try:
            laplacian = nx.laplacian_matrix(G).asfptype()
            eigenvalues = np.linalg.eigvalsh(laplacian.toarray())
            energy = np.sum(np.abs(eigenvalues))
            return float(energy)
        except Exception:
            return 0.0  # Handle disconnected graphs
        
################################################
#               RULE DEFINITIONS               #
################################################

class TestRule(Rule):
    """
    Test Rule: Creates balanced patterns through state transitions and edge formation.
    Demonstrates core cellular automata principles with configurable parameters.
    """

    # Parameter metadata and validation
    PARAMETER_METADATA = {
        "min_active_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required for survival",
            "min": 0,
            "max": 8
        },
        "max_active_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed for survival",
            "min": 0,
            "max": 8
        },
        "min_connected_active": {
            "type": int,
            "description": "Minimum number of connected active neighbors required for survival",
            "min": 0,
            "max": 8
        },
        "max_connected_active": {
            "type": int,
            "description": "Maximum number of connected active neighbors allowed for survival",
            "min": 0,
            "max": 8
        },
        "birth_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required for birth",
            "min": 0,
            "max": 8
        },
        "birth_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed for birth",
            "min": 0,
            "max": 8
        },
        "birth_min_connected": {
            "type": int,
            "description": "Minimum number of connected neighbors required for birth",
            "min": 0,
            "max": 8
        },
        "birth_max_connected": {
            "type": int,
            "description": "Maximum number of connected neighbors allowed for birth",
            "min": 0,
            "max": 8
        },
        "use_edge_density": {
            "type": bool,
            "description": "Enable edge density-based state changes"
        },
        "death_edge_density": {
            "type": float,
            "description": "Edge density threshold that triggers death",
            "min": 0.0,
            "max": 1.0
        },
        "birth_edge_density": {
            "type": float,
            "description": "Maximum edge density allowed for birth",
            "min": 0.0,
            "max": 1.0
        },
        "use_entropy": {
            "type": bool,
            "description": "Enable entropy-based state changes"
        },
        "death_entropy_threshold": {
            "type": float,
            "description": "Minimum entropy required to prevent death",
            "min": 0.0,
            "max": 1.0
        },
        "birth_entropy_threshold": {
            "type": float,
            "description": "Minimum entropy required for birth",
            "min": 0.0,
            "max": 1.0
        },
        "use_random_death": {
            "type": bool,
            "description": "Enable random death chance"
        },
        "death_probability": {
            "type": float,
            "description": "Probability of random death when enabled",
            "min": 0.0,
            "max": 1.0
        },
        "edge_formation_rate": {
            "type": float,
            "description": "Probability of forming a new edge",
            "min": 0.0,
            "max": 1.0
        },
        "edge_removal_rate": {
            "type": float,
            "description": "Probability of removing an existing edge",
            "min": 0.0,
            "max": 1.0
        },
        "min_shared_neighbors": {
            "type": int,
            "description": "Minimum shared neighbors required to maintain an edge",
            "min": 0,
            "max": 8
        },
        "max_edges_per_node": {
            "type": int,
            "description": "Maximum number of edges a node can have",
            "min": 0,
            "max": 8
        },
        "min_edges_per_node": {
            "type": int,
            "description": "Minimum number of edges a node must maintain",
            "min": 0,
            "max": 8
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": "RANDOM",
        "allowed_values": [
            "HIGHER_STATE",
            "LOWER_STATE",
            "MORE_CONNECTIONS",
            "FEWER_CONNECTIONS",
            "HIGHER_STATE_MORE_NEIGHBORS",
            "LOWER_STATE_FEWER_NEIGHBORS",
            "HIGHER_STATE_FEWER_NEIGHBORS",
            "LOWER_STATE_MORE_NEIGHBORS",
            "RANDOM",
            "AGREEMENT"
        ]
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData, 
                        node_positions: Optional[np.ndarray] = None, 
                        dimension_type: Optional[Dimension] = None) -> float:
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0:
            return -1.0

        active_neighbors = np.sum(neighborhood_data.states[neighbors] > 0)
        connected_active = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors] & 
                                (neighborhood_data.states[neighbors] > 0))

        if self.get_param('use_edge_density', False):
            edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)
        else:
            edge_density = 0.0

        if self.get_param('use_entropy', False):
            entropy = self.get_metric(StateEntropy, node_idx, neighborhood_data)
        else:
            entropy = 1.0

        if current_state > 0:
            min_active = self.get_param('min_active_neighbors')
            max_active = self.get_param('max_active_neighbors')
            min_connected = self.get_param('min_connected_active')
            max_connected = self.get_param('max_connected_active')
            
            if None in (min_active, max_active, min_connected, max_connected):
                return 0.0

            if edge_density > self.get_param('death_edge_density', 1.0):
                return 0.0

            if entropy < self.get_param('death_entropy_threshold', 0.0):
                return 0.0

            survives = (min_active <= active_neighbors <= max_active and
                    min_connected <= connected_active <= max_connected)

            if active_neighbors == min_active or active_neighbors == max_active:
                survives = TieBreaker.resolve(
                    current_state,
                    neighborhood_data.states[neighbors[0]],
                    connected_active,
                    np.sum(neighborhood_data.edge_matrix[neighbors[0]]),
                    TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
                )

            if (survives and 
                self.get_param('use_random_death', False) and
                random.random() < self.get_param('death_probability', 0.0)):
                return 0.0

            return 1.0 if survives else 0.0

        else:
            birth_min = self.get_param('birth_min_neighbors')
            birth_max = self.get_param('birth_max_neighbors')
            birth_min_connected = self.get_param('birth_min_connected')
            birth_max_connected = self.get_param('birth_max_connected')

            if None in (birth_min, birth_max, birth_min_connected, birth_max_connected):
                return 0.0

            if edge_density > self.get_param('birth_edge_density', 1.0):
                return 0.0

            if entropy < self.get_param('birth_entropy_threshold', 0.0):
                return 0.0

            born = (birth_min <= active_neighbors <= birth_max and
                birth_min_connected <= connected_active <= birth_max_connected)

            if active_neighbors == birth_min or active_neighbors == birth_max:
                born = TieBreaker.resolve(
                    current_state,
                    neighborhood_data.states[neighbors[0]],
                    connected_active,
                    np.sum(neighborhood_data.edge_matrix[neighbors[0]]),
                    TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
                )

            return 1.0 if born else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                        node_positions: Optional[np.ndarray] = None,
                        dimension_type: Optional[Dimension] = None) -> Set[int]:
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        edge_formation_rate = self.get_param('edge_formation_rate')
        edge_removal_rate = self.get_param('edge_removal_rate')
        min_shared = self.get_param('min_shared_neighbors')
        max_edges = self.get_param('max_edges_per_node')
        min_edges = self.get_param('min_edges_per_node')

        if None in (edge_formation_rate, edge_removal_rate, min_shared, max_edges, min_edges):
            return new_edges

        current_edges = {n for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}

        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            shared = np.sum(neighborhood_data.edge_matrix[node_idx] & 
                        neighborhood_data.edge_matrix[n])

            if n in current_edges:
                if shared >= min_shared:
                    new_edges.add(n)
                elif random.random() >= edge_removal_rate:
                    new_edges.add(n)
            else:
                if (len(new_edges) < max_edges and
                    shared >= min_shared and
                    random.random() < edge_formation_rate):
                    new_edges.add(n)

        if len(new_edges) < min_edges:
            available = [n for n in neighbors 
                        if neighborhood_data.states[n] > 0 and n not in new_edges]
            while len(new_edges) < min_edges and available:
                n = available.pop(random.randrange(len(available)))
                new_edges.add(n)

        return new_edges

class PureRuleTableRule(Rule):
    """
    PureRuleTable: A rule driven entirely by comprehensive rule tables for both state
    transitions and edge updates. Uses only local neighborhood state patterns and
    connection patterns without any additional metrics. Includes tiebreaker logic
    for equivalent patterns.
    """

    PARAMETER_METADATA = {
        "state_rule_table": {
            "type": dict,
            "description": "Rule table for state transitions. Keys are (current_state, neighbor_pattern, connection_pattern)",
            "default": {}
        },
        "edge_rule_table": {
            "type": dict,
            "description": "Rule table for edge updates. Keys are (self_state, neighbor_state, connection_pattern)",
            "default": {}
        },
        "state_memory": {
            "type": int,
            "description": "Number of steps to remember previous states",
            "min": 0,
            "max": 10
        },
        "edge_memory": {
            "type": int,
            "description": "Number of steps to remember previous edges",
            "min": 0,
            "max": 10
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": "RANDOM",
        "allowed_values": [
            "HIGHER_STATE",
            "LOWER_STATE",
            "MORE_CONNECTIONS",
            "FEWER_CONNECTIONS",
            "HIGHER_STATE_MORE_NEIGHBORS",
            "LOWER_STATE_FEWER_NEIGHBORS",
            "HIGHER_STATE_FEWER_NEIGHBORS",
            "LOWER_STATE_MORE_NEIGHBORS",
            "RANDOM",
            "AGREEMENT"
        ]
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self.state_history: List[Dict[int, float]] = []
        self.edge_history: List[Set[Tuple[int, int]]] = []

    def _validate_parameter(self, name: str, value: Any) -> bool:
        """Override parameter validation to handle rule tables"""
        if name == 'state_rule_table':
            return self._validate_state_rule_table(value)
        elif name == 'edge_rule_table':
            return self._validate_edge_rule_table(value)
        
        return super()._validate_parameter(name, value)

    def _validate_state_rule_table(self, table: Dict[str, int]) -> bool:
        """Validate state rule table format and values"""
        try:
            if "default" not in table:
                return False
                
            for key, value in table.items():
                if key == "default":
                    if value not in [-1, 0, 1]:
                        return False
                    continue
                    
                try:
                    # Parse key format: (current_state, neighbor_pattern, connection_pattern)
                    parts = key.strip("()").split(",")
                    if len(parts) != 3:
                        return False
                        
                    current_state = int(parts[0])
                    neighbor_pattern = parts[1].strip()
                    connection_pattern = parts[2].strip()
                    
                    # Validate current state
                    if current_state not in [-1, 0, 1]:
                        return False
                        
                    # Validate neighbor pattern (exactly 8 bits)
                    if len(neighbor_pattern) != 8 or not all(c in '01' for c in neighbor_pattern):
                        return False
                        
                    # Validate connection pattern (exactly 8 bits)
                    if len(connection_pattern) != 8 or not all(c in '01' for c in connection_pattern):
                        return False
                        
                    # Validate new state
                    if value not in [-1, 0, 1]:
                        return False
                        
                except ValueError:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"State rule table validation error: {e}")
            return False

    def _validate_edge_rule_table(self, table: Dict[str, str]) -> bool:
        """Validate edge rule table format and values"""
        try:
            if "default" not in table:
                return False
                
            valid_actions = {"add", "remove", "maintain"}
            
            for key, value in table.items():
                if key == "default":
                    if value not in valid_actions:
                        return False
                    continue
                    
                try:
                    # Parse key format: (self_state, neighbor_state, connection_pattern)
                    parts = key.strip("()").split(",")
                    if len(parts) != 3:
                        return False
                        
                    self_state = int(parts[0])
                    neighbor_state = int(parts[1])
                    connection_pattern = parts[2].strip()
                    
                    # Validate states
                    if self_state not in [0, 1] or neighbor_state not in [0, 1]:
                        return False
                        
                    # Validate connection pattern (exactly 8 bits)
                    if len(connection_pattern) != 8 or not all(c in '01' for c in connection_pattern):
                        return False
                        
                    # Validate action
                    if value not in valid_actions:
                        return False
                        
                except ValueError:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Edge rule table validation error: {e}")
            return False
        
    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based purely on rule table patterns"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0:
            return -1.0

        # Get rule table
        state_rule_table = self.get_param('state_rule_table')
        if state_rule_table is None:
            return 0.0

        # Create neighbor pattern string
        neighbor_pattern = ''.join(['1' if neighborhood_data.states[n] > 0 else '0' 
                                  for n in neighbors])
        
        # Create connection pattern string
        connection_pattern = ''.join(['1' if neighborhood_data.edge_matrix[node_idx, n] else '0'
                                    for n in neighbors])

        # Create lookup key
        key = f"({int(current_state)}, {neighbor_pattern}, {connection_pattern})"
        
        # Get new state from rule table
        new_state = state_rule_table.get(key, state_rule_table.get('default', 0))

        # Handle ties using tiebreaker
        if new_state == current_state:
            new_state = int(TieBreaker.resolve(
                current_state,
                neighborhood_data.states[neighbors[0]],
                np.sum(neighborhood_data.edge_matrix[node_idx]),
                np.sum(neighborhood_data.edge_matrix[neighbors[0]]),
                TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
            ))

        # Update state history
        state_memory = self.get_param('state_memory', 0)
        if state_memory > 0:
            self.state_history.append({node_idx: new_state})
            if len(self.state_history) > state_memory:
                self.state_history.pop(0)

        return float(new_state)

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based purely on rule table patterns"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get rule table
        edge_rule_table = self.get_param('edge_rule_table')
        if edge_rule_table is None:
            return new_edges

        # Get current connection pattern
        connection_pattern = ''.join(['1' if neighborhood_data.edge_matrix[node_idx, n] else '0'
                                    for n in neighbors])

        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            # Create lookup key
            key = f"({int(neighborhood_data.states[node_idx] > 0)}, {int(neighborhood_data.states[n] > 0)}, {connection_pattern})"
            
            # Get action from rule table
            action = edge_rule_table.get(key, edge_rule_table.get('default', 'maintain'))

            # Apply action
            if action == 'add':
                new_edges.add(n)
            elif action == 'maintain' and neighborhood_data.edge_matrix[node_idx, n]:
                new_edges.add(n)

        # Update edge history
        edge_memory = self.get_param('edge_memory', 0)
        if edge_memory > 0:
            self.edge_history.append(new_edges.copy())
            if len(self.edge_history) > edge_memory:
                self.edge_history.pop(0)

        return new_edges
    
class MajorityRule(Rule):
    """
    Highly Parameterized Majority Rule with extensive control over all aspects of the simulation.
    """

    PARAMETER_METADATA = {
        "activation_threshold": {
            "type": float,
            "description": "Ratio of active neighbors required for activation",
            "min": 0.0,
            "max": 1.0
        },
        "deactivation_threshold": {
            "type": float,
            "description": "Ratio of active neighbors below which a node deactivates",
            "min": 0.0,
            "max": 1.0
        },
        "rebellion_chance": {
            "type": float,
            "description": "Base probability of a node rebelling against the majority",
            "min": 0.0,
            "max": 1.0
        },
        "rebellion_activation_bonus": {
            "type": float,
            "description": "Bonus to activation probability when rebelling",
            "min": 0.0,
            "max": 1.0
        },
        "rebellion_deactivation_penalty": {
            "type": float,
            "description": "Penalty to deactivation probability when rebelling",
            "min": 0.0,
            "max": 1.0
        },
        "random_birth_rate": {
            "type": float,
            "description": "Probability of a new node being born randomly",
            "min": 0.0,
            "max": 1.0
        },
        "random_death_rate": {
            "type": float,
            "description": "Probability of a node dying randomly",
            "min": 0.0,
            "max": 1.0
        },
        "edge_formation_rate": {
            "type": float,
            "description": "Probability of forming new edges",
            "min": 0.0,
            "max": 1.0
        },
        "edge_removal_rate": {
            "type": float,
            "description": "Probability of removing existing edges",
            "min": 0.0,
            "max": 1.0
        },
        "min_shared_neighbors": {
            "type": int,
            "description": "Minimum shared neighbors required to maintain an edge",
            "min": 0,
            "max": 8
        },
        "max_connections": {
            "type": int,
            "description": "Maximum number of connections a node can have",
            "min": 0,
            "max": 8
        },
        "min_connections": {
            "type": int,
            "description": "Minimum number of connections a node must have",
            "min": 0,
            "max": 8
        },
        "birth_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required for birth",
            "min": 0,
            "max": 8
        },
        "birth_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed for birth",
            "min": 0,
            "max": 8
        },
        "death_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required for survival",
            "min": 0,
            "max": 8
        },
        "death_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed for survival",
            "min": 0,
            "max": 8
        },
        "connect_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required to form a connection",
            "min": 0,
            "max": 8
        },
        "connect_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed to form a connection",
            "min": 0,
            "max": 8
        },
        "disconnect_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required to maintain a connection",
            "min": 0,
            "max": 8
        },
        "disconnect_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed to maintain a connection",
            "min": 0,
            "max": 8
        },
        "rebellion_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required to trigger a rebellion",
            "min": 0,
            "max": 8
        },
        "rebellion_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed to trigger a rebellion",
            "min": 0,
            "max": 8
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT"
            ]
        },
        "edge_formation_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required to form an edge",
            "min": 0,
            "max": 8
        },
        "edge_formation_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed to form an edge",
            "min": 0,
            "max": 8
        },
        "edge_removal_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required to remove an edge",
            "min": 0,
            "max": 8
        },
        "edge_removal_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed to remove an edge",
            "min": 0,
            "max": 8
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on majority/minority relationships"""
        try:
            # Get current state
            current_state = neighborhood_data.states[node_idx]
            neighbors = neighborhood_data.get_neighbor_indices(node_idx)
            
            # Handle empty cell case
            if current_state == -1.0:
                if random.random() < self.get_param('random_birth_rate', 0.01):
                    return 1.0
                else:
                    return -1.0

            # Calculate metrics
            if len(neighbors) == 0:
                return 0.0
                
            active_neighbors = np.sum(neighborhood_data.states[neighbors] > 0)
            active_ratio = active_neighbors / len(neighbors) if len(neighbors) > 0 else 0.0

            # Get parameters with defaults
            activation_threshold = self.get_param('activation_threshold', 0.6)
            deactivation_threshold = self.get_param('deactivation_threshold', 0.4)
            rebellion_chance = self.get_param('rebellion_chance', 0.0)
            rebellion_activation_bonus = self.get_param('rebellion_activation_bonus', 0.1)
            rebellion_deactivation_penalty = self.get_param('rebellion_deactivation_penalty', 0.1)
            random_death_rate = self.get_param('random_death_rate', 0.01)
            birth_min_neighbors = self.get_param('birth_min_neighbors', 0)
            birth_max_neighbors = self.get_param('birth_max_neighbors', 8)
            death_min_neighbors = self.get_param('death_min_neighbors', 0)
            death_max_neighbors = self.get_param('death_max_neighbors', 8)
            rebellion_min_neighbors = self.get_param('rebellion_min_neighbors', 0)
            rebellion_max_neighbors = self.get_param('rebellion_max_neighbors', 8)

            # State transition logic
            if current_state > 0:  # Active cell
                # Simplified survival check
                survives = active_ratio >= deactivation_threshold and death_min_neighbors <= active_neighbors <= death_max_neighbors
                
                # Apply rebellion chance
                if rebellion_min_neighbors <= active_neighbors <= rebellion_max_neighbors and random.random() < rebellion_chance:
                    survives = not survives
                    if survives:
                        survives = survives and random.random() > rebellion_deactivation_penalty
                    else:
                        survives = survives or random.random() < rebellion_activation_bonus
                
                if random.random() < random_death_rate:
                    survives = False
                
                return 1.0 if survives else 0.0
            else:  # Inactive cell
                # Simplified birth check
                born = active_ratio >= activation_threshold and birth_min_neighbors <= active_neighbors <= birth_max_neighbors
                
                # Apply rebellion chance
                if rebellion_min_neighbors <= active_neighbors <= rebellion_max_neighbors and random.random() < rebellion_chance:
                    born = not born
                    if born:
                        born = born and random.random() > rebellion_deactivation_penalty
                    else:
                        born = born or random.random() < rebellion_activation_bonus
                    
                return 1.0 if born else 0.0

        except Exception as e:
            logger.error(f"Error in MajorityRule compute_state_update for node {node_idx}: {e}")
            logger.error(traceback.format_exc())
            return 0.0  # Default to inactive on error
                
    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on majority relationships and state matching"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get edge parameters
        edge_formation_rate = self.get_param('edge_formation_rate', 0.5)
        edge_removal_rate = self.get_param('edge_removal_rate', 0.1)
        min_shared_neighbors = self.get_param('min_shared_neighbors', 1)
        max_connections = self.get_param('max_connections', 8)
        min_connections = self.get_param('min_connections', 2)
        connect_min_neighbors = self.get_param('connect_min_neighbors', 0)
        connect_max_neighbors = self.get_param('connect_max_neighbors', 8)
        disconnect_min_neighbors = self.get_param('disconnect_min_neighbors', 0)
        disconnect_max_neighbors = self.get_param('disconnect_max_neighbors', 8)
        edge_formation_min_neighbors = self.get_param('edge_formation_min_neighbors', 0)
        edge_formation_max_neighbors = self.get_param('edge_formation_max_neighbors', 8)
        edge_removal_min_neighbors = self.get_param('edge_removal_min_neighbors', 0)
        edge_removal_max_neighbors = self.get_param('edge_removal_max_neighbors', 8)

        current_edges = {n for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}

        for n in neighbors:
            if neighborhood_data.states[n] > 0:
                # Check connection conditions
                if connect_min_neighbors <= len(neighbors) <= connect_max_neighbors:
                    if len(new_edges) < max_connections:
                        if random.random() < edge_formation_rate:
                            new_edges.add(n)
                # Check disconnection conditions
                if n in current_edges and disconnect_min_neighbors <= len(neighbors) <= disconnect_max_neighbors:
                    if random.random() < edge_removal_rate:
                        new_edges.discard(n)

        # Enforce maximum connections
        if len(new_edges) > max_connections:
            new_edges = set(random.sample(new_edges, max_connections))

        # Enforce minimum connections
        while len(new_edges) < min_connections and len(new_edges) < len(neighbors):
            remaining_neighbors = set(neighbors) - new_edges
            if remaining_neighbors:
                new_neighbor = random.choice(list(remaining_neighbors))
                new_edges.add(new_neighbor)

        return new_edges

class ConnectLife(Rule):
    """A network-based variant of Conway's Game of Life that uses rule tables for both state transitions and edge updates."""

    PARAMETER_METADATA = {
            "state_rule_table": {
                "type": dict,
                "description": "Rule table for state transitions. Keys are (current_state, neighbor_pattern, connection_pattern)",
                "default": {}
            },
            "edge_rule_table": {
                "type": dict,
                "description": "Rule table for edge updates. Keys are (self_state, neighbor_state, connection_pattern)",
                "default": {}
            },
            "min_connections": {
                "type": int,
                "description": "Minimum number of connections required for a cell to be considered active.",
                "min": 0,
                "max": 8
            },
            "max_connections": {
                "type": int,
                "description": "Maximum number of connections allowed for a cell to prevent overcrowding.",
                "min": 0,
                "max": 8
            },
            "min_shared_neighbors": {
                "type": int,
                "description": "Minimum number of shared neighbors required to maintain an edge.",
                "min": 0,
                "max": 8
            },
            "use_clustering": {
                "type": bool,
                "description": "Enable clustering coefficient-based state changes"
            },
            "min_clustering": {
                "type": float,
                "description": "Minimum clustering coefficient required for survival",
                "min": 0.0,
                "max": 1.0
            },
            "max_clustering": {
                "type": float,
                "description": "Maximum clustering coefficient allowed before death",
                "min": 0.0,
                "max": 1.0
            },
            "use_entropy": {
                "type": bool,
                "description": "Enable entropy-based state modifications"
            },
            "min_entropy": {
                "type": float,
                "description": "Minimum entropy required for state changes",
                "min": 0.0,
                "max": 1.0
            },
            "max_entropy": {
                "type": float,
                "description": "Maximum entropy allowed before state changes",
                "min": 0.0,
                "max": 1.0
            },
            "edge_memory": {
                "type": int,
                "description": "Number of steps to remember previous edges",
                "min": 0,
                "max": 10
            },
            "state_memory": {
                "type": int,
                "description": "Number of steps to remember previous states",
                "min": 0,
                "max": 10
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT"
                ]
            }
        }

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on rule table and metrics"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0:
            return -1.0

        # Get rule table and parameters
        state_rule_table = self.get_param('state_rule_table')
        min_connections = self.get_param('min_connections')
        max_connections = self.get_param('max_connections')
        use_clustering = self.get_param('use_clustering')
        use_entropy = self.get_param('use_entropy')

        # Get connection count
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])
        
        # Check connection bounds
        if not (min_connections <= connected_neighbors <= max_connections):
            return 0.0

        # Create neighbor pattern string
        neighbor_pattern = ''.join(['1' if neighborhood_data.states[n] > 0 else '0' 
                                  for n in neighbors])
        
        # Create connection pattern string
        connection_pattern = ''.join(['1' if neighborhood_data.edge_matrix[node_idx, n] else '0'
                                    for n in neighbors])

        # Create lookup key
        key = f"({int(current_state)}, {neighbor_pattern}, {connection_pattern})"
        
        # Get base state from rule table
        new_state = state_rule_table.get(key, state_rule_table.get('default', 0))

        # Apply metric modifiers if enabled
        survives = True
        
        if use_clustering:
            clustering = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)
            min_clustering = self.get_param('min_clustering')
            max_clustering = self.get_param('max_clustering')
            if not (min_clustering <= clustering <= max_clustering):
                survives = False

        if use_entropy and survives:
            entropy = self.get_metric(StateEntropy, node_idx, neighborhood_data)
            min_entropy = self.get_param('min_entropy')
            max_entropy = self.get_param('max_entropy')
            if not (min_entropy <= entropy <= max_entropy):
                survives = False

        # Handle ties using tiebreaker
        if new_state == current_state:
            new_state = int(TieBreaker.resolve(
                current_state,
                neighborhood_data.states[neighbors[0]] if len(neighbors) > 0 else 0,
                connected_neighbors,
                np.sum(neighborhood_data.edge_matrix[neighbors[0]]) if len(neighbors) > 0 else 0,
                TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
            ))

        # Update state history
        state_memory = self.get_param('state_memory')
        if state_memory > 0:
            if not hasattr(self, 'state_history'):
                self.state_history = []
            self.state_history.append({node_idx: new_state})
            if len(self.state_history) > state_memory:
                self.state_history.pop(0)

        return float(new_state) if survives else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on rule table and metrics"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get parameters
        edge_rule_table = self.get_param('edge_rule_table')
        min_connections = self.get_param('min_connections')
        max_connections = self.get_param('max_connections')
        min_shared_neighbors = self.get_param('min_shared_neighbors')

        # Get current connection pattern
        connection_pattern = ''.join(['1' if neighborhood_data.edge_matrix[node_idx, n] else '0'
                                    for n in neighbors])

        current_connections = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            # Calculate shared neighbors
            shared = np.sum(neighborhood_data.edge_matrix[node_idx] & 
                          neighborhood_data.edge_matrix[n])

            # Create lookup key
            key = f"({int(neighborhood_data.states[node_idx] > 0)}, {int(neighborhood_data.states[n] > 0)}, {connection_pattern})"
            
            # Get action from rule table
            action = edge_rule_table.get(key, edge_rule_table.get('default', 'maintain'))

            # Apply action with connection bounds
            should_connect = False
            
            if action == 'add':
                should_connect = True
            elif action == 'maintain' and neighborhood_data.edge_matrix[node_idx, n]:
                should_connect = True
            elif action == 'remove':
                should_connect = False

            # Check connection bounds
            if should_connect:
                if current_connections >= max_connections:
                    should_connect = False
                elif shared < min_shared_neighbors:
                    should_connect = False

            if should_connect:
                new_edges.add(n)

        # Ensure minimum connections if possible
        if len(new_edges) < min_connections:
            available = [n for n in neighbors 
                        if neighborhood_data.states[n] > 0 and n not in new_edges]
            while len(new_edges) < min_connections and available:
                n = available.pop(random.randrange(len(available)))
                new_edges.add(n)

        return new_edges
       
class LifeAndDeath(Rule):
    """
    Life and Death: Creates complex patterns through detailed birth/death dynamics.
    Features balanced thresholds for survival and reproduction, with multiple
    factors influencing state transitions and edge formation.
    """

    PARAMETER_METADATA = {
            "survival_min_active": {
                "type": int,
                "description": "Minimum active neighbors required for a cell to survive.",
                "min": 0,
                "max": 8
            },
            "survival_max_active": {
                "type": int,
                "description": "Maximum active neighbors allowed for a cell to survive.",
                "min": 0,
                "max": 8
            },
            "survival_min_connected": {
                "type": int,
                "description": "Minimum number of connected neighbors required for a cell to survive.",
                "min": 0,
                "max": 8
            },
            "survival_max_connected": {
                "type": int,
                "description": "Maximum number of connected neighbors allowed for a cell to survive.",
                "min": 0,
                "max": 8
            },
            "birth_min_active": {
                "type": int,
                "description": "Minimum number of active neighbors required for a new cell to be born.",
                "min": 0,
                "max": 8
            },
            "birth_max_active": {
                "type": int,
                "description": "Maximum number of active neighbors allowed for a new cell to be born.",
                "min": 0,
                "max": 8
            },
            "birth_min_connected": {
                "type": int,
                "description": "Minimum number of connected neighbors required for a new cell to be born.",
                "min": 0,
                "max": 8
            },
            "birth_max_connected": {
                "type": int,
                "description": "Maximum number of connected neighbors allowed for a new cell to be born.",
                "min": 0,
                "max": 8
            },
            "use_edge_density": {
                "type": bool,
                "description": "Enable edge density-based state changes"
            },
            "survival_min_density": {
                "type": float,
                "description": "Minimum edge density required for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "survival_max_density": {
                "type": float,
                "description": "Maximum edge density allowed for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "birth_min_density": {
                "type": float,
                "description": "Minimum edge density required for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "birth_max_density": {
                "type": float,
                "description": "Maximum edge density allowed for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "use_clustering": {
                "type": bool,
                "description": "Enable clustering coefficient-based state changes"
            },
            "survival_min_clustering": {
                "type": float,
                "description": "Minimum clustering coefficient required for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "survival_max_clustering": {
                "type": float,
                "description": "Maximum clustering coefficient allowed for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "birth_min_clustering": {
                "type": float,
                "description": "Minimum clustering coefficient required for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "birth_max_clustering": {
                "type": float,
                "description": "Maximum clustering coefficient allowed for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "use_entropy": {
                "type": bool,
                "description": "Enable entropy-based state changes"
            },
            "survival_min_entropy": {
                "type": float,
                "description": "Minimum entropy required for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "survival_max_entropy": {
                "type": float,
                "description": "Maximum entropy allowed for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "birth_min_entropy": {
                "type": float,
                "description": "Minimum entropy required for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "birth_max_entropy": {
                "type": float,
                "description": "Maximum entropy allowed for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "edge_formation_rate": {
                "type": float,
                "description": "Base probability of forming new edges",
                "min": 0.0,
                "max": 1.0
            },
            "edge_removal_rate": {
                "type": float,
                "description": "Base probability of removing existing edges",
                "min": 0.0,
                "max": 1.0
            },
            "min_edges": {
                "type": int,
                "description": "Minimum edges a node must maintain",
                "min": 0,
                "max": 8
            },
            "max_edges": {
                "type": int,
                "description": "Maximum edges a node can have",
                "min": 0,
                "max": 8
            },
            "min_shared_neighbors": {
                "type": int,
                "description": "Minimum shared neighbors required to maintain an edge",
                "min": 0,
                "max": 8
            },
            "use_random_death": {
                "type": bool,
                "description": "Enable random death chance"
            },
            "death_probability": {
                "type": float,
                "description": "Probability of random death when enabled",
                "min": 0.0,
                "max": 1.0
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on survival and birth rules"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        # Handle empty cell case
        if current_state == -1.0:
            return -1.0

        # Calculate basic metrics
        active_neighbors = np.sum(neighborhood_data.states[neighbors] > 0)
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        # Get optional metrics
        if self.get_param('use_edge_density', False):
            edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)
        else:
            edge_density = 0.5

        if self.get_param('use_clustering', False):
            clustering = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)
        else:
            clustering = 0.5

        if self.get_param('use_entropy', False):
            entropy = self.get_metric(StateEntropy, node_idx, neighborhood_data)
        else:
            entropy = 0.5

        if current_state > 0:  # Active node
            # Get survival parameters
            survival_min_active = self.get_param('survival_min_active')
            survival_max_active = self.get_param('survival_max_active')
            survival_min_connected = self.get_param('survival_min_connected')
            survival_max_connected = self.get_param('survival_max_connected')
            
            if None in (survival_min_active, survival_max_active,
                       survival_min_connected, survival_max_connected):
                return 0.0

            # Check basic survival conditions
            survives = (
                survival_min_active <= active_neighbors <= survival_max_active and
                survival_min_connected <= connected_neighbors <= survival_max_connected
            )

            # Check edge density if enabled
            if self.get_param('use_edge_density', False):
                survival_min_density = self.get_param('survival_min_density')
                survival_max_density = self.get_param('survival_max_density')
                if None not in (survival_min_density, survival_max_density):
                    survives = survives and (
                        survival_min_density <= edge_density <= survival_max_density
                    )

            # Check clustering if enabled
            if self.get_param('use_clustering', False):
                survival_min_clustering = self.get_param('survival_min_clustering')
                survival_max_clustering = self.get_param('survival_max_clustering')
                if None not in (survival_min_clustering, survival_max_clustering):
                    survives = survives and (
                        survival_min_clustering <= clustering <= survival_max_clustering
                    )

            # Check entropy if enabled
            if self.get_param('use_entropy', False):
                survival_min_entropy = self.get_param('survival_min_entropy')
                survival_max_entropy = self.get_param('survival_max_entropy')
                if None not in (survival_min_entropy, survival_max_entropy):
                    survives = survives and (
                        survival_min_entropy <= entropy <= survival_max_entropy
                    )

            # Handle random death
            if (survives and 
                self.get_param('use_random_death', False) and
                random.random() < self.get_param('death_probability', 0.0)):
                return 0.0

            # Handle ties
            if active_neighbors in (survival_min_active, survival_max_active):
                survives = TieBreaker.resolve(
                    current_state,
                    neighborhood_data.states[neighbors[0]],
                    connected_neighbors,
                    np.sum(neighborhood_data.edge_matrix[neighbors[0]]),
                    TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
                )

            return 1.0 if survives else 0.0

        else:  # Inactive node
            # Get birth parameters
            birth_min_active = self.get_param('birth_min_active')
            birth_max_active = self.get_param('birth_max_active')
            birth_min_connected = self.get_param('birth_min_connected')
            birth_max_connected = self.get_param('birth_max_connected')
            
            if None in (birth_min_active, birth_max_active,
                       birth_min_connected, birth_max_connected):
                return 0.0

            # Check basic birth conditions
            born = (
                birth_min_active <= active_neighbors <= birth_max_active and
                birth_min_connected <= connected_neighbors <= birth_max_connected
            )

            # Check edge density if enabled
            if self.get_param('use_edge_density', False):
                birth_min_density = self.get_param('birth_min_density')
                birth_max_density = self.get_param('birth_max_density')
                if None not in (birth_min_density, birth_max_density):
                    born = born and (
                        birth_min_density <= edge_density <= birth_max_density
                    )

            # Check clustering if enabled
            if self.get_param('use_clustering', False):
                birth_min_clustering = self.get_param('birth_min_clustering')
                birth_max_clustering = self.get_param('birth_max_clustering')
                if None not in (birth_min_clustering, birth_max_clustering):
                    born = born and (
                        birth_min_clustering <= clustering <= birth_max_clustering
                    )

            # Check entropy if enabled
            if self.get_param('use_entropy', False):
                birth_min_entropy = self.get_param('birth_min_entropy')
                birth_max_entropy = self.get_param('birth_max_entropy')
                if None not in (birth_min_entropy, birth_max_entropy):
                    born = born and (
                        birth_min_entropy <= entropy <= birth_max_entropy
                    )

            # Handle ties
            if active_neighbors in (birth_min_active, birth_max_active):
                born = TieBreaker.resolve(
                    current_state,
                    neighborhood_data.states[neighbors[0]],
                    connected_neighbors,
                    np.sum(neighborhood_data.edge_matrix[neighbors[0]]),
                    TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
                )

            return 1.0 if born else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on node states and metrics"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get edge parameters
        edge_formation_rate = self.get_param('edge_formation_rate')
        edge_removal_rate = self.get_param('edge_removal_rate')
        min_edges = self.get_param('min_edges')
        max_edges = self.get_param('max_edges')
        min_shared_neighbors = self.get_param('min_shared_neighbors')

        if None in (edge_formation_rate, edge_removal_rate,
                   min_edges, max_edges, min_shared_neighbors):
            return new_edges

        current_edges = {n for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}

        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            # Calculate shared neighbors
            shared = np.sum(neighborhood_data.edge_matrix[node_idx] &
                          neighborhood_data.edge_matrix[n])

            if n in current_edges:
                if shared >= min_shared_neighbors:
                    new_edges.add(n)
                elif random.random() >= edge_removal_rate:
                    new_edges.add(n)
            else:
                if random.random() < edge_formation_rate:
                    new_edges.add(n)

        # Ensure minimum edges if possible
        if len(new_edges) < min_edges:
            available = [n for n in neighbors 
                        if neighborhood_data.states[n] > 0 and n not in new_edges]
            while len(new_edges) < min_edges and available:
                n = available.pop(random.randrange(len(available)))
                new_edges.add(n)

        # Enforce maximum edges if needed
        if len(new_edges) > max_edges:
            new_edges = set(random.sample(list(new_edges), max_edges))

        return new_edges
                                            
class NetworkLife(Rule):
    """A rule that evolves the network based on node states, connection counts, neighbor states, and local network metrics."""

    PARAMETER_METADATA = {
            "activation_threshold": {
                "type": float,
                "description": "Threshold for active neighbor ratio to activate a node.",
                "min": 0.0,
                "max": 1.0
            },
            "deactivation_threshold": {
                "type": float,
                "description": "Threshold for active neighbor ratio below which an active node becomes inactive.",
                "min": 0.0,
                "max": 1.0
            },
            "min_connections": {
                "type": int,
                "description": "Minimum connections for a node to be considered active.",
                "min": 0,
                "max": 8
            },
            "max_connections": {
                "type": int,
                "description": "Max connections for a node to be considered active.",
                "min": 0,
                "max": 8
            },
            "connection_preference": {
                "type": str,
                "description": "Preference for connecting to neighbors ('similar', 'lower', 'higher').",
                "allowed_values": ['similar', 'lower', 'higher']
            },
            "similarity_tolerance": {
                "type": int,
                "description": "Tolerance for degree difference when connecting to 'similar' neighbors.",
                "min": 0,
                "max": 8
            },
            "connect_probability": {
                "type": float,
                "description": "Base probability of connecting to a valid neighbor.",
                "min": 0.0,
                "max": 1.0
            },
            "disconnect_probability": {
                "type": float,
                "description": "Probability of disconnecting from a neighbor.",
                "min": 0.0,
                "max": 1.0
            },
            "min_shared_neighbors": {
                "type": int,
                "description": "Minimum number of shared neighbors to maintain an edge.",
                "min": 0,
                "max": 8
            },
            "use_assortativity": {
                "type": bool,
                "description": "Enable assortativity-based edge formation"
            },
            "assortativity_weight": {
                "type": float,
                "description": "Weight given to degree assortativity in edge formation",
                "min": 0.0,
                "max": 1.0
            },
            "use_clustering": {
                "type": bool,
                "description": "Enable clustering coefficient-based state changes"
            },
            "min_clustering": {
                "type": float,
                "description": "Minimum clustering coefficient required for survival",
                "min": 0.0,
                "max": 1.0
            },
            "max_clustering": {
                "type": float,
                "description": "Maximum clustering coefficient allowed before death",
                "min": 0.0,
                "max": 1.0
            },
            "use_betweenness": {
                "type": bool,
                "description": "Enable betweenness centrality-based state changes"
            },
            "min_betweenness": {
                "type": float,
                "description": "Minimum betweenness centrality required for survival",
                "min": 0.0,
                "max": 1.0
            },
            "max_betweenness": {
                "type": float,
                "description": "Maximum betweenness centrality allowed before death",
                "min": 0.0,
                "max": 1.0
            },
            "use_eigenvector": {
                "type": bool,
                "description": "Enable eigenvector centrality-based state changes"
            },
            "min_eigenvector": {
                "type": float,
                "description": "Minimum eigenvector centrality required for survival",
                "min": 0.0,
                "max": 1.0
            },
            "max_eigenvector": {
                "type": float,
                "description": "Maximum eigenvector centrality allowed before death",
                "min": 0.0,
                "max": 1.0
            },
            "use_random_death": {
                "type": bool,
                "description": "Enable random death chance"
            },
            "death_probability": {
                "type": float,
                "description": "Probability of random death when enabled",
                "min": 0.0,
                "max": 1.0
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            },
            "connect_if_neighbor_connections_lt": {
                "type": int,
                "description": "Connect to a neighbor if the neighbor's connections are less than this value.",
                "min": 0,
                "max": 8
            },
            "connect_if_neighbor_connections_gt": {
                "type": int,
                "description": "Connect to a neighbor if the neighbor's connections are greater than this value.",
                "min": 0,
                "max": 8
            },
            "connect_if_neighbor_connections_eq": {
                "type": int,
                "description": "Connect to a neighbor if the neighbor's connections are equal to this value.",
                "min": 0,
                "max": 8
            },
            "disconnect_if_neighbor_connections_lt": {
                "type": int,
                "description": "Disconnect from a neighbor if the neighbor's connections are less than this value.",
                "min": 0,
                "max": 8
            },
            "disconnect_if_neighbor_connections_gt": {
                "type": int,
                "description": "Disconnect from a neighbor if the neighbor's connections are greater than this value.",
                "min": 0,
                "max": 8
            },
            "disconnect_if_neighbor_connections_eq": {
                "type": int,
                "description": "Disconnect from a neighbor if the neighbor's connections are equal to this value.",
                "min": 0,
                "max": 8
            }
        }

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on connection counts and neighbor states"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0:
            return -1.0

        # Get metrics
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        # Get parameters
        activation_threshold = self.get_param('activation_threshold')
        deactivation_threshold = self.get_param('deactivation_threshold')
        min_connections = self.get_param('min_connections')
        max_connections = self.get_param('max_connections')
        use_clustering = self.get_param('use_clustering')
        use_betweenness = self.get_param('use_betweenness')
        use_eigenvector = self.get_param('use_eigenvector')
        use_random_death = self.get_param('use_random_death')

        # Check survival conditions
        survives = True

        # Basic connection checks
        if not (min_connections <= connected_neighbors <= max_connections):
            survives = False

        # Active neighbor ratio check
        if current_state > 0 and active_ratio < deactivation_threshold:
            survives = False
        elif current_state <= 0 and active_ratio < activation_threshold:
            survives = False

        # Optional metric checks
        if survives and use_clustering:
            clustering = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)
            if not (self.get_param('min_clustering') <= clustering <= self.get_param('max_clustering')):
                survives = False

        if survives and use_betweenness:
            betweenness = self.get_metric(BetweennessCentrality, node_idx, neighborhood_data)
            if not (self.get_param('min_betweenness') <= betweenness <= self.get_param('max_betweenness')):
                survives = False

        if survives and use_eigenvector:
            eigenvector = self.get_metric(EigenvectorCentrality, node_idx, neighborhood_data)
            if not (self.get_param('min_eigenvector') <= eigenvector <= self.get_param('max_eigenvector')):
                survives = False

        # Random death check
        if survives and use_random_death and random.random() < self.get_param('death_probability'):
            survives = False

        return 1.0 if survives else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on connection rules"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get parameters
        connection_preference = self.get_param('connection_preference')
        similarity_tolerance = self.get_param('similarity_tolerance')
        min_shared_neighbors = self.get_param('min_shared_neighbors')
        
        # Connection thresholds
        connect_lt = self.get_param('connect_if_neighbor_connections_lt')
        connect_gt = self.get_param('connect_if_neighbor_connections_gt')
        connect_eq = self.get_param('connect_if_neighbor_connections_eq')
        disconnect_lt = self.get_param('disconnect_if_neighbor_connections_lt')
        disconnect_gt = self.get_param('disconnect_if_neighbor_connections_gt')
        disconnect_eq = self.get_param('disconnect_if_neighbor_connections_eq')

        self_connections = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            neighbor_connections = np.sum(neighborhood_data.edge_matrix[n, neighborhood_data.get_neighbor_indices(n)])
            shared = np.sum(neighborhood_data.edge_matrix[node_idx] & neighborhood_data.edge_matrix[n])

            # Check connection conditions
            should_connect = False

            # Connection preference check
            if connection_preference == 'similar':
                should_connect = abs(self_connections - neighbor_connections) <= similarity_tolerance
            elif connection_preference == 'lower':
                should_connect = neighbor_connections < self_connections
            elif connection_preference == 'higher':
                should_connect = neighbor_connections > self_connections

            # Connection threshold checks
            if connect_lt is not None and neighbor_connections < connect_lt:
                should_connect = True
            if connect_gt is not None and neighbor_connections > connect_gt:
                should_connect = True
            if connect_eq is not None and neighbor_connections == connect_eq:
                should_connect = True

            # Disconnection threshold checks
            if disconnect_lt is not None and neighbor_connections < disconnect_lt:
                should_connect = False
            if disconnect_gt is not None and neighbor_connections > disconnect_gt:
                should_connect = False
            if disconnect_eq is not None and neighbor_connections == disconnect_eq:
                should_connect = False

            # Shared neighbor check
            if shared >= min_shared_neighbors:
                should_connect = True

            if should_connect:
                new_edges.add(n)

        return new_edges
    
class NeighborConnections(Rule):
    """
    Neighbor Connections and States Rule: Determines connections based on
    the states and connection counts of both the self node and its neighbors.
    """

    PARAMETER_METADATA = {
        "connections_for_high_state": {
            "type": int,
            "description": "Number of connections for a cell to be considered in the 'high' state (1.0).",
            "min": 0,
            "max": 8
        },
        "connections_for_medium_state": {
            "type": int,
            "description": "Number of connections for a cell to be considered in the 'medium' state (0.5).",
            "min": 0,
            "max": 8
        },
        "connections_for_low_state": {
            "type": int,
            "description": "Number of connections for a cell to be considered in the 'low' state (0.2).",
            "min": 0,
            "max": 8
        },
        "connect_if_self_state_gt": {
            "type": float,
            "description": "Connect to a neighbor if the cell's own state is greater than this threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_self_state_lt": {
            "type": float,
            "description": "Connect to a neighbor if the cell's own state is less than this threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_self_state_eq": {
            "type": float,
            "description": "Connect to a neighbor if the cell's own state is equal to this value.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_neighbor_state_gt": {
            "type": float,
            "description": "Connect to a neighbor if the neighbor's state is greater than this threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_neighbor_state_lt": {
            "type": float,
            "description": "Connect to a neighbor if the neighbor's state is less than this threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_neighbor_state_eq": {
            "type": float,
            "description": "Connect to a neighbor if the neighbor's state is equal to this value.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_state_difference_gt": {
            "type": float,
            "description": "Connect to a neighbor if the absolute difference between the cell's state and the neighbor's state is greater than this threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_state_difference_lt": {
            "type": float,
            "description": "Connect to a neighbor if the absolute difference between the cell's state and the neighbor's state is less than this threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_neighbor_connections_gt_self": {
            "type": bool,
            "description": "If True, connect to neighbors that have more connections than the cell itself."
        },
        "connect_if_neighbor_connections_lt_self": {
            "type": bool,
            "description": "If True, connect to neighbors that have fewer connections than the cell itself."
        },
        "connect_if_neighbor_connections_eq_self": {
            "type": bool,
            "description": "If True, connect to neighbors that have the same number of connections as the cell itself."
        },
        "min_connections_allowed": {
            "type": int,
            "description": "Minimum number of connections a cell is allowed to have.",
            "min": 0,
            "max": 8
        },
        "max_connections_allowed": {
            "type": int,
            "description": "Maximum number of connections a cell is allowed to have.",
            "min": 0,
            "max": 8
        },
        "preferred_connection_range": {
            "type": list,
            "description": "Preferred range for the number of connections a cell should have.",
            "element_type": int,
            "min": 0,
            "max": 8,
            "length": 2
        },
        "base_connection_probability": {
            "type": float,
            "description": "Base probability of forming a new connection with a neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "state_multiplier_effect": {
            "type": float,
            "description": "Multiplier effect of the cell's state on the connection probability.",
            "min": 0.0,
            "max": 1.0
        },
        "maintain_existing_if_stable": {
            "type": bool,
            "description": "If True, maintain existing connections if the cell's state is stable."
        },
        "prevent_isolation": {
            "type": bool,
            "description": "If True, prevent cells from becoming completely isolated by removing their last connection."
        },
        "min_shared_neighbors": {
            "type": int,
            "description": "Minimum number of shared neighbors required to maintain a connection between two cells.",
            "min": 0,
            "max": 8
        },
        "max_shared_neighbors": {
            "type": int,
            "description": "Maximum number of shared neighbors allowed before considering disconnection.",
            "min": 0,
            "max": 8
        },
        "connection_distance_threshold": {
            "type": float,
            "description": "Maximum distance between two cells for a connection to be formed.",
            "min": 0.0
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
        "type": str,
        "description": "Method to resolve ties in state transitions",
        "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self.params: Dict[str, Any] = {}  # Initialize params as a dictionary
        self.state_history: List[Dict[int, float]] = []
        self.edge_history: List[Set[Tuple[int, int]]] = []

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData, node_positions: Optional[np.ndarray] = None, dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on connection count"""
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell case (-1.0)
        if current_state == -1.0:
            # You can define birth conditions here if needed
            return -1.0  # Remain empty for now

        # Determine state based on connection count
        high_state_threshold = self.get_param('connections_for_high_state')
        medium_state_threshold = self.get_param('connections_for_medium_state')
        low_state_threshold = self.get_param('connections_for_low_state')

        if high_state_threshold is not None and connected_neighbors >= high_state_threshold:
            return 1.0  # High state
        elif medium_state_threshold is not None and connected_neighbors >= medium_state_threshold:
            return 0.5  # Medium state
        elif low_state_threshold is not None and connected_neighbors >= low_state_threshold:
            return 0.2  # Low state
        else:
            return 0.0  # Inactive
                                    
    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData, node_positions: Optional[np.ndarray] = None, dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on connection rules and neighbor states"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        current_state = neighborhood_data.states[node_idx]
        self_connections = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        for n in neighbors:
            # Skip if neighbor is empty
            if neighborhood_data.states[n] == -1:
                continue

            neighbor_state = neighborhood_data.states[n]
            neighbor_indices = neighborhood_data.get_neighbor_indices(n)
            neighbor_connections = np.sum(neighborhood_data.edge_matrix[n, neighbor_indices]) if len(neighbor_indices) > 0 else 0 # ADDED check for empty neighbor indices

            connect = False

            # Apply connection rules based on parameters
            connect_if_self_state_gt = self.get_param('connect_if_self_state_gt')
            if connect_if_self_state_gt is not None and current_state > connect_if_self_state_gt:
                connect = True
            connect_if_self_state_lt = self.get_param('connect_if_self_state_lt')
            if connect_if_self_state_lt is not None and current_state < connect_if_self_state_lt:
                connect = True
            connect_if_self_state_eq = self.get_param('connect_if_self_state_eq')
            if connect_if_self_state_eq is not None and current_state == connect_if_self_state_eq:
                connect = True
            connect_if_neighbor_state_gt = self.get_param('connect_if_neighbor_state_gt')
            if connect_if_neighbor_state_gt is not None and isinstance(neighbor_state, (int, float)) and neighbor_state > connect_if_neighbor_state_gt:
                connect = True
            connect_if_neighbor_state_lt = self.get_param('connect_if_neighbor_state_lt')
            if connect_if_neighbor_state_lt is not None and isinstance(neighbor_state, (int, float)) and neighbor_state < connect_if_neighbor_state_lt:
                connect = True
            connect_if_neighbor_state_eq = self.get_param('connect_if_neighbor_state_eq')
            if self.get_param('connect_if_neighbor_state_eq') is not None and neighbor_state == self.params.get('connect_if_neighbor_state_eq'):
                connect = True
            if self.params.get('connect_if_neighbor_connections_gt_self') and (self_connections is not None and neighbor_connections is not None) and neighbor_connections > self_connections:
                connect = True
            if self.params.get('connect_if_neighbor_connections_lt_self') and (self_connections is not None and neighbor_connections is not None) and neighbor_connections < self_connections:
                connect = True
            if self.params.get('connect_if_neighbor_connections_eq_self') and (self_connections is not None and neighbor_connections is not None) and neighbor_connections == self_connections:
                connect = True

            # Apply disconnection rules (only if not already connected)
            if not neighborhood_data.edge_matrix[node_idx, n]:
                disconnect_if_neighbor_connections_lt = self.params.get('disconnect_if_neighbor_connections_lt')
                if disconnect_if_neighbor_connections_lt is not None and neighbor_connections < disconnect_if_neighbor_connections_lt:
                    connect = False  # Override connection
                disconnect_if_neighbor_connections_gt = self.params.get('disconnect_if_neighbor_connections_gt')
                if disconnect_if_neighbor_connections_gt is not None and neighbor_connections > disconnect_if_neighbor_connections_gt:
                    connect = False  # Override connection
                if self.params.get('disconnect_if_neighbor_connections_eq') is not None and neighbor_connections == self.params.get('disconnect_if_neighbor_connections_eq'):
                    connect = False  # Override connection
                disconnect_if_neighbor_states_sum_lt = self.params.get('disconnect_if_neighbor_states_sum_lt')
                
                if disconnect_if_neighbor_states_sum_lt is not None:
                    neighbor_state_sum = np.sum(neighborhood_data.states[neighbor_indices]) if len(neighbor_indices) > 0 else None
                    if neighbor_state_sum is not None and neighbor_state_sum < disconnect_if_neighbor_states_sum_lt:
                        connect = False  # Override connection
                
                disconnect_if_neighbor_states_sum_gt = self.params.get('disconnect_if_neighbor_states_sum_gt')
                if disconnect_if_neighbor_states_sum_gt is not None:
                    neighbor_state_sum = np.sum(neighborhood_data.states[neighbor_indices]) if len(neighbor_indices) > 0 else None
                    if neighbor_state_sum is not None and neighbor_state_sum > disconnect_if_neighbor_states_sum_gt:
                        connect = False  # Override connection
                
                disconnect_if_neighbor_states_sum_eq = self.params.get('disconnect_if_neighbor_states_sum_eq')
                if self.params.get('disconnect_if_neighbor_states_sum_eq') is not None:
                    neighbor_state_sum = np.sum(neighborhood_data.states[neighbor_indices]) if len(neighbor_indices) > 0 else None
                    if neighbor_state_sum is not None and neighbor_state_sum == self.params.get('disconnect_if_neighbor_states_sum_eq'):
                        connect = False  # Override connection

                if connect:
                    new_edges.add(n)

        return new_edges
    
class AdaptiveNetworkRule(Rule):
    """Creates adaptive networks based on local and global conditions."""

    PARAMETER_METADATA = {
            "state_rule_table": {
                "type": dict,
                "description": "Rule table for state transitions. Keys are (current_state, neighbor_pattern, connection_pattern)",
                "default": {}
            },
            "edge_rule_table": {
                "type": dict,
                "description": "Rule table for edge updates. Keys are (self_state, neighbor_state, connection_pattern)",
                "default": {}
            },
            "birth_threshold": {
                "type": float,
                "description": "Minimum active neighbor ratio for an empty cell to become active.",
                "min": 0.0,
                "max": 1.0
            },
            "death_threshold": {
                "type": float,
                "description": "Maximum active neighbor ratio for an active cell to become inactive.",
                "min": 0.0,
                "max": 1.0
            },
            "min_connections": {
                "type": int,
                "description": "Minimum connections for a node to be considered active.",
                "min": 0,
                "max": 8
            },
            "max_connections": {
                "type": int,
                "description": "Max connections for a node to be considered active.",
                "min": 0,
                "max": 8
            },
            "adaptation_rate": {
                "type": float,
                "description": "Rate at which the network adapts to changes.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_threshold": {
                "type": float,
                "description": "Threshold for stability to trigger adaptation.",
                "min": 0.0,
                "max": 1.0
            },
            "use_clustering": {
                "type": bool,
                "description": "Enable clustering coefficient-based state changes"
            },
            "min_clustering": {
                "type": float,
                "description": "Minimum clustering coefficient required for survival",
                "min": 0.0,
                "max": 1.0
            },
            "max_clustering": {
                "type": float,
                "description": "Maximum clustering coefficient allowed before death",
                "min": 0.0,
                "max": 1.0
            },
            "use_entropy": {
                "type": bool,
                "description": "Enable entropy-based state modifications"
            },
            "min_entropy": {
                "type": float,
                "description": "Minimum entropy required for state changes",
                "min": 0.0,
                "max": 1.0
            },
            "max_entropy": {
                "type": float,
                "description": "Maximum entropy allowed before state changes",
                "min": 0.0,
                "max": 1.0
            },
            "edge_memory": {
                "type": int,
                "description": "Number of steps to remember previous edges",
                "min": 0,
                "max": 10
            },
            "state_memory": {
                "type": int,
                "description": "Number of steps to remember previous states",
                "min": 0,
                "max": 10
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on rule table and adaptive metrics"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0:
            return -1.0

        # Get parameters
        state_rule_table = self.get_param('state_rule_table')
        birth_threshold = self.get_param('birth_threshold')
        death_threshold = self.get_param('death_threshold')
        min_connections = self.get_param('min_connections')
        max_connections = self.get_param('max_connections')
        adaptation_rate = self.get_param('adaptation_rate')
        stability_threshold = self.get_param('stability_threshold')

        # Get metrics
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])
        
        # Check connection bounds
        if not (min_connections <= connected_neighbors <= max_connections):
            return 0.0

        # Create neighbor pattern string
        neighbor_pattern = ''.join(['1' if neighborhood_data.states[n] > 0 else '0' 
                                  for n in neighbors])
        
        # Create connection pattern string
        connection_pattern = ''.join(['1' if neighborhood_data.edge_matrix[node_idx, n] else '0'
                                    for n in neighbors])

        # Create lookup key
        key = f"({int(current_state)}, {neighbor_pattern}, {connection_pattern})"
        
        # Get base state from rule table
        new_state = state_rule_table.get(key, state_rule_table.get('default', 0))

        # Apply adaptive modifiers
        survives = True

        # Birth/death thresholds
        if current_state > 0 and active_ratio < death_threshold:
            survives = False
        elif current_state <= 0 and active_ratio < birth_threshold:
            survives = False

        # Apply metric modifiers if enabled
        if self.get_param('use_clustering'):
            clustering = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)
            if not (self.get_param('min_clustering') <= clustering <= self.get_param('max_clustering')):
                survives = False

        if self.get_param('use_entropy') and survives:
            entropy = self.get_metric(StateEntropy, node_idx, neighborhood_data)
            if not (self.get_param('min_entropy') <= entropy <= self.get_param('max_entropy')):
                survives = False

        # Calculate stability score
        edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)
        stability_score = (active_ratio + edge_density) / 2

        # Apply adaptation based on stability
        if stability_score < stability_threshold:
            adaptation_chance = random.random() * adaptation_rate
            if adaptation_chance > stability_score:
                new_state = 1 - new_state  # Flip state

        # Handle ties using tiebreaker
        if new_state == current_state:
            new_state = int(TieBreaker.resolve(
                current_state,
                neighborhood_data.states[neighbors[0]] if len(neighbors) > 0 else 0,
                connected_neighbors,
                np.sum(neighborhood_data.edge_matrix[neighbors[0]]) if len(neighbors) > 0 else 0,
                TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
            ))

        # Update state memory
        state_memory = self.get_param('state_memory')
        if state_memory > 0:
            if not hasattr(self, 'state_history'):
                self.state_history = []
            self.state_history.append({node_idx: new_state})
            if len(self.state_history) > state_memory:
                self.state_history.pop(0)

        return float(new_state) if survives else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on rule table and adaptive metrics"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get parameters
        edge_rule_table = self.get_param('edge_rule_table')
        min_connections = self.get_param('min_connections')
        max_connections = self.get_param('max_connections')
        min_shared_neighbors = self.get_param('min_shared_neighbors')
        adaptation_rate = self.get_param('adaptation_rate')

        # Get current connection pattern
        connection_pattern = ''.join(['1' if neighborhood_data.edge_matrix[node_idx, n] else '0'
                                    for n in neighbors])

        current_connections = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            # Calculate shared neighbors
            shared = np.sum(neighborhood_data.edge_matrix[node_idx] & 
                          neighborhood_data.edge_matrix[n])

            # Create lookup key
            key = f"({int(neighborhood_data.states[node_idx] > 0)}, {int(neighborhood_data.states[n] > 0)}, {connection_pattern})"
            
            # Get action from rule table
            action = edge_rule_table.get(key, edge_rule_table.get('default', 'maintain'))

            # Apply action with adaptive modification
            should_connect = False
            
            if action == 'add':
                should_connect = True
            elif action == 'maintain' and neighborhood_data.edge_matrix[node_idx, n]:
                should_connect = True
            elif action == 'remove':
                should_connect = False

            # Apply adaptation
            if should_connect:
                # Calculate edge stability
                edge_stability = shared / max(current_connections, 1)
                
                # Chance to adapt based on stability
                if random.random() < adaptation_rate * (1 - edge_stability):
                    should_connect = not should_connect

            # Check connection bounds
            if should_connect:
                if current_connections >= max_connections:
                    should_connect = False
                elif shared < min_shared_neighbors:
                    should_connect = False

            if should_connect:
                new_edges.add(n)

        # Ensure minimum connections if possible
        if len(new_edges) < min_connections:
            available = [n for n in neighbors 
                        if neighborhood_data.states[n] > 0 and n not in new_edges]
            while len(new_edges) < min_connections and available:
                n = available.pop(random.randrange(len(available)))
                new_edges.add(n)

        return new_edges
        
class StablePolygons(Rule):
    """Creates and maintains stable polygonal structures."""

    PARAMETER_METADATA = {
            "min_active_neighbors": {
                "type": int,
                "description": "Minimum active neighbors required for a cell to survive.",
                "min": 0,
                "max": 8
            },
            "max_active_neighbors": {
                "type": int,
                "description": "Maximum active neighbors allowed for a cell to survive.",
                "min": 0,
                "max": 8
            },
            "target_neighbors": {
                "type": int,
                "description": "Target number of neighbors for stable polygon formation.",
                "min": 3,
                "max": 8
            },
            "angle_tolerance": {
                "type": float,
                "description": "Tolerance for the angular spacing between neighbors in a stable polygon.",
                "min": 0.0,
                "max": 1.0
            },
            "edge_length_variance": {
                "type": float,
                "description": "Allowed variance in the lengths of edges connecting a cell to its neighbors.",
                "min": 0.0,
                "max": 1.0
            },
            "use_symmetry": {
                "type": bool,
                "description": "Enable symmetry-based state changes"
            },
            "min_symmetry": {
                "type": float,
                "description": "Minimum symmetry score required for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "max_symmetry": {
                "type": float,
                "description": "Maximum symmetry score allowed for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "use_regularity": {
                "type": bool,
                "description": "Enable polygon regularity checks"
            },
            "min_regularity": {
                "type": float,
                "description": "Minimum regularity score required for survival",
                "min": 0.0,
                "max": 1.0
            },
            "max_regularity": {
                "type": float,
                "description": "Maximum regularity score allowed for survival",
                "min": 0.0,
                "max": 1.0
            },
            "growth_threshold": {
                "type": float,
                "description": "Minimum ratio of active neighbors required for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "growth_rate": {
                "type": float,
                "description": "Probability of a new cell forming in an empty space next to an active cell.",
                "min": 0.0,
                "max": 1.0
            },
            "min_density": {
                "type": float,
                "description": "Minimum density of active cells in the neighborhood required to maintain the structure.",
                "min": 0.0,
                "max": 1.0
            },
            "max_density": {
                "type": float,
                "description": "Maximum density of active cells allowed in the neighborhood to prevent overcrowding.",
                "min": 0.0,
                "max": 1.0
            },
            "edge_formation_rate": {
                "type": float,
                "description": "Base probability of forming new edges",
                "min": 0.0,
                "max": 1.0
            },
            "edge_removal_rate": {
                "type": float,
                "description": "Base probability of removing existing edges",
                "min": 0.0,
                "max": 1.0
            },
            "min_edges": {
                "type": int,
                "description": "Minimum edges a node must maintain",
                "min": 0,
                "max": 8
            },
            "max_edges": {
                "type": int,
                "description": "Maximum edges a node can have",
                "min": 0,
                "max": 8
            },
            "use_angle_preference": {
                "type": bool,
                "description": "Enable preferred angle-based edge formation"
            },
            "preferred_angle": {
                "type": float,
                "description": "Preferred angle between edges (in radians)",
                "min": 0.0,
                "max": 6.28318530718  # 2*pi
            },
            "angle_weight": {
                "type": float,
                "description": "Weight given to angle preference in edge formation",
                "min": 0.0,
                "max": 1.0
            },
            "use_random_death": {
                "type": bool,
                "description": "Enable random death chance"
            },
            "death_probability": {
                "type": float,
                "description": "Probability of random death when enabled",
                "min": 0.0,
                "max": 1.0
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on polygon formation rules"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0 or node_positions is None:
            return -1.0

        # Get parameters
        min_active = self.get_param('min_active_neighbors')
        max_active = self.get_param('max_active_neighbors')
        target_neighbors = self.get_param('target_neighbors')
        angle_tolerance = self.get_param('angle_tolerance')
        edge_length_variance = self.get_param('edge_length_variance')
        min_density = self.get_param('min_density')
        max_density = self.get_param('max_density')
        growth_threshold = self.get_param('growth_threshold')
        growth_rate = self.get_param('growth_rate')

        # Calculate basic metrics
        active_neighbors = np.sum(neighborhood_data.states[neighbors] > 0)
        edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        # Check basic survival conditions
        survives = (min_active <= active_neighbors <= max_active and
                   min_density <= edge_density <= max_density)

        if survives and len(neighbors) > 0:
            # Calculate polygon metrics
            center = node_positions[node_idx]
            active_positions = node_positions[neighbors][neighborhood_data.states[neighbors] > 0]
            
            if len(active_positions) > 0:
                # Calculate edge lengths and angles
                vectors = active_positions - center
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])
                lengths = np.linalg.norm(vectors, axis=1)
                
                # Sort angles and calculate differences
                angles = np.sort(angles)
                angle_diffs = np.diff(angles)
                angle_diffs = np.append(angle_diffs, 2*np.pi - (angles[-1] - angles[0]))
                
                # Calculate regularity metrics
                target_angle = 2*np.pi / target_neighbors
                angle_variance = np.std(angle_diffs) / target_angle
                length_variance = np.std(lengths) / np.mean(lengths)
                regularity_score = 1.0 - (angle_variance + length_variance) / 2

                # Check symmetry if enabled
                if self.get_param('use_symmetry'):
                    symmetry_score = 1.0 - np.std(angle_diffs) / np.pi
                    if not (self.get_param('min_symmetry') <= symmetry_score <= self.get_param('max_symmetry')):
                        survives = False

                # Check regularity if enabled
                if self.get_param('use_regularity'):
                    if not (self.get_param('min_regularity') <= regularity_score <= self.get_param('max_regularity')):
                        survives = False

                # Check edge length variance
                if length_variance > edge_length_variance:
                    survives = False

        # Birth logic for inactive nodes
        elif current_state <= 0:
            if (active_neighbors / len(neighbors) >= growth_threshold and
                random.random() < growth_rate):
                return 1.0

        # Handle random death
        if survives and self.get_param('use_random_death'):
            if random.random() < self.get_param('death_probability'):
                survives = False

        # Handle ties using tiebreaker
        if survives and active_neighbors in (min_active, max_active):
            survives = TieBreaker.resolve(
                current_state,
                neighborhood_data.states[neighbors[0]] if len(neighbors) > 0 else 0,
                connected_neighbors,
                np.sum(neighborhood_data.edge_matrix[neighbors[0]]) if len(neighbors) > 0 else 0,
                TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
            )

        return 1.0 if survives else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates to maintain polygon structure"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0 or node_positions is None:
            return new_edges

        # Get parameters
        target_neighbors = self.get_param('target_neighbors')
        angle_tolerance = self.get_param('angle_tolerance')
        edge_formation_rate = self.get_param('edge_formation_rate')
        edge_removal_rate = self.get_param('edge_removal_rate')
        min_edges = self.get_param('min_edges')
        max_edges = self.get_param('max_edges')
        use_angle_preference = self.get_param('use_angle_preference')
        preferred_angle = self.get_param('preferred_angle')
        angle_weight = self.get_param('angle_weight')

        center = node_positions[node_idx]
        current_edges = {n for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}

        # Calculate angles to all neighbors
        neighbor_angles = {}
        for n in neighbors:
            if neighborhood_data.states[n] > 0:
                direction = node_positions[n] - center
                angle = np.arctan2(direction[1], direction[0])
                neighbor_angles[n] = angle

        # Process each neighbor
        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            angle = neighbor_angles[n]
            should_connect = False

            if use_angle_preference:
                # Check alignment with preferred angle
                angle_diff = abs(angle - preferred_angle)
                if angle_diff > np.pi:
                    angle_diff = 2*np.pi - angle_diff
                
                # Calculate connection probability based on angle
                angle_factor = 1.0 - (angle_diff / np.pi)
                connection_prob = edge_formation_rate * (1.0 - angle_weight + angle_weight * angle_factor)
                
                should_connect = random.random() < connection_prob
            else:
                # Calculate ideal angle based on target neighbors
                target_angle = 2*np.pi / target_neighbors
                angle_diff = abs(angle % target_angle)
                
                if angle_diff <= angle_tolerance:
                    should_connect = random.random() < edge_formation_rate

            # Maintain existing edges with some probability
            if n in current_edges and not should_connect:
                should_connect = random.random() > edge_removal_rate

            if should_connect:
                new_edges.add(n)

        # Ensure minimum edges if possible
        if len(new_edges) < min_edges:
            available = [n for n in neighbors 
                        if neighborhood_data.states[n] > 0 and n not in new_edges]
            while len(new_edges) < min_edges and available:
                n = available.pop(random.randrange(len(available)))
                new_edges.add(n)

        # Limit maximum edges
        if len(new_edges) > max_edges:
            new_edges = set(sorted(list(new_edges), 
                                 key=lambda n: neighbor_angles.get(n, 0))[:max_edges])

        return new_edges
        
class GeometricAngle(Rule):
    """Creates geometric patterns based on neighbor angles and edge properties."""

    PARAMETER_METADATA = {
            "target_neighbors": {
                "type": int,
                "description": "Target number of neighbors each cell should have to form a stable geometric structure.",
                "min": 0,
                "max": 8
            },
            "preferred_angle": {
                "type": float,
                "description": "Preferred angle between neighbors in a stable geometric structure (in radians).",
                "min": 0.0,
                "max": 6.28318530718  # 2*pi
            },
            "angle_tolerance": {
                "type": float,
                "description": "Tolerance for the angular deviation between neighbors in a stable geometric structure.",
                "min": 0.0,
                "max": 1.0
            },
            "edge_length_variance": {
                "type": float,
                "description": "Allowed variance in the lengths of edges connecting a cell to its neighbors.",
                "min": 0.0,
                "max": 1.0
            },
            "growth_threshold": {
                "type": float,
                "description": "Minimum ratio of active neighbors required for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "growth_rate": {
                "type": float,
                "description": "Probability of a new cell forming in an empty space next to an active cell.",
                "min": 0.0,
                "max": 1.0
            },
            "min_density": {
                "type": float,
                "description": "Minimum density of active cells in the neighborhood required to maintain the structure.",
                "min": 0.0,
                "max": 1.0
            },
            "max_density": {
                "type": float,
                "description": "Maximum density of active cells allowed in the neighborhood to prevent overcrowding.",
                "min": 0.0,
                "max": 1.0
            },
            "min_clustering": {
                "type": float,
                "description": "Minimum clustering coefficient required to maintain the stability of the structure.",
                "min": 0.0,
                "max": 1.0
            },
            "edge_formation_rate": {
                "type": float,
                "description": "Probability of forming a new edge between two active cells.",
                "min": 0.0,
                "max": 1.0
            },
            "edge_removal_rate": {
                "type": float,
                "description": "Probability of removing an unstable edge between two cells.",
                "min": 0.0,
                "max": 1.0
            },
            "min_shared_neighbors": {
                "type": int,
                "description": "Minimum number of shared neighbors required to keep an edge between two cells.",
                "min": 0,
                "max": 8
            },
            "angular_stability": {
                "type": float,
                "description": "Angular stability factor that influences the survival of cells based on their angular alignment with neighbors.",
                "min": 0.0,
                "max": 1.0
            },
            "angular_influence": {
                "type": float,
                "description": "Influence of angular alignment on the overall state update of a cell.",
                "min": 0.0,
                "max": 1.0
            },
            "prevent_isolation": {
                "type": bool,
                "description": "If True, prevents cells from becoming completely isolated by removing their last connection."
            },
            "maintain_existing": {
                "type": bool,
                "description": "If True, maintains existing connections between cells, even if they don't perfectly fit the geometric criteria."
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on geometric relationships"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0 or node_positions is None:
            return -1.0

        # Get parameters
        target_neighbors = self.get_param('target_neighbors')
        min_connections = self.get_param('min_connections')
        max_connections = self.get_param('max_connections')
        preferred_angle = self.get_param('preferred_angle')
        angle_tolerance = self.get_param('angle_tolerance')
        min_density = self.get_param('min_density')
        max_density = self.get_param('max_density')
        angular_stability = self.get_param('angular_stability')
        angular_influence = self.get_param('angular_influence')

        # Calculate basic metrics
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)

        # Check basic survival conditions
        survives = (min_connections <= connected_neighbors <= max_connections and
                   min_density <= edge_density <= max_density)

        if survives and len(neighbors) > 0:
            # Calculate angles between neighbors
            center = node_positions[node_idx]
            neighbor_positions = node_positions[neighbors]
            vectors = neighbor_positions - center
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            
            # Sort angles and calculate differences
            angles = np.sort(angles)
            angle_diffs = np.diff(angles)
            angle_diffs = np.append(angle_diffs, 2*np.pi - (angles[-1] - angles[0]))
            
            # Calculate angular metrics
            target_angle = 2*np.pi / target_neighbors
            angle_deviations = np.abs(angle_diffs - target_angle)
            mean_deviation = float(np.mean(angle_deviations))
            
            # Calculate angular stability score
            stability_score = 1.0 - (mean_deviation / np.pi)
            
            # Apply angular influence
            if stability_score < angular_stability:
                survives = False
            elif random.random() > stability_score * angular_influence:
                survives = False

            # Check clustering if enabled
            if survives and self.get_param('use_clustering'):
                clustering = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)
                if clustering < self.get_param('min_clustering'):
                    survives = False

        # Handle ties using tiebreaker
        if survives and active_ratio == self.get_param('activation_threshold'):
            survives = TieBreaker.resolve(
                current_state,
                neighborhood_data.states[neighbors[0]] if len(neighbors) > 0 else 0,
                connected_neighbors,
                np.sum(neighborhood_data.edge_matrix[neighbors[0]]) if len(neighbors) > 0 else 0,
                TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
            )

        return 1.0 if survives else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on geometric relationships"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0 or node_positions is None:
            return new_edges

        # Get parameters
        preferred_angle = self.get_param('preferred_angle')
        angle_tolerance = self.get_param('angle_tolerance')
        edge_formation_rate = self.get_param('edge_formation_rate')
        edge_removal_rate = self.get_param('edge_removal_rate')
        min_shared_neighbors = self.get_param('min_shared_neighbors')
        maintain_existing = self.get_param('maintain_existing')
        prevent_isolation = self.get_param('prevent_isolation')

        center = node_positions[node_idx]
        current_edges = {n for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}

        # Calculate angles to all neighbors
        neighbor_angles = {}
        for n in neighbors:
            if neighborhood_data.states[n] > 0:
                direction = node_positions[n] - center
                angle = np.arctan2(direction[1], direction[0])
                neighbor_angles[n] = angle

        # Process each neighbor
        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            # Calculate shared neighbors
            shared = np.sum(neighborhood_data.edge_matrix[node_idx] & 
                          neighborhood_data.edge_matrix[n])

            # Check angle alignment
            angle = neighbor_angles[n]
            angle_diff = abs(angle - preferred_angle)
            if angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff

            should_connect = False

            # Check angle alignment
            if angle_diff <= angle_tolerance:
                should_connect = random.random() < edge_formation_rate
            elif n in current_edges:
                # Maintain existing edges with some probability
                if maintain_existing:
                    should_connect = random.random() > edge_removal_rate
                elif shared >= min_shared_neighbors:
                    should_connect = True

            if should_connect:
                new_edges.add(n)

        # Prevent isolation if enabled
        if prevent_isolation and len(new_edges) == 0 and len(current_edges) > 0:
            # Keep at least one existing edge
            new_edges.add(random.choice(list(current_edges)))

        return new_edges
    
class SymmetryRule(Rule):
    """
    Symmetry Rule: Creates and maintains symmetric patterns across multiple axes.
    """

    PARAMETER_METADATA = {
        "num_symmetry_axes": {
            "type": int,
            "description": "Number of symmetry axes to maintain in the pattern.",
            "min": 1,
            "max": 8
        },
        "axis_angle_tolerance": {
            "type": float,
            "description": "Tolerance for the angular alignment of cells with the symmetry axes.",
            "min": 0.0,
            "max": 1.0
        },
        "symmetry_radius": {
            "type": int,
            "description": "Radius within which to check for symmetry around a cell.",
            "min": 1,
            "max": 100
        },
        "symmetry_tolerance": {
            "type": float,
            "description": "Tolerance for matching the states of cells that are symmetrically positioned.",
            "min": 0.0,
            "max": 1.0
        },
        "min_symmetric_nodes": {
            "type": int,
            "description": "Minimum number of cells required to form a symmetric pattern.",
            "min": 1,
            "max": 100
        },
        "max_asymmetric_nodes": {
            "type": int,
            "description": "Maximum number of asymmetric cells allowed in a symmetric pattern.",
            "min": 0,
            "max": 100
        },
        "rotational_order": {
            "type": int,
            "description": "Order of rotational symmetry to enforce (e.g., 4 for 90-degree rotational symmetry).",
            "min": 1,
            "max": 8
        },
        "rotation_tolerance": {
            "type": float,
            "description": "Tolerance for the rotational alignment of cells in a rotationally symmetric pattern.",
            "min": 0.0,
            "max": 1.0
        },
        "pattern_density": {
            "type": float,
            "description": "Target density of active cells within a symmetric pattern.",
            "min": 0.0,
            "max": 1.0
        },
        "pattern_scale": {
            "type": float,
            "description": "Scale factor for the size of the symmetric pattern.",
            "min": 0.1,
            "max": 10.0
        },
        "min_pattern_size": {
            "type": int,
            "description": "Minimum number of cells required to form a symmetric pattern.",
            "min": 1,
            "max": 100
        },
        "max_pattern_size": {
            "type": int,
            "description": "Maximum number of cells allowed in a symmetric pattern.",
            "min": 1,
            "max": 100
        },
        "symmetry_break_threshold": {
            "type": float,
            "description": "Threshold for the symmetry score above which intentional symmetry breaking can occur.",
            "min": 0.0,
            "max": 1.0
        },
        "break_probability": {
            "type": float,
            "description": "Probability of a cell breaking the symmetry if the symmetry score is above the threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "max_broken_duration": {
            "type": int,
            "description": "Maximum number of simulation steps to maintain a broken symmetry.",
            "min": 0,
            "max": 100
        },
        "min_symmetric_connections": {
            "type": int,
            "description": "Minimum number of connections a cell must have to other cells in the symmetric pattern.",
            "min": 0,
            "max": 8
        },
        "max_connections_per_axis": {
            "type": int,
            "description": "Maximum number of connections allowed along each symmetry axis.",
            "min": 0,
            "max": 8
        },
        "connection_angle_weight": {
            "type": float,
            "description": "Weight of the angular alignment in the connection decision.",
            "min": 0.0,
            "max": 1.0
        },
        "maintain_existing_symmetry": {
            "type": bool,
            "description": "If True, preserves existing symmetric patterns by preventing cells from disrupting the symmetry."
        },
        "prevent_isolation": {
            "type": bool,
            "description": "If True, prevents cells from becoming completely isolated by removing their last connection."
        },
        "min_connections": {
            "type": int,
            "description": "Minimum number of connections a cell must have to prevent isolation.",
            "min": 0,
            "max": 8
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on symmetry analysis"""
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell
        if current_state == -1.0:
            # For now, no birth logic specific to symmetry.  Could be added.
            return -1.0

        if node_positions is None:
            return current_state

        center = node_positions[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        neighbor_positions = node_positions[neighbors]
        neighbor_states = neighborhood_data.states[neighbors]

        # Get parameters
        symmetry_tolerance = self.get_param('symmetry_tolerance', 0.2)

        # Analyze symmetry
        rule_params = RuleParameters(**self.params)
        symmetry_data = analyze_symmetry(center, neighbor_positions, neighbor_states, rule_params)

        # Deactivate if symmetry is too low
        if symmetry_data.score < symmetry_tolerance:
            return 0.0
        else:
            return 1.0
            
    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData, node_positions: Optional[np.ndarray] = None, dimension_type: Optional[Dimension] = None) -> Set[int]:
        """
        Compute edge updates to maintain or enhance symmetry.
        If the node is active and positions are provided, the function analyzes symmetry and updates edges accordingly.
        It can also maintain existing symmetry if the option is enabled.
        """
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)

        if neighborhood_data.states[node_idx] > 0 and node_positions is not None:
            center = node_positions[node_idx]
            
            # Get parameters
            maintain_existing_symmetry = self.get_param('maintain_existing_symmetry', False)
            
            # Analyze symmetry
            rule_params = RuleParameters(**self.params)
            symmetry_data = analyze_symmetry(center, node_positions[neighbors], neighborhood_data.states[neighbors], rule_params)

            for n_idx in neighbors:
                if neighborhood_data.states[n_idx] > 0:
                    # Add edges to symmetric pairs
                    for pair_idx in symmetry_data.symmetric_pairs:
                        if n_idx == neighbors[pair_idx[0]] or n_idx == neighbors[pair_idx[1]]:
                            new_edges.add(n_idx)
                            break

                    # Maintain existing edges if option is enabled
                    if maintain_existing_symmetry and neighborhood_data.edge_matrix[node_idx, n_idx]:
                        new_edges.add(n_idx)

        return new_edges
        
class FractalRule(Rule):
    """Creates self-similar patterns at different scales."""

    PARAMETER_METADATA = {
            "min_pattern_size": {
                "type": int,
                "description": "Minimum number of cells required to form a fractal pattern.",
                "min": 1,
                "max": 100
            },
            "max_pattern_size": {
                "type": int,
                "description": "Maximum number of cells allowed in a fractal pattern.",
                "min": 1,
                "max": 100
            },
            "target_branching": {
                "type": int,
                "description": "Target number of branches each cell should have in a fractal pattern.",
                "min": 1,
                "max": 8
            },
            "max_recursion_depth": {
                "type": int,
                "description": "Maximum depth of recursion allowed for fractal patterns.",
                "min": 0,
                "max": 10
            },
            "min_scale": {
                "type": float,
                "description": "Minimum scale factor for fractal patterns.",
                "min": 0.0,
                "max": 10.0
            },
            "max_scale": {
                "type": float,
                "description": "Maximum scale factor for fractal patterns.",
                "min": 0.0,
                "max": 10.0
            },
            "scale_ratio": {
                "type": float,
                "description": "Ratio between successive scales in the fractal pattern.",
                "min": 0.0,
                "max": 10.0
            },
            "scale_tolerance": {
                "type": float,
                "description": "Tolerance for deviations from the ideal scale ratio.",
                "min": 0.0,
                "max": 1.0
            },
            "min_self_similarity": {
                "type": float,
                "description": "Minimum self-similarity score required for a cell to be considered part of a fractal pattern.",
                "min": 0.0,
                "max": 1.0
            },
            "similarity_threshold": {
                "type": float,
                "description": "Threshold for the similarity score above which a new pattern is recognized as a fractal.",
                "min": 0.0,
                "max": 1.0
            },
            "similarity_tolerance": {
                "type": float,
                "description": "Tolerance for deviations in the similarity score.",
                "min": 0.0,
                "max": 1.0
            },
            "target_dimension": {
                "type": float,
                "description": "Target fractal dimension for the patterns.",
                "min": 0.0,
                "max": 3.0
            },
            "dimension_tolerance": {
                "type": float,
                "description": "Tolerance for deviations from the target fractal dimension.",
                "min": 0.0,
                "max": 1.0
            },
            "dimension_weight": {
                "type": float,
                "description": "Weight of the fractal dimension in the overall state update of a cell.",
                "min": 0.0,
                "max": 1.0
            },
            "min_branching_score": {
                "type": float,
                "description": "Minimum branching score required for a cell to be considered part of a branching structure.",
                "min": 0.0,
                "max": 1.0
            },
            "branching_tolerance": {
                "type": float,
                "description": "Tolerance for deviations from the ideal branching structure.",
                "min": 0.0,
                "max": 1.0
            },
            "branch_angle_variance": {
                "type": float,
                "description": "Allowed variance in the angles between branches.",
                "min": 0.0,
                "max": 1.0
            },
            "formation_threshold": {
                "type": float,
                "description": "Threshold for the formation score above which a new fractal pattern is created.",
                "min": 0.0,
                "max": 1.0
            },
            "recursion_threshold": {
                "type": float,
                "description": "Threshold for the recursion score above which a pattern will recursively generate smaller patterns.",
                "min": 0.0,
                "max": 1.0
            },
            "pattern_merge_threshold": {
                "type": float,
                "description": "Threshold for the similarity score above which two patterns will merge.",
                "min": 0.0,
                "max": 1.0
            },
            "min_pattern_stability": {
                "type": float,
                "description": "Minimum stability score required for a pattern to be considered stable.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_threshold": {
                "type": float,
                "description": "Threshold for the overall stability of a cell, based on its participation in stable fractal patterns.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_tolerance": {
                "type": float,
                "description": "Tolerance for deviations in the stability score.",
                "min": 0.0,
                "max": 1.0
            },
            "maintain_symmetry": {
                "type": bool,
                "description": "If True, maintains the symmetry of fractal patterns by preventing cells from disrupting the symmetry."
            },
            "prevent_pattern_collapse": {
                "type": bool,
                "description": "If True, prevents fractal patterns from collapsing by ensuring cells always have enough connections to maintain the pattern."
            },
            "min_pattern_connections": {
                "type": int,
                "description": "Minimum number of connections a cell must have to other cells in a stored pattern.",
                "min": 0,
                "max": 8
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on fractal properties"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0 or node_positions is None:
            return -1.0

        # Get parameters
        min_pattern_size = self.get_param('min_pattern_size')
        max_pattern_size = self.get_param('max_pattern_size')
        target_branching = self.get_param('target_branching')
        min_scale = self.get_param('min_scale')
        max_scale = self.get_param('max_scale')
        scale_ratio = self.get_param('scale_ratio')
        min_self_similarity = self.get_param('min_self_similarity')
        target_dimension = self.get_param('target_dimension')
        dimension_tolerance = self.get_param('dimension_tolerance')
        min_branching_score = self.get_param('min_branching_score')
        formation_threshold = self.get_param('formation_threshold')

        # Calculate fractal metrics
        if len(neighbors) > 0:
            # Calculate self-similarity score
            self_similarity, fractal_dim, role = calculate_fractal_metrics(
                node_idx, neighborhood_data, node_positions, reference_scale=1.0
            )

            # Calculate branching metrics
            branching_score, scale_potential = calculate_branching_metrics(
                node_idx, neighborhood_data, node_positions, target_branching
            )

            # Check survival conditions
            survives = True

            # Check self-similarity
            if self_similarity < min_self_similarity:
                survives = False

            # Check fractal dimension
            if abs(fractal_dim - target_dimension) > dimension_tolerance:
                survives = False

            # Check branching
            if branching_score < min_branching_score:
                survives = False

            # Check scale
            if node_positions is not None:
                center = node_positions[node_idx]
                neighbor_positions = node_positions[neighbors]
                distances = np.linalg.norm(neighbor_positions - center, axis=1)
                avg_scale = np.mean(distances)
                
                if not (min_scale <= avg_scale <= max_scale):
                    survives = False

            # Check pattern size
            pattern_size = np.sum(neighborhood_data.edge_matrix[node_idx])
            if not (min_pattern_size <= pattern_size <= max_pattern_size):
                survives = False

            # Birth logic for inactive nodes
            if current_state <= 0:
                active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
                if active_ratio >= formation_threshold and branching_score >= min_branching_score:
                    return 1.0

            # Apply stability checks
            if survives:
                stability_score = (self_similarity + branching_score) / 2
                if stability_score < self.get_param('stability_threshold'):
                    survives = False

            # Handle pattern maintenance
            if survives and self.get_param('maintain_symmetry'):
                symmetry_data = analyze_symmetry(
                    center,
                    neighbor_positions,
                    neighborhood_data.states[neighbors],
                    RuleParameters(**self.params)
                )
                if symmetry_data.score < self.get_param('min_symmetry'):
                    survives = False

            return 1.0 if survives else 0.0
        
        return 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates to maintain fractal structure"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0 or node_positions is None:
            return new_edges

        # Get parameters
        target_branching = self.get_param('target_branching')
        scale_ratio = self.get_param('scale_ratio')
        scale_tolerance = self.get_param('scale_tolerance')
        min_pattern_connections = self.get_param('min_pattern_connections')

        center = node_positions[node_idx]
        current_edges = {n for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}

        # Calculate distances to neighbors
        neighbor_distances = {}
        for n in neighbors:
            if neighborhood_data.states[n] > 0:
                distance = np.linalg.norm(node_positions[n] - center)
                neighbor_distances[n] = distance

        # Sort neighbors by distance
        sorted_neighbors = sorted(
            [n for n in neighbors if n in neighbor_distances],
            key=lambda n: neighbor_distances[n]
        )

        # Create fractal branching pattern
        if sorted_neighbors:
            base_distance = neighbor_distances[sorted_neighbors[0]]
            current_scale = base_distance

            for n in sorted_neighbors:
                distance = neighbor_distances[n]
                
                # Check if distance matches current scale
                scale_match = abs(distance / current_scale - 1.0) <= scale_tolerance

                # Check if distance matches next scale
                next_scale_match = abs(distance / (current_scale * scale_ratio) - 1.0) <= scale_tolerance

                if scale_match or next_scale_match:
                    # Calculate branching angle
                    direction = node_positions[n] - center
                    angle = np.arctan2(direction[1], direction[0])
                    
                    # Check if angle fits branching pattern
                    angle_step = 2 * np.pi / target_branching
                    angle_match = any(
                        abs((angle - i * angle_step) % (2 * np.pi)) <= self.get_param('branching_tolerance')
                        for i in range(target_branching)
                    )

                    if angle_match:
                        new_edges.add(n)

                    if next_scale_match:
                        current_scale *= scale_ratio

        # Ensure minimum connections
        if len(new_edges) < min_pattern_connections:
            available = [n for n in sorted_neighbors if n not in new_edges]
            while len(new_edges) < min_pattern_connections and available:
                new_edges.add(available.pop(0))

        # Prevent pattern collapse if enabled
        if self.get_param('prevent_pattern_collapse'):
            if len(current_edges) > 0 and len(new_edges) == 0:
                new_edges.add(next(iter(current_edges)))

        return new_edges
    
class ModularRule(Rule):
    """
    Modular Rule: Creates distinct functional modules with specific internal/external connection patterns.
    """

    PARAMETER_METADATA = {
        "min_module_size": {
            "type": int,
            "description": "Minimum number of cells required to form a module.",
            "min": 1,
            "max": 100
        },
        "max_module_size": {
            "type": int,
            "description": "Maximum number of cells allowed in a module.",
            "min": 1,
            "max": 100
        },
        "optimal_module_size": {
            "type": int,
            "description": "Target number of cells for a module.",
            "min": 1,
            "max": 100
        },
        "min_internal_density": {
            "type": float,
            "description": "Minimum density of connections within a module.",
            "min": 0.0,
            "max": 1.0
        },
        "max_external_density": {
            "type": float,
            "description": "Maximum density of connections from a module to other modules.",
            "min": 0.0,
            "max": 1.0
        },
        "core_node_ratio": {
            "type": float,
            "description": "Target ratio of core nodes within a module.",
            "min": 0.0,
            "max": 1.0
        },
        "boundary_node_ratio": {
            "type": float,
            "description": "Target ratio of boundary nodes within a module.",
            "min": 0.0,
            "max": 1.0
        },
        "bridge_node_ratio": {
            "type": float,
            "description": "Target ratio of bridge nodes within a module.",
            "min": 0.0,
            "max": 1.0
        },
        "specialization_rate": {
            "type": float,
            "description": "Rate at which a module becomes specialized.",
            "min": 0.0,
            "max": 1.0
        },
        "max_specialization": {
            "type": float,
            "description": "Maximum level of specialization a module can reach.",
            "min": 0.0,
            "max": 1.0
        },
        "specialization_threshold": {
            "type": float,
            "description": "Threshold for the specialization score above which a module is considered specialized.",
            "min": 0.0,
            "max": 1.0
        },
        "inter_module_connection_rate": {
            "type": float,
            "description": "Rate at which connections form between different modules.",
            "min": 0.0,
            "max": 1.0
        },
        "max_connected_modules": {
            "type": int,
            "description": "Maximum number of other modules a module can connect to.",
            "min": 0,
            "max": 8
        },
        "min_module_connections": {
            "type": int,
            "description": "Minimum number of connections a module must have to other modules.",
            "min": 0,
            "max": 8
        },
        "module_formation_threshold": {
            "type": float,
            "description": "Threshold for the formation score above which a new module is created.",
            "min": 0.0,
            "max": 1.0
        },
        "module_merge_threshold": {
            "type": float,
            "description": "Threshold for the similarity score above which two modules will merge.",
            "min": 0.0,
            "max": 1.0
        },
        "module_split_threshold": {
            "type": float,
            "description": "Threshold for the dissimilarity score below which a module will split.",
            "min": 0.0,
            "max": 1.0
        },
        "stability_threshold": {
            "type": float,
            "description": "Threshold for the stability score above which a module is considered stable.",
            "min": 0.0,
            "max": 1.0
        },
        "adaptation_rate": {
            "type": float,
            "description": "Rate at which a module adapts to changes in the environment.",
            "min": 0.0,
            "max": 1.0
        },
        "maintain_module_integrity": {
            "type": bool,
            "description": "If True, maintains the integrity of modules by preventing cells from leaving or joining without proper cause."
        },
        "prevent_module_isolation": {
            "type": bool,
            "description": "If True, prevents modules from becoming completely isolated by ensuring they always have at least one connection to other modules."
        },
        "min_module_connections_internal": {
            "type": int,
            "description": "Minimum number of internal connections a module must have to prevent collapse.",
            "min": 0,
            "max": 8
        },
        "same_level_density": {
            "type": float,
            "description": "Target density of connections within the same hierarchical level.",
            "min": 0.0,
            "max": 1.0
        },
        "up_connection_density": {
            "type": float,
            "description": "Target density of connections from a cell to cells in the level above it.",
            "min": 0.0,
            "max": 1.0
        },
        "down_connection_density": {
            "type": float,
            "description": "Target density of connections from a cell to cells in the level below it.",
            "min": 0.0,
            "max": 1.0
        },
        "min_connections_per_level": {
            "type": int,
            "description": "Minimum number of connections a cell must have to each adjacent level.",
            "min": 0,
            "max": 8
        },
        "max_connections_per_level": {
            "type": int,
            "description": "Maximum number of connections a cell is allowed to have to each adjacent level.",
            "min": 0,
            "max": 8
        },
        "up_influence_weight": {
            "type": float,
            "description": "Weight of the influence from cells in higher levels on a cell's state.",
            "min": 0.0,
            "max": 1.0
        },
        "down_influence_weight": {
            "type": float,
            "description": "Weight of the influence from cells in lower levels on a cell's state.",
            "min": 0.0,
            "max": 1.0
        },
        "same_level_weight": {
            "type": float,
            "description": "Weight of the influence from cells within the same level on a cell's state.",
            "min": 0.0,
            "max": 1.0
        },
        "activation_threshold": {
            "type": float,
            "description": "Threshold for the combined influence required for an inactive cell to become active.",
            "min": 0.0,
            "max": 1.0
        },
        "deactivation_threshold": {
            "type": float,
            "description": "Threshold for the combined influence below which an active cell becomes inactive.",
            "min": 0.0,
            "max": 1.0
        },
        "level_activation_bonus": {
            "type": float,
            "description": "Additional activation chance given to cells based on their level in the hierarchy.",
            "min": 0.0,
            "max": 1.0
        },
        "level_up_threshold": {
            "type": float,
            "description": "Threshold for the combined influence required for a cell to move up a level in the hierarchy.",
            "min": 0.0,
            "max": 1.0
        },
        "level_down_threshold": {
            "type": float,
            "description": "Threshold for the combined influence below which a cell moves down a level in the hierarchy.",
            "min": 0.0,
            "max": 1.0
        },
        "level_change_probability": {
            "type": float,
            "description": "Probability of a cell changing levels in the hierarchy, even if it doesn't meet the thresholds.",
            "min": 0.0,
            "max": 1.0
        },
        "maintain_level_diversity": {
            "type": bool,
            "description": "If True, maintains a mix of different levels in the hierarchy."
        },
        "prevent_level_isolation": {
            "type": bool,
            "description": "If True, prevents levels from becoming completely disconnected from each other."
        },
        "min_level_connections": {
            "type": int,
            "description": "Minimum number of connections a level must have to other levels.",
            "min": 0,
            "max": 8
        },
        "connection_stability_bonus": {
            "type": float,
            "description": "Bonus added to the stability of a cell if it has stable connections to other cells.",
            "min": 0.0,
            "max": 1.0
        },
        "level_stability_threshold": {
            "type": float,
            "description": "Threshold for the stability of a level, based on the consistency of cell states within the level.",
            "min": 0.0,
            "max": 1.0
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
            super().__init__(metadata)
            self.module_assignments: Dict[int, int] = {}
            self.specialization_scores: Dict[int, float] = {}

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on module analysis"""
        current_state = neighborhood_data.states[node_idx]
        
        # Get parameters safely using .get() with defaults
        activation_threshold = self.get_param('activation_threshold', 0.5)
        deactivation_threshold = self.get_param('deactivation_threshold', 0.3)
        level_activation_bonus = self.get_param('level_activation_bonus', 0.1)
        up_influence_weight = self.get_param('up_influence_weight', 0.3)
        down_influence_weight = self.get_param('down_influence_weight', 0.3)
        same_level_weight = self.get_param('same_level_weight', 0.4)
        
        # Get module assignments
        module_assignments = self.module_assignments  # Replace with actual module assignments
        
        # Analyze hierarchy
        hierarchy_data = analyze_hierarchy(
            node_idx, neighborhood_data, module_assignments, max_level=3
        )
        module_data = analyze_module(
            node_idx, neighborhood_data, module_assignments, self.specialization_scores
        )
        
        # Calculate combined influence
        influence = (
            hierarchy_data.influence_up * up_influence_weight +
            hierarchy_data.influence_down * down_influence_weight +
            hierarchy_data.level_density * same_level_weight
        )
        
        # Adjust influence based on level
        if hierarchy_data.level > 1:
            influence += level_activation_bonus
        
        # State update logic
        if current_state > 0:
            if influence < deactivation_threshold:
                return 0.0  # Deactivate
            else:
                return 1.0  # Maintain
        else:
            if influence > activation_threshold:
                return 1.0  # Activate
            else:
                return 0.0  # Remain inactive

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates to maintain module structure"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        # Get parameters safely using .get() with defaults
        inter_module_connection_rate = self.get_param('inter_module_connection_rate', 0.2)
        same_level_density = self.get_param('same_level_density', 0.5)
        up_connection_density = self.get_param('up_connection_density', 0.3)
        down_connection_density = self.get_param('down_connection_density', 0.2)
        min_connections_per_level = self.get_param('min_connections_per_level', 1)

        # Get module assignments
        module_assignments = self.module_assignments  # Replace with actual module assignments

        # Analyze hierarchy
        hierarchy_data = analyze_hierarchy(
            node_idx, neighborhood_data, module_assignments, max_level=3
        )
        
        # Get neighbors
        same_level_edges = hierarchy_data.level_neighbors.get(hierarchy_data.level, [])

        # Calculate state
        for n in same_level_edges:
            if random.random() < same_level_density:
                new_edges.add(n)
            
        # Add connections to higher and lower levels
        for level, neighbors in hierarchy_data.level_neighbors.items():
            if level < hierarchy_data.level and random.random() < down_connection_density:
                for n in neighbors:
                    new_edges.add(n)
            elif level > hierarchy_data.level and random.random() < up_connection_density:
                for n in neighbors:
                    new_edges.add(n)
                
        # Add inter-module connections
        module_data = analyze_module(
            node_idx, neighborhood_data, module_assignments, self.specialization_scores
        )
        
        for other_module, connected_nodes in module_data.inter_module_connections.items():
            if random.random() < inter_module_connection_rate:
                for n in connected_nodes:
                    new_edges.add(n)
                    
        return new_edges
       
class FlowRule(Rule):
    """Creates structures optimized for directional flow or information transfer."""

    PARAMETER_METADATA = {
            "flow_alignment_threshold": {
                "type": float,
                "description": "Minimum alignment of a cell's direction with the global flow direction to be considered part of the flow.",
                "min": 0.0,
                "max": 1.0
            },
            "flow_direction_weight": {
                "type": float,
                "description": "Weight of the flow direction in the overall state update of a cell.",
                "min": 0.0,
                "max": 1.0
            },
            "flow_variation_tolerance": {
                "type": float,
                "description": "Allowed variation from the main flow direction for a cell to be considered part of the flow.",
                "min": 0.0,
                "max": 1.0
            },
            "min_path_capacity": {
                "type": float,
                "description": "Minimum capacity required for a flow path to be considered viable.",
                "min": 0.0,
                "max": 1.0
            },
            "max_path_capacity": {
                "type": float,
                "description": "Maximum capacity allowed for a flow path to prevent congestion.",
                "min": 0.0,
                "max": 1.0
            },
            "optimal_capacity": {
                "type": float,
                "description": "Target capacity utilization for flow paths.",
                "min": 0.0,
                "max": 1.0
            },
            "max_bottleneck_severity": {
                "type": float,
                "description": "Maximum allowed severity of bottlenecks in the flow network.",
                "min": 0.0,
                "max": 1.0
            },
            "min_path_redundancy": {
                "type": float,
                "description": "Minimum path redundancy required to ensure flow can be rerouted around bottlenecks.",
                "min": 0.0,
                "max": 1.0
            },
            "bottleneck_threshold": {
                "type": float,
                "description": "Threshold for bottleneck detection.",
                "min": 0.0,
                "max": 1.0
            },
            "max_pressure_difference": {
                "type": float,
                "description": "Maximum allowed difference between upstream and downstream pressure.",
                "min": 0.0,
                "max": 1.0
            },
            "pressure_equalization_rate": {
                "type": float,
                "description": "Rate at which cells equalize pressure with their neighbors.",
                "min": 0.0,
                "max": 1.0
            },
            "pressure_threshold": {
                "type": float,
                "description": "Threshold for pressure response.",
                "min": 0.0,
                "max": 1.0
            },
            "path_optimization_rate": {
                "type": float,
                "description": "Rate at which cells optimize their connections to improve flow efficiency.",
                "min": 0.0,
                "max": 1.0
            },
            "path_efficiency_threshold": {
                "type": float,
                "description": "Required path efficiency for a cell to maintain its connections.",
                "min": 0.0,
                "max": 1.0
            },
            "alternative_path_weight": {
                "type": float,
                "description": "Weight given to alternative paths when optimizing flow.",
                "min": 0.0,
                "max": 1.0
            },
            "min_flow_stability": {
                "type": float,
                "description": "Minimum required flow stability for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_threshold": {
                "type": float,
                "description": "Threshold for stable flow patterns.",
                "min": 0.0,
                "max": 1.0
            },
            "flow_adaptation_rate": {
                "type": float,
                "description": "Rate at which cells adapt their behavior to maintain flow stability.",
                "min": 0.0,
                "max": 1.0
            },
            "maintain_critical_paths": {
                "type": bool,
                "description": "If True, preserves critical flow paths by preventing cells from disrupting the flow."
            },
            "prevent_flow_disruption": {
                "type": bool,
                "description": "If True, prevents cells from completely disrupting the flow by ensuring they always have enough connections to maintain the flow."
            },
            "min_flow_connections": {
                "type": int,
                "description": "Minimum number of flow-aligned connections a cell must have to prevent flow disruption.",
                "min": 0,
                "max": 8
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            },
            "activation_threshold": {
                "type": float,
                "description": "Threshold for active neighbor ratio to activate a node.",
                "min": 0.0,
                "max": 1.0
            },
            "deactivation_threshold": {
                "type": float,
                "description": "Threshold for active neighbor ratio below which an active node becomes inactive.",
                "min": 0.0,
                "max": 1.0
            }
        }

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on flow dynamics"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0 or node_positions is None:
            return -1.0

        # Get parameters
        flow_alignment_threshold = self.get_param('flow_alignment_threshold')
        flow_direction_weight = self.get_param('flow_direction_weight')
        min_path_capacity = self.get_param('min_path_capacity')
        max_path_capacity = self.get_param('max_path_capacity')
        optimal_capacity = self.get_param('optimal_capacity')
        max_bottleneck_severity = self.get_param('max_bottleneck_severity')
        min_path_redundancy = self.get_param('min_path_redundancy')
        min_flow_stability = self.get_param('min_flow_stability')
        activation_threshold = self.get_param('activation_threshold')
        deactivation_threshold = self.get_param('deactivation_threshold')

        # Calculate flow bias direction (e.g., left to right)
        flow_bias = np.array([1.0, 0.0, 0.0]) if dimension_type == Dimension.THREE_D else np.array([1.0, 0.0])

        # Get flow metrics
        flow_data = analyze_flow(
            node_idx,
            neighborhood_data,
            node_positions,
            flow_bias,
            optimal_capacity
        )

        # Check survival conditions
        survives = True

        # Check flow alignment
        alignment = np.dot(flow_data.flow_direction[:len(flow_bias)], flow_bias)
        alignment = alignment / (np.linalg.norm(flow_data.flow_direction[:len(flow_bias)]) * np.linalg.norm(flow_bias) + 1e-6)
        if abs(alignment) < flow_alignment_threshold:
            survives = False

        # Check path capacity
        if not (min_path_capacity <= flow_data.capacity_utilization <= max_path_capacity):
            survives = False

        # Check bottleneck severity
        if flow_data.bottleneck_score > max_bottleneck_severity:
            survives = False

        # Check path redundancy
        if flow_data.path_redundancy < min_path_redundancy:
            survives = False

        # Check flow stability
        if flow_data.flow_stability < min_flow_stability:
            survives = False

        # Check pressure balance
        max_pressure_diff = self.get_param('max_pressure_difference')
        if abs(flow_data.upstream_pressure - flow_data.downstream_pressure) > self.get_param('max_pressure_difference'):
            survives = False

        # Apply pressure equalization
        if survives:
            pressure_rate = self.get_param('pressure_equalization_rate')
            pressure_threshold = self.get_param('pressure_threshold')
            
            if abs(flow_data.upstream_pressure - flow_data.downstream_pressure) > pressure_threshold:
                if random.random() < pressure_rate:
                    survives = False

        # Apply path optimization
        if survives:
            optimization_rate = self.get_param('path_optimization_rate')
            efficiency_threshold = self.get_param('path_efficiency_threshold')
            
            path_efficiency = 1.0 - flow_data.bottleneck_score
            if path_efficiency < efficiency_threshold:
                if random.random() < optimization_rate:
                    survives = False

        # Maintain critical paths
        if survives and self.get_param('maintain_critical_paths'):
            if len(flow_data.critical_paths) > 0 and random.random() < 0.8:
                survives = True

        # Prevent flow disruption
        if survives and self.get_param('prevent_flow_disruption'):
            min_flow_conn = self.get_param('min_flow_connections')
            flow_aligned_connections = sum(1 for path in flow_data.critical_paths 
                                        if alignment > flow_alignment_threshold)
            if flow_aligned_connections < min_flow_conn:
                survives = False

        # Update state based on survival
        if current_state > 0:
            return 1.0 if survives else 0.0
        else:
            active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
            if active_ratio > activation_threshold:
                return 1.0
            else:
                return 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates to optimize flow"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0 or node_positions is None:
            return new_edges

        # Get parameters
        flow_alignment_threshold = self.get_param('flow_alignment_threshold')
        min_flow_connections = self.get_param('min_flow_connections')
        path_optimization_rate = self.get_param('path_optimization_rate')
        edge_formation_rate = self.get_param('edge_formation_rate')
        edge_removal_rate = self.get_param('edge_removal_rate')

        # Calculate flow bias direction
        flow_bias = np.array([1.0, 0.0, 0.0]) if dimension_type == Dimension.THREE_D else np.array([1.0, 0.0])

        # Get flow data
        flow_data = analyze_flow(
            node_idx,
            neighborhood_data,
            node_positions,
            flow_bias,
            self.get_param('optimal_capacity')
        )

        # Calculate angles to neighbors
        neighbor_angles = {}
        for n in neighbors:
            if neighborhood_data.states[n] > 0:
                direction = node_positions[n] - node_positions[node_idx]
                angle = np.arctan2(direction[1], direction[0])
                neighbor_angles[n] = angle

        # Process each neighbor
        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            # Calculate alignment with flow direction
            direction = node_positions[n] - node_positions[node_idx]
            direction = direction / np.linalg.norm(direction)
            alignment = np.dot(direction[:len(flow_bias)], flow_bias)

            # Check if part of critical path
            is_critical = (node_idx, n) in flow_data.critical_paths

            # Check if angle is aligned with flow
            angle = neighbor_angles.get(n, 0.0)
            angle_diff = abs(angle - np.arctan2(flow_bias[1], flow_bias[0]))
            if angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff

            # Calculate connection probability
            connect_prob = edge_formation_rate
            if abs(alignment) >= flow_alignment_threshold:
                connect_prob += 0.2  # Bonus for alignment

            # Connect or disconnect based on probability
            if is_critical or random.random() < connect_prob:
                new_edges.add(n)
            elif neighborhood_data.edge_matrix[node_idx, n] and random.random() < edge_removal_rate:
                continue  # Skip adding the edge, effectively removing it

        # Ensure minimum flow connections
        flow_aligned_edges = sum(1 for n in new_edges if (node_idx, n) in flow_data.critical_paths)
        if flow_aligned_edges < min_flow_connections:
            available = [n for n in neighbors 
                        if (neighborhood_data.states[n] > 0 and 
                            n not in new_edges and 
                            (node_idx, n) in flow_data.critical_paths)]
            while flow_aligned_edges < min_flow_connections and available:
                n = available.pop(random.randrange(len(available)))
                new_edges.add(n)
                flow_aligned_edges += 1

        return new_edges
    
class CompetitiveRule(Rule):
    """Creates structures based on competition for resources or connections."""

    PARAMETER_METADATA = {
            "min_territory_size": {
                "type": int,
                "description": "Minimum number of cells required to form a territory.",
                "min": 1,
                "max": 100
            },
            "max_territory_size": {
                "type": int,
                "description": "Maximum number of cells allowed in a territory.",
                "min": 1,
                "max": 100
            },
            "optimal_territory_size": {
                "type": int,
                "description": "Target number of cells for a territory.",
                "min": 1,
                "max": 100
            },
            "territory_spacing": {
                "type": float,
                "description": "Minimum spatial spacing between territories.",
                "min": 0.0,
                "max": 10.0
            },
            "min_resource_level": {
                "type": float,
                "description": "Minimum resource level required for a territory to maintain its control over a cell.",
                "min": 0.0,
                "max": 1.0
            },
            "optimal_resource_level": {
                "type": float,
                "description": "Target resource level for a territory.",
                "min": 0.0,
                "max": 1.0
            },
            "resource_decay_rate": {
                "type": float,
                "description": "Rate at which resources decay within a territory.",
                "min": 0.0,
                "max": 1.0
            },
            "competition_threshold": {
                "type": float,
                "description": "Threshold for active competition between territories.",
                "min": 0.0,
                "max": 1.0
            },
            "competitive_advantage_threshold": {
                "type": float,
                "description": "Minimum competitive advantage required for a territory to expand into a neighboring cell.",
                "min": 0.0,
                "max": 1.0
            },
            "territory_defense_threshold": {
                "type": float,
                "description": "Minimum defense strength required for a territory to maintain control over its cells.",
                "min": 0.0,
                "max": 1.0
            },
            "alliance_formation_threshold": {
                "type": float,
                "description": "Threshold for the similarity score above which two territories will form an alliance.",
                "min": 0.0,
                "max": 1.0
            },
            "max_alliances": {
                "type": int,
                "description": "Maximum number of alliances a territory can form.",
                "min": 0,
                "max": 8
            },
            "alliance_strength_weight": {
                "type": float,
                "description": "Weight of the alliance strength in the overall state update of a cell.",
                "min": 0.0,
                "max": 1.0
            },
            "expansion_threshold": {
                "type": float,
                "description": "Threshold for the expansion potential above which a territory will attempt to expand.",
                "min": 0.0,
                "max": 1.0
            },
            "max_expansion_rate": {
                "type": float,
                "description": "Maximum rate at which a territory can expand into neighboring cells.",
                "min": 0.0,
                "max": 1.0
            },
            "min_expansion_stability": {
                "type": float,
                "description": "Minimum stability required for a territory to expand.",
                "min": 0.0,
                "max": 1.0
            },
            "min_stability_threshold": {
                "type": float,
                "description": "Minimum stability required for a cell to remain part of a territory.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_bonus_threshold": {
                "type": float,
                "description": "Threshold for the stability bonus above which a cell gains an advantage in the competition for resources.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_decay_rate": {
                "type": float,
                "description": "Rate at which the stability of a territory decays over time.",
                "min": 0.0,
                "max": 1.0
            },
            "maintain_core_territory": {
                "type": bool,
                "description": "If True, maintains the core territory of each territory by preventing cells from leaving without proper cause."
            },
            "prevent_territory_collapse": {
                "type": bool,
                "description": "If True, prevents territories from completely collapsing by ensuring they always have enough cells to maintain their structure."
            },
            "min_territory_connections": {
                "type": int,
                "description": "Minimum number of connections a cell must have to other cells in its territory.",
                "min": 0,
                "max": 8
            },
            "edge_formation_rate": {
                "type": float,
                "description": "Probability of forming a new edge between two active cells.",
                "min": 0.0,
                "max": 1.0
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self.territories: Dict[int, Territory] = {}
        self.territory_assignments: Dict[int, int] = {}
        self.resource_distribution: np.ndarray = np.random.rand(GlobalSettings.Simulation.get_current_grid_size()**2)
        self.last_territory_id = 0  # Initialize territory ID counter

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on competition and territory dynamics"""
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell
        if current_state == -1.0:
            # Add birth logic here, possibly based on proximity to existing territories
            return -1.0  # Remain empty for now

        # Get parameters
        competition_threshold = self.get_param('competition_threshold')
        min_resource_level = self.get_param('min_resource_level')
        territory_defense_threshold = self.get_param('territory_defense_threshold')

        # Get territory assignment
        territory_id = self.territory_assignments.get(node_idx, -1)
        territory = self.territories.get(territory_id)

        # Calculate competition data
        competition_data = analyze_competition(
            node_idx, neighborhood_data, self.territories, self.territory_assignments, self.resource_distribution
        )

        # Check survival conditions
        survives = True

        # Check resource level
        if territory and territory.resource_level < min_resource_level:
            survives = False

        # Check defense strength
        if territory and competition_data.defense_strength < territory_defense_threshold:
            survives = False

        # Check competitive pressure
        if competition_data.competitive_pressure > competition_threshold:
            survives = False

        # Update state based on survival
        if current_state > 0:
            return 1.0 if survives else 0.0
        else:
            # Attempt to claim cell if conditions are favorable
            if competition_data.resource_access > 0.5 and competition_data.competitive_pressure < 0.3:
                return 1.0
            else:
                return 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates to maintain territory and compete with neighbors"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get parameters
        edge_formation_rate = self.get_param('edge_formation_rate')
        alliance_formation_threshold = self.get_param('alliance_formation_threshold')
        max_alliances = self.get_param('max_alliances')

        # Get territory assignment
        territory_id = self.territory_assignments.get(node_idx, -1)
        territory = self.territories.get(territory_id)

        # Calculate competition data
        competition_data = analyze_competition(
            node_idx, neighborhood_data, self.territories, self.territory_assignments, self.resource_distribution
        )

        # Process each neighbor
        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            neighbor_territory = self.territory_assignments.get(n, -1)

            # Connect to cells within the same territory
            if neighbor_territory == territory_id:
                new_edges.add(n)
            else:
                # Attempt to form alliances
                if (territory and neighbor_territory != -1 and
                    len(territory.alliances) < max_alliances and
                    competition_data.alliance_strength > alliance_formation_threshold and
                    random.random() < edge_formation_rate):
                    # Form alliance
                    territory.alliances.add(neighbor_territory)
                    new_edges.add(n)

        return new_edges
    
class AdaptiveMemoryRule(Rule):
    """Creates structures that can remember and recreate previous patterns."""

    PARAMETER_METADATA = {
            "max_patterns": {
                "type": int,
                "description": "Maximum number of patterns that can be stored in the memory.",
                "min": 1,
                "max": 100
            },
            "min_pattern_size": {
                "type": int,
                "description": "Minimum number of cells required to form a valid pattern in the memory.",
                "min": 1,
                "max": 100
            },
            "max_pattern_size": {
                "type": int,
                "description": "Maximum number of cells allowed in a pattern.",
                "min": 1,
                "max": 100
            },
            "pattern_match_threshold": {
                "type": float,
                "description": "Threshold for the pattern match score above which a stored pattern is considered a good match to the current state.",
                "min": 0.0,
                "max": 1.0
            },
            "pattern_completion_threshold": {
                "type": float,
                "description": "Threshold for the pattern completion score above which a stored pattern is considered to be actively reconstructed.",
                "min": 0.0,
                "max": 1.0
            },
            "min_pattern_stability": {
                "type": float,
                "description": "Minimum stability score required for a pattern to be stored in the memory.",
                "min": 0.0,
                "max": 1.0
            },
            "memory_decay_rate": {
                "type": float,
                "description": "Rate at which the stability of stored patterns decays over time.",
                "min": 0.0,
                "max": 1.0
            },
            "reinforcement_rate": {
                "type": float,
                "description": "Rate at which the stability of a matching pattern is reinforced.",
                "min": 0.0,
                "max": 1.0
            },
            "pattern_influence_rate": {
                "type": float,
                "description": "Rate at which stored patterns influence the state of cells in the simulation.",
                "min": 0.0,
                "max": 1.0
            },
            "new_pattern_threshold": {
                "type": float,
                "description": "Threshold for the novelty score above which a new pattern is formed.",
                "min": 0.0,
                "max": 1.0
            },
            "pattern_merge_threshold": {
                "type": float,
                "description": "Threshold for the similarity score above which two patterns are merged into a single pattern.",
                "min": 0.0,
                "max": 1.0
            },
            "pattern_split_threshold": {
                "type": float,
                "description": "Threshold for the dissimilarity score below which a pattern is split into smaller patterns.",
                "min": 0.0,
                "max": 1.0
            },
            "reconstruction_strength": {
                "type": float,
                "description": "Strength of the pattern reconstruction effect on cell states.",
                "min": 0.0,
                "max": 1.0
            },
            "completion_influence": {
                "type": float,
                "description": "Influence of pattern completion on the state of cells.",
                "min": 0.0,
                "max": 1.0
            },
            "partial_match_threshold": {
                "type": float,
                "description": "Threshold for the pattern match score above which a partial match is considered valid.",
                "min": 0.0,
                "max": 1.0
            },
            "adaptation_rate": {
                "type": float,
                "description": "Rate at which the rule adapts to changes in the environment by modifying stored patterns.",
                "min": 0.0,
                "max": 1.0
            },
            "novelty_threshold": {
                "type": float,
                "description": "Threshold for the novelty score above which a new pattern is considered novel and worth storing.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_threshold": {
                "type": float,
                "description": "Threshold for the stability score above which a pattern is considered stable and reliable.",
                "min": 0.0,
                "max": 1.0
            },
            "maintain_stable_patterns": {
                "type": bool,
                "description": "If True, maintains stable patterns by preventing them from being easily overwritten or modified."
            },
            "prevent_pattern_decay": {
                "type": bool,
                "description": "If True, prevents patterns from completely decaying and being lost from memory."
            },
            "min_pattern_connections": {
                "type": int,
                "description": "Minimum number of connections a cell must have to other cells in a stored pattern.",
                "min": 0,
                "max": 8
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self.memory_patterns: List[MemoryPattern] = []
        self.last_pattern_id = 0  # Initialize pattern ID counter

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on adaptive memory"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)

        if current_state == -1.0:
            return -1.0

        # Get parameters
        pattern_influence_rate = self.get_param('pattern_influence_rate')
        best_match_threshold = self.get_param('pattern_match_threshold')
        memory_decay_rate = self.get_param('memory_decay_rate')
        novelty_threshold = self.get_param('novelty_threshold')
        maintain_stable_patterns = self.get_param('maintain_stable_patterns')
        stability_threshold = self.get_param('stability_threshold')

        # Analyze memory
        memory_data = analyze_memory(
            node_idx, neighborhood_data, self.memory_patterns, self.cache_generation, memory_decay_rate
        )

        # Influence from matching patterns
        if memory_data.best_match_score > best_match_threshold:
            # Reconstruct based on best matching pattern
            best_pattern = memory_data.matching_patterns[0][0]
            pattern_state = best_pattern.node_states.get(node_idx, current_state)
            new_state = (current_state * (1 - pattern_influence_rate) +
                        pattern_state * pattern_influence_rate)

            # Reinforce the pattern if it's stable
            if best_pattern.stability > stability_threshold:
                best_pattern.frequency += 1

            return float(new_state)
        else:
            # Decay towards inactive state if no strong pattern match
            return float(current_state * 0.9)

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates to maintain patterns and adapt to changes"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get parameters
        pattern_completion_threshold = self.get_param('pattern_completion_threshold')
        edge_formation_rate = self.get_param('edge_formation_rate')
        memory_decay_rate = self.get_param('memory_decay_rate')
        maintain_stable_patterns = self.get_param('maintain_stable_patterns')
        min_pattern_connections = self.get_param('min_pattern_connections')
        novelty_threshold = self.get_param('novelty_threshold')

        # Analyze memory
        memory_data = analyze_memory(
            node_idx, neighborhood_data, self.memory_patterns, self.cache_generation, memory_decay_rate
        )

        # Reconstruct connections from best matching pattern
        if memory_data.best_match_score > 0.5:
            best_pattern = memory_data.matching_patterns[0][0]
            if memory_data.reconstruction_score > pattern_completion_threshold:
                for n in neighbors:
                    if (n in best_pattern.node_states and
                        random.random() < edge_formation_rate):
                        new_edges.add(n)

        # Store new patterns if novel enough
        if memory_data.best_match_score < novelty_threshold:
            self.store_pattern(node_idx, neighborhood_data)

        return new_edges

    def store_pattern(self, node_idx: int, neighborhood_data: NeighborhoodData):
        """Store a new pattern in memory"""
        # Get parameters
        max_patterns = self.get_param('max_patterns')
        min_pattern_size = self.get_param('min_pattern_size')
        max_pattern_size = self.get_param('max_pattern_size')
        stability_threshold = self.get_param('stability_threshold')

        # Get neighbors and states
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        # Create pattern
        pattern_states = {n: float(neighborhood_data.states[n]) for n in neighbors}
        pattern_connections = {(node_idx, n) for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}
        
        # Check pattern size
        if not (min_pattern_size <= len(pattern_states) <= max_pattern_size):
            return

        # Calculate pattern stability
        stability = self.calculate_pattern_stability(pattern_states, pattern_connections)
        if stability < stability_threshold:
            return

        # Create new pattern
        self.last_pattern_id += 1
        new_pattern = MemoryPattern(
            pattern_id=self.last_pattern_id,
            node_states=pattern_states,
            node_connections=pattern_connections,
            last_seen=self.cache_generation,
            frequency=1,
            stability=stability,
            influence=1.0
        )

        # Add to memory
        self.memory_patterns.append(new_pattern)

        # Enforce max patterns
        if len(self.memory_patterns) > max_patterns:
            self.memory_patterns.pop(0)

    def calculate_pattern_stability(self, pattern_states: Dict[int, float], pattern_connections: Set[Tuple[int, int]]) -> float:
        """Calculate stability of a pattern"""
        # Placeholder: Implement stability calculation based on state and connection consistency
        return 0.7  # Return a default value for now
             
class ArtificialLifeRule(Rule):
    """
    Artificial Life Rule: Creates evolving organisms with metabolism, reproduction, and adaptation.
    """

    PARAMETER_METADATA = {
        "base_metabolism": {
            "type": float,
            "description": "Base energy consumption rate for all cells.",
            "min": 0.0,
            "max": 1.0
        },
        "max_metabolism": {
            "type": float,
            "description": "Maximum metabolic rate a cell can reach.",
            "min": 0.0,
            "max": 1.0
        },
        "energy_transfer_rate": {
            "type": float,
            "description": "Rate at which energy is transferred between neighboring cells.",
            "min": 0.0,
            "max": 1.0
        },
        "min_survival_energy": {
            "type": float,
            "description": "Minimum energy level required for a cell to survive.",
            "min": 0.0,
            "max": 1.0
        },
        "reproduction_threshold": {
            "type": float,
            "description": "Energy level required for a cell to reproduce.",
            "min": 0.0,
            "max": 1.0
        },
        "min_reproduction_age": {
            "type": int,
            "description": "Minimum age (in simulation steps) a cell must reach before it can reproduce.",
            "min": 0,
            "max": 1000
        },
        "max_reproduction_size": {
            "type": int,
            "description": "Maximum size an organism can reach before it stops reproducing.",
            "min": 1,
            "max": 1000
        },
        "offspring_energy_ratio": {
            "type": float,
            "description": "Ratio of energy transferred from a parent cell to its offspring during reproduction.",
            "min": 0.0,
            "max": 1.0
        },
        "base_mutation_rate": {
            "type": float,
            "description": "Base rate of mutation for offspring cells.",
            "min": 0.0,
            "max": 1.0
        },
        "max_mutation_rate": {
            "type": float,
            "description": "Maximum mutation rate for offspring cells.",
            "min": 0.0,
            "max": 1.0
        },
        "mutation_effect_size": {
            "type": float,
            "description": "Size of the effect of a mutation on a cell's genome.",
            "min": 0.0,
            "max": 1.0
        },
        "beneficial_mutation_bias": {
            "type": float,
            "description": "Bias towards beneficial mutations.",
            "min": 0.0,
            "max": 1.0
        },
        "min_adaptation_score": {
            "type": float,
            "description": "Minimum adaptation score required for a cell to survive.",
            "min": 0.0,
            "max": 1.0
        },
        "adaptation_rate": {
            "type": float,
            "description": "Rate at which cells adapt to changes in the environment.",
            "min": 0.0,
            "max": 1.0
        },
        "phenotype_plasticity": {
            "type": float,
            "description": "Ability of a cell to change its phenotype (expressed behavior) in response to environmental changes.",
            "min": 0.0,
            "max": 1.0
        },
        "environmental_sensitivity": {
            "type": float,
            "description": "Sensitivity of cells to changes in the environment.",
            "min": 0.0,
            "max": 1.0
        },
        "min_organism_size": {
            "type": int,
            "description": "Minimum number of cells required to form a viable organism.",
            "min": 1,
            "max": 1000
        },
        "max_organism_size": {
            "type": int,
            "description": "Maximum number of cells allowed in an organism.",
            "min": 1,
            "max": 1000
        },
        "optimal_organism_size": {
            "type": int,
            "description": "Target number of cells for an organism.",
            "min": 1,
            "max": 1000
        },
        "max_organism_age": {
            "type": int,
            "description": "Maximum age (in simulation steps) an organism can reach before it dies.",
            "min": 0,
            "max": 10000
        },
        "specialization_threshold": {
            "type": float,
            "description": "Threshold for the specialization score above which a cell is considered specialized.",
            "min": 0.0,
            "max": 1.0
        },
        "membrane_ratio": {
            "type": float,
            "description": "Target ratio of membrane cells within an organism.",
            "min": 0.0,
            "max": 1.0
        },
        "core_cell_ratio": {
            "type": float,
            "description": "Target ratio of core cells within an organism.",
            "min": 0.0,
            "max": 1.0
        },
        "interaction_range": {
            "type": int,
            "description": "Range within which organisms can interact with each other.",
            "min": 0,
            "max": 100
        },
        "competition_strength": {
            "type": float,
            "description": "Strength of competition between organisms for resources.",
            "min": 0.0,
            "max": 1.0
        },
        "cooperation_bonus": {
            "type": float,
            "description": "Bonus given to cells that cooperate with other cells.",
            "min": 0.0,
            "max": 1.0
        },
        "maintain_integrity": {
            "type": bool,
            "description": "If True, maintains the integrity of organisms by preventing cells from leaving or joining without proper cause."
        },
        "prevent_fragmentation": {
            "type": bool,
            "description": "If True, prevents organisms from fragmenting into smaller, non-viable pieces."
        },
        "min_connected_cells": {
            "type": int,
            "description": "Minimum number of connected cells an organism must have to prevent fragmentation.",
            "min": 1,
            "max": 8
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self.organism_assignments: Dict[int, int] = {}
        self.energy_distribution: np.ndarray = np.zeros(GlobalSettings.Simulation.get_current_grid_size()**2)
        self.energy_levels: Dict[int, float] = {}

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on artificial life rules (placeholder)"""
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell
        if current_state == -1.0:
            # Add birth logic here, possibly based on proximity to existing organisms
            return -1.0  # Remain empty for now

        # Get parameters safely using .get() with defaults
        base_metabolism = self.get_param('base_metabolism', 0.1)
        energy_transfer_rate = self.get_param('energy_transfer_rate', 0.2)
        min_survival_energy = self.get_param('min_survival_energy', 0.3)
        
        # Get organism ID
        organism_id = self.organism_assignments.get(node_idx, -1)
        
        # If not part of an organism, decay and return
        if organism_id == -1:
            return float(max(current_state - base_metabolism, 0.0))
        
        # Get energy balance
        energy_balance, metabolic_rate = calculate_metabolic_metrics(
            node_idx, neighborhood_data, self.energy_distribution, base_metabolism
        )
        
        # Update energy level
        self.energy_levels[organism_id] = self.energy_levels.get(organism_id, 0.0) + energy_balance
        
        # Check for survival
        if self.energy_levels[organism_id] < min_survival_energy:
            return 0.0  # Cell dies due to lack of energy
        
        # Otherwise, maintain state
        return 1.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on artificial life rules (placeholder)"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        # Get parameters safely using .get() with defaults
        interaction_range = self.get_param('interaction_range', 3)
        edge_formation_rate = self.get_param('edge_formation_rate', 0.2)
        
        # Get organism ID
        organism_id = self.organism_assignments.get(node_idx, -1)
        
        # If not part of an organism, no edges
        if organism_id == -1:
            return new_edges
        
        # Connect to nearby cells within interaction range
        for n in neighbors:
            if n != node_idx:
                neighbor_organism = self.organism_assignments.get(n, -1)
                
                # Connect to cells in same organism
                if neighbor_organism == organism_id:
                    new_edges.add(n)
                # Connect to cells in other organisms within interaction range
                elif (neighbor_organism != -1 and 
                      node_positions is not None and np.linalg.norm(node_positions[node_idx] - node_positions[n]) < interaction_range and
                      random.random() < edge_formation_rate):
                    new_edges.add(n)
                
        return new_edges
    
class EdgeCountMatchingRule(Rule):
    """
    Edge Count Matching Rule: Nodes try to match the average number of edges
    of their neighbors.
    """

    PARAMETER_METADATA = {
        "activation_threshold": {
            "type": float,
            "description": "Threshold for active neighbor ratio to activate a node.",
            "min": 0.0,
            "max": 1.0
        },
        "edge_match_tolerance": {
            "type": float,
            "description": "How close the edge counts need to be for a connection (fraction of the average neighbor edge count).",
            "min": 0.0,
            "max": 1.0
        },
        "connect_probability": {
            "type": float,
            "description": "Probability of connecting to a valid neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "disconnect_probability": {
            "type": float,
            "description": "Probability of disconnecting from a neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on active neighbor ratio."""
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell case (-1.0)
        if current_state == -1.0:
            # Birth logic: activate if enough active neighbors
            if active_ratio > self.get_param('activation_threshold', 0.4):
                return 1.0  # Cell is born
            else:
                return -1.0 # Remain empty

        # For active cells, stay active based on the threshold
        if active_ratio >= self.get_param('activation_threshold', 0.4):
            return 1.0  # Survive
        else:
            return 0.0  # Become inactive

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on edge count matching."""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        self_connections = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        if neighborhood_data.states[node_idx] > 0:  # Active node
            edge_match_tolerance = self.get_param('edge_match_tolerance', 0.2)
            connect_probability = self.get_param('connect_probability', 0.7)
            disconnect_probability = self.get_param('disconnect_probability', 0.1)

            for n in neighbors:
                if neighborhood_data.states[n] > 0:  # Only consider active neighbors
                    neighbor_connections = np.sum(neighborhood_data.edge_matrix[n, neighborhood_data.get_neighbor_indices(n)])

                    # Calculate the difference in edge counts
                    edge_count_difference = abs(self_connections - neighbor_connections)

                    # Determine connection/disconnection based on tolerance
                    if edge_count_difference <= edge_match_tolerance * self_connections:
                        if random.random() < connect_probability:
                            new_edges.add(n)
                    elif neighborhood_data.edge_matrix[node_idx, n]:  # Existing edge
                        if random.random() < disconnect_probability:
                            continue  # Skip adding the edge, effectively removing it if it existed
                    
                    if neighborhood_data.edge_matrix[node_idx, n]:
                        new_edges.add(n)

        return new_edges
          
class PreferentialAttachmentRule(Rule):
    """
    Preferential Attachment Rule (Modified): Favors connections to nodes with
    degrees similar to the self node, promoting a more balanced network.
    Includes explicit birth/death and edge addition/removal logic.
    """

    PARAMETER_METADATA = {
            "activation_threshold": {
                "type": float,
                "description": "Threshold for active neighbor ratio to activate a node.",
                "min": 0.0,
                "max": 1.0
            },
            "deactivation_threshold": {
                "type": float,
                "description": "Threshold for active neighbor ratio below which an active node becomes inactive.",
                "min": 0.0,
                "max": 1.0
            },
            "min_connections": {
                "type": int,
                "description": "Minimum connections for a node to be considered active.",
                "min": 0,
                "max": 8
            },
            "max_connections": {
                "type": int,
                "description": "Max connections for a node to be considered active.",
                "min": 0,
                "max": 8
            },
            "connection_preference": {
                "type": str,
                "description": "Preference for connecting to neighbors ('similar', 'lower', 'higher').",
                "allowed_values": ['similar', 'lower', 'higher']
            },
            "similarity_tolerance": {
                "type": int,
                "description": "Tolerance for degree difference when connecting to 'similar' neighbors.",
                "min": 0,
                "max": 8
            },
            "connect_probability": {
                "type": float,
                "description": "Base probability of connecting to a valid neighbor.",
                "min": 0.0,
                "max": 1.0
            },
            "disconnect_probability": {
                "type": float,
                "description": "Probability of disconnecting from a neighbor.",
                "min": 0.0,
                "max": 1.0
            },
            "min_shared_neighbors": {
                "type": int,
                "description": "Minimum number of shared neighbors to maintain an edge.",
                "min": 0,
                "max": 8
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self.params: Dict[str, Any] = {}  # Initialize params as a dictionary

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData, node_positions: Optional[np.ndarray] = None, dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on active neighbor ratio and connectivity"""
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])
        current_state = neighborhood_data.states[node_idx]

        # Get parameter values
        activation_threshold = self.params.get('activation_threshold', 0.3) # ADDED default
        min_connections = self.params.get('min_connections', 1) # ADDED default
        max_connections = self.params.get('max_connections', 8) # ADDED default

        if current_state == -1.0: # Empty cell
            # Birth conditions
            if (active_ratio >= activation_threshold and
                connected_neighbors >= min_connections):
                return 1.0  # Born
            else:
                return -1.0 # Remain empty

        if current_state > 0:  # Active node
            # Survival conditions
            if (active_ratio >= activation_threshold and
                connected_neighbors >= min_connections and
                connected_neighbors <= max_connections):
                return 1.0  # Survive
            else:
                return 0.0  # Die
        else:  # Inactive node
            # Birth conditions
            if (active_ratio >= activation_threshold and
                connected_neighbors >= min_connections):
                return 1.0  # Born
            else:
                return 0.0  # Remain inactive
                
    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData, node_positions: Optional[np.ndarray] = None, dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on preferential attachment (modified)"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        self_connections = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        if neighborhood_data.states[node_idx] > 0:  # Active node
            # Get parameter values
            connection_preference = self.params.get('connection_preference', 'similar') # ADDED default
            similarity_tolerance = self.params.get('similarity_tolerance', 2) # ADDED default
            connect_probability = self.params.get('connect_probability', 0.6) # ADDED default
            disconnect_probability = self.params.get('disconnect_probability', 0.1) # ADDED default
            min_shared_neighbors = self.params.get('min_shared_neighbors', 1) # ADDED default

            for n in neighbors:
                neighbor_connections = np.sum(neighborhood_data.edge_matrix[n, neighborhood_data.get_neighbor_indices(n)])
                    
                # Skip if neighbor is empty
                if neighborhood_data.states[n] == -1:
                    continue

                # Connection logic based on preference
                connect = False
                if connection_preference == 'similar':
                    if abs(self_connections - neighbor_connections) <= similarity_tolerance:
                        connect = True
                elif connection_preference == 'lower':
                    if neighbor_connections < self_connections:
                        connect = True
                elif connection_preference == 'higher':
                    if neighbor_connections > self_connections:
                        connect = True

                if connect and random.random() < connect_probability:
                    new_edges.add(n)

                # Disconnection logic (remove edges if too many connections or not enough shared neighbors)
                if neighborhood_data.edge_matrix[node_idx, n]:  # If edge exists
                    shared_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx] & neighborhood_data.edge_matrix[n])
                    if (self_connections > self.params.get('max_connections', 8) or # ADDED default
                        neighbor_connections > self.params.get('max_connections', 8) or # ADDED default
                        shared_neighbors < min_shared_neighbors):
                        if random.random() < disconnect_probability:
                            continue # Skip adding, effectively removing the edge
                        
                    new_edges.add(n) # keep the edge

        return new_edges

class StateDependentEdgeAngleRule(Rule):
    """
    State-Dependent Edge Angle Rule: Creates connections based on node state and
    relative angles, allowing for multiple angle preferences.
    """

    PARAMETER_METADATA = {
        "activation_threshold": {
            "type": float,
            "description": "Threshold for active neighbor ratio to activate a node.",
            "min": 0.0,
            "max": 1.0
        },
        "active_angle_1": {
            "type": float,
            "description": "Preferred angle 1 for connections when the node is active (in degrees).",
            "min": 0.0,
            "max": 360.0
        },
        "active_tolerance_1": {
            "type": float,
            "description": "Tolerance for angle 1 when the node is active (in degrees).",
            "min": 0.0,
            "max": 180.0
        },
        "active_angle_2": {
            "type": float,
            "description": "Preferred angle 2 for connections when the node is active (in degrees).",
            "min": 0.0,
            "max": 360.0
        },
        "active_tolerance_2": {
            "type": float,
            "description": "Tolerance for angle 2 when the node is active (in degrees).",
            "min": 0.0,
            "max": 180.0
        },
        "inactive_angle_1": {
            "type": float,
            "description": "Preferred angle 1 for connections when the node is inactive (in degrees).",
            "min": 0.0,
            "max": 360.0
        },
        "inactive_tolerance_1": {
            "type": float,
            "description": "Tolerance for angle 1 when the node is inactive (in degrees).",
            "min": 0.0,
            "max": 180.0
        },
        "inactive_angle_2": {
            "type": float,
            "description": "Preferred angle 2 for connections when the node is inactive (in degrees).",
            "min": 0.0,
            "max": 360.0
        },
        "inactive_tolerance_2": {
            "type": float,
            "description": "Tolerance for angle 2 when the node is inactive (in degrees).",
            "min": 0.0,
            "max": 180.0
        },
        "connect_probability": {
            "type": float,
            "description": "Base probability of connecting to a valid neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "disconnect_probability": {
            "type": float,
            "description": "Base probability of disconnecting from a neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on active neighbor ratio."""
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell case (-1.0)
        if current_state == -1.0:
            # Birth logic: activate if enough active neighbors
            if active_ratio > self.get_param('activation_threshold', 0.4):
                return 1.0  # Cell is born
            else:
                return -1.0 # Remain empty

        # Active/inactive cell logic
        activation_threshold = self.get_param('activation_threshold', 0.4)
        if active_ratio >= activation_threshold:
            return 1.0  # Become/remain active
        else:
            return 0.0  # Become/remain inactive

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on neighborhood state matching."""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0 or node_positions is None:  # Only active nodes update edges
            return new_edges

        # Get parameters
        connect_probability = self.get_param('connect_probability', 0.7)
        disconnect_probability = self.get_param('disconnect_probability', 0.1)

        # Get preferred angles and tolerances based on node state
        if neighborhood_data.states[node_idx] > 0:  # Active
            angle1 = self.get_param('active_angle_1', 45.0)
            tolerance1 = self.get_param('active_tolerance_1', 15.0)
            angle2 = self.get_param('active_angle_2', 135.0)
            tolerance2 = self.get_param('active_tolerance_2', 15.0)
        else:  # Inactive
            angle1 = self.get_param('inactive_angle_1', 225.0)
            tolerance1 = self.get_param('inactive_tolerance_1', 15.0)
            angle2 = self.get_param('inactive_angle_2', 315.0)
            tolerance2 = self.get_param('inactive_tolerance_2', 15.0)

        # Convert angles to radians
        angle1_rad = np.radians(angle1)
        angle2_rad = np.radians(angle2)
        tolerance1_rad = np.radians(tolerance1)
        tolerance2_rad = np.radians(tolerance2)

        # Calculate neighbor positions
        center = node_positions[node_idx]
        for n in neighbors:
            if neighborhood_data.states[n] > 0 and node_positions is not None:
                # Calculate angle to neighbor
                direction = node_positions[n] - center
                angle = np.arctan2(direction[1], direction[0])

                # Check if angle is within tolerance of preferred angles
                angle_diff1 = abs(angle - angle1_rad)
                angle_diff2 = abs(angle - angle2_rad)

                # Normalize angle differences to be between 0 and pi
                angle_diff1 = min(angle_diff1, 2*np.pi - angle_diff1)
                angle_diff2 = min(angle_diff2, 2*np.pi - angle_diff2)

                # Connect or disconnect based on angle and probability
                if (angle_diff1 <= tolerance1_rad or angle_diff2 <= tolerance2_rad) and random.random() < connect_probability:
                    new_edges.add(n)
                elif neighborhood_data.edge_matrix[node_idx, n] and random.random() < disconnect_probability:
                    continue  # Skip adding the edge, effectively removing it if it existed
                
                if neighborhood_data.edge_matrix[node_idx, n]:
                    new_edges.add(n)

        return new_edges
        
class NeighborhoodStateMatchingRule(Rule):
    """
    Neighborhood State Matching Rule: Connects nodes based on the similarity
    of the average states of their neighborhoods.
    """

    PARAMETER_METADATA = {
        "activation_threshold": {
            "type": float,
            "description": "Threshold for active neighbor ratio to activate a node.",
            "min": 0.0,
            "max": 1.0
        },
        "state_match_threshold": {
            "type": float,
            "description": "Threshold for the difference in average neighbor states.  If the difference is BELOW this, connection is favored.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_probability": {
            "type": float,
            "description": "Base probability of connecting to a valid neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "disconnect_probability": {
            "type": float,
            "description": "Base probability of disconnecting from a neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on a weighted combination of metrics."""
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell case (-1.0)
        if current_state == -1.0:
            # Use a simple birth condition for now
            if self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data) > 0.5:
                return 1.0  # Cell is born
            return -1.0

        # Get metrics
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        avg_neighbor_degree = self.get_metric(AverageNeighborDegree, node_idx, neighborhood_data)
        clustering_coefficient = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)

        # Get parameter values
        activation_threshold = self.get_param('activation_threshold', 0.6)
        active_ratio_weight = self.get_param('active_neighbor_ratio_weight', 0.4)
        avg_degree_weight = self.get_param('average_neighbor_degree_weight', 0.3)
        clustering_weight = self.get_param('clustering_coefficient_weight', 0.3)

        # Calculate weighted sum
        weighted_sum = (active_ratio * active_ratio_weight +
                        avg_neighbor_degree * avg_degree_weight +
                        clustering_coefficient * clustering_weight)

        # Normalize by the sum of weights (to keep it between 0 and 1)
        total_weight = active_ratio_weight + avg_degree_weight + clustering_weight
        if total_weight > 0:
            normalized_sum = weighted_sum / total_weight
        else:
            normalized_sum = 0.0

        # Determine new state
        if current_state > 0:  # Active node
            if normalized_sum >= activation_threshold:
                return 1.0  # Survive
            else:
                return 0.0  # Die
        else:  # Inactive node
            if normalized_sum >= activation_threshold:
                return 1.0  # Born
            else:
                return 0.0  # Remain inactive

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on a weighted combination of metrics."""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:  # Only active nodes update edges
            return new_edges

        # Get parameter values
        connect_threshold = self.get_param('connect_threshold', 0.7)
        disconnect_threshold = self.get_param('disconnect_threshold', 0.3)
        self_edge_density_weight = self.get_param('self_edge_density_weight', 0.4)
        neighbor_edge_density_weight = self.get_param('neighbor_edge_density_weight', 0.4)
        shared_neighbors_weight = self.get_param('shared_neighbors_weight', 0.2)

        self_edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)

        for n in neighbors:
            if neighborhood_data.states[n] == -1:
                continue #skip if empty
            if neighborhood_data.states[n] > 0:  # Only consider active neighbors for connection
                neighbor_edge_density = self.get_metric(EdgeDensity, n, neighborhood_data)
                shared_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx] & neighborhood_data.edge_matrix[n])

                # Calculate weighted sum for edge connection
                edge_score = (self_edge_density * self_edge_density_weight +
                            neighbor_edge_density * neighbor_edge_density_weight +
                            shared_neighbors * shared_neighbors_weight)
                
                # Normalize
                total_weight = self_edge_density_weight + neighbor_edge_density_weight + shared_neighbors_weight
                if total_weight > 0:
                    edge_score /= total_weight

                if edge_score >= connect_threshold:
                    new_edges.add(n)
                
                # Disconnect logic (only if the edge exists)
                if neighborhood_data.edge_matrix[node_idx, n]:
                    if edge_score < disconnect_threshold:
                        # Don't add to new_edges, effectively removing it
                        continue

                # Maintain existing edges
                if neighborhood_data.edge_matrix[node_idx, n]:
                    new_edges.add(n)

        return new_edges
    
class CombinedMetricRule(Rule):
    """
    Combined Metric Rule: Uses a weighted combination of local and neighbor metrics
    to determine node states and edge connections.
    """

    PARAMETER_METADATA = {
        "active_neighbor_ratio_weight": {
            "type": float,
            "description": "Weight of active neighbor ratio in state update.",
            "min": 0.0,
            "max": 1.0
        },
        "average_neighbor_degree_weight": {
            "type": float,
            "description": "Weight of average neighbor degree in state update.",
            "min": 0.0,
            "max": 1.0
        },
        "clustering_coefficient_weight": {
            "type": float,
            "description": "Weight of local clustering coefficient in state update.",
            "min": 0.0,
            "max": 1.0
        },
        "activation_threshold": {
            "type": float,
            "description": "Threshold for combined metrics to activate a node.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_threshold": {
            "type": float,
            "description": "Threshold for combined metrics to add an edge.",
            "min": 0.0,
            "max": 1.0
        },
        "disconnect_threshold": {
            "type": float,
            "description": "Threshold for combined metrics to remove an edge.",
            "min": 0.0,
            "max": 1.0
        },
        "self_edge_density_weight": {
            "type": float,
            "description": "Weight of self node's edge density in edge update.",
            "min": 0.0,
            "max": 1.0
        },
        "neighbor_edge_density_weight": {
            "type": float,
            "description": "Weight of neighbor node's edge density in edge update.",
            "min": 0.0,
            "max": 1.0
        },
        "shared_neighbors_weight": {
            "type": float,
            "description": "Weight of shared neighbors count in edge update.",
            "min": 0.0,
            "max": 1.0
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on a weighted combination of metrics."""
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell
        if current_state == -1.0:
            # Add birth logic here, possibly based on proximity to existing patterns
            return -1.0  # Remain empty for now

        # Get metrics
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        avg_neighbor_degree = self.get_metric(AverageNeighborDegree, node_idx, neighborhood_data)
        clustering_coefficient = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)

        # Get parameter values
        active_ratio_weight = self.get_param('active_neighbor_ratio_weight', 0.4)
        avg_degree_weight = self.get_param('average_neighbor_degree_weight', 0.3)
        clustering_weight = self.get_param('clustering_coefficient_weight', 0.3)
        activation_threshold = self.get_param('activation_threshold', 0.6)

        # Calculate weighted sum
        weighted_sum = (active_ratio * active_ratio_weight +
                        avg_neighbor_degree * avg_degree_weight +
                        clustering_coefficient * clustering_weight)

        # Normalize by the sum of weights (to keep it between 0 and 1)
        total_weight = active_ratio_weight + avg_degree_weight + clustering_weight
        if total_weight > 0:
            normalized_sum = weighted_sum / total_weight
        else:
            normalized_sum = 0.0

        # Determine new state
        if current_state > 0:  # Active node
            if normalized_sum >= activation_threshold:
                return 1.0  # Survive
            else:
                return 0.0  # Die
        else:  # Inactive node
            if normalized_sum >= activation_threshold:
                return 1.0  # Born
            else:
                return 0.0  # Remain inactive

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on a weighted combination of metrics."""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:  # Only active nodes update edges
            return new_edges

        # Get parameter values
        connect_threshold = self.get_param('connect_threshold', 0.7)
        disconnect_threshold = self.get_param('disconnect_threshold', 0.3)
        self_edge_density_weight = self.get_param('self_edge_density_weight', 0.4)
        neighbor_edge_density_weight = self.get_param('neighbor_edge_density_weight', 0.4)
        shared_neighbors_weight = self.get_param('shared_neighbors_weight', 0.2)

        self_edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)

        for n in neighbors:
            if neighborhood_data.states[n] == -1:
                continue #skip if empty
            if neighborhood_data.states[n] > 0:  # Only consider active neighbors for connection
                neighbor_edge_density = self.get_metric(EdgeDensity, n, neighborhood_data)
                shared_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx] & neighborhood_data.edge_matrix[n])

                # Calculate weighted sum for edge connection
                edge_score = (self_edge_density * self_edge_density_weight +
                            neighbor_edge_density * neighbor_edge_density_weight +
                            shared_neighbors * shared_neighbors_weight)
                
                # Normalize
                total_weight = self_edge_density_weight + neighbor_edge_density_weight + shared_neighbors_weight
                if total_weight > 0:
                    edge_score /= total_weight

                if edge_score >= connect_threshold:
                    new_edges.add(n)
                
                # Disconnect logic (only if the edge exists)
                if neighborhood_data.edge_matrix[node_idx, n]:
                    if edge_score < disconnect_threshold:
                        # Don't add to new_edges, effectively removing it
                        continue

                # Maintain existing edges
                if neighborhood_data.edge_matrix[node_idx, n]:
                    new_edges.add(n)

        return new_edges

@dataclass
class PredictableRuleMetadata:
    """Metadata for rules in the library"""
    name: str
    type: str                  
    position: int              
    category: str              
    author: str               
    url: str                  
    email: str                
    date_created: str         
    date_modified: str        
    version: str              
    description: str          
    tags: List[str]           
    dimension_compatibility: List[str]
    neighborhood_compatibility: List[str]
    parent_rule: Optional[str]
    rating: Optional[int] = None  # New field for rating
    notes: Optional[str] = None   # New field for notes
    controller: Optional['SimulationController'] = None # Add controller attribute
    neighborhood_type: Optional[NeighborhoodType] = None # ADDED

class PredictableRule(Rule):
    """
    Predictable Rule: Creates random nodes near existing nodes and connects them.
    """

    PARAMETER_METADATA = {
        "birth_probability": {
            "type": float,
            "description": "Probability of creating a new node near an existing node.",
            "min": 0.0,
            "max": 1.0
        },
        "max_connections": {
            "type": int,
            "description": "Maximum number of connections a node can have.",
            "min": 0,
            "max": 8
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on proximity to existing nodes"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0:
            # Check if there are any active neighbors
            active_neighbors = np.sum(neighborhood_data.states[neighbors] > 0)
            if active_neighbors > 0 and random.random() < self.get_param('birth_probability', 0.1):
                return 1.0  # Create a new node
            else:
                return -1.0  # Remain empty
        else:
            # If a node has more than max_connections, it dies
            max_connections = self.get_param('max_connections', 4)
            connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])
            if connected_neighbors > max_connections:
                return 0.0  # Node dies
            else:
                return 1.0  # Node survives

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates to connect to all active neighbors"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] > 0:
            for n in neighbors:
                if neighborhood_data.states[n] > 0:
                    new_edges.add(n)  # Connect to all active neighbors
        
        return new_edges
     
################################################
#                 RULE LIBRARY                 #
################################################

class RuleLibrary:
    """Comprehensive library of balanced cellular automaton rules"""

    @staticmethod
    def create_test_rule(metadata: RuleMetadata) -> Rule:
        """TestRule: Toggles node states and edges for testing"""
        return TestRule(metadata)
   
    @staticmethod
    def create_pure_rule_table_rule(metadata: RuleMetadata) -> Rule:
        """Pure Rule Table Rule: Uses a predefined rule table for state transitions."""
        return PureRuleTableRule(metadata)
     
    @staticmethod
    def create_edge_count_matching_rule(metadata: RuleMetadata) -> Rule:
        """EdgeCountMatchingRule: Connects to neighbors with similar edge counts."""
        return EdgeCountMatchingRule(metadata)

    @staticmethod
    def create_state_dependent_edge_angle_rule(metadata: RuleMetadata) -> Rule:
        """StateDependentEdgeAngleRule: Connects based on state and relative angles."""
        return StateDependentEdgeAngleRule(metadata)

    @staticmethod
    def create_preferential_attachment_rule(metadata: RuleMetadata) -> Rule:
        """PreferentialAttachmentRule: Favors connections to nodes with similar degrees."""
        return PreferentialAttachmentRule(metadata)
    
    @staticmethod
    def create_neighborhood_state_matching_rule(metadata: RuleMetadata) -> Rule:
        """NeighborhoodStateMatchingRule: Connects based on similarity of neighborhood states."""
        return NeighborhoodStateMatchingRule(metadata)

    @staticmethod
    def create_combined_metric_rule(metadata: RuleMetadata) -> Rule:
        """CombinedMetricRule: Uses a weighted combination of metrics for edge updates."""
        return CombinedMetricRule(metadata)
    
    @staticmethod
    def create_majority_rule(metadata: RuleMetadata) -> Rule:
        """
        Majority Rule: Classic rule with balanced thresholds
        """
        return MajorityRule(metadata)

    @staticmethod
    def create_connect_life(metadata: RuleMetadata) -> Rule:
        """ConnectLife: Creates connected structures with dynamic edge formation"""
        return ConnectLife(metadata)

    @staticmethod
    def create_life_and_death(metadata: RuleMetadata) -> Rule:
        """LifeAndDeath: Advanced rule with detailed birth/death dynamics"""
        return LifeAndDeath(metadata)

    @staticmethod
    def create_neighbor_connections(metadata: RuleMetadata) -> Rule:
        """Neighbor Connections Rule: Determines connections based on the states and connection counts of both the self node and its neighbors"""
        return NeighborConnections(metadata)
    
    @staticmethod
    def create_network_life(metadata: RuleMetadata) -> Rule:
        """NetworkLife: A Life variant based on connection counts and neighbor states"""
        return NetworkLife(metadata)
     
    @staticmethod
    def create_adaptive_network_rule(metadata: RuleMetadata) -> Rule:
        """AdaptiveNetworkRule: Creates adaptive networks based on local and global conditions"""
        return AdaptiveNetworkRule(metadata)

    @staticmethod
    def create_stable_polygons(metadata: RuleMetadata) -> Rule:
        """StablePolygons: Creates stable polygonal structures with slow growth"""
        return StablePolygons(metadata)

    @staticmethod
    def create_geometric_angle(metadata: RuleMetadata) -> Rule:
        """GeometricAngle: Creates geometric patterns based on neighbor angles"""
        return GeometricAngle(metadata)

    @staticmethod
    def create_symmetry_rule(metadata: RuleMetadata) -> Rule:
        """SymmetryRule: Creates and maintains symmetric patterns"""
        return SymmetryRule(metadata)

    @staticmethod
    def create_fractal_rule(metadata: RuleMetadata) -> Rule:
        """FractalRule: Creates self-similar patterns at different scales"""
        return FractalRule(metadata)

    @staticmethod
    def create_modular_rule(metadata: RuleMetadata) -> Rule:
        """ModularRule: Creates distinct functional modules"""
        return ModularRule(metadata)

    @staticmethod
    def create_flow_rule(metadata: RuleMetadata) -> Rule:
        """FlowRule: Creates structures optimized for directional flow"""
        return FlowRule(metadata)

    @staticmethod
    def create_competitive_rule(metadata: RuleMetadata) -> Rule:
        """CompetitiveRule: Creates structures based on competition for resources"""
        return CompetitiveRule(metadata)

    @staticmethod
    def create_adaptive_memory_rule(metadata: RuleMetadata) -> Rule:
        """AdaptiveMemoryRule: Creates structures that can remember and recreate previous patterns"""
        return AdaptiveMemoryRule(metadata)

    @staticmethod
    def create_artificial_life_rule(metadata: RuleMetadata) -> Rule:
        """ArtificialLifeRule: Creates evolving organisms with metabolism, reproduction, and adaptation"""
        return ArtificialLifeRule(metadata)

    @staticmethod
    def create_predictable_rule(metadata: RuleMetadata) -> Rule:
        """PredictableRule: Creates random nodes near existing nodes and connects them"""
        return PredictableRule(metadata)

    # Dictionary mapping rule names to their factory functions
    RULES = {
        # Basic Rules
        "Test Rule": create_test_rule, 
        "Predictable Rule": create_predictable_rule,
        "Pure Rule Table Rule": create_pure_rule_table_rule,
        "Majority Rule": create_majority_rule,
        "ConnectLife": create_connect_life,
        "Life and Death": create_life_and_death,
        
        # Network Rules
        "Modular Rule": create_modular_rule,
        "Flow Rule": create_flow_rule,
        "Competitive Rule": create_competitive_rule,
        "Adaptive Memory Rule": create_adaptive_memory_rule,
        "Edge Count Matching Rule": create_edge_count_matching_rule,
        "State-Dependent Edge Angle Rule": create_state_dependent_edge_angle_rule,
        "Neighborhood State Matching Rule": create_neighborhood_state_matching_rule,
        "Combined Metric Rule": create_combined_metric_rule,
        
         # Geometric Rules
        "Stable Polygons": create_stable_polygons,
        "Geometric Angle": create_geometric_angle,
        "Symmetry Rule": create_symmetry_rule,
        "Fractal Rule": create_fractal_rule,
        
        # Advanced Rules
        "Network Life": create_network_life,
        "Neighbor Connections And States": create_neighbor_connections,
        "Adaptive Network Rule": create_adaptive_network_rule,
        "Preferential Attachment Rule": create_preferential_attachment_rule, # Added rule
        
        # Biological Rules
        "Artificial Life Rule": create_artificial_life_rule,
    }
    
    @classmethod
    def get_rule_names(cls) -> List[str]:
        """Get list of available rule names"""
        return list(cls.RULES.keys())

    @classmethod
    def get_rule_categories(cls) -> Dict[str, List[str]]:
        """Get dictionary of rule categories and their rules"""
        categories: Dict[str, List[str]] = defaultdict(list)
        for rule_name in cls.get_rule_names():
            try:
                # Get the factory function
                factory_function = cls.RULES[rule_name]
                # Create a temporary instance of the rule to access its metadata
                metadata = RuleLibraryManager.get_rule(rule_name)
                category = metadata['category']
                categories[category].append(rule_name)
            except:
                pass
        return dict(categories)

    @classmethod
    def get_rules_in_category(cls, category: str) -> List[str]:
        """Get list of rules in a specific category"""
        return [rule for rule in cls.get_rule_names() if cls.get_rule_category(rule) == category]

    @classmethod
    def create_rule(cls, rule_name: str, metadata: 'RuleMetadata') -> 'Rule':
        """Create rule instance by name"""
        if rule_name not in cls.RULES:
            raise ValueError(f"Unknown rule: {rule_name}")
        
        # Get the factory function
        factory_function = cls.RULES[rule_name]
        
        # Get rule data including parameters
        rule_data = RuleLibraryManager.get_rule(rule_name)
        
        # Create the rule with the metadata
        rule = factory_function(metadata)
        
        # Set the parameters if they exist in the rule data
        if 'params' in rule_data:
            logger.debug(f"Loading parameters for {rule_name}: {rule_data['params']}")
            rule.params = rule_data['params']
        else:
            logger.warning(f"No parameters found for rule {rule_name} in library")
            rule.params = {}
            
        return rule

    @classmethod
    def get_rule_description(cls, rule_name: str) -> str:
        """Get description of rule"""
        try:
            # Get the factory function
            # Create a temporary instance of the rule to access its metadata
            metadata = RuleLibraryManager.get_rule(rule_name)
            return metadata['description']
        except Exception:
            return "No description available"

    @classmethod
    def get_rule_category(cls, rule_name: str) -> str:
        """Get the category of a rule"""
        try:
            # Get the factory function
            # Create a temporary instance of the rule to access its metadata
            metadata = RuleLibraryManager.get_rule(rule_name)
            return metadata['category']
        except Exception:
            return "Unknown"

    @classmethod
    def get_rule_compatibility(cls, rule_name: str) -> List[str]:
        """Get the dimension compatibility of a rule"""
        try:
            # Get the factory function
            # Create a temporary instance of the rule to access its metadata
            metadata = RuleLibraryManager.get_rule(rule_name)
            return metadata['dimension_compatibility']
        except Exception:
            return []

    @classmethod
    def get_rule_neighborhood_compatibility(cls, rule_name: str) -> List[str]:
        """Get the neighborhood compatibility of a rule"""
        try:
            # Get the factory function
            # Create a temporary instance of the rule to access its metadata
            metadata = RuleLibraryManager.get_rule(rule_name)
            return metadata.get('neighborhood_compatibility', [])
        except Exception:
            return []

    @classmethod
    def list_rules_with_descriptions(cls) -> Dict[str, str]:
        """Get dictionary of all rules and their descriptions"""
        return {name: cls.get_rule_description(name) for name in cls.RULES}

    @classmethod
    def get_similar_rules(cls, rule_name: str, num_similar: int = 3) -> List[str]:
        """Get list of similar rules based on category"""
        if rule_name not in cls.RULES:
            raise ValueError(f"Unknown rule: {rule_name}")
            
        # Find category containing the rule
        category = cls.get_rule_category(rule_name)
        
        if category:
            # Get other rules from same category
            similar_rules = [r for r in cls.get_rules_in_category(category) 
                           if r != rule_name]
            return similar_rules[:num_similar]
        
        return []

    @classmethod
    def validate_rule_parameters(cls, rule_name: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for a rule"""
        try:
            if rule_name not in cls.RULES:
                raise ValueError(f"Unknown rule: {rule_name}")
            
            # Get rule data from library
            rule_data = RuleLibraryManager.get_rule(rule_name)
            
            # Get the factory function
            factory_function = cls.RULES[rule_name]
            
            # Create a temporary instance of the rule to access its metadata
            metadata = RuleMetadata(**rule_data)
            rule = factory_function(metadata)
            
            # Check all required parameters are present
            for param in rule.params:
                if param not in parameters:
                    return False
                
            # Check parameter types match defaults
            for param, value in parameters.items():
                if param in rule.params:
                    expected_type = type(rule.params[param] ['type'])
                    if not isinstance(value, expected_type):
                        return False
                    
            return True
        except Exception as e:
            logger.error(f"Error validating rule parameters: {e}")
            return False
        
class RuleLibraryInfo:
    """Global rule information and organization"""
    
    # Dictionary mapping rule names to their factory functions
    RULES = {
        # Basic Rules
        "Test Rule": RuleLibrary.create_test_rule, 
        "Predictable Rule": RuleLibrary.create_predictable_rule,
        "Pure Rule Table Rule": RuleLibrary.create_pure_rule_table_rule,
        "Majority Rule": RuleLibrary.create_majority_rule,
        "ConnectLife": RuleLibrary.create_connect_life,
        "Life and Death": RuleLibrary.create_life_and_death,
        
        # Advanced Rules
        "Network Life": RuleLibrary.create_network_life,
        "Neighbor Connections And States": RuleLibrary.create_neighbor_connections,
        "Adaptive Network Rule": RuleLibrary.create_adaptive_network_rule,
        
        # Geometric Rules
        "Stable Polygons": RuleLibrary.create_stable_polygons,
        "Geometric Angle": RuleLibrary.create_geometric_angle,
        "Symmetry Rule": RuleLibrary.create_symmetry_rule,
        "Fractal Rule": RuleLibrary.create_fractal_rule,
        
        # Network Rules
        "Modular Rule": RuleLibrary.create_modular_rule,
        "Flow Rule": RuleLibrary.create_flow_rule,
        "Competitive Rule": RuleLibrary.create_competitive_rule,
        "Adaptive Memory Rule": RuleLibrary.create_adaptive_memory_rule,
        
        # Biological Rules
        "Artificial Life Rule": RuleLibrary.create_artificial_life_rule,
    }

    @classmethod
    def get_rule_names(cls) -> List[str]:
        """Get list of available rule names"""
        return list(cls.RULES.keys())

    @classmethod
    def get_rule_categories(cls) -> Dict[str, List[str]]:
        """Get dictionary of rule categories and their rules"""
        categories: Dict[str, List[str]] = defaultdict(list)
        for rule_name in cls.get_rule_names():
            try:
                # Get rule data from library
                rule_data = RuleLibraryManager.get_rule(rule_name)
                category = rule_data['category']
                categories[category].append(rule_name)
            except:
                pass
        return dict(categories)

    @classmethod
    def get_rules_in_category(cls, category: str) -> List[str]:
        """Get list of rules in a specific category"""
        return [rule for rule in cls.get_rule_names() if cls.get_rule_category(rule) == category]

    @classmethod
    def get_rule_description(cls, rule_name: str) -> str:
        """Get description of rule"""
        try:
            # Get rule data from library
            rule_data = RuleLibraryManager.get_rule(rule_name)
            return rule_data['description']
        except Exception:
            return "No description available"

    @classmethod
    def get_rule_category(cls, rule_name: str) -> str:
        """Get the category of a rule"""
        try:
            # Get rule data from library
            rule_data = RuleLibraryManager.get_rule(rule_name)
            return rule_data['category']
        except Exception:
            return "Unknown"

    @classmethod
    def get_rule_compatibility(cls, rule_name: str) -> List[str]:
        """Get the dimension compatibility of a rule"""
        try:
            # Get rule data from library
            rule_data = RuleLibraryManager.get_rule(rule_name)
            return rule_data['dimension_compatibility']
        except Exception:
            return []

    @classmethod
    def get_rule_neighborhood_compatibility(cls, rule_name: str) -> List[str]:
        """Get the neighborhood compatibility of a rule"""
        try:
            # Get rule data from library
            rule_data = RuleLibraryManager.get_rule(rule_name)
            return rule_data.get('neighborhood_compatibility', [])
        except Exception:
            return []

    @classmethod
    def list_rules_with_descriptions(cls) -> Dict[str, str]:
        """Get dictionary of all rules and their descriptions"""
        return {name: cls.get_rule_description(name) for name in cls.RULES}

    @classmethod
    def get_similar_rules(cls, rule_name: str, num_similar: int = 3) -> List[str]:
        """Get list of similar rules based on category"""
        if rule_name not in cls.RULES:
            raise ValueError(f"Unknown rule: {rule_name}")
            
        # Find category containing the rule
        category = cls.get_rule_category(rule_name)
        
        if category:
            # Get other rules from same category
            similar_rules = [r for r in cls.get_rules_in_category(category) 
                           if r != rule_name]
            return similar_rules[:num_similar]
        
        return []

    @classmethod
    def validate_rule_parameters(cls, rule_name: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for a rule"""
        try:
            if rule_name not in cls.RULES:
                raise ValueError(f"Unknown rule: {rule_name}")
            
            # Get rule data from library
            rule_data = RuleLibraryManager.get_rule(rule_name)
            
            # Get the factory function
            factory_function = cls.RULES[rule_name]
            
            # Create a temporary instance of the rule to access its metadata
            metadata = RuleMetadata(**rule_data)
            rule = factory_function(metadata)
            
            # Check all required parameters are present
            for param in rule.params:
                if param not in parameters:
                    return False
                
            # Check parameter types match defaults
            for param, value in parameters.items():
                if param in rule.params:
                    expected_type = type(rule.params[param]['type'])
                    if not isinstance(value, expected_type):
                        return False
                    
            return True
        except Exception as e:
            logger.error(f"Error validating rule parameters: {e}")
            return False
               
class RuleLibraryManager:
    """Manages rule library loading, saving, and updates"""
    _instance: Optional['RuleLibraryManager'] = None
    _rules_cache: Dict[str, Dict[str, Any]] = {}  # Add cache to prevent recursion
    _initialized: bool = False

    def __init__(self, library_path: Optional[str] = None):
        if not RuleLibraryManager._initialized:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(script_dir, 'rules.json')
            self.library_path = library_path or default_path
            print(f"Rule library path: {self.library_path}")  # ADDED for debugging
            self.rules: Dict[str, Dict[str, Any]] = {}
            self.rule_metadata: Dict[str, RuleMetadata] = {}

            # Load and validate the library
            self.load_library()
            
            RuleLibraryManager._initialized = True
            
    @classmethod
    def get_instance(cls) -> 'RuleLibraryManager':
        if cls._instance is None:
            cls._instance = RuleLibraryManager()
        return cls._instance

    def save_rule(self, rule_name: str, rule_data: Dict[str, Any]):
        """Save or update a rule in the library"""
        try:
            # Validate rule data
            if 'name' not in rule_data:
                raise ValueError("Rule data must contain a name")
            if 'type' not in rule_data:
                raise ValueError("Rule data must contain a type")
            
            # Check if rule already exists
            if rule_name in self.rules:
                logger.info(f"Updating existing rule: {rule_name}")
                self.rules[rule_name] = rule_data
            else:
                logger.info(f"Adding new rule: {rule_name}")
                self.rules[rule_name] = rule_data
            
            # Save the updated library
            self.save_library()
            
            # Update the cache
            RuleLibraryManager._rules_cache[rule_name] = rule_data
            
            logger.info(f"Rule '{rule_name}' saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving rule {rule_name}: {e}")
            raise

    def delete_rule(self, rule_name: str):
        """Delete a rule from the library"""
        try:
            if rule_name not in self.rules:
                raise ValueError(f"Rule '{rule_name}' not found in library")
            
            del self.rules[rule_name]
            
            # Save the updated library
            self.save_library()
            
            # Remove from cache
            if rule_name in RuleLibraryManager._rules_cache:
                del RuleLibraryManager._rules_cache[rule_name]
            
            logger.info(f"Rule '{rule_name}' deleted successfully")
            
        except Exception as e:
            logger.error(f"Error deleting rule {rule_name}: {e}")
            raise

    @classmethod
    def get_rule(cls, rule_name: str) -> Dict[str, Any]:
        """Get rule data from the library cache"""
        try:
            # First check the cache
            if rule_name in cls._rules_cache:
                # Return a DEEP copy to avoid modifying the cached data
                rule_data = copy.deepcopy(cls._rules_cache[rule_name])
                
                # Ensure 'params' key exists, even if it's empty
                if 'params' not in rule_data:
                    rule_data['params'] = {}
                
                logger.debug(f"Rule data loaded for {rule_name} from cache: {rule_data}")
                return rule_data

            # If not in cache, load from instance
            instance = cls.get_instance()
            if rule_name in instance.rules:
                # Get a DEEP copy of the rule data
                rule_data = copy.deepcopy(instance.rules[rule_name])
                
                # Ensure 'params' key exists, even if it's empty
                if 'params' not in rule_data:
                    rule_data['params'] = {}
                
                # Update cache and return a DEEP copy
                cls._rules_cache[rule_name] = rule_data
                logger.debug(f"Rule data loaded for {rule_name} from instance: {rule_data}")
                return rule_data
            
            logger.error(f"Rule '{rule_name}' not found in instance.rules. Available rules: {list(instance.rules.keys())}")
            raise ValueError(f"Rule '{rule_name}' not found in library")

        except Exception as e:
            logger.error(f"Error getting rule {rule_name}: {e}")
            raise 
                             
    def load_library(self):
        """Load rule library from JSON and auto-fix metadata"""
        try:
            logger.debug(f"Attempting to load rule library from {self.library_path}")
            try:
                with open(self.library_path, 'r') as f:
                    library_data = json.load(f)
            except FileNotFoundError:
                logger.error(f"Rule library file not found at {self.library_path}")
                self.rules = {}
                return
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON in rule library: {e}")
                self.rules = {}
                return

            if 'rules' in library_data:
                # First validate and fix the library
                logger.info("Starting library validation...")
                is_valid, modified = self.validate_library_metadata(library_data) # CHANGED - get modified flag
                if not is_valid:
                    logger.error("Library validation failed")
                    return

                # Create backup only if modifications were made
                if modified: # CHANGED - only backup if modified
                    backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = os.path.join(APP_PATHS['config'], f"rules_backup_{backup_timestamp}.json")
                    with open(backup_path, 'w') as f:
                        json.dump(library_data, f, indent=2)
                    logger.info(f"Created backup of rule library at {backup_path}")

                # Now process the validated/fixed library
                processed_rules = []

                for rule in library_data['rules']:
                    rule_data_copy = rule.copy()
                    params = rule_data_copy.pop('params', {})

                    # Add tiebreaker_type if missing
                    if 'tiebreaker_type' not in params:
                        params['tiebreaker_type'] = 'RANDOM'
                        modified = True
                        logger.debug(f"Added missing tiebreaker_type to rule {rule['name']}")

                    rule_data_copy['params'] = params
                    processed_rules.append(rule_data_copy)

                # Update rules
                self.rules = {rule['name']: rule for rule in processed_rules}
                
                # Save if modifications were made
                if modified:
                    self.save_library()
                    logger.info("Saved updated rule library with fixes")

                logger.debug(f"Successfully loaded {len(self.rules)} rules from library")
                logger.info("Rule library metadata validation successful.")

        except Exception as e:
            logger.error(f"Error loading rule library: {e}")
            raise
                                                                                                                 
    def validate_library_metadata(self, library_data) -> Tuple[bool, bool]:
        """Validates and automatically fixes rule metadata and parameters in the library"""
        try:
            logger.debug("========= ENTERING validate_library_metadata =========")

            # Initialize tracking variables
            is_valid_library = True
            modified = False
            
            # Define required metadata fields and their default values
            required_metadata = {
                'name': None,  # Must be provided
                'type': None,  # Must be provided
                'category': "Basic",  # Must be provided
                'author': GlobalSettings.Defaults.DEFAULT_AUTHOR,
                'url': GlobalSettings.Defaults.DEFAULT_URL,
                'email': GlobalSettings.Defaults.DEFAULT_EMAIL,
                'date_created': datetime.now().strftime("%Y-%m-%d"),
                'date_modified': datetime.now().strftime("%Y-%m-%d"),
                'version': "1.0",
                'description': "A cellular automata rule.",  # Must be provided
                'tags': [],
                'dimension_compatibility': ["TWO_D", "THREE_D"],
                'neighborhood_compatibility': None,
                'parent_rule': None,
                'position': 1,
                'rating': None,
                'notes': None
            }

            # Add detailed logging of each rule's current state
            for rule in library_data['rules']:
                rule_name = rule.get('name', 'Unknown Rule')
                logger.debug(f"\nInitial state of Rule {rule_name}:")
                logger.debug(f"Present metadata fields: {set(rule.keys())}")
                logger.debug(f"Missing metadata fields: {set(required_metadata.keys()) - set(rule.keys())}")
                if 'params' in rule:
                    logger.debug(f"Present parameters: {set(rule['params'].keys())}")
                else:
                    logger.debug("No parameters dictionary found")

            # Process each rule
            updated_rules = []
            for rule_idx, rule_data in enumerate(library_data['rules']):
                rule_name = rule_data.get('name', f"Unknown Rule {rule_idx}")
                rule_type = rule_data.get('type', 'Unknown')
                
                logger.debug(f"\nValidating Rule {rule_idx + 1}: {rule_name} (Type: {rule_type})")
                
                fixed_metadata = rule_data.copy()
                rule_modified = False

                # Check each required metadata field explicitly
                for field, default_value in required_metadata.items():
                    # First check if field exists
                    if field not in fixed_metadata:
                        logger.warning(f"Rule {rule_name}: Missing required field '{field}'")
                        if default_value is None and field in ['name', 'type', 'category', 'description']:
                            logger.error(f"Rule {rule_name}: Required field '{field}' is missing and has no default")
                            is_valid_library = False
                            continue
                        fixed_metadata[field] = default_value
                        rule_modified = True
                        modified = True
                        logger.info(f"Added missing field '{field}' with default value: {default_value}")
                        continue

                    # Then check field value
                    field_value = fixed_metadata[field]
                    if field_value is None and default_value is not None:
                        logger.warning(f"Rule {rule_name}: Field '{field}' is None")
                        fixed_metadata[field] = default_value
                        rule_modified = True
                        modified = True
                        logger.info(f"Fixed None field '{field}' with default value: {default_value}")
                    elif field in ['url', 'author', 'email']:
                        if not isinstance(field_value, str) or field_value.strip() == '':
                            logger.warning(f"Rule {rule_name}: Empty or invalid field '{field}'")
                            fixed_metadata[field] = default_value
                            rule_modified = True
                            modified = True
                            logger.info(f"Fixed empty/invalid field '{field}' with default value: {default_value}")

                # Ensure params exists
                if 'params' not in fixed_metadata:
                    logger.warning(f"Rule {rule_name}: Missing params dictionary")
                    fixed_metadata['params'] = {}
                    rule_modified = True
                    modified = True

                # Get the rule class to check its parameters
                try:
                    rule_class = globals()[rule_type]
                    # Get all class variables that are parameter definitions
                    expected_params = {}
                    for name, value in vars(rule_class).items():
                        if isinstance(value, dict) and 'type' in value:
                            param_type = value['type']
                            if param_type == float:
                                default = 0.0
                            elif param_type == int:
                                default = 0
                            elif param_type == bool:
                                default = False
                            elif param_type == str:
                                default = ""
                            elif param_type == list:
                                default = []
                            elif param_type == dict:
                                default = {}
                            else:
                                default = None
                            expected_params[name] = {'type': param_type, 'default': default}
                            
                            # Log expected parameters
                            logger.debug(f"Expected parameter '{name}' of type {param_type} with default {default}")

                    # Check each expected parameter explicitly
                    for param_name, param_info in expected_params.items():
                        if param_name not in fixed_metadata['params']:
                            logger.warning(f"Rule {rule_name}: Missing parameter '{param_name}'")
                            fixed_metadata['params'] [param_name] = param_info['default']
                            rule_modified = True
                            modified = True
                            logger.info(f"Added missing parameter '{param_name}' with default value: {param_info['default']}")
                        elif fixed_metadata['params'] [param_name] is None:
                            logger.warning(f"Rule {rule_name}: Parameter '{param_name}' is None")
                            fixed_metadata['params'] [param_name] = param_info['default']
                            rule_modified = True
                            modified = True
                            logger.info(f"Fixed None parameter '{param_name}' with default value: {param_info['default']}")

                except KeyError as e:
                    logger.error(f"Could not find rule class for type {rule_type}: {e}")
                    is_valid_library = False
                    continue

                # Add tiebreaker_type if missing
                if 'tiebreaker_type' not in fixed_metadata['params']:
                    logger.warning(f"Rule {rule_name}: Missing tiebreaker_type parameter")
                    fixed_metadata['params'] ['tiebreaker_type'] = 'RANDOM'
                    rule_modified = True
                    modified = True

                # Validate rule tables if present
                if 'state_rule_table' in fixed_metadata['params']:
                    try:
                        self._validate_rule_table(fixed_metadata['params'] ['state_rule_table'], 'state', rule_name)
                    except ValueError as e:
                        logger.error(f"Invalid state_rule_table in rule {rule_name}: {e}")
                        is_valid_library = False

                if 'edge_rule_table' in fixed_metadata['params']:
                    try:
                        self._validate_rule_table(fixed_metadata['params'] ['edge_rule_table'], 'edge', rule_name)
                    except ValueError as e:
                        logger.error(f"Invalid edge_rule_table in rule {rule_name}: {e}")
                        is_valid_library = False

                if rule_modified:
                    logger.info(f"Rule {rule_name} was modified. Changes made:")
                    for key in fixed_metadata:
                        if key not in rule_data or rule_data[key] != fixed_metadata[key]:
                            logger.info(f"  {key}: {rule_data.get(key)} -> {fixed_metadata[key]}")

                updated_rules.append(fixed_metadata)

                # Log final state of rule
                logger.debug(f"\nFinal state of Rule {rule_name}:")
                logger.debug(f"Present metadata fields: {set(fixed_metadata.keys())}")
                logger.debug(f"Present parameters: {set(fixed_metadata['params'].keys())}")

            # Update rules
            library_data['rules'] = updated_rules

            logger.debug("========= VALIDATION COMPLETE =========")
            return is_valid_library, modified

        except Exception as e:
            logger.error(f"Error in library validation: {e}")
            logger.error(traceback.format_exc())
            return False, False
        
    def _validate_rule_table(self, table_data: Dict[str, Any], table_type: str, rule_name: str):
        """Validate rule table data"""
        if not isinstance(table_data, dict):
            raise ValueError(f"Rule {rule_name}: Rule table must be a dictionary")

        if "default" not in table_data:
            raise ValueError(f"Rule {rule_name}: Rule table must have a default value")

        for key, value in table_data.items():
            if key == "default":
                continue

            # Validate key format
            try:
                components = key.strip("()").split(",")
                if table_type == 'state':
                    if len(components) != 3:
                        raise ValueError(f"Rule {rule_name}: Invalid state rule key format: {key}")
                    current_state = int(components[0])
                    active_neighbors = int(components[1])
                    connected_neighbors = int(components[2])
                    if current_state not in [-1, 0, 1] or active_neighbors < 0 or connected_neighbors < 0:
                        raise ValueError(f"Rule {rule_name}: Invalid values in state rule key: {key}")
                    if value not in [-1, 0, 1]:
                        raise ValueError(f"Rule {rule_name}: Invalid value for state rule key {key}: {value}")
                elif table_type == 'edge':
                    if len(components) != 3:
                        raise ValueError(f"Rule {rule_name}: Invalid edge rule key format: {key}")
                    self_state = int(components[0])
                    neighbor_state = int(components[1])
                    connection_pattern = components[2].strip()
                    if self_state not in [0, 1] or neighbor_state not in [0, 1] or len(connection_pattern) != 8 or not all(c in '01' for c in connection_pattern):
                        raise ValueError(f"Rule {rule_name}: Invalid values in edge rule key: {key}")
                    if value not in ["add", "remove", "maintain"]:
                        raise ValueError(f"Rule {rule_name}: Invalid value for edge rule key {key}: {value}")
            except (ValueError, IndexError) as e:
                raise ValueError(f"Rule {rule_name}: Invalid rule table format: {e}")
            
    def _initialize_default_metadata(self):
        """Initialize metadata for all default rules"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Use the RuleLibraryInfo class to create metadata
        for rule_name in RuleLibraryInfo.RULES.keys():
            try:
                # Access the rule data directly from the class
                rule_data = self.get_rule(rule_name)
                self.rule_metadata[rule_name] = DefaultRuleMetadata.get_default_metadata(
                    rule_type=rule_data['type'],
                    rule_name=rule_name,
                    category=rule_data['category'],
                    description=rule_data['description'],
                    dimension_compatibility=rule_data.get('dimension_compatibility', ["TWO_D", "THREE_D"]),
                    neighborhood_compatibility=rule_data.get('neighborhood_compatibility', []),
                    rating=None,
                    notes=None
                )
            except Exception as e:
                logger.error(f"Error initializing metadata for {rule_name}: {e}")
                                            
    def save_library(self):
        """Save current library state to JSON, creating a backup first."""
        try:
            # Create a backup file with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(APP_PATHS['rules_backups'], f"rules_backup_{timestamp}.json")  # Use APP_PATHS
            logger.info(f"Creating backup of rules.json at: {backup_path}")

            # Ensure the directory exists
            os.makedirs(APP_PATHS['rules_backups'], exist_ok=True)

            # Read the current rules.json to the backup location
            try:
                with open(self.library_path, 'r') as original_file:
                    backup_content = original_file.read()
            except FileNotFoundError:
                backup_content = ""  # Handle case where original file doesn't exist

            try:
                with open(backup_path, 'w') as backup_file:
                    backup_file.write(backup_content)  # Write the content to the backup file
            except Exception as e:
                logger.error(f"Error creating backup file: {e}")
                raise

            logger.info(f"Backup created successfully at {backup_path}")

            # Now, save the current state to the main rules.json file
            library_data = {
                'library_metadata': {
                    'version': '1.0',
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'description': 'Lace Rule Library'
                },
                'rules': list(self.rules.values())
            }

            try:
                with open(self.library_path, 'w') as f:
                    json.dump(library_data, f, indent=2)
                logger.info(f"Rule library saved to {self.library_path}")
            except Exception as e:
                logger.error(f"Error saving rule library: {e}")
                raise

        except Exception as e:
            logger.error(f"Error saving rule library: {e}")
            raise
              
    @classmethod
    def load_rule_params(cls, rule_name: str) -> Dict[str, Any]:
        """Load rule parameters from the library"""
        try:
            rule_data = cls.get_rule(rule_name)
            if 'params' not in rule_data:
                logger.error(f"No parameters found for rule {rule_name}")
                return {}
                
            # Return a copy of the parameters to avoid modifying the original
            return rule_data['params'].copy()
            
        except Exception as e:
            logger.error(f"Error loading parameters for rule {rule_name}: {e}")
            return {}

    @classmethod
    def get_parameter_info(cls, rule_name: str) -> Dict[str, str]:
        """Get parameter names and descriptions for a rule"""
        try:
            # Get rule data from library
            rule_data = cls.get_rule(rule_name)
            if not rule_data or 'params' not in rule_data:
                logger.error(f"No parameter info found for rule {rule_name}")
                return {}

            # Extract parameter info
            param_info = {}
            for param_name, param_data in rule_data['params'].items():
                if isinstance(param_data, dict) and 'description' in param_data:
                    param_info[param_name] = param_data['description']
                else:
                    # For parameters without explicit descriptions
                    param_info[param_name] = f"Parameter: {param_name}"

            return param_info

        except Exception as e:
            logger.error(f"Error getting parameter info for {rule_name}: {e}")
            return {}
                   
 
################################################
#               SIMULATION CONTROL             #
################################################

class SimulationController:
    
    def __init__(self, rule_name: str,
                 neighborhood_type: Optional[NeighborhoodType] = None,
                 dimension_type: Optional[Dimension] = None,
                 initial_density: float = 0.3,
                 initialize_state: bool = True):
        
        # Generate a unique ID for this instance
        unique_id = str(uuid.uuid4())
        # Hash the UUID to create a shorter name
        self._unique_id = 'g' + hashlib.md5(unique_id.encode()).hexdigest()[:8]  # 'g' + first 8 characters of MD5 hash
        
        
        try:
            # Add interruption flag
            self.interrupt_requested = False
            
            # Add stabilization flag
            self.auto_stabilize = True
            
            # Use global settings if none specified
            self.dimension_type = dimension_type or GlobalSettings.Simulation.DIMENSION_TYPE
            self.neighborhood_type = neighborhood_type or GlobalSettings.Simulation.NEIGHBORHOOD_TYPE
            self.dimensions = GlobalSettings.Simulation.get_grid_dimensions()
            self.rule_name = rule_name
            
            # Load and create initial rule
            try:
                rule_data = RuleLibraryManager.get_rule(rule_name)
            except ValueError as e:
                logger.error(f"Error loading rule {rule_name}: {e}")
                raise
            
            # Separate metadata and params
            metadata_dict = {k: v for k, v in rule_data.items() if k != 'params'}
            metadata = RuleMetadata(**metadata_dict)
            
            # Create rule instance
            try:
                self.rule = RuleLibrary.create_rule(rule_name, metadata)
            except ValueError as e:
                logger.error(f"Error creating rule {rule_name}: {e}")
                raise
                
            # Set the controller attribute of the metadata
            self.rule.metadata.controller = self
            
            # Set parameters
            if 'params' in rule_data:
                logger.debug(f"Loading parameters for {rule_name}: {rule_data['params']}")
                self.rule.params = copy.deepcopy(rule_data['params'])
            
            # Initialize grid with rule instance
            self.grid = Grid(self.dimensions, self.neighborhood_type, self.dimension_type, self.rule, unique_id=self._unique_id)
            
            # Initialize core components
            self.generation = 0
            self.stats = SimulationStats()
            self.perf_monitor = PerformanceMonitor()
            
            # Initialize highlighted and updated sets
            self.highlighted_nodes: Set[int] = set()
            self.highlighted_edges: Set[Tuple[int, int]] = set()
            self.last_updated_nodes: Set[int] = set()
            self.last_updated_edges: Set[Tuple[int, int]] = set()
            
            # Initialize auto-tuning parameters
            self.spatial_hash = SpatialHashGrid(self.dimensions)
            self.spatial_hash.adapt_grid()
            
            # Initialize shared memory and process pool
            self.shared_mem = None
            self.shared_array = None
            
            try:
                if GlobalSettings.Simulation.NUM_PROCESSES > 1:
                    # First try to clean up any existing shared memory
                    shared_mem_name = f'/{self._unique_id}' # Shorten the name
                    try:
                        shared_memory.SharedMemory(name=shared_mem_name).unlink()
                    except FileNotFoundError:
                        pass  # Ignore if it doesn't exist
                        
                    self.process_pool = ProcessPoolExecutor(
                        max_workers=GlobalSettings.Simulation.NUM_PROCESSES,
                        initializer=Grid._process_initializer
                    )
                    logger.info(f"Created process pool with {GlobalSettings.Simulation.NUM_PROCESSES} workers")
                else:
                    self.process_pool = ThreadPoolExecutor(max_workers=1)
                    logger.info("Falling back to single-threaded executor")
                    
                # Pass the unique ID to the Grid
                self.grid = Grid(self.dimensions, self.neighborhood_type, self.dimension_type, self.rule, unique_id=self._unique_id)
                self.grid.setup_shared_memory()
                
            except Exception as e:
                logger.error(f"Failed to initialize process pool: {e}")
                self.process_pool = ThreadPoolExecutor(max_workers=1)
                logger.info("Falling back to single-threaded executor")
            
            # Add these lines back in
            self.state_updated = False
            self.process_chunk = Grid._process_chunk
            
            # Initialize new state
            logger.debug(f"Initializing random state with density: {initial_density}")
            density = initial_density if initial_density is not None else GlobalSettings.Simulation.INITIAL_NODE_DENSITY
            self._initialize_random_state(density)
            
            # Initialize edges
            self.grid.initialize_edges()
            
            # Add a queue for communication with worker processes
            self.result_queue: Queue = Queue()
            
        except Exception as e:
            logger.error(f"Error initializing SimulationController: {e}")
            raise
                                          
    @staticmethod
    def _init_worker():
        """Initialize worker process"""
        try:
            worker_id = mp.current_process().name
            print(f"Worker {worker_id} initializing")
            root_logger = logging.getLogger()
            
            # Log existing handlers before clearing
            print(f"Worker {worker_id} handlers before clearing: {[str(h) for h in root_logger.handlers]}")
            
            # Disable logging in worker processes
            root_logger.handlers = []
            root_logger.addHandler(logging.NullHandler())
            
            # Log final handler state
            print(f"Worker {worker_id} final handlers: {[str(h) for h in root_logger.handlers]}")
            
        except Exception as e:
            print(f"Error in worker initialization: {e}")
                                                    
    def step(self):
        """Perform one simulation step"""
        try:
            # Update grid state
            logger.debug("=============== GRID UPDATE (SimulationController.step) ===============")
            
            # Store pre-update state for verification
            pre_active = np.sum(self.grid.grid_array > 0)
            pre_edges = np.sum(self.grid.neighborhood_data.edge_matrix) // 2
            logger.debug(f"Pre-update state - Active nodes: {pre_active}, Edges: {pre_edges}")
            
            # Verify grid and process pool exist
            if not hasattr(self, 'grid') or self.grid is None:
                logger.error("Grid not initialized")
                raise RuntimeError("Grid not initialized")
                
            if not hasattr(self, 'process_pool') or self.process_pool is None:
                logger.error("Process pool not initialized")
                raise RuntimeError("Process pool not initialized")
                
            # Log process pool state
            logger.debug(f"Process pool exists: {self.process_pool is not None}")
            if hasattr(self.process_pool, '_max_workers'):
                if isinstance(self.process_pool, ThreadPoolExecutor):
                    logger.debug(f"Process pool workers: {self.process_pool._max_workers}")
                else:
                    logger.debug("Process pool is not a ThreadPoolExecutor")
                
            # Verify update_grid_parallel exists and is callable
            if not hasattr(self.grid, 'update_grid_parallel'):
                logger.error("update_grid_parallel method not found")
                raise RuntimeError("update_grid_parallel method not found")
                
            logger.debug("About to call update_grid_parallel")
            
            # Perform grid update with explicit try-except
            try:
                self.grid.update_grid_parallel()
                logger.debug("Grid update_grid_parallel call completed")
            except Exception as e:
                logger.error(f"Error in update_grid_parallel: {e}\nTraceback:\n{traceback.format_exc()}")
                raise
                    
            # Verify post-update state
            post_active = np.sum(self.grid.grid_array > 0)
            post_edges = np.sum(self.grid.neighborhood_data.edge_matrix) // 2
            logger.debug(f"Post-update state - Active nodes: {post_active}, Edges: {post_edges}")
            
            if post_active == pre_active and post_edges == pre_edges:
                logger.warning("Grid state unchanged after update - possible update failure")
                    
            # Update generation counter
            self.generation += 1
                
            # Update statistics
            self._update_statistics()
            
            # Check for stabilization if enabled
            if self.auto_stabilize:
                self._check_stabilization()
                    
            # Periodically cleanup ghost edges
            if self.generation % 10 == 0:
                self.grid.cleanup_ghost_edges()
                    
            logger.debug("=============== GRID UPDATE COMPLETE ===============")

        except Exception as e:
            logger.error(f"Error in simulation step: {e}\nTraceback:\n{traceback.format_exc()}")
            raise
                                    
    def cleanup(self):
        """Clean up controller resources"""
        try:
            if hasattr(self, 'grid'):
                self.grid.cleanup()

                # Clean up process pool
                if hasattr(self, 'process_pool') and self.process_pool is not None:
                    try:
                        self.process_pool.shutdown(wait=True)
                        logger.info("Process pool shut down successfully")
                    except Exception as e:
                        logger.error(f"Error shutting down process pool: {e}")
                    finally:
                        self.process_pool = None
                    
                    # Clean up the result queue
                    if hasattr(self, 'result_queue') and self.result_queue is not None:
                        try:
                            while not self.result_queue.empty():
                                self.result_queue.get_nowait()
                            logger.info("Result queue cleared")
                        except Exception as e:
                            logger.error(f"Error clearing result queue: {e}")
                    
                    # Clean up any other resources...
                    if hasattr(self, 'spatial_hash'):
                        del self.spatial_hash
                        logger.info("Spatial hash grid deleted")
                    
                    if hasattr(self, 'stats'):
                        del self.stats
                        logger.info("Simulation stats deleted")
                    
                    if hasattr(self, 'perf_monitor'):
                        del self.perf_monitor
                        logger.info("Performance monitor deleted")
                    
                    if hasattr(self, 'rule'):
                        del self.rule
                        logger.info("Rule deleted")
                    
        except Exception as e:
            logger.error(f"Error in controller cleanup: {e}")
                                            
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()

    def _create_rule_instance(self, rule_data: Dict[str, Any]) -> Rule:
            """Create rule instance based on rule data from library"""
            try:
                # Use the rule name to look up the factory function
                rule_name = rule_data['name']

                # Remove 'status' from rule_data before creating RuleMetadata
                rule_data_copy = rule_data.copy()
                rule_data_copy.pop('status', None)  # Remove 'status' if it exists
                params = rule_data_copy.pop('params', {}) # Get the params, and remove from the copy
                if 'position' not in rule_data_copy:
                    rule_data_copy['position'] = 1  # Or some other default value

                metadata = RuleMetadata(**rule_data_copy) # Create metadata object
                rule = RuleLibrary.create_rule(rule_name, metadata) # Create the rule instance

                # Set parameters
                rule.params = params

                # Initialize module assignments and specialization_scores for ModularRule
                if isinstance(rule, ModularRule):
                    rule.module_assignments = {}  # Initialize
                    rule.specialization_scores = {}  # Initialize

                return rule
            except Exception as e:
                logger.error(f"Error creating rule {rule_data['name']}: {e}")
                raise
                                
    def set_rule(self, rule_name: str):
        """Change simulation rule"""
        try:
            # Load rule data and create new rule instance
            try:
                rule_data = RuleLibraryManager.get_rule(rule_name)
            except ValueError as e:
                logger.error(f"Error loading rule {rule_name}: {e}")
                return False
            
            # Create metadata for the rule
            metadata_dict = {k: v for k, v in rule_data.items() if k != 'params'}
            metadata_dict['neighborhood_type'] = self.neighborhood_type # ADDED
            metadata = RuleMetadata(**metadata_dict)
            
            # Create new rule instance
            try:
                new_rule = RuleLibrary.create_rule(rule_name, metadata)
            except ValueError as e:
                logger.error(f"Error creating rule {rule_name}: {e}")
                return False
            
            # Get parameters from rule data
            params = rule_data.get('params', {})
            if not params:
                logger.warning(f"No parameters found for rule {rule_name} in library")
                # Removed call to _get_default_params
                params = {} # Initialize params to empty dict if not found in rule data
                
            # Deep copy parameters here
            new_rule.params = copy.deepcopy(params) # Deep copy params
            logger.debug(f"New rule parameters after deep copy: {new_rule.params}")
            
            # Update controller's rule
            self.rule = new_rule
            self.rule_name = rule_name
            
            # Update grid's rule
            if not self.grid.set_rule(new_rule):
                logger.error(f"Failed to set rule on grid for {rule_name}")
                return False
            
            logger.info(f"Changed rule to: {rule_name}")
            logger.debug(f"Controller rule parameters loaded: {new_rule.params}")
            
        except Exception as e:
            logger.error(f"Error changing rule: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't reraise - handle gracefully
            return False
            
        return True

    def _load_rule_from_library(self, rule_name: str, params: Optional[Dict[str, Any]] = None) -> Rule:
        """Load rule from library and set parameters"""
        try:
            # Get rule data from library
            rule_data = RuleLibraryManager.get_rule(rule_name)
            
            # Create rule instance
            rule_type = rule_data['type']
            metadata = RuleMetadata(**rule_data)
            rule = RuleLibrary.create_rule(rule_type, metadata)
            
            # Set parameters
            if params:
                rule.params = params
            else:
                rule.params = rule_data['params']
                
            return rule
        except Exception as e:
            logger.error(f"Error loading rule {rule_name}: {e}")
            raise
                                    
    def _initialize_random_state(self, density: float):
        """Initialize grid with random active cells based on density"""
        try:
            logger.debug(f"Entering _initialize_random_state with density {density}")
            
            # Calculate total cells based on current dimensions
            total_cells = np.prod(self.dimensions)
            active_cells = int(total_cells * density)
            
            # Clear the existing grid array before initializing
            self.grid.grid_array.fill(-1.0)
            
            # Create random active cell indices using vectorized operation
            active_indices = np.random.choice(
                total_cells, 
                size=active_cells, 
                replace=False
            )
            
            logger.debug(f"Total cells: {total_cells}, Active cells: {active_cells}")
            
            # Set active cells directly using flat indices
            num_activated = 0
            for idx in active_indices:
                # Convert flat index to coordinates
                coords = tuple(_unravel_index(idx, self.dimensions))
                
                # Validate coordinates are within bounds
                if all(0 <= c < d for c, d in zip(coords, self.dimensions)):
                    try:
                        self.grid.set_node_state(coords, 1.0)  # Set state to 1.0 for active
                        num_activated += 1
                    except Exception as e:
                        logger.error(f"Error setting node state at {coords}: {e}")
                else:
                    logger.warning(f"Coordinates {coords} out of bounds, skipping")
            
            logger.debug(f"Number of nodes activated: {num_activated}")
            
            # Note: We don't initialize edges here, as that will be done separately
            # to avoid redundant initialization
            
        except Exception as e:
            logger.error(f"Error in random state initialization: {str(e)}\nTraceback:\n{traceback.format_exc()}")
            raise
                                                                
    def reset(self, initial_density: Optional[float] = None):
        """Reset simulation to initial state"""
        try:
            logger.debug("Entering reset_simulation")
            
            # Store current spacing before reset
            current_spacing = GlobalSettings.Visualization.NODE_SPACING
            
            # Stop any running simulation
            self.running = False
            self.paused = False
            
            # Clear existing state
            logger.debug("Clearing existing state")
            
            if hasattr(self, 'grid'):
                self.grid.grid_array.fill(-1.0)  # Reset to empty state
                self.grid.neighborhood_data.clear()
                self.grid.neighborhood_data.edge_matrix[:] = False # Reset edge matrix
            
            # Reset counters and flags
            logger.debug("Resetting counters and flags")
            self.generation = 0
            self.step_count = 0
            self.is_running = False
            self.highlighted_nodes.clear()
            self.highlighted_edges.clear()
            self.last_updated_nodes.clear()
            self.last_updated_edges.clear()
            
            # Reset statistics
            logger.debug("Resetting statistics")
            self.stats.reset()
            self.perf_monitor.reset()
            
            # Initialize new state
            logger.debug(f"Initializing new state with density: {initial_density}")
            density = initial_density if initial_density is not None else GlobalSettings.Simulation.INITIAL_NODE_DENSITY
            self._initialize_random_state(density)
            
            # Reinitialize grid with current rule
            logger.debug("Reinitializing grid with current rule")
            
            # Store the rule and parameters before re-initialization
            current_rule_name = self.rule_name
            current_params = self.rule.params.copy()
            logger.debug(f"Stored current rule parameters: {current_params}")
            
            # Reinitialize shared memory and process pool
            try:
                if hasattr(self, 'grid'):
                    self.grid.cleanup()
                    
                if GlobalSettings.Simulation.NUM_PROCESSES > 1:
                    # First try to clean up any existing shared memory
                    shared_mem_name = f'/{self._unique_id}' # Shorten the name
                    try:
                        shared_memory.SharedMemory(name=shared_mem_name).unlink()
                        logger.info(f"Successfully unlinked existing shared memory: {shared_mem_name}")
                    except FileNotFoundError:
                        logger.warning(f"Shared memory block not found, skipping unlink: {shared_mem_name}")
                    except Exception as e:
                        logger.error(f"Error unlinking shared memory: {e}")
                        
                    self.process_pool = ProcessPoolExecutor(
                        max_workers=GlobalSettings.Simulation.NUM_PROCESSES,
                        initializer=Grid._process_initializer
                    )
                    logger.info(f"Recreated process pool with {GlobalSettings.Simulation.NUM_PROCESSES} workers")
                else:
                    self.process_pool = ThreadPoolExecutor(max_workers=1)
                    logger.info("Falling back to single-threaded executor")
                    
                    # Reinitialize the grid
                    self.grid = Grid(self.dimensions, self.neighborhood_type, self.dimension_type, self.rule, unique_id=self._unique_id)
                    self.grid.setup_shared_memory()
                    
            except Exception as e:
                logger.error(f"Failed to reinitialize process pool: {e}")
                self.process_pool = ThreadPoolExecutor(max_workers=1)
                logger.info("Falling back to single-threaded executor")
            
            # Restore the rule and parameters
            if not self.set_rule(current_rule_name):
                logger.error(f"Failed to restore rule {current_rule_name}")
                # Handle the error appropriately, perhaps by setting a default rule
                
            # Reapply parameters
            self.rule.params = current_params
            logger.debug(f"Restored rule parameters: {self.rule.params}")
            
            # Initialize edges with optimized method - use grid directly
            self.grid.initialize_edges()
            
            # Restore spacing after reset
            GlobalSettings.Visualization.set_node_spacing(current_spacing)
            
            logger.info("Simulation reset complete")
            
            # Log the rule parameters after the reset
            logger.debug(f"Rule parameters AFTER reset: {self.rule.params}")
            
        except Exception as e:
            logger.error(f"Error in simulation reset: {str(e)}")
            raise
                                                
    def initialize_edges(self):
        """Initialize edges between nearby active cells, using batched processing"""
        try:
            logger.debug("SimulationController.initialize_edges: Starting edge initialization")
            
            # Get indices of all active nodes using vectorized operation through grid
            active_indices = np.where(self.grid.grid_array.ravel() > GlobalSettings.Visualization.NODE_VISIBILITY_THRESHOLD)[0]
            
            # Create batches to process edges in chunks
            batch_size = 1000
            num_batches = (len(active_indices) + batch_size - 1) // batch_size
            
            logger.debug(f"Processing {len(active_indices)} active nodes in {num_batches} batches")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(active_indices))
                batch_indices = active_indices[start_idx:end_idx]
                
                # Process each node in the batch
                for idx in batch_indices:
                    # Get neighboring active cells (unscaled indices)
                    neighbors = self.grid.neighborhood_data.get_neighbor_indices(idx)
                    active_neighbors = neighbors[
                        self.grid.neighborhood_data.states[neighbors] > 
                        GlobalSettings.Visualization.NODE_VISIBILITY_THRESHOLD
                    ]
                    
                    # Use vectorized operations for edge creation
                    if len(active_neighbors) > 0:
                        # Create random mask for edge creation
                        edge_mask = np.random.random(len(active_neighbors)) < GlobalSettings.Simulation.INITIAL_EDGE_DENSITY
                        selected_neighbors = active_neighbors[edge_mask]
                        
                        if len(selected_neighbors) > 0:
                            # Update edges in batch
                            self.grid.neighborhood_data.update_edges(idx, set(selected_neighbors))
                
                # Log progress for large grids
                if batch_idx % 10 == 0:
                    logger.debug(f"Processed batch {batch_idx + 1}/{num_batches}")
                    
            logger.debug("Edge initialization completed")
            
        except Exception as e:
            logger.error(f"Error in edge initialization: {str(e)}\nTraceback:\n{traceback.format_exc()}")
            raise
        
    def request_interrupt(self):
        """Request interruption of current step calculation"""
        self.interrupt_requested = True
        logger.info("Interruption requested")
                      
    def _update_edges(self):
        """Update edges based on rule"""
        total_nodes = np.prod(self.dimensions)
        
        for idx in range(total_nodes):
            if self.grid.neighborhood_data.states[idx] > 0:  # Only update active nodes
                new_edges = self.rule.compute_edge_updates(idx, self.grid.neighborhood_data)
                self.grid.neighborhood_data.update_edges(idx, new_edges)
                    
    def _update_statistics(self):
        """Update simulation statistics"""
        active_cells = np.sum(self.grid.grid_array > 0)
        total_cells = np.prod(self.dimensions)
        
        try:
            edge_density = float(np.mean([
                self.grid.neighborhood_data.get_metric(idx, 'edge_density')
                for idx in range(total_cells)
            ]))
        except:
            edge_density = 0.0 # Handle cases where edge density might be undefined
            
        performance_stats = self.perf_monitor.get_stats()
        
        self.stats.update(
            generation=self.generation,
            active_ratio=active_cells / total_cells,
            edge_density=edge_density,
            # Add individual performance metrics instead of the whole dictionary
            simulation_avg_time=performance_stats.get('step', {}).get('avg', 0.0),
            grid_avg_time=self.grid.get_performance_stats().get('grid_stats', {}).get('avg_update_time', 0.0),
            rule_avg_time=self.rule.get_performance_stats().get('avg_compute_time', 0.0)
        )
        
    def _check_stabilization(self):
        """Check if simulation has stabilized"""
        if self.generation > GlobalSettings.Simulation.MIN_GENERATIONS:
            recent_activity = self.stats.get_recent_activity(
                window=GlobalSettings.Simulation.STABILITY_WINDOW
            )
            
            if np.std(recent_activity) < GlobalSettings.Simulation.STABILITY_THRESHOLD:
                logger.info(f"Simulation stabilized at generation {self.generation}")
                self.is_running = False
                    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set controller state from dictionary"""
        try:
            # Update grid state
            if 'grid_state' in state:
                np.copyto(self.grid.grid_array, state['grid_state'])
            
            # Update edge matrix
            if 'edge_matrix' in state:
                np.copyto(self.grid.neighborhood_data.edge_matrix, state['edge_matrix'])
            
            # Update generation count
            if 'generation' in state:
                self.generation = state['generation']
            
            # Update other attributes if needed
            if 'stats' in state:
                self.stats.update(**state['stats'])
                
            logger.info(f"Controller state updated from dictionary")
        except Exception as e:
            logger.error(f"Error setting controller state: {e}")
            raise
        
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state with direct array access"""
        return {
            'generation': self.generation,
            'grid_state': self.grid.grid_array,  # Direct array access
            'edge_matrix': self.grid.neighborhood_data.edge_matrix,  # Direct array access
            'stats': self.stats.get_current(),
            'performance': self.perf_monitor.get_stats()
        }
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        return {
            'simulation': self.perf_monitor.get_stats(),
            'grid': self.grid.get_performance_stats(),
            'rule': self.rule.get_performance_stats()
        }

    def set_neighborhood_type(self, neighborhood_type: NeighborhoodType):
        """Change the neighborhood type and reinitialize grid"""
        try:
            if neighborhood_type != self.neighborhood_type:
                # Validate compatibility
                if (neighborhood_type == NeighborhoodType.HEX_PRISM and
                    self.dimension_type != Dimension.THREE_D):
                    self.dimension_type = Dimension.THREE_D
                    GlobalSettings.Simulation.DIMENSION_TYPE = Dimension.THREE_D
                    self.dimensions = GlobalSettings.Simulation.get_grid_dimensions()
                    logger.info(f"HEX_PRISM neighborhood requires 3D, changed dimension type to: {self.dimension_type.name}")
                elif (neighborhood_type == NeighborhoodType.HEX and
                    self.dimension_type == Dimension.THREE_D):
                    self.dimension_type = Dimension.TWO_D
                    GlobalSettings.Simulation.DIMENSION_TYPE = Dimension.TWO_D
                    self.dimensions = GlobalSettings.Simulation.get_grid_dimensions()

                    logger.info(f"HEX neighborhood requires 2D, changed dimension type to: {self.dimension_type.name}")
                    
                # Update neighborhood type
                self.neighborhood_type = neighborhood_type
                
                # Reinitialize grid with new neighborhood type
                self.grid = Grid(self.dimensions, self.neighborhood_type, self.dimension_type, self.rule, unique_id=self._unique_id)
                
                # Reset simulation state
                self.generation = 0
                self.is_running = False

                logger.info(f"Reinitialized grid with new neighborhood type: {neighborhood_type.name}")
                
        except Exception as e:
            logger.error(f"Error setting neighborhood type: {e}")
            raise

class ValidatedEntry(tk.Entry):
    """Entry widget that supports tooltips"""
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self._tooltip: Optional[ToolTip] = None 
               
class ToolTip:
    """Creates a tooltip for a given widget"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        
        widget.bind("<Enter>", self.enter)
        widget.bind("<Leave>", self.leave)

    def showtip(self, event=None):
        """Display text in tooltip window"""
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        # Creates a toplevel window
        self.tipwindow = tw = tk.Toplevel(self.widget)
        # Leaves only the label and destroys other windows
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font="tahoma 8 normal")
        label.pack()

    def hidetip(self):
        """Hides the tooltip"""
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

    def enter(self, event=None):
        """Handles mouse enter event"""
        self.id = self.widget.after(500, self.showtip)

    def leave(self, event=None):
        """Handles mouse leave event"""
        if self.id:
            self.widget.after_cancel(self.id)
        self.id = None
        if self.tipwindow:
            self.hidetip()

    def destroy(self):
        """Unbind events to prevent memory leaks"""
        self.widget.unbind("<Enter>", None)
        self.widget.unbind("<Leave>", None)
        if self.tipwindow:
            self.hidetip()

class ScrollableFrame(tk.Frame):
    """A scrollable frame that works consistently across platforms"""
    def __init__(self, container: tk.Widget, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # Create frame inside canvas for content
        self.scrolled_frame = tk.Frame(self.canvas)
        self.scrolled_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrolled_frame, anchor="nw")
        
        # Configure canvas scrolling
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Bind canvas resize to frame resize
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Setup mouse wheel scrolling
        self._bind_mouse_scroll(self.scrolled_frame)
        self._bind_mouse_scroll(self.canvas)
        
        # Pack scrollbar and canvas
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
    def _on_canvas_configure(self, event):
        """Handle canvas resize"""
        # Update the width of the frame to fill canvas
        self.canvas.itemconfig(self.canvas_frame, width=event.width)
        
    def _on_mousewheel(self, event):
        """Cross-platform mouse wheel scrolling"""
        if self.scrollbar.get() != (0.0, 1.0):  # Only scroll if there's something to scroll
            if event.num == 4 or event.num == 5:  # Linux
                delta = -1 if event.num == 4 else 1
            else:  # Windows/macOS
                # Convert delta to consistent unit
                delta = -1 * (event.delta // 120 if abs(event.delta) >= 120 else event.delta)
            
            self.canvas.yview_scroll(delta, "units")
            
    def _bind_mouse_scroll(self, widget):
        """Bind mouse wheel events for all platforms"""
        # Windows/macOS
        widget.bind("<MouseWheel>", self._on_mousewheel)
        
        # Linux
        widget.bind("<Button-4>", self._on_mousewheel)
        widget.bind("<Button-5>", self._on_mousewheel)
        
    def bind_child_scroll(self, widget):
        """Bind mouse wheel scrolling to a child widget"""
        self._bind_mouse_scroll(widget)

class ChangeTracker:
    """Tracks changes to rule parameters and provides undo/redo functionality"""
    def __init__(self):
        self.original_values = {}
        self.current_values = {}
        self.undo_stack = []
        self.redo_stack = []
        self._is_modified = False

    def initialize(self, params: Dict[str, Any]):
        """Initialize with current parameter values"""
        self.original_values = copy.deepcopy(params)
        self.current_values = copy.deepcopy(params)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._is_modified = False

    def track_change(self, param_name: str, old_value: Any, new_value: Any):
        """Track a parameter change"""
        if old_value != new_value:
            self.undo_stack.append({
                'param': param_name,
                'old': old_value,
                'new': new_value
            })
            self.current_values[param_name] = new_value
            self.redo_stack.clear()  # Clear redo stack on new change
            self._is_modified = True

    def undo(self) -> Optional[Dict[str, Any]]:
        """Undo last change"""
        if self.undo_stack:
            change = self.undo_stack.pop()
            self.redo_stack.append(change)
            self.current_values[change['param']] = change['old']
            self._update_modified_state()
            return change
        return None

    def redo(self) -> Optional[Dict[str, Any]]:
        """Redo last undone change"""
        if self.redo_stack:
            change = self.redo_stack.pop()
            self.undo_stack.append(change)
            self.current_values[change['param']] = change['new']
            self._update_modified_state()
            return change
        return None

    def is_modified(self) -> bool:
        """Check if any values are modified from original"""
        return self._is_modified

    def _update_modified_state(self):
        """Update the modified state based on current values"""
        self._is_modified = any(
            self.current_values.get(k) != v 
            for k, v in self.original_values.items()
        )
        
class RuleTableEditor(tk.Frame):

    def __init__(self, parent: tk.Frame, simulation_gui: 'SimulationGUI', 
                    table_name: str, table_data: dict, table_type: str, param_info: dict):
            super().__init__(parent)
            self.simulation_gui = simulation_gui
            self.table_name = table_name
            self.table_data = table_data.copy()
            self.table_type = table_type
            self.param_info = param_info
            self._original_table_data = copy.deepcopy(table_data)

            # Create scrollable frame
            self.scrollable = ScrollableFrame(self)
            self.scrollable.pack(fill=tk.BOTH, expand=True)

            # Create table entries in the scrolled frame
            self._create_table_entries(self.table_data, self.table_type)
            self.simulation_gui = simulation_gui
            self._is_updating = False
            self._error_state = False
            self._previous_state = {}
            self._current_state = self.table_data.copy()

            # Create scrolled frame for table
            self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
            self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.scrollable_frame = tk.Frame(self.canvas)

            self.scrollable_frame.bind(
                "<Configure>",
                lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            )

            self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
            self.canvas.configure(yscrollcommand=self.scrollbar.set)

            # Pack scrollbar and canvas
            self.scrollbar.pack(side="right", fill="y")
            self.canvas.pack(side="left", fill="both", expand=True)

            # Create table entries
            self._create_table_entries(self.table_data, self.table_type)
            
            # Create buttons
            self._create_buttons()
            
            # Bind mousewheel scrolling
            self.scrollable_frame.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
    def _safe_update(self, update_func: Callable) -> bool:
        """Safely execute an update with error recovery"""
        if self._is_updating:
            return False
            
        self._is_updating = True
        self._error_state = False
        
        try:
            # Store current state before update
            self._previous_state = self._current_state.copy()
            
            # Execute update
            update_func()
            
            # Update succeeded - store new state
            self._current_state = self.get_table_data()
            return True
            
        except Exception as e:
            logger.error(f"Error during table update: {e}")
            self._error_state = True
            
            # Attempt recovery
            try:
                self.set_table_data(self._previous_state)
                logger.info("Successfully recovered to previous state")
            except Exception as recovery_error:
                logger.error(f"Error during recovery: {recovery_error}")
                messagebox.showerror(
                    "Critical Error",
                    "Failed to recover from error. Please close and reopen the editor."
                )
            return False
            
        finally:
            self._is_updating = False

    def _safe_delete_row(self, row_frame: tk.Frame):
        """Safely delete a row with confirmation and recovery"""
        if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this rule?"):
            def delete_row():
                row_frame.destroy()
                # Update current state after deletion
                self._current_state = self.get_table_data()
                
            self._safe_update(delete_row)

    def _create_validated_menu(self, parent: tk.Frame, var: tk.StringVar, options: List[str]) -> tk.OptionMenu:
        """Create an option menu with validation and error recovery"""
        menu = tk.OptionMenu(parent, var, *options)
        
        def on_select(*args):
            if not self._is_updating:
                def update():
                    # Validation happens in _safe_update
                    pass
                self._safe_update(update)
                
        var.trace('w', on_select)
        return menu
        
    def _get_column_validators(self) -> List[Callable[[str], bool]]:
        """Get validators for each column based on table type"""
        if self.table_type == 'state':
            return [
                lambda x: x in ['-1', '0', '1'],  # current_state
                lambda x: x.isdigit(),  # active_neighbors
                lambda x: x.isdigit(),  # connected_neighbors
                lambda x: x in ['-1', '0', '1']  # new_state
            ]
        else:
            return [
                lambda x: x in ['0', '1'],  # self_state
                lambda x: x in ['0', '1'],  # neighbor_state
                lambda x: x.isdigit(),  # self_connections
                lambda x: x.isdigit(),  # neighbor_connections
                lambda x: self._is_valid_float(x, 0, 1),  # self_clustering
                lambda x: self._is_valid_float(x, 0, 1),  # neighbor_clustering
            ]

    def _is_valid_float(self, value: str, min_val: float, max_val: float) -> bool:
        """Validate float value in range"""
        try:
            if not value:
                return True
            f = float(value)
            return min_val <= f <= max_val
        except ValueError:
            return False
        
    def _get_current_row_data(self) -> List[str]:
        """Get data from current row being edited"""
        try:
            focused = self.scrollable_frame.focus_get()
            if focused is None:
                return []
            if not hasattr(focused, 'master'):
                return []
            current_row = []
            if focused.master is not None:
                for widget in focused.master.winfo_children():
                    if isinstance(widget, tk.Entry):
                        current_row.append(widget.get())
            return current_row
        except (AttributeError, tk.TclError):
            return []

    def get_table_data(self) -> dict:
        """Get the current table data with validation"""
        data = {}
        
        # Collect data from entries
        for child in self.scrollable_frame.winfo_children():
            if isinstance(child, tk.Frame):
                entries = [w for w in child.winfo_children() 
                        if isinstance(w, (tk.Entry, tk.OptionMenu))]
                if entries:
                    # Construct key from components
                    key_parts = []
                    for e in entries[:-1]:
                        if isinstance(e, tk.Entry):
                            key_parts.append(e.get())
                        else:  # OptionMenu
                            var = e.cget('textvariable')
                            if var:
                                key_parts.append(str(self.getvar(var)))
                    key = f"({','.join(key_parts)})"
                    
                    # Get value
                    last_entry = entries[-1]
                    if isinstance(last_entry, tk.Entry):
                        value = last_entry.get()
                    else:  # OptionMenu
                        var = last_entry.cget('textvariable')
                        value = str(self.getvar(var)) if var else ""
                    
                    # Convert state values to int
                    if self.table_type == 'state':
                        try:
                            value = int(value)
                        except ValueError:
                            value = 0
                            
                    data[key] = value
        
        # Get default value
        default_frame = next((c for c in self.scrollable_frame.winfo_children() 
                            if isinstance(c, tk.Frame)), None)
        if default_frame:
            default_widget = next((w for w in default_frame.winfo_children()
                                if isinstance(w, (tk.Entry, tk.OptionMenu))), None)
            if default_widget:
                if isinstance(default_widget, tk.Entry):
                    try:
                        data["default"] = int(default_widget.get())
                    except ValueError:
                        data["default"] = 0
                else:  # OptionMenu
                    var = default_widget.cget('textvariable')
                    data["default"] = str(self.getvar(var)) if var else "maintain"
        
        return data

    def _create_header(self):
        """Create table header based on rule type"""
        header_frame = tk.Frame(self.scrollable_frame)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        if self.table_type == 'state':
            headers = ["Current State", "Active Neighbors", "Connected Neighbors", "New State"]
        else:  # edge
            headers = ["Self State", "Neighbor State", "Self Connections", 
                      "Neighbor Connections", "Self Clustering", "Neighbor Clustering", "Action"]
            
        for i, header in enumerate(headers):
            tk.Label(header_frame, text=header, font=("TkDefaultFont", 10, "bold")).grid(
                row=0, column=i, padx=5, pady=2)
                    
    def _create_entries(self):
            """Create table entries with error recovery"""
            def safe_create_entries():
                # Clear existing entries
                for child in self.scrollable_frame.winfo_children():
                    child.destroy()
                    
                # Create new entries
                for key, value in self.table_data.items():
                    if key == "default":
                        continue
                        
                    row_frame = tk.Frame(self.scrollable_frame)
                    row_frame.pack(fill=tk.X, padx=5, pady=2)
                    
                    # Parse the tuple string into components
                    components = key.strip("()").split(",")
                    
                    # Create entries for each component with validation
                    for i, comp in enumerate(components):
                        entry = self._create_validated_entry(row_frame, i)
                        entry.insert(0, comp.strip())
                        entry.grid(row=0, column=i, padx=2)
                        
                    # Create value entry/dropdown with validation
                    if self.table_type == 'state':
                        value_var = tk.StringVar(value=str(value))
                        value_entry = self._create_validated_entry(row_frame, len(components))
                        value_entry.config(textvariable=value_var)
                        value_entry.grid(row=0, column=len(components), padx=2)
                    else:
                        value_var = tk.StringVar(value=value)
                        value_menu = self._create_validated_menu(
                            row_frame, 
                            value_var,
                            ["add", "remove", "maintain"]
                        )
                        value_menu.grid(row=0, column=len(components), padx=2)
                        
                    # Add delete button with confirmation
                    delete_btn = tk.Button(
                        row_frame, 
                        text="X",
                        command=lambda r=row_frame: self._safe_delete_row(r)
                    )
                    delete_btn.grid(row=0, column=len(components)+1, padx=2)

            self._safe_update(safe_create_entries)
         
    def _create_buttons(self):
        """Create buttons for adding rules and handling default"""
        button_frame = tk.Frame(self.scrollable_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Button(button_frame, text="Add Rule", 
                command=self._add_new_rule).pack(side="left", padx=5)
                
        # Default value handling
        tk.Label(button_frame, text="Default:").pack(side="left", padx=5)
        if self.table_type == 'state':
            default_var = tk.StringVar(value=str(self.table_data.get("default", "0")))
            tk.Entry(button_frame, textvariable=default_var, width=10).pack(side="left")
        else:
            default_var = tk.StringVar(value=self.table_data.get("default", "maintain"))
            tk.OptionMenu(button_frame, default_var, "add", "remove", "maintain").pack(side="left")

    def _on_table_change(self, new_table_data: Dict[str, Any]):
        try:
            # Get change tracker through simulation_gui
            if tracker := self.simulation_gui._get_change_tracker():
                tracker.track_change(
                    self.table_name,
                    self._original_table_data,
                    new_table_data
                )
            
            # Update the table
            self.table_data = new_table_data
            
            # Update the rule parameters
            for key, value in new_table_data.items():
                self.simulation_gui._on_parameter_change(
                    f"{self.table_name}.{key}",
                    value,
                    self.simulation_gui._parameter_entries
                )
            
        except Exception as e:
            logger.error(f"Error updating table: {e}")
            messagebox.showerror("Error", f"Failed to update table: {e}")

    def _randomize_rule_table(self):
        """Randomize table values with change tracking"""
        try:
            if self.table_type == 'state':
                new_table = {"default": random.choice([-1, 0, 1])}
                for _ in range(random.randint(5, 10)):
                    current_state = random.choice([-1, 0, 1])
                    active_neighbors = random.randint(0, 8)
                    connected_neighbors = random.randint(0, min(active_neighbors, 8))
                    new_state = random.choice([-1, 0, 1])
                    key = f"({current_state},{active_neighbors},{connected_neighbors})"
                    new_table[key] = new_state
            else:
                new_table = {"default": random.choice(["add", "remove", "maintain"])}
                for _ in range(random.randint(5, 10)):
                    self_state = random.choice([0, 1])
                    neighbor_state = random.choice([0, 1])
                    connection_pattern = ''.join(random.choice(['0', '1']) for _ in range(8))
                    action = random.choice(["add", "remove", "maintain"])
                    key = f"({self_state},{neighbor_state},{connection_pattern})"
                    new_table[key] = action

            # Track and apply changes
            self._on_table_change(new_table)
            self.set_table_data(new_table)
            
        except Exception as e:
            logger.error(f"Error randomizing table: {e}")
            messagebox.showerror("Error", f"Failed to randomize table: {e}")

    def set_table_data(self, table_data: Dict[str, Any]):
        """Set table data with change tracking"""
        if table_data != self.table_data:
            self._on_table_change(table_data)
            self._create_table_entries(table_data, self.table_type)
            
    def _add_new_rule(self):
        """Add a new blank rule entry"""
        row_frame = tk.Frame(self.scrollable_frame)
        row_frame.pack(fill="x", padx=5, pady=2)
        
        # Create blank entries based on rule type
        if self.table_type == 'state':
            num_components = 3  # current_state, active_neighbors, connected_neighbors
        else:
            num_components = 6  # self_state, neighbor_state, etc.
            
        for i in range(num_components):
            tk.Entry(row_frame, width=10).grid(row=0, column=i, padx=2)
            
        # Add value entry/dropdown
        if self.table_type == 'state':
            value_var = tk.StringVar(value="0")
            tk.Entry(row_frame, textvariable=value_var, width=10).grid(
                row=0, column=num_components, padx=2)
        else:
            value_var = tk.StringVar(value="maintain")
            tk.OptionMenu(row_frame, value_var, "add", "remove", "maintain").grid(
                row=0, column=num_components, padx=2)
                    
        # Add delete button
        tk.Button(row_frame, text="X", command=lambda: row_frame.destroy()).grid(
            row=0, column=num_components+1, padx=2)

    def _set_tooltip(self, widget: Union[tk.Entry, ValidatedEntry], message: str):
        """Safely set tooltip on widget"""
        if hasattr(widget, '_tooltip'):
            widget._tooltip.destroy()  # type: ignore
        tooltip = ToolTip(widget, message)
        setattr(widget, '_tooltip', tooltip)

    def _remove_tooltip(self, widget: Union[tk.Entry, ValidatedEntry]):
        """Safely remove tooltip from widget"""
        if hasattr(widget, '_tooltip'):
            widget._tooltip.destroy()  # type: ignore
            delattr(widget, '_tooltip')
            
    def _validate_rule_table_entry(self, table_type: str, key_components: List[str], value: str) -> bool:
        """Validate a single rule table entry"""
        try:
            if table_type == 'state':
                # Validate state table entry
                if len(key_components) != 3:
                    raise ValueError("State rule key must have 3 components: (current_state, active_neighbors, connected_neighbors)")
                    
                current_state = int(key_components[0])
                active_neighbors = int(key_components[1])
                connected_neighbors = int(key_components[2])
                new_state = int(value)
                
                # Validate ranges
                if current_state not in [-1, 0, 1]:
                    raise ValueError("Current state must be -1, 0, or 1")
                if active_neighbors < 0:
                    raise ValueError("Active neighbors cannot be negative")
                if connected_neighbors < 0:
                    raise ValueError("Connected neighbors cannot be negative")
                if connected_neighbors > active_neighbors:
                    raise ValueError("Connected neighbors cannot exceed active neighbors")
                if new_state not in [-1, 0, 1]:
                    raise ValueError("New state must be -1, 0, or 1")
                    
            else:  # edge table
                # Validate edge table entry
                if len(key_components) != 6:
                    raise ValueError("Edge rule key must have 6 components: (self_state, neighbor_state, self_connections, neighbor_connections, self_clustering, neighbor_clustering)")
                    
                self_state = int(key_components[0])
                neighbor_state = int(key_components[1])
                self_connections = int(key_components[2])
                neighbor_connections = int(key_components[3])
                self_clustering = float(key_components[4])
                neighbor_clustering = float(key_components[5])
                
                # Validate ranges
                if self_state not in [0, 1]:
                    raise ValueError("Self state must be 0 or 1")
                if neighbor_state not in [0, 1]:
                    raise ValueError("Neighbor state must be 0 or 1")
                if self_connections < 0:
                    raise ValueError("Self connections cannot be negative")
                if neighbor_connections < 0:
                    raise ValueError("Neighbor connections cannot be negative")
                if not 0 <= self_clustering <= 1:
                    raise ValueError("Self clustering must be between 0 and 1")
                if not 0 <= neighbor_clustering <= 1:
                    raise ValueError("Neighbor clustering must be between 0 and 1")
                if value not in ['add', 'remove', 'maintain']:
                    raise ValueError("Edge action must be 'add', 'remove', or 'maintain'")
                    
            return True
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return False

    def _validate_rule_table_consistency(self, table_data: dict, table_type: str) -> bool:
        """Validate consistency across all rule table entries"""
        try:
            if table_type == 'state':
                # Check state transition consistency
                for key, value in table_data.items():
                    if key == 'default':
                        continue
                        
                    components = key.strip('()').split(',')
                    current_state = int(components[0])
                    active_neighbors = int(components[1])
                    
                    # Check for conflicting rules
                    conflicting_keys = [k for k in table_data.keys() 
                                    if k != 'default' and k != key and 
                                    k.strip('()').split(',')[0] == str(current_state) and
                                    k.strip('()').split(',')[1] == str(active_neighbors)]
                                    
                    if conflicting_keys:
                        raise ValueError(f"Conflicting rules found for state {current_state} with {active_neighbors} active neighbors")
                        
            else:  # edge table
                # Check edge rule consistency
                for key, value in table_data.items():
                    if key == 'default':
                        continue
                        
                    components = key.strip('()').split(',')
                    self_state = int(components[0])
                    neighbor_state = int(components[1])
                    
                    # Check for conflicting rules
                    conflicting_keys = [k for k in table_data.keys()
                                    if k != 'default' and k != key and
                                    k.strip('()').split(',')[0] == str(self_state) and
                                    k.strip('()').split(',')[1] == str(neighbor_state)]
                                    
                    if conflicting_keys:
                        raise ValueError(f"Conflicting rules found for states ({self_state}, {neighbor_state})")
                        
            return True
            
        except ValueError as e:
            logger.error(f"Consistency error: {e}")
            return False

    def _validate_rule_table_completeness(self, table_data: dict, table_type: str) -> bool:
        """Validate that the rule table covers all necessary cases"""
        try:
            if table_type == 'state':
                # Check for required state transitions
                required_states = [-1, 0, 1]
                for current_state in required_states:
                    # Check basic transitions
                    basic_cases = [(current_state, 0, 0)]  # Add more basic cases as needed
                    for case in basic_cases:
                        key = f"({case[0]}, {case[1]}, {case[2]})"
                        if key not in table_data and 'default' not in table_data:
                            raise ValueError(f"Missing rule for basic case: {key}")
                            
            else:  # edge table
                # Check for required edge rules
                required_states = [0, 1]
                for self_state in required_states:
                    for neighbor_state in required_states:
                        # Check basic connections
                        basic_cases = [(self_state, neighbor_state, 0, 0, 0.0, 0.0)]
                        for case in basic_cases:
                            key = f"({case[0]}, {case[1]}, {case[2]}, {case[3]}, {case[4]:.1f}, {case[5]:.1f})"
                            if key not in table_data and 'default' not in table_data:
                                raise ValueError(f"Missing rule for basic case: {key}")
                                
            return True
            
        except ValueError as e:
            logger.error(f"Completeness error: {e}")
            return False
            
    def _validate_entry(self, value: str, column_type: str) -> bool:
        """Validate entry based on column type and parameter info"""
        try:
            # Get current row data
            row = self._get_current_row_data()
            if not row:
                return True  # Allow typing if row is not complete
            
            # Validate entire entry
            return self._validate_rule_table_entry(
                self.table_type,
                row,
                value if column_type == 'action' else row[-1]
            )
        except Exception:
            return False
      
    def _create_entry(self, parent: tk.Frame, column_type: str) -> tk.Entry:
        """Create validated entry widget"""
        vcmd = (self.register(lambda val: self._validate_entry(val, column_type)), '%P')
        entry = tk.Entry(parent, validate='key', validatecommand=vcmd)
        return entry

    def _validate_rule_table_cell(self, table_type: str, column_index: int, value: str, widget: Optional[tk.Entry] = None) -> bool:
        """Validate rule table cell value"""
        try:
            is_valid = False
            error_msg = ""
            
            if table_type == 'state':
                if column_index == 0:  # current_state
                    try:
                        val = int(value)
                        is_valid = val in [-1, 0, 1]
                        error_msg = "Must be -1, 0, or 1"
                    except ValueError:
                        error_msg = "Must be an integer"
                elif column_index in [1, 2]:  # active_neighbors, connected_neighbors
                    try:
                        val = int(value)
                        is_valid = val >= 0
                        error_msg = "Must be non-negative"
                    except ValueError:
                        error_msg = "Must be an integer"
                elif column_index == 3:  # new_state
                    try:
                        val = int(value)
                        is_valid = val in [-1, 0, 1]
                        error_msg = "Must be -1, 0, or 1"
                    except ValueError:
                        error_msg = "Must be an integer"
            else:  # edge table
                if column_index in [0, 1]:  # self_state, neighbor_state
                    try:
                        val = int(value)
                        is_valid = val in [0, 1]
                        error_msg = "Must be 0 or 1"
                    except ValueError:
                        error_msg = "Must be an integer"
                elif column_index == 2:  # connection_pattern
                    is_valid = all(c in '01' for c in value) and len(value) == 8
                    error_msg = "Must be 8 digits of 0s and 1s"
                elif column_index == 3:  # action
                    is_valid = value in ["add", "remove", "maintain"]
                    error_msg = "Must be 'add', 'remove', or 'maintain'"

            # Apply visual feedback if widget provided
            if widget is not None:
                if is_valid:
                    widget.configure(bg='white')
                    self._remove_tooltip(widget)
                else:
                    widget.configure(bg='#ffebeb')  # Light red
                    self._set_tooltip(widget, error_msg)

            return is_valid

        except Exception as e:
            logger.error(f"Error validating rule table cell: {e}")
            return False

    def _create_validated_entry(self, parent: tk.Frame, column_index: int) -> ValidatedEntry:
        """Create entry widget with validation for rule table"""
        entry = ValidatedEntry(parent)  # Use ValidatedEntry instead of tk.Entry
        
        def validate(value: str) -> bool:
            # During typing, allow empty value
            if not value:
                self._remove_tooltip(entry)
                entry.configure(bg='white')
                return True
            return self._validate_rule_table_cell(self.table_type, column_index, value, entry)
            
        def on_focus_out(event):
            value = entry.get()
            if not value:  # Don't allow empty on focus out
                self._set_tooltip(entry, "Value cannot be empty")
                entry.configure(bg='#ffebeb')
                entry.delete(0, tk.END)
                entry.insert(0, self._get_last_valid_value(column_index))
            elif not self._validate_rule_table_cell(self.table_type, column_index, value, entry):
                entry.delete(0, tk.END)
                entry.insert(0, self._get_last_valid_value(column_index))
            
        vcmd = (parent.register(validate), '%P')
        entry.config(validate='key', validatecommand=vcmd)
        entry.bind('<FocusOut>', on_focus_out)
        entry.bind('<Return>', on_focus_out)
        
        return entry

    def _get_last_valid_value(self, column_index: int) -> str:
        """Get the last valid value for a column"""
        if self.table_type == 'state':
            if column_index == 0:  # current_state
                return "0"
            elif column_index in [1, 2]:  # active/connected neighbors
                return "0"
            elif column_index == 3:  # new_state
                return "0"
        else:  # edge table
            if column_index in [0, 1]:  # states
                return "0"
            elif column_index == 2:  # connection pattern
                return "00000000"
            elif column_index == 3:  # action
                return "maintain"
        return ""
    
    def _create_table_entries(self, table_data: Dict[str, Any], table_type: str):
        """Create table entries with validation"""
        # Clear existing entries
        for child in self.scrollable_frame.winfo_children():
            child.destroy()
            
        # Create headers
        header_frame = tk.Frame(self.scrollable_frame)
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        headers = []
        if table_type == 'state':
            headers = ["Current State", "Active Neighbors", "Connected Neighbors", "New State"]
        else:
            headers = ["Self State", "Neighbor State", "Connection Pattern", "Action"]
            
        for i, header in enumerate(headers):
            tk.Label(header_frame, text=header, font=("TkDefaultFont", 10, "bold")).grid(
                row=0, column=i, padx=5, pady=2)
                
        # Create entries for each rule
        for key, value in table_data.items():
            if key == "default":
                continue
                
            row_frame = tk.Frame(self.scrollable_frame)
            row_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Parse key components
            components = key.strip("()").split(",")
            
            # Create entries for each component
            for i, comp in enumerate(components):
                entry = self._create_validated_entry(row_frame, i)
                entry.insert(0, comp.strip())
                entry.grid(row=0, column=i, padx=2)
                
            # Create value entry/dropdown
            if table_type == 'state':
                value_entry = self._create_validated_entry(row_frame, len(components))
                value_entry.insert(0, str(value))
                value_entry.grid(row=0, column=len(components), padx=2)
            else:
                value_var = tk.StringVar(value=value)
                tk.OptionMenu(row_frame, value_var, "add", "remove", "maintain").grid(
                    row=0, column=len(components), padx=2)
                    
            # Add delete button
            tk.Button(row_frame, text="X", 
                    command=lambda r=row_frame: self._safe_delete_row(r)).grid(
                        row=0, column=len(components)+1, padx=2)

        # Add default value row
        default_frame = tk.Frame(self.scrollable_frame)
        default_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(default_frame, text="Default:").pack(side="left", padx=5)
        if table_type == 'state':
            default_entry = self._create_validated_entry(default_frame, 3)  # Use new_state validation
            default_entry.insert(0, str(table_data.get("default", "0")))
            default_entry.pack(side="left")
        else:
            default_var = tk.StringVar(value=table_data.get("default", "maintain"))
            tk.OptionMenu(default_frame, default_var, "add", "remove", "maintain").pack(side="left")
     
class SimulationGUI:
    """GUI for cellular automaton simulation using new controller architecture"""

    # Global application lock
    _app_lock = threading.Lock()
    change_tracker: Optional[ChangeTracker]
    _params_modified: bool

    def __init__(self, rule_name: str, initial_params: Optional[Dict[str, Any]] = None):
        """Initialize the simulation GUI"""
        self._initialization_complete = False  # Add flag at very start
        self._pending_rule_data = None  # Add this to fix the attribute error
        self._tk_destroyed = False 
        self.change_tracker = ChangeTracker()  # Initialize change tracker
        self._parameter_entries = {}
        self._params_modified = False
        self.param_canvas = None
        self.canvas_frame = None
        self.metadata_frame = None

        logger.debug("SimulationGUI.__init__ START")

        # Initialize matplotlib backend first
        plt.style.use('default')
        matplotlib.use('TkAgg')

        # Create root window first
        self.root = tk.Tk()
        self.root.configure(bg=GlobalSettings.Colors.BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Calculate proper window width to ensure control panel is fully visible
        control_panel_width = GlobalSettings.Visualization.RULE_EDITOR_METADATA_WIDTH
        total_padding = (GlobalSettings.Visualization.WINDOW_PADDING * 2) + (GlobalSettings.Visualization.CONTROL_PADDING * 2)
        window_width = (GlobalSettings.Visualization.WINDOW_SIZE[0] + 
                    control_panel_width + 
                    total_padding)
        window_height = GlobalSettings.Visualization.WINDOW_SIZE[1]
        
        # Set initial window size and position
        self.root.geometry(f"{window_width}x{window_height}")

        # Create main container with padding
        self.main_frame = tk.Frame(self.root, bg=GlobalSettings.Colors.BACKGROUND)
        self.main_frame.pack(fill=tk.BOTH, expand=True, 
                            padx=GlobalSettings.Visualization.WINDOW_PADDING,
                            pady=GlobalSettings.Visualization.WINDOW_PADDING)
        
        # Create visualization frame first with full size
        self.viz_frame = tk.Frame(self.main_frame, bg=GlobalSettings.Colors.BACKGROUND)
        self.viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                        padx=(0, GlobalSettings.Visualization.CONTROL_PADDING))

        # Initialize visualization components IMMEDIATELY to ensure they appear at the right size
        self.fig = Figure(figsize=GlobalSettings.Visualization.FIGURE_SIZE)
        self.fig.set_facecolor(GlobalSettings.Colors.BACKGROUND)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().configure(bg=GlobalSettings.Colors.BACKGROUND)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create empty initial axes with proper background
        if GlobalSettings.Simulation.DIMENSION_TYPE == Dimension.THREE_D:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self._current_azim = 45  # Set initial azimuth for depth view
            self._current_elev = 30   # Set initial elevation
            self.ax.view_init(elev=self._current_elev, azim=self._current_azim)
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_aspect('equal')
        
        # Configure axes appearance immediately
        self.ax.set_facecolor(GlobalSettings.Colors.BACKGROUND)
        self.ax.grid(False)
        self.ax.set_axisbelow(True)
        self.ax.tick_params(colors='gray')
        
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        # Force an immediate draw to ensure canvas is visible
        self.canvas.draw()
        
        # Create control panel with fixed width
        self.control_panel = tk.Frame(
            self.main_frame, 
            width=control_panel_width,
            bg='#404040'
        )
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_panel.pack_propagate(False)
        
        # Create canvas and scrollbar for control panel
        self.control_canvas = tk.Canvas(
            self.control_panel,
            bg='#404040',
            highlightthickness=0,
            width=control_panel_width - 20
        )
        self.control_scrollbar = tk.Scrollbar(
            self.control_panel,
            orient="vertical",
            command=self.control_canvas.yview
        )
        
        # Create scrollable frame
        self.scrollable_control_frame = tk.Frame(self.control_canvas, bg='#404040')
        
        # Configure canvas scrolling
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        
        # Create window in canvas
        self.control_canvas.create_window(
            (0, 0),
            window=self.scrollable_control_frame,
            anchor="nw",
            width=control_panel_width - 25
        )
        
        # Pack scrollbar and canvas
        self.control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Force update to ensure proper layout
        self.root.update_idletasks()

        try:
            # Now continue with the rest of initialization
            self._init_variables()

            # Initialize rule selection variables
            self.rule_type_var = tk.StringVar()
            self.rule_instance_var = tk.StringVar()
            
            # Initialize dimension and neighborhood variables
            self.dimension_var = tk.StringVar()
            self.neighborhood_var = tk.StringVar()
            
            # Initialize tiebreaker variables
            self.enable_tiebreakers_var = tk.BooleanVar(value=GlobalSettings.ENABLE_TIEBREAKERS)
            self.tiebreaker_type_var = tk.StringVar()
            
            # Initialize continuous run variables
            self.run_continuously = tk.BooleanVar(value=False)
            self.num_steps_var = tk.StringVar(value=str(GlobalSettings.Simulation.NUM_STEPS))

            # Initialize controller but don't initialize state yet
            self._init_controller(rule_name)

            # Set initial values for rule selection
            self.rule_type_var.set(RuleLibraryInfo.get_rule_category(self.controller.rule_name))
            self.rule_instance_var.set(self.controller.rule_name)
            
            # Set initial values for dimension and neighborhood
            self.dimension_var.set(self.controller.dimension_type.name)
            self.neighborhood_var.set(self.controller.neighborhood_type.name)
            
            # Set initial tiebreaker type
            self.tiebreaker_type_var.set(self.controller.rule.get_param('tiebreaker_type', 'RANDOM'))
            
            # Setup event handlers now that the canvas is ready
            self._setup_event_handlers()

            # Setup controls directly in the control panel
            logger.debug("Setting up control panel content")
            print("About to setup control panel content")
            self._setup_controls_content(self.scrollable_control_frame)
            print("Control panel content setup completed")
            
            # Force update to ensure proper layout
            self.root.update_idletasks()
            
            # Initialize change tracking
            self._parameter_entries = {}
            
            # Initialize change tracker with current parameters
            if hasattr(self, 'controller') and self.controller.rule:
                self.change_tracker.initialize(self.controller.rule.params)
            
            # Verify integration
            if not self._verify_change_detection_integration():
                logger.warning("Change detection integration verification failed")
            
            # Start periodic change detection
            self._check_for_changes()

            # Set additional state variables
            self._tk_destroyed = False
            self._is_shutting_down = False
            self._is_cleaning_up = False
            self._stop_requested = False
            self._toggle_debounce_timer = None
            self._is_toggling = False
            self._last_start_time = None

            # Now mark initialization as complete and proceed with final initialization
            self._initialization_complete = True
            self._complete_initialization()

            logger.debug("SimulationGUI initialization sequence completed")

        except Exception as e:
            logger.error(f"Error in GUI setup: {str(e)}")
            raise
                
    def _complete_initialization(self):
            """Complete initialization after GUI is ready"""
            try:
                logger.debug("Completing final initialization")

                # Now initialize grid state
                self.controller.grid = Grid(
                    self.controller.dimensions,
                    self.controller.neighborhood_type,
                    self.controller.dimension_type,
                    rule=self.controller.rule,
                    unique_id=self.controller._unique_id  # Access unique_id from controller
                )

                # Apply pending rule data if exists
                if hasattr(self, '_pending_rule_data'):
                    if self._pending_rule_data and 'params' in self._pending_rule_data:
                        self.controller.rule.params = copy.deepcopy(self._pending_rule_data['params'])
                    delattr(self, '_pending_rule_data')

                # Initialize state
                density = GlobalSettings.Simulation.INITIAL_NODE_DENSITY
                self._initialize_random_state(density)

                # Initialize edges AFTER the state is initialized
                # Set the edge_initialization parameter to RANDOM if it's not already set
                if 'edge_initialization' not in self.controller.rule.params:
                    self.controller.rule.params['edge_initialization'] = 'RANDOM'
                
                # Initialize edges AFTER the state is initialized
                self.controller.grid.initialize_edges() # KEEP THIS LINE

                # Force initial render
                self._safe_plot_update()

                logger.info("Initialization complete")

            except Exception as e:
                logger.error(f"Error in final initialization: {e}")
                raise
                
              
    def _init_matplotlib(self):
        """Initialize matplotlib backend"""
        plt.style.use('default')
        matplotlib.use('TkAgg')

    def _init_window_and_frames(self):
        """Initialize main window and frame structure"""
        # Create root window
        self.root = tk.Tk()
        self.root.configure(bg=GlobalSettings.Colors.BACKGROUND)
        
        # Bind window closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Calculate proper window width to ensure control panel is fully visible
        control_panel_width = GlobalSettings.Visualization.RULE_EDITOR_METADATA_WIDTH
        total_padding = (GlobalSettings.Visualization.WINDOW_PADDING * 2) + (GlobalSettings.Visualization.CONTROL_PADDING * 2)
        window_width = (GlobalSettings.Visualization.WINDOW_SIZE[0] + 
                    control_panel_width + 
                    total_padding)
        window_height = GlobalSettings.Visualization.WINDOW_SIZE[1]
        
        # Set initial window size and position
        self.root.geometry(f"{window_width}x{window_height}")

        # Create main container with padding
        self.main_frame = tk.Frame(self.root, bg=GlobalSettings.Colors.BACKGROUND)
        self.main_frame.pack(fill=tk.BOTH, expand=True, 
                            padx=GlobalSettings.Visualization.WINDOW_PADDING,
                            pady=GlobalSettings.Visualization.WINDOW_PADDING)
        
        # Create visualization frame first - make it expand fully
        self.viz_frame = tk.Frame(self.main_frame, bg=GlobalSettings.Colors.BACKGROUND)
        self.viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                        padx=(0, GlobalSettings.Visualization.CONTROL_PADDING))
        
        # Create loading frame - this will initially show the background color
        # and be replaced by the canvas
        self.loading_frame = tk.Frame(self.viz_frame, bg=GlobalSettings.Colors.BACKGROUND)
        self.loading_frame.pack(fill=tk.BOTH, expand=True)
                                    
        # Create control panel with fixed width
        self.control_panel = tk.Frame(
            self.main_frame, 
            width=control_panel_width,
            bg='#404040'
        )
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_panel.pack_propagate(False)
        
        # Create canvas and scrollbar
        self.control_canvas = tk.Canvas(
            self.control_panel,
            bg='#404040',
            highlightthickness=0,
            width=control_panel_width - 20
        )
        self.control_scrollbar = tk.Scrollbar(
            self.control_panel,
            orient="vertical",
            command=self.control_canvas.yview
        )
        
        # Create scrollable frame
        self.scrollable_control_frame = tk.Frame(self.control_canvas, bg='#404040')
        
        # Configure canvas scrolling
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        
        # Create window in canvas
        self.control_canvas.create_window(
            (0, 0),
            window=self.scrollable_control_frame,
            anchor="nw",
            width=control_panel_width - 25
        )
        
        # Pack scrollbar and canvas
        self.control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Force update to ensure proper layout
        self.root.update_idletasks()
        
    def _bind_mousewheel(self):
        """Bind mousewheel to scrolling for macOS"""
        def _on_mousewheel(event):
            # Get the widget that triggered the event
            widget = event.widget
            
            # Check if the widget is within the control panel
            parent = widget
            while parent is not None:
                if parent == self.control_panel:
                    # We're in the control panel, process the scroll
                    if event.delta:
                        self.control_canvas.yview_scroll(int(-1 * (event.delta/120)), "units")
                    return
                parent = parent.master

        # Bind to the control panel itself
        self.control_panel.bind('<MouseWheel>', _on_mousewheel)
        
        # Bind to the canvas
        self.control_canvas.bind('<MouseWheel>', _on_mousewheel)
        
        # Bind to the scrollable frame
        self.scrollable_control_frame.bind('<MouseWheel>', _on_mousewheel)
        
        # Recursively bind to all widgets in the control panel
        def bind_recursive(widget):
            widget.bind('<MouseWheel>', _on_mousewheel)
            for child in widget.winfo_children():
                bind_recursive(child)
        
        # Apply bindings to all widgets
        bind_recursive(self.control_panel)
        
        # Bind to the root window when mouse enters control panel
        def _on_enter_control_panel(event):
            self.root.bind_all('<MouseWheel>', _on_mousewheel)
            
        def _on_leave_control_panel(event):
            self.root.unbind_all('<MouseWheel>')
        
        # Set up enter/leave bindings for the control panel
        self.control_panel.bind('<Enter>', _on_enter_control_panel)
        self.control_panel.bind('<Leave>', _on_leave_control_panel)
            
    def _update_scroll_bindings(self):
        """Update scroll bindings when new widgets are added"""
        if hasattr(self, 'control_panel') and not self._tk_destroyed:
            self._bind_mousewheel()
            # Schedule next update
            self.root.after(1000, self._update_scroll_bindings)
            
    def _on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        
        # Rebind mousewheel to all widgets
        self._bind_mousewheel()
        
    def _bound_to_mousewheel(self, event):
        """Bind mousewheel when mouse enters the frame"""
        self.control_canvas.bind_all("<MouseWheel>", 
            lambda e: self.control_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

    def _unbound_to_mousewheel(self, event):
        """Unbind mousewheel when mouse leaves the frame"""
        self.control_canvas.unbind_all("<MouseWheel>")
    
    def _init_variables(self):
            """Initialize instance variables"""
            self.step_button_pressed = False
            self.anim = None
            self.step_count = 0
            self.running = False
            self.paused = False
            self.new_nodes = set()
            self.new_edges = set()
            self.node_marker = MarkerStyle('o')
            self.step_delay = GlobalSettings.Simulation.STEP_DELAY
            self.last_frame_time = time.time()
            self.current_frame_time = 0.0
            self.undo_stack = []
            self.redo_stack = []
            self._plot_cache = {}
            self._last_plot_hash = None
            self.tooltip = None
            self._view_state = {
                'xlim': None,
                'ylim': None,
                'zlim': None,
                'elev': 30,
                'azim': 0,
                'zoom_factor': 1.0
            }
            self._last_mouse_pos = None
            self._current_azim = 0
            self._current_elev = 30
            self.rotation_enabled = False
            self.panning_enabled = False
            self.performance_stats = {
                'step_times': [],
                'render_times': [],
                'frame_times': []
            }
            self._update_lock = threading.Lock()
            self._parameter_lock = threading.Lock()
            self._simulation_event = threading.Event()
            self._gui_event = threading.Event()
            self._stop_event = threading.Event()
            self._is_updating = False
            self._last_update_time = time.time()
            self._update_queue = Queue()
            self._frame_times = []
            self._event_bindings = []
            self._is_initialized = False
            self._is_setting_defaults = False
            self.neighborhood_types = {
                "VON_NEUMANN": NeighborhoodType.VON_NEUMANN,
                "MOORE": NeighborhoodType.MOORE,
                "HEX": NeighborhoodType.HEX,
                "HEX_PRISM": NeighborhoodType.HEX_PRISM
            }
            
            # ADD THIS SECTION
            self.dimension_types = {
                "TWO_D": Dimension.TWO_D,
                "THREE_D": Dimension.THREE_D
            }

    def _init_controller(self, rule_name: str):
        """Initialize simulation controller"""
        try:
            self.controller = SimulationController(
                rule_name=rule_name,
                initialize_state=False
            )
            # Now correctly set the controller reference
            self.controller.rule.metadata = dataclasses.replace(
                self.controller.rule.metadata,
                controller=self.controller
            )
            
            # Load parameters
            rule_data = RuleLibraryManager.get_rule(rule_name)
            if 'params' in rule_data:
                logger.debug(f"Loading parameters for {rule_name}: {rule_data['params']}")
                self.controller.rule.params = copy.deepcopy(rule_data['params'])
                
        except Exception as e:
            logger.error(f"Error initializing controller: {e}")
            raise

    def _init_matplotlib_components(self):
        """Initialize matplotlib figure and canvas"""
        # Create figure
        self.fig = Figure(figsize=GlobalSettings.Visualization.FIGURE_SIZE)
        self.fig.set_facecolor(GlobalSettings.Colors.BACKGROUND)
        
        # Create canvas with proper background color
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().configure(bg=GlobalSettings.Colors.BACKGROUND)
        
        # Create axes
        if GlobalSettings.Simulation.DIMENSION_TYPE == Dimension.THREE_D:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.view_init(elev=self._current_elev, azim=self._current_azim)
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_aspect('equal')
            
        # Configure axes
        self.ax.set_facecolor(GlobalSettings.Colors.BACKGROUND)
        self.ax.grid(False)
        self.ax.set_axisbelow(True)
        self.ax.tick_params(colors='gray')
        
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        # Pack canvas with expand=True to fill the entire area from the start
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Force a draw to ensure it's visible
        self.canvas.draw()
        self.root.update_idletasks()
            
    def _init_gui_controls(self):
        """Initialize GUI control elements"""
        # Initialize variables
        self.highlight_var = tk.BooleanVar(value=True)
        self.rule_type_var = tk.StringVar()
        self.rule_instance_var = tk.StringVar()
        self.dimension_var = tk.StringVar()
        self.neighborhood_var = tk.StringVar()
        self.initial_conditions_var = tk.StringVar(value="Random")
        
        # Create control frames
        self.scrollable_control_frame = tk.Frame(self.control_panel, bg=self.control_panel.cget('bg'))
        
        # Create labels
        self.step_label = tk.Label(self.control_panel, text="Step: 0", bg=self.control_panel.cget('bg'), fg='white')
        self.perf_label = tk.Label(self.control_panel, text="Avg Step: 0.0ms", bg=self.control_panel.cget('bg'), fg='white')
        self.grid_size_label = tk.Label(self.control_panel, text="Size: 0", bg=self.control_panel.cget('bg'), fg='white')
                
    def _init_scales(self):
        """Initialize scale widgets"""
        # Grid size scale
        self.grid_size_scale = tk.Scale(self.control_panel, from_=10, to=100, orient=tk.HORIZONTAL, 
                                    command=self._on_grid_size_change)
        
        # Speed scale
        self.speed_scale = tk.Scale(self.control_panel, from_=10, to=1000, orient=tk.HORIZONTAL,
                                command=self._on_speed_change)
        
        # Node spacing scale
        self.spacing_scale = tk.Scale(self.control_panel, from_=0.0, 
                                    to=GlobalSettings.Visualization.MAX_NODE_SPACING,
                                    resolution=0.1, orient=tk.HORIZONTAL, length=200,
                                    showvalue=True, digits=2, sliderrelief=tk.SOLID)
        
        # Node density scale
        self.node_density_scale = tk.Scale(
            self.control_panel,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            command=self._on_node_density_change
        )
        self.node_density_scale.set(GlobalSettings.Simulation.INITIAL_NODE_DENSITY)
        
        # Edge density scale
        self.edge_density_scale = tk.Scale(
            self.control_panel,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            command=self._on_edge_density_change
        )
        self.edge_density_scale.set(GlobalSettings.Simulation.INITIAL_EDGE_DENSITY)

    def _init_buttons(self):
        """Initialize button widgets"""
        self.start_button = tk.Button(self.control_panel, text="Start", width=15, 
                                    command=self.toggle_simulation)
        self.pause_button = tk.Button(self.control_panel, text="Pause", width=15,
                                    command=self.toggle_pause)
        self.step_button = tk.Button(self.control_panel, text="Step", width=15,
                                    command=self.step_button_clicked)
        self.edit_rule_button = tk.Button(self.control_panel, text="Edit Rule",
                                        command=lambda: self._create_rule_editor_window(self.controller.rule_name))
        self.zoom_in_button = tk.Button(self.control_panel, text="Zoom +", width=15,
                                    command=self.zoom_in)
        self.zoom_out_button = tk.Button(self.control_panel, text="Zoom -", width=15,
                                        command=self.zoom_out)
        self.save_button = tk.Button(self.control_panel, text="Save", width=15,
                                    command=self.save_state)
        self.load_button = tk.Button(self.control_panel, text="Load", width=15,
                                    command=self.load_state)

    def _init_selectors(self):
        """Initialize selector widgets"""
        self.initial_conditions_selector = tk.OptionMenu(
            self.control_panel, self.initial_conditions_var,
            "Random", "2D - Circle", "2D - Square", "3D - Sphere", "3D - Cube",
            command=lambda value: self._on_initial_conditions_change(self.initial_conditions_var.get())
        )
        
        self.rule_type_selector = tk.OptionMenu(
            self.control_panel, self.rule_type_var,
            *RuleLibraryInfo.get_rule_categories().keys(),
            command=lambda value: self._on_rule_type_change(self.rule_type_var.get())
        )
        
        self.rule_instance_selector = tk.OptionMenu(
            self.control_panel, self.rule_instance_var,
            *RuleLibraryInfo.get_rule_names(),
            command=lambda value: self._on_rule_instance_change(self.rule_instance_var.get())
        )
                                    
    def _initialize_variables(self):
        """Helper method to initialize variables"""
        self.step_button_pressed = False
        self.anim = None
        self.step_count = 0
        self.running = False
        self.paused = False
        self.new_nodes = set()
        self.new_edges = set()
        self.node_marker = MarkerStyle('o')
        self.step_delay = GlobalSettings.Simulation.STEP_DELAY
        self.last_frame_time = time.time()
        self.current_frame_time = 0.0
        self.undo_stack = []
        self.redo_stack = []
        self._plot_cache = {}
        self._last_plot_hash = None
        self.tooltip = None
        self._view_state = {
            'xlim': None,
            'ylim': None,
            'zlim': None,
            'elev': 30,
            'azim': 0,
            'zoom_factor': 1.0
        }
        self.pan_start_x = None
        self.pan_start_y = None
        self._last_mouse_pos = None
        self._current_azim = 0
        self._current_elev = 30
        self.rotation_enabled = False
        self.panning_enabled = False
        self.performance_stats = {
            'step_times': [],
            'render_times': [],
            'frame_times': []
        }
        self._update_lock = threading.Lock()
        self._parameter_lock = threading.Lock()
        self._simulation_event = threading.Event()
        self._gui_event = threading.Event()
        self._stop_event = threading.Event()
        self._is_updating = False
        self._last_update_time = time.time()
        self._update_queue = Queue()
        self._frame_times = []
        self._event_bindings = []
        self._is_initialized = False
        self._is_setting_defaults = False
        self.neighborhood_types = {
            "VON_NEUMANN": NeighborhoodType.VON_NEUMANN,
            "MOORE": NeighborhoodType.MOORE,
            "HEX": NeighborhoodType.HEX,
            "HEX_PRISM": NeighborhoodType.HEX_PRISM
        }
        
    def _initialize_controller(self, rule_name: str):
        """Helper method to initialize the SimulationController"""
        try:
            rule_data = RuleLibraryManager.get_rule(rule_name)
            metadata_dict = {k: v for k, v in rule_data.items() if k != 'params'}
            metadata = RuleMetadata(**metadata_dict)
            self.controller = SimulationController(
                rule_name=rule_name,
                initialize_state=False
            )
            self.controller.rule = RuleLibrary.create_rule(rule_name, metadata)
            # Remove this line:
            # self.controller.rule.metadata.controller = self
            if 'params' in rule_data:
                logger.debug(f"Loading parameters for {rule_name}: {rule_data['params']}")
                self.controller.rule.params = copy.deepcopy(rule_data['params'])
        except ValueError as e:
            logger.error(f"Error initializing SimulationController: {e}")
            messagebox.showerror("Error", f"Failed to initialize SimulationController: {e}")
            raise

    def _setup_gui_layout(self, main_frame, viz_frame):
        """Helper method to setup the GUI layout"""
        # Create visualization frame with background color
        viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                    padx=GlobalSettings.Visualization.WINDOW_PADDING,
                    pady=GlobalSettings.Visualization.WINDOW_PADDING)

        # Create control panel with background color
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y,
                            padx=GlobalSettings.Visualization.WINDOW_PADDING,
                            pady=GlobalSettings.Visualization.WINDOW_PADDING)
  

        # Pack canvas after axes are configured
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _backup_rules_file(self) -> Optional[str]:
        """Create backup of rules.json before modifications"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(APP_PATHS['rules_backups'], f"rules_backup_{timestamp}.json")
            rules_path = RuleLibraryManager.get_instance().library_path #self.library_path #os.path.join(APP_PATHS['config'], 'rules.json')

            logger.info(f"Creating backup of rules.json at: {backup_path}")

            # Read current rules.json
            with open(rules_path, 'r') as f:
                current_data = json.load(f)
                
            # Save backup
            with open(backup_path, 'w') as f:
                json.dump(current_data, f, indent=2)
            
            logger.info("Backup created successfully")
            return backup_path

        except Exception as e:
            logger.error(f"Error creating rules backup: {e}")
            messagebox.showerror("Error", f"Failed to create backup: {e}")
            return None
         
    def _on_window_close(self, event=None):
        """Handle main window closing with change detection"""
        try:
            if self.change_tracker and self.change_tracker.is_modified():
                response = messagebox.askyesnocancel(
                    "Unsaved Changes",
                    "There are unsaved changes. Would you like to save them before closing?",
                    icon='warning'
                )
                
                if response is None:  # Cancel
                    return
                elif response:  # Yes, save
                    current_rule_data = self._get_current_rule_data()
                    if not self._save_rule_with_confirmation(self.controller.rule_name, current_rule_data):
                        return

            self.cleanup()
            
        except Exception as e:
            logger.error(f"Error handling window close: {e}")
            self.cleanup()  # Force cleanup on error
        
    def cleanup(self):
        """Clean up GUI resources"""
        try:
            # Check if already cleaned up
            if hasattr(self, '_cleanup_complete'):
                return

            # Check for unsaved changes before proceeding
            response = True  # Initialize response to True
            if hasattr(self, 'change_tracker') and self.change_tracker is not None and self.change_tracker.is_modified():
                # Only show the messagebox if the root window still exists
                if hasattr(self, 'root') and self.root is not None and not self._tk_destroyed:
                    response = messagebox.askyesnocancel(
                        "Unsaved Changes",
                        "There are unsaved changes. Would you like to save them before closing?",
                        icon='warning'
                    )
                    
                    if response is None:  # Cancel
                        return
                    elif response:  # Yes, save
                        try:
                            current_rule_data = self._get_current_rule_data()
                            if not self._save_rule_with_confirmation(self.controller.rule_name, current_rule_data):
                                return  # Don't proceed if save failed or was cancelled
                        except Exception as e:
                            logger.error(f"Error saving changes during cleanup: {e}")
                            if not messagebox.askyesno("Error", 
                                "Failed to save changes. Would you like to continue closing anyway?"):
                                return
                    
                # Stop simulation first
                self.running = False
                if hasattr(self, 'controller'):
                    self.controller.interrupt_requested = True
                
                # Clean up matplotlib resources first
                plt.close('all')
                
                # Clean up event handlers
                if hasattr(self, '_event_bindings'):
                    self._cleanup_event_handlers()
                
                # Clean up animation
                if hasattr(self, 'anim') and self.anim is not None:
                    self.anim.event_source.stop()
                    self.anim = None
                    
                # Clean up event loop
                if hasattr(self, 'event_loop') and self.event_loop is not None:
                    self._cleanup_event_loop()
                    
                # Clean up controller after event loop
                if hasattr(self, 'controller'):
                    # Check if controller has a cleanup method and call it
                    if hasattr(self.controller, 'cleanup') and callable(getattr(self.controller, 'cleanup')):
                        self.controller.cleanup()
                    else:
                        logger.warning("Controller does not have a cleanup method")
                    
                # Clean up change tracker
                if hasattr(self, 'change_tracker'):
                    self.change_tracker = None
                    
                if response:  # Yes, save
                    try:
                        backup_path = self._backup_rules_file()
                        if backup_path:
                            current_rule_data = self._get_current_rule_data()
                            if not self._save_rule_with_confirmation(self.controller.rule_name, current_rule_data):
                                return  # Don't proceed if save failed or was cancelled
                        else:
                            if not (hasattr(self, 'root') and self.root is not None and not self._tk_destroyed and messagebox.askyesno("Warning", 
                                "Failed to create backup. Would you like to continue closing anyway?")):
                                return
                    except Exception as e:
                        logger.error(f"Error saving changes during cleanup: {e}")
                    
                # Clean up root window last, but only if not already destroyed
                if hasattr(self, 'root') and self.root is not None and not self._tk_destroyed:
                    try:
                        self.root.quit()
                        self.root.destroy()
                        self._tk_destroyed = True
                        logger.info("Root window destroyed successfully")
                    except Exception as e:
                        logger.warning(f"Error destroying root window: {e}")
                    
                # Mark cleanup as complete
                self._cleanup_complete = True
                            
            except Exception as e:
                logger.error(f"Error in GUI cleanup: {e}")
                                            
    def on_closing(self):
        """Handle window closing event with change tracking"""
        try:
            # Check for unsaved changes first
            if self.change_tracker is not None and self.change_tracker.is_modified():
                response = messagebox.askyesnocancel(
                    "Unsaved Changes",
                    "There are unsaved changes. Would you like to save them before closing?",
                    icon='warning'
                )
                
                if response is None:  # Cancel
                    return
                elif response:  # Yes, save
                    backup_path = self._backup_rules_file()
                    if backup_path:
                        current_rule_data = self._get_current_rule_data()
                        if not self._save_rule_with_confirmation(self.controller.rule_name, current_rule_data):
                            return
                    else:
                        if not messagebox.askyesno("Warning", 
                            "Failed to create backup. Would you like to continue closing anyway?"):
                            return

            # Stop any running simulation
            self.running = False
            if hasattr(self, 'controller'):
                self.controller.interrupt_requested = True
            
            # Clean up event loop first
            if hasattr(self, 'event_loop') and self.event_loop is not None:
                self._cleanup_event_loop()
                
            # Clean up controller
            if hasattr(self, 'controller'):
                self.controller.cleanup()
                
            # Clean up matplotlib resources
            plt.close('all')
                
            # Clean up event handlers
            if hasattr(self, '_event_bindings'):
                self._cleanup_event_handlers()
                
            # Set destroyed flag
            self._tk_destroyed = True
            
            # Destroy root window and quit mainloop
            if hasattr(self, 'root') and self.root is not None:
                self.root.quit()
                self.root.destroy()
                
            logger.info("Application shutdown completed")
                
        except Exception as e:
            logger.error(f"Error during window closing: {str(e)}")
            # Force quit even if there was an error
            if hasattr(self, 'root') and self.root is not None:
                self.root.quit()
                self.root.destroy()
                                
    def _destroy_root(self):
        """Safely destroy root window"""
        try:
            if (hasattr(self, 'root') and self.root is not None and 
                self.root.winfo_exists()):
                self.root.quit()
                self.root.update_idletasks()  # Process any pending events
                self.root.destroy()
                logger.info("Root window destroyed successfully")
        except Exception as e:
            logger.warning(f"Error destroying root window: {e}")

    def _handle_closing(self):
        """Handle the actual closing sequence"""
        try:
            # Clean up everything except root window
            self.running = False
            if hasattr(self, 'controller'):
                self.controller.cleanup()
                
            # Clean up event handlers
            self._cleanup_event_handlers()
            
            # Clean up matplotlib resources
            plt.close('all')
            
            # Clean up event loop
            if hasattr(self, 'event_loop'):
                self._cleanup_event_loop()
                
            # Finally destroy root window
            if (hasattr(self, 'root') and self.root is not None and 
                self.root.winfo_exists()):
                self.root.quit()
                self.root.update_idletasks()
                self.root.destroy()
                logger.info("Application shutdown completed")
                
        except Exception as e:
            logger.error(f"Error in closing handler: {e}")

    def _complete_cleanup(self):
        """Complete the cleanup process"""
        try:
            # Clean up controller first
            if hasattr(self, 'controller'):
                self.controller.cleanup()
                
            # Clean up event handlers
            self._cleanup_event_handlers()
            
            # Clean up matplotlib resources
            plt.close('all')
            
            # Clean up event loop if it exists
            if hasattr(self, 'event_loop') and self.event_loop is not None:
                try:
                    if self.event_loop.is_running():
                        self.event_loop.call_soon_threadsafe(self.event_loop.stop)
                    if hasattr(self, 'loop_thread') and self.loop_thread.is_alive():
                        self.loop_thread.join(timeout=1.0)
                except Exception as e:
                    logger.warning(f"Error stopping event loop: {e}")
                    
            # Finally destroy the root window
            if hasattr(self, 'root') and self.root is not None:
                try:
                    self.root.quit()
                    self.root.destroy()
                except Exception as e:
                    logger.warning(f"Error destroying root window: {e}")
                    
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error in final cleanup: {e}")
                                                                
    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            self.cleanup()
        except Exception as e:
            if logger:  # Check if logger exists before using
                logger.error(f"Error in GUI deletion: {e}")
            else:
                print(f"Error in GUI deletion: {e}")

    def _get_change_tracker(self) -> Optional[ChangeTracker]:
        """Safely get change tracker"""
        return self.change_tracker if hasattr(self, 'change_tracker') else None
 
    @log_errors
    def update_plot_2d(self, frame):
        """Update 2D visualization"""
        return self._update_2d_plot()
                  
    @log_errors
    def update_plot_3d(self, frame):
        """Update 3D visualization"""
        return self._update_3d_plot()
   
    def on_spacing_release(self, event):
        """Only update spacing when slider is released"""
        if self.spacing_scale is not None: # ADDED CHECK
            value = self.spacing_scale.get()
            self._on_spacing_change(str(value))
                                             
    def _setup_gui(self):
        """Initialize the GUI components"""
        try:
            # Force matplotlib to initialize its backend first
            plt.close('all')
            
            # Create main window
            logger.debug("SimulationGUI._setup_gui: Creating main window")
            self.root.title("Network Automata Simulation")
            
            # Set window background color immediately
            self.root.configure(bg=GlobalSettings.Colors.BACKGROUND)

            # Use GlobalSettings for window size
            window_width = GlobalSettings.Visualization.WINDOW_SIZE[0] + 300  # Add space for control panel
            window_height = GlobalSettings.Visualization.WINDOW_SIZE[1]
            
            # Set initial window size and position
            self.root.geometry(f"{window_width}x{window_height}")

            # Create main container with background color
            logger.debug("SimulationGUI._setup_gui: Creating main frame")
            main_frame = tk.Frame(self.root, 
                                width=window_width, 
                                height=window_height,
                                bg=GlobalSettings.Colors.BACKGROUND)
            main_frame.pack(fill=tk.BOTH, expand=True)
            main_frame.pack_propagate(False)

            # Create visualization frame with background color
            logger.debug("SimulationGUI._setup_gui: Creating visualization frame")
            viz_frame = tk.Frame(main_frame, 
                                bg=GlobalSettings.Colors.BACKGROUND)
            viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                        padx=GlobalSettings.Visualization.WINDOW_PADDING,
                        pady=GlobalSettings.Visualization.WINDOW_PADDING)

            # Create control panel with background color
            logger.debug("SimulationGUI._setup_gui: Creating control panel")
            self.control_panel = tk.Frame(main_frame, 
                                        width=300,
                                        bg='#404040')  # Dark grey background
            self.control_panel.pack(side=tk.RIGHT, fill=tk.Y,
                                padx=GlobalSettings.Visualization.WINDOW_PADDING,
                                pady=GlobalSettings.Visualization.WINDOW_PADDING)
           

            # Create matplotlib figure
            logger.debug("SimulationGUI._setup_gui: Creating matplotlib figure")
            plt.style.use('default')  # Reset to default style
            self.fig = Figure(figsize=GlobalSettings.Visualization.FIGURE_SIZE)
            self.fig.set_facecolor(GlobalSettings.Colors.BACKGROUND)
            
            # Create canvas before axes
            logger.debug("SimulationGUI._setup_gui: Creating canvas")
            self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
            self.canvas.get_tk_widget().configure(bg=GlobalSettings.Colors.BACKGROUND)
            
            # Create appropriate axes
            logger.debug("SimulationGUI._setup_gui: Creating axes")
            if GlobalSettings.Simulation.DIMENSION_TYPE == Dimension.THREE_D:
                self.ax = self.fig.add_subplot(111, projection='3d')
                self._current_azim = 0
                self._current_elev = 30
                self.ax.view_init(elev=self._current_elev, azim=self._current_azim)
            else:
                self.ax = self.fig.add_subplot(111)
                self.ax.set_aspect('equal')
                
            # Set axes properties
            self.ax.set_facecolor(GlobalSettings.Colors.BACKGROUND)
            self.ax.grid(False)
            self.ax.set_axisbelow(True)
            self.ax.tick_params(colors='gray')
            
            # Remove spines
            for spine in self.ax.spines.values():
                spine.set_visible(False)

            # Pack canvas after axes are configured
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Force an immediate draw
            self.canvas.draw()
            self.root.update_idletasks()

            # Create scrollable control frame
            self.scrollable_control_frame = tk.Frame(self.control_panel, bg=self.control_panel.cget('bg'))
            
            # Create canvas for scrolling
            self.control_canvas = tk.Canvas(self.scrollable_control_frame, bg=self.control_panel.cget('bg'), highlightthickness=0)
            self.control_scrollbar = tk.Scrollbar(self.scrollable_control_frame, orient="vertical", command=self.control_canvas.yview)
            self.control_scrollable_frame = tk.Frame(self.control_canvas, bg=self.control_panel.cget('bg'))

            # Configure scrolling
            self.control_scrollable_frame.bind(
                "<Configure>",
                lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
            )

            self.control_canvas.create_window((0, 0), window=self.control_scrollable_frame, anchor="nw")
            self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
            
            # Bind the canvas to update the scrollregion
            self.control_canvas.bind("<Configure>", lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))

            # Pack scrollbar and canvas
            self.control_scrollbar.pack(side="right", fill="y")
            self.control_canvas.pack(side="left", fill="both", expand=True)
            self.scrollable_control_frame.pack(fill=tk.BOTH, expand=True)

            # Setup controls
            self._setup_controls_content(self.control_scrollable_frame)
            
            # Perform post-setup tasks
            self._post_setup_controls()

        except Exception as e:
            logger.error(f"Error in GUI setup: {str(e)}")
            raise
        finally:
            # Ensure that the GUI is marked as initialized, even if there was an error
            self._is_initialized = True
                                                                  
    def _setup_controls(self):
        """Setup control panel with buttons and labels in vertical layout"""
        try:
            self._is_setting_defaults = True  # Set flag
            logger.debug("Starting _setup_controls")

            # Create main control section
            control_section = tk.LabelFrame(self.control_panel, text="Controls")
            control_section.pack(fill=tk.X, padx=5, pady=5)
            logger.debug("Created and packed control section")
            
            # Status section (Create this BEFORE triggering rule selection)
            status_section = tk.LabelFrame(self.control_panel, text="Status")
            status_section.pack(fill=tk.X, padx=5, pady=5)
            logger.debug("Created and packed status section")
            
            self.step_label = tk.Label(status_section, text="Step: 0")
            self.step_label.pack(pady=2)
            logger.debug("Created and packed step label")
            
            self.perf_label = tk.Label(status_section, text="Avg Step: 0.0ms")
            self.perf_label.pack(pady=2)
            logger.debug("Created and packed perf label")
            
            # Add Visualization Options section
            viz_section = tk.LabelFrame(self.control_panel, text="Visualization")
            viz_section.pack(fill=tk.X, padx=5, pady=5)
            logger.debug("Created and packed viz section")

            # Add highlight toggle
            self.highlight_var = tk.BooleanVar(value=True)  # Default to on
            self.highlight_checkbox = tk.Checkbutton(
                viz_section,
                text="Show Highlights",
                variable=self.highlight_var,
                command=self._on_highlight_toggle
            )
            self.highlight_checkbox.pack(pady=2)
            logger.debug("Created and packed highlight checkbox")
                
            # Simulation controls
            self.start_button = tk.Button(control_section, 
                                        text="Start",
                                        width=15,
                                        command=self.toggle_simulation)
            self.start_button.pack(pady=2)
            logger.debug("Created and packed start button")
            
            self.pause_button = tk.Button(control_section,
                                        text="Pause",
                                        width=15,
                                        command=self.toggle_pause)
            self.pause_button.pack(pady=2)
            logger.debug("Created and packed pause button")
            
            self.step_button = tk.Button(control_section,
                                        text="Step",
                                        width=15,
                                        command=self.step_button_clicked)
            self.step_button.pack(pady=2)
            logger.debug("Created and packed step button")
            
            # Add reset button
            self.reset_button = tk.Button(control_section,
                                        text="Reset",
                                        width=15,
                                        command=self.reset_simulation)
            self.reset_button.pack(pady=2)
            logger.debug("Created and packed reset button")

            # Rule selection section
            rule_section = tk.LabelFrame(self.control_panel, text="Rule Selection")
            rule_section.pack(fill=tk.X, padx=5, pady=5)
            logger.debug("Created and packed rule section")

            initial_rule_type = RuleLibraryInfo.get_rule_category(self.controller.rule_name)
            self.rule_type_var = tk.StringVar(value=initial_rule_type)
            self.rule_type_selector = tk.OptionMenu(
                rule_section,
                self.rule_type_var,
                *RuleLibraryInfo.get_rule_categories().keys(),
                command=lambda value: self._on_rule_type_change(self.rule_type_var.get())
            )
            self.rule_type_selector.configure(width=25)
            self.rule_type_selector['menu'].configure(font=('TkDefaultFont', 10))
            self.rule_type_selector.pack(fill=tk.X, padx=5, pady=2)
            logger.debug("Created and packed rule type selector")

            self.rule_instance_var = tk.StringVar(value=self.controller.rule_name)
            self.rule_instance_selector = tk.OptionMenu(
                rule_section,
                self.rule_instance_var,
                *RuleLibraryInfo.get_rules_in_category(initial_rule_type),
                command=lambda _: self._on_rule_instance_change(self.rule_instance_var.get())
            )
            self.rule_instance_selector.configure(width=25)
            self.rule_instance_selector['menu'].configure(font=('TkDefaultFont', 10))
            self.rule_instance_selector.pack(fill=tk.X, padx=5, pady=2)
            logger.debug("Created and packed rule instance selector")

            # Add parameter editing button
            param_button_frame = tk.Frame(self.control_panel)
            param_button_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.edit_rule_button = tk.Button(
                param_button_frame,
                text="Edit Rule",
                command=lambda: self._create_rule_editor_window(self.controller.rule_name)
            )
            self.edit_rule_button.pack(fill=tk.X)
            logger.debug("Created and packed edit rule button")
            
            # Add initial density sliders
            density_section = tk.LabelFrame(self.control_panel, text="Initial Density")
            density_section.pack(fill=tk.X, padx=5, pady=5)
            logger.debug("Created and packed density section")

            self.node_density_label = tk.Label(density_section, text="Node Density:")
            self.node_density_label.pack(pady=2)
            logger.debug("Creating node density scale")
            self.node_density_scale = tk.Scale(
                density_section,
                from_=0.0,
                to=1.0,
                resolution=0.05,
                orient=tk.HORIZONTAL,
                command=self._on_node_density_change
            )
            if self.node_density_scale is not None:
                self.node_density_scale.set(GlobalSettings.Simulation.INITIAL_NODE_DENSITY)
            self.node_density_scale.pack(fill=tk.X, padx=5, pady=2)
            logger.debug("Node density scale created and packed")

            self.edge_density_label = tk.Label(density_section, text="Edge Density:")
            if self.edge_density_scale is not None:
                self.edge_density_scale.pack(pady=2)
            logger.debug("Creating edge density scale")
            self.edge_density_scale = tk.Scale(
                density_section,
                from_=0.0,
                to=1.0,
                resolution=0.05,
                orient=tk.HORIZONTAL,
                command=self._on_edge_density_change
            )
            if self.edge_density_scale is not None:
                self.edge_density_scale.set(GlobalSettings.Simulation.INITIAL_EDGE_DENSITY)
            self.edge_density_scale.pack(fill=tk.X, padx=5, pady=2)
            logger.debug("Edge density scale created and packed")

            # Initial Conditions Section
            initial_conditions_section = tk.LabelFrame(self.control_panel, text="Initial Conditions")
            initial_conditions_section.pack(fill=tk.X, padx=5, pady=5)

            self.initial_conditions_var = tk.StringVar(value="Random")
            self.initial_conditions_selector = tk.OptionMenu(
                initial_conditions_section,
                self.initial_conditions_var,
                "Random",  # Default option
                "2D - Circle", "2D - Square",  # Example 2D options
                "3D - Sphere", "3D - Cube",  # Example 3D options
                command=lambda value: self._on_initial_conditions_change(value.get())
            )
            self.initial_conditions_selector.pack(fill=tk.X, padx=5, pady=2)

            # Save/Load section
            io_section = tk.LabelFrame(self.control_panel, text="Save/Load")
            io_section.pack(fill=tk.X, padx=5, pady=5)

            self.save_button = tk.Button(io_section,
                                        text="Save",
                                        width=15,
                                        command=self.save_state)
            self.save_button.pack(pady=2)

            self.load_button = tk.Button(io_section,
                                        text="Load",
                                        width=15,
                                        command=self.load_state)
            self.load_button.pack(pady=2)
                
            # Add Neighborhood Selection section
            neighborhood_section = tk.LabelFrame(self.control_panel, text="Neighborhood Type")
            neighborhood_section.pack(fill=tk.X, padx=5, pady=5)
            
            self.neighborhood_var = tk.StringVar(value=self.controller.neighborhood_type.name)
            
            # Dimension selection section
            dimension_section = tk.LabelFrame(self.control_panel, text="Dimension")
            dimension_section.pack(fill=tk.X, padx=5, pady=5)
            
            self.dimension_var = tk.StringVar(value=self.controller.dimension_type.name)
            
            # Grid size control
            grid_size_section = tk.LabelFrame(self.control_panel, text="Grid Size")
            grid_size_section.pack(fill=tk.X, padx=5, pady=5)
            
            current_size = GlobalSettings.Simulation.get_current_grid_size()
            self.grid_size_label = tk.Label(grid_size_section, text=f"Size: {current_size}")
            self.grid_size_label.pack(pady=2)
            if self.grid_size_label is not None:
                self.grid_size_label.config(text=f"Size: {current_size}")  # Ensure it's initialized
            
            self.grid_size_scale = tk.Scale(
                grid_size_section,
                from_=10,
                to=100,
                resolution=5,
                orient=tk.HORIZONTAL,
                command=self._on_grid_size_change)
            self.grid_size_scale.set(current_size)
            self.grid_size_scale.pack(fill=tk.X, padx=5, pady=2)
                
            # Speed control section
            speed_section = tk.LabelFrame(self.control_panel, text="Speed")
            speed_section.pack(fill=tk.X, padx=5, pady=5)
            
            self.speed_scale = tk.Scale(speed_section,
                                        from_=10,
                                        to=1000,
                                        orient=tk.HORIZONTAL,
                                        command=self._on_speed_change)
            self.speed_scale.set(1000)  # Set to maximum speed by default
            self.speed_scale.pack(fill=tk.X, padx=5, pady=2)
                
            # View control section
            view_section = tk.LabelFrame(self.control_panel, text="View Controls")
            view_section.pack(fill=tk.X, padx=5, pady=5)
                
            self.zoom_in_button = tk.Button(view_section,
                                        text="Zoom +",
                                        width=15,
                                        command=self.zoom_in)
            self.zoom_in_button.pack(pady=2)
            
            self.zoom_out_button = tk.Button(view_section,
                                            text="Zoom -",
                                            width=15,
                                            command=self.zoom_out)
            self.zoom_out_button.pack(pady=2)
                
            # Add spacing control section
            spacing_section = tk.LabelFrame(self.control_panel, text="Node Spacing")
            spacing_section.pack(fill=tk.X, padx=5, pady=5)
                
            def on_spacing_release(event):
                """Only update spacing when slider is released"""
                if self.spacing_scale is not None:
                    value = self.spacing_scale.get()
                self._on_spacing_change(str(value))
                
            self.spacing_scale = tk.Scale(
                spacing_section,
                from_=0.0,
                to=GlobalSettings.Visualization.MAX_NODE_SPACING,
                resolution=0.1,
                orient=tk.HORIZONTAL,
                length=200,
                showvalue=True,
                digits=2,
                sliderrelief=tk.SOLID
            )
            self.spacing_scale.set(GlobalSettings.Visualization.NODE_SPACING)
            self.spacing_scale.pack(fill=tk.X, padx=5, pady=2)
                
            # Bind to button release instead of continuous updates
            self.spacing_scale.bind("<ButtonRelease-1>", on_spacing_release)

            # Now that all GUI elements are created, trigger initial rule selection
            self._on_rule_instance_change(self.controller.rule_name)
        finally:
            self._is_setting_defaults = False # Clear flag
                                                                                                        
    def _setup_controls_content(self, parent_frame):
        """Master method to setup all control panel content"""
        logger.debug("Starting _setup_controls_content")
        print("Starting control panel setup")  # Debug print

        try:
            # Test widget to verify the frame is working
            test_label = tk.Label(parent_frame, text="Control Panel", bg='#404040', fg='white')
            test_label.pack(fill=tk.X, padx=5, pady=5)

            # Create all control sections in correct order
            self._create_control_section(parent_frame)
            self._create_status_section(parent_frame)
            self._create_step_control_section(parent_frame)
            self._create_speed_section(parent_frame)
            self._create_view_control_section(parent_frame)
            self._create_visualization_section(parent_frame)
            self._create_tiebreaker_section(parent_frame)
            self._create_rule_selection_section(parent_frame)
            self._create_neighborhood_section(parent_frame)
            self._create_dimension_section(parent_frame)
            self._create_initial_conditions_section(parent_frame)
            self._create_grid_size_section(parent_frame)
            self._create_density_section(parent_frame)
            self._create_spacing_section(parent_frame)
            self._create_save_load_section(parent_frame)


            # Force update to ensure proper layout
            parent_frame.update_idletasks()
            
            # Rebind mousewheel to all widgets
            self._bind_mousewheel()
            
            logger.debug("Control panel content setup completed")
            
        except Exception as e:
            logger.error(f"Error in _setup_controls_content: {e}")
            raise
        
    def _create_control_section(self, parent_frame):
        """Create main control buttons section"""
        control_section = tk.LabelFrame(parent_frame, text="Controls", bg='#404040', fg='white')
        control_section.pack(fill=tk.X, padx=5, pady=5)

        button_width = 15
        button_configs = [
            ("Start", self.toggle_simulation),
            ("Pause", self.toggle_pause),
            ("Step", self.step_button_clicked),
            ("Reset", self.reset_simulation)
        ]
        
        for text, command in button_configs:
            btn = tk.Button(control_section, text=text, width=button_width, command=command)
            btn.pack(fill=tk.X, padx=5, pady=2)
            if text == "Start":
                self.start_button = btn
            elif text == "Pause":
                self.pause_button = btn
            elif text == "Step":
                self.step_button = btn
            elif text == "Reset":
                self.reset_button = btn

    def _create_status_section(self, parent_frame):
        """Create status display section"""
        status_section = tk.LabelFrame(parent_frame, text="Status", bg='#404040', fg='white')
        status_section.pack(fill=tk.X, padx=5, pady=5)

        self.step_label = tk.Label(status_section, text="Step: 0", bg='#404040', fg='white')
        self.step_label.pack(fill=tk.X, padx=5, pady=2)

        self.perf_label = tk.Label(status_section, text="Avg Step: 0.0ms", bg='#404040', fg='white')
        self.perf_label.pack(fill=tk.X, padx=5, pady=2)

        self.grid_size_label = tk.Label(status_section, text="Size: 0", bg='#404040', fg='white')
        self.grid_size_label.pack(fill=tk.X, padx=5, pady=2)

    def _create_rule_selection_section(self, parent_frame):
        """Create rule selection section"""
        rule_section = tk.LabelFrame(parent_frame, text="Rule Selection", bg='#404040', fg='white')
        rule_section.pack(fill=tk.X, padx=5, pady=5)

        # Rule type dropdown
        initial_rule_type = RuleLibraryInfo.get_rule_category(self.controller.rule_name)
        self.rule_type_var.set(initial_rule_type)
        self.rule_type_selector = tk.OptionMenu(
            rule_section,
            self.rule_type_var,
            initial_rule_type,
            *RuleLibraryInfo.get_rule_categories().keys(),
            command=lambda value: self._on_rule_type_change(self.rule_type_var.get())
        )
        self.rule_type_selector.configure(width=25)
        self.rule_type_selector['menu'].configure(font=('TkDefaultFont', 10))
        self.rule_type_selector.pack(fill=tk.X, padx=5, pady=2)

        # Rule instance dropdown
        initial_rule = self.controller.rule_name
        self.rule_instance_var.set(initial_rule)
        self.rule_instance_selector = tk.OptionMenu(
            rule_section,
            self.rule_instance_var,
            initial_rule,
            *RuleLibraryInfo.get_rules_in_category(initial_rule_type),
            command=lambda _: self._on_rule_instance_change(self.rule_instance_var.get())
        )
        self.rule_instance_selector.configure(width=25)
        self.rule_instance_selector['menu'].configure(font=('TkDefaultFont', 10))
        self.rule_instance_selector.pack(fill=tk.X, padx=5, pady=2)

        # Edit rule button
        self.edit_rule_button = tk.Button(
            rule_section,
            text="Edit Rule",
            command=lambda: self._create_rule_editor_window(self.controller.rule_name)
        )
        self.edit_rule_button.pack(fill=tk.X, padx=5, pady=2)

    def _create_step_control_section(self, parent_frame):
        """Create step control section"""
        step_section = tk.LabelFrame(parent_frame, text="Step Control", bg='#404040', fg='white')
        step_section.pack(fill=tk.X, padx=5, pady=5)

        # Run continuously checkbox
        self.run_continuously = tk.BooleanVar(value=False)
        self.run_continuously_check = tk.Checkbutton(
            step_section,
            text="Run Continuously",
            variable=self.run_continuously,
            bg='#404040',
            fg='white',
            selectcolor='#404040',
            command=self._on_run_continuously_change
        )
        self.run_continuously_check.pack(fill=tk.X, padx=5, pady=2)

        # Number of steps entry
        steps_label = tk.Label(step_section, text="Number of Steps:", bg='#404040', fg='white')
        steps_label.pack(fill=tk.X, padx=5, pady=2)

        self.num_steps_var = tk.StringVar(value=str(GlobalSettings.Simulation.NUM_STEPS))
        self.num_steps_entry = tk.Entry(
            step_section, 
            textvariable=self.num_steps_var,
            state=tk.DISABLED if self.run_continuously.get() else tk.NORMAL
        )
        self.num_steps_entry.pack(fill=tk.X, padx=5, pady=2)

    def _create_tiebreaker_section(self, parent_frame):
        """Create tiebreaker settings section"""
        tiebreaker_section = tk.LabelFrame(parent_frame, text="Tiebreaker Settings", bg='#404040', fg='white')
        tiebreaker_section.pack(fill=tk.X, padx=5, pady=5)

        # Tiebreaker enable checkbox
        self.enable_tiebreakers_var = tk.BooleanVar(value=GlobalSettings.ENABLE_TIEBREAKERS)
        enable_tiebreakers_checkbox = tk.Checkbutton(
            tiebreaker_section,
            text="Enable Tiebreakers",
            variable=self.enable_tiebreakers_var,
            bg='#404040',
            fg='white',
            selectcolor='#404040',
            command=self._on_enable_tiebreakers_change
        )
        enable_tiebreakers_checkbox.pack(fill=tk.X, padx=5, pady=2)

        # Tiebreaker type selector
        initial_tiebreaker = self.controller.rule.get_param('tiebreaker_type', 'RANDOM')
        self.tiebreaker_type_var.set(initial_tiebreaker)
        tiebreaker_type_selector = tk.OptionMenu(
            tiebreaker_section,
            self.tiebreaker_type_var,
            initial_tiebreaker,
            *TieBreaker.__members__.keys(),
            command=lambda value: self._on_tiebreaker_type_change(self.tiebreaker_type_var.get())
        )
        tiebreaker_type_selector.pack(fill=tk.X, padx=5, pady=2)

    def _create_neighborhood_section(self, parent_frame):
        """Create neighborhood selection section"""
        neighborhood_section = tk.LabelFrame(parent_frame, text="Neighborhood Type", bg='#404040', fg='white')
        neighborhood_section.pack(fill=tk.X, padx=5, pady=5)

        # Get valid neighborhood types for current dimension
        valid_neighborhoods = self._get_valid_neighborhoods()
        
        self.neighborhood_var.set(self.controller.neighborhood_type.name)
        self.neighborhood_selector = tk.OptionMenu(
            neighborhood_section,
            self.neighborhood_var,
            self.controller.neighborhood_type.name,
            *valid_neighborhoods,
            command=lambda value: self._on_neighborhood_change(self.neighborhood_var.get())
        )
        self.neighborhood_selector.pack(fill=tk.X, padx=5, pady=2)

    def _create_dimension_section(self, parent_frame):
        """Create dimension selection section"""
        dimension_section = tk.LabelFrame(parent_frame, text="Dimension", bg='#404040', fg='white')
        dimension_section.pack(fill=tk.X, padx=5, pady=5)

        self.dimension_var.set(self.controller.dimension_type.name)
        self.dimension_selector = tk.OptionMenu(
            dimension_section,
            self.dimension_var,
            self.controller.dimension_type.name,
            *[dim.name for dim in Dimension],
            command=lambda value: self._on_dimension_change(self.dimension_var.get())
        )
        self.dimension_selector.pack(fill=tk.X, padx=5, pady=2)

    def _create_initial_conditions_section(self, parent_frame):
        """Create initial conditions selection section"""
        initial_conditions_section = tk.LabelFrame(parent_frame, text="Initial Conditions", bg='#404040', fg='white')
        initial_conditions_section.pack(fill=tk.X, padx=5, pady=5)

        self.initial_conditions_var = tk.StringVar(value="Random")
        self.initial_conditions_selector = tk.OptionMenu(
            initial_conditions_section,
            self.initial_conditions_var,
            "Random",  # Default option
            "2D - Circle", 
            "2D - Square",
            "3D - Sphere", 
            "3D - Cube",
            command=lambda value: self._on_initial_conditions_change(self.initial_conditions_var.get())
        )
        self.initial_conditions_selector.pack(fill=tk.X, padx=5, pady=2)

    def _create_grid_size_section(self, parent_frame):
        """Create grid size control section"""
        grid_size_section = tk.LabelFrame(parent_frame, text="Grid Size", bg='#404040', fg='white')
        grid_size_section.pack(fill=tk.X, padx=5, pady=5)

        current_size = GlobalSettings.Simulation.get_current_grid_size()
        self.grid_size_label = tk.Label(grid_size_section, 
                                    text=f"Size: {current_size}", 
                                    bg='#404040', 
                                    fg='white')
        self.grid_size_label.pack(fill=tk.X, padx=5, pady=2)

        self.grid_size_scale = tk.Scale(
            grid_size_section,
            from_=10,
            to=100,
            resolution=5,
            orient=tk.HORIZONTAL,
            command=self._on_grid_size_change
        )
        self.grid_size_scale.set(current_size)
        self.grid_size_scale.pack(fill=tk.X, padx=5, pady=2)

    def _create_density_section(self, parent_frame):
        """Create density control section"""
        density_section = tk.LabelFrame(parent_frame, text="Initial Density", bg='#404040', fg='white')
        density_section.pack(fill=tk.X, padx=5, pady=5)

        # Node density
        self.node_density_label = tk.Label(density_section, text="Node Density:", bg='#404040', fg='white')
        self.node_density_label.pack(fill=tk.X, padx=5, pady=2)

        self.node_density_scale = tk.Scale(
            density_section,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            command=self._on_node_density_change
        )
        self.node_density_scale.set(GlobalSettings.Simulation.INITIAL_NODE_DENSITY)
        self.node_density_scale.pack(fill=tk.X, padx=5, pady=2)

        # Edge density
        self.edge_density_label = tk.Label(density_section, text="Edge Density:", bg='#404040', fg='white')
        self.edge_density_label.pack(fill=tk.X, padx=5, pady=2)

        self.edge_density_scale = tk.Scale(
            density_section,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            command=self._on_edge_density_change
        )
        self.edge_density_scale.set(GlobalSettings.Simulation.INITIAL_EDGE_DENSITY)
        self.edge_density_scale.pack(fill=tk.X, padx=5, pady=2)

    def _create_speed_section(self, parent_frame):
        """Create speed control section"""
        speed_section = tk.LabelFrame(parent_frame, text="Speed", bg='#404040', fg='white')
        speed_section.pack(fill=tk.X, padx=5, pady=5)

        self.speed_scale = tk.Scale(
            speed_section,
            from_=10,
            to=1000,
            orient=tk.HORIZONTAL,
            command=self._on_speed_change
        )
        self.speed_scale.set(1000)  # Set to maximum speed by default
        self.speed_scale.pack(fill=tk.X, padx=5, pady=2)

    def _create_spacing_section(self, parent_frame):
        """Create node spacing control section"""
        spacing_section = tk.LabelFrame(parent_frame, text="Node Spacing", bg='#404040', fg='white')
        spacing_section.pack(fill=tk.X, padx=5, pady=5)

        self.spacing_scale = tk.Scale(
            spacing_section,
            from_=0.0,
            to=GlobalSettings.Visualization.MAX_NODE_SPACING,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=200,
            showvalue=True,
            digits=2,
            sliderrelief=tk.SOLID
        )
        self.spacing_scale.set(GlobalSettings.Visualization.NODE_SPACING)
        self.spacing_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Bind to button release instead of continuous updates
        self.spacing_scale.bind("<ButtonRelease-1>", self.on_spacing_release)
            
    def _create_visualization_section(self, parent_frame):
        """Create visualization options section"""
        viz_section = tk.LabelFrame(parent_frame, text="Visualization", bg='#404040', fg='white')
        viz_section.pack(fill=tk.X, padx=5, pady=5)

        # Highlight checkbox
        self.highlight_var = tk.BooleanVar(value=True)
        self.highlight_checkbox = tk.Checkbutton(
            viz_section,
            text="Show Highlights",
            variable=self.highlight_var,
            bg='#404040',
            fg='white',
            selectcolor='#404040',
            command=self._on_highlight_toggle
        )
        self.highlight_checkbox.pack(fill=tk.X, padx=5, pady=2)

    def _create_save_load_section(self, parent_frame):
        """Create save/load control section"""
        io_section = tk.LabelFrame(parent_frame, text="Save/Load", bg='#404040', fg='white')
        io_section.pack(fill=tk.X, padx=5, pady=5)

        button_width = 15
        
        self.save_button = tk.Button(
            io_section,
            text="Save",
            width=button_width,
            command=self.save_state
        )
        self.save_button.pack(fill=tk.X, padx=5, pady=2)

        self.load_button = tk.Button(
            io_section,
            text="Load",
            width=button_width,
            command=self.load_state
        )
        self.load_button.pack(fill=tk.X, padx=5, pady=2)

    def _create_view_control_section(self, parent_frame):
        """Create view control section"""
        view_section = tk.LabelFrame(parent_frame, text="View Controls", bg='#404040', fg='white')
        view_section.pack(fill=tk.X, padx=5, pady=5)

        # Zoom controls
        zoom_frame = tk.Frame(view_section, bg='#404040')
        zoom_frame.pack(fill=tk.X, padx=5, pady=2)

        self.zoom_in_button = tk.Button(
            zoom_frame,
            text="Zoom +",
            width=15,
            command=self.zoom_in
        )
        self.zoom_in_button.pack(side=tk.LEFT, padx=2, pady=2, expand=True, fill=tk.X)

        self.zoom_out_button = tk.Button(
            zoom_frame,
            text="Zoom -",
            width=15,
            command=self.zoom_out
        )
        self.zoom_out_button.pack(side=tk.RIGHT, padx=2, pady=2, expand=True, fill=tk.X)

        # Rotation controls for 3D
        if self.controller.dimension_type == Dimension.THREE_D:
            rotation_frame = tk.Frame(view_section, bg='#404040')
            rotation_frame.pack(fill=tk.X, padx=5, pady=2)

            self.rotation_enabled_var = tk.BooleanVar(value=False)
            self.rotation_checkbox = tk.Checkbutton(
                rotation_frame,
                text="Enable Rotation",
                variable=self.rotation_enabled_var,
                bg='#404040',
                fg='white',
                selectcolor='#404040'
            )
            self.rotation_checkbox.pack(fill=tk.X, pady=2)

            # Rotation speed control
            speed_label = tk.Label(rotation_frame, text="Rotation Speed:", bg='#404040', fg='white')
            speed_label.pack(fill=tk.X, pady=2)

            self.rotation_speed_scale = tk.Scale(
                rotation_frame,
                from_=0.0,
                to=2.0,
                resolution=0.1,
                orient=tk.HORIZONTAL,
                command=lambda v: setattr(GlobalSettings.Visualization, 'ROTATION_SPEED', float(v))
            )
            self.rotation_speed_scale.set(GlobalSettings.Visualization.ROTATION_SPEED)
            self.rotation_speed_scale.pack(fill=tk.X, pady=2)

        # Reset view button
        self.reset_view_button = tk.Button(
            view_section,
            text="Reset View",
            width=15,
            command=self._reset_view
        )
        self.reset_view_button.pack(fill=tk.X, padx=5, pady=2)
                
    def _post_setup_controls(self):
        """Perform post-setup tasks that require GUI elements to be initialized"""
        try:
            logger.debug("Entering _post_setup_controls")
            
            # Set initial values for GUI elements
            self.highlight_var.set(True)  # Default to on
            self.rule_type_var.set(self.controller.rule_name)
            self.rule_instance_var.set(self.controller.rule_name)
            
            # Set initial state of buttons
            self.start_button.config(text="Start")
            self.pause_button.config(text="Pause")
            
            # Set initial values for sliders
            if self.grid_size_scale is not None: # ADDED CHECK
                self.grid_size_scale.set(GlobalSettings.Simulation.get_current_grid_size())
            if self.speed_scale is not None: # ADDED CHECK
                self.speed_scale.set(1000)
            if self.spacing_scale is not None: # ADDED CHECK
                self.spacing_scale.set(GlobalSettings.Visualization.NODE_SPACING)
            if self.node_density_scale is not None:
                self.node_density_scale.set(GlobalSettings.Simulation.INITIAL_NODE_DENSITY)
            if self.edge_density_scale is not None:
                self.edge_density_scale.set(GlobalSettings.Simulation.INITIAL_EDGE_DENSITY)
            
            logger.debug("Post-setup controls completed successfully")
            
            # Set the _is_initialized flag to True after all setup is complete
            self._is_initialized = True
            
        except Exception as e:
            logger.error(f"Error in _post_setup_controls: {e}")
            raise
             
    def _create_standard_settings_section(self, parent_frame: tk.Frame, rule_name: str) -> tk.LabelFrame:
        """Create the standard settings section in the Rule Editor"""
        settings_frame = tk.LabelFrame(parent_frame, text="Standard Settings")
        
        

        # Dimension selection
        dimension_label = tk.Label(settings_frame, text="Dimension:")
        dimension_label.pack(anchor="w")
        dimension_var = tk.StringVar(value=self.controller.dimension_type.name)
        dimension_menu = tk.OptionMenu(settings_frame, dimension_var, *[dim.name for dim in Dimension],
                                        command=lambda value: self._on_dimension_change(value.get()))
        dimension_menu.pack(fill=tk.X, padx=5, pady=2)

        # Neighborhood selection
        neighborhood_label = tk.Label(settings_frame, text="Neighborhood Type:")
        neighborhood_label.pack(anchor="w")
        neighborhood_var = tk.StringVar(value=self.controller.neighborhood_type.name)
        neighborhood_menu = tk.OptionMenu(settings_frame, neighborhood_var, *[n.name for n in NeighborhoodType],
                                            command=lambda value: self._on_neighborhood_change(value.get()))
        neighborhood_menu.pack(fill=tk.X, padx=5, pady=2)

        # Initial Conditions selection
        initial_conditions_label = tk.Label(settings_frame, text="Initial Conditions:")
        initial_conditions_label.pack(anchor="w")
        initial_conditions_var = tk.StringVar(value="Random")  # Default value
        initial_conditions_menu = tk.OptionMenu(
            settings_frame,
            initial_conditions_var,
            "Random",  # Default option
            "2D - Circle", "2D - Square",  # Example 2D options
            "3D - Sphere", "3D - Cube",  # Example 3D options
            command=lambda value: self._on_initial_conditions_change(value.get())
        )
        initial_conditions_menu.pack(fill=tk.X, padx=5, pady=2)

        return settings_frame
                                                                                     
    def _update_plot_limits(self):
        """Dynamically update plot limits based on node coordinates"""
        try:
            # Get coordinates directly from coords_map
            grid_array = self.controller.grid.grid_array
            visible_mask = grid_array.ravel() > GlobalSettings.Visualization.NODE_VISIBILITY_THRESHOLD
            visible_indices = np.where(visible_mask)[0]
            
            if len(visible_indices) == 0:
                # Use grid dimensions to set fixed limits
                grid_size = GlobalSettings.Visualization.EDGE_SCALE * max(self.controller.dimensions)
                margin = grid_size * 0.1
                self.ax.set_xlim(-margin, grid_size + margin)
                self.ax.set_ylim(-margin, grid_size + margin)
                if isinstance(self.ax, Axes3DType): # type: ignore
                    self.ax.set_zlim(-margin, grid_size + margin)
                return
                
            # Calculate coordinates
            coords_map = self._calculate_node_coordinates()
            
            if not coords_map:
                # Use grid dimensions as fallback
                grid_size = GlobalSettings.Visualization.EDGE_SCALE * max(self.controller.dimensions)
                margin = grid_size * 0.1
                self.ax.set_xlim(-margin, grid_size + margin)
                self.ax.set_ylim(-margin, grid_size + margin)
                if isinstance(self.ax, Axes3DType): # type: ignore
                    self.ax.set_zlim(-margin, grid_size + margin)
                return
                
            # Set fixed tick intervals based on grid size
            grid_size = GlobalSettings.Visualization.EDGE_SCALE * max(self.controller.dimensions)
            tick_interval = max(1, grid_size // 10)  # At most 10 ticks
            
            # Set fixed ticks
            self.ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
            self.ax.yaxis.set_major_locator(MultipleLocator(tick_interval))
            
            if isinstance(self.ax, Axes3DType): # type: ignore
                self.ax.zaxis.set_major_locator(MultipleLocator(tick_interval))
                
            # Calculate limits with fixed margins
            margin = grid_size * 0.1
            self.ax.set_xlim(-margin, grid_size + margin)
            self.ax.set_ylim(-margin, grid_size + margin)
            
            if isinstance(self.ax, Axes3DType): # type: ignore
                self.ax.set_zlim(-margin, grid_size + margin)
                self.ax.set_box_aspect([1, 1, 1])

        except Exception as e:
            logger.error(f"Error updating plot limits: {e}")
            raise
            
    def _validate_state_rule_table(self, table_data: Dict[str, Any]) -> None:
        """Validate state rule table data"""
        if not isinstance(table_data, dict):
            raise ValueError("State rule table must be a dictionary")
            
        if "default" not in table_data:
            raise ValueError("State rule table must have a default value")
            
        for key, value in table_data.items():
            if key == "default":
                if not isinstance(value, int) or value not in [-1, 0, 1]:
                    raise ValueError("Default value must be -1, 0, or 1")
                continue
                
            # Validate key format
            try:
                components = key.strip("()").split(",")
                if len(components) != 3:
                    raise ValueError(f"Invalid key format: {key}")
                current_state = int(components[0])
                active_neighbors = int(components[1])
                connected_neighbors = int(components[2])
                if current_state not in [0, 1] or active_neighbors < 0 or connected_neighbors < 0:
                    raise ValueError(f"Invalid values in key: {key}")
            except (ValueError, IndexError):
                raise ValueError(f"Invalid key format: {key}")
                
            # Validate value
            if not isinstance(value, int) or value not in [-1, 0, 1]:
                raise ValueError(f"Invalid value for key {key}: {value}")
                
    def _convert_parameter_value(self, param_name: str, value_str: str) -> Any:
        """Convert parameter string value to appropriate type"""
        try:
            # Get rule data
            rule_data = RuleLibraryManager.get_rule(self.controller.rule_name)
            if rule_data is None:
                raise ValueError(f"Rule data not found for {self.controller.rule_name}")

            # Get parameter info
            param_info = rule_data['params'].get(param_name)
            if param_info is None:
                raise ValueError(f"Unknown parameter: {param_name}")

            # Get parameter type
            if isinstance(param_info, dict):
                param_type = param_info.get('type')
            else:
                # If param_info is not a dict, use the current value's type
                current_value = self.controller.rule.params.get(param_name)
                param_type = type(current_value)

            # Convert value based on type
            if param_type == float:
                new_value = float(value_str)
                # Check bounds if specified
                min_val = param_info.get('min') if isinstance(param_info, dict) else None
                max_val = param_info.get('max') if isinstance(param_info, dict) else None
                if min_val is not None and new_value < min_val:
                    raise ValueError(f"Value must be at least {min_val}")
                if max_val is not None and new_value > max_val:
                    raise ValueError(f"Value must be at most {max_val}")
            elif param_type == int:
                new_value = int(float(value_str))  # Allow float strings that represent integers
                # Check bounds if specified
                min_val = param_info.get('min') if isinstance(param_info, dict) else None
                max_val = param_info.get('max') if isinstance(param_info, dict) else None
                if min_val is not None and new_value < min_val:
                    raise ValueError(f"Value must be at least {min_val}")
                if max_val is not None and new_value > max_val:
                    raise ValueError(f"Value must be at most {max_val}")
            elif param_type == bool:
                value_str = value_str.lower()
                if value_str in ('true', '1', 'yes', 'on'):
                    new_value = True
                elif value_str in ('false', '0', 'no', 'off'):
                    new_value = False
                else:
                    raise ValueError("Boolean value must be true/false, 1/0, yes/no, or on/off")
            elif param_type == tuple:
                # Handle tuple parameters (like preferred_connection_range)
                try:
                    # Evaluate string as tuple, ensuring it's safe
                    new_value = ast.literal_eval(value_str)
                    if not isinstance(new_value, tuple):
                        raise ValueError("Value must be a tuple")
                except:
                    raise ValueError("Invalid tuple format")
            else:
                # For other types, use the string value as-is
                new_value = value_str

            return new_value

        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise ValueError(f"Invalid value: {str(e)}")

    def _validate_edge_rule_table(self, table_data: Dict[str, Any]) -> None:
        """Validate edge rule table data"""
        if not isinstance(table_data, dict):
            raise ValueError("Edge rule table must be a dictionary")
            
        if "default" not in table_data:
            raise ValueError("Edge rule table must have a default value")
            
        valid_actions = {"add", "remove", "maintain"}
        
        for key, value in table_data.items():
            if key == "default":
                if value not in valid_actions:
                    raise ValueError(f"Default value must be one of: {valid_actions}")
                continue
                
            # Validate key format
            try:
                components = key.strip("()").split(",")
                if len(components) != 6:  # self_state, neighbor_state, self_connections, neighbor_connections, self_clustering, neighbor_clustering
                    raise ValueError(f"Invalid key format: {key}")
                self_state = int(components[0])
                neighbor_state = int(components[1])
                self_connections = int(components[2])
                neighbor_connections = int(components[3])
                self_clustering = float(components[4])
                neighbor_clustering = float(components[5])
                if (self_state not in [0, 1] or neighbor_state not in [0, 1] or
                    self_connections < 0 or neighbor_connections < 0 or
                    not 0 <= self_clustering <= 1 or not 0 <= neighbor_clustering <= 1):
                    raise ValueError(f"Invalid values in key: {key}")
            except (ValueError, IndexError):
                raise ValueError(f"Invalid key format: {key}")
                
            # Validate value
            if value not in valid_actions:
                raise ValueError(f"Invalid value for key {key}: {value}")
                                    
    def _create_rule_editor_window(self, rule_name: str):
        try:
            # Create window and frames
            editor_window = tk.Toplevel(self.root)
            
            # Create scrollable canvas for parameters
            main_container = tk.Frame(editor_window)
            param_frame = tk.Frame(main_container)
            param_canvas = tk.Canvas(param_frame, bd=0, highlightthickness=0)
            param_scrollbar = tk.Scrollbar(param_frame, orient="vertical", command=param_canvas.yview)
            
            # Pack scrollbar and canvas
            param_scrollbar.pack(side="right", fill=tk.Y)
            param_canvas.pack(side="left", fill=tk.BOTH, expand=True)

            # Configure scrolling
            param_canvas.configure(yscrollcommand=param_scrollbar.set)

            # Store canvas references for scroll verification
            self.param_canvas = param_canvas  # After creating parameter canvas
            canvas_frame = tk.Frame(param_canvas)
            param_canvas.create_window((0, 0), window=canvas_frame, anchor="nw", 
                                    width=GlobalSettings.Visualization.RULE_EDITOR_PARAM_WIDTH - 50)
            self.canvas_frame = canvas_frame  # After creating canvas frame
            metadata_frame = tk.Frame(param_canvas)  # Define metadata_frame
            self.metadata_frame = metadata_frame  # After creating metadata frame 
            
            
            logger.info(f"Opening Rule Editor for rule: {rule_name}")

            # Create new top-level window
            #editor_window = tk.Toplevel(self.root) # REMOVE THIS LINE
            editor_window.title(f"Rule Editor - {rule_name}")

            # Make window independent
            editor_window.transient(self.root)  # Make it a child of the main window

            # Use correct GlobalSettings for window size
            window_width = GlobalSettings.Visualization.RULE_EDITOR_WINDOW_SIZE[0]
            window_height = GlobalSettings.Visualization.RULE_EDITOR_WINDOW_SIZE[1]

            # Position to the right of main window
            main_window_x = self.root.winfo_x()
            main_window_y = self.root.winfo_y()
            x_position = main_window_x + GlobalSettings.Visualization.WINDOW_SIZE[0] + 20
            y_position = main_window_y

            # Set geometry
            editor_window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

            # Create main container with two columns
            main_container = tk.Frame(editor_window)
            main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Configure column widths using correct GlobalSettings
            main_container.grid_columnconfigure(0, weight=0, minsize=GlobalSettings.Visualization.RULE_EDITOR_PARAM_WIDTH)
            main_container.grid_columnconfigure(1, weight=0, minsize=GlobalSettings.Visualization.RULE_EDITOR_METADATA_WIDTH)
            main_container.grid_rowconfigure(0, weight=1)

            # Left column - Parameters
            param_frame = tk.LabelFrame(main_container, text="Rule Parameters", 
                                    font=("TkDefaultFont", GlobalSettings.Visualization.RULE_EDITOR_HEADING_FONT_SIZE))
            param_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

            # Create scrollable canvas for parameters
            param_canvas = tk.Canvas(param_frame, bd=0, highlightthickness=0)
            param_scrollbar = tk.Scrollbar(param_frame, orient="vertical", command=param_canvas.yview)
            
            # Pack scrollbar and canvas
            param_scrollbar.pack(side="right", fill=tk.Y)
            param_canvas.pack(side="left", fill=tk.BOTH, expand=True)

            # Configure scrolling
            param_canvas.configure(yscrollcommand=param_scrollbar.set)

            # Create frame for parameters
            canvas_frame = tk.Frame(param_canvas)
            param_canvas.create_window((0, 0), window=canvas_frame, anchor="nw", 
                                            width=GlobalSettings.Visualization.RULE_EDITOR_PARAM_WIDTH - 50)

            # Setup scrolling for parameters
            self._setup_scrolling(param_canvas, canvas_frame)

            # Create standard settings section
            standard_settings_frame = self._create_standard_settings_section(canvas_frame, rule_name)
            standard_settings_frame.pack(fill=tk.X, padx=5, pady=5)

            # Create parameter fields
            parameter_entries = {}
            self._create_parameter_fields(canvas_frame, rule_name, parameter_entries)

            # Add bottom padding
            bottom_padding = tk.Frame(canvas_frame, 
                                            height=GlobalSettings.Visualization.RULE_EDITOR_BOTTOM_PADDING)
            bottom_padding.pack(fill=tk.X)

            # Right column - Metadata
            metadata_frame = tk.LabelFrame(main_container, text="Rule Metadata",
                                                font=("TkDefaultFont", GlobalSettings.Visualization.RULE_EDITOR_HEADING_FONT_SIZE))
            metadata_frame, metadata_fields = self._create_metadata_section(metadata_frame, rule_name, editor_window)
            metadata_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

            # Setup scrolling for metadata
            metadata_canvas = cast(tk.Canvas, metadata_frame.winfo_children()[0])  # Get the canvas
            metadata_scrollable = cast(tk.Frame, metadata_canvas.winfo_children()[0])  # Get the frame
            self._setup_scrolling(metadata_canvas, metadata_scrollable)

            # Create button section
            button_frame = self._create_button_section(main_container, rule_name, editor_window, 
                                                            metadata_fields, parameter_entries)
            button_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

            # Force an update of the scrollregion
            editor_window.update_idletasks()
            param_canvas.configure(scrollregion=param_canvas.bbox("all"))

            # Bind window closing event
            editor_window.protocol("WM_DELETE_WINDOW", lambda: self._on_rule_editor_close(editor_window, rule_name))

            logger.info(f"Rule Editor opened for rule: {rule_name}")

        except Exception as e:
            logger.error(f"Error creating rule editor: {str(e)}\nTraceback:\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"Failed to create rule editor: {e}")
                                                            
    def _setup_parameter_bindings(self, parameter_entries: Dict[str, Union[tk.Entry, RuleTableEditor]]):
        """Setup bindings for parameter updates"""
        def create_table_callback(name: str) -> Callable[[Any], None]:
            def callback(event: Any) -> None:
                self._update_parameter(name, "", parameter_entries)
            return callback
            
        def create_entry_callback(name: str, widget: tk.Entry) -> Callable[[Any], None]:
            def callback(event: Any) -> None:
                self._update_parameter(name, widget.get(), parameter_entries)
            return callback

        for param_name, widget in parameter_entries.items():
            if isinstance(widget, RuleTableEditor):
                # Bind table editor updates
                callback = create_table_callback(param_name)
                widget.bind("<<TableModified>>", callback, add="+")
            else:
                # Bind regular entry updates
                callback = create_entry_callback(param_name, widget)
                widget.bind("<FocusOut>", callback, add="+")
                widget.bind("<Return>", callback, add="+")
                
    def _destroy_editor_window(self, editor_window):
        """Destroy the editor window and its tooltip"""
        if hasattr(self, 'tooltip') and self.tooltip:
            self.tooltip.destroy()
        editor_window.destroy()
                    
    def _is_default_rule(self, rule_name: str) -> bool:
        """Check if a rule is one of the default rules"""
        try:
            # Load the rule from the library
            rule_data = RuleLibraryManager.get_rule(rule_name)
            return rule_data.get('default', False)
        except Exception as e:
            logger.error(f"Error getting rule data: {str(e)}")
            return False

    def _get_rule_metadata(self, rule_name: str) -> Dict[str, Any]:
        """Get metadata for a rule"""
        try:
            # Load the rule from the library
            rule_data = RuleLibraryManager.get_rule(rule_name)
            
            # Extract the metadata fields
            metadata = {
                'name': rule_data['name'],
                'category': rule_data['category'],
                'author': rule_data.get('author', 'Unknown'),
                'url': rule_data.get('url', ''),
                'email': rule_data.get('email', ''),
                'date_created': rule_data.get('date_created', datetime.now().strftime("%Y-%m-%d")),
                'date_modified': rule_data.get('date_modified', datetime.now().strftime("%Y-%m-%d")),
                'version': rule_data.get('version', '1.0'),
                'description': rule_data['description'],
                'tags': rule_data['tags'],
                'dimension_compatibility': rule_data.get('dimension_compatibility', []),
                'neighborhood_compatibility': rule_data.get('neighborhood_compatibility', []),
                'parent_rule': rule_data.get('parent_rule', None),
                'rating': rule_data.get('rating', None),
                'notes': rule_data.get('notes', None)
            }
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting rule metadata: {str(e)}")
            return {}
        
    def _highlight_field(self, entry: tk.Entry):
        """Highlight a parameter field"""
        original_bg = entry.cget('bg')
        original_fg = entry.cget('fg')  # Store original text color
        entry.config(bg="yellow", fg="black")  # Invert colors

        def restore_bg():
            entry.config(bg=original_bg, fg=original_fg)  # Restore original colors
        
        self.root.after(500, restore_bg)  # Highlight for 500ms
        
    def _create_parameter_section(self, main_container: tk.Frame, rule_name: str) -> Tuple[tk.LabelFrame, Dict[str, tk.Entry], tk.Frame, tk.Canvas]:
        """Create the parameter section of the Rule Editor window"""
        param_frame = tk.LabelFrame(main_container, text="Rule Parameters")
        param_canvas = tk.Canvas(param_frame, bd=0, highlightthickness=0)
        param_scrollbar = tk.Scrollbar(param_frame, orient="vertical", command=param_canvas.yview)
        param_scrollable = tk.Frame(param_canvas)
        
        # Configure scrolling
        param_scrollable.bind(
            "<Configure>",
            lambda e: param_canvas.configure(scrollregion=param_canvas.bbox("all"))
        )
        
        # Create window in canvas with fixed width
        canvas_frame = tk.Frame(param_scrollable)  # Container frame
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        param_canvas.create_window((0, 0), window=canvas_frame, anchor="nw", width=GlobalSettings.Visualization.RULE_EDITOR_PARAM_WIDTH // 2 - 50)
        
        # Pack scrollbar and canvas
        param_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        param_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure canvas scrolling
        param_canvas.configure(yscrollcommand=param_scrollbar.set)

        parameter_entries: Dict[str, tk.Entry] = {}  # Store references to the entry widgets

        # Get parameter information
        rule_data = RuleLibraryManager.get_rule(rule_name)
        if rule_data is None:
            logger.error(f"Rule data not found for {rule_name}")
            return param_frame, parameter_entries, canvas_frame, param_canvas

        # Extract parameter definitions from the rule data
        params = rule_data['params']
        
        # Create parameter fields with descriptions
        for param_name, param_info in params.items():
            # Format parameter name for display
            display_name = ' '.join(word.capitalize() for word in param_name.split('_'))
            
            # Get description from metadata
            description = param_info.get('description')
            
            # Parameter container
            param_container = tk.Frame(canvas_frame)
            param_container.pack(fill=tk.X, padx=5, pady=5)
            
            # Parameter name
            name_label = tk.Label(param_container, text=display_name, anchor="w", 
                                font=("TkDefaultFont", 10, "bold"))
            name_label.pack(fill=tk.X)
            
            # Parameter value entry
            entry = tk.Entry(param_container, name=param_name)
            entry.insert(0, str(self.controller.rule.params.get(param_name, "")))  # Get from controller
            entry.pack(fill=tk.X, pady=(2, 2))
            parameter_entries[param_name] = entry  # Store entry widget
            
            # Description text with proper wrapping
            if description:
                # Calculate wrap width based on container width
                param_container.update_idletasks()  # Ensure we have current dimensions
                wrap_width = int((param_container.winfo_width() - 30) * 0.75)  # 25% narrower
                if wrap_width <= 0:  # Fallback if width not available
                    wrap_width = 300
                
                desc_text = tk.Text(param_container, 
                                  wrap=tk.WORD,
                                  width=0,  # Let it expand to container width
                                  font=("TkDefaultFont", 9),
                                  relief=tk.FLAT,
                                  bg=param_container.cget('bg'))
                desc_text.insert('1.0', description)
                # Calculate required height based on content
                desc_text.update_idletasks()
                line_count = int(desc_text.index('end-1c').split('.')[0])
                desc_text.configure(height=line_count)
                desc_text.configure(state='disabled')
                desc_text.pack(fill=tk.X, padx=(15, 5), pady=(0, 5))
            
            # Add update binding
            update_callback = lambda e, name=param_name, widget=entry, parameter_entries=parameter_entries: self._update_parameter(name, widget.get(), parameter_entries)
            entry.bind('<FocusOut>', update_callback)
            entry.bind('<Return>', update_callback)
        
        # Add bottom padding
        bottom_padding = tk.Frame(canvas_frame, height=GlobalSettings.Visualization.RULE_EDITOR_BOTTOM_PADDING)
        bottom_padding.pack(fill=tk.X)

        return param_frame, parameter_entries, canvas_frame, param_canvas
                        
    def _create_metadata_section(self, metadata_frame: tk.LabelFrame, rule_name: str, editor_window: tk.Toplevel) -> Tuple[tk.LabelFrame, Dict[str, tk.Widget]]:
        """Create the metadata section of the Rule Editor window"""
        rule_data = RuleLibraryManager.get_rule(rule_name)
        if not rule_data:
            logger.error(f"No rule data found for {rule_name}")
            return metadata_frame, {}

        # Create scrollable frame
        canvas = tk.Canvas(metadata_frame, bd=0, highlightthickness=0)
        scrollbar = tk.Scrollbar(metadata_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", 
                            width=GlobalSettings.Visualization.RULE_EDITOR_METADATA_WIDTH - 30)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack scrollbar and canvas
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        metadata_fields = {}
        metadata_order = [
            'name', 'category', 'author', 'version', 'dimension_compatibility',
            'neighborhood_compatibility', 'description', 'tags', 'email', 'url',
            'date_created', 'date_modified', 'rating', 'notes'
        ]

        for field_name in metadata_order:
            if field_name in rule_data:
                field_container = tk.Frame(scrollable_frame)
                field_container.pack(fill=tk.X, padx=5, pady=5)

                label = tk.Label(field_container,
                            text=field_name.replace('_', ' ').title(),
                            font=("TkDefaultFont", GlobalSettings.Visualization.RULE_EDITOR_FIELD_FONT_SIZE, "bold"),
                            anchor="w")
                label.pack(fill=tk.X, padx=5, pady=(5,2))

                if field_name == 'description':
                    widget = tk.Text(field_container, height=4, wrap=tk.WORD, 
                                width=GlobalSettings.Visualization.RULE_EDITOR_FIELD_ENTRY_WIDTH)
                    widget.insert('1.0', str(rule_data[field_name]))
                else:
                    widget = tk.Entry(field_container, 
                                    width=GlobalSettings.Visualization.RULE_EDITOR_FIELD_ENTRY_WIDTH)
                    value = rule_data[field_name]
                    if isinstance(value, (list, tuple)):
                        widget.insert(0, ', '.join(str(x) for x in value))
                    else:
                        widget.insert(0, str(value))

                widget.pack(fill=tk.X, padx=5, pady=(2,5))
                metadata_fields[field_name] = widget

        # Enable mouse wheel scrolling for the metadata area
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        scrollable_frame.bind_all("<MouseWheel>", _on_mousewheel)

        return metadata_frame, metadata_fields

    def _create_button_section(self, main_container: tk.Frame, rule_name: str, editor_window: tk.Toplevel, 
                            metadata_fields: Dict[str, tk.Widget], parameter_entries: Dict[str, Union[tk.Entry, tk.OptionMenu, RuleTableEditor]]) -> tk.Frame:
        """Create button section with undo/redo and reset functionality"""
        button_frame = tk.Frame(main_container)
        button_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Left side buttons (editing controls)
        left_frame = tk.Frame(button_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Undo button
        undo_btn = tk.Button(left_frame, text="↶ Undo",
                            command=lambda: self._undo_parameter_change(parameter_entries))
        undo_btn.pack(side=tk.LEFT, padx=5)

        # Redo button
        redo_btn = tk.Button(left_frame, text="↷ Redo",
                            command=lambda: self._redo_parameter_change(parameter_entries))
        redo_btn.pack(side=tk.LEFT, padx=5)

        # Reset to Defaults button
        reset_btn = tk.Button(left_frame, text="Reset to Defaults",
                            command=lambda: self._reset_parameters_to_defaults(rule_name, parameter_entries))
        reset_btn.pack(side=tk.LEFT, padx=5)

        # Right side buttons (save/close)
        right_frame = tk.Frame(button_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.X)

        save_btn = tk.Button(right_frame, text="Save",
                            command=lambda: self._save_rule_with_confirmation(rule_name, self._get_current_rule_data(), editor_window))
        save_btn.pack(side=tk.LEFT, padx=5)

        close_btn = tk.Button(right_frame, text="Close",
                            command=lambda: self._on_rule_editor_close(editor_window, rule_name))
        close_btn.pack(side=tk.LEFT, padx=5)

        # Store buttons for state updates
        self._editor_buttons = {
            'undo': undo_btn,
            'redo': redo_btn,
            'save': save_btn,
            'reset': reset_btn
        }

        # Initial button states
        self._update_editor_buttons()

        return button_frame

    def _undo_parameter_change(self, parameter_entries: Dict[str, Union[tk.Entry, tk.OptionMenu, RuleTableEditor]]):
        """Handle undo button click"""
        try:
            if self.change_tracker:
                change = self.change_tracker.undo()
            else:
                change = None
            if change:
                param_name = change['param']
                old_value = change['old']
                
                # Update parameter in controller
                self.controller.rule.params[param_name] = old_value
                self.controller.rule.invalidate_cache()
                
                # Update UI
                widget = parameter_entries.get(param_name)
                if isinstance(widget, tk.Entry):
                    widget.delete(0, tk.END)
                    widget.insert(0, str(old_value))
                elif isinstance(widget, tk.OptionMenu):
                    widget.setvar(widget.cget('textvariable'), str(old_value))
                elif isinstance(widget, RuleTableEditor):
                    widget.set_table_data(old_value)
                
                # Update visualization if running
                if self.running:
                    self._safe_plot_update()
                
                # Update button states
                self._update_editor_buttons()
                
                logger.info(f"Undid change to {param_name}")
                
        except Exception as e:
            logger.error(f"Error in undo: {e}")
            messagebox.showerror("Error", f"Failed to undo change: {e}")

    def _redo_parameter_change(self, parameter_entries: Dict[str, Union[tk.Entry, tk.OptionMenu, RuleTableEditor]]):
        """Handle redo button click"""
        try:
            if self.change_tracker:
                change = self.change_tracker.redo()
            else:
                change = None
            if change:
                param_name = change['param']
                new_value = change['new']
                
                # Update parameter in controller
                self.controller.rule.params[param_name] = new_value
                self.controller.rule.invalidate_cache()
                
                # Update UI
                widget = parameter_entries.get(param_name)
                if isinstance(widget, tk.Entry):
                    widget.delete(0, tk.END)
                    widget.insert(0, str(new_value))
                elif isinstance(widget, tk.OptionMenu):
                    widget.setvar(widget.cget('textvariable'), str(new_value))
                elif isinstance(widget, RuleTableEditor):
                    widget.set_table_data(new_value)
                
                # Update visualization if running
                if self.running:
                    self._safe_plot_update()
                
                # Update button states
                self._update_editor_buttons()
                
                logger.info(f"Redid change to {param_name}")
                
        except Exception as e:
            logger.error(f"Error in redo: {e}")
            messagebox.showerror("Error", f"Failed to redo change: {e}")

    def _reset_parameters_to_defaults(self, rule_name: str, parameter_entries: Dict[str, Union[tk.Entry, tk.OptionMenu, RuleTableEditor]]):
        """Reset parameters to their default values"""
        try:
            if messagebox.askyesno("Confirm Reset", 
                                "Are you sure you want to reset all parameters to their default values?"):
                # Get default values from rule library
                rule_data = RuleLibraryManager.get_rule(rule_name)
                default_params = copy.deepcopy(rule_data['params'])
                
                # Store current values for undo
                old_params = copy.deepcopy(self.controller.rule.params)
                
                # Track the change
                if tracker := self._get_change_tracker(): tracker.track_change('all_params', old_params, default_params)
                
                # Update parameters
                self.controller.rule.params = default_params
                self.controller.rule.invalidate_cache()
                
                # Update all UI elements
                for param_name, widget in parameter_entries.items():
                    value = default_params.get(param_name)
                    if isinstance(widget, tk.Entry):
                        widget.delete(0, tk.END)
                        widget.insert(0, str(value))
                    elif isinstance(widget, tk.OptionMenu):
                        widget.setvar(widget.cget('textvariable'), str(value))
                    elif isinstance(widget, RuleTableEditor):
                        widget.set_table_data(value)
                
                # Update visualization if running
                if self.running:
                    self._safe_plot_update()
                
                # Update button states
                self._update_editor_buttons()
                
                logger.info("Reset all parameters to defaults")
                
        except Exception as e:
            logger.error(f"Error resetting parameters: {e}")
            messagebox.showerror("Error", f"Failed to reset parameters: {e}")

    # In SimulationGUI._update_editor_buttons:
    def _update_editor_buttons(self):
        try:
            if hasattr(self, '_editor_buttons'):
                if tracker := self._get_change_tracker():
                    undo_state = 'normal' if tracker.undo_stack else 'disabled'
                    redo_state = 'normal' if tracker.redo_stack else 'disabled'
                    save_state = 'normal' if tracker.is_modified() else 'disabled'

                    self._editor_buttons['undo'].configure(state=undo_state, disabledforeground='light grey')
                    self._editor_buttons['redo'].configure(state=redo_state, disabledforeground='light grey')
                    self._editor_buttons['save'].configure(state=save_state, disabledforeground='light grey')
                    self._editor_buttons['reset'].configure(state='normal', disabledforeground='light grey') # Ensure reset is always enabled
        except Exception as e:
            logger.error(f"Error updating button states: {e}")
            
    def _save_rule(self, rule_name: str, rule_data: Dict[str, Any], window: tk.Toplevel, metadata_fields: Dict[str, tk.Widget]):
        """Save rule with current parameters and metadata"""
        try:
            with self._update_lock:
                # Collect metadata
                metadata = {}
                for field_name, widget in metadata_fields.items():
                    if isinstance(widget, tk.Text):
                        value = widget.get("1.0", tk.END).strip()
                    else:
                        if isinstance(widget, tk.Entry):
                            value = widget.get().strip()
                        elif isinstance(widget, tk.Text):
                            value = widget.get("1.0", tk.END).strip()
                        else:
                            raise TypeError(f"Unsupported widget type: {type(widget)}")
                    metadata[field_name] = value
                    
                # Save rule
                try:
                    RuleLibraryManager.get_instance().save_rule(rule_name, rule_data)
                    messagebox.showinfo("Success", f"Rule '{rule_name}' saved successfully")
                    window.destroy()
                except Exception as e:
                    logger.error(f"Error saving rule: {str(e)}")
                    messagebox.showerror("Error", f"Failed to save rule: {str(e)}")

        except Exception as e:
            logger.error(f"Error saving rule: {str(e)}")
            messagebox.showerror("Error", f"Failed to save rule: {str(e)}")
            
    def _delete_rule(self, rule_name: str):
        """Delete a custom rule from the library"""
        try:
            with self._update_lock:
                # Delete rule
                try:
                    RuleLibraryManager.get_instance().delete_rule(rule_name)
                    messagebox.showinfo("Success", f"Rule '{rule_name}' deleted successfully")
                except Exception as e:
                    logger.error(f"Error deleting rule: {str(e)}")
                    messagebox.showerror("Error", f"Failed to delete rule: {str(e)}")
                    return
                
                # Reset simulation to a valid rule
                available_rules = RuleLibraryInfo.get_rule_names()
                if available_rules:
                    new_rule_name = available_rules[0]
                    self._on_rule_instance_change(new_rule_name)
                else:
                    logger.error("No rules available after deletion")
                    messagebox.showerror("Error", "No rules available after deletion")

        except Exception as e:
            logger.error(f"Error deleting rule: {str(e)}")
            messagebox.showerror("Error", f"Failed to delete rule: {str(e)}")
        
    def step_button_clicked(self):
        """Handle step button click"""
        try:
            # Reset all state flags
            self.controller.interrupt_requested = False
            self.running = False
            self.paused = False  # Temporarily unpause to allow step
            
            # Run the step directly
            self.step_simulation()
            
            # Reset state after step
            self.paused = True
            self.running = False
            
        except Exception as e:
            logger.error(f"Error in step button handler: {e}")
            
    def _on_highlight_toggle(self):
        """Handle highlight toggle"""
        logger.debug(f"Highlight toggle: {self.highlight_var.get()}")
        # Force redraw to apply change
        self._safe_plot_update()

    def _on_enable_tiebreakers_change(self):
        """Handle enable tiebreakers checkbox change"""
        enabled = self.enable_tiebreakers_var.get()
        GlobalSettings.ENABLE_TIEBREAKERS = enabled
        logger.info(f"Tiebreakers enabled: {enabled}")
        # Force redraw to apply change
        self._safe_plot_update()

    def _on_tiebreaker_type_change(self, value: str):   
        """Handle tiebreaker type change"""
        try:
            # Update tiebreaker type in rule parameters
            if self.controller.rule.update_parameter('tiebreaker_type', value):
                logger.info(f"Changed tiebreaker type to: {value}")
                # Invalidate metric cache to ensure new parameters are used
                self.controller.rule.invalidate_cache()
                # Force redraw to apply change
                self._safe_plot_update()
            else:
                logger.warning(f"Invalid tiebreaker type: {value}")
                messagebox.showerror("Error", f"Invalid tiebreaker type: {value}")
                # Reset to current type
                self.tiebreaker_type_var.set(self.controller.rule.get_param('tiebreaker_type', 'RANDOM'))
        except Exception as e:
            logger.error(f"Error changing tiebreaker type: {e}")
            messagebox.showerror("Error", f"Failed to change tiebreaker type: {e}")
            # Reset to current type
            self.tiebreaker_type_var.set(self.controller.rule.get_param('tiebreaker_type', 'RANDOM'))
                                                                                    
    def _on_node_density_change(self, value_str: str):
        """Handle node density slider change"""
        try:
            if not self._initialization_complete:
                logger.debug("Skipping density change during initialization") 
                return
                
            logger.debug(f"Node density change requested: {value_str}")
            density = float(value_str)
            if 0.0 <= density <= 1.0:
                GlobalSettings.Simulation.INITIAL_NODE_DENSITY = density
                logger.info(f"Changed initial node density to: {density}")
                
                # Only reset if initialization is complete
                if self._initialization_complete:
                    self.reset_simulation(initial_density=density)
                else:
                    logger.debug("Skipping reset_simulation during initialization")
            else:
                logger.warning("Node density out of range")
                messagebox.showerror("Error", "Node density must be between 0.0 and 1.0")
                if self.node_density_scale is not None:
                    self.node_density_scale.set(GlobalSettings.Simulation.INITIAL_NODE_DENSITY)
        except ValueError as e:
            logger.error(f"Invalid node density value: {value_str}")
            messagebox.showerror("Error", f"Invalid node density value: {value_str}")
                   
    def _on_edge_density_change(self, value_str: str):
        """Handle edge density slider change"""
        try:
            if not self._initialization_complete:
                logger.debug("Skipping edge density change during initialization")
                return
            
            logger.debug(f"Edge density change requested: {value_str}")
            density = float(value_str)
            if 0.0 <= density <= 1.0:
                GlobalSettings.Simulation.INITIAL_EDGE_DENSITY = density
                logger.info(f"Changed initial edge density to: {density}")
                
                # Only reset if the GUI is initialized and not setting defaults
                if hasattr(self, '_is_initialized') and self._is_initialized and not (hasattr(self, '_is_setting_defaults') and self._is_setting_defaults):
                    self.reset_simulation(initial_density=GlobalSettings.Simulation.INITIAL_NODE_DENSITY)
                else:
                    logger.debug("Skipping reset_simulation because GUI is not yet initialized or setting defaults")
            else:
                logger.warning("Edge density out of range")
                messagebox.showerror("Error", "Edge density must be between 0.0 and 1.0")
                if self.edge_density_scale is not None: # ADDED CHECK
                    self.edge_density_scale.set(GlobalSettings.Simulation.INITIAL_EDGE_DENSITY)
        except ValueError as e:
            logger.error(f"Invalid edge density value: {value_str}")
            messagebox.showerror("Error", f"Invalid edge density value: {value_str}")
                        
    def _on_initial_conditions_change(self, value: str):
        """Handle initial conditions selection"""
        logger.info(f"Selected initial conditions: {value}")
        # TODO: Implement loading of initial conditions
        # For now, just reset the simulation
        self.reset_simulation()
    
    def _update_rule_instances(self, rule_type: str):
        """Update the rule instance selector with available rules"""
        # Get available rules for the selected rule type
        available_rules = RuleLibrary.get_rules_in_category(rule_type)
        
        # Update the rule instance selector options
        menu = self.rule_instance_selector["menu"]
        menu.delete(0, "end")
        for rule_name in available_rules:
            menu.add_command(label=rule_name, command=lambda value=rule_name: self._on_rule_instance_change(value))
            
        # Set the selected rule instance to the first available rule
        if available_rules:
            self.rule_instance_var.set(available_rules[0])
        else:
            self.rule_instance_var.set("")
                        
    def _on_rule_type_change(self, rule_type: str):
        """Handle rule type change"""
        try:
            # Get available rules for this type
            available_rules = RuleLibraryInfo.get_rules_in_category(rule_type)
            
            # Update rule instance dropdown
            menu = self.rule_instance_selector["menu"]
            menu.delete(0, "end")
            
            # Add new options
            for rule_name in available_rules:
                menu.add_command(
                    label=rule_name,
                    command=lambda name=rule_name: self._on_rule_instance_change(name)
                )
            
            # Set to first rule in category and trigger update
            if available_rules:
                first_rule = available_rules[0]
                self.rule_instance_var.set(first_rule)
                self._on_rule_instance_change(first_rule)
                
            # Store current rule type
            self.rule_type_var.set(rule_type)
                
            logger.info(f"Changed rule type to: {rule_type}")
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error changing rule type: {e}")
                        
    def _on_rule_instance_change(self, rule_name: str):
        try:
            if rule_name == self.controller.rule_name:
                return

            # Check for changes using safe access
            if tracker := self._get_change_tracker():
                if tracker.is_modified():
                    response = messagebox.askyesnocancel(
                        "Unsaved Changes",
                        "There are unsaved changes. Would you like to save them before changing rules?",
                        icon='warning'
                    )
                
                if response is None:  # Cancel
                    self.rule_instance_var.set(self.controller.rule_name)
                    return
                elif response:  # Yes, save
                    backup_path = self._backup_rules_file()
                    if backup_path:
                        current_rule_data = self._get_current_rule_data()
                        if not self._save_rule_with_confirmation(self.controller.rule_name, current_rule_data):
                            self.rule_instance_var.set(self.controller.rule_name)
                            return
                    else:
                        if not messagebox.askyesno("Warning", 
                            "Failed to create backup. Would you like to continue changing rules anyway?"):
                            self.rule_instance_var.set(self.controller.rule_name)
                            return

            with self._update_lock:
                # Stop any running simulation
                self.running = False
                self.paused = False
                
                logger.info(f"Changing to rule: {rule_name}")
                try:
                    rule_data = RuleLibraryManager.get_rule(rule_name)
                    
                    # Create new rule instance
                    metadata_dict = {k: v for k, v in rule_data.items() if k != 'params'}
                    metadata_dict['neighborhood_type'] = self.controller.neighborhood_type
                    metadata = RuleMetadata(**metadata_dict)
                    new_rule = RuleLibrary.create_rule(rule_name, metadata)
                    
                    # Get parameters and initialize change tracker
                    params = rule_data.get('params', {})
                    new_rule.params = copy.deepcopy(params)
                    if tracker := self._get_change_tracker():
                        tracker.initialize

                    
                    # Update controller
                    self.controller.rule = new_rule
                    self.controller.rule_name = rule_name
                    
                    if not self.controller.grid.set_rule(new_rule):
                        raise ValueError(f"Failed to set rule on grid for {rule_name}")
                    
                    # Update UI
                    self._update_ui_for_new_rule(rule_data)
                    
                    logger.info(f"Changed to rule: {rule_name}")
                    logger.debug(f"Controller rule parameters loaded: {new_rule.params}")
                    
                except Exception as e:
                    logger.error(f"Error setting rule {rule_name}: {e}")
                    messagebox.showerror("Error", f"Failed to set rule: {e}")
                    self.rule_instance_var.set(self.controller.rule_name)
                    return

        except Exception as e:
            logger.error(f"Error changing rule: {str(e)}")
            messagebox.showerror("Error", f"Failed to change rule: {e}")
            self.rule_instance_var.set(self.controller.rule_name)

    def _update_ui_for_new_rule(self, rule_data: Dict[str, Any]):
        """Update UI elements for new rule"""
        try:
            # Update initial conditions
            initial_conditions = rule_data['params'].get('initial_conditions', ["Random"])
            if initial_conditions:
                self.initial_conditions_var.set(initial_conditions[0])
                self._on_initial_conditions_change(self.initial_conditions_var.get())
            
            # Update rule instance selector
            self.rule_instance_var.set(rule_data['name'])
            
            # Update edit button
            self.edit_rule_button.configure(text="Edit Rule")
            
            # Force complete redraw
            self._safe_plot_update()
            
        except Exception as e:
            logger.error(f"Error updating UI for new rule: {e}")
            raise

    def _get_current_rule_data(self) -> Dict[str, Any]:
        """Get current rule data including changes"""
        try:
            rule_data = RuleLibraryManager.get_rule(self.controller.rule_name)
            current_data = {
                'name': self.controller.rule_name,
                'type': self.controller.rule.__class__.__name__,
                'category': rule_data['category'],
                'author': rule_data.get('author', GlobalSettings.Defaults.DEFAULT_AUTHOR),
                'url': rule_data.get('url', GlobalSettings.Defaults.DEFAULT_URL),
                'email': rule_data.get('email', GlobalSettings.Defaults.DEFAULT_EMAIL),
                'date_created': rule_data.get('date_created', datetime.now().strftime("%Y-%m-%d")),
                'date_modified': datetime.now().strftime("%Y-%m-%d"),
                'version': rule_data.get('version', '1.0'),
                'description': rule_data['description'],
                'tags': rule_data.get('tags', []),
                'dimension_compatibility': rule_data.get('dimension_compatibility', []),
                'neighborhood_compatibility': rule_data.get('neighborhood_compatibility', []),
                'parent_rule': rule_data.get('parent_rule', None),
                'position': rule_data.get('position', 1),
                'rating': rule_data.get('rating', None),
                'notes': rule_data.get('notes', None),
                'params': self.controller.rule.params
            }
            return current_data
        except Exception as e:
            logger.error(f"Error getting current rule data: {e}")
            raise
        
    def _check_unsaved_changes(self) -> bool:
        """Check for unsaved changes and prompt user"""
        try:
            if hasattr(self, '_params_modified') and self._params_modified:
                response = messagebox.askyesnocancel(
                    "Unsaved Changes",
                    "There are unsaved changes to the current rule. Would you like to save them?",
                    icon='warning'
                )
                
                if response is None:  # Cancel
                    return False
                elif response:  # Yes, save
                    current_rule_data = {
                        'name': self.controller.rule_name,
                        'type': self.controller.rule.__class__.__name__,
                        'category': RuleLibraryManager.get_rule(self.controller.rule_name)['category'],
                        'author': GlobalSettings.Defaults.DEFAULT_AUTHOR,
                        'url': GlobalSettings.Defaults.DEFAULT_URL,
                        'email': GlobalSettings.Defaults.DEFAULT_EMAIL,
                        'date_created': datetime.now().strftime("%Y-%m-%d"),
                        'date_modified': datetime.now().strftime("%Y-%m-%d"),
                        'version': '1.0',
                        'description': RuleLibraryManager.get_rule(self.controller.rule_name)['description'],
                        'tags': RuleLibraryManager.get_rule(self.controller.rule_name)['tags'],
                        'dimension_compatibility': RuleLibraryManager.get_rule(self.controller.rule_name)['dimension_compatibility'],
                        'neighborhood_compatibility': RuleLibraryManager.get_rule(self.controller.rule_name)['neighborhood_compatibility'],
                        'parent_rule': None,
                        'position': 1,
                        'rating': None,
                        'notes': None,
                        'params': self.controller.rule.params
                    }
                    return self._save_rule_with_confirmation(self.controller.rule_name, current_rule_data)
                    
                # No, don't save
                return True
                
            return True  # No unsaved changes
            
        except Exception as e:
            logger.error(f"Error checking unsaved changes: {e}")
            return False  # On error, prevent action to be safe
            
    def _save_rule_with_confirmation(self, rule_name: str, rule_data: Dict[str, Any], window: Optional[tk.Toplevel] = None) -> bool:
        """Save rule with current parameters and metadata"""
        try:
            # Create backup first
            backup_path = self._backup_rules_file()
            if not backup_path:
                if not messagebox.askyesno("Warning", 
                    "Failed to create backup. Would you like to continue saving anyway?"):
                    return False

            # Get change tracker
            tracker = self._get_change_tracker()

            # Ask user what they want to do
            if window:
                dialog = tk.Toplevel(window)
            else:
                dialog = tk.Toplevel(self.root)
            dialog.title("Save Rule")
            dialog.transient(window if window else self.root)
            dialog.grab_set()

            tk.Label(dialog, text="How would you like to save the rule?").pack(pady=10, padx=10)

            save_choice = tk.StringVar(value="overwrite")
            
            tk.Radiobutton(dialog, text="Overwrite existing rule", 
                        variable=save_choice, value="overwrite").pack(anchor="w", padx=10)
            tk.Radiobutton(dialog, text="Save as new rule", 
                        variable=save_choice, value="new").pack(anchor="w", padx=10)

            new_name_frame = tk.Frame(dialog)
            new_name_frame.pack(fill="x", padx=10, pady=5)
            tk.Label(new_name_frame, text="New rule name:").pack(side="left")
            new_name_entry = tk.Entry(new_name_frame)
            new_name_entry.pack(side="left", fill="x", expand=True, padx=5)
            new_name_entry.insert(0, rule_name + "_copy")
            
            def on_save():
                nonlocal rule_name, rule_data
                if save_choice.get() == "new":
                    new_name = new_name_entry.get().strip()
                    if not new_name:
                        messagebox.showerror("Error", "Please enter a name for the new rule")
                        return
                    rule_name = new_name
                    
                    # Update metadata for new rule
                    rule_data.update({
                        'name': new_name,
                        'date_created': datetime.now().strftime("%Y-%m-%d"),
                        'date_modified': datetime.now().strftime("%Y-%m-%d"),
                        'version': '1.0'
                    })
                else:
                    # Update modification date for existing rule
                    rule_data['date_modified'] = datetime.now().strftime("%Y-%m-%d")

                try:
                    # Update library metadata
                    rule_library_path = RuleLibraryManager.get_instance().library_path #self.library_path #os.path.join(APP_PATHS['config'], 'rules.json') # CHANGED
                    with open(rule_library_path, 'r') as f:
                        current_data = json.load(f)
                    current_data['library_metadata'].update({
                        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                    # Find and update/add rule
                    rule_found = False
                    for i, rule in enumerate(current_data['rules']):
                        if rule['name'] == rule_name:
                            current_data['rules'][i] = rule_data
                            rule_found = True
                            break
                    
                    if not rule_found:
                        current_data['rules'].append(rule_data)

                    # Save updated library
                    rules_path = RuleLibraryManager.get_instance().library_path #self.library_path #os.path.join(APP_PATHS['config'], 'rules.json') # CHANGED
                    with open(rules_path, 'w') as f:
                        json.dump(current_data, f, indent=2)

                    logger.info(f"Rule saved successfully as: {rule_name}")
                    dialog.destroy()
                    if window:
                        window.destroy()
                    messagebox.showinfo("Success", f"Rule saved as: {rule_name}")
                except Exception as e:
                    logger.error(f"Error saving rule: {e}")
                    messagebox.showerror("Error", f"Failed to save rule: {e}")

            def on_cancel():
                dialog.destroy()

            button_frame = tk.Frame(dialog)
            button_frame.pack(fill="x", pady=10)
            tk.Button(button_frame, text="Save", command=on_save).pack(side="left", padx=10)
            tk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="right", padx=10)

            # Center dialog on parent window
            dialog.update_idletasks()
            parent = window if window else self.root
            x = parent.winfo_x() + (parent.winfo_width() - dialog.winfo_width()) // 2
            y = parent.winfo_y() + (parent.winfo_height() - dialog.winfo_height()) // 2
            dialog.geometry(f"+{x}+{y}")

            dialog.wait_window()
            return True

        except Exception as e:
            logger.error(f"Error in save dialog: {e}")
            messagebox.showerror("Error", f"Failed to save rule: {e}")
            return False
             
    def _reset_view(self):
        """Reset view to default position and zoom"""
        try:
            # Reset zoom factor
            self._view_state['zoom_factor'] = 1.0
            
            # Reset view limits
            grid_size = GlobalSettings.Visualization.EDGE_SCALE * max(self.controller.dimensions)
            margin = grid_size * 0.1
            
            self.ax.set_xlim(-margin, grid_size + margin)
            self.ax.set_ylim(-margin, grid_size + margin)
            
            # Reset 3D specific views
            if isinstance(self.ax, Axes3DType): # type: ignore
                self.ax.set_zlim(-margin, grid_size + margin)
                self._current_azim = 0
                self._current_elev = 30
                self.ax.view_init(elev=self._current_elev, azim=self._current_azim)
                self.ax.set_box_aspect([1, 1, 1])
            
            # Reset spacing to default
            GlobalSettings.Visualization.set_node_spacing(0.0)
            if hasattr(self, 'spacing_scale'):
                self.spacing_scale.set(0.0)
            
            # Force complete redraw
            self._safe_plot_update()
            
            logger.debug("View reset to default position")
            
        except Exception as e:
            logger.error(f"Error resetting view: {e}")
    
    def _set_tooltip(self, widget: Union[tk.Entry, ValidatedEntry], message: str):
        """Safely set tooltip on widget"""
        if hasattr(widget, '_tooltip'):
            widget._tooltip.destroy()  # type: ignore
        tooltip = ToolTip(widget, message)
        setattr(widget, '_tooltip', tooltip)

    def _remove_tooltip(self, widget: Union[tk.Entry, ValidatedEntry]):
        """Safely remove tooltip from widget"""
        if hasattr(widget, '_tooltip'):
            widget._tooltip.destroy()  # type: ignore
            delattr(widget, '_tooltip')
                                                                                                             
    def _update_parameter(self, param_name: str, value_str: str, parameter_entries: Mapping[str, Union[tk.Entry, RuleTableEditor]]) -> None:
        """Update parameter with validation and highlight effect"""
        try:
            with self._update_lock:
                # Get current value for type inference
                current_value = self.controller.rule.params.get(param_name)
                
                # Convert string value to correct type
                try:
                    if isinstance(current_value, float):
                        new_value = float(value_str)
                    elif isinstance(current_value, int):
                        new_value = int(float(value_str))
                    elif isinstance(current_value, bool):
                        new_value = value_str.lower() in ('true', '1', 'yes', 'on')
                    else:
                        new_value = value_str
                    
                    # Update parameter with validation
                    if self.controller.rule.update_parameter(param_name, new_value):
                        # Invalidate metric cache to ensure new parameters are used
                        self.controller.rule.invalidate_cache()
                        
                        # Force a synchronization point
                        self.root.update_idletasks()
                        
                        logger.info(f"Updated parameter {param_name} to {new_value}")
                        
                        # Find the entry widget and highlight it
                        entry_widget = parameter_entries.get(param_name)
                        if entry_widget and isinstance(entry_widget, tk.Entry):
                            self._highlight_field(entry_widget)
                            
                        # Update Tiebreaker dropdown if changed
                        if param_name == 'tiebreaker_type':
                            self._update_tiebreaker_dropdown(str(new_value))
                    else:
                        # Revert the entry field to the previous value
                        entry_widget = parameter_entries.get(param_name)
                        if entry_widget and isinstance(entry_widget, tk.Entry):
                            entry_widget.delete(0, tk.END)
                            entry_widget.insert(0, str(current_value))
                            
                except ValueError as e:
                    logger.error(f"Invalid value for parameter {param_name}: {e}")
                    messagebox.showerror("Error", f"Invalid value for parameter {param_name}: {e}")
                    
                    # Revert the entry field to the previous value
                    entry_widget = parameter_entries.get(param_name)
                    if entry_widget and isinstance(entry_widget, tk.Entry):
                        entry_widget.delete(0, tk.END)
                        entry_widget.insert(0, str(current_value))
        except Exception as e:
            logger.error(f"Error updating parameter {param_name}: {e}")
            messagebox.showerror("Error", f"Error updating parameter: {e}")
                                                                                                                     
    def _create_parameter_metadata_text(self, param_name: str, param_info: Any) -> str:
        """Create formatted metadata text for parameter"""
        metadata_parts = []
        
        if isinstance(param_info, dict):
            # Add description
            if 'description' in param_info:
                metadata_parts.append(param_info['description'])
            
            # Add type information
            if 'type' in param_info:
                type_str = f"Type: {param_info['type'].__name__}"
                if 'min' in param_info:
                    type_str += f", Min: {param_info['min']}"
                if 'max' in param_info:
                    type_str += f", Max: {param_info['max']}"
                if 'allowed_values' in param_info:
                    type_str += f"\nAllowed values: {', '.join(map(str, param_info['allowed_values']))}"
                metadata_parts.append(type_str)
                
        return "\n".join(metadata_parts)

    def _create_validated_entry(self, parent: tk.Frame, param_name: str, param_info: Any) -> tk.Entry:
        """Create entry widget with validation"""
        def validate(value):
            try:
                if isinstance(param_info, dict):
                    param_type = param_info.get('type')
                    if param_type == float:
                        val = float(value)
                        min_val = param_info.get('min')
                        max_val = param_info.get('max')
                        if min_val is not None and val < min_val:
                            return False
                        if max_val is not None and val > max_val:
                            return False
                    elif param_type == int:
                        val = int(float(value))
                        min_val = param_info.get('min')
                        max_val = param_info.get('max')
                        if min_val is not None and val < min_val:
                            return False
                        if max_val is not None and val > max_val:
                            return False
                    elif param_type == bool:
                        return value.lower() in ('true', 'false', '1', '0', 'yes', 'no', 'on', 'off', '')
                return True
            except ValueError:
                return False

        vcmd = (parent.register(validate), '%P')
        entry = tk.Entry(parent, validate='key', validatecommand=vcmd)
        
        # Set initial value
        if param_name in self.controller.rule.params:
            entry.insert(0, str(self.controller.rule.params[param_name]))
            
        return entry
        
    def _setup_scrolling(self, canvas: tk.Canvas, frame: tk.Frame):
        """Setup cross-platform scrolling for a canvas/frame pair"""
        def _on_mousewheel(event):
            """Handle mousewheel scrolling for all platforms"""
            try:
                # Check if there's anything to scroll
                if canvas.yview() != (0.0, 1.0):
                    if event.num == 4 or event.num == 5:  # Linux
                        delta = -1 if event.num == 4 else 1
                    else:  # Windows/macOS
                        # Convert delta to consistent units
                        delta = -1 * (event.delta // 120 if abs(event.delta) >= 120 else event.delta)
                    
                    canvas.yview_scroll(delta, "units")
            except Exception as e:
                logger.error(f"Error in mousewheel handler: {e}")

        def _on_frame_configure(event):
            """Update scroll region when frame size changes"""
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        def _on_canvas_configure(event):
            """Update frame width to match canvas"""
            # Update the frame's width to fill the canvas
            canvas_width = event.width
            canvas.itemconfig(canvas.find_withtag("window")[0], width=canvas_width)

        # Bind frame resizing
        frame.bind("<Configure>", _on_frame_configure)
        
        # Bind canvas resizing
        canvas.bind("<Configure>", _on_canvas_configure)

        # Bind mouse wheel for all platforms
        # Windows and macOS
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        frame.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Linux
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)
        frame.bind_all("<Button-4>", _on_mousewheel)
        frame.bind_all("<Button-5>", _on_mousewheel)

        def _on_enter(event):
            """Bind scrolling when mouse enters widget"""
            if platform.system() == "Linux":
                canvas.bind_all("<Button-4>", _on_mousewheel)
                canvas.bind_all("<Button-5>", _on_mousewheel)
            else:
                canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _on_leave(event):
            """Unbind scrolling when mouse leaves widget"""
            if platform.system() == "Linux":
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
            else:
                canvas.unbind_all("<MouseWheel>")

        # Bind enter/leave events
        canvas.bind("<Enter>", _on_enter)
        canvas.bind("<Leave>", _on_leave)
        frame.bind("<Enter>", _on_enter)
        frame.bind("<Leave>", _on_leave)
        
    def _create_parameter_fields(self, parent_frame: tk.Frame, rule_name: str, 
                           parameter_entries: Dict[str, Union[ValidatedEntry, tk.OptionMenu, 'RuleTableEditor']]) -> None:
        """Create parameter fields with enhanced metadata display and value tracking"""
        rule_data = RuleLibraryManager.get_rule(rule_name)
        if rule_data is None:
            logger.error(f"Rule data not found for {rule_name}")
            return

        # Store original parameter values for change detection
        self._original_params = copy.deepcopy(self.controller.rule.params)

        # Get parameter metadata from the rule's PARAMETER_METADATA
        rule_class = globals()[rule_data['type']]
        parameter_metadata = rule_class.PARAMETER_METADATA

        # Get list of parameter names
        param_names = list(parameter_metadata.keys())

        for i, param_name in enumerate(param_names):
            param_info = parameter_metadata[param_name]

            # Create container
            param_container = tk.Frame(parent_frame)
            param_container.pack(fill=tk.X, padx=5, pady=5)

            # Parameter name
            display_name = ' '.join(word.capitalize() for word in param_name.split('_'))
            name_label = tk.Label(param_container, 
                                text=display_name,
                                font=("TkDefaultFont", GlobalSettings.Visualization.RULE_EDITOR_FIELD_FONT_SIZE, "bold"),
                                anchor="w")
            name_label.pack(fill=tk.X)

            # Create appropriate widget based on parameter type
            if param_name == 'initial_conditions':
                # Initial conditions dropdown
                initial_conditions = [
                    "Random",
                    "2D - Circle", "2D - Square", "2D - Grid", "2D - Random Walk",
                    "3D - Sphere", "3D - Cube", "3D - Grid", "3D - Random Walk"
                ]
                var = tk.StringVar(value=str(self.controller.rule.params.get(param_name, "Random")))
                dropdown = tk.OptionMenu(
                    param_container, 
                    var, 
                    *initial_conditions,
                    command=lambda value, name=param_name: self._on_parameter_change(name, value.get(), parameter_entries)
                )
                dropdown.config(width=20)
                dropdown.pack(fill=tk.X, pady=2)
                parameter_entries[param_name] = dropdown

            elif param_name.endswith('_rule_table'):
                # Rule table editor
                table_type = 'state' if param_name == 'state_rule_table' else 'edge'
                
                table_editor = RuleTableEditor(
                    param_container, 
                    self, 
                    param_name,
                    self.controller.rule.params.get(param_name, {}),
                    table_type, 
                    param_info
                )
                table_editor.pack(fill=tk.X, pady=2)
                parameter_entries[param_name] = table_editor

                # Create randomize button for rule table
                randomize_btn = tk.Button(
                    param_container,
                    text=f"Randomize {table_type.title()} Rule Table",
                    command=lambda editor=table_editor: editor._randomize_rule_table()
                )
                randomize_btn.pack(fill=tk.X, pady=(0, 5))

            elif 'allowed_values' in param_info:
                # Dropdown for parameters with preset options
                var = tk.StringVar(value=str(self.controller.rule.params.get(param_name, "")))
                dropdown = tk.OptionMenu(
                    param_container, 
                    var, 
                    *param_info['allowed_values'],
                    command=lambda value, name=param_name: self._on_parameter_change(name, value.get(), parameter_entries)
                )
                dropdown.config(width=20)
                dropdown.pack(fill=tk.X, pady=2)
                parameter_entries[param_name] = dropdown

            elif param_info.get('type') == bool:
                # Dropdown for boolean parameters
                var = tk.StringVar(value=str(self.controller.rule.params.get(param_name, False)))
                dropdown = tk.OptionMenu(
                    param_container, 
                    var, 
                    "True", 
                    "False",
                    command=lambda value, name=param_name: self._on_parameter_change(name, str(value == "True"), parameter_entries)
                )
                dropdown.config(width=10)
                dropdown.pack(fill=tk.X, pady=2)
                parameter_entries[param_name] = dropdown

            else:
                # Regular entry field
                entry = tk.Entry(param_container)
                entry.insert(0, str(self.controller.rule.params.get(param_name, "")))
                entry.pack(fill=tk.X, pady=2)
                parameter_entries[param_name] = cast(ValidatedEntry, entry)

                # Bind validation and update events
                entry.bind('<FocusOut>', lambda e, name=param_name, w=entry: self._handle_entry_validation(e, name, param_info, w))
                entry.bind('<Return>', lambda e, name=param_name, w=entry: self._handle_entry_validation(e, name, param_info, w))
                

            # Parameter description
            if isinstance(param_info, dict) and 'description' in param_info:
                desc_frame = tk.Frame(param_container, bg=parent_frame.cget('bg'))
                desc_frame.pack(fill=tk.X, padx=10, pady=(2,5))

                description_text = param_info['description']
                limits_text = ""
                if 'min' in param_info and 'max' in param_info:
                    limits_text = f" (Min: {param_info['min']}, Max: {param_info['max']})"
                elif 'min' in param_info:
                    limits_text = f" (Min: {param_info['min']})"
                elif 'max' in param_info:
                    limits_text = f" (Max: {param_info['max']})"

                desc_label = tk.Label(
                    desc_frame,
                    text=description_text + limits_text,
                    wraplength=GlobalSettings.Visualization.RULE_EDITOR_PARAM_WIDTH - 60,
                    justify=tk.LEFT,
                    bg=parent_frame.cget('bg'),
                    fg='white',
                    font=("TkDefaultFont", GlobalSettings.Visualization.RULE_EDITOR_FONT_SIZE)
                )
                desc_label.pack(fill=tk.X, padx=5, pady=2)

    def _handle_entry_validation(self, event, param_name: str, param_info: Dict[str, Any], widget: tk.Entry):
        """Handle validation and update on entry events"""
        value = widget.get()
        if self._validate_entry_value(value, param_name, param_info, widget):
            # Valid value - update parameter
            self._on_parameter_change(param_name, value, self._parameter_entries)
        else:
            # Invalid value - restore last valid value
            widget.delete(0, tk.END)
            widget.insert(0, str(self._original_params.get(param_name, "")))

    def _verify_scroll_regions(self):
        """Verify all scroll regions are properly set"""
        try:
            # Get canvas references from actual widgets
            param_canvas = None
            canvas_frame = None
            metadata_frame = None
            
            # Find the canvases in the widget hierarchy
            for child in self.root.winfo_children():
                if isinstance(child, tk.Canvas):
                    if not param_canvas:
                        param_canvas = child
                    elif not metadata_frame:
                        metadata_frame = child
                        
            if param_canvas:
                param_scroll_region = param_canvas.bbox("all")
                logger.debug(f"Parameter scroll region: {param_scroll_region}")
                
            if metadata_frame:
                metadata_scroll_region = metadata_frame.bbox("all")
                logger.debug(f"Metadata scroll region: {metadata_scroll_region}")
                
            # Check rule table scroll regions
            for widget in self._parameter_entries.values():
                if isinstance(widget, RuleTableEditor):
                    table_canvas = widget.canvas
                    table_scroll_region = table_canvas.bbox("all")
                    logger.debug(f"Rule table scroll region: {table_scroll_region}")
                        
            return True
                
        except Exception as e:
            logger.error(f"Error verifying scroll regions: {e}")
            return False
            
    def _on_parameter_change(self, param_name: str, value: str, parameter_entries: Mapping[str, Union[tk.Entry, tk.OptionMenu, RuleTableEditor]]) -> None:
        """Handle parameter value changes with change tracking"""
        try:
            # Convert and validate the value
            new_value = self._convert_parameter_value(param_name, value)
            old_value = self.controller.rule.params.get(param_name)
            
            if new_value != old_value:
                # Use safe change tracker access
                if tracker := self._get_change_tracker():
                    tracker.track_change(param_name, old_value, new_value)
                    
                # Update the parameter
                self.controller.rule.params[param_name] = new_value
                self.controller.rule.invalidate_cache()
                
                # Update UI if needed
                if self.running:
                    self._safe_plot_update()
                    
                # Update button states if editor is open
                if hasattr(self, '_update_editor_buttons'):
                    self._update_editor_buttons()
                    
                # Update widget display if needed
                widget = parameter_entries.get(param_name)
                if isinstance(widget, tk.Entry):
                    widget.delete(0, tk.END)
                    widget.insert(0, str(new_value))
                elif isinstance(widget, tk.OptionMenu):
                    widget.setvar(widget.cget('textvariable'), str(new_value))
                    
                logger.info(f"Updated parameter {param_name} to {new_value}")
                
        except ValueError as e:
            logger.error(f"Invalid value for {param_name}: {e}")
            messagebox.showerror("Error", f"Invalid value for {param_name}: {e}")
            
            # Restore original value in widget
            widget = parameter_entries.get(param_name)
            if isinstance(widget, tk.Entry):
                widget.delete(0, tk.END)
                widget.insert(0, str(old_value))
                                                                            
    def _validate_entry_value(self, value: str, param_name: str, param_info: Dict[str, Any], widget: Optional[Union[tk.Entry, ValidatedEntry]] = None) -> bool:
        """Validate entry value based on parameter info"""
        try:
            is_valid = False
            error_msg = ""
            
            # Get the expected type
            if isinstance(param_info, dict):
                param_type = param_info.get('type')
            else:
                # If param_info is not a dict, use the current value's type
                current_value = self.controller.rule.params.get(param_name)
                param_type = type(current_value)

            # Convert the value
            try:
                if param_type == float:
                    val = float(value)
                    if 'min' in param_info and val < param_info['min']:
                        is_valid = False
                        error_msg = f"Value must be at least {param_info['min']}"
                    elif 'max' in param_info and val > param_info['max']:
                        is_valid = False
                        error_msg = f"Value must be at most {param_info['max']}"
                    else:
                        is_valid = True
                elif param_type == int:
                    val = int(float(value))
                    if 'min' in param_info and val < param_info['min']:
                        is_valid = False
                        error_msg = f"Value must be at least {param_info['min']}"
                    elif 'max' in param_info and val > param_info['max']:
                        is_valid = False
                        error_msg = f"Value must be at most {param_info['max']}"
                    else:
                        is_valid = True
                elif param_type == bool:
                    is_valid = value.lower() in ('true', 'false', '1', '0', 'yes', 'no', 'on', 'off', '')
                    if not is_valid:
                        error_msg = "Must be true/false, yes/no, 1/0, or on/off"
                else:
                    is_valid = True
            except ValueError:
                is_valid = False
                error_msg = f"Invalid value" if param_type is None else f"Invalid {param_type.__name__} value"

            # Apply visual feedback if widget provided
            if widget is not None:
                if is_valid:
                    widget.configure(bg='white')
                    self._remove_tooltip(widget)
                else:
                    widget.configure(bg='#ffebeb')  # Light red
                    self._set_tooltip(widget, error_msg)

            return is_valid

        except Exception as e:
            logger.error(f"Error validating entry value: {e}")
            return False

    def _add_undo_redo_buttons(self, button_frame: tk.Frame):
        """Add undo/redo buttons to button frame"""
        undo_btn = tk.Button(button_frame, text="Undo", 
                            command=self._undo_change)
        undo_btn.pack(side=tk.LEFT, padx=5)
        
        redo_btn = tk.Button(button_frame, text="Redo", 
                            command=self._redo_change)
        redo_btn.pack(side=tk.LEFT, padx=5)

    def _undo_change(self):
        """Handle undo button click"""
        if self.change_tracker is not None:
            change = self.change_tracker.undo()
            if change:
                param_name = change['param']
                value = change['old']
                self.controller.rule.params[param_name] = value
                self.controller.rule.invalidate_cache()
                
                # Update UI
                self._update_parameter_display(param_name, value)
                if self.running:
                    self._safe_plot_update()
        if change:
            param_name = change['param']
            value = change['old']
            self.controller.rule.params[param_name] = value
            self.controller.rule.invalidate_cache()
            
            # Update UI
            self._update_parameter_display(param_name, value)
            if self.running:
                self._safe_plot_update()
                    
    def _verify_change_detection_integration(self) -> bool:
        """Verify all change detection points are properly integrated"""
        try:
            missing_integrations = []
            
            # Check required attributes
            required_attrs = [
                'change_tracker',
                '_parameter_entries',
            ]
            
            for attr in required_attrs:
                if not hasattr(self, attr):
                    missing_integrations.append(f"Missing attribute: {attr}")
            
            if missing_integrations:
                logger.error("Change detection integration incomplete:")
                for missing in missing_integrations:
                    logger.error(f"  {missing}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error verifying change detection integration: {e}")
            return False

    def _check_for_changes(self):
        """Periodically check for unsaved changes"""
        try:
            if tracker := self._get_change_tracker():
                if tracker.is_modified():
                    # Update save button state if editor is open
                    if hasattr(self, '_update_editor_buttons'):
                        self._update_editor_buttons()
                        
            # Schedule next check
            self.root.after(1000, self._check_for_changes)
                
        except Exception as e:
            logger.error(f"Error checking for changes: {e}")

    def _validate_and_update_parameter(self, param_name: str, value: str, widget: Union[tk.Entry, tk.OptionMenu]) -> bool:
        """Validate and update parameter with change tracking"""
        try:
            # Convert and validate value
            new_value = self._convert_parameter_value(param_name, value)
            old_value = self.controller.rule.params.get(param_name)
            
            if new_value != old_value:
                # Track the change
                if tracker := self._get_change_tracker(): tracker.track_change(param_name, old_value, new_value)
                
                # Update parameter
                self.controller.rule.params[param_name] = new_value
                self.controller.rule.invalidate_cache()
                
                # Update UI
                if isinstance(widget, tk.Entry):
                    widget.delete(0, tk.END)
                    widget.insert(0, str(new_value))
                
                if self.running:
                    self._safe_plot_update()
                    
                logger.info(f"Updated parameter {param_name} to {new_value}")
                return True
                
            return True  # No change needed
            
        except ValueError as e:
            logger.error(f"Invalid value for {param_name}: {e}")
            messagebox.showerror("Error", f"Invalid value for {param_name}: {e}")
            
            # Restore original value
            if isinstance(widget, tk.Entry):
                widget.delete(0, tk.END)
                widget.insert(0, str(old_value))
            return False
        
    def _update_parameter_display(self, param_name: str, value: Any):
        """Update parameter display in UI"""
        if param_name in self._parameter_entries:
            widget = self._parameter_entries[param_name]
            if isinstance(widget, tk.Entry):
                widget.delete(0, tk.END)
                widget.insert(0, str(value))
            elif isinstance(widget, tk.OptionMenu):
                widget.setvar(widget.cget('textvariable'), str(value))

    def _redo_change(self):
        """Handle redo button click"""
        if self.change_tracker is not None:
            change = self.change_tracker.redo()
        else:
            change = None
        if change:
            param_name = change['param']
            value = change['new']
            self.controller.rule.params[param_name] = value
            self.controller.rule.invalidate_cache()
            
            # Update UI
            self._update_parameter_display(param_name, value)
            if self.running:
                self._safe_plot_update()
                
    def _update_tiebreaker_dropdown(self, tiebreaker_value: str):
        """Update the Tiebreaker dropdown in the main window"""
        try:
            # Update the Tiebreaker dropdown in the main window
            self.tiebreaker_type_var.set(tiebreaker_value)
            logger.info(f"Updated Tiebreaker dropdown to: {tiebreaker_value}")
        except Exception as e:
            logger.error(f"Error updating Tiebreaker dropdown: {e}")
                
    def _on_rule_editor_close(self, editor_window: tk.Toplevel, rule_name: str):
        try:
            # Check for changes using safe access
            if tracker := self._get_change_tracker():
                if tracker.is_modified():
                    response = messagebox.askyesnocancel(
                        "Unsaved Changes",
                        "There are unsaved changes. Would you like to save them before closing?",
                        icon='warning'
                    )
                
                    if response is None:  # Cancel
                        return
                    elif response:  # Yes, save
                        current_rule_data = {
                            'name': rule_name,
                            'type': self.controller.rule.__class__.__name__,
                            'category': RuleLibraryManager.get_rule(rule_name)['category'],
                            'author': GlobalSettings.Defaults.DEFAULT_AUTHOR,
                            'url': GlobalSettings.Defaults.DEFAULT_URL,
                            'email': GlobalSettings.Defaults.DEFAULT_EMAIL,
                            'date_created': RuleLibraryManager.get_rule(rule_name)['date_created'],
                            'date_modified': datetime.now().strftime("%Y-%m-%d"),
                            'version': RuleLibraryManager.get_rule(rule_name)['version'],
                            'description': RuleLibraryManager.get_rule(rule_name)['description'],
                            'tags': RuleLibraryManager.get_rule(rule_name)['tags'],
                            'dimension_compatibility': RuleLibraryManager.get_rule(rule_name)['dimension_compatibility'],
                            'neighborhood_compatibility': RuleLibraryManager.get_rule(rule_name)['neighborhood_compatibility'],
                            'parent_rule': None,
                            'position': 1,
                            'rating': None,
                            'notes': None,
                            'params': self.change_tracker.current_values if self.change_tracker else self.controller.rule.params
                        }
                        
                        if not self._save_rule_with_confirmation(rule_name, current_rule_data):
                            return  # Don't close if save failed or was cancelled
            else:
                response = True # ADDED - if there is no change tracker, assume it is ok to close

            logger.info(f"Closing Rule Editor for rule: {rule_name}")
            editor_window.destroy()
            
        except Exception as e:
            logger.error(f"Error closing Rule Editor: {e}")
            messagebox.showerror("Error", f"Error closing editor: {e}")
                     
    def _add_reset_button(self, button_frame: tk.Frame):
        """Add reset to defaults button"""
        reset_btn = tk.Button(
            button_frame,
            text="Reset to Defaults",
            command=self._reset_to_defaults
        )
        reset_btn.pack(side=tk.LEFT, padx=5)

    def _reset_to_defaults(self):
        """Reset parameters to default values"""
        if messagebox.askyesno(
            "Confirm Reset",
            "Are you sure you want to reset all parameters to their default values?"
        ):
            try:
                # Get default values from rule data
                rule_data = RuleLibraryManager.get_rule(self.controller.rule_name)
                default_params = copy.deepcopy(rule_data['params'])
                
                # Track the change
                if tracker := self._get_change_tracker():
                    tracker.track_change(
                    'all_params',
                    self.controller.rule.params,
                    default_params
                )
                
                # Apply defaults
                self.controller.rule.params = default_params
                self.controller.rule.invalidate_cache()
                
                # Update all parameter displays
                for param_name, widget in self._parameter_entries.items():
                    self._update_parameter_display(param_name, default_params.get(param_name))
                
                if self.running:
                    self._safe_plot_update()
                    
                logger.info("Reset all parameters to defaults")
                
            except Exception as e:
                logger.error(f"Error resetting to defaults: {e}")
                messagebox.showerror("Error", f"Failed to reset parameters: {e}")
                                                                                                                                              
    def _on_spacing_change(self, value: str):
        """Handle spacing slider change"""
        try:
            with self._update_lock:
                try:
                    spacing = float(value)
                except ValueError as e:
                    logger.error(f"Invalid spacing value: {value}")
                    messagebox.showerror("Error", f"Invalid spacing value: {value}")
                    return
                
                current_spacing = GlobalSettings.Visualization.NODE_SPACING
                
                # Only update if actually changed
                if spacing != current_spacing:
                    # Validate spacing value
                    if not (0.0 <= spacing <= GlobalSettings.Visualization.MAX_NODE_SPACING):
                        logger.error(f"Spacing value {spacing} out of range")
                        messagebox.showerror("Error", f"Spacing value must be between 0.0 and {GlobalSettings.Visualization.MAX_NODE_SPACING}")
                        if self.spacing_scale is not None: # ADDED CHECK
                            self.spacing_scale.set(current_spacing)  # Reset to valid value
                        return
                    
                    # Update the spacing
                    GlobalSettings.Visualization.set_node_spacing(spacing)
                    
                    # Store current view state
                    current_xlim = self.ax.get_xlim()
                    current_ylim = self.ax.get_ylim()
                    current_zlim = self.ax.get_zlim() if isinstance(self.ax, Axes3DType) else None  # type: ignore
                    
                    # Clear everything
                    self.ax.clear()
                    self.fig.clear()
                    
                    # Recreate the axes
                    if self.controller.dimension_type == Dimension.THREE_D:
                        self.ax = self.fig.add_subplot(111, projection='3d')
                        if hasattr(self, '_current_elev') and hasattr(self, '_current_azim'):
                            self.ax.view_init(elev=self._current_elev, azim=self._current_azim)
                    else:
                        self.ax = self.fig.add_subplot(111)
                        self.ax.set_aspect('equal')
                    
                    # Force complete redraw
                    self._safe_plot_update()
                    
                    # Restore view state
                    if isinstance(self.ax, Axes3DType):  # type: ignore
                        self.ax.set_xlim(current_xlim[0], current_xlim[1])
                        self.ax.set_ylim(current_ylim[0], current_ylim[1])
                        if current_zlim:
                            self.ax.set_zlim(current_zlim[0], current_zlim[1])  # type: ignore
                        self.ax.set_box_aspect([1, 1, 1])
                    else:
                        self.ax.set_xlim(current_xlim[0], current_xlim[1])
                        self.ax.set_ylim(current_ylim[0], current_ylim[1])
                    
                    # Force immediate draw
                    self.canvas.draw()
                    self.canvas.flush_events()
                    
                    logger.info(f"Changed spacing from {current_spacing:.1f} to {spacing:.1f}")
                    
        except ValueError as e:
            logger.error(f"Invalid spacing value: {value}")
            messagebox.showerror("Error", f"Invalid spacing value: {value}")
                                                    
    def reset_coordinates(self):
        """Force recalculation of node coordinates with current settings"""
        try:
            # Reset view state to force full redraw
            self._view_state = {
                'xlim': None,
                'ylim': None,
                'zlim': None,
                'elev': 30 if self.controller.dimension_type == Dimension.THREE_D else None,
                'azim': 45 if self.controller.dimension_type == Dimension.THREE_D else None,
                'zoom_factor': 1.0
            }
            
            # Recalculate plot limits with new spacing
            self._update_plot_limits()
            
            # Force controller to update node positions
            if hasattr(self.controller, 'grid'):
                # Get current state
                current_state = self.controller.get_state()
                
                # Reset grid with current state
                self.controller.grid = Grid(
                    self.controller.dimensions,
                    self.controller.neighborhood_type,
                    self.controller.dimension_type,
                    rule=self.controller.rule,  # Pass the rule instance
                    unique_id=self.controller._unique_id # Access unique_id from controller
                )
                
                # Restore state
                self.controller.set_state(current_state)
                
            logger.debug("Node coordinates reset with new spacing")
            
        except Exception as e:
            logger.error(f"Error resetting coordinates: {str(e)}")
            raise
            
    def _get_valid_neighborhoods(self) -> List[str]:
        """Get list of valid neighborhood types for current dimension"""
        if self.controller.dimension_type == Dimension.TWO_D:
            return ["VON_NEUMANN", "MOORE", "HEX"]
        else:  # THREE_D
            # Note: We still show HEX as an option, it will be auto-converted to HEX_PRISM
            return ["VON_NEUMANN", "MOORE", "HEX", "HEX_PRISM"]
                                        
    def _on_dimension_change(self, dimension_str: str):
        """Handle dimension type change"""
        try:
            with self._update_lock:
                # Get the value from the settings
                dimension_type = Dimension[dimension_str]
                
                # Set global settings
                GlobalSettings.Simulation.DIMENSION_TYPE = dimension_type
                
                # Reset spacing to default and update slider
                GlobalSettings.Visualization.NODE_SPACING = 0.0
                if self.spacing_scale is not None:
                    self.spacing_scale.set(0.0)
                
                # Create new controller with new dimension type
                try:
                    self.controller = SimulationController(
                        rule_name=self.controller.rule_name,
                        dimension_type=dimension_type,
                        initialize_state=True
                    )
                except Exception as e:
                    logger.error(f"Error creating new controller: {e}")
                    messagebox.showerror("Error", f"Failed to create new controller: {e}")
                    return
                    
                # Clear and recreate the figure and axes
                self.fig.clear()
                if dimension_type == Dimension.THREE_D:
                    self.ax = self.fig.add_subplot(111, projection='3d')
                    if hasattr(self, '_current_elev') and hasattr(self, '_current_azim'):
                        self._current_azim = 45  # Set initial azimuth for depth view
                        self._current_elev = 30   # Set initial elevation
                        self.ax.view_init(elev=self._current_elev, azim=self._current_azim)
                else:
                    self.ax = self.fig.add_subplot(111)
                    self.ax.set_aspect('equal')
                
                # Reset view state
                self._view_state = {
                    'xlim': None,
                    'ylim': None,
                    'zlim': None,
                    'elev': 20 if dimension_type == Dimension.THREE_D else None,
                    'azim': 45 if dimension_type == Dimension.THREE_D else None,
                    'zoom_factor': 1.0
                }
                
                # Update grid size slider to show current dimension's size
                current_size = GlobalSettings.Simulation.get_current_grid_size()
                if self.grid_size_scale is not None:
                    self.grid_size_scale.set(current_size)
                if self.grid_size_label is not None:
                    self.grid_size_label.config(text=f"Size: {current_size}")
                
                # Reset simulation with new dimension type
                self.reset_simulation()
                
                # Reinitialize change tracker with current parameters
                if hasattr(self, 'controller') and self.controller.rule:
                    if self.change_tracker is not None:
                        self.change_tracker.initialize(self.controller.rule.params)
                
                # Force plot update
                self._safe_plot_update()
                
                logger.info(f"Changed dimension type to: {dimension_type.name}")
                
        except (ValueError, TypeError) as e:
            logger.error(f"Error changing dimension type: {e}")
            messagebox.showerror("Error", f"Failed to change dimension type: {e}")
                                                                    
    def _on_grid_size_change(self, new_size_str: str):
        """Handle grid size change"""
        try:
            # Prevent recursive calls
            if hasattr(self, '_is_resizing') and self._is_resizing:
                logger.warning("Recursive call to _on_grid_size_change, skipping")
                return
            
            # Prevent initialization during setup
            if not self._is_initialized:
                logger.debug("Skipping _on_grid_size_change during initialization")
                return
            
            self._is_resizing = True  # Set flag
            
            try:
                with self._update_lock:
                    try:
                        new_size = int(float(new_size_str))
                        logger.debug(f"New grid size: {new_size}")
                    except ValueError as e:
                        logger.error(f"Invalid grid size value: {new_size_str}")
                        messagebox.showerror("Error", f"Invalid grid size value: {new_size_str}")
                        return
                    
                    # Update global settings with appropriate dimension type
                    GlobalSettings.Simulation.set_grid_size(new_size, self.controller.dimension_type)
                    
                    # Pause simulation if running
                    was_running = self.running
                    self.running = False
                    
                    try:
                        # Store old grid state
                        logger.debug("Storing old grid state")
                        old_grid_state = self.controller.grid.grid_array.copy()
                        
                        # Clean up old grid
                        self.controller.grid.cleanup()
                        
                        # Update controller dimensions
                        logger.debug("Updating controller dimensions")
                        self.controller.dimensions = GlobalSettings.Simulation.get_grid_dimensions()
                        
                        # Create new grid with new dimensions
                        logger.debug("Creating new grid with new dimensions")
                        new_grid = Grid(
                            self.controller.dimensions,
                            self.controller.neighborhood_type,
                            self.controller.dimension_type,
                            rule=self.controller.rule,  # Pass the rule instance
                            unique_id=self.controller._unique_id
                        )
                        
                        self.controller.grid = new_grid
                        
                        # Restore old grid state
                        self._restore_grid_state(old_grid_state)
                        
                        # Update label
                        current_size = GlobalSettings.Simulation.get_current_grid_size()
                        if self.grid_size_label is not None:
                            self.grid_size_label.config(text=f"Size: {current_size}")
                        
                        # Restore the view state
                        if isinstance(self.ax, Axes3DType):  # type: ignore
                            self.ax.view_init(elev=self._current_elev, azim=self._current_azim)  # type: ignore
                        
                        # Save existing limits prior to update
                        if hasattr(self, '_view_state') and isinstance(self.ax, Axes): # type: ignore
                            self._view_state['xlim'] = self.ax.get_xlim()
                            self._view_state['ylim'] = self.ax.get_ylim()
                            if hasattr(self.ax, 'get_zlim'): # type: ignore
                                self._view_state['zlim'] = self.ax.get_zlim() # type: ignore
                        
                        # Force plot update
                        logger.debug("Forcing plot update")
                        self._safe_plot_update()
                        
                        # Restore simulation state
                        self.running = was_running
                        
                        logger.info(f"Changed {self.controller.dimension_type.name} grid size to: {new_size}")
                        
                    except Exception as e:
                        logger.error(f"Error changing grid size: {e}")
                        self.running = False
                        messagebox.showerror("Error", f"Failed to change grid size: {e}")
                        raise
                        
            except ValueError as e:
                logger.error(f"Invalid grid size value: {e}")
                messagebox.showerror("Error", f"Invalid grid size value: {e}")
        finally:
            if hasattr(self, '_is_resizing'):
                self._is_resizing = False  # Clear flag
                 
    def _restore_grid_state(self, old_grid_state):
        """Restores as much of the old grid state as possible to the new grid"""
        try:
            # Get new dimensions
            new_dimensions = self.controller.dimensions
            new_grid_array = self.controller.grid.grid_array
            
            # If we are in 3D
            if self.controller.dimension_type == Dimension.THREE_D:
                # Get old dimensions
                old_z, old_y, old_x = old_grid_state.shape
                
                # Determine minimum and maximum overlap dimensions
                min_z = min(old_z, new_dimensions[0])
                min_y = min(old_y, new_dimensions[1])
                min_x = min(old_x, new_dimensions[2])
                
                # Copy the overlapping parts
                for z in range(min_z):
                    for y in range(min_y):
                        for x in range(min_x):
                            new_grid_array[z, y, x] = old_grid_state[z, y, x]
                            
            else:
                old_y, old_x = old_grid_state.shape
                # Determine minimum and maximum overlap dimensions
                min_y = min(old_y, new_dimensions[0])
                min_x = min(old_x, new_dimensions[1])
                
                for y in range(min_y):
                    for x in range(min_x):
                        new_grid_array[y, x] = old_grid_state[y, x]
                        
            # Update the grid with the new state
            self.controller.grid.grid_array = new_grid_array
            
            # Initialize the edges with the new grid state
            self.controller.grid.initialize_edges()
                        
            
            logger.info("Grid state restored")
        except Exception as e:
            logger.error(f"Error restoring grid state: {e}")
                   
    def _on_neighborhood_change(self, neighborhood_str: str):
        """Handle neighborhood type change"""
        try:
            with self._update_lock:
                # Convert string to enum, handling HEX -> HEX_PRISM conversion for 3D
                if (neighborhood_str == "HEX" and 
                    self.controller.dimension_type == Dimension.THREE_D):
                    new_type = NeighborhoodType.HEX_PRISM
                    # Update the display in the dropdown to show HEX_PRISM
                    self.neighborhood_var.set("HEX_PRISM")
                    logger.info("Automatically converted HEX to HEX_PRISM for 3D")
                else:
                    try:
                        new_type = self.neighborhood_types[neighborhood_str]
                    except KeyError as e:
                        logger.error(f"Invalid neighborhood type: {neighborhood_str}")
                        messagebox.showerror("Error", f"Invalid neighborhood type: {neighborhood_str}")
                        return

                # Validate compatibility
                if ((new_type == NeighborhoodType.HEX_PRISM and 
                    self.controller.dimension_type == Dimension.TWO_D) or
                    (new_type == NeighborhoodType.HEX and 
                    self.controller.dimension_type == Dimension.THREE_D)):
                    tk.messagebox.showerror( # type: ignore
                        "Invalid Selection",
                        "This neighborhood type is not compatible with the current dimension type."
                    )
                    # Reset to current type
                    self.neighborhood_var.set(self.controller.neighborhood_type.name)
                    return
                    
                # Update controller with new neighborhood type
                try:
                    self.controller.set_neighborhood_type(new_type)
                except Exception as e:
                    logger.error(f"Error setting neighborhood type: {e}")
                    messagebox.showerror("Error", f"Failed to set neighborhood type: {e}")
                    return
                
                # Reset view state
                self._view_state = {
                    'xlim': None,
                    'ylim': None,
                    'zlim': None,
                    'elev': 30,
                    'azim': 0,
                    'zoom_factor': 1.0
                }
                
                # Reset simulation
                self.reset_simulation()
                
                logger.info(f"Changed neighborhood type to: {new_type.name}")
                
        except (ValueError, TypeError) as e:
            logger.error(f"Error changing neighborhood type: {e}")
            messagebox.showerror("Error", f"Failed to change neighborhood type: {e}")
            # Reset to current type
            self.neighborhood_var.set(self.controller.neighborhood_type.name)
                                 
    def step_simulation(self):
        """Perform one step of simulation using the controller"""
        try:
            with self._update_lock:
                # Ensure the grid is using the currently selected rule
                current_rule_name = self.rule_instance_var.get()
                if current_rule_name != self.controller.rule_name:
                    self.controller.set_rule(current_rule_name) # Update the rule in the controller

                # No need to set rule_name since Grid stores the Rule instance
                
                # Reset all state flags
                self.controller.interrupt_requested = False
                self.running = False
                self.paused = False  # Temporarily unpause to allow step
                
                # Run the step directly
                self.step_simulation_logic()

                # Reset state after step
                self.paused = True
                self.running = False
                
        except Exception as e:
            logger.error(f"Error in step button handler: {e}")

    def step_simulation_logic(self):
        """Encapsulated step simulation logic to be called from step and run methods"""
        try:
            start_time = time.time()

            # Check for interruption before starting step
            if self.controller.interrupt_requested or self._stop_requested:
                self.running = False
                self.paused = True
                self.start_button.config(text="Start", state=tk.NORMAL)  # Re-enable the button
                logger.info("Step simulation interrupted before execution")
                return

            # Add detailed logging for state before update
            logger.debug(f"Pre-step state: Active nodes = {np.sum(self.controller.grid.grid_array > 0)}")
            logger.debug(f"Pre-step edges: {np.sum(self.controller.grid.neighborhood_data.edge_matrix)}")

            # Use controller to perform simulation step
            try:
                logger.info("==================================================================================================== ") 
                logger.info(f"=================================== Performing simulation step {self.step_count} =================================== ") 
                logger.info("==================================================================================================== ") 
                self.controller.step() # Call the controller's step method
            except Exception as step_error:
                logger.error(f"Error during controller step: {step_error}\nTraceback:\n{traceback.format_exc()}")
                raise

            # Log post-step state
            logger.debug(f"Post-step state: Active nodes = {np.sum(self.controller.grid.grid_array > 0)}")
            logger.debug(f"Post-step edges: {np.sum(self.controller.grid.neighborhood_data.edge_matrix)}")

            # Check if step was interrupted
            if self.controller.interrupt_requested or self._stop_requested:
                self.running = False
                self.paused = True
                self.start_button.config(text="Start", state=tk.NORMAL)  # Re-enable the button
                logger.info("Step simulation interrupted during execution")
                return
            
            # Update step counter and labels
            self.step_count = self.controller.generation
            if self.step_label is not None:
                self.step_label.config(text=f"Step: {self.step_count}")

            # Calculate frame time
            current_time = time.time()
            frame_time = current_time - self.last_frame_time
            
            # Update step delay
            try:
                new_delay = GlobalSettings.Simulation.adjust_step_delay(frame_time)
                self.step_delay = new_delay
            except Exception as delay_error:
                logger.error(f"Error adjusting step delay: {delay_error}")
                self.step_delay = GlobalSettings.Simulation.STEP_DELAY

            # Update performance metrics
            step_time = time.time() - start_time
            self.performance_stats['step_times'].append(step_time)
            self.performance_stats['frame_times'].append(frame_time)
            
            if len(self.performance_stats['step_times']) > 100 and self.perf_label is not None:
                times = self.performance_stats['step_times'] [-100:]
                avg_time = float(np.mean(times))
                self.perf_label.config(text=f"Avg Step: {avg_time*1000:.1f}ms")

            # Check if simulation should stop
            if not self.run_continuously.get() and self.step_count >= int(self.num_steps_var.get()):
                self.running = False
                self.paused = True
                self.start_button.config(text="Start", state=tk.NORMAL)  # Re-enable the button
                logger.info("Simulation completed")

            # Update timing
            self.last_frame_time = current_time

            # Update plot with error trapping
            try:
                self._safe_plot_update() # MOVE THIS LINE
            except Exception as plot_error:
                logger.error(f"Error updating plot: {plot_error}\nTraceback:\n{traceback.format_exc()}")
                raise

        except Exception as e:
            logger.error(f"Error in step simulation: {e}\nTraceback:\n{traceback.format_exc()}")
            self.running = False
            self.paused = True
            self.start_button.config(text="Start", state=tk.NORMAL)  # Re-enable the button
            raise
                                            
    def handle_step_button(self):
        """Handle step button press with interruption support"""
        try:
            # Set step button state
            self.step_button_pressed = True
            
            # If currently running a step, interrupt it
            if self.running:
                self.controller.interrupt_requested = False
                # Small delay to ensure current step completes
                time.sleep(0.1)
            
            # Perform one step
            self.step_simulation()
            
            # Force plot update
            if self.controller.dimension_type == Dimension.THREE_D:
                self.update_plot_3d(self.step_count)
            else:
                self.update_plot_2d(self.step_count)
            self.canvas.draw()
            
            # Reset step button state
            self.step_button_pressed = False
            
            # Keep simulation paused after step
            self.running = False
            self.paused = True
            
        except Exception as e:
            logger.error(f"Error in step button handler: {str(e)}")
            self.step_button_pressed = False  # Make sure to reset on error
            raise
        
    def _on_speed_change(self, value: str):
        """Handle speed slider change"""
        try:
            speed = float(value)
            # Convert speed (0-1000) to delay (100-10ms)
            self.step_delay = int(100 - (speed * 0.09))  # Gives range 100ms to 10ms
            self.step_delay = int(100 - (speed * 0.09))  # Gives range 100ms to 10ms
        except ValueError:
            logger.error("Invalid speed value")
            
    def setup_animation(self):
        """Initialize the animation"""
        logger.debug("Starting animation setup")
        
        try:            
            # Update the canvas
            logger.debug("Updating canvas")
            self.canvas.draw()
            
            logger.debug("Animation setup completed")
        except Exception as e:
            logger.error(f"Error in animation setup: {str(e)}")
            raise
                                            
    def toggle_simulation(self):
        """Start/Stop simulation with thread safety"""
        try:
            logger.info("Entering toggle_simulation")

            with self._update_lock:
                if self.running:
                    logger.info("================= Entering toggle_simulation - Stopping simulation ================= ")  # ADDED
                    # Request interruption
                    self._stop_requested = True
                    self.start_button.config(state=tk.DISABLED)  # Disable the button
                    self.start_button.config(text="Stopping...")
                    logger.info("================= Stop requested, waiting for current step to complete =================")
                    
                    # Log the time since the last start
                    if self._last_start_time is not None:
                        time_since_start = time.time() - self._last_start_time
                        logger.info(f"Time since last start: {time_since_start:.3f} seconds")
                    
                else:
                    logger.info("================= Entering toggle_simulation - Starting simulation ================= ")  # ADDED

                    # Check if we are in the process of shutting down
                    if self._is_shutting_down:
                        logger.warning("Attempting to start simulation while shutting down, aborting")
                        return

                    # Reset flags before starting
                    self.controller.interrupt_requested = False
                    self._stop_requested = False
                    self._stop_event.clear()
                    self.running = True
                    self.paused = False
                    self.start_button.config(state=tk.DISABLED)  # Disable the button
                    self.start_button.config(text="Starting...")
                    
                    # Store the start time
                    self._last_start_time = time.time()

                    # Log the actual rule that will be running
                    logger.info(f"Starting simulation with rule: {self.controller.rule_name}")
                    logger.info(f"Rule instance variable shows: {self.rule_instance_var.get()}")
                    logger.info(f"Controller rule parameters: {self.controller.rule.params}")

                    # Force initial render
                    self._safe_plot_update()
                    self.canvas.draw()

                    # Setup shared memory
                    self.controller.grid.setup_shared_memory()

                    logger.info("About to start simulation loop")
                    # Start simulation loop directly
                    self._run_simulation_step()

        except Exception as e:
            logger.error(f"Error in toggle_simulation: {e}")
            self.running = False
            self.start_button.config(text="Start", state=tk.NORMAL)  # Re-enable the button
            self.controller.interrupt_requested = False
        finally:
            # Set button text based on simulation state
            if self.running:
                self.start_button.config(text="Stop")
            else:
                self.start_button.config(text="Start")
            self.start_button.config(state=tk.NORMAL)  # Re-enable the button
                                                                                      
    def _run_simulation_step(self):
        """Run a single simulation step and schedule the next one"""
        try:
            if not self.running or self.paused:
                logger.info("Simulation paused or stopped")
                return

            logger.info("Starting simulation step...")

            # Ensure the grid is using the currently selected rule
            current_rule_name = self.rule_instance_var.get()
            if current_rule_name != self.controller.rule_name:
                self.controller.set_rule(current_rule_name) # Update the rule in the controller

            # No need to set rule_name since Grid stores the Rule instance
            
            # Schedule next step if still running
            if self.running and not self.paused and not self.controller.interrupt_requested:
                try:
                    # Perform step logic using the encapsulated method
                    self.step_simulation_logic()

                    # Update plot with error trapping
                    try:
                        self._safe_plot_update()
                    except Exception as plot_error:
                        logger.error(f"Error updating plot: {plot_error}\nTraceback:\n{traceback.format_exc()}")
                        raise

                    logger.debug("Scheduling next step")
                    self.root.after(int(self.step_delay), self._run_simulation_step)
                    logger.info("Next update scheduled")
                except Exception as e:
                    logger.error(f"Error in simulation step: {e}")
                    self.running = False
                    self.start_button.config(text="Start")

        except Exception as e:
            logger.error(f"Error in simulation step: {e}")
            self.running = False
            self.start_button.config(text="Start")
                                                                         
    def toggle_pause(self):
        """Pause/Resume simulation with thread safety"""
        try:
            with self._update_lock:
                if not self.paused:
                    if self.running:
                        self.controller.request_interrupt()
                    self.paused = True
                    self.pause_button.config(text="Resume")
                    logger.info("Simulation paused")
                else:
                    self.controller.interrupt_requested = False
                    self._stop_requested = False # ADD THIS LINE
                    self.paused = False
                    self.pause_button.config(text="Pause")
                    if self.running:
                        self.root.after(int(self.step_delay), self._run_simulation_step)
                        logger.info("Simulation resumed")
                        
        except Exception as e:
            logger.error(f"Error in toggle_pause: {e}")
            self.paused = True
            self.pause_button.config(text="Resume")
            self.controller.interrupt_requested = False
            
    @log_errors
    def change_rule(self, rule_name: str, new_params: Optional[Dict[str, Any]] = None):
        """Change the current rule"""
        try:
            # Change rule in controller
            self.controller.set_rule(rule_name)
            
            # Reset simulation with new rule
            self.reset_simulation()
            
            #Update plot limits based on magnification
            self._update_plot_limits()
            
            logger.info(f"Changed to rule: {rule_name}")
        except Exception as e:
            logger.error(f"Error changing rule: {str(e)}")

    @log_errors
    def _update_simulation(self):
        """Internal method to update simulation state"""
        try:
            while self.running and not self.paused:
                # Check if interruption was requested
                if self.controller.interrupt_requested:
                    self.running = False
                    self.start_button.config(text="Start")
                    logger.info("Simulation loop interrupted")
                    return

                if self.step_count < GlobalSettings.Simulation.NUM_STEPS:
                    self.step_simulation()
                    # Schedule next update with delay
                    self.root.after(int(self.step_delay), self._update_simulation)
                else:
                    self.running = False
                    self.start_button.config(text="Start")
                    logger.debug("Simulation run completed")
                    break

        except Exception as e:
            logger.error(f"Error in simulation update loop: {e}")
            self.running = False
            self.start_button.config(text="Start")
            raise

    def _cleanup_event_loop(self):
        """Clean up event loop properly"""
        try:
            if hasattr(self, 'event_loop') and self.event_loop is not None:
                # First check if we're in the event loop thread
                current_thread = threading.current_thread()
                is_event_loop_thread = (
                    hasattr(self, 'loop_thread') and 
                    current_thread == self.loop_thread
                )
                
                if self.event_loop.is_running():
                    if is_event_loop_thread:
                        # If we're in the event loop thread, just stop it
                        self.event_loop.stop()
                    else:
                        # If we're in a different thread, use call_soon_threadsafe
                        self.event_loop.call_soon_threadsafe(self.event_loop.stop)
                        
                        # Wait for the event loop to stop
                        if hasattr(self, 'loop_thread') and self.loop_thread.is_alive():
                            self.loop_thread.join(timeout=1.0)
                            logger.info("Event loop thread joined")
                
                # Don't close the event loop - just let it be garbage collected
                # This avoids the "Cannot close a running event loop" error
                self.event_loop = None
                
                logger.info("Event loop cleanup completed")
                
        except Exception as e:
            logger.error(f"Error cleaning up event loop: {e}")

    def _setup_event_loop(self):
        """Setup event loop with proper logging configuration, ensuring single loop"""
        try:
            if not hasattr(self, 'event_loop') or self.event_loop is None:
                # Create event loop in the main thread only if it doesn't exist
                self.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.event_loop)
                
                # Start the event loop in a separate thread
                def run_event_loop():
                    # Use existing logger
                    global logger
                    logger.info("Starting event loop thread")
                    asyncio.set_event_loop(self.event_loop)
                    try:
                        if self.event_loop is not None:
                            self.event_loop.run_forever()
                    except Exception as e:
                        logger.error(f"Error in event loop thread: {e}")
                    finally:
                        logger.info("Event loop thread finished")
                        
                self.loop_thread = threading.Thread(target=run_event_loop, daemon=True)
                self.loop_thread.start()
                logger.info("Event loop thread started")
                
            else:
                logger.info("Event loop already running, reusing existing loop")
                
        except Exception as e:
            logger.error(f"Error setting up event loop: {e}")
            raise

    def _validate_num_steps(self, value):
        """Validate the number of steps entry"""
        if value == "":
            return True  # Allow empty string
        try:
            num = int(value)
            if num > 0:
                return True
            else:
                return False
        except ValueError:
            return False

    def _on_run_continuously_change(self):
        """Handle run continuously checkbox change"""
        self._update_num_steps_entry_state()

    def _update_num_steps_entry_state(self):
        """Update the state of the number of steps entry based on the run continuously checkbox"""
        if self.run_continuously.get():
            self.num_steps_entry.config(state=tk.DISABLED)
        else:
            self.num_steps_entry.config(state=tk.NORMAL)
                        
    def run(self):
            """Start the GUI and main simulation loop with interruption support"""
            global logger
            logger.info("Starting GUI main loop")

            try:
                # Log the actual running rule at startup
                logger.info(f"Starting application with rule: {self.controller.rule_name}")
                logger.info(f"Rule instance variable shows: {self.rule_instance_var.get()}")
                logger.info(f"Controller rule parameters: {self.controller.rule.params}")

                # Setup event loop
                self._setup_event_loop()

                # Make window visible
                self.root.deiconify()
                
                # Bind window closing event
                self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

                # Start tkinter mainloop
                self.root.mainloop()

            except (ValueError, TypeError) as e:
                logger.error(f"Error in GUI main loop: {e}")
            finally:
                if not self._tk_destroyed:
                    try:
                        # Clean up if not already done
                        self.cleanup()
                    except Exception as e:
                        logger.error(f"Error during GUI cleanup: {e}")
                
                logger.info("GUI main loop finished")
                                                                    
    def reset_simulation(self, initial_density: Optional[float] = None):
        """Reset the simulation to initial state"""
        try:
            if not self._initialization_complete:
                logger.debug("Skipping reset during initialization")
                return

            logger.debug("Entering reset_simulation")

            # Store current spacing before reset
            current_spacing = GlobalSettings.Visualization.NODE_SPACING

            # Stop any running simulation
            self.running = False
            self.paused = False

            # Store the rule and parameters before re-initialization
            current_rule_name = self.controller.rule_name
            current_params = self.controller.rule.params.copy()
            logger.debug(f"Stored current rule parameters: {current_params}")

            # Reinitialize shared memory and process pool
            try:
                # Clean up process pool
                if hasattr(self, 'process_pool') and self.process_pool is not None:
                    try:
                        self.process_pool.shutdown(wait=True, cancel_futures=True)
                        logger.info("Process pool shut down successfully")
                    except Exception as e:
                        logger.error(f"Error shutting down process pool: {e}")
                    finally:
                        self.process_pool = None  # Ensure it's set to None

                # Reinitialize the grid
                # DO NOT DELETE THIS COMMENT: It is CRUCIAL to clear the existing grid data BEFORE re-initializing the Grid object.
                # DO NOT DELETE THIS COMMENT: Failing to do so will result in the accumulation of nodes and edges with each reset.
                # DO NOT DELETE THIS COMMENT: This ensures that the new random state is initialized on a clean slate.
                if hasattr(self, 'controller') and hasattr(self.controller, 'grid'):
                    self.controller.grid.grid_array.fill(-1.0)
                    self.controller.grid.neighborhood_data.edge_matrix[:] = False  # Reset edge matrix

                    # Clean up the grid
                    self.controller.grid.cleanup()

                if GlobalSettings.Simulation.NUM_PROCESSES > 1:
                    self.process_pool = ProcessPoolExecutor(
                        max_workers=GlobalSettings.Simulation.NUM_PROCESSES,
                        initializer=Grid._process_initializer
                    )
                    logger.info(f"Recreated process pool with {GlobalSettings.Simulation.NUM_PROCESSES} workers")
                else:
                    self.process_pool = ThreadPoolExecutor(max_workers=1)
                    logger.info("Falling back to single-threaded executor")

                self.controller.grid = Grid(
                    self.controller.dimensions,  # Access through controller
                    self.controller.neighborhood_type,  # Access through controller
                    self.controller.dimension_type,  # Access through controller
                    rule=self.controller.rule,  # Pass the rule instance
                    unique_id=self.controller._unique_id  # Access through controller
                )
                self.controller.grid.setup_shared_memory()

            except Exception as e:
                logger.error(f"Failed to reinitialize process pool: {e}")
                self.process_pool = ThreadPoolExecutor(max_workers=1)
                logger.info("Falling back to single-threaded executor")

            # Restore the rule and parameters
            if not self.controller.set_rule(current_rule_name):  # Access through controller
                logger.error(f"Failed to restore rule {current_rule_name}")
                # Handle the error appropriately, perhaps by setting a default rule

            # Reapply parameters
            self.controller.rule.params = current_params
            logger.debug(f"Restored rule parameters: {self.controller.rule.params}")

            # Initialize new state
            logger.debug(f"Initializing new state with density: {initial_density}")
            density = initial_density if initial_density is not None else GlobalSettings.Simulation.INITIAL_NODE_DENSITY
            self._initialize_random_state(density)

            # Restore spacing after reset
            GlobalSettings.Visualization.set_node_spacing(current_spacing)

            # Reset step counter
            self.step_count = 0
            if self.step_label is not None:
                self.step_label.config(text=f"Step: {self.step_count}")

            logger.info("Simulation reset complete")

            # Log the rule parameters after the reset
            logger.debug(f"Rule parameters AFTER reset: {self.controller.rule.params}")

            # Force complete redraw
            logger.debug("Forcing complete redraw")
            self._safe_plot_update()

        except Exception as e:
            logger.error(f"Error in simulation reset: {str(e)}\nTraceback:\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"Failed to reset simulation: {e}")
                                                        
    def _initialize_random_state(self, density: float):
        """Initialize grid with random active cells based on density"""
        try:
            logger.debug(f"Entering _initialize_random_state with density {density}")
            
            # Calculate total cells based on current dimensions
            total_cells = np.prod(self.dimensions)
            active_cells = int(total_cells * density)
            
            # Clear the existing grid array before initializing
            self.grid.grid_array.fill(-1.0)
            
            # Create random active cell indices using vectorized operation
            active_indices = np.random.choice(
                total_cells, 
                size=active_cells, 
                replace=False
            )
            
            logger.debug(f"Total cells: {total_cells}, Active cells: {active_cells}")
            # logger.debug(f"Active indices: {active_indices}")
            
            # Set active cells directly using flat indices
            num_activated = 0
            for idx in active_indices:
                # Convert flat index to coordinates
                coords = tuple(_unravel_index(idx, self.dimensions))
                
                # Validate coordinates are within bounds
                if all(0 <= c < d for c, d in zip(coords, self.dimensions)):
                    try:
                        self.grid.set_node_state(coords, 1.0)  # Set state to 1.0 for active
                        num_activated += 1
                        # logger.debug(f"Activated node at coordinates: {coords}")
                    except Exception as e:
                        logger.error(f"Error setting node state at {coords}: {e}")
                    else:
                        logger.warning(f"Coordinates {coords} out of bounds, skipping")
            
            logger.debug(f"Number of nodes activated: {num_activated}")
            
            # REMOVE THIS SECTION
            # Initialize edges with optimized method - use grid directly
            #try:
            #    self.grid.initialize_edges()
            #    logger.debug("Edge initialization completed")
            #except Exception as e:
            #    logger.error(f"Error initializing edges: {e}")
            
        except Exception as e:
            logger.error(f"Error in random state initialization: {str(e)}\nTraceback:\n{traceback.format_exc()}")
            raise
                    
    @log_errors
    def zoom_in(self):
        """Zoom in to the visualization"""
        self.ax.set_xlim(self.ax.get_xlim()[0] / GlobalSettings.Visualization.ZOOM_FACTOR,
                        self.ax.get_xlim()[1] / GlobalSettings.Visualization.ZOOM_FACTOR)
        self.ax.set_ylim(self.ax.get_ylim()[0] / GlobalSettings.Visualization.ZOOM_FACTOR,
                        self.ax.get_ylim()[1] / GlobalSettings.Visualization.ZOOM_FACTOR)
        
        if self.controller.dimension_type == Dimension.THREE_D and isinstance(self.ax, Axes3DType): # type: ignore
            z_min, z_max = self.ax.get_zlim()
            self.ax.set_zlim(z_min / GlobalSettings.Visualization.ZOOM_FACTOR,
                            z_max / GlobalSettings.Visualization.ZOOM_FACTOR)
        
        self.canvas.draw()
        logger.debug("Zoomed in")

    @log_errors
    def zoom_out(self):
        """Zoom out from the visualization"""
        self.ax.set_xlim(self.ax.get_xlim()[0] * GlobalSettings.Visualization.ZOOM_FACTOR,
                        self.ax.get_xlim()[1] * GlobalSettings.Visualization.ZOOM_FACTOR)
        self.ax.set_ylim(self.ax.get_ylim()[0] * GlobalSettings.Visualization.ZOOM_FACTOR,
                        self.ax.get_ylim()[1] * GlobalSettings.Visualization.ZOOM_FACTOR)
        
        if self.controller.dimension_type == Dimension.THREE_D and isinstance(self.ax, Axes3DType): # type: ignore
            z_min, z_max = self.ax.get_zlim() # type: ignore
            self.ax.set_zlim(z_min * GlobalSettings.Visualization.ZOOM_FACTOR, # type: ignore
                            z_max * GlobalSettings.Visualization.ZOOM_FACTOR)
        
        self.canvas.draw()
        logger.debug("Zoomed out")

    def _safe_plot_update(self):
        """Optimized plot update with caching"""
        current_hash = self._calculate_plot_hash()
        # logger.debug(f"Calculated plot hash: {current_hash}")
        
        # Clear the figure and create new axes
        self.fig.clear()
        
        # Create appropriate axes type
        if self.controller.dimension_type == Dimension.THREE_D:
            self.ax = self.fig.add_subplot(111, projection='3d')
            if hasattr(self, '_current_elev') and hasattr(self, '_current_azim'):
                self.ax.view_init(elev=self._current_elev, azim=self._current_azim)
            logger.debug("Created 3D axes")
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_aspect('equal')
            logger.debug("Created 2D axes")
        
        # Set axes properties
        self.ax.set_facecolor(GlobalSettings.Colors.BACKGROUND)
        self.ax.grid(False)
        self.ax.set_axisbelow(True)
        self.ax.tick_params(colors='gray')
        
        # Remove spines
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        # Update based on dimension
        logger.debug("Updating plot based on dimension")
        if self.controller.dimension_type == Dimension.THREE_D:
            self._update_3d_plot()
        else:
            self._update_2d_plot()

        # Force integer ticks - corrected MultipleLocator usage
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if self.controller.dimension_type == Dimension.THREE_D:
            self.ax.zaxis.set_major_locator(MaxNLocator(integer=True)) # type: ignore

        # Force immediate draw
        self.canvas.draw()
        self.canvas.flush_events()
                                    
    def _calculate_plot_hash(self) -> str:
        """Calculate hash of current plot state for cache comparison"""
        state = self.controller.get_state()
        grid_hash = hash(state['grid_state'].tobytes())
        edge_hash = hash(state['edge_matrix'].tobytes())
        highlighted_hash = hash(frozenset(self.controller.highlighted_edges))
        view_hash = hash((self._current_azim, self._current_elev))
        return f"{grid_hash}_{edge_hash}_{highlighted_hash}_{view_hash}"

    def _manage_plot_cache(self):
        """Manage plot cache size and cleanup old entries"""
        current_time = time.time()
        cache_size = len(self._plot_cache)
        
        if cache_size > 100:  # Maximum cache entries
            # Remove oldest entries
            oldest_entries = sorted(self._plot_cache.items(), 
                                key=lambda x: x[1]['timestamp'])[:cache_size-50]
            for key, _ in oldest_entries:
                del self._plot_cache[key]

    def _store_view_state(self):
        """Store current view state"""
        self._view_state = {
            'xlim': self.ax.get_xlim(),
            'ylim': self.ax.get_ylim(),
            'zlim': self.ax.get_zlim() if isinstance(self.ax, Axes3DType) else None, # type: ignore
            'azim': self._current_azim,
            'elev': self._current_elev
            }
            
    def _draw_edges_batch(self, edges_data):
        """Draw edges in batches using LineCollection"""
        segments = []
        colors = []
        for (start, end), color in edges_data:
            segments.append([start, end])
            colors.append(color)
            
        line_collection = LineCollection(segments, 
                                    colors=colors,
                                    alpha=GlobalSettings.Visualization.EDGE_OPACITY,
                                    linewidths=GlobalSettings.Visualization.EDGE_WIDTH)
        self.ax.add_collection(line_collection) # type: ignore

    def _calculate_coordinates_vectorized(self, visible_indices):
        """Vectorized coordinate calculation"""
        if self.controller.dimension_type == Dimension.THREE_D:
            grid_size = self.controller.dimensions[0]
            z = visible_indices // (grid_size * grid_size)
            remainder = visible_indices % (grid_size * grid_size)
            y = remainder // grid_size
            x = remainder % grid_size
            
            scale = GlobalSettings.Visualization.EDGE_SCALE * (1.0 + GlobalSettings.Visualization.NODE_SPACING)
            return np.column_stack([x * scale, y * scale, z * scale])
        else:
            grid_size = self.controller.dimensions[0]
            y = visible_indices // grid_size
            x = visible_indices % grid_size
            
            scale = GlobalSettings.Visualization.EDGE_SCALE * (1.0 + GlobalSettings.Visualization.NODE_SPACING)
            return np.column_stack([x * scale, y * scale])
                                                   
    def _calculate_node_coordinates(self) -> Dict[int, Tuple[float, ...]]:
        """Calculate scaled coordinates for all active nodes"""
        # Get current state
        grid_array = self.controller.grid.grid_array
        
        # Get visible nodes
        visible_mask = grid_array.ravel() > GlobalSettings.Visualization.NODE_VISIBILITY_THRESHOLD
        visible_indices = np.where(visible_mask)[0]
        
        # Create a mapping from 1D index to scaled coordinates
        coords_map = {}
        
        # Get current spacing and scale
        base_scale = GlobalSettings.Visualization.EDGE_SCALE
        spacing = GlobalSettings.Visualization.NODE_SPACING
        total_scale = base_scale * (1.0 + spacing)
        
        if self.controller.dimension_type == Dimension.THREE_D:
            # For 3D grid, convert 1D indices to 3D coordinates
            grid_size = self.controller.dimensions[0]  # Assuming cubic grid
            for idx in visible_indices:
                # Convert 1D index to 3D coordinates
                z = idx // (grid_size * grid_size)
                remainder = idx % (grid_size * grid_size)
                y = remainder // grid_size
                x = remainder % grid_size
                
                # Scale coordinates
                x_scaled = x * total_scale
                y_scaled = y * total_scale
                z_scaled = z * total_scale
                
                coords_map[idx] = (x_scaled, y_scaled, z_scaled)
        else:
            # For 2D grid, convert 1D indices to 2D coordinates
            grid_size = self.controller.dimensions[0]  # Assuming square grid
            for idx in visible_indices:
                # Convert 1D index to 2D coordinates
                row = idx // grid_size  # This gives y coordinate
                col = idx % grid_size   # This gives x coordinate
                
                # Scale coordinates
                x_scaled = col * total_scale
                y_scaled = row * total_scale
                
                coords_map[idx] = (x_scaled, y_scaled)
        
        return coords_map

    def _should_redraw(self) -> bool:
        """Determine if redraw is needed based on state changes"""
        current_hash = self._calculate_plot_hash()
        if current_hash != self._last_plot_hash:
            self._last_plot_hash = current_hash
            return True
        return False
                                                                                                        
    def _update_2d_plot(self):
        """Update the 2D plot with optimized logging"""
        try:
            # Get current state directly from grid arrays
            grid_array = self.controller.grid.grid_array
            edge_matrix = self.controller.grid.neighborhood_data.edge_matrix
            
            # Log initial edge count
            total_edges = np.sum(np.triu(edge_matrix))  # Count only upper triangle
            logger.debug(f"Initial edge matrix contains {total_edges} edges")
            
            # Identify active nodes
            visible_mask = grid_array.ravel() > GlobalSettings.Visualization.NODE_VISIBILITY_THRESHOLD
            visible_indices = np.where(visible_mask)[0]
            
            logger.debug(f"Updating plot with {len(visible_indices)} visible nodes")
            
            # Track edges - directly from edge matrix
            current_edges = set()
            visible_edge_count = 0
            
            # Check all pairs of visible nodes for edges
            for i_idx, i in enumerate(visible_indices):
                for j_idx, j in enumerate(visible_indices[i_idx+1:], i_idx+1):
                    # Get the actual node indices from visible_indices
                    node_i = visible_indices[i_idx]
                    node_j = visible_indices[j_idx]
                    
                    if edge_matrix[node_i, node_j]:
                        current_edges.add((node_i, node_j))
                        visible_edge_count += 1

            logger.debug(f"Found {visible_edge_count} edges between visible nodes")
            
            # Track new nodes and edges
            if self.controller.generation == 0:
                # On initial render, don't highlight anything
                new_nodes = set()
                new_edges = set()
                self.controller.highlighted_nodes = set(visible_indices)
                self.controller.highlighted_edges = current_edges
            else:
                # Calculate new nodes and edges
                previous_nodes = self.controller.highlighted_nodes
                new_nodes = set(visible_indices) - previous_nodes
                self.controller.highlighted_nodes = set(visible_indices)
                
                previous_edges = self.controller.highlighted_edges
                new_edges = current_edges - previous_edges if self.highlight_var.get() else set()
                self.controller.highlighted_edges = current_edges

            # Compute coordinates
            coords_map = self._calculate_node_coordinates()
            
            # Check if coords_map is empty
            if not coords_map:
                logger.warning("No active nodes to plot.")
                self.ax.clear()
                self.canvas.draw()
                return
            
            x_coords, y_coords, states_plot = [], [], []
            
            # Collect coordinates for all visible nodes
            for idx in visible_indices:
                if idx in coords_map:
                    x, y = coords_map[idx]
                    x_coords.append(x)
                    y_coords.append(y)
                    states_plot.append(grid_array.ravel()[idx])
                    
            # Convert to numpy arrays
            x_coords = np.array(x_coords, dtype=np.float64)
            y_coords = np.array(y_coords, dtype=np.float64)
            states_plot = np.array(states_plot, dtype=np.float64)

            # Update plot limits dynamically
            self._update_plot_limits()
            
            # Draw edges
            edge_count = 0
            logger.debug(f"Starting to draw {len(current_edges)} edges")
            
            # Draw all edges between visible nodes
            for i, j in current_edges:
                if i in coords_map and j in coords_map:
                    x1, y1 = coords_map[i]
                    x2, y2 = coords_map[j]
                    # Color edges based on highlighting setting
                    edge_color = (GlobalSettings.Colors.EDGE_HIGHLIGHT 
                                if (i, j) in new_edges and self.highlight_var.get()
                                else GlobalSettings.Colors.EDGE_NORMAL)
                    self.ax.plot([x1, x2], [y1, y2], 
                            c=edge_color,
                            alpha=GlobalSettings.Visualization.EDGE_OPACITY,
                            linewidth=GlobalSettings.Visualization.EDGE_WIDTH)
                    edge_count += 1

            # Draw nodes with optional highlighting
            if len(x_coords) > 0:
                edge_colors = np.array([GlobalSettings.Colors.NODE_HIGHLIGHT if (idx in new_nodes and self.highlight_var.get())
                                    else 'none' for idx in visible_indices])
                edge_widths = np.array([2.0 if (idx in new_nodes and self.highlight_var.get())
                                    else GlobalSettings.Visualization.EDGE_WIDTH for idx in visible_indices])
                
                self.ax.scatter(x_coords, y_coords, 
                        c=states_plot, 
                        cmap='viridis', 
                        norm=Normalize(0, 1),
                        alpha=GlobalSettings.Visualization.NODE_OPACITY, 
                        s=GlobalSettings.Visualization.NODE_SIZE * 100,
                        edgecolors=edge_colors.tolist(),
                        linewidths=edge_widths)

            # Set axes properties
            self.ax.set_facecolor(GlobalSettings.Colors.BACKGROUND)
            self.ax.grid(False)
            self.ax.set_axisbelow(True)
            self.ax.tick_params(colors='gray')
            
            # Remove spines
            for spine in self.ax.spines.values():
                spine.set_visible(False)

            # Log detailed counts
            logger.debug(f"Drew {edge_count} edges and {len(x_coords)} nodes")
            logger.debug(f"Edge matrix contains {total_edges} total edges, {visible_edge_count} between visible nodes")
            logger.debug(f"New nodes: {len(new_nodes)}, New edges: {len(new_edges)}")

        except Exception as e:
            logger.error(f"Error in _update_2d_plot: {e}")
            logger.error(traceback.format_exc())
            raise
                                                                                            
    def _update_3d_plot(self):
        """Update the 3D plot"""
        try:
            # Get current state directly from grid arrays
            grid_array = self.controller.grid.grid_array
            edge_matrix = self.controller.grid.neighborhood_data.edge_matrix
            
            # Log initial edge count
            total_edges = np.sum(np.triu(edge_matrix))  # Count only upper triangle
            logger.debug(f"Initial edge matrix contains {total_edges} edges")
            
            # Identify active nodes
            visible_mask = grid_array.ravel() > GlobalSettings.Visualization.NODE_VISIBILITY_THRESHOLD
            visible_indices = np.where(visible_mask)[0]
            
            logger.debug(f"Updating 3D plot with {len(visible_indices)} visible nodes")
            
            # Compute coordinates
            coords_map = self._calculate_node_coordinates()
            
            # Check if coords_map is empty
            if not coords_map:
                logger.warning("No active nodes to plot.")
                self.ax.clear()
                self.canvas.draw()
                return
            
            x_coords, y_coords, z_coords, states_plot = [], [], [], []
            
            # Collect coordinates for all visible nodes
            for idx in visible_indices:
                if idx in coords_map:
                    x, y, z = coords_map[idx]
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(z)
                    states_plot.append(grid_array.ravel()[idx])
                    
            # Convert to numpy arrays
            x_coords = np.array(x_coords, dtype=np.float64)
            y_coords = np.array(y_coords, dtype=np.float64)
            z_coords = np.array(z_coords, dtype=np.float64)
            states_plot = np.array(states_plot, dtype=np.float64)

            # Update plot limits dynamically
            self._update_plot_limits()
            
            # Clear the axes
            self.ax.cla()

            # Track edges - directly from edge matrix
            current_edges = set()
            visible_edge_count = 0
            
            # Check all pairs of visible nodes for edges
            for i_idx, i in enumerate(visible_indices):
                for j_idx, j in enumerate(visible_indices[i_idx+1:], i_idx+1):
                    # Get the actual node indices from visible_indices
                    node_i = visible_indices[i_idx]
                    node_j = visible_indices[j_idx]
                    
                    if edge_matrix[node_i, node_j]:
                        current_edges.add((node_i, node_j))
                        visible_edge_count += 1

            logger.debug(f"Found {visible_edge_count} edges between visible nodes")
            
            # Track new nodes and edges
            if self.controller.generation == 0:
                # On initial render, don't highlight anything
                new_nodes = set()
                new_edges = set()
                self.controller.highlighted_nodes = set(visible_indices)
                self.controller.highlighted_edges = current_edges
            else:
                # Calculate new nodes and edges
                previous_nodes = self.controller.highlighted_nodes
                new_nodes = set(visible_indices) - previous_nodes
                self.controller.highlighted_nodes = set(visible_indices)
                
                previous_edges = self.controller.highlighted_edges
                new_edges = current_edges - previous_edges if self.highlight_var.get() else set()
                self.controller.highlighted_edges = current_edges
                
            # Draw edges
            edge_count = 0
            logger.debug(f"Starting to draw {len(current_edges)} edges")
            
            # Prepare edge segments and colors
            segments = []
            colors = []
            
            # Draw all edges between visible nodes
            for i, j in current_edges:
                if i in coords_map and j in coords_map:
                    x1, y1, z1 = coords_map[i]
                    x2, y2, z2 = coords_map[j]
                    # Color edges based on highlighting setting
                    edge_color = (GlobalSettings.Colors.EDGE_HIGHLIGHT 
                                if (i, j) in new_edges and self.highlight_var.get()
                                else GlobalSettings.Colors.EDGE_NORMAL)
                    segments.append([(x1, y1, z1), (x2, y2, z2)])
                    colors.append(edge_color)
            
            # Draw edges using Line3DCollection
            if segments:
                line_collection = Line3DCollection(segments, 
                                            colors=colors,
                                            alpha=GlobalSettings.Visualization.EDGE_OPACITY,
                                            linewidths=GlobalSettings.Visualization.EDGE_WIDTH)
                self.ax.add_collection3d(line_collection) # type: ignore

            # Draw nodes with highlighted outlines for new nodes
            norm = Normalize(0, 1)
            
            # Set edge color based on whether all nodes are highlighted (frame 0)
            edge_color = GlobalSettings.Colors.EDGE_HIGHLIGHT if self.controller.generation == 0 else 'face'
            
            scatter = self.ax.scatter(x_coords, y_coords, z_coords,
                        c=states_plot, 
                        cmap='viridis', 
                        norm=norm,
                        alpha=GlobalSettings.Visualization.NODE_OPACITY, 
                        edgecolors=edge_color,
                        linewidths=GlobalSettings.Visualization.EDGE_WIDTH)

            # Set axes properties
            self.ax.set_facecolor(GlobalSettings.Colors.BACKGROUND)
            self.ax.grid(False)
            self.ax.set_axisbelow(True)
            self.ax.tick_params(colors='gray')
            
            # Remove spines
            for spine in self.ax.spines.values():
                spine.set_visible(False)

            # Set the view angle
            if isinstance(self.ax, Axes3DType): # type: ignore
                self.ax.view_init(elev=self._current_elev, azim=self._current_azim) # type: ignore
                
            # Log only total counts
            logger.debug(f"Drew {edge_count} edges and {len(x_coords)} nodes in 3D")
            visible_edge_count = np.sum(edge_matrix[visible_mask][:, visible_mask]) // 2
            logger.debug(f"Edge matrix contains {np.sum(np.triu(edge_matrix)) // 2} total edges, {visible_edge_count} between visible nodes")
            logger.debug(f"New nodes: {len(new_nodes)}, New edges: {len(new_edges)}")

            # Force draw
            self.canvas.draw()

        except Exception as e:
            logger.error(f"Error in _update_3d_plot: {e}")
            logger.error(traceback.format_exc())
            raise
                           
    def _manage_cache(self):
        """Manage plot cache size and cleanup"""
        current_time = time.time()
        # Remove old cache entries
        for hash_key in list(self._plot_cache.keys()):
            if current_time - self._plot_cache[hash_key]['timestamp'] > 30:  # 30 second cache
                del self._plot_cache[hash_key]
                                  
    def _on_mouse_press(self, event):
        """Handle mouse press"""
        try:
            if not hasattr(event, 'inaxes'):
                return
                
            if event.inaxes == self.ax:
                if self.controller.dimension_type == Dimension.THREE_D:
                    # For 3D plots, enable rotation
                    self._last_mouse_pos = (event.xdata, event.ydata)
                    self.rotation_enabled = True
                else:
                    # For 2D plots, enable panning
                    self.panning_enabled = True
                    self.pan_start_x = event.xdata
                    self.pan_start_y = event.ydata
                    self.x_min, self.x_max = self.ax.get_xlim()
                    self.y_min, self.y_max = self.ax.get_ylim()
        except AttributeError as e:
            logger.warning(f"AttributeError in _on_mouse_press: {e}")
            return
                    
    @log_errors
    def save_state(self):
        """Save current simulation state"""
        try:
            with self._update_lock:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'simulation_state_{timestamp}.json'
                filepath = os.path.join(APP_PATHS['saves'], filename)
                
                # Get state from controller
                state = self.controller.get_state()
                
                # Convert NumPy arrays to lists
                state['grid_state'] = self.controller.grid.grid_array.tolist()
                state['edge_matrix'] = self.controller.grid.neighborhood_data.edge_matrix.tolist()
                
                try:
                    with open(filepath, 'w') as f:
                        json.dump(state, f, indent=2)
                    logger.info(f"Simulation state saved to {filename}")
                except Exception as e:
                    logger.error(f"Error saving state to file: {e}")
                    messagebox.showerror("Error", f"Failed to save state to file: {e}")

        except Exception as e:
            logger.error(f"Error saving simulation state: {str(e)}")
            messagebox.showerror("Error", f"Failed to save simulation state: {str(e)}")
        
    @log_errors
    def load_state(self):
        """Load simulation state from file"""
        try:
            with self._update_lock:
                filename = filedialog.askopenfilename(
                    initialdir=APP_PATHS['saves'],
                    title="Select simulation state file",
                    filetypes=[("JSON files", "*.json")]
                )
                
                if filename:
                    try:
                        with open(filename, 'r') as f:
                            state = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        logger.error(f"Error loading state from file: {e}")
                        messagebox.showerror("Error", f"Failed to load state from file: {e}")
                        return

                    try:
                        # Reset controller with loaded state
                        self.controller.reset()
                        # Update controller state
                        self.controller.set_state(state)
                        # Reset GUI counters
                        self.step_count = self.controller.generation
                        self.new_nodes.clear()
                        self.new_edges.clear()
                        # Update display
                        self._safe_plot_update()
                        logger.info(f"Simulation state loaded from {filename}")
                    except Exception as e:
                        logger.error(f"Error setting state: {e}")
                        messagebox.showerror("Error", f"Failed to set state: {e}")

        except Exception as e:
            logger.error(f"Error loading simulation state: {str(e)}")
                
    def _on_mouse_move(self, event):
        """Handle mouse movement for 3D rotation"""
        if event.inaxes == self.ax and hasattr(event, 'button') and event.button == 1:
            if not hasattr(self, '_last_mouse_pos') or self._last_mouse_pos is None:
                self._last_mouse_pos = (event.xdata, event.ydata)
                return
                
            if event.xdata is None or event.ydata is None:
                return
                
            dx = event.xdata - self._last_mouse_pos[0]
            dy = event.ydata - self._last_mouse_pos[1]
            
            if isinstance(self.ax, Axes3DType): # type: ignore
                self._current_azim = (self._current_azim + dx * 50) % 360
                self._current_elev = np.clip(self._current_elev + dy * 50, -90, 90)
                self.ax.view_init(elev=self._current_elev, azim=self._current_azim)
                self.canvas.draw()
            
            self._last_mouse_pos = (event.xdata, event.ydata)
            
    def _on_mouse_release(self, event):
        """Handle mouse release"""
        self._last_mouse_pos = None
        self.rotation_enabled = False
        self.panning_enabled = False
        self.pan_start_x = None
        self.pan_start_y = None
        
    def _on_mouse_drag(self, event):
        """Handle mouse drag for 3D rotation or 2D panning"""
        try:
            if not hasattr(event, 'inaxes') or event.inaxes != self.ax:
                return

            if self.controller.dimension_type == Dimension.THREE_D:
                # Perform 3D rotation
                if self.rotation_enabled:
                    if event.xdata is None or event.ydata is None or self._last_mouse_pos is None:
                        return

                    # Calculate movement
                    dx = event.xdata - self._last_mouse_pos[0]
                    dy = event.ydata - self._last_mouse_pos[1]

                    # Scale the rotation based on the plot size
                    rotation_scale = 100.0 / self.canvas.get_tk_widget().winfo_width()
                    
                    # Update rotation angles - swap dx and dy for left/right rotation
                    self._current_azim = (self._current_azim + dy * 50 * rotation_scale) % 360
                    self._current_elev = np.clip(self._current_elev + dx * 50 * rotation_scale, -90, 90)

                    # Apply rotation
                    self.ax.view_init(elev=self._current_elev, azim=self._current_azim) # type: ignore
                    
                    # Store new position
                    self._last_mouse_pos = (event.xdata, event.ydata)
                    
                    # Redraw
                    self.canvas.draw()
                else:
                    # Perform 2D panning
                    if self.panning_enabled:
                        if event.xdata is None or event.ydata is None or self.pan_start_x is None or self.pan_start_y is None:
                            return

                        # Calculate pan distance in data coordinates
                        dx = (event.xdata - self.pan_start_x);
                        dy = (event.ydata - self.pan_start_y);

                        # Get current limits
                        x_min, x_max = self.ax.get_xlim()
                        y_min, y_max = self.ax.get_ylim()
                        
                        # Set new limits
                        self.ax.set_xlim(x_min - dx, x_max - dx)
                        self.ax.set_ylim(y_min - dy, y_max - dy)
                        
                        # Store current position
                        self.pan_start_x = event.xdata
                        self.pan_start_y = event.ydata
                        # Redraw
                        self.canvas.draw()
                    
        except Exception as e:
            logger.error(f"Error in mouse drag handling: {str(e)}")
            self.rotation_enabled = False
            self.panning_enabled = False
                    
    def _initialize_3d_plot(self):
        """Initialize 3D plot settings"""
        try:
            if not isinstance(self.ax, Axes3DType): # type: ignore
                logger.debug("Initializing 3D plot")
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection='3d')
                
                # Set initial view
                self._current_azim = 0
                self._current_elev = 30
                self.ax.view_init(elev=self._current_elev, azim=self._current_azim)
                
                # Set equal aspect ratio
                self.ax.set_box_aspect([1, 1, 1])
                
                # Clear any existing plots
                self.ax.clear()
                
                # Update plot limits
                self._update_plot_limits()
                
                # Force draw
                self.canvas.draw()
                self.canvas.flush_events()
                
                logger.debug("3D plot initialized successfully")
                self.fig.clear()
                self.ax = self.fig.add_subplot(111, projection='3d')
            
            # Set initial view
            self._current_azim = 0
            self._current_elev = 30
            self.ax.view_init(elev=self._current_elev, azim=self._current_azim)
            
            # Set equal aspect ratio
            self.ax.set_box_aspect([1, 1, 1])
            
            # Clear any existing plots
            self.ax.clear()
            
            # Update plot limits
            self._update_plot_limits()
            
            # Force draw
            self.canvas.draw()
            self.canvas.flush_events()
            
            logger.debug("3D plot initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing 3D plot: {str(e)}\nTraceback:\n{traceback.format_exc()}")
            raise

    def _handle_mouse_move(self, event):
        """Handle mouse movement"""
        if hasattr(self.ax, 'get_proj') and self.rotation_enabled:  # Only for 3D
            # Convert tkinter coordinates to matplotlib data coordinates
            x = self.ax.get_xlim()[0] + (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * event.x / self.canvas.get_tk_widget().winfo_width()
            y = self.ax.get_ylim()[0] + (self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * (1 - event.y / self.canvas.get_tk_widget().winfo_height())
            
            if self._last_mouse_pos is not None:
                dx = x - self._last_mouse_pos[0]
                dy = y - self._last_mouse_pos[1]
                
                rotation_scale = 100.0 / self.canvas.get_tk_widget().winfo_width()
                self._current_azim = (self._current_azim - dx * 50 * rotation_scale) % 360
                self._current_elev = np.clip(self._current_elev + dy * 50 * rotation_scale, -90, 90)
                self.ax.view_init(elev=self._current_elev, azim=self._current_azim) # type: ignore
                self.canvas.draw()
            
            self._last_mouse_pos = (x, y)

    def _handle_scroll(self, event):
        """Handle scroll events"""
        if event.delta:
            factor = 0.9 if event.delta > 0 else 1.1
            self._zoom(factor)
                
    def _handle_mouse_press(self, event):
        """Handle mouse press"""
        try: # ADD TRY-EXCEPT BLOCK HERE
            if event.inaxes == self.ax:
                if self.controller.dimension_type == Dimension.THREE_D:
                    # For 3D plots, enable rotation
                    self._last_mouse_pos = (event.xdata, event.ydata)
                    self.rotation_enabled = True
                else:
                    # For 2D plots, enable panning
                    self.panning_enabled = True
                    self.pan_start_x = event.xdata
                    self.pan_start_y = event.ydata
                    self.x_min, self.x_max = self.ax.get_xlim()
                    self.y_min, self.y_max = self.ax.get_ylim()
        except AttributeError as e: # CATCH AttributeError
            logger.warning(f"AttributeError in _handle_mouse_press: {e}") # LOG WARNING
            return # GRACEFULLY RETURN
        
    def _handle_mouse_release(self, event):
        """Handle mouse release"""
        self.rotation_enabled = False
        self.panning_enabled = False
        self._last_mouse_pos = None
        self.pan_start_x = None
        self.pan_start_y = None

    def _handle_mouse_drag(self, event):
        """Handle mouse drag for 3D rotation or 2D panning"""
        try:
            if event.inaxes != self.ax:
                return

            if self.controller.dimension_type == Dimension.THREE_D:
                # Perform 3D rotation
                if self.rotation_enabled:
                    if event.xdata is None or event.ydata is None or self._last_mouse_pos is None:
                        return

                    # Calculate movement
                    dx = event.xdata - self._last_mouse_pos[0]
                    dy = event.ydata - self._last_mouse_pos[1]

                    # Scale the rotation based on the plot size
                    rotation_scale = 100.0 / self.canvas.get_tk_widget().winfo_width()
                    
                    # Update rotation angles
                    self._current_azim = (self._current_azim - dx * 50 * rotation_scale) % 360
                    self._current_elev = np.clip(self._current_elev + dy * 50 * rotation_scale, -90, 90)

                    # Apply rotation
                    self.ax.view_init(elev=self._current_elev, azim=self._current_azim) # type: ignore
                    
                    # Store new position
                    self._last_mouse_pos = (event.xdata, event.ydata)
                    
                    # Redraw
                    self.canvas.draw()
            else:
                # Perform 2D panning
                if self.panning_enabled:
                    if event.xdata is None or event.ydata is None or self.pan_start_x is None or self.pan_start_y is None:
                        return

                    # Calculate pan distance in data coordinates
                    dx = (event.xdata - self.pan_start_x);
                    dy = (event.ydata - self.pan_start_y);

                    # Get current limits
                    x_min, x_max = self.ax.get_xlim()
                    y_min, y_max = self.ax.get_ylim()
                    
                    # Set new limits
                    self.ax.set_xlim(x_min - dx, x_max - dx)
                    self.ax.set_ylim(y_min - dy, y_max - dy)
                    
                    # Store current position
                    self.pan_start_x = event.xdata
                    self.pan_start_y = event.ydata
                    # Redraw
                    self.canvas.draw()
                
        except Exception as e:
            logger.error(f"Error in mouse drag handling: {str(e)}")
            self.rotation_enabled = False
            self.panning_enabled = False
    
    @log_errors
    def _pan(self, dx: float, dy: float):
        """Pan the plot"""
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        
        self.ax.set_xlim(x_min - dx, x_max - dx)
        self.ax.set_ylim(y_min - dy, y_max - dy)

        # Store the new limits in the view state
        self._view_state['xlim'] = self.ax.get_xlim()
        self._view_state['ylim'] = self.ax.get_ylim()
        
    @log_errors
    def _on_scroll(self, event):
        """Handle mouse scroll"""
        logger.debug("Handling scroll event")
        factor = 0.9 if event.delta > 0 else 1.1
        self._zoom(factor)
        
    def _calculate_effective_spacing(self) -> float:
        """Calculate effective spacing compensated for zoom level"""
        base_spacing = GlobalSettings.Visualization.NODE_SPACING
        
        # Get current zoom factor from view state
        zoom_factor = self._view_state.get('zoom_factor', 1.0)
        
        # Compensate spacing based on zoom
        # When zoomed out (zoom_factor > 1), increase spacing
        # When zoomed in (zoom_factor < 1), decrease spacing
        effective_spacing = base_spacing + (1.0 - zoom_factor) # Linear Spacing
        
        # Ensure spacing stays within reasonable bounds
        min_spacing = 0.0
        max_spacing = GlobalSettings.Visualization.MAX_NODE_SPACING
        
        return float(np.clip(effective_spacing, min_spacing, max_spacing))
        
    def _zoom(self, factor: float):
        """Zoom both graph and plot with spacing compensation"""
        try:
            if isinstance(self.ax, Axes3DType): # type: ignore
                # 3D zoom handling remains the same
                # Update zoom factor
                self._view_state['zoom_factor'] *= factor
                
                # Get initial limits from full view
                max_coord = max(self.controller.dimensions) * GlobalSettings.Visualization.EDGE_SCALE
                margin = max_coord * 0.1
                initial_xlim = (-margin, max_coord + margin)
                initial_ylim = (-margin, max_coord + margin)
                initial_zlim = (-margin, max_coord + margin)
                
                # Get current view center
                xcenter = (initial_xlim[1] + initial_xlim[0]) / 2
                ycenter = (initial_ylim[1] + initial_ylim[0]) / 2
                zcenter = (initial_zlim[1] + initial_zlim[0]) / 2
                
                # Calculate new width based on zoom factor
                xwidth = (initial_xlim[1] - initial_xlim[0]) * self._view_state['zoom_factor']
                ywidth = (initial_ylim[1] - initial_ylim[0]) * self._view_state['zoom_factor']
                zwidth = (initial_zlim[1] - initial_zlim[0]) * self._view_state['zoom_factor']
                
                # Set new limits
                new_xlim = (xcenter - xwidth/2, xcenter + xwidth/2)
                new_ylim = (ycenter - ywidth/2, ycenter + ywidth/2)
                new_zlim = (zcenter - zwidth/2, zcenter + zwidth/2)
                
                self.ax.set_xlim(new_xlim)
                self.ax.set_ylim(new_ylim)
                self.ax.set_zlim(new_zlim)
                
                # Store new limits in view state
                self._view_state['xlim'] = new_xlim
                self._view_state['ylim'] = new_ylim
                self._view_state['zlim'] = new_zlim
                
                # Adjust node size and edge width based on zoom factor
                node_size = GlobalSettings.Visualization.NODE_SIZE / self._view_state['zoom_factor']
                edge_width = GlobalSettings.Visualization.EDGE_WIDTH / self._view_state['zoom_factor']
                
                # Clamp node size and edge width to reasonable values
                min_node_size = 0.1
                min_edge_width = 0.1
                
                node_size = max(node_size, min_node_size)
                edge_width = max(edge_width, min_edge_width)
                
                # Store adjusted values
                self._adjusted_node_size = node_size
                self._adjusted_edge_width = edge_width
                
            else:
                # Handle Zoom for 2D plot
                x_min, x_max = self.ax.get_xlim()
                y_min, y_max = self.ax.get_ylim()
                
                x_mid = (x_min + x_max) / 2
                y_mid = (y_min + y_max) / 2
                
                # Update zoom factor
                self._view_state['zoom_factor'] *= factor
                
                # Calculate new widths
                x_width = (x_max - x_min) * factor
                y_width = (y_max - y_min) * factor
                
                # Set new limits
                self.ax.set_xlim(x_mid - x_width / 2, x_mid + x_width / 2)
                self.ax.set_ylim(y_mid - y_width / 2, y_mid + y_width / 2)
                
                # Store new limits in view state
                self._view_state['xlim'] = self.ax.get_xlim()
                self._view_state['ylim'] = self.ax.get_ylim()
                
                # Force redraw with compensated spacing
                self._safe_plot_update()

            self.canvas.draw()
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error in zoom: {str(e)}")
                    
    def _setup_event_handlers(self):
        """Setup event handlers with proper cleanup tracking"""
        try:
            # Ensure canvas widget exists and is mapped
            if not hasattr(self, 'canvas') or not self.canvas:
                logger.error("Canvas not initialized")
                return
                
            canvas_widget = self.canvas.get_tk_widget()
            if not canvas_widget.winfo_exists():
                logger.error("Canvas widget does not exist")
                return
                
            # Wait for widget to be mapped
            self.root.update_idletasks()
            
            handlers = [
                ("<MouseWheel>", self._on_scroll, "scroll"),
                ("<ButtonPress-1>", self._on_mouse_press, "mouse_press"),
                ("<ButtonRelease-1>", self._on_mouse_release, "mouse_release"),
                ("<B1-Motion>", self._on_mouse_drag, "mouse_drag"),
                ("<Configure>", self._on_window_configure, "window_configure"),
                ("<Destroy>", self._on_window_destroy, "window_destroy")
            ]

            # Initialize binding list if not exists
            if not hasattr(self, '_event_bindings'):
                self._event_bindings = []

            # Bind handlers and track binding IDs
            for event, handler, name in handlers:
                try:
                    binding_id = canvas_widget.bind(event, handler)
                    self._event_bindings.append((event, binding_id, name))
                    logger.debug(f"Successfully bound {name} handler")
                except Exception as e:
                    logger.error(f"Error binding {name} handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error in event handler setup: {e}")
            raise
            
    def _cleanup_event_handlers(self):
        """Clean up event handlers to prevent memory leaks"""
        if not self._tk_destroyed and hasattr(self, 'canvas') and self.canvas is not None:
            widget = self.canvas.get_tk_widget()
            for event, binding_id, name in self._event_bindings:
                try:
                    widget.unbind(event, binding_id)
                except Exception as e:
                    logger.warning(f"Error unbinding {name} handler: {e}")
        self._event_bindings.clear()
        
    def _on_window_destroy(self, event):
        """Handle window destruction with proper cleanup"""
        try:
            # Clean up event handlers
            self._cleanup_event_handlers()

            # Clean up animation
            if hasattr(self, 'anim') and self.anim is not None:
                self.anim.event_source.stop()
                self.anim = None

            # Clean up matplotlib resources
            plt.close('all')

            # Clean up controller resources
            if hasattr(self, 'controller'):
                self.controller.interrupt_requested = True
                # Give time for any running operations to complete
                self.root.after(100, self._complete_cleanup)

        except Exception as e:
            logger.error(f"Error in window destruction cleanup: {e}")

    def _on_window_configure(self, event):
        """Handle window resize events"""
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = None

        # Cancel previous timer
        if self._resize_timer is not None:
            self.root.after_cancel(self._resize_timer)

        # Set new timer
        self._resize_timer = self.root.after(100, self._handle_resize)

    def _handle_resize(self):
        """Handle window resize with debouncing"""
        try:
            self._resize_timer = None
            if self.running:
                # Pause simulation during resize
                was_running = True
                self.running = False
            else:
                was_running = False

            # Update plot
            self._safe_plot_update()

            # Restore simulation state
            if was_running:
                self.running = True
                self.root.after(int(self.step_delay), self._run_simulation_step)

        except Exception as e:
            logger.error(f"Error handling resize: {e}")

    def _safe_simulation_update(self):
        """Safely update simulation state with thread synchronization"""
        if self._is_updating:
            return
            
        try:
            self._is_updating = True
            
            # Acquire locks in consistent order to prevent deadlocks
            with self._update_lock:
                with self._parameter_lock:
                    # Signal simulation thread to pause
                    self._simulation_event.clear()
                    
                    # Perform simulation step
                    self.controller.step()
                    
                    # Update visualization
                    if self.controller.dimension_type == Dimension.THREE_D:
                        self.update_plot_3d(self.step_count)
                    else:
                        self.update_plot_2d(self.step_count)
                        
                    # Force immediate draw
                    self.canvas.draw()
                    
                    # Signal simulation thread to resume
                    self._simulation_event.set()
                    
        except Exception as e:
            logger.error(f"Error in safe simulation update: {e}")
            raise
        finally:
            self._is_updating = False
            
    def _safe_parameter_update(self, param_name: str, value: Any, 
                             parameter_entries: Mapping[str, Union[tk.Entry, RuleTableEditor]]) -> None:
        """Safely update parameters with thread synchronization"""
        try:
            # Acquire parameter lock
            with self._parameter_lock:
                # Signal simulation to pause
                self._simulation_event.clear()
                
                # Perform parameter update
                self._update_parameter(param_name, value, parameter_entries)
                
                # Signal simulation to resume
                self._simulation_event.set()
                
        except Exception as e:
            logger.error(f"Error updating parameter {param_name}: {e}")
            messagebox.showerror("Error", f"Parameter update error: {e}")

    def _gui_update(self):
        """Update GUI with performance optimization"""
        try:
            logger.info("Entering _gui_update")
            if not self.running or self.paused:
                logger.info("Not running or paused, exiting _gui_update")
                return

            start_time = time.time()

            # Only update if needed
            if self._should_redraw():
                logger.info("Starting simulation step")
                # Perform simulation step
                try:
                    self.step_simulation()
                    logger.info("Simulation step completed")
                except Exception as step_error:
                    logger.error(f"Error in step_simulation: {step_error}\nTraceback:\n{traceback.format_exc()}")
                    raise
                
                # Check if interrupted
                if self.controller.interrupt_requested:
                    logger.info("Update interrupted, exiting")
                    return
                    
                # Measure frame time
                frame_time = time.time() - start_time
                self._frame_times.append(frame_time)
                
                logger.info(f"Frame time: {frame_time*1000:.1f}ms")

            # Manage cache periodically
            if time.time() - self._last_frame_time > 5:  # Every 5 seconds
                self._manage_cache()
                self._last_frame_time = time.time()

            # Schedule next update if still running
            if self.running and not self.paused and not self.controller.interrupt_requested:
                logger.info("Scheduling next update")
                try:
                    self.root.after(int(self.step_delay), self._gui_update)
                    logger.info("Next update scheduled")
                except Exception as schedule_error:
                    logger.error(f"Error scheduling next update: {schedule_error}\nTraceback:\n{traceback.format_exc()}")
                    raise

        except Exception as e:
            logger.error(f"Error in GUI update: {e}\nTraceback:\n{traceback.format_exc()}")
            self.running = False
            self.start_button.config(text="Start")
            self.controller.interrupt_requested = False
            
    def _force_initial_render(self):
        """Force initial render after GUI is fully initialized"""
        try:
            # Only update if no plot exists
            if len(self.ax.get_children()) == 0:
                self._safe_plot_update()
                
            self.canvas.draw()
            self.canvas.flush_events()
            
        except Exception as e:
            logger.error(f"Error in force initial render: {e}")
                                  
if __name__ == "__main__":
    print("Entering __main__ block")
    try:
        # Initialize directory structure first
        APP_PATHS, BASE_PATH = setup_directories()
        
        # Setup single logger instance
        logger = setup_logging(APP_PATHS['logs'])
        
        # Store logger globally
        global_logger = logger
        
        logger.info("Starting application")
        
        # Initialize the simulation with initial rule
        logger.debug("Initializing simulation")
        initial_rule = GlobalSettings.Defaults.DEFAULT_RULE
        
        # Log before GUI creation
        logger.debug("Before SimulationGUI instantiation")
        
        # Create GUI
        gui = None
        try:
            gui = SimulationGUI(initial_rule)
            
            # Log after GUI creation
            logger.debug("After SimulationGUI instantiation")
            
            gui.run()
        except Exception as e:
            logger.error(f"Error running GUI: {e}")
            raise
        finally:
            if gui is not None:
                try:
                    gui.cleanup()
                except Exception as cleanup_error:
                    logger.error(f"Error during GUI cleanup: {cleanup_error}")
            
            # Final matplotlib cleanup
            try:
                plt.close('all')
            except Exception as plt_error:
                logger.error(f"Error closing matplotlib windows: {plt_error}")
                
        logger.info("Application completed normally")
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
        else:
            print(f"Fatal error before logger initialization: {str(e)}\n{traceback.format_exc()}")
        raise
    finally:
        # Final cleanup
        try:
            plt.close('all')
        except:
            pass
