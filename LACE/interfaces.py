# =========== START of interfaces.py ===========
from __future__ import annotations
import ast
import math
import inspect
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from typing import ClassVar, Dict, List, Set, Tuple, Optional, Union, Any, TypeVar, TYPE_CHECKING
from numba import njit
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
plt.ioff()
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import traceback
import numpy as np
from typing_extensions import Protocol
import warnings
import copy

# --- Keep TYPE_CHECKING block if needed elsewhere ---
if TYPE_CHECKING:
    from .lace_app import Grid # Keep this if other methods in Rule NEED the type hint at check time
    pass
# ---

from .logging_config import logger, LogSettings
from .enums import Dimension, NeighborhoodType, StateType, ShapeType, TieBreaker
from .settings import GlobalSettings
from .utils import (
    timer_decorator, _ravel_multi_index, _unravel_index
    )   


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
#                     RULES                    #
################################################

        
class BaseMetrics:
    """Base metrics implementation that follows RuleMetrics protocol"""

    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float:  # Changed parameter
        """Default compute implementation"""
        return 0.0

    @staticmethod
    @njit(cache=True)
    def active_neighbor_ratio(states: npt.NDArray[np.float64],
                            neighbor_indices: npt.NDArray[np.int64],
                            neighborhood_data: 'NeighborhoodData') -> float:
        """Ratio of active neighbors to total neighbors"""
        # Handle the case where neighbor_indices is None or empty
        if neighbor_indices is None or len(neighbor_indices) == 0:
            return 0.0
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        return float(np.sum(states[valid_neighbors] > 0) / len(valid_neighbors))

    @staticmethod
    @njit(cache=True)
    def edge_density_jit(states: npt.NDArray[np.float64],
                        neighbor_indices: npt.NDArray[np.int64],
                        edge_exists: npt.NDArray[np.bool_]) -> float:
        """JIT-compiled version of edge_density that works with primitive types"""
        # Handle the case where neighbor_indices is None or empty
        if neighbor_indices is None or len(neighbor_indices) == 0:
            return 0.0
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) < 2:
            return 0.0

        possible_edges = len(valid_neighbors) * (len(valid_neighbors) - 1) / 2
        if possible_edges == 0:
            return 0.0

        actual_edges = 0.0
        for i in range(len(valid_neighbors)):
            for j in range(i + 1, len(valid_neighbors)):
                # Check if the edge exists using the edge_exists array
                idx = i * len(valid_neighbors) + j
                if idx < len(edge_exists) and edge_exists[idx]:
                    actual_edges += 1.0

        return float(actual_edges / possible_edges)


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
    neighborhood_compatibility: List[str] # Keep this, might be useful later
    parent_rule: Optional[str]
    rating: Optional[int] = None
    notes: Optional[str] = None
    neighborhood_type: Optional['NeighborhoodType'] = None # Keep this
    allowed_initial_conditions: List[str] = field(default_factory=list)
    allow_rule_tables: bool = True
    favorite: bool = False
    


class Rule(ABC):
    PARAMETER_METADATA: ClassVar[Dict[str, Dict[str, Any]]] = {}
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = set()
    produces_binary_edges: ClassVar[bool] = False
    needs_node_history: ClassVar[bool] = False # ADDED: Flag for history requirement
    # --- ADDED: State Type Information ---
    node_state_type: ClassVar[StateType] = StateType.BINARY # Default to binary
    edge_state_type: ClassVar[StateType] = StateType.BINARY # Default to binary (use produces_binary_edges for check)
    # --- ADDED: State Range Parameters (Defaults assume 0-1 range) ---
    min_node_state: ClassVar[float] = 0.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0
    # ---
    # --- ADDED: Flags for JIT Optimization Paths (Round 11 & 14) ---
    use_jit_state_phase: ClassVar[bool] = False # Set to True in subclasses that implement _compute_new_state_jit
    use_jit_edge_phase: ClassVar[bool] = False # Set to True in subclasses that implement _compute_new_edges_jit
    # ---

    def __init__(self, metadata: 'RuleMetadata'):
        super().__init__()
        # --- Ensure base metadata is populated ---
        if not Rule.PARAMETER_METADATA:
             Rule._populate_base_metadata()
        # ---
        self.metadata = metadata
        self.name = metadata.name
        self.metric_cache: Dict[Any, Any] = {}
        self.cache_generation = 0
        self._params: Dict[str, Any] = {}
        self.dependency_depth: int = 1
        self.disable_chunking: bool = False
        self.requires_post_edge_state_update: bool = False
        self.needs_neighbor_degrees: bool = False
        self.needs_neighbor_active_counts: bool = False
        self.skip_standard_tiebreakers: bool = False
        self.sets_states_to_degree: bool = False # TODO: Check if we implemented this or still need it or will need it

        self.perf_stats: Dict[str, Any] = {
            'compute_times': [], 'cache_hits': 0, 'cache_misses': 0
        }

        # Initialize instance _params with defaults from class attributes
        for param_name, param_info in Rule.PARAMETER_METADATA.items():
            if 'default' in param_info and param_name not in self._params:
                 self._params[param_name] = param_info['default']
        subclass_param_metadata = getattr(self, 'PARAMETER_METADATA', {})
        if subclass_param_metadata:
            for param_name, param_info in subclass_param_metadata.items():
                 if 'default' in param_info and param_name not in self._params:
                     self._params[param_name] = param_info['default']

        self._initialize_rule_tables()
        
    @staticmethod
    def _populate_base_metadata():
        from .initial_conditions import InitialConditionManager # Keep local import
        """Populates the base Rule.PARAMETER_METADATA class variable with enhanced descriptions."""
        if Rule.PARAMETER_METADATA: # Only populate if empty
            return

        logger.debug("Rule._populate_base_metadata: Populating base parameter metadata...")

        # Define standard colormap choices
        _standard_colormaps = sorted([
            'viridis', 'plasma', 'inferno', 'magma', 'cividis', # Perceptually Uniform
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', # Sequential
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', # Diverging
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            'twilight', 'twilight_shifted', 'hsv',
            'Pastel1', 'Pastel2', 'Paired', 'Accent', # Qualitative
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar', 'turbo'
        ])

        # PARAMETER_METADATA Setup with enhanced descriptions
        default_params_meta = {
            # Rule Tables
            'state_rule_table': {"type": dict, "default": {}, "parameter_group": "Rule Tables", "description": "(Optional) Rule table for state transitions."},
            'edge_rule_table': {"type": dict, "default": {}, "parameter_group": "Rule Tables", "description": "(Optional) Rule table for edge updates."},
            # Initialization
            'initial_continuous_node_state': {'type': float, 'default': 0.1, "parameter_group": "Initialization", 'min': -1.0, 'max': 1.0, 'description': 'Base state for newly active continuous nodes.'},
            'initial_continuous_edge_state': {'type': float, 'default': 0.1, "parameter_group": "Initialization", 'min': -1.0, 'max': 1.0, 'description': 'Base state for newly formed continuous edges.'},
            'edge_initialization': {'type': str, 'default': 'RANDOM', "parameter_group": "Initialization", 'allowed_values': ['RANDOM', 'FULL', 'DISTANCE', 'NEAREST', 'SIMILARITY', 'NONE'], 'description': 'Method for initializing edges'},
            'initial_conditions': {'type': str, 'default': 'Random', "parameter_group": "Initialization", 'allowed_values': InitialConditionManager.get_instance().get_all_names(), 'description': 'Initial grid state pattern'}, # Use manager
            'initial_density': {"type": float, 'default': GlobalSettings.Simulation.INITIAL_NODE_DENSITY, "description": "Initial density of active nodes (state=1) for 'Random' condition.", "min": 0.0, "max": 1.0, "parameter_group": "Initialization"},
            'connect_probability': {"type": float, "description": "Probability (0-1) for RANDOM edge initialization.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
            'min_edge_weight': {'type': float, 'default': 0.0, "parameter_group": "Initialization", 'min': -1.0, 'max': 1.0, 'description': 'Minimum initial random edge weight (-1 to 1).'}, # Adjusted range
            'max_edge_weight': {'type': float, 'default': 1.0, "parameter_group": "Initialization", 'min': -1.0, 'max': 1.0, 'description': 'Maximum initial random edge weight (-1 to 1).'}, # Adjusted range
            # Core
            'tiebreaker_type': {"type": str, "default": "RANDOM", "parameter_group": "Tiebreaker", "allowed_values": [e.name for e in TieBreaker], "description": "Method to resolve ties."},
            'grid_boundary': {'type': str, 'default': 'bounded', "parameter_group": "Core", 'allowed_values': ['bounded', 'wrap'], 'description': 'Grid boundary behavior (bounded or wrap)'},
            'neighborhood_type': {'type': str, 'default': "MOORE", "parameter_group": "Core", 'allowed_values': [n.name for n in NeighborhoodType], 'description': 'Neighborhood definition.'},
            'dimension_type': {'type': str, 'default': "TWO_D", "parameter_group": "Core", 'allowed_values': [d.name for d in Dimension], 'description': 'Grid dimension.'},
            # History
            'node_history_depth': {'type': int, 'default': 10, "parameter_group": "History", 'min': 0, 'max': 100, 'description': 'Number of previous states to store for each node'},
            # Visualization - Nodes
            'use_state_coloring': {"type": bool, "description": "Color nodes based on their stored state value.", "default": False, "parameter_group": "Visualization: Nodes"},
            'color_nodes_by_degree': {"type": bool, "description": "If Use State Coloring is True, color nodes based on connection count (degree) in the current step.", "default": False, "parameter_group": "Visualization: Nodes"},
            'color_nodes_by_active_neighbors': {"type": bool, "description": "If Use State Coloring is True, color nodes based on active neighbor count in the previous step.", "default": False, "parameter_group": "Visualization: Nodes"},
            'node_colormap': {"type": str, "description": "Colormap for node coloring.", "default": GlobalSettings.Visualization.DEFAULT_NODE_COLORMAP, "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
            'node_color_norm_vmin': {"type": float, "description": "Min value for node color normalization. Valid range depends on rule's node_state_type.", "default": 0.0, "parameter_group": "Visualization: Nodes"},
            'node_color_norm_vmax': {"type": float, "description": "Max value for node color normalization. Valid range depends on rule's node_state_type.", "default": 8.0, "parameter_group": "Visualization: Nodes"},
            # Visualization - Edges
            'use_state_coloring_edges': {"type": bool, "description": "Color edges based on their state value.", "default": True, "parameter_group": "Visualization: Edges"},
            'color_edges_by_neighbor_degree': {"type": bool, "description": "If Use State Coloring is True, edge state/color = avg degree of endpoints (prev step).", "default": False, "parameter_group": "Visualization: Edges"},
            'color_edges_by_neighbor_active_neighbors': {"type": bool, "description": "If Use State Coloring is True, edge state/color = avg active neighbor count of endpoints (prev step).", "default": False, "parameter_group": "Visualization: Edges"},
            'edge_colormap': {"type": str, "description": "Colormap for edge coloring.", "default": GlobalSettings.Visualization.DEFAULT_EDGE_COLORMAP, "parameter_group": "Visualization: Edges", "allowed_values": ["(None)"] + _standard_colormaps},
            'edge_color_norm_vmin': {"type": float, "description": "Min value for edge color normalization. Valid range depends on rule's edge_state_type.", "default": 0.0, "parameter_group": "Visualization: Edges"},
            'edge_color_norm_vmax': {"type": float, "description": "Max value for edge color normalization. Valid range depends on rule's edge_state_type.", "default": 1.0, "parameter_group": "Visualization: Edges"},

            # --- ADDED: Visualization Overrides Toggle ---
            'use_rule_specific_colors': {
                "type": bool, "default": False,
                "description": "Check this to use the specific color/colormap settings defined below for this rule, overriding the global color theme.",
                "parameter_group": "Visualization Overrides", "editor_sort_key": 0 # Ensure it appears first
            },
            # --- (Keep existing override color parameters below this) ---
            'rule_background_color': {
                "type": str, "default": None, # Default to None (use global theme)
                "description": "Override background color (e.g., #1a1a1a). Leave blank to use global theme.",
                "parameter_group": "Visualization Overrides", "editor_sort_key": 10
            },
            'rule_node_base_color': {
                "type": str, "default": None,
                "description": "Override base/inactive node color (e.g., #303030). Leave blank to use global theme.",
                "parameter_group": "Visualization Overrides", "editor_sort_key": 20
            },
            'rule_node_color': {
                "type": str, "default": None,
                "description": "Override active node color (or outline for non-state coloring) (e.g., #FFA500). Leave blank to use global theme.",
                "parameter_group": "Visualization Overrides", "editor_sort_key": 30
            },
            'rule_new_node_color': {
                "type": str, "default": None,
                "description": "Override highlight color for newly active node outlines (e.g., #FF00FF). Leave blank to use global theme.",
                "parameter_group": "Visualization Overrides", "editor_sort_key": 40
            },
            'rule_default_edge_color': {
                "type": str, "default": None,
                "description": "Override default edge color (or old edge outline) (e.g., #00FFFF). Leave blank to use global theme.",
                "parameter_group": "Visualization Overrides", "editor_sort_key": 50
            },
            'rule_new_edge_color': {
                "type": str, "default": None,
                "description": "Override highlight color for new edges (e.g., #FFFF00). Leave blank to use global theme.",
                "parameter_group": "Visualization Overrides", "editor_sort_key": 60
            },
        }
        Rule.PARAMETER_METADATA.update(default_params_meta) # Update the class attribute
        logger.debug(f"Rule._populate_base_metadata: Populated {len(Rule.PARAMETER_METADATA)} base parameters.")

    # --- Optional method for post-edge state calculation ---
    @timer_decorator
    def _compute_final_state(self,
                             node_idx: int,
                             current_proxy_state: float, # Eligibility/state from current step's computation
                             final_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]], # Edges calculated for *current* step
                             dimensions: Tuple[int,...],
                             # --- Added Previous State Data ---
                             previous_node_states: npt.NDArray[np.float64],
                             previous_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
                             previous_edge_states: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float],
                             previous_node_degrees: Optional[npt.NDArray[np.int32]], # Array of degrees for all nodes
                             previous_active_neighbors: Optional[npt.NDArray[np.int32]], # Array of active neighbor counts for all nodes
                             # --- ADDED: Parameter for eligibility proxies ---
                             eligibility_proxies: Optional[np.ndarray] = None # Full array from Phase 1
                             ) -> float:
        """
        (Optional) Compute the final node state after edge updates are complete.
        Rules needing specific final state calculation (like degree-based rules)
        MUST override this. The base implementation returns the proxy state.
        (Round 17: Added eligibility_proxies parameter to base signature)
        """
        # Default implementation returns the proxy state if not overridden
        return current_proxy_state

    def _initialize_rule_tables(self):
        self._cached_state_rule_table = self.get_param('state_rule_table', {})
        self._cached_edge_rule_table = self.get_param('edge_rule_table', {})

    @property
    def category(self) -> str: return getattr(self.metadata, 'category', 'Unknown')


    # --- MODIFIED: Make initialize_grid_state abstract ---
    @abstractmethod
    def initialize_grid_state(self, grid: 'Grid'):
        """
        (Abstract Method) Defines how a rule *would* initialize the grid.
        The actual implementation calling InitialConditionManager should
        be handled externally (e.g., in SimulationGUI or SimulationController).
        Subclasses might override this if they have truly unique init logic
        that doesn't fit the standard manager patterns, but it's generally
        better to add new conditions to the InitialConditionManager.
        """
        # --- Implementation removed ---
        raise NotImplementedError

    @timer_decorator
    def compute_updates(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> Tuple[float, Dict[Tuple[int, int], float]]:
        """
        Computes node state and edge updates for a given neighborhood.
        (Round 25: Added detailed_logging_enabled flag)
        """
        new_state: float = 0.0
        new_edges: Dict[Tuple[int, int], float] = {}
        try:
            # Pass the flag down to internal methods
            new_state = self._compute_new_state(neighborhood, detailed_logging_enabled)
            new_edges = self._compute_new_edges(neighborhood, detailed_logging_enabled)
            return new_state, new_edges
        except Exception as e:
            logger.error(f"Error computing updates for node {neighborhood.node_index}: {e}\n{traceback.format_exc()}")
            # Return original state and empty edges on error
            return neighborhood.node_state, {}

    @timer_decorator  
    @abstractmethod
    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """
        (Abstract) Compute the new state for the central node based on the neighborhood.
        (Round 25: Added detailed_logging_enabled flag)
        """
        raise NotImplementedError

    @timer_decorator
    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> Dict[Tuple[int, int], float]:
        """
        Compute edge state using GoL-like rules based ONLY on the neighbor's degree
        from the previous step. Mutual eligibility check is REMOVED.
        (Round 25: Added detailed_logging_enabled flag)
        (Round 16: Added logging for neighbor degree used)
        """
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        # detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING # Use passed arg
        node_idx = neighborhood.node_index

        # Fetch parameters
        edge_birth_degrees = self.get_param('edge_birth_scores', [0, 3], neighborhood=neighborhood) # Using score name, but refers to neighbor degree
        edge_survival_degrees = self.get_param('edge_survival_scores', [2, 3], neighborhood=neighborhood) # Using score name, but refers to neighbor degree
        random_flip_prob = self.get_param('random_edge_flip_prob', 0.0, neighborhood=neighborhood)

        self_prev_degree = int(neighborhood.node_state) # Degree from prev step

        if detailed_logging_enabled:
            log_func = logger.info # Use INFO level for this diagnostic
            log_func(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]
            log_func(f"    Self Prev Degree: {self_prev_degree}") # type: ignore [attr-defined]
            log_func(f"    Neighbor Degrees Dict Received: {neighborhood.neighbor_degrees}") # type: ignore [attr-defined]
        else:
            log_func = lambda *args, **kwargs: None # No-op if detailed logging off

        for neighbor_idx, neighbor_prev_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            # Get neighbor's degree from previous step
            neighbor_prev_degree = 0
            if neighborhood.neighbor_degrees is not None:
                 neighbor_prev_degree = neighborhood.neighbor_degrees.get(neighbor_idx, 0)
                 log_func(f"  Neighbor {neighbor_idx}: Using Prev Degree = {neighbor_prev_degree} from neighbor_degrees dict.") # type: ignore [attr-defined]
            else:
                 logger.warning(f"Node {node_idx}: neighbor_degrees missing for neighbor {neighbor_idx}")
                 neighbor_prev_degree = int(neighbor_prev_state) if neighbor_prev_state > 0 else 0 # Fallback
                 log_func(f"  Neighbor {neighbor_idx}: Using Fallback Prev Degree = {neighbor_prev_degree} from neighbor_prev_state.") # type: ignore [attr-defined]

            edge = (node_idx, neighbor_idx) # Canonical order
            has_current_edge = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0) > 1e-6

            new_edge_state = 0.0 # Default death/no birth
            decision_reason = "Default (No Edge)"

            # --- Logic Based ONLY on neighbor_prev_degree ---
            if not has_current_edge: # Edge Birth Check
                if neighbor_prev_degree in edge_birth_degrees:
                    new_edge_state = 1.0
                    decision_reason = f"Birth (NeighPrevDeg={neighbor_prev_degree} in {edge_birth_degrees})"
                else:
                     decision_reason = f"No Birth (NeighPrevDeg={neighbor_prev_degree} not in {edge_birth_degrees})"
            else: # Edge Survival Check
                if neighbor_prev_degree in edge_survival_degrees:
                    new_edge_state = 1.0
                    decision_reason = f"Survival (NeighPrevDeg={neighbor_prev_degree} in {edge_survival_degrees})"
                else:
                     decision_reason = f"Death (NeighPrevDeg={neighbor_prev_degree} not in {edge_survival_degrees})"
            # ---

            # Apply random flip
            random_flip_applied = False
            if np.random.random() < random_flip_prob:
                new_edge_state = 1.0 - new_edge_state
                random_flip_applied = True
                decision_reason += " + Random Flip"

            log_func(f"    Edge {edge}: Decision: {decision_reason}. Final State={new_edge_state:.0f}") # type: ignore [attr-defined]

            if new_edge_state > 0.5:
                new_edges[edge] = 1.0 # Binary edges

        return new_edges
    

    @staticmethod
    # @njit # Keep disabled
    def _get_neighbor_pattern(neighborhood: NeighborhoodData) -> str:
        pattern = ""; max_len = len(neighborhood.neighbor_states)
        for i in range(max_len): pattern += "1" if neighborhood.neighbor_states[i] > 1e-6 else "0"
        return pattern

    @staticmethod
    # @njit # Keep disabled
    def _get_connection_pattern(neighborhood: NeighborhoodData) -> str:
        pattern = ""; node_idx = neighborhood.node_index; max_len = len(neighborhood.neighbor_indices)
        for i in range(max_len):
            neighbor_idx = neighborhood.neighbor_indices[i]
            if neighbor_idx != -1: pattern += "1" if neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0) > 1e-6 else "0"
            else: pattern += "0"
        return pattern

    @staticmethod
    # @njit # Keep disabled for now
    def _get_direction_from_delta(delta: np.ndarray) -> str:
        """
        Determines the canonical direction string ('N', 'NE', 'U', 'Diag', etc.)
        based on the coordinate difference (delta = neighbor_coord - node_coord).
        """
        dims = len(delta) # Determine dimensions from the delta vector

        if dims == 2:
            dy, dx = delta[0], delta[1] # delta_row, delta_col
            # Handle zero delta case (shouldn't happen for neighbors, but safety)
            if dx == 0 and dy == 0: return "Center"

            angle_rad = math.atan2(dy, dx) # atan2(y, x)
            angle_deg = math.degrees(angle_rad)

            # Map angle to cardinal/intercardinal directions (Moore neighborhood assumption)
            if -22.5 <= angle_deg < 22.5: return 'E'
            elif 22.5 <= angle_deg < 67.5: return 'NE'
            elif 67.5 <= angle_deg < 112.5: return 'N'
            elif 112.5 <= angle_deg < 157.5: return 'NW'
            elif angle_deg >= 157.5 or angle_deg < -157.5: return 'W'
            elif -157.5 <= angle_deg < -112.5: return 'SW'
            elif -112.5 <= angle_deg < -67.5: return 'S'
            elif -67.5 <= angle_deg < -22.5: return 'SE'
            else: return "Unknown_2D" # Should not happen

        elif dims == 3:
            dy, dx, dz = delta[0], delta[1], delta[2] # delta_row, delta_col, delta_depth
            # Handle zero delta case
            if dx == 0 and dy == 0 and dz == 0: return "Center"

            abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)

            # Check for cardinal directions (including Up/Down) first
            if dx == 0 and dy == 0 and dz != 0: return 'U' if dz > 0 else 'D'
            if dx == 0 and dz == 0 and dy != 0: return 'N' if dy > 0 else 'S'
            if dy == 0 and dz == 0 and dx != 0: return 'E' if dx > 0 else 'W'

            # Refined 3D Diagonal Logic (Still an approximation)
            # Determine primary XY direction
            if abs_dx >= abs_dy: primary_xy = 'E' if dx > 0 else 'W'
            else: primary_xy = 'N' if dy > 0 else 'S'
            # Determine primary Z direction
            primary_z = ""
            if abs_dz > max(abs_dx, abs_dy) * 0.5: # If z movement is significant
                 primary_z = 'U' if dz > 0 else 'D'

            # Combine
            if primary_z: return primary_z + primary_xy # e.g., UNE, DS
            else: return primary_xy # Purely planar diagonal relative to center

        else:
            return "Unsupported_Dim"

    def get_metric(self, metric_name: str, neighborhood: 'NeighborhoodData') -> Union[float, npt.NDArray[np.float64]]:
        cache_key = (neighborhood.node_index, metric_name)
        if cache_key in self.metric_cache: self.perf_stats['cache_hits'] += 1; return self.metric_cache[cache_key]
        self.perf_stats['cache_misses'] += 1; value: Union[float, npt.NDArray[np.float64]]
        if metric_name in neighborhood.neighborhood_metrics: value = neighborhood.neighborhood_metrics[metric_name]
        elif metric_name in neighborhood.neighbor_metrics: value = neighborhood.neighbor_metrics[metric_name]
        else:
            if hasattr(BaseMetrics, metric_name):
                method = getattr(BaseMetrics, metric_name); sig = inspect.signature(method)
                if 'grid' not in sig.parameters:
                    try: value = method(neighborhood.neighbor_states, neighborhood.neighbor_indices, neighborhood)
                    except TypeError as te: raise ValueError(f"Metric '{metric_name}' calc error: {te}") from te
                else: raise ValueError(f"Metric '{metric_name}' requires grid access.")
            else: raise ValueError(f"Unknown metric: {metric_name}")
        self.metric_cache[cache_key] = value; return value

    def invalidate_cache(self): self.metric_cache.clear(); self.cache_generation += 1

    def get_performance_stats(self) -> Dict[str, Any]:
        avg_time = 0.0
        if self.perf_stats['compute_times']:
             valid_times = [t for t in self.perf_stats['compute_times'][-100:] if isinstance(t, (int, float))]
             if valid_times: avg_time = float(np.mean(valid_times))
        return {'name': self.name, 'avg_compute_time': avg_time, 'cache_hits': self.perf_stats['cache_hits'], 'cache_misses': self.perf_stats['cache_misses'], 'cache_size': len(self.metric_cache), 'cache_generation': self.cache_generation}

    @property
    def params(self) -> Dict[str, Any]: return self._params

    @params.setter
    def params(self, new_params: Dict[str, Any]):
        """Setter for rule parameters with validation and transition check.
           (Round 1: More robust list/tuple parsing from string)"""
        logger = logging.getLogger(__name__)
        log_prefix = f"Rule({self.name}).params.setter (R1 List/Tuple Fix): " # Updated round
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING

        is_applying_preset = getattr(self, '_applying_preset_params', False)
        validated_params = self._params.copy() # Start with current params

        for name, value in new_params.items():
            param_info = self.PARAMETER_METADATA.get(name, {})
            param_type = param_info.get('type')
            converted_value = value # Start with the input value
            original_value_repr = repr(value) # For logging

            if param_type:
                try:
                    if param_type == bool: converted_value = str(value).lower() in ('true', '1', 'yes', 'on')
                    elif param_type == int: converted_value = int(float(value))
                    elif param_type == float: converted_value = float(value)
                    # --- MODIFIED List/Tuple Parsing ---
                    elif param_type == list and isinstance(value, str):
                        if detailed_logging_enabled: logger.detail(f"{log_prefix}Attempting LIST conversion for '{name}'. Input: '{value}'") # type: ignore[attr-defined]
                        try:
                            # Attempt literal_eval first
                            parsed = ast.literal_eval(value)
                            if isinstance(parsed, list):
                                converted_value = parsed
                                if detailed_logging_enabled: logger.detail(f"  literal_eval LIST success: {converted_value}") # type: ignore[attr-defined]
                            else:
                                logger.warning(f"{log_prefix}literal_eval for LIST '{name}' did not return list (got {type(parsed).__name__}). Input: '{value}'. Trying fallback.")
                                # Fallback: Try splitting by comma if eval failed or gave wrong type
                                converted_value = [item.strip() for item in value.strip('[]').split(',') if item.strip()]
                                if detailed_logging_enabled: logger.detail(f"  Comma split LIST fallback result: {converted_value}") # type: ignore[attr-defined]
                        except (ValueError, SyntaxError, TypeError) as e_eval:
                            logger.warning(f"{log_prefix}literal_eval FAILED for LIST '{name}': {e_eval}. Input: '{value}'. Trying fallback.")
                            # Fallback: Try splitting by comma
                            try:
                                converted_value = [item.strip() for item in value.strip('[]').split(',') if item.strip()]
                                if detailed_logging_enabled: logger.detail(f"  Comma split LIST fallback result: {converted_value}") # type: ignore[attr-defined]
                            except Exception as e_split:
                                logger.error(f"{log_prefix}Fallback comma split FAILED for LIST '{name}': {e_split}. Using empty list.")
                                converted_value = [] # Final fallback: empty list
                        # Ensure final result is a list
                        if not isinstance(converted_value, list): converted_value = []

                    elif param_type == tuple and isinstance(value, str):
                        if detailed_logging_enabled: logger.detail(f"{log_prefix}Attempting TUPLE conversion for '{name}'. Input: '{value}'") # type: ignore[attr-defined]
                        try:
                            parsed = ast.literal_eval(value)
                            if isinstance(parsed, tuple):
                                converted_value = parsed
                                if detailed_logging_enabled: logger.detail(f"  literal_eval TUPLE success: {converted_value}") # type: ignore[attr-defined]
                            else:
                                logger.warning(f"{log_prefix}literal_eval for TUPLE '{name}' did not return tuple (got {type(parsed).__name__}). Input: '{value}'. Using empty tuple.")
                                converted_value = () # Fallback
                        except (ValueError, SyntaxError, TypeError) as e_eval:
                            logger.warning(f"{log_prefix}literal_eval FAILED for TUPLE '{name}': {e_eval}. Input: '{value}'. Using empty tuple.")
                            converted_value = () # Fallback
                        # Ensure final result is a tuple
                        if not isinstance(converted_value, tuple): converted_value = ()
                    # --- END MODIFIED ---
                    elif param_type == dict and isinstance(value, str):
                         try: converted_value = ast.literal_eval(value)
                         except: raise ValueError("Invalid dict string")
                         if not isinstance(converted_value, dict): raise ValueError("Not dict")
                except (ValueError, TypeError, SyntaxError) as e:
                    logger.warning(f"{log_prefix}Conversion error for '{name}' (Value: {original_value_repr}): {e}")
                    # Keep original value on conversion error before validation? Or use default? Let's keep original for now.
                    converted_value = value # Keep original value if conversion fails

            # --- Validation ---
            if detailed_logging_enabled: logger.detail(f"{log_prefix}Validating param '{name}'. Value before validation: {repr(converted_value)} (Type: {type(converted_value).__name__})") # type: ignore[attr-defined]
            if self._validate_parameter(name, converted_value):
                validated_params[name] = converted_value
            else:
                logger.warning(f"{log_prefix}Invalid param '{name}' after conversion/validation: {repr(converted_value)}")
                existing_val = self._params.get(name, param_info.get('default')) # Get current or default
                validated_params[name] = existing_val # Revert to existing or default on validation failure

        # --- Apply validated parameters ---
        self._params = validated_params
        if 'state_rule_table' in self._params or 'edge_rule_table' in self._params:
            self._initialize_rule_tables()

        # --- Log final state of problematic params ---
        if detailed_logging_enabled:
            for pname in self._params:
                if pname.endswith("_values") or pname.endswith("_range"): # Log both ranges and values
                    pval = self._params[pname]
                    logger.detail(f"{log_prefix}FINAL state in self._params for '{pname}': {repr(pval)} (Type: {type(pval).__name__})") # type: ignore[attr-defined]
        # ---

    def update_parameter(self, name: str, value: Any) -> bool:
        """Updates a single parameter, performing validation and type conversion.
           (Round 1: More robust list/tuple parsing from string)"""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        log_prefix = f"Rule({self.name}).update_parameter (R1 List/Tuple Fix): " # Updated round
        logger.debug(f"{log_prefix}Attempting update for '{name}' with value '{repr(value)}' (type: {type(value)})")

        is_applying_preset = getattr(self, '_applying_preset_params', False)
        # --- Combine logic, avoid repetition ---
        try:
            param_info = self.PARAMETER_METADATA.get(name, {}); param_type = param_info.get('type'); converted_value = value
            original_value_repr = repr(value) # For logging

            # --- Type Conversion ---
            if param_type:
                if name in ('node_colormap', 'edge_colormap') and value is None: converted_value = None
                elif param_type == bool: converted_value = str(value).lower() in ('true', '1', 'yes', 'on')
                elif param_type == int: converted_value = int(float(value))
                elif param_type == float: converted_value = float(value)
                # --- MODIFIED List/Tuple Parsing ---
                elif param_type == list and isinstance(value, str):
                    if detailed_logging_enabled: logger.detail(f"{log_prefix}Attempting LIST conversion for '{name}'. Input: '{value}'") # type: ignore[attr-defined]
                    try:
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, list):
                            converted_value = parsed
                            if detailed_logging_enabled: logger.detail(f"  literal_eval LIST success: {converted_value}") # type: ignore[attr-defined]
                        else:
                            logger.warning(f"{log_prefix}literal_eval for LIST '{name}' did not return list (got {type(parsed).__name__}). Input: '{value}'. Trying fallback.")
                            converted_value = [item.strip() for item in value.strip('[]').split(',') if item.strip()]
                            if detailed_logging_enabled: logger.detail(f"  Comma split LIST fallback result: {converted_value}") # type: ignore[attr-defined]
                    except (ValueError, SyntaxError, TypeError) as e_eval:
                        logger.warning(f"{log_prefix}literal_eval FAILED for LIST '{name}': {e_eval}. Input: '{value}'. Trying fallback.")
                        try:
                            converted_value = [item.strip() for item in value.strip('[]').split(',') if item.strip()]
                            if detailed_logging_enabled: logger.detail(f"  Comma split LIST fallback result: {converted_value}") # type: ignore[attr-defined]
                        except Exception as e_split:
                            logger.error(f"{log_prefix}Fallback comma split FAILED for LIST '{name}': {e_split}. Using empty list.")
                            converted_value = []
                    if not isinstance(converted_value, list): converted_value = []

                elif param_type == tuple and isinstance(value, str):
                    if detailed_logging_enabled: logger.detail(f"{log_prefix}Attempting TUPLE conversion for '{name}'. Input: '{value}'") # type: ignore[attr-defined]
                    try:
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, tuple):
                            converted_value = parsed
                            if detailed_logging_enabled: logger.detail(f"  literal_eval TUPLE success: {converted_value}") # type: ignore[attr-defined]
                        else:
                            logger.warning(f"{log_prefix}literal_eval for TUPLE '{name}' did not return tuple (got {type(parsed).__name__}). Input: '{value}'. Using empty tuple.")
                            converted_value = ()
                    except (ValueError, SyntaxError, TypeError) as e_eval:
                        logger.warning(f"{log_prefix}literal_eval FAILED for TUPLE '{name}': {e_eval}. Input: '{value}'. Using empty tuple.")
                        converted_value = ()
                    if not isinstance(converted_value, tuple): converted_value = ()
                # --- END MODIFIED ---
                elif param_type == dict and isinstance(value, str):
                     try: converted_value = ast.literal_eval(value)
                     except: raise ValueError("Invalid dict string")
                     if not isinstance(converted_value, dict): raise ValueError("Not dict")
                else:
                     if converted_value is not None and not isinstance(converted_value, param_type):
                          try: converted_value = param_type(value)
                          except (ValueError, TypeError): raise ValueError(f"Cannot convert '{value}' to type {param_type}")
            # --- End Type Conversion ---

            # --- Validation ---
            if detailed_logging_enabled: logger.detail(f"{log_prefix}Validating param '{name}'. Value before validation: {repr(converted_value)} (Type: {type(converted_value).__name__})") # type: ignore[attr-defined]
            if self._validate_parameter(name, converted_value):
                old_value = self._params.get(name)
                # --- Check if value actually changed before logging success ---
                value_changed = False
                if isinstance(converted_value, float) and isinstance(old_value, (int, float)): value_changed = not np.isclose(converted_value, float(old_value))
                elif isinstance(converted_value, int) and isinstance(old_value, (int, float)): value_changed = not np.isclose(float(converted_value), float(old_value))
                elif isinstance(converted_value, bool) and isinstance(old_value, (int, bool)): value_changed = bool(converted_value) != bool(old_value)
                else: value_changed = converted_value != old_value

                self._params[name] = converted_value
                if value_changed:
                    logger.info(f"{log_prefix}Successfully updated '{name}' from '{repr(old_value)}' to '{repr(converted_value)}' in self._params.")
                else:
                    logger.debug(f"{log_prefix}Value for '{name}' validated but was not different from existing value '{repr(old_value)}'.")
                if name in ['state_rule_table', 'edge_rule_table']: self._initialize_rule_tables()
                return True
            else:
                logger.warning(f"{log_prefix}Invalid value '{repr(converted_value)}' for parameter '{name}'. Update rejected.")
                return False

        except (ValueError, TypeError, SyntaxError) as e:
            logger.warning(f"{log_prefix}Conversion/Validation error for '{name}' ('{original_value_repr}'): {e}")
            return False
        except Exception as e:
            logger.error(f"{log_prefix}Unexpected error updating param '{name}': {e}")
            logger.error(traceback.format_exc())
            return False
        
    def _validate_parameter(self, name: str, value: Any) -> bool:
        """
        Validate parameter value against metadata constraints.
        (Round 18: Improve list/tuple validation for JSON loading)
        """
        try:
            if not hasattr(self, 'PARAMETER_METADATA') or name not in self.PARAMETER_METADATA:
                # logger.debug(f"Validation skipped for '{name}': No metadata found.") # Reduce noise
                return True # No metadata, assume valid

            metadata = self.PARAMETER_METADATA[name]
            param_type = metadata.get('type')
            converted_value = value # Start with the original value

            if param_type:
                correct_type = False
                # Handle list/tuple conversion earlier
                if param_type == list:
                    parsed_list = None
                    if isinstance(value, str):
                        try: parsed_list = ast.literal_eval(value)
                        except (ValueError, SyntaxError): pass # Handle potential eval errors later
                    elif isinstance(value, list):
                        parsed_list = value # Already a list

                    if isinstance(parsed_list, list):
                        element_type = metadata.get('element_type')
                        if element_type == tuple:
                            # Attempt conversion to list of tuples FIRST
                            try:
                                converted_value = [tuple(item) if isinstance(item, list) else item for item in parsed_list]
                                # Check if conversion was successful AND final type is list of tuples
                                if isinstance(converted_value, list) and all(isinstance(item, tuple) for item in converted_value):
                                    correct_type = True
                                else:
                                     logger.warning(f"Validation: Failed inner list->tuple conversion for '{name}'. Value: {parsed_list}")
                            except TypeError:
                                logger.warning(f"Validation: TypeError during inner list->tuple conversion for '{name}'. Value: {parsed_list}")
                        elif element_type:
                            # Check if all elements match the specified type
                            if all(isinstance(item, element_type) for item in parsed_list):
                                converted_value = parsed_list # Keep the list
                                correct_type = True
                        else: # No element type specified, just check if it's a list
                            converted_value = parsed_list
                            correct_type = True
                    # If parsing/conversion failed or original wasn't list/str, correct_type remains False
                elif param_type == tuple:
                     parsed_tuple = None
                     if isinstance(value, str):
                         try: parsed_tuple = ast.literal_eval(value)
                         except (ValueError, SyntaxError): pass
                     elif isinstance(value, tuple):
                         parsed_tuple = value
                     if isinstance(parsed_tuple, tuple):
                         converted_value = parsed_tuple
                         correct_type = True
                elif name in ('node_colormap', 'edge_colormap') and value is None:
                    converted_value = None
                    correct_type = True # None is valid for colormaps
                elif param_type == float and isinstance(value, (int, float)):
                    converted_value = float(value) # Ensure float
                    correct_type = True
                elif param_type == int and isinstance(value, int):
                    converted_value = value
                    correct_type = True
                elif param_type == bool and isinstance(value, bool):
                    converted_value = value
                    correct_type = True
                elif param_type == str and isinstance(value, str):
                    converted_value = value
                    correct_type = True
                elif param_type == dict and isinstance(value, dict):
                     converted_value = value
                     correct_type = True
                # Add other explicit type checks if needed

                if not correct_type:
                    logger.warning(f"Validation failed for '{name}': Type mismatch. Expected {param_type}, got {type(value)}.")
                    return False

            # Value Constraint Checks (Min/Max/Allowed)
            if 'min' in metadata and isinstance(converted_value, (int, float)) and converted_value < metadata['min']:
                logger.warning(f"Validation failed for '{name}': Value {converted_value} < Min {metadata['min']}.")
                return False
            if 'max' in metadata and isinstance(converted_value, (int, float)) and converted_value > metadata['max']:
                logger.warning(f"Validation failed for '{name}': Value {converted_value} > Max {metadata['max']}.")
                return False
            if 'allowed_values' in metadata:
                allowed_values = metadata['allowed_values']
                # Allow None for colormaps even if "(None)" isn't explicitly in allowed_values in metadata
                is_colormap = name in ('node_colormap', 'edge_colormap')
                if is_colormap and converted_value is None:
                    pass # Allow None for colormaps
                elif converted_value not in allowed_values:
                    logger.warning(f"Validation failed for '{name}': Value '{converted_value}' not in allowed values {allowed_values}.")
                    return False

            # logger.debug(f"Validation successful for '{name}' with value '{converted_value}'.") # Reduce noise
            return True
        except Exception as e:
            logger.error(f"Parameter validation error for '{name}' (Value: {value}): {e}")
            logger.error(traceback.format_exc())
            return False

    def get_param(self, name: str, default: Any = None, neighborhood: Optional[NeighborhoodData] = None) -> Any:
        """
        Gets a parameter value, checking neighborhood context first, then internal _params,
        then PARAMETER_METADATA defaults. Adds detailed logging for retrieved values.
        (Round 32: Use logger.error for *_values_* param logging to force visibility)
        """
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        log_prefix = f"Rule({self.name}).get_param(name='{name}' R32): " # Updated round
        value = None
        source = "Not Found"

        # [ Value retrieval logic remains the same ]
        # Check neighborhood context first
        if neighborhood and neighborhood.rule_params:
            value = neighborhood.rule_params.get(name)
            if value is not None:
                source = "NeighborhoodData.rule_params"

        # Check internal _params dictionary next if not found in neighborhood
        if value is None:
            value = self._params.get(name)
            if value is not None:
                source = "self._params"

        # If not in _params, check PARAMETER_METADATA for a default
        if value is None:
            if not Rule.PARAMETER_METADATA:
                 Rule._populate_base_metadata()
            if hasattr(self, 'PARAMETER_METADATA') and name in self.PARAMETER_METADATA:
                metadata_default = self.PARAMETER_METADATA[name].get('default')
                if metadata_default is not None:
                    value = metadata_default
                    source = "PARAMETER_METADATA default"
                    self._params[name] = copy.deepcopy(value) # Store default

        # --- Determine Final Value and Log ---
        final_value = value if value is not None else default
        final_source = source if value is not None else "Default Argument"
        type_to_log = type(final_value).__name__

        # --- MODIFIED: Explicit Logging for *_values_* parameters using ERROR level ---
        if name.endswith("_values"):
            # Log with ERROR level to ensure visibility
            logger.error(f"{log_prefix}FORCE LOG: Retrieving *_values_* param '{name}'. Source='{final_source}'. Value BEFORE return: {repr(final_value)} (Type: {type_to_log})")
            if isinstance(final_value, list):
                 element_types = [type(item).__name__ for item in final_value[:10]] # Check first 10
                 logger.error(f"{log_prefix}FORCE LOG:   '{name}' is LIST. Element Types (First 10): {element_types}")
            elif isinstance(final_value, str):
                 logger.critical(f"{log_prefix}FORCE LOG:   '{name}' is STRING ('{final_value}'). THIS IS THE PROBLEM!") # Use CRITICAL for string
            else:
                 logger.error(f"{log_prefix}FORCE LOG:   '{name}' is unexpected type: {type_to_log}")
        # --- END MODIFIED ---
        elif detailed_logging_enabled: # Standard detailed logging for other params
            log_msg = f"{log_prefix}Source='{final_source}', Retrieved Value='{str(final_value)[:100]}', Type='{type_to_log}'"
            if isinstance(final_value, list):
                 element_types = [type(item).__name__ for item in final_value[:5]]
                 log_msg += f", List Elements (First 5 Types): {element_types}"
            logger.detail(log_msg) # type: ignore[attr-defined]

        return final_value
   
    def _get_max_neighbors_for_validation(self) -> int:
        try:
            neighborhood_type_str=self.get_param("neighborhood_type","MOORE"); dimension_type_str=self.get_param("dimension_type","TWO_D")
            neighborhood_type=NeighborhoodType[neighborhood_type_str]; dimension_type=Dimension[dimension_type_str]
            if neighborhood_type == NeighborhoodType.VON_NEUMANN: return 6 if dimension_type == Dimension.THREE_D else 4
            elif neighborhood_type == NeighborhoodType.MOORE: return 26 if dimension_type == Dimension.THREE_D else 8
            elif neighborhood_type == NeighborhoodType.HEX: return 6
            elif neighborhood_type == NeighborhoodType.HEX_PRISM: return 12
            else: return 8
        except: return 8

    def get_active_neighbors(self, neighborhood: NeighborhoodData) -> List[int]:
        active_neighbors: List[int] = []; activity_threshold = self.get_param('node_activity_threshold', 0.1) # Use threshold if defined
        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx >= 0 and neighbor_state > activity_threshold: active_neighbors.append(neighbor_idx)
        return active_neighbors

    def get_neighborhood_metric(self, node_idx: int, metric_name: str) -> float: raise NotImplementedError

    def get_state_table_entry(self, current_state, neighbor_pattern, connection_pattern):
        key = f"({int(current_state)}, {neighbor_pattern}, {connection_pattern})"
        return float(self._cached_state_rule_table.get(key, self._cached_state_rule_table.get('default', 0)))

    def get_edge_table_entry(self, self_state, neighbor_state, connection_pattern):
        key = f"({int(self_state)}, {int(neighbor_state)}, {connection_pattern})"
        return self._cached_edge_rule_table.get(key, self._cached_edge_rule_table.get('default', 'maintain'))

    def _get_parameters(self, *param_names: str) -> List[Any]: return [self.get_param(name) for name in param_names]

    @staticmethod
    # @njit # Keep disabled
    def _count_active_neighbors(neighbor_states: np.ndarray) -> int:
        count = 0; activity_threshold = 0.1 # Default threshold if not available via params
        # Cannot access self.get_param in static/njit method easily
        try:
            if isinstance(neighbor_states, np.ndarray): count = int(np.sum(np.greater(neighbor_states, activity_threshold)))
            else: count = sum(1 for state in neighbor_states if state > activity_threshold)
        except Exception as e: logger.error(f"Err count neighbors: {e}"); count = 0
        return count

    def validate_parameter_context(self, param_name: str, current_params: Dict[str, Any]) -> Optional[str]:
        """
        (Optional Override) Perform context-dependent validation for a parameter after basic checks.

        This method is called by the RuleEditorWindow after a parameter's basic
        type/range validation passes and its value has potentially changed in the UI's
        tracked state. Subclasses can override this to implement checks that
        depend on the values of *other* parameters (e.g., checking for overlaps
        between life and death ranges based on the current metric selection).

        Args:
            param_name: The name of the parameter that was just validated/changed.
            current_params: A dictionary representing the complete set of parameters
                            currently reflected in the editor's state (including the
                            potentially new value for 'param_name').

        Returns:
            A string containing a warning message if a contextual conflict or issue
            is detected, otherwise None. Returning a message will typically trigger
            a non-blocking warning dialog in the UI.
        """
        # Base implementation performs no contextual checks.
        return None

    def get_dynamic_parameter_metadata(self, param_base_name: str, metric_type: str, aggregation: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        (Optional Override) Dynamically generate metadata for permutation parameters.

        This method is called by the RuleEditorWindow when it needs metadata for a
        parameter that follows a permutation pattern (e.g., based on metric/aggregation)
        and is not found in the static PARAMETER_METADATA. Subclasses that use such
        parameters should override this method.

        Args:
            param_base_name: The base name of the parameter (e.g., "birth_eligibility_range").
            metric_type: The selected metric type (e.g., "DEGREE").
            aggregation: The selected aggregation type (e.g., "SUM", "AVG"), or None if not applicable.

        Returns:
            A dictionary containing the full metadata for the specific requested
            parameter permutation (including type, default, description with calculated
            ranges, parameter_group, editor_sort_key, etc.), or None if the rule
            does not support generating metadata for this specific combination.
        """
        # Base implementation does not generate dynamic metadata.
        return None

    @staticmethod
    def _get_expected_dynamic_param_names_from_selectors(params: Dict[str, Any]) -> Set[str]:
        """
        (Optional Override) Returns the set of expected dynamic parameter names
        based on the selector values currently present in the provided params dict.
        Base implementation returns an empty set.
        """
        return set()

class Shape(Protocol):
    """Protocol for shapes to be placed on the grid."""

    def get_relative_coordinates(self) -> List[Tuple[int, ...]]:
        """Returns a list of relative coordinates defining the shape."""
        ...

    def get_connectivity(self) -> str:
        """Returns the connectivity type ('full', 'perimeter', or 'none')."""
        ...

    def get_dimensions(self) -> int:
        """Returns the dimensionality of the shape (2 or 3)."""
        ...

    def is_filled(self) -> bool:
        """Returns True if the shape is filled, False otherwise."""
        ...
        
    def get_shape_type(self) -> ShapeType:
        """Returns the shape type"""
        ...

    def get_bounding_box(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Returns the bounding box of the shape as (min_coords, max_coords)."""
        ...


################################################
#                NEIGHBORHOOD DATA             #
################################################

@dataclass
class NeighborhoodData:
    """Manages neighborhood relationships for the grid (read-only access)."""
    node_index: int
    node_coords: Tuple[int, ...]
    node_state: float
    neighbor_indices: npt.NDArray[np.int64]
    neighbor_states: npt.NDArray[np.float64]
    edges: Set[Tuple[int, int]] # Edges within the immediate neighborhood (node to neighbor), using indices
    dimensions: Tuple[int, ...]
    neighborhood_type: NeighborhoodType
    grid_boundary: str
    neighbor_edge_states: Dict[int, float] # Neighbor index -> edge state from prev step

    # --- ADDED: Rule parameters specific to this step/context ---
    rule_params: Dict[str, Any] = field(default_factory=dict)
    # ---

    # --- ADDED: Optional 2nd Order Info ---
    neighbor_degrees: Optional[Dict[int, int]] = None # Neighbor index -> degree from prev step
    neighbor_active_counts: Optional[Dict[int, int]] = None # Neighbor index -> its active neighbor count from prev step
    # ---

    # Pre-calculated metrics (initialized by Grid._reconstruct_neighborhood_data or __post_init__)
    neighbor_metrics: Dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)
    neighborhood_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # --- MODIFIED: Only calculate metrics if they weren't pre-calculated ---
        if 'avg_neighbor_edge_state' not in self.neighborhood_metrics:
            if self.neighbor_edge_states:
                valid_edge_states = [state for idx, state in self.neighbor_edge_states.items() if idx >= 0]
                self.neighborhood_metrics['avg_neighbor_edge_state'] = float(np.mean(valid_edge_states)) if valid_edge_states else 0.0
            else:
                self.neighborhood_metrics['avg_neighbor_edge_state'] = 0.0
        if 'neighbor_state' not in self.neighbor_metrics:
            self.neighbor_metrics['neighbor_state'] = self.neighbor_states
        # --- END MODIFIED ---


################################################
#                  COORDINATES                 #
################################################

class CoordinateSystem:
    """Unified coordinate system handler for grid visualization"""

    def __init__(self, grid_dimensions, edge_scale=1.0, node_spacing=0.0, dimension_type=Dimension.TWO_D):
        """Initialize the coordinate system with grid parameters"""
        self.grid_dimensions = grid_dimensions
        self.edge_scale = edge_scale
        self.node_spacing = node_spacing # Store the desired spacing from .settings
        self.dimension_type = dimension_type

        # --- MODIFIED: Calculate scale_factor using effective spacing ---
        self._update_scale_factor()
        # ---

        # Default display bounds (these are just initial values, not actively used)
        self.display_bounds = self._calculate_default_display_bounds()

        logger.debug(f"CoordinateSystem initialized with dimensions: {grid_dimensions}, "
                    f"edge_scale: {edge_scale}, node_spacing_setting: {node_spacing}, "
                    f"calculated scale_factor: {self.scale_factor:.3f}")

    def _calculate_default_display_bounds(self):
        """Calculate default display bounds based on grid dimensions (not actively used)."""
        if self.dimension_type == Dimension.THREE_D:
            return {
                'x': (-0.5, self.grid_dimensions[1] - 0.5),
                'y': (-0.5, self.grid_dimensions[0] - 0.5),
                'z': (-0.5, self.grid_dimensions[2] - 0.5)
            }
        else:  # TWO_D
            return {
                'x': (-0.5, self.grid_dimensions[1] - 0.5),
                'y': (-0.5, self.grid_dimensions[0] - 0.5)
            }

    def grid_to_display(self, grid_coords):
        """Convert grid coordinates to display coordinates."""
        # logger.debug(f"grid_to_display called with grid_coords: {grid_coords}, scale_factor: {self.scale_factor}") # Added logging
        if self.dimension_type == Dimension.THREE_D:
            assert len(grid_coords) == 3, "3D grid requires 3 coordinates"
            i, j, k = grid_coords
            # For 3D, we use (j, i, k) for (x, y, z) in display
            return (j * self.scale_factor, i * self.scale_factor, k * self.scale_factor)
        else:  # TWO_D
            assert len(grid_coords) == 2, "2D grid requires 2 coordinates"
            i, j = grid_coords
            # For 2D, we use (j, i) for (x, y) in display
            return (j * self.scale_factor, i * self.scale_factor)
        
    def display_to_grid(self, display_coords):
        """Convert display coordinates to grid coordinates (approximate)."""
        # logger.debug(f"display_to_grid called with display_coords: {display_coords}, scale_factor: {self.scale_factor}")
        
        if self.dimension_type == Dimension.THREE_D:
            assert len(display_coords) == 3, "3D display requires 3 coordinates"
            x, y, z = display_coords
            
            # For 3D, we use (y, x, z) for (i, j, k) in grid
            # CRITICAL FIX: Divide by scale_factor to convert from display to grid
            i = y / self.scale_factor
            j = x / self.scale_factor
            k = z / self.scale_factor
            
            grid_coords = (i, j, k)
            logger.debug(f"display_to_grid: 3D grid_coords = {grid_coords}")
            return grid_coords
        else:  # TWO_D
            assert len(display_coords) == 2, "2D display requires 2 coordinates"
            x, y = display_coords
            
            # For 2D, we use (y, x) for (i, j) in grid
            # CRITICAL FIX: Divide by scale_factor to convert from display to grid
            i = y / self.scale_factor
            j = x / self.scale_factor
            
            grid_coords = (i, j)
            # logger.debug(f"display_to_grid: 2D grid_coords = {grid_coords}")
            return grid_coords
        
    def index_to_display(self, idx, grid_dimensions=None):
        """Convert a flat index to display coordinates."""
        if grid_dimensions is None:
            grid_dimensions = self.grid_dimensions

        # Convert flat index to grid coordinates
        grid_coords = _unravel_index(idx, grid_dimensions)

        # Convert grid coordinates to display coordinates
        return self.grid_to_display(grid_coords)

    def display_to_index(self, display_coords, grid_dimensions=None):
        """Convert display coordinates to a flat index (approximate)."""
        if grid_dimensions is None:
            grid_dimensions = self.grid_dimensions

        # Convert display coordinates to grid coordinates
        grid_coords = self.display_to_grid(display_coords)

        # Check if grid coordinates are within bounds
        for i, dim in enumerate(grid_coords):
            if dim < 0 or dim >= grid_dimensions[i]:
                return None  # Out of bounds

        # Convert grid coordinates to flat index
        return _ravel_multi_index(np.array(grid_coords), grid_dimensions)

    def update_parameters(self, edge_scale=None, node_spacing=None, dimension_type=None, grid_dimensions=None):
        """Update coordinate system parameters."""
        log_prefix = "CoordinateSystem.update_parameters: "
        updated = False
        if edge_scale is not None and edge_scale != self.edge_scale:
            self.edge_scale = edge_scale; updated = True
            logger.debug(f"{log_prefix}Updated edge_scale to {self.edge_scale}")
        if node_spacing is not None and node_spacing != self.node_spacing:
            self.node_spacing = node_spacing; updated = True # Store the new desired spacing
            logger.debug(f"{log_prefix}Updated node_spacing setting to {self.node_spacing}")
        if dimension_type is not None and dimension_type != self.dimension_type:
            self.dimension_type = dimension_type; updated = True
            logger.debug(f"{log_prefix}Updated dimension_type to {self.dimension_type}")
        if grid_dimensions is not None and grid_dimensions != self.grid_dimensions:
            self.grid_dimensions = grid_dimensions; updated = True
            logger.debug(f"{log_prefix}Updated grid_dimensions to {self.grid_dimensions}")

        if updated:
            # --- MODIFIED: Recalculate scale factor using effective spacing ---
            self._update_scale_factor()
            # ---
            # Recalculate default display bounds (not actively used, but kept for consistency)
            self.display_bounds = self._calculate_default_display_bounds()
            logger.debug(f"{log_prefix}Parameters updated. New scale_factor={self.scale_factor:.3f}")

    def _update_scale_factor(self):
        """Calculates the scale factor based on edge_scale and node_spacing setting."""
        try:
            # Use the NODE_SPACING setting directly from GlobalSettings
            current_spacing_setting = GlobalSettings.Visualization.NODE_SPACING
            self.scale_factor = self.edge_scale * (1.0 + current_spacing_setting)
            logger.debug(f"_update_scale_factor: EdgeScale={self.edge_scale:.2f}, SpacingSetting={current_spacing_setting:.3f} -> ScaleFactor={self.scale_factor:.3f}")
        except Exception as e:
            logger.error(f"Error calculating scale factor: {e}")
            # Fallback to a simple calculation on error
            self.scale_factor = self.edge_scale * (1.0 + self.node_spacing)

# =========== END of interfaces.py ===========