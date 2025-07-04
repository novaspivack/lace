# =========== START of rules.py ===========
from __future__ import annotations
from abc import abstractmethod
import json
import ast
import re
import shutil
import copy
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from typing import ClassVar, Dict, List, Set, Tuple, Optional, Type, Union, Any, Callable, TypeVar, cast
import numpy as np
import numpy.typing as npt
import itertools
import matplotlib.pyplot as plt
plt.ioff()
from enum import Enum, auto
from dataclasses import dataclass, field, fields
from collections import defaultdict
import logging
import random
import os
from datetime import datetime
import traceback
from numba import njit, prange
from numba import types
from numba.typed import Dict as NumbaDict
import numpy as np
from typing_extensions import Protocol
import warnings

from .logging_config import logger, APP_PATHS, LogSettings
from .enums import Dimension, NeighborhoodType, StateType
from .settings import GlobalSettings
from .interfaces import Rule, NeighborhoodData, RuleMetadata
from .utils import (
    PerformanceLogger, _unravel_index, _njit_unravel_index, _njit_ravel_multi_index, perf_logger, timer_decorator
    )   

# --- ADDED TYPE_CHECKING block ---
from typing import TYPE_CHECKING, Optional, Dict, Any, Tuple, Set, List, Callable, Union, ClassVar, Type
if TYPE_CHECKING:
    # Keep forward reference for type hints if needed elsewhere in this file
    from lace_app import Grid
# --- END ADDED ---


logger.info(f"Global PerformanceLogger instance created in lace_app.py (ID: {id(perf_logger)})")

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
#                   RULE ENGINE                #
################################################

class RuleMetrics(Protocol):
    """Protocol defining the interface for rule metrics"""
    
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                adj_list: Dict[int, Set[int]]) -> float:
        """Compute metric value"""
        ...

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

# @njit(cache=True)
def _calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculate Euclidean distance between two positions."""
    return np.sqrt(np.sum(np.square(pos1 - pos2)))

# @njit(cache=True)
def _calculate_angle_2d(center: np.ndarray, point: np.ndarray) -> float:
    """Calculate angle (in radians) between center and point in 2D."""
    vector = point - center
    return np.arctan2(vector[1], vector[0])

# @njit(cache=True)
def _calculate_angle_3d(center: np.ndarray, point: np.ndarray) -> Tuple[float, float]:
    """Calculate azimuth and elevation angles (in radians) between center and point in 3D."""
    vector = point - center
    xy_dist = np.sqrt(vector[0]**2 + vector[1]**2)
    azimuth = np.arctan2(vector[1], vector[0])
    elevation = np.arctan2(vector[2], xy_dist)
    return azimuth, elevation

_standard_colormaps = sorted([
    'viridis', 'prism', 'inferno', 'magma', 'cividis', # Perceptually Uniform
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
    'flag', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar', 'turbo'
])

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
                self._default_params = kwargs
    
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
         

# ──────────────────────────────────────────────────────────────────────
#   Rule Helper Functions
# ──────────────────────────────────────────────────────────────────────

# NOTE: Helper Implementation: The _get_opposing_neighbor_indices helper currently only has logic for 2D Moore and 2D Von Neumann neighborhoods. For other types, it returns an empty list, resulting in a symmetry score of 0.0.
# TODO: For 3D support we will need to update this logic to handle 3D Moore (26 neighbors) and Von Neumann (6 neighbors) neighborhoods. This will require mapping the 26/6 indices to their opposing positions.
# The SYMMETRY_STATE and SYMMETRY_DEGREE metrics will return 0.0 (or potentially error if data passing isn't fixed) for all other neighborhood types (3D, Hex, HexPrism) until this helper is fully implemented.

def _get_opposing_neighbor_indices(
    node_idx: int,
    dimensions: Tuple[int, ...],
    neighborhood_type: NeighborhoodType,
    all_neighbor_indices: np.ndarray, # Shape (total_nodes, max_neighbors)
    max_neighbors: int
) -> List[Tuple[int, int]]:
    """
    Identifies pairs of opposing neighbor indices for a given node.

    Currently supports:
        - 2D Moore (8 neighbors)
        - 2D Von Neumann (4 neighbors)

    Returns an empty list for unsupported types or if errors occur.

    Args:
        node_idx: The index of the central node.
        dimensions: The dimensions of the grid.
        neighborhood_type: The type of neighborhood.
        all_neighbor_indices: Pre-calculated neighbor indices array.
        max_neighbors: The maximum number of neighbors for this grid type.

    Returns:
        A list of tuples, where each tuple contains the indices of an
        opposing neighbor pair (idx1, idx2) with idx1 < idx2.
    """
    opposing_pairs: List[Tuple[int, int]] = []
    num_dims = len(dimensions)

    if not (0 <= node_idx < all_neighbor_indices.shape[0]):
        logger.warning(f"_get_opposing_neighbor_indices: node_idx {node_idx} out of bounds.")
        return opposing_pairs

    neighbor_indices_for_node = all_neighbor_indices[node_idx]
    valid_neighbors = neighbor_indices_for_node[neighbor_indices_for_node != -1]

    if len(valid_neighbors) == 0:
        return opposing_pairs # No neighbors, no pairs

    # --- Logic for specific neighborhood types ---
    # This relies on the consistent ordering defined in _populate_neighbor_array_optimized
    if num_dims == 2:
        if neighborhood_type == NeighborhoodType.MOORE and max_neighbors == 8:
            # Order: N, NE, E, SE, S, SW, W, NW (Indices 0 to 7)
            # Pairs: (N, S), (NE, SW), (E, W), (SE, NW) -> (0, 4), (1, 5), (2, 6), (3, 7)
            pair_map_indices = [(0, 4), (1, 5), (2, 6), (3, 7)]
        elif neighborhood_type == NeighborhoodType.VON_NEUMANN and max_neighbors == 4:
            # Order: N, E, S, W (Indices 0 to 3)
            # Pairs: (N, S), (E, W) -> (0, 2), (1, 3)
            pair_map_indices = [(0, 2), (1, 3)]
        else:
            logger.warning(f"Opposing neighbor calculation not implemented for 2D {neighborhood_type.name}. Returning empty list.")
            return opposing_pairs

        # Check if the indices exist in the actual neighbor list for the node
        for idx1_map, idx2_map in pair_map_indices:
            if idx1_map < len(neighbor_indices_for_node) and idx2_map < len(neighbor_indices_for_node):
                n1 = neighbor_indices_for_node[idx1_map]
                n2 = neighbor_indices_for_node[idx2_map]
                # Ensure both neighbors are valid (not -1) before adding the pair
                if n1 != -1 and n2 != -1:
                    # Ensure consistent order (smaller index first)
                    pair = tuple(sorted((int(n1), int(n2))))
                    if pair[0] != pair[1]: # Avoid self-pairing if somehow possible
                         opposing_pairs.append(cast(Tuple[int, int], pair))

    # --- Placeholder for 3D ---
    elif num_dims == 3:
        # TODO: Implement logic for 3D Moore (26 neighbors) and Von Neumann (6 neighbors)
        # This requires mapping the 26/6 indices to their opposing positions.
        logger.warning(f"Opposing neighbor calculation not yet implemented for 3D {neighborhood_type.name}. Returning empty list.")
        return opposing_pairs
    else:
        logger.warning(f"Opposing neighbor calculation not implemented for {num_dims}D. Returning empty list.")
        return opposing_pairs

    # Remove duplicate pairs (shouldn't happen with sorted tuples, but safe)
    return list(set(opposing_pairs))

def _calculate_symmetry_metric(
    opposing_pairs: List[Tuple[int, int]],
    data_array: np.ndarray, # e.g., previous_node_states or previous_node_degrees
    node_idx: int # For logging context
) -> float:
    """
    Calculates the average absolute difference between values of opposing neighbors.

    Args:
        opposing_pairs: List of opposing neighbor index pairs from _get_opposing_neighbor_indices.
        data_array: The NumPy array containing the data (e.g., states, degrees) for all nodes.
        node_idx: Index of the central node (for logging).

    Returns:
        The average absolute difference (float), or 0.0 if no valid pairs exist.
    """
    total_diff = 0.0
    pair_count = 0
    max_index = data_array.size - 1

    if not opposing_pairs:
        # logger.debug(f"Node {node_idx}: No opposing pairs provided for symmetry calculation.")
        return 0.0

    for idx1, idx2 in opposing_pairs:
        try:
            # Check bounds carefully
            if 0 <= idx1 <= max_index and 0 <= idx2 <= max_index:
                value1 = float(data_array[idx1]) # Ensure float for difference
                value2 = float(data_array[idx2])
                total_diff += abs(value1 - value2)
                pair_count += 1
            else:
                logger.warning(f"Node {node_idx}: Invalid index in opposing pair ({idx1}, {idx2}) for data array size {max_index+1}. Skipping pair.")
        except IndexError:
             logger.warning(f"Node {node_idx}: IndexError accessing data for pair ({idx1}, {idx2}). Skipping pair.")
        except Exception as e:
             logger.error(f"Node {node_idx}: Error calculating difference for pair ({idx1}, {idx2}): {e}")

    if pair_count > 0:
        avg_diff = total_diff / pair_count
        # logger.debug(f"Node {node_idx}: Calculated symmetry metric: {avg_diff:.4f} (from {pair_count} pairs)")
        return avg_diff
    else:
        # logger.debug(f"Node {node_idx}: No valid opposing pairs found to calculate symmetry.")
        return 0.0

def calculate_max_neighbors(
    dimension_type: Dimension,
    neighborhood_type: NeighborhoodType
) -> int:
    """
    Calculate maximum possible neighbors based on dimension and neighborhood type.

    Args:
        dimension_type: The grid dimension (Dimension.TWO_D or Dimension.THREE_D).
        neighborhood_type: The neighborhood type enum.

    Returns:
        The maximum number of neighbors.

    Raises:
        ValueError: If an invalid combination is provided.
    """
    if neighborhood_type == NeighborhoodType.VON_NEUMANN:
        if dimension_type == Dimension.TWO_D: return 4
        elif dimension_type == Dimension.THREE_D: return 6
        else: raise ValueError(f"Invalid dimension type {dimension_type} for VON_NEUMANN neighborhood")
    elif neighborhood_type == NeighborhoodType.MOORE:
        if dimension_type == Dimension.TWO_D: return 8
        elif dimension_type == Dimension.THREE_D: return 26
        else: raise ValueError(f"Invalid dimension type {dimension_type} for MOORE neighborhood")
    elif neighborhood_type == NeighborhoodType.HEX:
        if dimension_type == Dimension.TWO_D: return 6
        else: raise ValueError(f"Invalid dimension type {dimension_type} for HEX neighborhood")
    elif neighborhood_type == NeighborhoodType.HEX_PRISM:
        if dimension_type == Dimension.THREE_D: return 12
        else: raise ValueError(f"Invalid dimension type {dimension_type} for HEX_PRISM neighborhood")
    else:
        raise ValueError(f"Invalid neighborhood type: {neighborhood_type}")
# --- END ADDED ---

################################################
#            RULE CLASS DEFINITIONS            #
################################################


class RealmOfLace(Rule):
    """
    A fascinating new class of Lace rule where node state represents its degree (connection count).
    Node eligibility (for edge formation/survival) is determined by the SUM of neighbor degrees
    from the previous step falling within specified ranges. 
    Edges (binary 0/1) exist only between mutually eligible nodes. 
    Requires post-update step.
    We abbreviate this rule name as "ROL" (Realm of Lace). 
    Variants of this rule with different parameter settings also exhibit interesting behaviors.
    (Round 1: Corrected default parameter types)
    """
    # --- MODIFIED: Update EXCLUDE_EDITOR_PARAMS and PARAMETER_METADATA ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'min_edge_weight', 'max_edge_weight', # Edges are binary
        'connect_probability', # Not used by core logic, only init
        # Keep edge coloring params as they are now relevant
        'color_edges_by_neighbor_degree', # Replaced by edge_coloring_mode
        'color_edges_by_neighbor_active_neighbors', # Replaced by edge_coloring_mode
        'node_history_depth'
        # tiebreaker_type is used by edge logic, keep it
    }

    node_state_type: ClassVar[StateType] = StateType.INTEGER
    edge_state_type: ClassVar[StateType] = StateType.BINARY # Edges are 0/1
    min_node_state: ClassVar[float] = 0.0 # Degree cannot be negative
    max_node_state: ClassVar[float] = 26.0 # Theoretical max degree (Moore 3D)
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern (Nodes initialized to 0, state becomes degree).", "default": "Random", "allowed_values": ['Random'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Approx initial density of nodes with connections (influences edge init).", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},

        # === Node Eligibility Logic (Based on Neighbor Degree SUM) ===
        "birth_neighbor_degree_sum_range": {
            'type': list, 'element_type': tuple,
            'default': [(3.0, 5.0), (10.0, 14.0)],
            'description': "Node Birth Eligibility: List of (min_sum, max_sum) ranges for neighbor degree sum (prev step). Node eligible if sum falls in any range.",
            "parameter_group": "Node Eligibility"
        },
        "survival_neighbor_degree_sum_range": {
            'type': list, 'element_type': tuple,
            'default': [(2.0, 4.0), (8.0, 12.0)],
            'description': "Node Survival Eligibility: List of (min_sum, max_sum) ranges for neighbor degree sum (prev step). Node eligible if sum falls in any range.",
            "parameter_group": "Node Eligibility"
        },

        # === Final State Death Conditions ===
        "final_death_degree_counts": {
            'type': list, 'element_type': int,
            'default': [0, 1, 7, 8],
            'description': "Final State: Node state becomes 0 if its *final* calculated degree is in this list.",
            "parameter_group": "Final State Death"
        },
        # --- ADDED: Optional Death Range ---
        "final_death_degree_range": {
            'type': list, 'element_type': tuple,
            'default': [], # Default to empty list (not used)
            'description': "Final State: Optional list of (min_deg, max_deg) ranges. Node state becomes 0 if final degree falls in ANY range. Checked *in addition* to specific counts.",
            "parameter_group": "Final State Death"
        },
        # ---

        # === Edge Logic (Mutual Eligibility) ===
        # (No specific parameters needed for this basic mutual eligibility)

        # === Visualization: Nodes ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (connection count).", "default": True, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states (degrees).", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization.", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization.", "default": 8.0, "parameter_group": "Visualization: Nodes"},

        # === Visualization: Edges ===
        "use_state_coloring_edges": { "type": bool, "description": "Enable edge coloring based on selected mode (overrides simple binary).", "default": False, "parameter_group": "Visualization: Edges"},
        # --- ADDED: Edge Coloring Mode ---
        "edge_coloring_mode": {
            'type': str, 'default': 'Default',
            'allowed_values': ['Default', 'ActiveNeighbors', 'DegreeSum'],
            'description': "Edge Color: 'Default' (binary 0/1), 'ActiveNeighbors' (avg active neighbors of endpoints), 'DegreeSum' (sum of endpoint degrees). Uses prev step data.",
            "parameter_group": "Visualization: Edges"
        },
        # ---
        "edge_colormap": { "type": str, "description": "Colormap for edge coloring (if enabled & not Default).", "default": "prism", "parameter_group": "Visualization: Edges", "allowed_values": ["(None)"] + _standard_colormaps},
        "edge_color_norm_vmin": { "type": float, "description": "Min value for edge color normalization (for ActiveNeighbors/DegreeSum modes).", "default": 0.0, "parameter_group": "Visualization: Edges"},
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization (e.g., 16 for DegreeSum in Moore 2D).", "default": 16.0, "parameter_group": "Visualization: Edges"}, # Default max for DegreeSum

        # === Other ===
        "tiebreaker_type": { "type": str, "description": "Tiebreaker for edge conflicts (Not typically relevant).", "allowed_values": ["RANDOM", "AGREEMENT"], "default": "RANDOM", "parameter_group": "Tiebreaker"},
        "node_history_depth": { 'type': int, 'description': 'Grid Setting: History depth (not used by rule).', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Realm of Lace"
        metadata.description = "Node state = degree. Eligibility based on sum of neighbor degrees (prev step) within ranges. Edges exist only between mutually eligible nodes. Requires post-update step." # Updated description
        metadata.category = "Connectivity-Based"
        metadata.tags = ["Connectivity", "ROL", "Degree", "Discrete State", "Edges", "Eligibility", "Dynamic"] # Added Dynamic
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Realm of Lace"
        self._params = {}
        self.requires_post_edge_state_update = True # Requires final degree calculation
        self.needs_neighbor_degrees = True
        # --- MODIFIED: Set needs_neighbor_active_counts to True ---
        # Required for the 'ActiveNeighbors' edge coloring mode
        self.needs_neighbor_active_counts = True
        # ---
        self.skip_standard_tiebreakers = True # Uses its own edge logic
        # Set coloring defaults AFTER super().__init__ which populates _params
        self._params.setdefault('use_state_coloring', True)
        self._params.setdefault('node_colormap', 'prism')
        self._params.setdefault('node_color_norm_vmax', 8.0)
        # Edge coloring defaults are handled by PARAMETER_METADATA

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _check_sum_ranges(self, value: float, ranges: List[Tuple[float, float]]) -> bool:
        """Checks if a value falls within any of the specified (min_sum, max_sum) ranges."""
        if not ranges: return False # Fail if no ranges specified (must meet a condition)
        # --- ADDED: Ensure inner elements are tuples of numbers ---
        valid_ranges = []
        for r in ranges:
            if isinstance(r, (list, tuple)) and len(r) == 2:
                try:
                    min_val = float(r[0])
                    max_val = float(r[1])
                    valid_ranges.append((min_val, max_val))
                except (ValueError, TypeError):
                    logger.warning(f"_check_sum_ranges: Skipping invalid range element '{r}' (cannot convert to float tuple).")
            else:
                 logger.warning(f"_check_sum_ranges: Skipping invalid range element '{r}' (not list/tuple of length 2).")
        # --- END ADDED ---
        return any(min_val <= value <= max_val for min_val, max_val in valid_ranges) # Use the filtered list'

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """
        Calculates the node's ELIGIBILITY proxy state (0 or 1) based on the SUM
        of its neighbors' degrees from the previous step falling within specified ranges.
        (Round 21 Fix: Use SUM instead of AVERAGE)
        """
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters using the CORRECTED names
        birth_sum_ranges = self.get_param('birth_neighbor_degree_sum_range', [[2.0, 15.0]], neighborhood=neighborhood)
        survival_sum_ranges = self.get_param('survival_neighbor_degree_sum_range', [[1.0, 20.0]], neighborhood=neighborhood)

        # --- Calculate SUM of neighbor degrees from previous step ---
        sum_neighbor_degree = 0.0
        neighbor_degree_count = 0 # Count how many neighbors we could get a degree for
        if neighborhood.neighbor_degrees is not None:
            for neighbor_idx in neighborhood.neighbor_indices:
                 if neighbor_idx >= 0: # Consider all valid neighbors
                      sum_neighbor_degree += neighborhood.neighbor_degrees.get(neighbor_idx, 0)
                      neighbor_degree_count += 1
        else:
            # Fallback: Use neighbor states as degree proxy if neighbor_degrees is missing
            logger.warning(f"Node {node_idx}: neighbor_degrees missing! Falling back to using neighbor states as degree proxy.")
            for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
                 if neighbor_idx >= 0:
                      sum_neighbor_degree += neighbor_state # Add the state directly
                      neighbor_degree_count += 1
        # ---

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state (Eligibility Proxy) ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State (Degree): {neighborhood.node_state:.0f}") # type: ignore [attr-defined]
            logger.detail(f"    Neighbor Degree Sum (Prev Step): {sum_neighbor_degree:.2f} (from {neighbor_degree_count} neighbors)") # type: ignore [attr-defined]
            logger.detail(f"    Birth Sum Ranges: {birth_sum_ranges}") # type: ignore [attr-defined]
            logger.detail(f"    Survival Sum Ranges: {survival_sum_ranges}") # type: ignore [attr-defined]

        eligibility_proxy = 0.0
        decision_reason = "Default (Ineligible)"

        # --- Check eligibility based on SUM ranges ---
        if neighborhood.node_state <= 1e-6: # --- Check Birth Eligibility ---
            passes_birth_sum = self._check_sum_ranges(sum_neighbor_degree, birth_sum_ranges)
            if passes_birth_sum:
                eligibility_proxy = 1.0
                decision_reason = f"Eligible for Birth (Sum={sum_neighbor_degree:.2f} in {birth_sum_ranges})"
            else:
                decision_reason = f"Ineligible (Birth Sum={sum_neighbor_degree:.2f} not in {birth_sum_ranges})"

        else: # --- Check Survival Eligibility ---
            passes_survival_sum = self._check_sum_ranges(sum_neighbor_degree, survival_sum_ranges)
            if passes_survival_sum:
                eligibility_proxy = 1.0
                decision_reason = f"Eligible for Survival (Sum={sum_neighbor_degree:.2f} in {survival_sum_ranges})"
            else:
                decision_reason = f"Ineligible (Death: Sum={sum_neighbor_degree:.2f} not in {survival_sum_ranges})"
        # ---

        if detailed_logging_enabled:
            logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    Returning Eligibility Proxy: {eligibility_proxy:.1f}") # type: ignore [attr-defined]

        return eligibility_proxy

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> Dict[Tuple[int, int], float]:
        """Determine edge existence based on mutual eligibility using proxy states."""

        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # --- Get Eligibility Proxies from rule_params ---
        eligibility_proxies_flat = neighborhood.rule_params.get('_eligibility_proxies')
        if eligibility_proxies_flat is None:
            logger.error(f"Node {node_idx}: Eligibility proxies missing in rule_params!")
            return new_edges # Cannot determine mutual eligibility
        # ---

        # Determine eligibility of the current node for the *next* step from proxy
        self_is_eligible = False
        if 0 <= node_idx < eligibility_proxies_flat.size:
            self_is_eligible = eligibility_proxies_flat[node_idx] > 0.5
        else:
            logger.warning(f"Node {node_idx}: Index out of bounds for eligibility proxies.")

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Eligible Next (from proxy): {self_is_eligible}") # type: ignore [attr-defined]

        # If self is ineligible, no edges can form/survive from its perspective
        if not self_is_eligible:
             return new_edges

        for neighbor_idx in neighborhood.neighbor_indices: # Iterate through valid neighbor indices
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue # Skip invalid and avoid double check

            # --- Get Neighbor Eligibility from Proxy Array ---
            neighbor_is_eligible = False
            if 0 <= neighbor_idx < eligibility_proxies_flat.size:
                neighbor_is_eligible = eligibility_proxies_flat[neighbor_idx] > 0.5
            else:
                logger.warning(f"Node {node_idx}: Neighbor index {neighbor_idx} out of bounds for eligibility proxies.")
                continue # Skip if neighbor index is invalid
            # ---

            edge = (node_idx, neighbor_idx) # Canonical order
            propose_edge = False
            decision_reason = "Default (No Edge)"

            # Propose edge ONLY if BOTH nodes are eligible based on proxy states
            if self_is_eligible and neighbor_is_eligible:
                propose_edge = True
                decision_reason = "Propose Edge (Both eligible based on proxy)"
            # else: # If one or both are ineligible, no edge is proposed

            if detailed_logging_enabled:
                logger.detail(f"  Neighbor {neighbor_idx}: Eligible Next(Proxy)={neighbor_is_eligible}. Decision: {decision_reason}. Propose={propose_edge}") # type: ignore [attr-defined]

            # Add to proposed edges if it should exist
            if propose_edge:
                new_edges[edge] = 1.0 # Binary edges

        return new_edges

    def _compute_final_state(self,
                            node_idx: int,
                            current_proxy_state: float, # Eligibility/state from current step's computation
                            final_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]], # Edges calculated for *current* step
                            dimensions: Tuple[int,...],
                            # --- Accept all arguments even if unused ---
                            previous_node_states: npt.NDArray[np.float64],
                            previous_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
                            previous_edge_states: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float],
                            previous_node_degrees: Optional[npt.NDArray[np.int32]],
                            previous_active_neighbors: Optional[npt.NDArray[np.int32]],
                            eligibility_proxies: Optional[np.ndarray] = None,
                            detailed_logging_enabled: bool = False
                            ) -> float:
        """
        Calculates the final state (degree) based on eligibility and final edge count,
        applying death list AND death range.
        (Round 4: Added death range check)
        """
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_final_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Proxy State (Eligibility): {current_proxy_state:.1f}") # type: ignore [attr-defined]

        # If proxy state indicates ineligible (0.0), final state is 0
        if current_proxy_state < 0.5:
            if detailed_logging_enabled: logger.detail(f"    Node was ineligible based on prev step conditions, final state is 0.") # type: ignore [attr-defined]
            return 0.0

        # If eligible (proxy was 1.0), calculate final degree based on the FINAL edges for this step
        node_coords = tuple(_unravel_index(node_idx, dimensions))
        final_degree = sum(1 for edge in final_edges if node_coords in edge)

        if detailed_logging_enabled:
            logger.detail(f"    Node was eligible. Final Calculated Degree: {final_degree}") # type: ignore [attr-defined]

        # Apply death list and range based on the FINAL degree
        final_death_degrees = self.get_param('final_death_degree_counts', [0])
        # --- ADDED: Get death range parameter ---
        final_death_range = self.get_param('final_death_degree_range', [])
        # ---

        final_state = 0.0 # Default to death
        decision_reason = "Default (Death)"

        # --- MODIFIED: Check both counts and ranges ---
        dies_by_count = final_degree in final_death_degrees
        dies_by_range = self._check_sum_ranges(float(final_degree), final_death_range) # Use float for range check

        if dies_by_count:
            decision_reason = f"Final Death (Final Degree={final_degree} in death list {final_death_degrees})"
        elif dies_by_range:
            decision_reason = f"Final Death (Final Degree={final_degree} in death range {final_death_range})"
        else:
            # Survived final death checks, state is the final degree
            final_state = float(final_degree)
            decision_reason = f"Final Survival (Final Degree={final_degree} not in death criteria)"
        # --- END MODIFIED ---

        if detailed_logging_enabled:
            logger.detail(f"    Final Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    Final State: {final_state:.1f}") # type: ignore [attr-defined]
        return final_state

# TODO: The SYMMETRY_STATE and SYMMETRY_DEGREE metrics will return 0.0 (or potentially error if data passing isn't fixed) for all other neighborhood types (3D, Hex, HexPrism) until this helper is fully implemented (Deferred)
class RealmOfLaceUnified(Rule):
    """
    **Realm of Lace (Unified) - "ROL-U"**
    Node state=degree. Eligibility uses configurable metrics (Degree, Clustering,
    Betweenness, Active Neigh Count, Symmetry, Variance) and aggregation (Sum, Average)
    independently for birth/survival, checked against counts OR ranges.
    Optional perturbations. Edges based on mutual eligibility. Final state check uses
    configurable metric and life/death criteria.
    Default Viz: Nodes colored by Degree, Edges by DegreeSum.
    (Round 17: Updated default visualization parameters)
    """

    # ──────────────────────────────────────────────────────────────────────
    #   Class Attributes & Metadata
    # ──────────────────────────────────────────────────────────────────────

    node_state_type: ClassVar[StateType] = StateType.INTEGER
    edge_state_type: ClassVar[StateType] = StateType.BINARY
    min_node_state: ClassVar[float] = 0.0
    max_node_state: ClassVar[float] = 26.0 # Theoretical max degree (Moore 3D)
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0
    produces_binary_edges: ClassVar[bool] = True
    use_jit_state_phase: ClassVar[bool] = True
    use_jit_edge_phase: ClassVar[bool] = True

    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'min_edge_weight', 'max_edge_weight', # Edges are binary
        'connect_probability', # Not used by core logic, only init
        'color_edges_by_neighbor_degree', # Replaced by edge_coloring_mode
        'color_edges_by_neighbor_active_neighbors', # Replaced by edge_coloring_mode
        'node_history_depth', # Grid setting, not rule logic
        'tiebreaker_type', # Not used by this rule's logic
        # Remove old dynamic description params
        'birth_eligibility_range', 'birth_eligibility_counts',
        'survival_eligibility_range', 'survival_eligibility_counts',
        'final_death_metric_counts', 'final_death_metric_range',
    }

    FINAL_CHECK_METRIC_DETAILS: ClassVar[Dict[str, Dict[str, str]]] = {
        "DEGREE": {
            "Meaning": "Final connection count of the node after edge updates.",
            "Calculation": "Count of edges connected to the node in the final graph.",
            "Value Type": "Integer (used for counts), Float (used for ranges)",
            "Range": "[0, MaxN] (e.g., 0-8 for 2D Moore)",
            "Example Counts": "[0, 1, 7, 8]",
            "Example Ranges": "[[0.0, 1.5], [6.5, 8.0]]"
        },
        "CLUSTERING": {
            "Meaning": "Proxy clustering coefficient based on final degree.",
            "Calculation": "Computed as (deg * (deg - 1)) / (MaxN * (MaxN - 1)) if deg > 1 else 0.0.",
            "Value Type": "Float",
            "Range": "[0.0, 1.0]",
            "Example Counts": "[0, 1]", # Rounded value compared
            "Example Ranges": "[[0.0, 0.1], [0.8, 1.0]]" # Unrounded value compared
        },
        "BETWEENNESS": {
            "Meaning": "Proxy betweenness based on final degree.",
            "Calculation": "Computed as 1.0 / deg if deg > 0 else 0.0.",
            "Value Type": "Float",
            "Range": "[0.0, 1.0]",
            "Example Counts": "[0, 1]", # Rounded value compared
            "Example Ranges": "[[0.0, 0.2], [0.9, 1.0]]" # Unrounded value compared
        },
        "ACTIVE_NEIGHBOR_COUNT": {
            "Meaning": "Count of neighbors whose eligibility proxy was > 0.5.",
            "Calculation": "Number of neighbors considered active based on eligibility.",
            "Value Type": "Integer (used for counts), Float (used for ranges)",
            "Range": "[0, MaxN] (e.g., 0-8 for 2D Moore)",
            "Example Counts": "[0, 1, 2]",
            "Example Ranges": "[[0.0, 2.5]]"
        }
    }

    PARAMETER_METADATA = {
        # === Core === (Group Sort Key: 0)
        "dimension_type": {
            'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"],
            'description': "Grid dimensionality (TWO_D or THREE_D).",
            "parameter_group": "Core", "editor_sort_key": 10
        },
        "neighborhood_type": {
            'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"],
            'description': "Defines adjacent cells (e.g., MOORE includes diagonals, VON_NEUMANN does not). HEX requires 2D, HEX_PRISM requires 3D.",
            "parameter_group": "Core", "editor_sort_key": 20
        },
        'grid_boundary': {
            'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap',
            'description': "Grid boundary behavior ('wrap' connects edges, 'bounded' stops at edges).",
            "parameter_group": "Core", "editor_sort_key": 30
        },

        # === Initialization === (Group Sort Key: 1)
        "initial_conditions": {
            "type": str, "default": "Random", "allowed_values": ['Random'],
            "description": "Initial grid state pattern (Nodes initialized to 0, state becomes degree). Currently only 'Random' is directly supported by this rule's init logic, others require presets.",
            "parameter_group": "Initialization", "editor_sort_key": 10
        },
        "initial_density": {
            "type": float, "min": 0.0, "max": 1.0, "default": 0.5,
            "description": "Approx initial density of nodes with connections (influences random edge initialization).",
            "parameter_group": "Initialization", "editor_sort_key": 20
        },
        "edge_initialization": {
            'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM',
            'description': "Initial edge setup: NONE=No edges, FULL=Connect all active neighbors, RANDOM=Connect active neighbors with probability based on Initial Density.",
            "parameter_group": "Initialization", "editor_sort_key": 30
        },

        # === Node Eligibility Logic === (Group Sort Key: 2)
        "birth_metric_type": {
            "type": str, "default": "DEGREE",
            "allowed_values": ["DEGREE", "CLUSTERING", "BETWEENNESS", "ACTIVE_NEIGHBOR_COUNT",
                               "SYMMETRY_STATE", "SYMMETRY_DEGREE",
                               "NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV"],
            "description": ("BIRTH Metric: Selects the metric used for birth eligibility (applied to inactive nodes).\n"
                           "Calculated based on neighbor data from the *previous* step."),
            "parameter_group": "Node Eligibility", "editor_sort_key": 10
        },
        "birth_metric_aggregation": {
            "type": str, "default": "SUM",
            "allowed_values": ["SUM", "AVERAGE"],
            "description": "BIRTH Aggregation: How per-neighbour metrics (except Variance/StdDev/Symmetry) are combined.",
            "parameter_group": "Node Eligibility", "editor_sort_key": 20
        },
        "survival_metric_type": {
            "type": str, "default": "DEGREE",
            "allowed_values": ["DEGREE", "CLUSTERING", "BETWEENNESS", "ACTIVE_NEIGHBOR_COUNT",
                               "SYMMETRY_STATE", "SYMMETRY_DEGREE",
                               "NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV"],
            "description": ("SURVIVAL Metric: Selects the metric used for survival eligibility (applied to active nodes).\n"
                           "Calculated based on neighbor data from the *previous* step."),
            "parameter_group": "Node Eligibility", "editor_sort_key": 50
        },
        "survival_metric_aggregation": {
            "type": str, "default": "SUM",
            "allowed_values": ["SUM", "AVERAGE"],
            "description": "SURVIVAL Aggregation: How per-neighbour metrics (except Variance/StdDev/Symmetry) are combined.",
            "parameter_group": "Node Eligibility", "editor_sort_key": 60
        },
        "clustering_denominator_type": {
            "type": str, "default": "ACTUAL", "allowed_values": ["ACTUAL", "THEORETICAL"],
            "description": ("Denominator for CLUSTERING proxy calculation:\n"
                           "- **ACTUAL:** Uses `N*(N-1)` where N is the *actual* number of valid neighbors found for the central node (like original ROLM). More sensitive to local density.\n"
                           "- **THEORETICAL:** Uses `MaxN*(MaxN-1)` where MaxN is the *theoretical maximum* neighbors for the grid type (e.g., 8 for 2D Moore). More consistent normalization across the grid."),
            "parameter_group": "Node Eligibility", "editor_sort_key": 90
        },
        # --- Base Templates for Dynamic Permutations (Unchanged) ---
        "birth_eligibility_range_BASE": {
            'type': list, 'element_type': tuple, 'parameter_group': "Node Eligibility", 'editor_sort_key': 40,
            '_description_template': ("Node Birth Eligibility (Ranges - {METRIC}/{AGG_STR}):\n"
                                     "- Value Type: List of Float Ranges `[[min, max], ...]`. Default: {DEFAULT}\n"
                                     "- Comparison: The calculated metric value (Float) is compared **directly (unrounded)** against these ranges.\n"
                                     "- Value Ranges (Metric): {RANGES}\n"
                                     "- Explanation: Metric is the {METRIC_EXPLANATION}. Use this parameter to define continuous intervals where birth is allowed. Float precision matters.\n"
                                     "- Definition: Inactive node is eligible if its **unrounded** metric value falls within ANY [min, max] range in this list. Checked *in addition* to Specific Values.\n"
                                     "- Example: {EXAMPLE}")
        },
        "birth_eligibility_values_BASE": {
            'type': list, 'element_type': object, 'parameter_group': "Node Eligibility", 'editor_sort_key': 30, # Use object for Union[int, float]
            '_description_template': ("Node Birth Eligibility (Specific Values - {METRIC}/{AGG_STR}):\n"
                                     "- Value Type: List of Numbers `[n1, n2, ...]`, Integers or Floats. Default: {DEFAULT}\n"
                                     "- Comparison Logic:\n"
                                     "    - Integer Target (e.g., `3`): Calculated metric (Float) is **rounded** to the nearest integer. Eligible if rounded result equals target integer.\n"
                                     "    - Float Target (e.g., `4.5`): Eligible if **unrounded** calculated metric (Float) is within +/- 0.005 of the target float.\n"
                                     "- Value Ranges (Metric): {RANGES}\n"
                                     "- Explanation: Metric is the {METRIC_EXPLANATION}. Use this to specify individual values. Integers match rounded values (like B/S counts). Floats match values very close to the specified float.\n"
                                     "- Definition: Inactive node is eligible if its metric value matches ANY value in this list (using logic above). Checked *in addition* to Ranges.\n"
                                     "- Example: {EXAMPLE}")
        },
        "survival_eligibility_range_BASE": {
            'type': list, 'element_type': tuple, 'parameter_group': "Node Eligibility", 'editor_sort_key': 80,
             '_description_template': ("Node Survival Eligibility (Ranges - {METRIC}/{AGG_STR}):\n"
                                     "- Value Type: List of Float Ranges `[[min, max], ...]`. Default: {DEFAULT}\n"
                                     "- Comparison: The calculated metric value (Float) is compared **directly (unrounded)** against these ranges.\n"
                                     "- Value Ranges (Metric): {RANGES}\n"
                                     "- Explanation: Metric is the {METRIC_EXPLANATION}. Use this parameter to define continuous intervals where survival is allowed. Float precision matters.\n"
                                     "- Definition: Active node is eligible if its **unrounded** metric value falls within ANY [min, max] range in this list. Checked *in addition* to Specific Values.\n"
                                     "- Example: {EXAMPLE}")
        },
        "survival_eligibility_values_BASE": {
            'type': list, 'element_type': object, 'parameter_group': "Node Eligibility", 'editor_sort_key': 70,
            '_description_template': ("Node Survival Eligibility (Specific Values - {METRIC}/{AGG_STR}):\n"
                                     "- Value Type: List of Numbers `[n1, n2, ...]`, Integers or Floats. Default: {DEFAULT}\n"
                                     "- Comparison Logic:\n"
                                     "    - Integer Target (e.g., `3`): Calculated metric (Float) is **rounded** to the nearest integer. Eligible if rounded result equals target integer.\n"
                                     "    - Float Target (e.g., `4.5`): Eligible if **unrounded** calculated metric (Float) is within +/- 0.005 of the target float.\n"
                                     "- Value Ranges (Metric): {RANGES}\n"
                                     "- Explanation: Metric is the {METRIC_EXPLANATION}. Use this to specify individual values. Integers match rounded values (like B/S counts). Floats match values very close to the specified float.\n"
                                     "- Definition: Active node is eligible if its metric value matches ANY value in this list (using logic above). Checked *in addition* to Ranges.\n"
                                     "- Example: {EXAMPLE}")
        },
        # --- END BASE TEMPLATES ---

        # === Final State Check Conditions === (Group Sort Key: 3)
        "final_check_metric": {
            "type": str, "default": "DEGREE",
            "allowed_values": ["DEGREE", "CLUSTERING", "BETWEENNESS", "ACTIVE_NEIGHBOR_COUNT"],
            "description": ("Metric used for Final Life/Death Checks. This check happens *after* node eligibility is determined and edges for the step are finalized.\n"
                           "DEGREE (Integer) uses the final connection count. ACTIVE_NEIGHBOR_COUNT (Integer) uses the count of neighbors whose eligibility proxy (from Phase 1) was > 0.5. CLUSTERING (Float Proxy) and BETWEENNESS (Float Proxy) use fast proxies based on final degree.\n"
                           "*** PERFORMANCE NOTE: ***\n"
                           "Using DEGREE or ACTIVE_NEIGHBOR_COUNT is fast. Using CLUSTERING or BETWEENNESS uses inexpensive proxies based on the final degree and should also be fast."),
            "parameter_group": "Final State Check", "editor_sort_key": 10
        },
        # --- Base Templates for Final State Dynamic Permutations (Unchanged) ---
        "final_death_metric_range_BASE": {
            'type': list, 'element_type': tuple, 'parameter_group': "Final State Check", 'editor_sort_key': 30,
            '_description_template': ("Final State Death (Range - {METRIC}):\n"
                                     "- Value Type: List of Float Ranges `[[min, max], ...]`. Default: {DEFAULT}\n"
                                     "- Comparison: The calculated final metric value (Type: {VALUE_TYPE_STR}) is compared **directly (unrounded)** against these ranges.\n"
                                     "- Value Ranges (Metric): {RANGES}\n"
                                     "- Explanation: Metric is the {METRIC_EXPLANATION}. Use this parameter to define continuous intervals where the node's final state becomes 0 (death).\n"
                                     "- Definition: Node state becomes 0 if its **unrounded** final metric value falls in ANY range (unless life condition met). Checked *in addition* to Specific Values.\n"
                                     "- Example: {EXAMPLE}")
        },
        "final_death_metric_values_BASE": {
            'type': list, 'element_type': object, 'parameter_group': "Final State Check", 'editor_sort_key': 20,
            '_description_template': ("Final State Death (Specific Values - {METRIC}):\n"
                                     "- Value Type: List of Numbers `[n1, n2, ...]`, Integers or Floats. Default: {DEFAULT}\n"
                                     "- Comparison Logic:\n"
                                     "    - Integer Target (e.g., `3`): Calculated final metric (Type: {VALUE_TYPE_STR}) is **rounded** to the nearest integer. Death occurs if rounded result equals target integer.\n"
                                     "    - Float Target (e.g., `0.1`): Death occurs if **unrounded** calculated final metric (Type: {VALUE_TYPE_STR}) is within +/- 0.005 of the target float.\n"
                                     "- Value Ranges (Metric): {RANGES}\n"
                                     "- Explanation: Metric is the {METRIC_EXPLANATION}. Use this to specify individual values that cause the node's final state to become 0 (death).\n"
                                     "- Definition: Node state becomes 0 if its final metric value matches ANY value in this list (using logic above, unless life condition met). Checked *in addition* to Ranges.\n"
                                     "- Example: {EXAMPLE}")
        },
        "final_life_metric_range_BASE": {
            'type': list, 'element_type': tuple, 'parameter_group': "Final State Check", 'editor_sort_key': 50,
            '_description_template': ("Final State Life (Range - {METRIC}):\n"
                                     "- Value Type: List of Float Ranges `[[min, max], ...]`. Default: {DEFAULT}\n"
                                     "- Comparison: The calculated final metric value (Type: {VALUE_TYPE_STR}) is compared **directly (unrounded)** against these ranges.\n"
                                     "- Value Ranges (Metric): {RANGES}\n"
                                     "- Explanation: Metric is the {METRIC_EXPLANATION}. Use this parameter to define continuous intervals where the node survives (state becomes final degree).\n"
                                     "- Definition: Node survives if its **unrounded** final metric value falls in ANY range (overrides death conditions). Checked *in addition* to Specific Values.\n"
                                     "- Example: {EXAMPLE}")
        },
        "final_life_metric_values_BASE": {
            'type': list, 'element_type': object, 'parameter_group': "Final State Check", 'editor_sort_key': 40,
            '_description_template': ("Final State Life (Specific Values - {METRIC}):\n"
                                     "- Value Type: List of Numbers `[n1, n2, ...]`, Integers or Floats. Default: {DEFAULT}\n"
                                     "- Comparison Logic:\n"
                                     "    - Integer Target (e.g., `3`): Calculated final metric (Type: {VALUE_TYPE_STR}) is **rounded** to the nearest integer. Survival occurs if rounded result equals target integer.\n"
                                     "    - Float Target (e.g., `0.5`): Survival occurs if **unrounded** calculated final metric (Type: {VALUE_TYPE_STR}) is within +/- 0.005 of the target float.\n"
                                     "- Value Ranges (Metric): {RANGES}\n"
                                     "- Explanation: Metric is the {METRIC_EXPLANATION}. Use this to specify individual values that cause the node to survive (state becomes final degree).\n"
                                     "- Definition: Node survives if its final metric value matches ANY value in this list (using logic above, overrides death conditions). Checked *in addition* to Ranges.\n"
                                     "- Example: {EXAMPLE}")
        },
        # --- END BASE TEMPLATES ---

        # === Perturbations === (Group Sort Key: 4)
        "perturbation_enable": { "type": bool, "default": False, "description": "Enable random perturbations to prevent static states.", "parameter_group": "Perturbations", "editor_sort_key": 10 },
        "random_state_flip_probability": { "type": float, "min": 0.0, "max": 1.0, "default": 0.0, "description": "Probability per node per step to flip its calculated eligibility proxy (0->1 or 1->0). Applied BEFORE edge calculation.", "parameter_group": "Perturbations", "editor_sort_key": 20 },
        "random_edge_toggle_probability": { "type": float, "min": 0.0, "max": 1.0, "default": 0.0, "description": "Probability per potential edge connection per step to toggle its existence (add if missing, remove if present) *after* main edge logic.", "parameter_group": "Perturbations", "editor_sort_key": 30 },

        # === Visualization: Nodes === (Group Sort Key: 5)
        # --- MODIFIED: Update defaults ---
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (connection count).", "default": True, "parameter_group": "Visualization: Nodes", "editor_sort_key": 10},
        "color_nodes_by_degree": { "type": bool, "description": "If Use State Coloring is True, color nodes based on connection count (degree) in the current step.", "default": True, "parameter_group": "Visualization: Nodes", "editor_sort_key": 15}, # Added sort key
        "color_nodes_by_active_neighbors": { "type": bool, "description": "If Use State Coloring is True, color nodes based on active neighbor count in the previous step.", "default": False, "parameter_group": "Visualization: Nodes", "editor_sort_key": 17}, # Added sort key
        "node_colormap": { "type": str, "description": "Colormap for node states (degrees).", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps, "editor_sort_key": 20},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization.", "default": 0.0, "parameter_group": "Visualization: Nodes", "editor_sort_key": 30},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization.", "default": 8.0, "parameter_group": "Visualization: Nodes", "editor_sort_key": 40}, # Default for degree
        # --- END MODIFIED ---

        # === Visualization: Edges === (Group Sort Key: 6)
        # --- MODIFIED: Update defaults ---
        "use_state_coloring_edges": { "type": bool, "description": "Enable edge coloring based on selected mode (overrides simple binary).", "default": True, "parameter_group": "Visualization: Edges", "editor_sort_key": 10},
        "edge_coloring_mode": { 'type': str, 'default': 'DegreeSum', 'allowed_values': ['Default', 'ActiveNeighbors', 'DegreeSum'], 'description': "Edge Color: 'Default' (binary 0/1), 'ActiveNeighbors' (avg active neighbors of endpoints), 'DegreeSum' (sum of endpoint degrees). Uses prev step data.", "parameter_group": "Visualization: Edges", "editor_sort_key": 20 },
        "edge_colormap": { "type": str, "description": "Colormap for edge coloring (if enabled & not Default).", "default": "prism", "parameter_group": "Visualization: Edges", "allowed_values": ["(None)"] + _standard_colormaps, "editor_sort_key": 30},
        "edge_color_norm_vmin": { "type": float, "description": "Min value for edge color normalization (for ActiveNeighbors/DegreeSum modes).", "default": 0.0, "parameter_group": "Visualization: Edges", "editor_sort_key": 40},
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization (e.g., 16 for DegreeSum in Moore 2D).", "default": 16.0, "parameter_group": "Visualization: Edges", "editor_sort_key": 50}, # Default for DegreeSum
        # --- END MODIFIED ---

        # === Visualization Overrides === (Group Sort Key: 7) - Keep these as they are not permutations
        'use_rule_specific_colors': {
            "type": bool, "default": False,
            "description": "Check this to use the specific color/colormap settings defined below for this rule, overriding the global color theme.",
            "parameter_group": "Visualization Overrides", "editor_sort_key": 0
        },
        'rule_background_color': {
            "type": str, "default": None,
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
        # === Other === (Group Sort Key: 8) - Tiebreaker removed
    }

    def __init__(self, metadata: 'RuleMetadata'):
        """Initialize the RealmOfLaceUnified rule.
           (Round 17: Set new visualization defaults in self._params)"""
        # Update metadata before calling parent init
        metadata.name = "Realm of Lace Unified"
        # Updated description to include new metrics and dual eligibility
        metadata.description = "Unified ROL variant. Node state=degree. Eligibility uses configurable metrics (Degree, Clustering, Betweenness, Active Neigh Count, Symmetry, Variance) and aggregation (Sum, Average) independently for birth/survival, checked against counts OR ranges. Optional perturbations."
        metadata.category = "Realm of Lace Unified"
        metadata.tags = ["Connectivity", "ROL", "Degree", "Metric", "Clustering", "Betweenness", "Active Neighbors", "Symmetry", "Variance", "Configurable", "Eligibility", "Dynamic", "Unified"] # Added new tags

        # Call the base Rule's __init__
        super().__init__(metadata)
        self.name = "Realm of Lace Unified" # Ensure name is set correctly

        # --- Initialize flags directly ---
        self.requires_post_edge_state_update = True
        # Needs degrees for DEGREE, CLUSTERING, BETWEENNESS, SYMMETRY_DEGREE, VARIANCE, STDDEV
        self.needs_neighbor_degrees = True
        self.needs_neighbor_active_counts = True # Needed for ACTIVE_NEIGHBOR_COUNT metric and edge coloring modes
        self.skip_standard_tiebreakers = True # Tiebreakers not used
        self.sets_states_to_degree = True
        # ---

        # --- MODIFIED: Ensure NEW visualization defaults are set in self._params ---
        # Use setdefault to avoid overwriting values loaded from JSON/preset
        self._params.setdefault('use_state_coloring', True)
        self._params.setdefault('color_nodes_by_degree', True)
        self._params.setdefault('color_nodes_by_active_neighbors', False) # Keep this False by default
        self._params.setdefault('node_colormap', 'prism')
        self._params.setdefault('node_color_norm_vmin', 0.0)
        self._params.setdefault('node_color_norm_vmax', 8.0) # Default for degree coloring
        self._params.setdefault('use_state_coloring_edges', True)
        self._params.setdefault('edge_coloring_mode', 'DegreeSum')
        self._params.setdefault('edge_colormap', 'prism')
        self._params.setdefault('edge_color_norm_vmin', 0.0)
        self._params.setdefault('edge_color_norm_vmax', 16.0) # Default for DegreeSum coloring
        # --- END MODIFIED ---

        # Ensure other parameters have defaults in _params if not already set by base init
        self._params.setdefault('birth_metric_type', 'DEGREE')
        self._params.setdefault('birth_metric_aggregation', 'SUM')
        self._params.setdefault('survival_metric_type', 'DEGREE')
        self._params.setdefault('survival_metric_aggregation', 'SUM')
        # Defaults for dynamic params are handled by get_dynamic_parameter_metadata
        self._params.setdefault('clustering_denominator_type', 'ACTUAL')
        self._params.setdefault('perturbation_enable', False)
        self._params.setdefault('random_state_flip_probability', 0.0)
        self._params.setdefault('random_edge_toggle_probability', 0.0)
        # Ensure final death parameters have defaults
        self._params.setdefault('final_check_metric', 'DEGREE')
        # Defaults for dynamic final params handled by get_dynamic_parameter_metadata
        # Ensure visualization override defaults are set
        self._params.setdefault('use_rule_specific_colors', False)
        self._params.setdefault('rule_background_color', None)
        self._params.setdefault('rule_node_base_color', None)
        self._params.setdefault('rule_node_color', None)
        self._params.setdefault('rule_new_node_color', None)
        self._params.setdefault('rule_default_edge_color', None)
        self._params.setdefault('rule_new_edge_color', None)

    # ──────────────────────────────────────────────────────────────────────
    #   Grid Initialization
    # ──────────────────────────────────────────────────────────────────────

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        Relies on external InitialConditionManager based on 'initial_conditions' param.
        """
        logger = logging.getLogger(__name__)
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here.

    # ──────────────────────────────────────────────────────────────────────
    #   Helpers
    # ──────────────────────────────────────────────────────────────────────

    # --- Description Templates (Keep these near the top, after PARAMETER_METADATA) ---
    _DESC_TEMPLATE_RANGE = """
    Node {PURPOSE} Eligibility (Ranges - {METRIC}/{AGG_STR}):
    - Value Type: List of number pairs `[[min, max], (min, max), ...]`. Default: {DEFAULT}
    - Format: Outer list uses `[]`. Inner pairs can use `[]` or `()`.
    - Comparison: The calculated metric value (Float) is compared **directly (unrounded)** against these ranges.
    - Value Ranges (Metric): {RANGES}
    - Explanation: Metric is the {METRIC_EXPLANATION}. Use this parameter to define continuous intervals where {PURPOSE_LOWER} is allowed. Float precision matters.
    - Definition: {NODE_TYPE_CAP} node is eligible if its **unrounded** metric value falls within ANY [min, max] range in this list. Checked *in addition* to Specific Values.
    - Example: {EXAMPLE}
    """.strip()

    _DESC_TEMPLATE_VALUES = """
    Node {PURPOSE} Eligibility (Specific Values - {METRIC}/{AGG_STR}):
    - Value Type: List of Numbers `[n1, n2, ...]`, Integers or Floats. Default: {DEFAULT}
    - Format: A flat list `[...]` containing numbers.
    - Comparison Logic:
        - Integer Target (e.g., `3`): Calculated metric (Float) is **rounded** to the nearest integer. Eligible if rounded result equals target integer.
        - Float Target (e.g., `4.5`): Eligible if **unrounded** calculated metric (Float) is within +/- 0.005 of the target float.
    - Value Ranges (Metric): {RANGES}
    - Explanation: Metric is the {METRIC_EXPLANATION}. Use this to specify individual values. Integers match rounded values (like B/S counts). Floats match values very close to the specified float.
    - Definition: {NODE_TYPE_CAP} node is eligible if its metric value matches ANY value in this list (using logic above). Checked *in addition* to Ranges.
    - Example: {EXAMPLE}
        """.strip()

    _DESC_TEMPLATE_FINAL_RANGE = """
    Final State {PURPOSE} (Range - {METRIC}):
    - Value Type: List of number pairs `[[min, max], (min, max), ...]`. Default: {DEFAULT}
    - Format: Outer list uses `[]`. Inner pairs can use `[]` or `()`.
    - Comparison: The calculated final metric value (Type: {VALUE_TYPE_STR}) is compared **directly (unrounded)** against these ranges.
    - Value Ranges (Metric): {RANGES}
    - Explanation: Metric is the {METRIC_EXPLANATION}. Use this parameter to define continuous intervals where the node's final state {PURPOSE_ACTION}.
    - Definition: Node {PURPOSE_LOWER}s if its **unrounded** final metric value falls in ANY range ({PRIORITY}). Checked *in addition* to Specific Values.
    - Example: {EXAMPLE}
        """.strip()

    _DESC_TEMPLATE_FINAL_VALUES = """
    Final State {PURPOSE} (Specific Values - {METRIC}):
    - Value Type: List of Numbers `[n1, n2, ...]`, Integers or Floats. Default: {DEFAULT}
    - Format: A flat list `[...]` containing numbers.
    - Comparison Logic:
        - Integer Target (e.g., `3`): Calculated final metric (Type: {VALUE_TYPE_STR}) is **rounded** to the nearest integer. {PURPOSE_CAP} occurs if rounded result equals target integer.
        - Float Target (e.g., `0.1`): {PURPOSE_CAP} occurs if **unrounded** calculated final metric (Type: {VALUE_TYPE_STR}) is within +/- 0.005 of the target float.
    - Value Ranges (Metric): {RANGES}
    - Explanation: Metric is the {METRIC_EXPLANATION}. Use this to specify individual values that cause the node's final state to {PURPOSE_ACTION}.
    - Definition: Node {PURPOSE_LOWER}s if its final metric value matches ANY value in this list (using logic above, {PRIORITY}). Checked *in addition* to Ranges.
    - Example: {EXAMPLE}
        """.strip()
    # --- End Description Templates ---

    # --- Dynamic Metadata Generation ---
    def get_dynamic_parameter_metadata(self, param_base_name: str, metric_type: str, aggregation: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Dynamically generates metadata for ROL-U's permutation parameters.
        (Round 17: Use _get_permutation_default helper)
        """
        log_prefix = f"ROL-U.get_dynamic_metadata({param_base_name}, {metric_type}, {aggregation}): "
        logger.debug(log_prefix + "Generating dynamic metadata...")

        # --- Determine Parameter Type (Range/Values) and Purpose (Birth/Survival/Life/Death) ---
        parts = param_base_name.split('_')
        purpose_indicator = parts[0] # 'birth', 'survival', 'final'
        target_indicator = parts[1] if purpose_indicator == 'final' else parts[0] # 'life', 'death' or 'birth', 'survival'
        type_indicator = parts[-1] # 'range' or 'values'

        # Determine full parameter name suffix
        is_final_check = purpose_indicator == 'final'
        is_non_aggregated_eligibility = metric_type in ["SYMMETRY_STATE", "SYMMETRY_DEGREE", "NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV"] and not is_final_check
        agg_str = aggregation if aggregation and not is_non_aggregated_eligibility else "N/A"
        suffix = f"_{metric_type}"
        if aggregation and not is_non_aggregated_eligibility and not is_final_check:
            suffix += f"_{aggregation}"
        full_param_name = f"{param_base_name}{suffix}"
        logger.debug(f"{log_prefix}Full Param Name: {full_param_name}, Type: {type_indicator}, Purpose: {target_indicator}")

        # --- Select Template and Base Info ---
        template = None
        base_info_key = ""
        if type_indicator == 'range':
            if is_final_check: template = self._DESC_TEMPLATE_FINAL_RANGE; base_info_key = f"{purpose_indicator}_{target_indicator}_metric_range_BASE"
            else: template = self._DESC_TEMPLATE_RANGE; base_info_key = f"{target_indicator}_eligibility_range_BASE"
        elif type_indicator == 'values':
            if is_final_check: template = self._DESC_TEMPLATE_FINAL_VALUES; base_info_key = f"{purpose_indicator}_{target_indicator}_metric_values_BASE"
            else: template = self._DESC_TEMPLATE_VALUES; base_info_key = f"{target_indicator}_eligibility_values_BASE"

        if template is None:
            logger.warning(f"{log_prefix}No description template found for base '{param_base_name}' and type '{type_indicator}'.")
            return None

        base_info = copy.deepcopy(self.PARAMETER_METADATA.get(base_info_key, {}))
        if not base_info: logger.warning(f"{log_prefix}No base info found for key '{base_info_key}'.")

        # --- Gather Dynamic Content using Helpers ---
        try:
            range_str = self._calculate_range_string(metric_type, aggregation if not is_final_check else None, is_final_check)
            metric_explanation = self._get_metric_explanation(metric_type, aggregation if not is_final_check else None, is_final_check)
            # --- MODIFIED: Use helper for default ---
            default_value = self._get_permutation_default(full_param_name)
            # ---
            example = self._get_param_example(full_param_name)
            purpose_str = target_indicator.capitalize()
            purpose_lower_str = target_indicator.lower()
            node_type_cap_str = "Active" if target_indicator == "survival" else "Inactive"
            value_type_str = "Float" if is_final_check and metric_type in ["CLUSTERING", "BETWEENNESS"] else "Integer"
            purpose_action = "becomes 0 (death)" if target_indicator == "death" else "becomes its final degree"
            priority = "overrides death conditions" if target_indicator == "life" else "unless life condition met"

            # --- Format Description ---
            formatted_description = template.format(
                PURPOSE=purpose_str,
                METRIC=metric_type,
                AGG_STR=agg_str,
                DEFAULT=default_value, # Use the fetched default value
                RANGES=range_str,
                METRIC_EXPLANATION=metric_explanation,
                PURPOSE_LOWER=purpose_lower_str,
                NODE_TYPE_CAP=node_type_cap_str,
                EXAMPLE=example,
                VALUE_TYPE_STR=value_type_str,
                PURPOSE_ACTION=purpose_action,
                PURPOSE_CAP=purpose_str,
                PRIORITY=priority
            )
        except Exception as e:
            logger.error(f"{log_prefix}Error formatting description for {full_param_name}: {e}")
            formatted_description = f"Error generating description for {full_param_name}."

        # --- Construct Final Metadata ---
        dynamic_meta = {
            'type': list,
            'default': default_value, # Use fetched default
            'description': formatted_description,
            'parameter_group': base_info.get('parameter_group', "Unknown Group"),
            'editor_sort_key': base_info.get('editor_sort_key', 999)
        }
        if type_indicator == 'range': dynamic_meta['element_type'] = tuple
        elif type_indicator == 'values': dynamic_meta['element_type'] = object

        logger.debug(f"{log_prefix}Generated metadata for {full_param_name}.")
        return dynamic_meta
    
    # --- Helper Methods for Dynamic Metadata ---
    def _calculate_range_string(self, metric_type: str, aggregation: Optional[str], is_final_check: bool) -> str:
        """
        Calculates the theoretical value range string for descriptions.
        (Round 37: Improved ranges for Symmetry metrics)
        """
        ranges = {}
        neighborhoods = {
            "M2D": (8, "Moore 2D"), "M3D": (26, "Moore 3D"),
            "VN2D": (4, "VN 2D"), "VN3D": (6, "VN 3D"),
            "H2D": (6, "Hex 2D"), "HP3D": (12, "Hex Prism 3D")
        }
        # Note: Hex/HexPrism symmetry calculation is not implemented, range is theoretical.

        for code, (maxn, _) in neighborhoods.items():
            min_val: Union[int, float] = 0.0 # Default min to float 0.0
            max_val: Union[int, float] = 0.0
            is_float = True # Most metrics result in float

            if is_final_check:
                # Final check metrics are calculated on the node itself
                if metric_type == "DEGREE": min_val=0; max_val = maxn; is_float = False
                elif metric_type == "ACTIVE_NEIGHBOR_COUNT": min_val=0; max_val = maxn; is_float = False
                elif metric_type in ["CLUSTERING", "BETWEENNESS"]: min_val=0.0; max_val=1.0; is_float=True
                else: min_val=0; max_val=0; is_float=False # Unknown final metric
            else: # Eligibility check (based on neighbors)
                if metric_type == "DEGREE":
                    if aggregation == "SUM": max_val = maxn * maxn
                    else: max_val = float(maxn) # AVG
                elif metric_type == "CLUSTERING":
                    if aggregation == "SUM": max_val = float(maxn) # Sum of values in [0,1]
                    else: max_val = 1.0 # AVG
                elif metric_type == "BETWEENNESS":
                    if aggregation == "SUM": max_val = float(maxn) # Sum of values in [0,1]
                    else: max_val = 1.0 # AVG
                elif metric_type == "ACTIVE_NEIGHBOR_COUNT":
                    if aggregation == "SUM": max_val = maxn * maxn
                    else: max_val = float(maxn) # AVG
                elif metric_type == "SYMMETRY_STATE":
                    # Avg diff of states (0 or 1). Max diff is 1. Avg is also [0,1].
                    min_val = 0.0; max_val = 1.0
                elif metric_type == "SYMMETRY_DEGREE":
                    # Avg diff of degrees (0 to maxn). Max diff is maxn. Avg is [0, maxn].
                    min_val = 0.0; max_val = float(maxn)
                elif metric_type in ["NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV"]:
                    # Non-negative, theoretically unbounded but practically limited
                    min_val = 0.0; max_val = float('inf')
                else: min_val=0.0; max_val=0.0 # Unknown eligibility metric

            # Format the range string
            if max_val == float('inf'):
                ranges[code] = f">= {min_val:.1f}"
            elif is_float:
                ranges[code] = f"[{min_val:.1f}, {max_val:.1f}]"
            else: # Integer range
                ranges[code] = f"[{int(min_val)}, {int(max_val)}]"

        return ", ".join([f"{k}={v}" for k, v in ranges.items()])

    def _get_metric_explanation(self, metric_type: str, aggregation: Optional[str], is_final_check: bool) -> str:
        """
        Provides a brief explanation of the metric.
        (Round 37: Improved explanations for Symmetry metrics)
        """
        # Define explanations directly here
        metric_meanings = {
            "DEGREE": "connection count (degree)",
            "CLUSTERING": "clustering proxy (local density)",
            "BETWEENNESS": "betweenness proxy (centrality)",
            "ACTIVE_NEIGHBOR_COUNT": "active neighbor count (eligibility proxy > 0.5)",
            "SYMMETRY_STATE": "state symmetry (avg abs difference of opposing neighbor states)",
            "SYMMETRY_DEGREE": "degree symmetry (avg abs difference of opposing neighbor degrees)",
            "NEIGHBOR_DEGREE_VARIANCE": "neighbor degree variance",
            "NEIGHBOR_DEGREE_STDDEV": "neighbor degree standard deviation"
        }
        base_explanation = metric_meanings.get(metric_type, f"unknown metric '{metric_type}'")
        calculation_basis = "(previous step)" if not is_final_check else "(current step)"

        if is_final_check:
            # For final checks, the metric applies directly to the node itself
            return f"Final node {base_explanation} {calculation_basis}"
        else: # Eligibility check (based on neighbors)
            is_non_aggregated_eligibility = metric_type in [
                "SYMMETRY_STATE", "SYMMETRY_DEGREE",
                "NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV"
            ]
            if aggregation and not is_non_aggregated_eligibility:
                # Aggregated neighbor metric
                return f"{aggregation.capitalize()} of neighbor {base_explanation} {calculation_basis}"
            else:
                # Non-aggregated metric (Symmetry, Variance, StdDev)
                # These are calculated for the node based on its neighborhood
                return f"Node's {base_explanation} {calculation_basis}"
                   
    @staticmethod
    def _get_permutation_default(full_param_name: str) -> Any:
        """
        Gets the default value for a specific permutation parameter.
        Ensures lists are returned as actual lists, not strings.
        (Round 6: Made static)
        """
        # Define defaults here. Crucially, return actual lists, not strings.
        defaults = {
            "birth_eligibility_range_DEGREE_SUM": [(3.0, 5.0), (10.0, 14.0)],
            "birth_eligibility_values_DEGREE_SUM": [],
            "birth_eligibility_range_DEGREE_AVG": [(2.5, 3.8)],
            "birth_eligibility_values_DEGREE_AVG": [],
            "birth_eligibility_range_CLUSTERING_SUM": [(1.5, 3.5)],
            "birth_eligibility_values_CLUSTERING_SUM": [],
            "birth_eligibility_range_CLUSTERING_AVG": [(0.2, 0.6)],
            "birth_eligibility_values_CLUSTERING_AVG": [],
            "birth_eligibility_range_BETWEENNESS_SUM": [(0.5, 2.5)],
            "birth_eligibility_values_BETWEENNESS_SUM": [],
            "birth_eligibility_range_BETWEENNESS_AVG": [(0.1, 0.4)],
            "birth_eligibility_values_BETWEENNESS_AVG": [],
            "birth_eligibility_range_ACTIVE_NEIGHBOR_COUNT_SUM": [(8.0, 16.0)],
            "birth_eligibility_values_ACTIVE_NEIGHBOR_COUNT_SUM": [],
            "birth_eligibility_range_ACTIVE_NEIGHBOR_COUNT_AVG": [(1.5, 3.5)],
            "birth_eligibility_values_ACTIVE_NEIGHBOR_COUNT_AVG": [],
            "birth_eligibility_range_SYMMETRY_STATE": [(0.0, 0.2)],
            "birth_eligibility_values_SYMMETRY_STATE": [],
            "birth_eligibility_range_SYMMETRY_DEGREE": [(0.0, 1.5)],
            "birth_eligibility_values_SYMMETRY_DEGREE": [],
            "birth_eligibility_range_NEIGHBOR_DEGREE_VARIANCE": [(0.5, 2.5)],
            "birth_eligibility_values_NEIGHBOR_DEGREE_VARIANCE": [],
            "birth_eligibility_range_NEIGHBOR_DEGREE_STDDEV": [(0.5, 1.5)],
            "birth_eligibility_values_NEIGHBOR_DEGREE_STDDEV": [],

            "survival_eligibility_range_DEGREE_SUM": [(2.0, 4.0), (8.0, 12.0)],
            "survival_eligibility_values_DEGREE_SUM": [],
            "survival_eligibility_range_DEGREE_AVG": [(1.8, 3.2)],
            "survival_eligibility_values_DEGREE_AVG": [],
            "survival_eligibility_range_CLUSTERING_SUM": [(1.0, 4.0)],
            "survival_eligibility_values_CLUSTERING_SUM": [],
            "survival_eligibility_range_CLUSTERING_AVG": [(0.1, 0.8)],
            "survival_eligibility_values_CLUSTERING_AVG": [],
            "survival_eligibility_range_BETWEENNESS_SUM": [(0.8, 3.0)],
            "survival_eligibility_values_BETWEENNESS_SUM": [],
            "survival_eligibility_range_BETWEENNESS_AVG": [(0.05, 0.6)],
            "survival_eligibility_values_BETWEENNESS_AVG": [],
            "survival_eligibility_range_ACTIVE_NEIGHBOR_COUNT_SUM": [(4.0, 18.0)],
            "survival_eligibility_values_ACTIVE_NEIGHBOR_COUNT_SUM": [],
            "survival_eligibility_range_ACTIVE_NEIGHBOR_COUNT_AVG": [(0.8, 4.2)],
            "survival_eligibility_values_ACTIVE_NEIGHBOR_COUNT_AVG": [],
            "survival_eligibility_range_SYMMETRY_STATE": [(0.0, 0.5)],
            "survival_eligibility_values_SYMMETRY_STATE": [],
            "survival_eligibility_range_SYMMETRY_DEGREE": [(0.0, 2.5)],
            "survival_eligibility_values_SYMMETRY_DEGREE": [],
            "survival_eligibility_range_NEIGHBOR_DEGREE_VARIANCE": [(0.0, 4.0)],
            "survival_eligibility_values_NEIGHBOR_DEGREE_VARIANCE": [],
            "survival_eligibility_range_NEIGHBOR_DEGREE_STDDEV": [(0.0, 2.0)],
            "survival_eligibility_values_NEIGHBOR_DEGREE_STDDEV": [],

            "final_death_metric_range_DEGREE": [],
            "final_death_metric_values_DEGREE": [], # Default to empty list
            "final_death_metric_range_CLUSTERING": [],
            "final_death_metric_values_CLUSTERING": [],
            "final_death_metric_range_BETWEENNESS": [],
            "final_death_metric_values_BETWEENNESS": [],
            "final_death_metric_range_ACTIVE_NEIGHBOR_COUNT": [],
            "final_death_metric_values_ACTIVE_NEIGHBOR_COUNT": [],

            "final_life_metric_range_DEGREE": [],
            "final_life_metric_values_DEGREE": [],
            "final_life_metric_range_CLUSTERING": [],
            "final_life_metric_values_CLUSTERING": [],
            "final_life_metric_range_BETWEENNESS": [],
            "final_life_metric_values_BETWEENNESS": [],
            "final_life_metric_range_ACTIVE_NEIGHBOR_COUNT": [],
            "final_life_metric_values_ACTIVE_NEIGHBOR_COUNT": [],
        }
        default = defaults.get(full_param_name)
        if default is None:
             if "_range" in full_param_name or "_values" in full_param_name: return []
             else: return None
        return copy.deepcopy(default)

    def _get_param_example(self, full_param_name: str) -> str:
        """Gets an example value string for a specific permutation parameter."""
        if "range" in full_param_name: return "[[1.5, 4.1]]"
        if "precise" in full_param_name: return "[2.75, 4.0]"
        if "counts" in full_param_name: return "[2, 3]"
        if "final_death_metric_values_DEGREE" in full_param_name: return "[0, 1, 8]"
        if "final_life_metric_values_DEGREE" in full_param_name: return "[3, 4]"
        return "[]"

    def _get_param_definition(self, param_base_name: str) -> str:
        """Gets the definition part of the description."""
        if param_base_name.startswith("birth_"): return "Inactive node is eligible for birth if its"
        if param_base_name.startswith("survival_"): return "Active node is eligible for survival if its"
        if param_base_name.startswith("final_life_"): return "Node survives (state becomes final degree) if its"
        if param_base_name.startswith("final_death_"): return "Node state becomes 0 (death) if its"
        return "Node is eligible if its"

    def _get_max_neighbors_for_rule(self) -> int:
        """Helper to get the theoretical maximum neighbors based on rule parameters."""
        try:
            dimension_str = self.get_param("dimension_type", "TWO_D")
            neighborhood_str = self.get_param("neighborhood_type", "MOORE")
            dimension = Dimension[dimension_str]
            neighborhood = NeighborhoodType[neighborhood_str]

            if neighborhood == NeighborhoodType.VON_NEUMANN: return 6 if dimension == Dimension.THREE_D else 4
            elif neighborhood == NeighborhoodType.MOORE: return 26 if dimension == Dimension.THREE_D else 8
            elif neighborhood == NeighborhoodType.HEX: return 6
            elif neighborhood == NeighborhoodType.HEX_PRISM: return 12
            else: logger.warning(f"Unknown neighborhood type '{neighborhood_str}', returning 8."); return 8
        except KeyError as e: logger.error(f"Invalid dimension/neighborhood string: {e}. Returning 8."); return 8
        except Exception as e: logger.error(f"Error getting max neighbors: {e}. Returning 8."); return 8

    def _check_sum_ranges(self, value: float, ranges: List[Tuple[float, float]]) -> bool:
        """Checks if a value falls within any of the specified (min_sum, max_sum) ranges."""
        if not ranges: return False
        valid_ranges = [tuple(r) for r in ranges if isinstance(r, (list, tuple)) and len(r) == 2]
        return any(min_val <= value <= max_val for min_val, max_val in valid_ranges)

    @timer_decorator
    def _compute_all_neighbour_metrics(self, neighborhood: 'NeighborhoodData') -> Dict[str, float]:
        """
        Calculates ONLY the required aggregated metrics for the neighborhood based on
        the rule's configuration (birth/survival metric type & aggregation) and
        previous step data. Handles different clustering denominator types and new metrics.
        Returns a dictionary containing keys for the required metrics.
        (Round 14: Added entry/exit logging and input data logging)
        """
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index
        log_prefix = f"Node {node_idx} _compute_all_neighbour_metrics (R14 Log): " # Updated round

        # --- ADDED: Entry Logging ---
        if detailed_logging_enabled:
            logger.detail(f"{log_prefix}--- ENTRY ---") # type: ignore[attr-defined]
            logger.detail(f"  Received NeighborhoodData for Node {node_idx}") # type: ignore[attr-defined]
            logger.detail(f"  neighbor_indices: {neighborhood.neighbor_indices}") # type: ignore[attr-defined]
            logger.detail(f"  neighbor_states: {neighborhood.neighbor_states}") # type: ignore[attr-defined]
            logger.detail(f"  neighbor_degrees available: {neighborhood.neighbor_degrees is not None}") # type: ignore[attr-defined]
            if neighborhood.neighbor_degrees is not None:
                 logger.detail(f"    neighbor_degrees: {neighborhood.neighbor_degrees}") # type: ignore[attr-defined]
            logger.detail(f"  neighbor_active_counts available: {neighborhood.neighbor_active_counts is not None}") # type: ignore[attr-defined]
            if neighborhood.neighbor_active_counts is not None:
                 logger.detail(f"    neighbor_active_counts: {neighborhood.neighbor_active_counts}") # type: ignore[attr-defined]
            # Log SHM arrays passed via rule_params if they exist
            all_neighbors_shm = neighborhood.rule_params.get('_all_neighbor_indices_shm')
            prev_states_shm = neighborhood.rule_params.get('_previous_node_states')
            prev_degrees_shm = neighborhood.rule_params.get('_previous_node_degrees')
            logger.detail(f"  SHM _all_neighbor_indices_shm available: {all_neighbors_shm is not None}") # type: ignore[attr-defined]
            logger.detail(f"  SHM _previous_node_states available: {prev_states_shm is not None}") # type: ignore[attr-defined]
            logger.detail(f"  SHM _previous_node_degrees available: {prev_degrees_shm is not None}") # type: ignore[attr-defined]
        # --- END ADDED ---

        # --- Determine required metrics ---
        birth_metric = self.get_param('birth_metric_type', 'DEGREE', neighborhood=neighborhood)
        birth_agg = self.get_param('birth_metric_aggregation', 'SUM', neighborhood=neighborhood)
        survival_metric = self.get_param('survival_metric_type', 'DEGREE', neighborhood=neighborhood)
        survival_agg = self.get_param('survival_metric_aggregation', 'SUM', neighborhood=neighborhood)

        required_metrics = set()
        non_aggregated_metrics = ["NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV",
                                  "SYMMETRY_STATE", "SYMMETRY_DEGREE"]
        if birth_metric not in non_aggregated_metrics: required_metrics.add(f"{birth_metric}_{birth_agg}")
        else: required_metrics.add(birth_metric)
        if survival_metric not in non_aggregated_metrics: required_metrics.add(f"{survival_metric}_{survival_agg}")
        else: required_metrics.add(survival_metric)

        if detailed_logging_enabled: logger.detail(f"{log_prefix}Required metrics: {required_metrics}") # type: ignore[attr-defined]
        aggregated_metrics: Dict[str, float] = {key: 0.0 for key in required_metrics}

        # --- Get neighbor data ---
        valid_neighbor_indices = [idx for idx in neighborhood.neighbor_indices if idx >= 0]
        num_valid_neighbors = len(valid_neighbor_indices)
        if detailed_logging_enabled: logger.detail(f"{log_prefix}Found {num_valid_neighbors} valid neighbors: {valid_neighbor_indices}") # type: ignore[attr-defined]

        if num_valid_neighbors == 0:
            if detailed_logging_enabled: logger.detail(f"{log_prefix}No valid neighbors found. Returning zeros.") # type: ignore[attr-defined]
            # Ensure specific metrics are zeroed if required
            if "NEIGHBOR_DEGREE_VARIANCE" in required_metrics: aggregated_metrics["NEIGHBOR_DEGREE_VARIANCE"] = 0.0
            if "NEIGHBOR_DEGREE_STDDEV" in required_metrics: aggregated_metrics["NEIGHBOR_DEGREE_STDDEV"] = 0.0
            if "SYMMETRY_STATE" in required_metrics: aggregated_metrics["SYMMETRY_STATE"] = 0.0
            if "SYMMETRY_DEGREE" in required_metrics: aggregated_metrics["SYMMETRY_DEGREE"] = 0.0
            return aggregated_metrics # Return zeros if no neighbors

        degrees = neighborhood.neighbor_degrees; active_counts = neighborhood.neighbor_active_counts; neighbor_states = neighborhood.neighbor_states

        # --- Determine which raw data is needed ---
        needs_degree = any('DEGREE' in key or 'CLUSTERING' in key or 'BETWEENNESS' in key or 'SYMMETRY_DEGREE' in key or 'VARIANCE' in key or 'STDDEV' in key for key in required_metrics)
        needs_active_count = any('ACTIVE_NEIGHBOR_COUNT' in key for key in required_metrics)
        needs_state = any('SYMMETRY_STATE' in key for key in required_metrics)

        if detailed_logging_enabled:
            logger.detail(f"{log_prefix}Needs Degree: {needs_degree} (Available: {degrees is not None})") # type: ignore[attr-defined]
            logger.detail(f"{log_prefix}Needs Active Count: {needs_active_count} (Available: {active_counts is not None})") # type: ignore[attr-defined]
            logger.detail(f"{log_prefix}Needs State: {needs_state} (Available: {neighbor_states is not None})") # type: ignore[attr-defined]

        if needs_degree and degrees is None: logger.warning(f"{log_prefix}neighbor_degrees missing! DEGREE/CLUSTERING/BETWEENNESS/SYMMETRY_DEGREE/VARIANCE/STDDEV metrics will use state proxy.")
        if needs_active_count and active_counts is None: logger.warning(f"{log_prefix}neighbor_active_counts missing! ACTIVE_NEIGHBOR_COUNT metric will be 0.")

        # --- Calculate theoretical max neighbors ---
        max_deg_theory = 0
        if any('CLUSTERING' in key for key in required_metrics) or any('SYMMETRY' in key for key in required_metrics):
             try:
                 rule_dim_str = self.get_param('dimension_type', 'TWO_D')
                 rule_neigh_str = self.get_param('neighborhood_type', 'MOORE')
                 rule_dim = Dimension[rule_dim_str]; rule_neigh = NeighborhoodType[rule_neigh_str]
                 max_deg_theory = calculate_max_neighbors(rule_dim, rule_neigh)
             except (KeyError, ValueError) as e: logger.error(f"{log_prefix}Error getting dimension/neighborhood for MaxN calc: {e}. Using default 8."); max_deg_theory = 8
             if detailed_logging_enabled: logger.detail(f"{log_prefix}Theoretical Max Neighbors (MaxN): {max_deg_theory}") # type: ignore[attr-defined]

        # --- Calculate individual metrics per neighbor ---
        neighbor_degree_values = []; neighbor_clustering_values = []; neighbor_betweenness_values = []
        neighbor_active_count_values = []; neighbor_state_values = []
        clustering_denom_type = self.get_param('clustering_denominator_type', 'ACTUAL', neighborhood=neighborhood)
        if detailed_logging_enabled: logger.detail(f"{log_prefix}Clustering Denominator Type: {clustering_denom_type}") # type: ignore[attr-defined]
        if detailed_logging_enabled: logger.detail(f"{log_prefix}Calculating per-neighbor metrics...") # type: ignore[attr-defined]

        for i, nbr_idx in enumerate(valid_neighbor_indices):
            # Find the original index in the padded neighbor_indices array to get the correct state
            original_indices_list = np.where(neighborhood.neighbor_indices == nbr_idx)[0]
            if not original_indices_list.size > 0:
                logger.warning(f"{log_prefix}Could not find original index for valid neighbor {nbr_idx}. Skipping metric calculation for this neighbor.")
                continue
            original_idx = original_indices_list[0] # Get the first match

            nbr_state = 0.0
            if neighbor_states is not None and original_idx < len(neighbor_states):
                 nbr_state = neighbor_states[original_idx] # Use original_idx to access neighbor_states
            else: logger.warning(f"{log_prefix}Neighbor state unavailable for index {original_idx} (neighbor {nbr_idx}).")

            if needs_state: neighbor_state_values.append(float(nbr_state))

            deg = 0
            if needs_degree:
                deg = degrees.get(nbr_idx, 0) if degrees is not None else (int(nbr_state) if nbr_state > 0 else 0)
                if any(m in key for m in ['DEGREE', 'VARIANCE', 'STDDEV', 'SYMMETRY_DEGREE'] for key in required_metrics):
                    neighbor_degree_values.append(float(deg))

            cluster_val = 0.0
            if any('CLUSTERING' in key for key in required_metrics):
                max_deg_for_calc = num_valid_neighbors if clustering_denom_type == 'ACTUAL' else max_deg_theory
                max_deg_term = float(max_deg_for_calc * (max_deg_for_calc - 1)) if max_deg_for_calc > 1 else 1.0
                if max_deg_term <= 0: max_deg_term = 1.0
                cluster_val = (deg * (deg - 1)) / max_deg_term if deg > 1 else 0.0
                neighbor_clustering_values.append(cluster_val)

            between_val = 0.0
            if any('BETWEENNESS' in key for key in required_metrics):
                between_val = 1.0 / float(deg) if deg > 0 else 0.0; neighbor_betweenness_values.append(between_val)

            active_count = 0
            if needs_active_count:
                active_count = active_counts.get(nbr_idx, 0) if active_counts is not None else 0
                neighbor_active_count_values.append(float(active_count))

            if detailed_logging_enabled: logger.detail(f"  Neighbor {nbr_idx}: State={nbr_state:.1f}, Degree={deg}, Cluster={cluster_val:.3f}, Between={between_val:.3f}, ActiveNbrs={active_count}") # type: ignore[attr-defined]

        # --- Calculate/Aggregate Required Metrics ---
        opposing_pairs = []
        needs_symmetry = any('SYMMETRY' in key for key in required_metrics)
        if needs_symmetry:
            all_neighbors_array = neighborhood.rule_params.get('_all_neighbor_indices_shm')
            if all_neighbors_array is not None:
                 opposing_pairs = _get_opposing_neighbor_indices(node_idx, neighborhood.dimensions, neighborhood.neighborhood_type, all_neighbors_array, max_deg_theory)
            else: logger.warning(f"{log_prefix}Cannot calculate symmetry, precalculated neighbor array missing.")

        for key in required_metrics:
            if key == "NEIGHBOR_DEGREE_VARIANCE": aggregated_metrics[key] = float(np.var(neighbor_degree_values)) if len(neighbor_degree_values) > 1 else 0.0; continue
            if key == "NEIGHBOR_DEGREE_STDDEV": aggregated_metrics[key] = float(np.std(neighbor_degree_values)) if len(neighbor_degree_values) > 1 else 0.0; continue
            if key == "SYMMETRY_STATE":
                prev_states = neighborhood.rule_params.get('_previous_node_states')
                if prev_states is not None: aggregated_metrics[key] = _calculate_symmetry_metric(opposing_pairs, prev_states, node_idx)
                else: logger.warning(f"{log_prefix}Cannot calculate SYMMETRY_STATE, previous_node_states missing."); aggregated_metrics[key] = 0.0
                continue
            if key == "SYMMETRY_DEGREE":
                prev_degrees = neighborhood.rule_params.get('_previous_node_degrees')
                if prev_degrees is not None: aggregated_metrics[key] = _calculate_symmetry_metric(opposing_pairs, prev_degrees, node_idx)
                else: logger.warning(f"{log_prefix}Cannot calculate SYMMETRY_DEGREE, previous_node_degrees missing."); aggregated_metrics[key] = 0.0
                continue

            try: metric_type_agg, agg_type = key.rsplit('_', 1)
            except ValueError: logger.error(f"{log_prefix}Could not split metric key '{key}'. Skipping."); continue

            values_to_agg = []
            if metric_type_agg == 'DEGREE': values_to_agg = neighbor_degree_values
            elif metric_type_agg == 'CLUSTERING': values_to_agg = neighbor_clustering_values
            elif metric_type_agg == 'BETWEENNESS': values_to_agg = neighbor_betweenness_values
            elif metric_type_agg == 'ACTIVE_NEIGHBOR_COUNT': values_to_agg = neighbor_active_count_values

            if values_to_agg:
                metric_sum = sum(values_to_agg)
                if agg_type == 'SUM': aggregated_metrics[key] = metric_sum
                elif agg_type == 'AVERAGE': aggregated_metrics[key] = metric_sum / num_valid_neighbors if num_valid_neighbors > 0 else 0.0

        # --- ADDED: Exit Logging ---
        if detailed_logging_enabled:
            logger.detail(f"{log_prefix}--- EXIT --- Returning Calculated Metrics:") # type: ignore[attr-defined]
            for key, val in aggregated_metrics.items(): logger.detail(f"  {key}: {val:.4f}") # type: ignore[attr-defined]
        # --- END ADDED ---

        return aggregated_metrics
    
    def validate_parameter_context(self, param_name: str, current_params: Dict[str, Any]) -> Optional[str]:
        """
        Overrides base method to check for overlapping final life/death conditions
        when a relevant parameter is changed.
        (Round 7: Corrected relevant_params definition without METRIC_DETAILS)
        """
        # Define the possible metric types explicitly
        possible_final_metrics = ["DEGREE", "CLUSTERING", "BETWEENNESS", "ACTIVE_NEIGHBOR_COUNT"]

        # Define the set of parameters that trigger this contextual check
        relevant_params = {
            f"final_life_metric_values_{m}" for m in possible_final_metrics # Use _values
        } | {
            f"final_life_metric_range_{m}" for m in possible_final_metrics
        } | {
            f"final_death_metric_values_{m}" for m in possible_final_metrics # Use _values
        } | {
            f"final_death_metric_range_{m}" for m in possible_final_metrics
        } | { # Also check when the metric itself changes
            "final_check_metric"
        }

        # Only perform the check if the changed parameter is relevant
        if param_name in relevant_params:
            # Call the helper method using the provided current parameters
            overlap_message = self._check_final_condition_overlap(current_params)
            if overlap_message:
                # Return the specific warning message
                return overlap_message

        # No overlap detected or parameter wasn't relevant
        return None
    
    def _check_final_condition_overlap(self, current_params: Dict[str, Any]) -> Optional[str]:
        """
        Checks for overlaps between the *active* final life and final death conditions
        based on the current 'final_check_metric', considering both ranges and the
        unified values list (handling int/float comparison logic).

        Args:
            current_params: A dictionary representing the parameters to check.

        Returns:
            A string describing the first detected overlap, or None if no overlap found.
        """
        metric_type = current_params.get('final_check_metric', 'DEGREE')
        metric_key_suffix = f"_{metric_type}"
        tolerance = 0.005 # Tolerance for float comparisons in the values list

        # Determine if the metric produces float values for comparison purposes
        metric_is_float = metric_type in ["CLUSTERING", "BETWEENNESS"]

        # Get the *active* life/death parameters based on the current metric
        life_ranges = current_params.get(f'final_life_metric_range{metric_key_suffix}', [])
        life_values = current_params.get(f'final_life_metric_values{metric_key_suffix}', [])
        death_ranges = current_params.get(f'final_death_metric_range{metric_key_suffix}', [])
        death_values = current_params.get(f'final_death_metric_values{metric_key_suffix}', [])

        # --- Data Cleaning and Validation ---
        try:
            # Ensure ranges are lists of valid tuples/lists with 2 numbers
            life_ranges = [(float(r[0]), float(r[1])) for r in life_ranges if isinstance(r, (list, tuple)) and len(r) == 2]
            death_ranges = [(float(r[0]), float(r[1])) for r in death_ranges if isinstance(r, (list, tuple)) and len(r) == 2]
            # Ensure values are lists of numbers (int or float)
            life_values = [v for v in life_values if isinstance(v, (int, float))]
            death_values = [v for v in death_values if isinstance(v, (int, float))]
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Error cleaning parameter lists during overlap check: {e}")
            return f"Invalid format in life/death parameter lists for metric '{metric_type}'."
        # ---

        # --- Overlap Checks ---

        # 1. Range vs. Range
        for lr_min, lr_max in life_ranges:
            for dr_min, dr_max in death_ranges:
                # Check for overlap: max(starts) <= min(ends)
                if max(lr_min, dr_min) <= min(lr_max, dr_max):
                    return f"Life Range [{lr_min}, {lr_max}] overlaps with Death Range [{dr_min}, {dr_max}] for metric '{metric_type}'."

        # 2. Values vs. Values (Interval Overlap Logic)
        for lv in life_values:
            for dv in death_values:
                overlap = False
                if isinstance(lv, int) and isinstance(dv, int):
                    # Integer vs Integer: Overlap if they are the same integer
                    if lv == dv: overlap = True
                elif isinstance(lv, float) and isinstance(dv, float):
                    # Float vs Float: Overlap if their tolerance intervals overlap
                    if abs(lv - dv) < (2 * tolerance): overlap = True
                elif isinstance(lv, int) and isinstance(dv, float):
                    # Integer vs Float: Overlap if float tolerance interval overlaps integer rounding interval
                    if abs(lv - dv) < (0.5 + tolerance): overlap = True
                elif isinstance(lv, float) and isinstance(dv, int):
                    # Float vs Integer: Overlap if float tolerance interval overlaps integer rounding interval
                    if abs(lv - dv) < (0.5 + tolerance): overlap = True

                if overlap:
                    return f"Life Value '{lv}' conflicts with Death Value '{dv}' (considering rounding/tolerance) for metric '{metric_type}'."

        # 3. Life Range vs. Death Values
        for lr_min, lr_max in life_ranges:
            for dv in death_values:
                overlap = False
                if isinstance(dv, int):
                    # Check if integer falls within the life range
                    if lr_min <= dv <= lr_max: overlap = True
                elif isinstance(dv, float):
                    # Check if the float's tolerance interval overlaps the life range
                    if max(lr_min, dv - tolerance) < min(lr_max, dv + tolerance): overlap = True

                if overlap:
                    return f"Life Range [{lr_min}, {lr_max}] overlaps with Death Value '{dv}' (considering rounding/tolerance) for metric '{metric_type}'."

        # 4. Death Range vs. Life Values
        for dr_min, dr_max in death_ranges:
            for lv in life_values:
                overlap = False
                if isinstance(lv, int):
                    # Check if integer falls within the death range
                    if dr_min <= lv <= dr_max: overlap = True
                elif isinstance(lv, float):
                    # Check if the float's tolerance interval overlaps the death range
                    if max(dr_min, lv - tolerance) < min(dr_max, lv + tolerance): overlap = True

                if overlap:
                    return f"Death Range [{dr_min}, {dr_max}] overlaps with Life Value '{lv}' (considering rounding/tolerance) for metric '{metric_type}'."

        return None # No overlap detected
    
    @staticmethod
    def _get_current_parameter_metadata() -> Dict[str, Dict[str, Any]]:
        """Statically gets the merged and filtered parameter metadata for the current class."""
        # Ensure base metadata is populated if it hasn't been already
        if not Rule.PARAMETER_METADATA:
            Rule._populate_base_metadata()

        base_meta = getattr(Rule, 'PARAMETER_METADATA', {})
        sub_meta = getattr(RealmOfLaceUnified, 'PARAMETER_METADATA', {})
        # Replicate merging logic statically
        merged = copy.deepcopy(base_meta)
        for param_name, sub_info in sub_meta.items():
            if param_name in merged:
                base_info = merged[param_name]
                if 'type' in sub_info and 'type' in base_info and sub_info['type'] != base_info['type']:
                    base_info['type'] = sub_info['type'] # Prefer subclass type
                # Include editor_sort_key in the keys to override
                for key in ['default', 'allowed_values', 'min', 'max', 'parameter_group', 'description', 'element_type', 'editor_sort_key']:
                    if key in sub_info:
                        base_info[key] = sub_info[key]
                merged[param_name] = base_info
            else:
                merged[param_name] = copy.deepcopy(sub_info)
        # Filter excluded
        exclude_set = getattr(RealmOfLaceUnified, 'EXCLUDE_EDITOR_PARAMS', set())
        final_meta = {k: v for k, v in merged.items() if k not in exclude_set}
        return final_meta

    @staticmethod
    def _get_expected_static_params() -> Set[str]:
         """Statically gets the expected static parameter names."""
         all_meta = RealmOfLaceUnified._get_current_parameter_metadata()
         static_params = set()
         for name in all_meta:
             # Exclude templates and dynamically generated parameter bases
             if name.endswith('_BASE'): continue
             if not (name.startswith("birth_eligibility_") or \
                     name.startswith("survival_eligibility_") or \
                     name.startswith("final_death_metric_") or \
                     name.startswith("final_life_metric_")):
                 static_params.add(name)
         return static_params

    @staticmethod
    def _get_expected_dynamic_param_names(birth_metric, birth_agg, survival_metric, survival_agg, final_metric) -> Set[str]:
         """Statically constructs expected dynamic names based on selectors."""
         names = set()
         is_non_agg_birth = birth_metric in ["SYMMETRY_STATE", "SYMMETRY_DEGREE", "NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV"]
         is_non_agg_survival = survival_metric in ["SYMMETRY_STATE", "SYMMETRY_DEGREE", "NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV"]

         # Adjust aggregation to None if metric doesn't use it
         birth_agg_used = birth_agg if not is_non_agg_birth else None
         survival_agg_used = survival_agg if not is_non_agg_survival else None

         birth_suffix = f"_{birth_metric}" + (f"_{birth_agg_used}" if birth_agg_used else "")
         survival_suffix = f"_{survival_metric}" + (f"_{survival_agg_used}" if survival_agg_used else "")
         final_suffix = f"_{final_metric}"

         names.add(f"birth_eligibility_range{birth_suffix}")
         names.add(f"birth_eligibility_values{birth_suffix}")
         names.add(f"survival_eligibility_range{survival_suffix}")
         names.add(f"survival_eligibility_values{survival_suffix}")
         names.add(f"final_life_metric_range{final_suffix}")
         names.add(f"final_life_metric_values{final_suffix}")
         names.add(f"final_death_metric_range{final_suffix}")
         names.add(f"final_death_metric_values{final_suffix}")
         return names

    @staticmethod
    def _parse_dynamic_param_name(param_name: str) -> Tuple[str, str, Optional[str]]:
        """
        Robustly parses base, metric, agg from a dynamic name.
        Handles names with or without aggregation suffixes, and metrics with underscores.
        Accepts both AVG and AVERAGE as aggregation suffixes.
        """
        logger = logging.getLogger(__name__)
        log_prefix = "ROL-U._parse_dynamic_param_name (Final rsplit): "

        known_metrics = {
            "DEGREE", "CLUSTERING", "BETWEENNESS", "ACTIVE_NEIGHBOR_COUNT",
            "SYMMETRY_STATE", "SYMMETRY_DEGREE", "NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV"
        }
        known_aggregations = {"SUM", "AVG", "AVERAGE"}
        agg_normalization = {"AVERAGE": "AVG", "AVG": "AVG", "SUM": "SUM"}

        base_prefixes = sorted([
            "birth_eligibility_range", "birth_eligibility_values",
            "survival_eligibility_range", "survival_eligibility_values",
            "final_life_metric_range", "final_life_metric_values",
            "final_death_metric_range", "final_death_metric_values"
        ], key=len, reverse=True)

        base = param_name
        metric = "UNKNOWN"
        agg: Optional[str] = None

        matched_base = None
        for prefix in base_prefixes:
            if param_name.startswith(prefix + '_'):
                matched_base = prefix
                break

        if matched_base:
            base = matched_base
            remainder = param_name[len(matched_base) + 1:]
            # Split from the right on the last underscore
            if '_' in remainder:
                metric_candidate, agg_candidate = remainder.rsplit('_', 1)
                if agg_candidate in known_aggregations:
                    agg = agg_normalization[agg_candidate]
                    metric = metric_candidate
                else:
                    metric = remainder
                    agg = None
            else:
                metric = remainder
                agg = None

            if metric not in known_metrics:
                logger.warning(f"{log_prefix}Parsed metric '{metric}' from '{param_name}' is NOT in known_metrics {known_metrics}. Setting to UNKNOWN.")
                metric = "UNKNOWN"
                agg = None
        else:
            logger.warning(f"{log_prefix}Could not reliably parse metric from dynamic param name: {param_name}")

        logger.debug(f"{log_prefix}Parsed '{param_name}' -> Base='{base}', Metric='{metric}', Agg='{agg}'")
        return base, metric or "UNKNOWN", agg

    @staticmethod
    def _get_default_params() -> Dict[str, Any]:
        """Statically gets current default parameters for ROL-U."""
        defaults = {}
        try:
            all_meta = RealmOfLaceUnified._get_current_parameter_metadata()
            for name, info in all_meta.items():
                if 'default' in info:
                    # Ensure list-of-lists defaults are converted to list-of-tuples for ranges
                    default_val = info['default']
                    if name.endswith("_range") and isinstance(default_val, list) and info.get('element_type') == tuple:
                         try: defaults[name] = [tuple(item) if isinstance(item, list) else item for item in default_val]
                         except TypeError: defaults[name] = default_val # Fallback if conversion fails
                    else: defaults[name] = default_val
        except Exception as e:
            logger.error(f"Error getting default params for ROL-U: {e}")
        return defaults

    @staticmethod
    def migrate_params_vX_to_vY(params_from_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrates parameters from older ROL-U formats to the current format.
        Ensures all *expected* dynamic parameters for the current selectors
        are present in the output, using defaults if not migrated.
        """
        migrated_params = {}
        processed_old_keys = set()

        # 1. Get current selectors from the JSON data (use defaults if missing)
        birth_metric = params_from_json.get('birth_metric_type', 'DEGREE')
        birth_agg = params_from_json.get('birth_metric_aggregation', 'SUM')
        survival_metric = params_from_json.get('survival_metric_type', 'DEGREE')
        survival_agg = params_from_json.get('survival_metric_aggregation', 'SUM')
        final_metric = params_from_json.get('final_check_metric', 'DEGREE')

        # 2. Copy known static parameters directly
        expected_static = RealmOfLaceUnified._get_expected_static_params()
        for name in expected_static:
            if name in params_from_json:
                migrated_params[name] = params_from_json[name]
                processed_old_keys.add(name)

        # 3. Identify and store OLD dynamic/generic parameters
        old_dynamic_params = {}
        old_generic_params = {}
        patterns = [
            r"^(birth_eligibility_range)_([A-Z_]+)(?:_([A-Z]+))?$", r"^(birth_eligibility_values)_([A-Z_]+)(?:_([A-Z]+))?$",
            r"^(survival_eligibility_range)_([A-Z_]+)(?:_([A-Z]+))?$", r"^(survival_eligibility_values)_([A-Z_]+)(?:_([A-Z]+))?$",
            r"^(final_life_metric_range)_([A-Z_]+)$", r"^(final_life_metric_values)_([A-Z_]+)$",
            r"^(final_death_metric_range)_([A-Z_]+)$", r"^(final_death_metric_values)_([A-Z_]+)$",
            r"^(birth_eligibility_range)$", r"^(survival_eligibility_range)$",
            r"^(final_death_degree_counts)$", r"^(final_death_state_range)$",
            r"^(final_life_metric_counts)_([A-Z_]+)$", r"^(final_death_metric_counts)_([A-Z_]+)$",
            r"^(final_life_metric_counts)$", r"^(final_death_metric_counts)$",
        ]
        for old_key, old_value in params_from_json.items():
            if old_key in processed_old_keys: continue
            for pattern in patterns:
                match = re.match(pattern, old_key)
                if match:
                    if pattern.endswith(")$"): old_generic_params[old_key] = old_value
                    else: old_dynamic_params[old_key] = old_value
                    processed_old_keys.add(old_key)
                    break

        # 4. Determine expected NEW dynamic parameter names
        expected_new_dynamic_names = RealmOfLaceUnified._get_expected_dynamic_param_names(
            birth_metric, birth_agg, survival_metric, survival_agg, final_metric
        )

        # 5. Map values from old to new expected parameters AND add missing defaults
        for new_name in expected_new_dynamic_names:
            if new_name in migrated_params:
                continue

            base_name, metric, agg = RealmOfLaceUnified._parse_dynamic_param_name(new_name)
            is_non_agg_eligibility = metric in ["SYMMETRY_STATE", "SYMMETRY_DEGREE", "NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV"] and not base_name.startswith("final_")
            agg_for_lookup = agg if not is_non_agg_eligibility else None

            old_specific_key = f"{base_name}_{metric}"
            if agg_for_lookup: old_specific_key += f"_{agg_for_lookup}"

            # Comprehensive mapping for all possible legacy keys
            old_generic_key = None
            if base_name == "birth_eligibility_range": old_generic_key = "birth_eligibility_range"
            elif base_name == "survival_eligibility_range": old_generic_key = "survival_eligibility_range"
            elif base_name == "final_death_metric_values" and metric == "DEGREE":
                if "final_death_metric_counts_DEGREE" in params_from_json:
                    old_specific_key = "final_death_metric_counts_DEGREE"
                elif "final_death_degree_counts" in params_from_json:
                    old_generic_key = "final_death_degree_counts"
            elif base_name == "final_death_metric_range" and metric == "DEGREE":
                if "final_death_metric_range_DEGREE" in params_from_json:
                    old_specific_key = "final_death_metric_range_DEGREE"
                elif "final_death_state_range" in params_from_json:
                    old_generic_key = "final_death_state_range"
            elif base_name == "final_life_metric_values" and metric == "DEGREE":
                if "final_life_metric_counts_DEGREE" in params_from_json:
                    old_specific_key = "final_life_metric_counts_DEGREE"
            elif base_name == "final_life_metric_range" and metric == "DEGREE":
                if "final_life_metric_range_DEGREE" in params_from_json:
                    old_specific_key = "final_life_metric_range_DEGREE"
            elif base_name == "final_death_metric_values":
                if f"final_death_metric_counts_{metric}" in params_from_json:
                    old_specific_key = f"final_death_metric_counts_{metric}"
            elif base_name == "final_death_metric_range":
                if f"final_death_metric_range_{metric}" in params_from_json:
                    old_specific_key = f"final_death_metric_range_{metric}"
            elif base_name == "final_life_metric_values":
                if f"final_life_metric_counts_{metric}" in params_from_json:
                    old_specific_key = f"final_life_metric_counts_{metric}"
            elif base_name == "final_life_metric_range":
                if f"final_life_metric_range_{metric}" in params_from_json:
                    old_specific_key = f"final_life_metric_range_{metric}"

            value_found = None; source_key = None
            if old_specific_key in params_from_json:
                value_found = params_from_json[old_specific_key]; source_key = old_specific_key
            elif old_specific_key in old_dynamic_params:
                value_found = old_dynamic_params[old_specific_key]; source_key = old_specific_key
            elif old_generic_key and old_generic_key in params_from_json:
                value_found = params_from_json[old_generic_key]; source_key = old_generic_key
            elif old_generic_key and old_generic_key in old_generic_params:
                value_found = old_generic_params[old_generic_key]; source_key = old_generic_key

            if value_found is not None:
                # Convert string lists/tuples if needed
                if base_name.endswith(("_range", "_values")):
                    if isinstance(value_found, str):
                        try:
                            converted_list = ast.literal_eval(value_found)
                            if isinstance(converted_list, list):
                                if base_name.endswith("_range"): migrated_params[new_name] = [tuple(item) if isinstance(item, list) else item for item in converted_list]
                                else: migrated_params[new_name] = converted_list
                            else: migrated_params[new_name] = value_found
                        except: migrated_params[new_name] = value_found
                    elif isinstance(value_found, list):
                         if base_name.endswith("_range"): migrated_params[new_name] = [tuple(item) if isinstance(item, list) else item for item in value_found]
                         else: migrated_params[new_name] = value_found
                    else: migrated_params[new_name] = value_found
                else: migrated_params[new_name] = value_found
            else:
                try:
                    default_val = RealmOfLaceUnified._get_permutation_default(new_name)
                    migrated_params[new_name] = default_val
                except Exception:
                    pass

        # 6. Add missing static params with defaults (if not already copied)
        current_defaults = RealmOfLaceUnified._get_default_params()
        for static_name in expected_static:
            if static_name not in migrated_params:
                if static_name in current_defaults:
                    migrated_params[static_name] = current_defaults[static_name]

        return migrated_params

    @staticmethod
    def _get_expected_dynamic_param_names_from_selectors(params: Dict[str, Any]) -> set:
        """
        Returns the set of expected dynamic parameter names for ROL-U
        based on the selector values currently present in the provided params dict.
        Always uses 'AVG' as the canonical aggregation suffix.
        """
        # Get selectors from the input params, falling back to defaults if missing
        birth_metric = params.get('birth_metric_type', 'DEGREE')
        birth_agg = params.get('birth_metric_aggregation', 'SUM')
        survival_metric = params.get('survival_metric_type', 'DEGREE')
        survival_agg = params.get('survival_metric_aggregation', 'SUM')
        final_metric = params.get('final_check_metric', 'DEGREE')

        # Normalize aggregation to 'AVG' if user/UI uses 'AVERAGE'
        def normalize_agg(agg):
            if agg == "AVERAGE":
                return "AVG"
            return agg

        birth_agg = normalize_agg(birth_agg)
        survival_agg = normalize_agg(survival_agg)

        names = set()
        is_non_agg_birth = birth_metric in ["SYMMETRY_STATE", "SYMMETRY_DEGREE", "NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV"]
        is_non_agg_survival = survival_metric in ["SYMMETRY_STATE", "SYMMETRY_DEGREE", "NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV"]

        # Adjust aggregation to None if metric doesn't use it
        birth_agg_used = birth_agg if not is_non_agg_birth else None
        survival_agg_used = survival_agg if not is_non_agg_survival else None

        birth_suffix = f"_{birth_metric}" + (f"_{birth_agg_used}" if birth_agg_used else "")
        survival_suffix = f"_{survival_metric}" + (f"_{survival_agg_used}" if survival_agg_used else "")
        final_suffix = f"_{final_metric}"

        names.add(f"birth_eligibility_range{birth_suffix}")
        names.add(f"birth_eligibility_values{birth_suffix}")
        names.add(f"survival_eligibility_range{survival_suffix}")
        names.add(f"survival_eligibility_values{survival_suffix}")
        names.add(f"final_life_metric_range{final_suffix}")
        names.add(f"final_life_metric_values{final_suffix}")
        names.add(f"final_death_metric_range{final_suffix}")
        names.add(f"final_death_metric_values{final_suffix}")
        return names

    # ──────────────────────────────────────────────────────────────────────
    #   Core Computation Logic
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    @njit(cache=True, parallel=True, fastmath=True)
    def _compute_new_state_jit(
        chunk_indices: npt.NDArray[np.int64],
        original_states_flat: npt.NDArray[np.float64],
        neighbor_indices_array: npt.NDArray[np.int64], # Shape (total_nodes, max_neighbors)
        previous_degree_array: Optional[npt.NDArray[np.int32]], # Flat array, can be None
        previous_active_neighbor_array: Optional[npt.NDArray[np.int32]], # Flat array, can be None
        # REMOVED: params_dict_for_jit: Dict[str, Any],
        grid_dimensions_arr: npt.NDArray[np.int64],
        # ADDED: Primitive parameters needed for metric calculation
        birth_metric_type_str: str,
        birth_agg_str: str,
        survival_metric_type_str: str,
        survival_agg_str: str,
        clustering_denom_type_str: str,
        max_deg_theory: int, # Pre-calculated theoretical max neighbors
        detailed_logging_enabled: bool # Keep for potential internal Numba print (limited)
    ) -> npt.NDArray[np.float64]: # Return calculated METRIC values
        """
        Numba-jitted helper to calculate the relevant eligibility METRIC for a chunk of nodes.
        Does NOT perform eligibility checks or perturbations.
        (Round 9: New JIT helper function)
        (Round 13: Refactored to accept primitive params and return metric value)
        """
        num_nodes_in_chunk = len(chunk_indices)
        # --- MODIFIED: Initialize array for metric results ---
        calculated_metrics = np.zeros(num_nodes_in_chunk, dtype=np.float64)
        # ---
        total_nodes = len(original_states_flat)
        max_neighbors = neighbor_indices_array.shape[1] # Use actual shape

        # --- Pre-calculate MaxN term for clustering ---
        max_deg_term_theory = float(max_deg_theory * (max_deg_theory - 1)) if max_deg_theory > 1 else 1.0
        if max_deg_term_theory <= 0: max_deg_term_theory = 1.0

        # --- Loop over nodes in the chunk ---
        for i in prange(num_nodes_in_chunk): # Use prange for parallel execution
            node_idx = chunk_indices[i]
            current_node_state = original_states_flat[node_idx]
            check_type = "Birth" if current_node_state <= 1e-6 else "Survival"

            # Determine which metric/aggregation to use for THIS node
            metric_type = birth_metric_type_str if check_type == "Birth" else survival_metric_type_str
            aggregation = birth_agg_str if check_type == "Birth" else survival_agg_str

            # --- Calculate Aggregated Metric ---
            metric_value_for_node = 0.0 # Initialize metric for this node
            neighbor_indices = neighbor_indices_array[node_idx]
            valid_neighbor_indices = neighbor_indices[neighbor_indices != -1]
            num_valid_neighbors = len(valid_neighbor_indices)

            if num_valid_neighbors > 0:
                # --- Extract neighbor data ---
                # Ensure indices are within bounds before accessing arrays
                valid_neighbor_indices = valid_neighbor_indices[valid_neighbor_indices < total_nodes] # Ensure indices are valid for state array
                num_valid_neighbors = len(valid_neighbor_indices) # Recalculate count after filtering

                if num_valid_neighbors > 0: # Proceed only if still valid neighbors
                    neighbor_states = original_states_flat[valid_neighbor_indices]

                    neighbor_degrees = np.zeros(num_valid_neighbors, dtype=np.int32)
                    if previous_degree_array is not None:
                        # Check bounds against the specific array being accessed
                        valid_degree_mask = valid_neighbor_indices < len(previous_degree_array)
                        valid_degree_indices = valid_neighbor_indices[valid_degree_mask]
                        if len(valid_degree_indices) > 0: # Check if any valid indices remain
                            neighbor_degrees[valid_degree_mask] = previous_degree_array[valid_degree_indices]

                    neighbor_active_counts = np.zeros(num_valid_neighbors, dtype=np.int32)
                    if previous_active_neighbor_array is not None:
                        valid_active_mask = valid_neighbor_indices < len(previous_active_neighbor_array)
                        valid_active_indices = valid_neighbor_indices[valid_active_mask]
                        if len(valid_active_indices) > 0: # Check if any valid indices remain
                            neighbor_active_counts[valid_active_mask] = previous_active_neighbor_array[valid_active_indices]

                    # --- Calculate per-neighbor metric values ---
                    values_to_aggregate = np.zeros(num_valid_neighbors, dtype=np.float64)
                    is_non_aggregated = False # Flag for metrics calculated directly

                    if metric_type == 'DEGREE':
                        values_to_aggregate = neighbor_degrees.astype(np.float64)
                    elif metric_type == 'ACTIVE_NEIGHBOR_COUNT':
                        values_to_aggregate = neighbor_active_counts.astype(np.float64)
                    elif metric_type == 'CLUSTERING':
                        max_deg_for_calc = float(num_valid_neighbors) if clustering_denom_type_str == 'ACTUAL' else float(max_deg_theory)
                        max_deg_term = max_deg_for_calc * (max_deg_for_calc - 1.0) if max_deg_for_calc > 1.0 else 1.0
                        if max_deg_term <= 0: max_deg_term = 1.0
                        for j in range(num_valid_neighbors):
                            deg = float(neighbor_degrees[j])
                            values_to_aggregate[j] = (deg * (deg - 1.0)) / max_deg_term if deg > 1.0 else 0.0
                    elif metric_type == 'BETWEENNESS':
                        for j in range(num_valid_neighbors):
                            deg = float(neighbor_degrees[j])
                            values_to_aggregate[j] = 1.0 / deg if deg > 0.0 else 0.0
                    elif metric_type == 'NEIGHBOR_DEGREE_VARIANCE':
                        if num_valid_neighbors > 1: metric_value_for_node = np.var(neighbor_degrees.astype(np.float64))
                        else: metric_value_for_node = 0.0
                        is_non_aggregated = True
                    elif metric_type == 'NEIGHBOR_DEGREE_STDDEV':
                        if num_valid_neighbors > 1: metric_value_for_node = np.std(neighbor_degrees.astype(np.float64))
                        else: metric_value_for_node = 0.0
                        is_non_aggregated = True
                    # --- SYMMETRY needs more complex logic, potentially pre-calculation ---
                    # Placeholder for SYMMETRY - returns 0 for now in JIT context
                    elif metric_type.startswith('SYMMETRY'):
                        metric_value_for_node = 0.0 # Placeholder
                        is_non_aggregated = True
                    else: # Unknown metric type
                         metric_value_for_node = 0.0
                         is_non_aggregated = True

                    # --- Aggregate values (if not already calculated) ---
                    if not is_non_aggregated:
                        if aggregation == "SUM":
                            metric_value_for_node = np.sum(values_to_aggregate)
                        elif aggregation == "AVERAGE":
                            metric_value_for_node = np.mean(values_to_aggregate)
                        else: # Default to sum if aggregation unknown (shouldn't happen)
                            metric_value_for_node = np.sum(values_to_aggregate)

            # --- Assign the calculated metric value ---
            calculated_metrics[i] = metric_value_for_node

        # --- Return the array of calculated metric values ---
        return calculated_metrics

    # @timer_decorator # Remove timer from the non-JIT version
    def _compute_new_state(self, neighborhood: 'NeighborhoodData', detailed_logging_enabled: bool) -> float:
        """
        Calculates the node's ELIGIBILITY proxy state (0 or 1).
        This method now primarily acts as a wrapper to call the JIT-compiled helper.
        It extracts necessary data from NeighborhoodData which might not be JIT-compatible.
        (Round 9: Refactored to call JIT helper - Placeholder implementation)
        """
        logger = logging.getLogger(__name__)
        node_idx = neighborhood.node_index
        log_prefix = f"Node {node_idx} _compute_new_state (Wrapper R9): "

        # --- This non-JIT version is NO LONGER THE MAIN COMPUTATION PATH ---
        # --- It might be called in sequential mode or if JIT fails ---
        # --- We need to replicate the logic here without Numba ---
        # --- OR raise an error indicating JIT should be used ---

        # --- Option 1: Replicate Logic (Less performant) ---
        # logger.warning(f"{log_prefix}Executing non-JIT version. Performance will be lower.")
        # required_metrics_dict = self._compute_all_neighbour_metrics(neighborhood)
        # current_node_state = neighborhood.node_state
        # check_type = "Birth" if current_node_state <= 1e-6 else "Survival"
        # metric_type = self.get_param(f'{check_type.lower()}_metric_type', 'DEGREE', neighborhood=neighborhood)
        # aggregation = self.get_param(f'{check_type.lower()}_metric_aggregation', 'SUM', neighborhood=neighborhood)
        # is_non_agg = metric_type in ["SYMMETRY_STATE", "SYMMETRY_DEGREE", "NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV"]
        # agg_used = aggregation if not is_non_agg else None
        # metric_key = f"{metric_type}" + (f"_{agg_used}" if agg_used else "")
        # metric_value_to_check = required_metrics_dict.get(metric_key, 0.0)
        # range_param_name = f"{check_type.lower()}_eligibility_range_{metric_key}"
        # values_param_name = f"{check_type.lower()}_eligibility_values_{metric_key}"
        # ranges_to_check = self.get_param(range_param_name, [], neighborhood=neighborhood)
        # values_to_check = self.get_param(values_param_name, [], neighborhood=neighborhood)
        # passes_check = False
        # if self._check_sum_ranges(metric_value_to_check, ranges_to_check): passes_check = True
        # elif values_to_check:
        #     tolerance = 0.005; metric_value_rounded = int(round(metric_value_to_check))
        #     for target_value in values_to_check:
        #         match = False
        #         if isinstance(target_value, int): match = (metric_value_rounded == target_value)
        #         elif isinstance(target_value, float): match = (abs(metric_value_to_check - target_value) < tolerance)
        #         if match: passes_check = True; break
        # eligibility_proxy = 1.0 if passes_check else 0.0
        # perturbation_enabled = self.get_param('perturbation_enable', False)
        # state_flip_prob = self.get_param('random_state_flip_probability', 0.0)
        # if perturbation_enabled and state_flip_prob > 0 and random.random() < state_flip_prob:
        #     eligibility_proxy = 1.0 - eligibility_proxy
        # return eligibility_proxy

        # --- Option 2: Raise Error (Force use of parallel path) ---
        logger.error(f"{log_prefix}Non-JIT _compute_new_state called. This path is not optimized. Parallel execution with JIT helper should be used.")
        # Returning the current state might be safer than erroring if sequential mode is ever needed
        return neighborhood.node_state # Return current state as fallback
    
    @staticmethod
    @njit(cache=True, parallel=True, fastmath=True)
    def _compute_new_edges_jit(
        chunk_indices: npt.NDArray[np.int64],
        eligibility_proxies_flat: npt.NDArray[np.float64],
        neighbor_indices_array: npt.NDArray[np.int64], # Shape (total_nodes, max_neighbors)
        perturbation_enable: bool,
        edge_toggle_prob: float
    # --- MODIFIED: Return a list of lists/arrays per node ---
    ) -> List[npt.NDArray[np.int64]]: # Return List[neighbor_indices_to_connect_to]
        """
        Numba-jitted helper to determine proposed edges based on mutual eligibility
        and optional perturbation. Returns a list where each element corresponds
        to a node in chunk_indices and contains an array of neighbor indices
        to which an edge should be formed/kept.
        (Round 14: New JIT helper function)
        (Round 16: Change return type and logic for safer parallel collection)
        """
        num_nodes_in_chunk = len(chunk_indices)
        total_nodes = len(eligibility_proxies_flat)
        # --- MODIFIED: Create a list to store results for each node ---
        # Numba requires explicit typing for lists containing arrays.
        # We'll use a temporary list of lists during computation and convert later if needed,
        # or use Numba's typed list if performance is critical (more complex).
        # For simplicity now, let's build Python lists inside, knowing it might limit parallelism benefits.
        # A more advanced approach would use per-thread output buffers.
        # Let's try returning a list of arrays directly. Numba might handle this.
        results_per_node: List[npt.NDArray[np.int64]] = [np.empty(0, dtype=np.int64) for _ in range(num_nodes_in_chunk)]
        # ---

        for i in prange(num_nodes_in_chunk): # Use prange for parallel execution
            node_idx = chunk_indices[i]
            proposed_neighbors_for_node = [] # Temp list for this node's proposed edges

            # Get self eligibility
            self_is_eligible = False
            if 0 <= node_idx < total_nodes:
                self_is_eligible = eligibility_proxies_flat[node_idx] > 0.5
            else: continue # Skip if node index is invalid

            # If self is ineligible, no edges proposed from its perspective
            if not self_is_eligible:
                # --- MODIFIED: Assign empty array for this node ---
                results_per_node[i] = np.empty(0, dtype=np.int64)
                continue
                # ---

            # Iterate through neighbors
            neighbor_indices = neighbor_indices_array[node_idx]
            for neighbor_idx in neighbor_indices:
                # Skip invalid neighbors (no need to check neighbor_idx <= node_idx here,
                # as we store neighbors for the *current* node_idx)
                if neighbor_idx < 0: continue

                # Get neighbor eligibility
                neighbor_is_eligible = False
                if 0 <= neighbor_idx < total_nodes:
                    neighbor_is_eligible = eligibility_proxies_flat[neighbor_idx] > 0.5
                else: continue # Skip if neighbor index is invalid

                # --- Determine if edge should exist based on eligibility ---
                propose_edge = self_is_eligible and neighbor_is_eligible
                # ---

                # --- Apply Perturbation ---
                if perturbation_enable and edge_toggle_prob > 0 and np.random.rand() < edge_toggle_prob:
                    propose_edge = not propose_edge # Flip the decision
                # ---

                if propose_edge:
                    # --- MODIFIED: Append neighbor index to this node's list ---
                    proposed_neighbors_for_node.append(neighbor_idx)
                    # ---

            # --- MODIFIED: Assign the collected neighbors for this node ---
            # Convert the list to a NumPy array for the final result structure
            if proposed_neighbors_for_node:
                results_per_node[i] = np.array(proposed_neighbors_for_node, dtype=np.int64)
            else:
                 results_per_node[i] = np.empty(0, dtype=np.int64)
            # ---

        # Return the list of neighbor index arrays
        return results_per_node

    # @timer_decorator # Remove timer from non-JIT version
    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> Dict[Tuple[int, int], float]:
        """
        Determine edge existence based on mutual eligibility using proxy states.
        This is now a fallback method if the JIT path is not used.
        (Round 14: Refactored to be fallback)
        """
        logger = logging.getLogger(__name__)
        node_idx = neighborhood.node_index
        log_prefix = f"Node {node_idx} _compute_new_edges (Fallback R14): "

        # --- Log that the fallback is being used ---
        logger.warning(f"{log_prefix}Executing non-JIT version. Performance will be lower.")
        # ---

        new_edges: Dict[Tuple[int, int], float] = {}
        # detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING # Use passed arg

        # --- Get Eligibility Proxies from rule_params ---
        eligibility_proxies_flat = neighborhood.rule_params.get('_eligibility_proxies')
        if eligibility_proxies_flat is None:
            logger.error(f"{log_prefix}Eligibility proxies missing in rule_params!")
            return new_edges # Cannot determine mutual eligibility
        # ---

        # --- Get Perturbation Parameters ---
        perturbation_enabled = self.get_param('perturbation_enable', False, neighborhood=neighborhood)
        edge_toggle_prob = self.get_param('random_edge_toggle_probability', 0.0, neighborhood=neighborhood)
        apply_edge_toggle = perturbation_enabled and edge_toggle_prob > 0
        # ---

        # Determine eligibility of the current node for the *next* step from proxy
        self_is_eligible = False
        if 0 <= node_idx < eligibility_proxies_flat.size:
            self_is_eligible = eligibility_proxies_flat[node_idx] > 0.5
        else:
            logger.warning(f"{log_prefix}Node index out of bounds for eligibility proxies.")

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges (Fallback) ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Eligible Next (from proxy): {self_is_eligible}") # type: ignore [attr-defined]
            logger.detail(f"    Apply Random Edge Toggle: {apply_edge_toggle} (Prob: {edge_toggle_prob:.3f})") # type: ignore [attr-defined]

        # If self is ineligible, no edges can form/survive from its perspective
        if not self_is_eligible:
             if detailed_logging_enabled: logger.detail("    Self ineligible, proposing no edges.") # type: ignore [attr-defined]
             return new_edges

        for neighbor_idx in neighborhood.neighbor_indices: # Iterate through valid neighbor indices
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue # Skip invalid and avoid double check

            # Get Neighbor Eligibility from Proxy Array
            neighbor_is_eligible = False
            if 0 <= neighbor_idx < eligibility_proxies_flat.size:
                neighbor_is_eligible = eligibility_proxies_flat[neighbor_idx] > 0.5
            else:
                logger.warning(f"{log_prefix}Neighbor index {neighbor_idx} out of bounds for eligibility proxies.")
                continue # Skip if neighbor index is invalid

            edge = (node_idx, neighbor_idx) # Canonical order
            propose_edge = False
            decision_reason = "Default (No Edge)"

            # Propose edge ONLY if BOTH nodes are eligible based on proxy states
            if self_is_eligible and neighbor_is_eligible:
                propose_edge = True
                decision_reason = "Propose Edge (Both eligible based on proxy)"
            # else: # If one or both are ineligible, no edge is proposed

            # --- Apply Random Edge Toggle Perturbation ---
            random_toggle_applied = False
            if apply_edge_toggle and np.random.rand() < edge_toggle_prob:
                propose_edge = not propose_edge # Flip the decision
                random_toggle_applied = True
                decision_reason += " + Random Toggle"
            # ---

            if detailed_logging_enabled:
                logger.detail(f"  Neighbor {neighbor_idx}: Eligible Next(Proxy)={neighbor_is_eligible}. Decision: {decision_reason}. Final Propose={propose_edge}") # type: ignore [attr-defined]

            # Add to proposed edges if it should exist
            if propose_edge:
                new_edges[edge] = 1.0 # Binary edges

        if detailed_logging_enabled:
            logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges

    # TODO: Note: this has been updated for the new more efficient controller.step, grid._process_final_chunk - we may need to update the other rules with final states this way too
    def _compute_final_state(self,
                            node_idx: int,
                            current_proxy_state: float, # Eligibility proxy from Phase 1
                            final_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
                            dimensions: Tuple[int,...],
                            previous_node_states: np.ndarray, # Full array from previous step
                            previous_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
                            previous_edge_states: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float],
                            previous_node_degrees: Optional[np.ndarray], # Array from previous step
                            previous_active_neighbors: Optional[np.ndarray], # Array from previous step
                            eligibility_proxies: Optional[np.ndarray] = None, # Full array from Phase 1
                            detailed_logging_enabled: bool = False,
                            # --- ADDED: Accept final degree array ---
                            final_degree_array: Optional[np.ndarray] = None, # Array from Phase 2.5
                            # ---
                            ) -> float:
        """
        Calculates the final state based on eligibility and the chosen final check metric.
        Applies life conditions first, then death conditions. Defaults to survival.
        Uses pre-calculated final degree if available and metric requires it.
        (Round 7: Use pre-calculated final degree)
        (Round 25: Accept detailed_logging_enabled flag, remove LogSettings access)
        """
        logger = logging.getLogger(__name__)
        log_prefix = f"Node {node_idx} _compute_final_state (ROL-U R7 Final Degree): " # Updated round

        if detailed_logging_enabled: logger.detail(f"{log_prefix}--- ENTRY --- Eligibility Proxy State: {current_proxy_state:.1f}") # type: ignore [attr-defined]

        if current_proxy_state < 0.5:
            if detailed_logging_enabled: logger.detail(f"    Node ineligible based on Phase 1, final state is 0.") # type: ignore [attr-defined]
            return 0.0

        # --- Calculate the value to check ---
        final_check_metric_type = self.get_param('final_check_metric', 'DEGREE')
        value_to_check: float = 0.0
        metric_is_float = False
        final_degree = 0.0 # Initialize

        # --- MODIFIED: Use pre-calculated degree if available ---
        if final_degree_array is not None and 0 <= node_idx < final_degree_array.size:
            final_degree = float(final_degree_array[node_idx])
            if detailed_logging_enabled: logger.detail(f"    Using pre-calculated final degree: {final_degree}") # type: ignore [attr-defined]
        else:
            # Fallback: Calculate degree if pre-calculated array is missing
            if final_check_metric_type in ["DEGREE", "CLUSTERING", "BETWEENNESS"]:
                logger.warning(f"{log_prefix}Final degree array missing or index out of bounds. Recalculating degree...")
                node_coords = tuple(_njit_unravel_index(node_idx, np.array(dimensions, dtype=np.int64)))
                final_degree = float(sum(1 for edge in final_edges if node_coords in edge))
                logger.warning(f"    Recalculated final degree: {final_degree}")
            else:
                 final_degree = 0.0 # Cannot calculate if not needed by metric
        # --- END MODIFIED ---

        if final_check_metric_type == "DEGREE": value_to_check = final_degree; metric_is_float = False
        elif final_check_metric_type == "ACTIVE_NEIGHBOR_COUNT":
            count = 0
            if eligibility_proxies is not None:
                neighbor_indices_set = set()
                node_coords = tuple(_njit_unravel_index(node_idx, np.array(dimensions, dtype=np.int64))) # Need node coords
                for edge in final_edges:
                    if node_coords == edge[0]: neighbor_indices_set.add(_njit_ravel_multi_index(np.array(edge[1], dtype=np.int64), np.array(dimensions, dtype=np.int64)))
                    elif node_coords == edge[1]: neighbor_indices_set.add(_njit_ravel_multi_index(np.array(edge[0], dtype=np.int64), np.array(dimensions, dtype=np.int64)))
                for neighbor_idx in neighbor_indices_set:
                    if 0 <= neighbor_idx < eligibility_proxies.size:
                        if eligibility_proxies[neighbor_idx] > 0.5: count += 1
            value_to_check = float(count); metric_is_float = False
        elif final_check_metric_type == "CLUSTERING":
            max_deg_theory = self._get_max_neighbors_for_rule(); max_deg_term = float(max_deg_theory * (max_deg_theory - 1)) if max_deg_theory > 1 else 1.0
            if max_deg_term <= 0: max_deg_term = 1.0
            value_to_check = (final_degree * (final_degree - 1)) / max_deg_term if final_degree > 1 else 0.0; metric_is_float = True
        elif final_check_metric_type == "BETWEENNESS":
            value_to_check = 1.0 / float(final_degree) if final_degree > 0 else 0.0; metric_is_float = True
        else: value_to_check = final_degree; metric_is_float = False # Default to DEGREE

        if detailed_logging_enabled: logger.detail(f"    Final Check Metric: {final_check_metric_type}, Value: {value_to_check:.4f}, IsFloatMetric: {metric_is_float}, FinalDegree: {final_degree}") # type: ignore [attr-defined]

        # --- Apply Perturbation ---
        # [ Perturbation logic remains the same ]
        value_after_perturbation = value_to_check
        perturbation_enabled = self.get_param('perturbation_enable', False)
        state_flip_prob = self.get_param('random_state_flip_probability', 0.0)
        if perturbation_enabled and state_flip_prob > 0 and random.random() < state_flip_prob:
            if value_to_check > 1e-6: value_after_perturbation = 0.0; perturbation_desc = f"active({value_to_check:.2f})->0"
            else: value_after_perturbation = 1.0; perturbation_desc = f"inactive({value_to_check:.2f})->1"
            if detailed_logging_enabled: logger.detail(f"    PERTURBATION (Final Metric): Flipped value ({perturbation_desc}). Value now: {value_after_perturbation:.2f}") # type: ignore [attr-defined]
        # ---

        # --- Get life/death condition parameters ---
        # [ Parameter fetching remains the same ]
        metric_key_suffix = f"_{final_check_metric_type}"
        final_life_range_list = self.get_param(f'final_life_metric_range{metric_key_suffix}', [])
        final_life_values_list = self.get_param(f'final_life_metric_values{metric_key_suffix}', [])
        final_death_range_list = self.get_param(f'final_death_metric_range{metric_key_suffix}', [])
        final_death_values_list = self.get_param(f'final_death_metric_values{metric_key_suffix}', [])
        if detailed_logging_enabled:
            logger.detail(f"    --- Final Condition Params ---") # type: ignore [attr-defined]
            logger.detail(f"    Life Ranges: {final_life_range_list}") # type: ignore [attr-defined]
            logger.detail(f"    Life Values: {final_life_values_list}") # type: ignore [attr-defined]
            logger.detail(f"    Death Ranges: {final_death_range_list}") # type: ignore [attr-defined]
            logger.detail(f"    Death Values: {final_death_values_list}") # type: ignore [attr-defined]
        # ---

        # --- Check Life/Death Conditions ---
        # [ Life/Death check logic remains the same ]
        final_state = 0.0 # Default state
        tolerance = 0.005
        value_rounded = int(round(value_after_perturbation))
        if detailed_logging_enabled: logger.detail(f"    --- Checking Conditions (Value={value_after_perturbation:.4f}, Rounded={value_rounded}) ---") # type: ignore [attr-defined]
        meets_life_condition = False; life_reason_detail = ""
        life_range_match = self._check_sum_ranges(value_after_perturbation, final_life_range_list)
        if life_range_match: meets_life_condition = True; life_reason_detail = f"in ranges {final_life_range_list}"
        if detailed_logging_enabled: logger.detail(f"      Life Range Check: {life_range_match}") # type: ignore [attr-defined]
        life_values_match = False
        if not meets_life_condition and final_life_values_list:
            for target_value in final_life_values_list:
                match = False
                if isinstance(target_value, int): match = (value_rounded == target_value)
                elif isinstance(target_value, float) and metric_is_float: match = (abs(value_after_perturbation - target_value) < tolerance)
                elif isinstance(target_value, float) and not metric_is_float: continue
                if match: life_values_match = True; break
            if life_values_match: meets_life_condition = True; life_reason_detail = f"matched value {target_value} in list {final_life_values_list}"
        if detailed_logging_enabled: logger.detail(f"      Life Values Check: {life_values_match}") # type: ignore [attr-defined]
        meets_death_condition = False; death_reason_detail = ""
        death_range_match = self._check_sum_ranges(value_after_perturbation, final_death_range_list)
        if death_range_match: meets_death_condition = True; death_reason_detail = f"in ranges {final_death_range_list}"
        if detailed_logging_enabled: logger.detail(f"      Death Range Check: {death_range_match}") # type: ignore [attr-defined]
        death_values_match = False
        if not meets_death_condition and final_death_values_list:
            for target_value in final_death_values_list:
                match = False
                if isinstance(target_value, int): match = (value_rounded == target_value)
                elif isinstance(target_value, float) and metric_is_float: match = (abs(value_after_perturbation - target_value) < tolerance)
                elif isinstance(target_value, float) and not metric_is_float: continue
                if match: death_values_match = True; break
            if death_values_match: meets_death_condition = True; death_reason_detail = f"matched value {target_value} in list {final_death_values_list}"
        if detailed_logging_enabled: logger.detail(f"      Death Values Check: {death_values_match}") # type: ignore [attr-defined]
        # ---

        # --- Apply Logic (Life overrides Death, Default Survival) ---
        # [ Apply logic remains the same ]
        decision_reason = ""
        if meets_life_condition:
            final_state = final_degree # Use the pre-calculated or fallback degree
            decision_reason = f"Final Life (Metric={value_after_perturbation:.3f} {life_reason_detail})"
        elif meets_death_condition:
            final_state = 0.0
            decision_reason = f"Final Death (Metric={value_after_perturbation:.3f} {death_reason_detail})"
        else:
            final_state = final_degree # Use the pre-calculated or fallback degree
            decision_reason = f"Final Survival (Metric={value_after_perturbation:.3f} not in explicit life/death criteria)"
        # ---

        if detailed_logging_enabled:
            logger.detail(f"    --- Final Decision ---") # type: ignore [attr-defined]
            logger.detail(f"    Meets Life: {meets_life_condition}, Meets Death: {meets_death_condition}") # type: ignore [attr-defined]
            logger.detail(f"    Decision Reason: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    Returning Final State (Degree): {final_state:.1f}") # type: ignore [attr-defined]

        return final_state

class GameOfLife(Rule):
    """
    Conway's Game of Life (B3/S23) with configurable birth/survival rules.
    - Birth: An inactive cell becomes active if its number of active neighbors is in the 'birth_neighbor_counts' list.
    - Survival: An active cell remains active if its number of active neighbors is in the 'survival_neighbor_counts' list.
    - Death: All other cells become or remain inactive.
    Edges are not used or modified by this rule. Standard GoL uses B[3]/S[2, 3].
    """
    # --- ADDED: Exclude unused base parameters from editor ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'edge_initialization', 'connect_probability', 'min_edge_weight', 'max_edge_weight',
        'tiebreaker_type', 'node_history_depth', 'state_rule_table', 'edge_rule_table',
        'use_state_coloring_edges', 'color_edges_by_neighbor_degree',
        'color_edges_by_neighbor_active_neighbors', 'edge_colormap',
        'edge_color_norm_vmin', 'edge_color_norm_vmax', 'node_history_depth'
    }
    # ---
    
    produces_binary_edges: ClassVar[bool] = True
    node_state_type: ClassVar[StateType] = StateType.BINARY
    edge_state_type: ClassVar[StateType] = StateType.BINARY
    min_node_state: ClassVar[float] = 0.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0
    
    PARAMETER_METADATA = {
        # --- State Update ---
        "birth_neighbor_counts": {
            'type': list, 'element_type': int, 'default': [3],
            'description': "List of exact active neighbor counts required for an inactive cell to become active (birth). Example: [3]",
            "parameter_group": "State Update (GoL Counts)"
        },
        "survival_neighbor_counts": {
            'type': list, 'element_type': int, 'default': [2, 3],
            'description': "List of exact active neighbor counts required for an active cell to remain active (survival). Example: [2, 3]",
            "parameter_group": "State Update (GoL Counts)"
        },
        # --- Core Parameters (Fixed for standard GoL, but keep for potential variation) ---
        "neighborhood_type": {
            'type': str, 'default': "MOORE",
            'allowed_values': ["MOORE", "VON_NEUMANN"],
            'description': "Neighborhood definition (Adjacent cells). MOORE (8 neighbors in 2D, 26 in 3D) is standard for GoL.",
            "parameter_group": "Core"
        },
        "dimension_type": {
            'type': str, 'default': "TWO_D",
            'allowed_values': ["TWO_D", "THREE_D"],
            'description': "Grid dimension (TWO_D is standard for GoL).",
            "parameter_group": "Core"
        },
        'grid_boundary': {
            'type': str, 'default': 'wrap',
            'allowed_values': ['bounded', 'wrap'],
            'description': 'Grid boundary behavior (wrap connects edges, bounded stops at edges). Wrap is common for GoL patterns.',
            "parameter_group": "Core"
        },
        # --- Initialization Parameters ---
        "initial_density": {
            "type": float, 'default': GlobalSettings.Simulation.INITIAL_NODE_DENSITY,
            "description": "Initial density of active nodes (0.0 to 1.0) when using 'Random' initial conditions.",
            "min": 0.0, "max": 1.0,
            "parameter_group": "Initialization"
        },
        "initial_conditions": {
            "type": str, 'default': "Random",
            "description": "Method for setting the initial grid state.",
            "allowed_values": ['Random', 'Glider Pattern', 'Gosper Glider Gun Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'],
            "parameter_group": "Initialization"
        },
         # --- Visualization Parameters (Keep node coloring options) ---
        "use_state_coloring": {
            "type": bool, "description": "Color nodes based on state (0 or 1).",
            "default": False, "parameter_group": "Visualization: Nodes"
        },
        "color_nodes_by_degree": {
            "type": bool, "description": "Color nodes based on connection count (degree). (Not applicable to GoL)",
            "default": False, "parameter_group": "Visualization: Nodes"
        },
        "color_nodes_by_active_neighbors": {
            "type": bool, "description": "Color nodes based on active neighbor count (previous step).",
            "default": False, "parameter_group": "Visualization: Nodes"
        },
        "node_colormap": {
            "type": str, "description": "Colormap for node coloring (if enabled).",
            "default": "prism", "parameter_group": "Visualization: Nodes",
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "node_color_norm_vmin": {
            "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..",
            "default": 0.0, "parameter_group": "Visualization: Nodes"
        },
        "node_color_norm_vmax": { 
            "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", 
            "default": 1.0, "parameter_group": "Visualization: Nodes"},
    }

    def __init__(self, metadata: 'RuleMetadata'):
        # --- Ensure correct metadata is passed ---
        metadata.name = "Game of Life"
        metadata.description = "Conway's Game of Life. Birth/Survival rules (B/S counts) are configurable. Standard GoL is B3/S23. Edges are not used." # Updated description
        metadata.category = "Life-Like"
        metadata.tags = ["Life", "Conway", "B3/S23", "Standard", "Configurable"] # Updated tags
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        metadata.allow_rule_tables = False # GoL doesn't use tables
        # ---
        super().__init__(metadata)
        self.name = "Game of Life" # Set name explicitly
        # Set fixed core param for edge initialization AFTER super().__init__
        self._params['edge_initialization'] = "NONE"

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute new state using configurable B/S rules."""
        logger = logging.getLogger(__name__)
        # detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING # Use passed arg
        node_idx = neighborhood.node_index

        # Fetch parameters using get_param with neighborhood context
        birth_rules = self.get_param('birth_neighbor_counts', [3], neighborhood=neighborhood)
        survival_rules = self.get_param('survival_neighbor_counts', [2, 3], neighborhood=neighborhood)

        # Log fetched parameters if detailed logging is on
        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Current State: {neighborhood.node_state:.1f}") # type: ignore [attr-defined]
            logger.detail(f"    Using Birth Rules: {birth_rules}, Survival Rules: {survival_rules}") # type: ignore [attr-defined]

        num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)

        if detailed_logging_enabled:
            logger.detail(f"    Calculated Active Neighbors: {num_active_neighbors}") # type: ignore [attr-defined]

        new_state = 0.0
        decision_reason = "Default (Death/Inactive)"
        if neighborhood.node_state > 0: # Active node
            if num_active_neighbors in survival_rules:
                new_state = 1.0
                decision_reason = f"Survival (Neighbors={num_active_neighbors} in {survival_rules})"
            else:
                decision_reason = f"Death (Neighbors={num_active_neighbors} not in {survival_rules})"
        else: # Inactive node
            if num_active_neighbors in birth_rules:
                new_state = 1.0
                decision_reason = f"Birth (Neighbors={num_active_neighbors} in {birth_rules})"
            else:
                 decision_reason = f"Remain Dead (Neighbors={num_active_neighbors} not in {birth_rules})"

        if detailed_logging_enabled:
            logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    New State: {new_state:.1f}") # type: ignore [attr-defined]

        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> Dict[Tuple[int, int], float]:
        """Game of Life does not use or modify edges."""
        return {}
    
class TwoDCAMasterRule(Rule):
    """
    General purpose 2D Cellular Automaton rule using the Moore neighborhood.
    This rule provides a flexible framework for defining 2D CA behavior.

    Default Behavior (B/S Counts):
    - By default, node state (0 for inactive, 1 for active) is determined by birth/survival rules based on the count of active neighbors in the previous step.
    - Birth: An inactive node becomes active if its active neighbor count is present in the 'birth_neighbor_counts' list (e.g., [3] for standard Game of Life).
    - Survival: An active node remains active if its active neighbor count is present in the 'survival_neighbor_counts' list (e.g., [2, 3] for standard Game of Life).
    - Death: Nodes that don't meet birth or survival criteria become or remain inactive (state 0).

    Rule Table Override:
    - If the 'state_rule_table' parameter (a dictionary) is populated with entries, the B/S count logic is *completely ignored*.
    - The state transition is then determined solely by looking up a key in the state_rule_table.
    - The key format is a string: '(current_state, neighbor_pattern, connection_pattern)'
        - current_state: 0 (inactive) or 1 (active).
        - neighbor_pattern: An 8-digit binary string representing the states (0/1) of the 8 Moore neighbors in a fixed order.
        - connection_pattern: Ignored by this rule (always treated as '00000000') since edges are not used.
    - The value associated with the key in the table should be 0 or 1, representing the node's next state.
    - **Important:** If a specific state/pattern key is *not* found in the provided table, the node's next state will default to 0 (inactive/death), unless a specific key named 'default' exists in the table dict, in which case that value will be used.

    Fixed Settings:
    - This rule is fixed to operate on a 2D grid.
    - It always uses the Moore neighborhood (8 neighbors).
    - It does not use or modify edges ('edge_initialization' is fixed to NONE).
    """
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'edge_initialization', 'connect_probability', 'min_edge_weight', 'max_edge_weight',
        'tiebreaker_type', 'node_history_depth', 'edge_rule_table', # Keep state_rule_table
        'use_state_coloring_edges', 'color_edges_by_neighbor_degree',
        'color_edges_by_neighbor_active_neighbors', 'edge_colormap',
        'edge_color_norm_vmin', 'edge_color_norm_vmax',
        'dimension_type', 'neighborhood_type', 'node_history_depth',
    }

    produces_binary_edges: ClassVar[bool] = True
    node_state_type: ClassVar[StateType] = StateType.BINARY
    edge_state_type: ClassVar[StateType] = StateType.BINARY
    min_node_state: ClassVar[float] = 0.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0

    PARAMETER_METADATA = {
        # --- Core (Fixed - Defined here but excluded from editor) ---
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D"], 'description': "Grid Dimension (Fixed to TWO_D).", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE"], 'description': "Neighborhood Type (Fixed to MOORE).", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'default': 'wrap', 'allowed_values': ['bounded', 'wrap'], 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # --- State Update (Default Logic) ---
        "birth_neighbor_counts": { 'type': list, 'element_type': int, 'default': [3], 'description': "Default Logic: List of exact active neighbor counts for birth (e.g., [3]). Used only if State Rule Table is empty.", "parameter_group": "State Update (Default B/S Logic)" },
        "survival_neighbor_counts": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Default Logic: List of exact active neighbor counts for survival (e.g., [2, 3]). Used only if State Rule Table is empty.", "parameter_group": "State Update (Default B/S Logic)" },
        # --- Rule Table (Optional Override) ---
        "state_rule_table": { "type": dict, "default": {}, "parameter_group": "State Update (Rule Table Override)", "description": "Optional Override: If non-empty, this table dictates state changes, overriding B/S counts. Keys are strings '(state, neighbors, connections)', e.g., '(1, 00100100, 00000000)'. Values are 0 or 1. Connection pattern is ignored. Missing keys default to state 0 unless a 'default' key exists." },
        # --- Initialization ---
        "initial_density": { "type": float, 'default': GlobalSettings.Simulation.INITIAL_NODE_DENSITY, "description": "Initial density of active nodes (0.0 to 1.0) when using 'Random' initial conditions.", "min": 0.0, "max": 1.0, "parameter_group": "Initialization" },
        "initial_conditions": { "type": str, 'default': "Random", "description": "Method for setting the initial grid state.", "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization" },
        # --- Visualization (Keep node options) ---
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (0 or 1).", "default": False, "parameter_group": "Visualization: Nodes" },
        "color_nodes_by_degree": { "type": bool, "description": "Color nodes based on connection count (degree). (Not applicable)", "default": False, "parameter_group": "Visualization: Nodes" },
        "color_nodes_by_active_neighbors": { "type": bool, "description": "Color nodes based on active neighbor count (previous step).", "default": False, "parameter_group": "Visualization: Nodes" },
        "node_colormap": { "type": str, "description": "Colormap for node coloring (if enabled).", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps },
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes" },
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization: Nodes"},
    }

    def __init__(self, metadata: 'RuleMetadata'):
        # --- Ensure correct metadata is passed ---
        metadata.name = "2D CA Master Rule"
        metadata.description = "General 2D CA (Moore neighborhood) using configurable B/S counts. State rule table can override." # Keep concise here
        metadata.category = "TwoD CA Master Rule"
        metadata.tags = ["2D", "Moore", "configurable", "master", "rule table"]
        metadata.neighborhood_compatibility = ["MOORE"] # Fixed
        metadata.dimension_compatibility = ["TWO_D"] # Fixed
        metadata.allow_rule_tables = True # Allows state table
        # ---
        super().__init__(metadata)
        self.name = "2D CA Master Rule" # Set name explicitly
        # Set fixed core params AFTER super().__init__
        self._params['dimension_type'] = "TWO_D"
        self._params['neighborhood_type'] = "MOORE"
        self._params['edge_initialization'] = "NONE"


    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute new state using rule table if provided, otherwise B/S counts."""
        logger = logging.getLogger(__name__)
        # detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING # Use passed arg
        node_idx = neighborhood.node_index

        # Fetch parameters using get_param with neighborhood context
        state_rule_table = self.get_param('state_rule_table', {}, neighborhood=neighborhood)
        birth_rules = self.get_param('birth_neighbor_counts', [3], neighborhood=neighborhood)
        survival_rules = self.get_param('survival_neighbor_counts', [2, 3], neighborhood=neighborhood)

        # --- Rule Table Logic (if table is not empty) ---
        if state_rule_table: # Check if the dictionary is populated
            neighbor_pattern = self._get_neighbor_pattern(neighborhood)
            # Connection pattern is ignored, use all zeros for Moore (8 neighbors)
            connection_pattern = '00000000'
            current_state_int = 1 if neighborhood.node_state > 0.5 else 0 # Treat state as binary

            key = f"({current_state_int}, {neighbor_pattern}, {connection_pattern})"
            default_outcome = state_rule_table.get('default', 0) # Get table default, fallback to 0
            new_state_float = float(state_rule_table.get(key, default_outcome))

            if detailed_logging_enabled:
                logger.detail(f"Node {node_idx}: Using Rule Table. Key='{key}'. Found={key in state_rule_table}. Result={new_state_float}") # type: ignore [attr-defined]
            return new_state_float

        # --- Fallback to B/S Count Logic ---
        else:
            if detailed_logging_enabled:
                logger.detail(f"Node {node_idx}: Using B/S Logic. Birth={birth_rules}, Survival={survival_rules}") # type: ignore [attr-defined]

            num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
            new_state = 0.0
            decision_reason = "Default (Death/Inactive)"

            if neighborhood.node_state <= 0: # Birth check
                if num_active_neighbors in birth_rules:
                    new_state = 1.0
                    decision_reason = f"Birth (Neighbors={num_active_neighbors} in {birth_rules})"
                else:
                    decision_reason = f"Remain Dead (Neighbors={num_active_neighbors} not in {birth_rules})"
            else: # Survival check
                if num_active_neighbors in survival_rules:
                    new_state = 1.0
                    decision_reason = f"Survival (Neighbors={num_active_neighbors} in {survival_rules})"
                else:
                    decision_reason = f"Death (Neighbors={num_active_neighbors} not in {survival_rules})"

            if detailed_logging_enabled:
                logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
                logger.detail(f"    New State: {new_state:.1f}") # type: ignore [attr-defined]
            return new_state
        
    # --- ADDED: Implementation for _compute_new_edges ---
    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> Dict[Tuple[int, int], float]:
        """This rule does not use or modify edges."""
        return {}
    # --- END ADDED ---

class LaceLifeWithEdges(Rule):
    """
    Game of Life variant with configurable edge dynamics and effects.
    Node state logic uses GoL B/S counts, optionally modified by edge connectivity.
    Edges are binary (0/1). Node state stored can be binary or degree/neighbor count.
    (Round 14: Added missing PARAMETER_METADATA)
    """
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'min_edge_weight', 'max_edge_weight', # Edges are binary
        'color_edges_by_neighbor_degree', # Edges are binary, state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edges are binary, state not based on this
        'tiebreaker_type', 'node_history_depth',
    }
    
    produces_binary_edges: ClassVar[bool] = True # Edges are 0 or 1

    # --- ADDED PARAMETER_METADATA ---
    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup (Binary Edges).', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern.", "default": "Random", "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Initial density of active nodes (state=1).", "min": 0.0, "max": 1.0, "default": 0.35, "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability for RANDOM edge init.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        # === Node Update Logic ===
        "birth_neighbor_counts": { 'type': list, 'element_type': int, 'default': [3], 'description': "Node Logic: List of active neighbor counts for birth.", "parameter_group": "Node Logic: GoL Counts"},
        "survival_neighbor_counts": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Node Logic: List of active neighbor counts for survival.", "parameter_group": "Node Logic: GoL Counts"},
        "use_edge_effects_on_nodes": { 'type': bool, 'default': False, 'description': "Node Logic: Enable probabilistic boost/penalty to node survival based on edge connectivity.", "parameter_group": "Node Logic: Edge Effects"},
        "survival_boost_factor": { 'type': float, 'default': 0.1, 'description': "Node Logic: Probability (0-1) to survive death if connected to active neighbors.", "min": 0.0, "max": 1.0, "parameter_group": "Node Logic: Edge Effects"},
        "death_boost_factor": { 'type': float, 'default': 0.1, 'description': "Node Logic: Probability (0-1) to die despite survival conditions if NOT connected to active neighbors.", "min": 0.0, "max": 1.0, "parameter_group": "Node Logic: Edge Effects"},
        # === Edge Update Logic ===
        "edge_support_neighbor_threshold": { 'type': int, 'default': 1, 'description': "Edge Logic: Min active neighbors (prev step) BOTH nodes need to form/maintain edge.", "min": 0, "max": 26, "parameter_group": "Edge Logic"},
        # === Visualization ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (degree).", "default": True, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states (degrees).", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 8.0, "parameter_group": "Visualization: Nodes"},
        "use_state_coloring_edges": { "type": bool, "description": "Color edges based on state (0 or 1).", "default": False, "parameter_group": "Visualization: Edges"},
        "edge_colormap": { "type": str, "description": "Colormap for edge coloring (if enabled).", "default": "binary", "parameter_group": "Visualization: Edges", "allowed_values": ["(None)"] + _standard_colormaps},
        "edge_color_norm_vmin": { "type": float, "description": "Min value for edge color normalization.", "default": 0.0, "parameter_group": "Visualization: Edges"},
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # === Other ===
        "node_history_depth": { 'type': int, 'description': 'Number of previous states to store.', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }
    # --- END ADDED ---

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Lace Life with Edges"
        metadata.description = "Node state uses GoL B/S counts, optionally modified by edge connectivity. Binary edges based on node activity/support. Node state/color can be binary, degree, or neighbor count."
        metadata.category = "Life-Like"
        metadata.tags = ["Life", "Edges", "Binary", "Configurable", "B3/S23", "Connectivity", "Degree", "Average Degree"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Lace Life with Edges"
        # --- REMOVED: _params initialization here, handled by base class ---
        self.requires_post_edge_state_update = True # Final state is degree
        self.needs_neighbor_degrees = False # Not needed for core logic
        self.needs_neighbor_active_counts = True # Needed if dynamic edge logic is used
        # --- Set coloring flags AFTER super().__init__ ---
        self._params.setdefault('use_state_coloring', True)
        self._params.setdefault('color_nodes_by_degree', True) # Default to degree coloring
        self._params.setdefault('color_nodes_by_active_neighbors', False)
        self._params.setdefault('node_colormap', 'prism')
        self._params.setdefault('node_color_norm_vmax', 8.0)
        self._params.setdefault('use_state_coloring_edges', False) # Default False for binary edges
        self._params.setdefault('edge_colormap', 'binary')
        self._params.setdefault('edge_color_norm_vmin', 0.0)
        self._params.setdefault('edge_color_norm_vmax', 1.0)

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    # --- Methods (_compute_new_state, _compute_new_edges, _compute_final_state) remain unchanged ---
    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute ELIGIBILITY proxy state (0 or 1) using GoL counts, optionally modified by edge effects."""
        # This logic is identical to LaceLifeWithEdges._compute_new_state
        logger = logging.getLogger(__name__)
        # detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING # Use passed arg
        node_idx = neighborhood.node_index

        # Fetch parameters
        birth_rules = self.get_param('birth_neighbor_counts', [3], neighborhood=neighborhood)
        survival_rules = self.get_param('survival_neighbor_counts', [2, 3], neighborhood=neighborhood)
        use_edge_effects = self.get_param('use_edge_effects_on_nodes', False, neighborhood=neighborhood)
        survival_boost = self.get_param('survival_boost_factor', 0.1, neighborhood=neighborhood)
        death_boost = self.get_param('death_boost_factor', 0.1, neighborhood=neighborhood)

        num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
        neighborhood.rule_params['_active_neighbor_count'] = num_active_neighbors

        # --- Step 1: Determine Base GoL State ---
        gol_state = 0.0
        if neighborhood.node_state > 0: # Check if node was active (degree > 0)
            if num_active_neighbors in survival_rules: gol_state = 1.0
        else: # Birth check
            if num_active_neighbors in birth_rules: gol_state = 1.0

        if detailed_logging_enabled:
            logger.detail(f"Node {node_idx}: PrevState(Degree)={neighborhood.node_state:.0f}, ActiveN={num_active_neighbors} -> BaseGoLState={gol_state:.0f}") # type: ignore [attr-defined]

        # --- Step 2: Apply Optional Edge Effects ---
        final_state = gol_state # Start with GoL state
        if use_edge_effects and neighborhood.node_state > 0: # Only apply to nodes that were active
            connected_active_neighbors = sum(1 for idx, state in neighborhood.neighbor_edge_states.items()
                                             if idx >= 0 and state > 1e-6 and neighborhood.neighbor_states[neighborhood.neighbor_indices == idx][0] > 0)

            if gol_state <= 0: # GoL predicts death
                if connected_active_neighbors > 0:
                    if np.random.random() < survival_boost:
                        final_state = 1.0 # Probabilistic survival boost
                        if detailed_logging_enabled: logger.detail(f"    EdgeEffect: SURVIVAL BOOST applied (ConnectedActiveN={connected_active_neighbors})") # type: ignore [attr-defined]
            elif gol_state > 0: # GoL predicts survival
                if connected_active_neighbors == 0:
                    if np.random.random() < death_boost:
                        final_state = 0.0 # Probabilistic death penalty
                        if detailed_logging_enabled: logger.detail(f"    EdgeEffect: DEATH PENALTY applied (ConnectedActiveN=0)") # type: ignore [attr-defined]

        if detailed_logging_enabled:
            logger.detail(f"    Final State (Eligibility Proxy) calculated: {final_state:.1f}") # type: ignore [attr-defined]

        return final_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> Dict[Tuple[int, int], float]:
        """Propose binary edges based on selected logic (simple or dynamic)."""
        # This logic is identical to LaceLifeWithEdges._compute_new_edges
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        use_dynamic_logic = self.get_param('use_dynamic_edge_logic', False, neighborhood=neighborhood)
        support_thr = self.get_param('edge_support_neighbor_threshold', 1, neighborhood=neighborhood)

        # Determine eligibility of the current node for the *next* step
        self_is_eligible = self._compute_new_state(neighborhood, detailed_logging_enabled) > 0.5
        # Get current node's active neighbor count from *previous* step (needed for dynamic logic)
        self_prev_active_count = self._count_active_neighbors(neighborhood.neighbor_states)

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges (Dynamic={use_dynamic_logic}) ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Eligible Next: {self_is_eligible}, Self Prev Active Neighbors: {self_prev_active_count}") # type: ignore [attr-defined]

        if not self_is_eligible:
            if detailed_logging_enabled: logger.detail("    Self ineligible, proposing no edges.") # type: ignore [attr-defined]
            return new_edges # Propose no edges if self will be dead

        # If current node will be alive, check neighbors
        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0: continue

            # Determine neighbor's eligibility for the *next* step (approximation)
            neighbor_is_eligible = neighbor_state > 0 # Check if neighbor was active (degree > 0)

            propose_edge = False # Default: no edge
            decision_reason = "Default (No Edge)"

            if self_is_eligible and neighbor_is_eligible:
                if use_dynamic_logic:
                    # Dynamic Logic: Check support threshold using PREVIOUS active counts
                    neighbor_prev_active_count = 0
                    if neighborhood.neighbor_active_counts is not None:
                         neighbor_prev_active_count = neighborhood.neighbor_active_counts.get(neighbor_idx, 0)
                    else: logger.warning(f"Node {node_idx}: neighbor_active_counts missing for neighbor {neighbor_idx}")

                    if self_prev_active_count >= support_thr and neighbor_prev_active_count >= support_thr:
                        propose_edge = True
                        decision_reason = f"Propose/Maintain (Dynamic: Eligible & Support Met >= {support_thr})"
                    else:
                        decision_reason = f"Break/No Form (Dynamic: Support Low < {support_thr})"
                else:
                    # Simple Logic: Just connect if both are eligible
                    propose_edge = True
                    decision_reason = "Propose/Maintain (Simple: Both Eligible)"

            else: # One or both ineligible
                decision_reason = "Break/No Form (One Ineligible)"

            if detailed_logging_enabled:
                logger.detail(f"  Neighbor {neighbor_idx}: Eligible Next(Proxy)={neighbor_is_eligible}. Decision: {decision_reason}") # type: ignore [attr-defined]

            if propose_edge:
                edge = (node_idx, neighbor_idx) if node_idx < neighbor_idx else (neighbor_idx, node_idx)
                new_edges[edge] = 1.0 # Binary edges

        if detailed_logging_enabled:
            logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges

    def _compute_final_state(self,
                             node_idx: int,
                             current_proxy_state: float, # Eligibility/state from current step's computation
                             final_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]], # Edges calculated for *current* step
                             dimensions: Tuple[int,...],
                             # --- Accept all arguments even if unused ---
                             previous_node_states: npt.NDArray[np.float64],
                             previous_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
                             previous_edge_states: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float],
                             previous_node_degrees: Optional[npt.NDArray[np.int32]],
                             previous_active_neighbors: Optional[npt.NDArray[np.int32]],
                             eligibility_proxies: Optional[np.ndarray] = None,
                             detailed_logging_enabled: bool = False
                             ) -> float:
        """
        Calculates the final node state (degree) based on eligibility and final edge count.
        """
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_final_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Proxy State (Eligibility): {current_proxy_state:.1f}") # type: ignore [attr-defined]

        # If proxy state indicates ineligible (0.0), final state is 0
        if current_proxy_state < 0.5:
             if detailed_logging_enabled: logger.detail(f"    Node was ineligible based on prev step conditions, final state is 0.") # type: ignore [attr-defined]
             return 0.0

        # If eligible (proxy was 1.0), calculate final degree based on the FINAL edges for this step
        node_coords = tuple(_unravel_index(node_idx, dimensions))
        final_degree = sum(1 for edge in final_edges if node_coords in edge)

        if detailed_logging_enabled:
            logger.detail(f"    Node was eligible. Final Calculated Degree: {final_degree}") # type: ignore [attr-defined]
            logger.detail(f"    Setting Final State to: {float(final_degree):.1f}") # type: ignore [attr-defined]

        # The state IS the degree if eligible
        return float(final_degree)
    
class LifeWithEdgeGrowth(Rule):
    """
    Game of Life variant where node survival/birth requires meeting BOTH
    standard GoL active neighbor counts AND minimum edge strength sum thresholds.
    Edges have continuous states (0-1) that grow or decay based on node activity.
    (Round 33: Added edge visualization parameters)
    """

    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'color_edges_by_neighbor_degree', # Edge state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edge state not based on this
        'tiebreaker_type', 'node_history_depth'
    }
    # --- END ADDED ---

    node_state_type: ClassVar[StateType] = StateType.BINARY # Nodes are 0/1
    edge_state_type: ClassVar[StateType] = StateType.REAL   # Edges are 0.0-1.0
    min_node_state: ClassVar[float] = 0.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern.", "default": "Random", "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Initial density of active nodes (state=1).", "min": 0.0, "max": 1.0, "default": 0.35, "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability for RANDOM edge init.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "min_edge_weight": { "type": float, "description": "Minimum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Initialization"},
        "max_edge_weight": { "type": float, "description": "Maximum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        # === State Update ===
        "birth_neighbor_counts": { 'type': list, 'element_type': int, 'default': [3], 'description': "Node Logic: List of active neighbor counts for potential birth.", "parameter_group": "Node Logic"},
        "birth_min_edge_strength_sum": { 'type': float, 'description': "Node Logic: Min sum of connecting edge states to active neighbors required for birth.", "min": 0.0, "max": 26.0, "default": 0.5, "parameter_group": "Node Logic"},
        "survival_neighbor_counts": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Node Logic: List of active neighbor counts for potential survival.", "parameter_group": "Node Logic"},
        "survival_min_edge_strength_sum": { 'type': float, 'description': "Node Logic: Min sum of connecting edge states to active neighbors required for survival.", "min": 0.0, "max": 26.0, "default": 0.3, "parameter_group": "Node Logic"},
        # === Edge Update ===
        "edge_growth_rate": { "type": float, "description": "Edge Logic: Rate (0-1) at which existing edges strengthen if nodes are active.", "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Edge Logic: Growth/Decay"},
        "edge_decay_rate": { "type": float, "description": "Edge Logic: Rate (0-1) at which existing edges weaken if a node is inactive.", "min": 0.0, "max": 1.0, "default": 0.2, "parameter_group": "Edge Logic: Growth/Decay"},
        "new_edge_initial_strength": { "type": float, "description": "Edge Logic: Initial strength (0-1) of a newly formed edge.", "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Edge Logic: Growth/Decay"},
        "min_edge_state_to_keep": { "type": float, "description": "Edge Logic: Minimum edge state required to be kept.", "min": 0.0, "max": 0.5, "default": 0.01, "parameter_group": "Edge Logic: Pruning"},
        "use_probabilistic_degree_edges": { 'type': bool, 'default': False, 'description': "Edge Logic: Enable probabilistic edge changes based on neighbor degree.", "parameter_group": "Edge Logic: Probabilistic Degree"},
        "prob_connect_degree_threshold": { 'type': int, 'default': 3, 'description': "Edge Logic: Connect prob if neighbor degree > this.", "min": 0, "max": 26, "parameter_group": "Edge Logic: Probabilistic Degree"},
        "prob_connect_if_above_threshold": { 'type': float, 'default': 0.1, 'description': "Edge Logic: Probability to form edge if neighbor degree above threshold.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Logic: Probabilistic Degree"},
        "prob_disconnect_degree_threshold": { 'type': int, 'default': 2, 'description': "Edge Logic: Disconnect prob if neighbor degree < this.", "min": 0, "max": 26, "parameter_group": "Edge Logic: Probabilistic Degree"},
        "prob_disconnect_if_below_threshold": { 'type': float, 'default': 0.2, 'description': "Edge Logic: Probability to remove edge if neighbor degree below threshold.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Logic: Probabilistic Degree"},
        # === Visualization ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (0 or 1).", "default": False, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states.", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization: Nodes"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on their state value (0-1).",
            "default": True, "parameter_group": "Visualization: Edges" # Default True for continuous
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "prism", "parameter_group": "Visualization: Edges",
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # === Other ===
        "node_history_depth": { 'type': int, 'description': 'Number of previous states to store.', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Life with Edge Growth"
        metadata.description = "GoL B/S counts AND min edge strength sum required for birth/survival. Continuous edges (0-1) grow/decay based on node activity. Optional probabilistic edge changes based on neighbor degree."
        metadata.category = "Life-Like"
        metadata.tags = ["Life", "Edges", "Growth", "Weighted", "Connectivity", "B3/S23", "Probabilistic", "Degree"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Life with Edge Growth"
        self._params = {}
        self.requires_post_edge_state_update = False
        self.needs_neighbor_degrees = True # Needed for probabilistic edge logic
        self.needs_neighbor_active_counts = False
        # --- ADDED: Ensure edge coloring defaults are set ---
        self._params.setdefault('use_state_coloring_edges', True)
        self._params.setdefault('edge_colormap', 'prism')
        self._params.setdefault('edge_color_norm_vmin', 0.0)
        self._params.setdefault('edge_color_norm_vmax', 1.0)
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    # --- Methods (_compute_new_state, _compute_new_edges) remain unchanged ---
    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute new state using GoL neighbor counts first, then edge strength sum."""
        logger = logging.getLogger(__name__)
        # detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING # Use passed arg
        node_idx = neighborhood.node_index

        # Fetch parameters
        birth_neighbor_counts = self.get_param('birth_neighbor_counts', [3], neighborhood=neighborhood)
        survival_neighbor_counts = self.get_param('survival_neighbor_counts', [2, 3], neighborhood=neighborhood)
        birth_min_edge_sum = self.get_param('birth_min_edge_strength_sum', 0.5, neighborhood=neighborhood)
        survival_min_edge_sum = self.get_param('survival_min_edge_strength_sum', 0.3, neighborhood=neighborhood)

        # Count active neighbors
        num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
        # Sum edge strengths ONLY to active neighbors
        sum_edge_strength = sum(
            neighborhood.neighbor_edge_states.get(idx, 0.0)
            for idx, state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states)
            if idx >= 0 and state > 0 # Only edges to ACTIVE neighbors
        )

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State: {neighborhood.node_state:.1f}") # type: ignore [attr-defined]
            logger.detail(f"    Active Neighbors: {num_active_neighbors}") # type: ignore [attr-defined]
            logger.detail(f"    Sum Edge Strength (to active): {sum_edge_strength:.4f}") # type: ignore [attr-defined]

        new_state = 0.0 # Default to death
        decision_reason = "Default (Death/Inactive)"

        if neighborhood.node_state <= 0: # --- Birth Condition ---
            passes_neighbor_check = num_active_neighbors in birth_neighbor_counts
            passes_edge_check = sum_edge_strength >= birth_min_edge_sum
            if passes_neighbor_check and passes_edge_check:
                new_state = 1.0
                decision_reason = f"Birth (NeighborsOk={passes_neighbor_check}, EdgeSumOk={passes_edge_check})"
            else:
                decision_reason = f"Remain Dead (NeighborsOk={passes_neighbor_check}, EdgeSumOk={passes_edge_check})"

        else: # --- Survival Condition ---
            passes_neighbor_check = num_active_neighbors in survival_neighbor_counts
            passes_edge_check = sum_edge_strength >= survival_min_edge_sum
            if passes_neighbor_check and passes_edge_check:
                new_state = 1.0
                decision_reason = f"Survival (NeighborsOk={passes_neighbor_check}, EdgeSumOk={passes_edge_check})"
            else:
                decision_reason = f"Death (NeighborsOk={passes_neighbor_check}, EdgeSumOk={passes_edge_check})"

        if detailed_logging_enabled:
            logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    New State: {new_state:.1f}") # type: ignore [attr-defined]

        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> Dict[Tuple[int, int], float]:
        """Edges grow if both nodes active, decay otherwise. Optional probabilistic changes based on neighbor degree."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch base parameters
        edge_growth_rate = self.get_param('edge_growth_rate', 0.1, neighborhood=neighborhood)
        edge_decay_rate = self.get_param('edge_decay_rate', 0.2, neighborhood=neighborhood)
        new_edge_strength = self.get_param('new_edge_initial_strength', 0.1, neighborhood=neighborhood)
        min_keep = self.get_param('min_edge_state_to_keep', 0.01, neighborhood=neighborhood)

        # Fetch probabilistic parameters
        use_prob_logic = self.get_param('use_probabilistic_degree_edges', False, neighborhood=neighborhood)
        prob_conn_thr = self.get_param('prob_connect_degree_threshold', 3, neighborhood=neighborhood)
        prob_conn_p = self.get_param('prob_connect_if_above_threshold', 0.1, neighborhood=neighborhood)
        prob_disc_thr = self.get_param('prob_disconnect_degree_threshold', 2, neighborhood=neighborhood)
        prob_disc_p = self.get_param('prob_disconnect_if_below_threshold', 0.2, neighborhood=neighborhood)

        # Determine the *next* state of the current node for edge logic
        next_node_state = self._compute_new_state(neighborhood, detailed_logging_enabled)
        self_is_active_next = next_node_state > 0.5

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges (Probabilistic={use_prob_logic}) ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Active Next: {self_is_active_next}") # type: ignore [attr-defined]

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue # Skip invalid and avoid double check

            edge = (node_idx, neighbor_idx) # Canonical order
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0)

            # Approximation: Use current neighbor state as proxy for neighbor's next state.
            neighbor_is_active_next = neighbor_state > 0.5

            # --- Base Growth/Decay Logic ---
            new_edge_state = current_edge_state # Start with current
            base_decision = "Maintain"
            if self_is_active_next and neighbor_is_active_next:
                # Both nodes likely active: Strengthen or Form
                if current_edge_state > 1e-6: # If edge exists, strengthen
                    new_edge_state += edge_growth_rate
                    base_decision = f"Strengthen (Rate={edge_growth_rate:.2f})"
                else: # If edge doesn't exist, form it
                    new_edge_state = new_edge_strength
                    base_decision = f"Form (Strength={new_edge_strength:.2f})"
            else:
                # One or both nodes likely inactive: Decay
                new_edge_state -= edge_decay_rate
                base_decision = f"Decay (Rate={edge_decay_rate:.2f})"
            # ---

            # --- Probabilistic Override Logic ---
            prob_override_applied = False
            if use_prob_logic:
                # Get neighbor's degree from previous step
                neighbor_prev_degree = 0
                if neighborhood.neighbor_degrees is not None:
                    neighbor_prev_degree = neighborhood.neighbor_degrees.get(neighbor_idx, 0)
                else:
                    logger.warning(f"Node {node_idx}: neighbor_degrees missing for neighbor {neighbor_idx} in probabilistic check.")

                # Check Disconnection Probability
                if current_edge_state >= min_keep: # Only check existing edges
                    if neighbor_prev_degree < prob_disc_thr:
                        if np.random.random() < prob_disc_p:
                            new_edge_state = 0.0 # Force disconnect
                            base_decision += f" | OVERRIDE: Prob Disconnect (NeighDeg={neighbor_prev_degree}<{prob_disc_thr}, P={prob_disc_p:.2f})"
                            prob_override_applied = True

                # Check Connection Probability (only if not disconnected above and edge didn't exist)
                elif neighbor_idx in neighborhood.neighbor_edge_states and neighborhood.neighbor_edge_states[neighbor_idx] > 0:
                    if neighbor_prev_degree > prob_conn_thr:
                        if np.random.random() < prob_conn_p:
                            new_edge_state = new_edge_strength # Force connect
                            base_decision += f" | OVERRIDE: Prob Connect (NeighDeg={neighbor_prev_degree}>{prob_conn_thr}, P={prob_conn_p:.2f})"
                            prob_override_applied = True
            # ---

            clipped_new_edge_state = float(np.clip(new_edge_state, 0.0, 1.0))

            if detailed_logging_enabled:
                logger.detail(f"    Edge {edge}: Curr={current_edge_state:.3f}, BaseDecision='{base_decision}', ProbOverride={prob_override_applied} -> New(clipped)={clipped_new_edge_state:.3f}") # type: ignore [attr-defined]

            # Only propose edges with a state greater than the minimum threshold
            if clipped_new_edge_state >= min_keep:
                 new_edges[edge] = clipped_new_edge_state

        if detailed_logging_enabled:
            logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges
    
class LifeWithConnectivityWeighting(Rule):
    """
    Game of Life variant where the standard neighbor count for birth/survival
    is replaced by a *weighted* count. Connected active neighbors contribute
    more (or less) to the sum than unconnected active neighbors. Edges are binary (0/1).
    (Round 34: Added edge visualization parameters and updated description)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'min_edge_weight', 'max_edge_weight', # Edges are binary
        # 'use_state_coloring_edges', # Keep
        # 'edge_colormap', # Keep
        # 'edge_color_norm_vmin', # Keep
        # 'edge_color_norm_vmax', # Keep
        'color_edges_by_neighbor_degree', # Edges are binary, state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edges are binary, state not based on this
        'tiebreaker_type', 'node_history_depth'
    }
    # --- END MODIFIED ---

    node_state_type: ClassVar[StateType] = StateType.BINARY
    edge_state_type: ClassVar[StateType] = StateType.BINARY
    min_node_state: ClassVar[float] = 0.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0
    produces_binary_edges: ClassVar[bool] = True

    PARAMETER_METADATA = {
        # === Node Update Logic ===
        "birth_min_weight": { 'type': float, 'default': 2.9, 'description': "Node Logic: Minimum weighted neighbor sum for birth.", "min": 0.0, "max": 26.0, "parameter_group": "Node Logic: Weighted Counts"},
        "birth_max_weight": { 'type': float, 'default': 3.1, 'description': "Node Logic: Maximum weighted neighbor sum for birth.", "min": 0.0, "max": 26.0, "parameter_group": "Node Logic: Weighted Counts"},
        "survival_min_weight": { 'type': float, 'default': 1.9, 'description': "Node Logic: Minimum weighted neighbor sum for survival.", "min": 0.0, "max": 26.0, "parameter_group": "Node Logic: Weighted Counts"},
        "survival_max_weight": { 'type': float, 'default': 3.1, 'description': "Node Logic: Maximum weighted neighbor sum for survival.", "min": 0.0, "max": 26.0, "parameter_group": "Node Logic: Weighted Counts"},
        "connected_neighbor_weight": { 'type': float, 'description': "Node Logic: Weight contribution of an active neighbor *connected* by an edge.", "min": 0.0, "max": 2.0, "default": 1.0, "parameter_group": "Node Logic: Weighted Counts"},
        "unconnected_neighbor_weight": { 'type': float, 'description': "Node Logic: Weight contribution of an active neighbor *not connected* by an edge.", "min": 0.0, "max": 2.0, "default": 0.8, "parameter_group": "Node Logic: Weighted Counts"},

        # === Edge Update Logic ===
        "random_edge_flip_prob": { "type": float, "description": "Edge Logic: Probability (0-1) of randomly flipping a potential edge's state (0->1 or 1->0).", "min": 0.0, "max": 0.1, "default": 0.002, "parameter_group": "Edge Logic: Base"},
        "use_probabilistic_weight_edges": { 'type': bool, 'default': False, 'description': "Edge Logic: Enable probabilistic edge changes based on neighbor's edge weight sum (prev step).", "parameter_group": "Edge Logic: Probabilistic Weight"},
        "prob_connect_weight_sum_threshold": { 'type': float, 'default': 2.0, 'description': "Edge Logic: Connect prob if neighbor's edge weight sum > this.", "min": 0.0, "max": 26.0, "parameter_group": "Edge Logic: Probabilistic Weight"},
        "prob_connect_if_sum_above": { 'type': float, 'default': 0.1, 'description': "Edge Logic: Probability to form edge if neighbor's edge weight sum above threshold.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Logic: Probabilistic Weight"},
        "prob_disconnect_weight_sum_threshold": { 'type': float, 'default': 1.0, 'description': "Edge Logic: Disconnect prob if neighbor's edge weight sum < this.", "min": 0.0, "max": 26.0, "parameter_group": "Edge Logic: Probabilistic Weight"},
        "prob_disconnect_if_sum_below": { 'type': float, 'default': 0.2, 'description': "Edge Logic: Probability to remove edge if neighbor's edge weight sum below threshold.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Logic: Probabilistic Weight"},

        # === Core Parameters ===
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood definition.", "parameter_group": "Core"},
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Grid dimension.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},

        # === Initialization Parameters ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup (Edges are binary).', "parameter_group": "Initialization"},
         "connect_probability": { "type": float, "description": "Probability (0-1) for RANDOM edge initialization.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "initial_density": { "type": float, 'default': GlobalSettings.Simulation.INITIAL_NODE_DENSITY, "description": "Initial density of active nodes (state=1).", "min": 0.0, "max": 1.0, "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, 'default': "Random", "description": "Method for setting the initial grid state.", "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},

        # === Visualization ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (0 or 1).", "default": False, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states.", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization: Nodes"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on state (0 or 1).",
            "default": False, "parameter_group": "Visualization: Edges" # Default False for binary
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "binary", "parameter_group": "Visualization: Edges", # Default binary
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # === Other ===
        "node_history_depth": { 'type': int, 'description': 'Number of previous states to store.', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Life with Connectivity Weighting"
        # --- UPDATED DESCRIPTION ---
        metadata.description = "GoL B/S logic based on weighted neighbor count. Connected neighbors contribute more/less based on weights. Edges are binary (0/1). Optional probabilistic edge changes based on neighbor's incoming edge weight sum."
        # ---
        metadata.category = "Life-Like"
        metadata.tags = ["Life", "Edges", "Weighted", "Connectivity", "B3/S23", "Probabilistic"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Life with Connectivity Weighting"
        self._params = {}
        self.requires_post_edge_state_update = False
        self.needs_neighbor_degrees = True # Needed for probabilistic edge logic
        self.needs_neighbor_active_counts = False
        # --- ADDED: Ensure edge coloring defaults are set ---
        self._params.setdefault('use_state_coloring_edges', False)
        self._params.setdefault('edge_colormap', 'binary')
        self._params.setdefault('edge_color_norm_vmin', 0.0)
        self._params.setdefault('edge_color_norm_vmax', 1.0)
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute new state using B/S rules based on *weighted* neighbor count ranges."""
        logger = logging.getLogger(__name__)
        # detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING # Use passed arg
        node_idx = neighborhood.node_index

        # Fetch parameters
        birth_min_w = self.get_param('birth_min_weight', 2.9, neighborhood=neighborhood)
        birth_max_w = self.get_param('birth_max_weight', 3.1, neighborhood=neighborhood)
        survival_min_w = self.get_param('survival_min_weight', 1.9, neighborhood=neighborhood)
        survival_max_w = self.get_param('survival_max_weight', 3.1, neighborhood=neighborhood)
        connected_weight = self.get_param('connected_neighbor_weight', 1.0, neighborhood=neighborhood)
        unconnected_weight = self.get_param('unconnected_neighbor_weight', 0.8, neighborhood=neighborhood)

        # Calculate weighted neighbor count
        weighted_neighbor_count = 0.0
        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx >= 0 and neighbor_state > 0: # If neighbor is active
                # Check if an edge exists (state > 0 for binary edges)
                if neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0) > 1e-6:
                    weighted_neighbor_count += connected_weight
                else:
                    weighted_neighbor_count += unconnected_weight

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State: {neighborhood.node_state:.1f}") # type: ignore [attr-defined]
            logger.detail(f"    Weighted Neighbor Count: {weighted_neighbor_count:.4f}") # type: ignore [attr-defined]

        new_state = 0.0 # Default to death
        decision_reason = "Default (Death/Inactive)"

        # Check against weight ranges
        if neighborhood.node_state <= 0: # --- Birth Condition ---
            if birth_min_w <= weighted_neighbor_count <= birth_max_w:
                new_state = 1.0
                decision_reason = f"Birth (WeightedCount={weighted_neighbor_count:.4f} in [{birth_min_w:.4f}-{birth_max_w:.4f}])"
            else:
                decision_reason = f"Remain Dead (WeightedCount={weighted_neighbor_count:.4f} OUT of [{birth_min_w:.4f}-{birth_max_w:.4f}])"
        else: # --- Survival Condition ---
            if survival_min_w <= weighted_neighbor_count <= survival_max_w:
                new_state = 1.0
                decision_reason = f"Survival (WeightedCount={weighted_neighbor_count:.4f} in [{survival_min_w:.4f}-{survival_max_w:.4f}])"
            else:
                decision_reason = f"Death (WeightedCount={weighted_neighbor_count:.4f} OUT of [{survival_min_w:.4f}-{survival_max_w:.4f}])"

        if detailed_logging_enabled:
            logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    New State: {new_state:.1f}") # type: ignore [attr-defined]

        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> Dict[Tuple[int, int], float]:
        """Edges exist (state 1.0) iff both connected nodes will be active, plus probabilistic overrides and random flips."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch base parameters
        random_flip_prob = self.get_param('random_edge_flip_prob', 0.002, neighborhood=neighborhood)

        # Fetch probabilistic parameters
        use_prob_logic = self.get_param('use_probabilistic_weight_edges', False, neighborhood=neighborhood)
        prob_conn_thr = self.get_param('prob_connect_weight_sum_threshold', 2.0, neighborhood=neighborhood)
        prob_conn_p = self.get_param('prob_connect_if_sum_above', 0.1, neighborhood=neighborhood)
        prob_disc_thr = self.get_param('prob_disconnect_weight_sum_threshold', 1.0, neighborhood=neighborhood)
        prob_disc_p = self.get_param('prob_disconnect_if_sum_below', 0.2, neighborhood=neighborhood)

        # Determine eligibility of the current node for the *next* step
        self_is_eligible = self._compute_new_state(neighborhood, detailed_logging_enabled) > 0.5

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges (Probabilistic={use_prob_logic}) ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Eligible Next: {self_is_eligible}") # type: ignore [attr-defined]

        # Iterate through neighbors
        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue # Skip invalid and avoid double check

            # Approximation: Use current neighbor state as proxy for neighbor's next state eligibility.
            neighbor_is_eligible = neighbor_state > 0.5

            edge = (node_idx, neighbor_idx) # Canonical order
            has_current_edge = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0) > 1e-6

            # --- Base Decision: Connect if both eligible ---
            propose_edge = self_is_eligible and neighbor_is_eligible
            base_decision = "Maintain/Form (Both Eligible)" if propose_edge else "Break/No Form (One Ineligible)"
            # ---

            # --- Probabilistic Override Logic ---
            prob_override_applied = False
            if use_prob_logic and (self_is_eligible and neighbor_is_eligible): # Only apply if both eligible
                # Calculate neighbor's incoming edge weight sum from PREVIOUS step
                # This requires accessing the neighbor's neighborhood data, which isn't directly available.
                # WORKAROUND: We approximate by summing the neighbor's neighbor's states (degrees)
                # This is NOT the same as edge weight sum but provides a proxy for connectivity.
                # A more accurate implementation would require passing 2nd degree neighbor states/edges.
                neighbor_prev_degree_sum_proxy = 0.0
                if neighborhood.neighbor_degrees is not None:
                    # Get the neighbor's neighbors (2nd degree from self)
                    # This requires a grid reference or passing more data - MAJOR LIMITATION
                    # For now, we'll use the neighbor's own degree as a proxy for its connectivity sum
                    neighbor_prev_degree = neighborhood.neighbor_degrees.get(neighbor_idx, 0)
                    neighbor_prev_degree_sum_proxy = float(neighbor_prev_degree) # Use degree as proxy
                    logger.warning(f"Node {node_idx}, Neighbor {neighbor_idx}: Using neighbor's degree ({neighbor_prev_degree}) as proxy for edge weight sum in probabilistic check.")
                else:
                    logger.warning(f"Node {node_idx}: neighbor_degrees missing, cannot perform probabilistic check for neighbor {neighbor_idx}.")

                # Check Disconnection Probability
                if has_current_edge: # Only check existing edges
                    if neighbor_prev_degree_sum_proxy < prob_disc_thr:
                        if np.random.random() < prob_disc_p:
                            propose_edge = False # Force disconnect
                            base_decision += f" | OVERRIDE: Prob Disconnect (NeighSumProxy={neighbor_prev_degree_sum_proxy:.2f}<{prob_disc_thr:.2f}, P={prob_disc_p:.2f})"
                            prob_override_applied = True

                # Check Connection Probability (only if not disconnected above and edge didn't exist)
                elif not has_current_edge: # Only check non-existing edges
                    if neighbor_prev_degree_sum_proxy > prob_conn_thr:
                        if np.random.random() < prob_conn_p:
                            propose_edge = True # Force connect
                            base_decision += f" | OVERRIDE: Prob Connect (NeighSumProxy={neighbor_prev_degree_sum_proxy:.2f}>{prob_conn_thr:.2f}, P={prob_conn_p:.2f})"
                            prob_override_applied = True
            # ---

            # Apply random flip
            random_flip_applied = False
            if np.random.random() < random_flip_prob:
                propose_edge = not propose_edge
                random_flip_applied = True
                base_decision += " + Random Flip"

            if detailed_logging_enabled:
                logger.detail(f"  Neighbor {neighbor_idx}: Eligible Next(Proxy)={neighbor_is_eligible}. Decision: {base_decision}. ProbOverride={prob_override_applied}. RandomFlip={random_flip_applied}. Final Propose={propose_edge}") # type: ignore [attr-defined]

            # Add to proposed edges if it should exist
            if propose_edge:
                new_edges[edge] = 1.0 # Binary edges

        if detailed_logging_enabled:
            logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges

class LifeWithDynamicEdges(Rule):
    """
    Game of Life variant where node state (binary 0/1) depends on GoL neighbor counts
    AND the sum of incoming edge weights (continuous 0-1). Edges change dynamically
    based on connected node activity, decay, and randomness.
    (Round 34: Added edge visualization parameters)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'color_edges_by_neighbor_degree', # Edge state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edge state not based on this
        'tiebreaker_type' , 'node_history_depth'
    }
    # --- END MODIFIED ---

    node_state_type: ClassVar[StateType] = StateType.BINARY # Nodes are 0/1
    edge_state_type: ClassVar[StateType] = StateType.REAL   # Edges are 0.0-1.0
    min_node_state: ClassVar[float] = 0.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0

    PARAMETER_METADATA = {
        # === Node Update Logic ===
        "birth_neighbor_counts": { 'type': list, 'element_type': int, 'default': [3], 'description': "Node Logic: List of exact active neighbor counts for potential birth.", "parameter_group": "Node Group A: GoL Counts"},
        "birth_min_edge_sum": { 'type': float, 'description': "Node Logic: Min sum of incoming edge weights (prev step) required for birth.", "min": 0.0, "max": 26.0, "default": 0.5, "parameter_group": "Node Group B: Edge Sum Condition"},
        "survival_neighbor_counts": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Node Logic: List of exact active neighbor counts for potential survival.", "parameter_group": "Node Group A: GoL Counts"},
        "survival_min_edge_sum": { 'type': float, 'description': "Node Logic: Min sum of incoming edge weights (prev step) required for survival.", "min": 0.0, "max": 26.0, "default": 0.3, "parameter_group": "Node Group B: Edge Sum Condition"},
        "survival_max_edge_sum": { 'type': float, 'description': "Node Logic: Max sum of incoming edge weights (prev step) allowed for survival.", "min": 0.0, "max": 26.0, "default": 2.5, "parameter_group": "Node Group B: Edge Sum Condition"},
        "use_edge_effects_on_nodes_group": { 'type': bool, 'default': False, 'description': "Node Logic: Enable probabilistic boost/penalty to node survival based on edge sum.", "parameter_group": "Node Group C: Edge Effects on Nodes"},
        "node_survival_boost_edge_sum_min": { 'type': float, 'default': 0.5, 'description': "Node Logic: Min edge sum (prev step) to potentially boost survival probability.", "min": 0.0, "max": 26.0, "parameter_group": "Node Group C: Edge Effects on Nodes"},
        "node_survival_boost_prob": { 'type': float, 'default': 0.1, 'description': "Node Logic: Probability (0-1) to survive death if edge sum is sufficient.", "min": 0.0, "max": 1.0, "parameter_group": "Node Group C: Edge Effects on Nodes"},
        "node_death_penalty_edge_sum_max": { 'type': float, 'default': 0.2, 'description': "Node Logic: Max edge sum (prev step) below which death probability is boosted.", "min": 0.0, "max": 26.0, "parameter_group": "Node Group C: Edge Effects on Nodes"},
        "node_death_penalty_prob": { 'type': float, 'default': 0.1, 'description': "Node Logic: Probability (0-1) to die despite survival conditions if edge sum is too low.", "min": 0.0, "max": 1.0, "parameter_group": "Node Group C: Edge Effects on Nodes"},

        # === Edge Update Logic (Continuous) ===
        "edge_change_rate": { "type": float, "description": "Edge Logic: Rate (0-1) edge state moves towards target (0 or 1 based on mutual activity).", "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Edge Group A: Base Activity/Decay"},
        "edge_decay_rate": { "type": float, "description": "Edge Logic: Constant decay subtracted from edge state each step.", "min": 0.0, "max": 0.1, "default": 0.02, "parameter_group": "Edge Group A: Base Activity/Decay"},
        "random_edge_change_prob": { "type": float, "description": "Edge Logic: Probability (0-1) of applying random change to edge state.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Group B: Randomness"},
        "random_edge_change_amount": { "type": float, "description": "Edge Logic: Max amount (+/-) of random change applied to edge state.", "min": 0.0, "max": 0.5, "default": 0.1, "parameter_group": "Edge Group B: Randomness"},
        "min_edge_state_to_keep": { "type": float, "description": "Edge Logic: Minimum edge state required to be kept (non-zero).", "min": 0.0, "max": 0.5, "default": 0.02, "parameter_group": "Edge Logic: Pruning"},
        "use_probabilistic_degree_edges_group": { 'type': bool, 'default': False, 'description': "Edge Logic: Enable probabilistic edge creation/removal based on neighbor degree (prev step). Overrides base target.", "parameter_group": "Edge Group C: Probabilistic Degree"},
        "prob_connect_degree_threshold": { 'type': int, 'default': 3, 'description': "Edge Logic: Connect probabilistically if neighbor degree (prev step) > this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Probabilistic Degree"},
        "prob_connect_if_above_threshold": { 'type': float, 'default': 0.1, 'description': "Edge Logic: Probability (0-1) to form edge if neighbor degree above threshold.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Group C: Probabilistic Degree"},
        "prob_disconnect_degree_threshold": { 'type': int, 'default': 2, 'description': "Edge Logic: Disconnect probabilistically if neighbor degree (prev step) < this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Probabilistic Degree"},
        "prob_disconnect_if_below_threshold": { 'type': float, 'default': 0.2, 'description': "Edge Logic: Probability (0-1) to remove edge if neighbor degree below threshold.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Group C: Probabilistic Degree"},
        "new_edge_initial_strength_prob": { "type": float, "description": "Edge Logic: Initial strength (0-1) of a newly formed edge via probabilistic connect.", "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Edge Group C: Probabilistic Degree"},
        "use_node_effects_on_edges_group": { 'type': bool, 'default': False, 'description': "Edge Logic: Enable neighbor count influencing edge target/rate (uses prev step counts). Overrides Groups A & C.", "parameter_group": "Edge Group D: Node Effects on Edges"},
        "edge_survival_neighbor_min": { 'type': int, 'default': 2, 'description': "Edge Logic: Min active neighbors (prev step) for nodes to boost edge survival (target 1.0).", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects on Edges"},
        "edge_survival_neighbor_max": { 'type': int, 'default': 4, 'description': "Edge Logic: Max active neighbors (prev step) for nodes to boost edge survival (target 1.0).", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects on Edges"},
        "edge_survival_boost_rate": { 'type': float, 'default': 0.1, 'description': "Edge Logic: Rate edge moves towards 1 if node neighbor counts optimal.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Group D: Node Effects on Edges"},
        "edge_death_neighbor_min": { 'type': int, 'default': 1, 'description': "Edge Logic: Edge death boost (target 0.0) if node neighbor count (prev step) < this.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects on Edges"},
        "edge_death_neighbor_max": { 'type': int, 'default': 5, 'description': "Edge Logic: Edge death boost (target 0.0) if node neighbor count (prev step) > this.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects on Edges"},
        "edge_death_boost_rate": { 'type': float, 'default': 0.2, 'description': "Edge Logic: Rate edge moves towards 0 if node neighbor counts non-optimal.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Group D: Node Effects on Edges"},

        # === Core Parameters (Repeated for clarity) ===
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood definition.", "parameter_group": "Core"},
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Grid dimension.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},

        # === Initialization Parameters (Repeated for clarity) ===
        "initial_density": { "type": float, 'default': GlobalSettings.Simulation.INITIAL_NODE_DENSITY, "description": "Initial density of active nodes (state=1).", "min": 0.0, "max": 1.0, "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, 'default': "Random", "description": "Method for setting the initial grid state.", "allowed_values": ['Random', 'Glider Pattern', 'Gosper Glider Gun Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability (0-1) for RANDOM edge initialization.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "min_edge_weight": { "type": float, "description": "Minimum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Initialization"},
        "max_edge_weight": { "type": float, "description": "Maximum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.8, "parameter_group": "Initialization"},

        # === Visualization ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (0 or 1).", "default": False, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states.", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization: Nodes"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on their state value (0-1).",
            "default": True, "parameter_group": "Visualization: Edges" # Default True for continuous
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "prism", "parameter_group": "Visualization: Edges",
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # === Other Parameters ===
        "node_history_depth": { 'type': int, 'description': 'Number of previous states to store.', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Life with Dynamic Edges"
        metadata.description = "GoL B/S counts combined with edge sum conditions (birth min, survival range). Continuous edges (0-1) change based on node activity, decay, randomness, and optional probabilistic/node-effect logic groups."
        metadata.category = "Life-Like"
        metadata.tags = ["Life", "Edges", "Dynamic", "Connectivity", "Weighted", "B3/S23", "Probabilistic", "Modular"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Life with Dynamic Edges"
        self._params = {}
        self.requires_post_edge_state_update = False
        self.needs_neighbor_degrees = True # Needed for Group C
        self.needs_neighbor_active_counts = True # Needed for Group D
        # --- ADDED: Ensure edge coloring defaults are set ---
        self._params.setdefault('use_state_coloring_edges', True)
        self._params.setdefault('edge_colormap', 'prism')
        self._params.setdefault('edge_color_norm_vmin', 0.0)
        self._params.setdefault('edge_color_norm_vmax', 1.0)
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute new state based on neighbor counts and edge sum conditions, with optional probabilistic edge effects."""
        logger = logging.getLogger(__name__)
        # detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING # Use passed arg
        node_idx = neighborhood.node_index

        # Fetch base parameters
        birth_neighbor_counts = self.get_param('birth_neighbor_counts', [3], neighborhood=neighborhood)
        survival_neighbor_counts = self.get_param('survival_neighbor_counts', [2, 3], neighborhood=neighborhood)
        birth_min_sum = self.get_param('birth_min_edge_sum', 0.5, neighborhood=neighborhood)
        survival_min_sum = self.get_param('survival_min_edge_sum', 0.3, neighborhood=neighborhood)
        survival_max_sum = self.get_param('survival_max_edge_sum', 2.5, neighborhood=neighborhood)

        # Fetch edge effect parameters
        use_edge_effects = self.get_param('use_edge_effects_on_nodes_group', False, neighborhood=neighborhood)
        survival_boost_thr = self.get_param('node_survival_boost_edge_sum_min', 0.5, neighborhood=neighborhood)
        survival_boost_p = self.get_param('node_survival_boost_prob', 0.1, neighborhood=neighborhood)
        death_penalty_thr = self.get_param('node_death_penalty_edge_sum_max', 0.2, neighborhood=neighborhood)
        death_penalty_p = self.get_param('node_death_penalty_prob', 0.1, neighborhood=neighborhood)

        # Calculate metrics
        num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
        # Sum incoming edge weights from previous step
        incoming_edge_sum = sum(
            neighborhood.neighbor_edge_states.get(idx, 0.0)
            for idx in neighborhood.neighbor_indices if idx >= 0
        )

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state (EdgeEffects={use_edge_effects}) ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State: {neighborhood.node_state:.1f}") # type: ignore [attr-defined]
            logger.detail(f"    Active Neighbors: {num_active_neighbors}") # type: ignore [attr-defined]
            logger.detail(f"    Incoming Edge Sum (Prev): {incoming_edge_sum:.4f}") # type: ignore [attr-defined]

        # --- Determine Base Eligibility ---
        base_eligible = False
        if neighborhood.node_state <= 0: # Birth Condition
            passes_neighbor_check = num_active_neighbors in birth_neighbor_counts
            passes_edge_sum_check = incoming_edge_sum >= birth_min_sum
            base_eligible = passes_neighbor_check and passes_edge_sum_check
            decision_reason = f"Base Birth Check (NeighborsOk={passes_neighbor_check}, EdgeSumOk={passes_edge_sum_check}) -> Eligible={base_eligible}"
        else: # Survival Condition
            passes_neighbor_check = num_active_neighbors in survival_neighbor_counts
            passes_edge_sum_check = survival_min_sum <= incoming_edge_sum <= survival_max_sum
            base_eligible = passes_neighbor_check and passes_edge_sum_check
            decision_reason = f"Base Survival Check (NeighborsOk={passes_neighbor_check}, EdgeSumOk={passes_edge_sum_check}) -> Eligible={base_eligible}"

        if detailed_logging_enabled: logger.detail(f"    {decision_reason}") # type: ignore [attr-defined]

        # --- Apply Optional Edge Effects (Probabilistic Override) ---
        final_eligible = base_eligible
        override_reason = "None"
        if use_edge_effects:
            if not base_eligible and neighborhood.node_state > 0: # Potential Survival Boost (was active, base logic says die)
                if incoming_edge_sum >= survival_boost_thr:
                    if np.random.random() < survival_boost_p:
                        final_eligible = True # Probabilistic survival
                        override_reason = f"SURVIVAL BOOST (EdgeSum={incoming_edge_sum:.2f}>={survival_boost_thr:.2f}, P={survival_boost_p:.2f})"
            elif base_eligible and neighborhood.node_state > 0: # Potential Death Penalty (was active, base logic says survive)
                if incoming_edge_sum < death_penalty_thr:
                    if np.random.random() < death_penalty_p:
                        final_eligible = False # Probabilistic death
                        override_reason = f"DEATH PENALTY (EdgeSum={incoming_edge_sum:.2f}<{death_penalty_thr:.2f}, P={death_penalty_p:.2f})"

        if detailed_logging_enabled and override_reason != "None":
             logger.detail(f"    Override Applied: {override_reason} -> Final Eligible={final_eligible}") # type: ignore [attr-defined]

        new_state = 1.0 if final_eligible else 0.0

        if detailed_logging_enabled:
            logger.detail(f"    New State: {new_state:.1f}") # type: ignore [attr-defined]

        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> Dict[Tuple[int, int], float]:
        """Compute new continuous edge states based on enabled logic groups with priority."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # --- Get Toggle Parameters ---
        use_base_activity = True # Base logic is always considered first
        use_prob_degree = self.get_param('use_probabilistic_degree_edges_group', False, neighborhood=neighborhood)
        use_node_effects = self.get_param('use_node_effects_on_edges_group', False, neighborhood=neighborhood)
        use_decay = True # Base decay is always applied
        use_random = self.get_param('random_edge_change_prob', 0.0) > 0 and self.get_param('random_edge_change_amount', 0.0) > 0

        # --- Get Value Parameters ---
        # Base Activity/Decay (Group A)
        edge_change_rate_base = self.get_param('edge_change_rate', 0.4, neighborhood=neighborhood)
        edge_decay_rate = self.get_param('edge_decay_rate', 0.02, neighborhood=neighborhood)
        # Probabilistic Degree (Group C)
        prob_conn_thr = self.get_param('prob_connect_degree_threshold', 3, neighborhood=neighborhood)
        prob_conn_p = self.get_param('prob_connect_if_above_threshold', 0.1, neighborhood=neighborhood)
        prob_disc_thr = self.get_param('prob_disconnect_degree_threshold', 2, neighborhood=neighborhood)
        prob_disc_p = self.get_param('prob_disconnect_if_below_threshold', 0.2, neighborhood=neighborhood)
        new_edge_strength_prob = self.get_param('new_edge_initial_strength_prob', 0.1, neighborhood=neighborhood)
        # Node Effects (Group D)
        edge_survival_neighbor_min = self.get_param('edge_survival_neighbor_min', 2, neighborhood=neighborhood)
        edge_survival_neighbor_max = self.get_param('edge_survival_neighbor_max', 4, neighborhood=neighborhood)
        edge_survival_boost_rate = self.get_param('edge_survival_boost_rate', 0.1, neighborhood=neighborhood)
        edge_death_neighbor_min = self.get_param('edge_death_neighbor_min', 1, neighborhood=neighborhood)
        edge_death_neighbor_max = self.get_param('edge_death_neighbor_max', 5, neighborhood=neighborhood)
        edge_death_boost_rate = self.get_param('edge_death_boost_rate', 0.2, neighborhood=neighborhood)
        # Randomness (Group F)
        random_edge_prob = self.get_param('random_edge_change_prob', 0.01, neighborhood=neighborhood)
        random_edge_amount = self.get_param('random_edge_change_amount', 0.1, neighborhood=neighborhood)
        # Pruning
        min_keep = self.get_param('min_edge_state_to_keep', 0.02, neighborhood=neighborhood)

        # Determine the *next* state of the current node (proxy 0 or 1)
        next_node_state = self._compute_new_state(neighborhood, detailed_logging_enabled)
        self_is_active_next = next_node_state > 0.5

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Active Next: {self_is_active_next}") # type: ignore [attr-defined]
            logger.detail(f"    Enabled Edge Groups: Base={use_base_activity}, ProbDegree={use_prob_degree}, NodeEffects={use_node_effects}, Decay={use_decay}, Random={use_random}") # type: ignore [attr-defined]

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            edge = (node_idx, neighbor_idx) # Canonical order
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0)

            # Approximation: Use current neighbor state as proxy for neighbor's next state.
            neighbor_is_active_next = neighbor_state > 0.5

            # --- Determine Target State & Change Rate based on Priority ---
            target_edge_state = current_edge_state # Default: maintain
            effective_change_rate = 0.0 # Default: no change
            applied_group = "None"
            prob_override_applied = False # Flag for probabilistic logic

            # 1. Base Target/Rate (Group A - always calculated first if enabled)
            if use_base_activity:
                target_edge_state = 1.0 if (self_is_active_next and neighbor_is_active_next) else 0.0
                effective_change_rate = edge_change_rate_base
                applied_group = "A:BaseActivity"

            # 2. Probabilistic Degree Override (Group C - overrides Group A target)
            if use_prob_degree:
                neighbor_prev_degree = 0
                if neighborhood.neighbor_degrees is not None:
                    neighbor_prev_degree = neighborhood.neighbor_degrees.get(neighbor_idx, 0)
                else: logger.warning(f"Node {node_idx}: neighbor_degrees missing for Group C!")

                if current_edge_state < min_keep: # Check formation
                    if neighbor_prev_degree > prob_conn_thr and np.random.random() < prob_conn_p:
                        target_edge_state = new_edge_strength_prob # Target formation strength
                        # Use base rate for formation? Or a specific one? Using base for now.
                        effective_change_rate = edge_change_rate_base
                        applied_group = "C:ProbDegree(Form)"
                        prob_override_applied = True
                elif current_edge_state >= min_keep: # Check destruction
                    if neighbor_prev_degree < prob_disc_thr and np.random.random() < prob_disc_p:
                        target_edge_state = 0.0 # Target destruction
                        # Use base rate for destruction? Or a specific one? Using base for now.
                        effective_change_rate = edge_change_rate_base
                        applied_group = "C:ProbDegree(Destroy)"
                        prob_override_applied = True

            # 3. Node Effects Override (Group D - overrides Groups A & C target/rate)
            if use_node_effects and not prob_override_applied: # Only if probabilistic didn't override
                self_prev_active_count = neighborhood.neighbor_active_counts.get(node_idx, 0) if neighborhood.neighbor_active_counts else 0
                neighbor_prev_active_count = neighborhood.neighbor_active_counts.get(neighbor_idx, 0) if neighborhood.neighbor_active_counts else 0

                self_survival_boost = (edge_survival_neighbor_min <= self_prev_active_count <= edge_survival_neighbor_max)
                neighbor_survival_boost = (edge_survival_neighbor_min <= neighbor_prev_active_count <= edge_survival_neighbor_max)
                self_death_boost = (self_prev_active_count < edge_death_neighbor_min or self_prev_active_count > edge_death_neighbor_max)
                neighbor_death_boost = (neighbor_prev_active_count < edge_death_neighbor_min or neighbor_prev_active_count > edge_death_neighbor_max)

                if self_survival_boost and neighbor_survival_boost:
                    target_edge_state = 1.0
                    effective_change_rate = edge_survival_boost_rate
                    applied_group = "D:NodeEffects(Surv)"
                elif self_death_boost or neighbor_death_boost:
                    target_edge_state = 0.0
                    effective_change_rate = edge_death_boost_rate
                    applied_group = "D:NodeEffects(Death)"

            # --- Apply Changes ---
            new_edge_state = current_edge_state + (target_edge_state - current_edge_state) * effective_change_rate

            # Apply Decay (Group E)
            if use_decay:
                new_edge_state -= edge_decay_rate

            # Apply Randomness (Group F)
            if use_random and np.random.random() < random_edge_prob:
                change = (np.random.random() * 2 - 1) * random_edge_amount
                new_edge_state += change

            # Clip and Prune
            clipped_new_edge_state = float(np.clip(new_edge_state, 0.0, 1.0))

            if detailed_logging_enabled:
                logger.detail(f"    Edge {edge}: Curr={current_edge_state:.3f}, Target={target_edge_state:.3f}, Rate={effective_change_rate:.3f} (Group={applied_group}) -> New(clipped)={clipped_new_edge_state:.3f}") # type: ignore [attr-defined]

            if clipped_new_edge_state >= min_keep:
                 new_edges[edge] = clipped_new_edge_state

        if detailed_logging_enabled:
            logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges

class LifeWithContinuousEdges(Rule):
    """
    Standard Game of Life (B/S) node logic combined with continuous edge dynamics.
    Edges represent connection strength (0.0-1.0) and evolve based on local activity.

    Node Logic (Binary State 0/1):
    - Follows standard Conway's Game of Life rules based on the count of active neighbors (state > 0) in the previous step.
    - Birth: Inactive node becomes active if active neighbor count is in 'birth_neighbor_counts' (Default: [3]).
    - Survival: Active node remains active if active neighbor count is in 'survival_neighbor_counts' (Default: [2, 3]).
    - Death: Otherwise, node becomes or remains inactive (state 0).

    Edge Logic (Continuous State 0-1):
    - Target State: An edge's target strength is determined by mutual support. If BOTH connected nodes had an active neighbor count >= 'edge_support_neighbor_threshold' in the *previous* step, the target state is 1.0. Otherwise, the target state is 0.0.
    - Inertia: The edge state moves towards its target state at a rate determined by 'edge_change_rate'.
    - Decay: A constant 'edge_decay_rate' is subtracted from the edge state each step.
    - Randomness: A small random fluctuation (+/- 'random_edge_change_amount') can be applied with 'random_edge_change_prob'.
    - Clipping: Edge strength is always kept within the [0.0, 1.0] range.
    - Pruning: Edges with strength below 'min_edge_state_to_keep' are removed entirely (state becomes 0).

    Requires 2nd-order neighbor active counts (from the previous step) for edge logic.
   """

    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'node_history_depth'
    }
    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": {
            'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"],
            'description': "Grid dimension.", "parameter_group": "Core"
        },
        "neighborhood_type": {
            'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"],
            'description': "Neighborhood type.", "parameter_group": "Core"
        },
        'grid_boundary': {
            'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap',
            'description': 'Grid boundary behavior.', "parameter_group": "Core"
        },
        # === Initialization ===
        "edge_initialization": {
            'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM',
            'description': 'Initial edge setup. Edges are continuous (0-1).', "parameter_group": "Initialization"
        },
        "initial_conditions": {
            "type": str, "description": "Initial grid state pattern.", "default": "Random",
            "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'],
            "parameter_group": "Initialization"
        },
        "initial_density": {
            "type": float, "description": "Initial density of active nodes (state=1) for 'Random' condition.",
            "min": 0.0, "max": 1.0, "default": 0.35, "parameter_group": "Initialization"
        },
        "connect_probability": { # Used by RANDOM edge initialization
            "type": float, "description": "Probability (0-1) for RANDOM edge initialization.",
            "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"
        },
        "min_edge_weight": { # Used by RANDOM edge initialization
            "type": float, "description": "Minimum initial random edge weight (for RANDOM init).",
            "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Initialization"
        },
        "max_edge_weight": { # Used by RANDOM edge initialization
            "type": float, "description": "Maximum initial random edge weight (for RANDOM init).",
            "min": 0.0, "max": 1.0, "default": 0.8, "parameter_group": "Initialization"
        },
        # === State Update (GoL Logic) ===
        "birth_neighbor_counts": {
            'type': list, 'element_type': int, 'default': [3],
            'description': "Node Logic: List of active neighbor counts for birth.",
            "parameter_group": "Node Group A: GoL Counts" # Grouped
        },
        "survival_neighbor_counts": {
            'type': list, 'element_type': int, 'default': [2, 3],
            'description': "Node Logic: List of active neighbor counts for survival.",
            "parameter_group": "Node Group A: GoL Counts" # Grouped
        },
        # --- ADDED: Node Logic Group B: Edge Effects on Nodes ---
        "use_edge_effects_on_nodes_group": {
            'type': bool, 'default': False,
            'description': "Node Logic: Enable probabilistic boost/penalty to node survival based on edge sum.",
            "parameter_group": "Node Group B: Edge Effects on Nodes"
        },
        "node_survival_boost_edge_sum_min": {
            'type': float, 'default': 0.5,
            'description': "Node Logic: Min edge sum (prev step) to potentially boost survival probability.",
            "min": 0.0, "max": 26.0, "parameter_group": "Node Group B: Edge Effects on Nodes"
        },
        "node_survival_boost_prob": {
            'type': float, 'default': 0.1,
            'description': "Node Logic: Probability (0-1) to survive death if edge sum is sufficient.",
            "min": 0.0, "max": 1.0, "parameter_group": "Node Group B: Edge Effects on Nodes"
        },
        "node_death_penalty_edge_sum_max": {
            'type': float, 'default': 0.2,
            'description': "Node Logic: Max edge sum (prev step) below which death probability is boosted.",
            "min": 0.0, "max": 26.0, "parameter_group": "Node Group B: Edge Effects on Nodes"
        },
        "node_death_penalty_prob": {
            'type': float, 'default': 0.1,
            'description': "Node Logic: Probability (0-1) to die despite survival conditions if edge sum is too low.",
            "min": 0.0, "max": 1.0, "parameter_group": "Node Group B: Edge Effects on Nodes"
        },
        # --- END ADDED ---

        # === Edge Update ===
        "edge_support_neighbor_threshold": {
            'type': int, 'default': 2,
            'description': "Edge Logic: Min active neighbors (prev step) BOTH nodes need to target edge state 1.0.",
            "min": 0, "max": 26, "parameter_group": "Edge Group A: Base Activity/Support" # Grouped
        },
        "edge_change_rate": {
            "type": float, "description": "Edge Logic: Rate (0-1) at which edge state moves towards target (0 or 1).",
            "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Edge Group A: Base Activity/Support" # Grouped
        },
        "edge_decay_rate": {
            "type": float, "description": "Edge Logic: Constant decay subtracted from edge state each step.",
            "min": 0.0, "max": 0.1, "default": 0.02, "parameter_group": "Edge Group B: Decay & Randomness" # Grouped
        },
        "random_edge_change_prob": {
            "type": float, "description": "Edge Logic: Probability (0-1) of applying a small random change to edge state.",
            "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Group B: Decay & Randomness" # Grouped
        },
         "random_edge_change_amount": {
            "type": float, "description": "Edge Logic: Max amount (+/-) of random change applied to edge state.",
            "min": 0.0, "max": 0.5, "default": 0.1, "parameter_group": "Edge Group B: Decay & Randomness" # Grouped
        },
        "min_edge_state_to_keep": {
            "type": float, "description": "Edge Logic: Minimum edge state required to be kept (non-zero). Edges below this are removed.",
            "min": 0.0, "max": 0.5, "default": 0.05, "parameter_group": "Edge Logic: Pruning" # Grouped
        },
        # --- ADDED: Edge Logic Group C: Node Effects on Edges ---
        "use_node_effects_on_edges_group": {
            'type': bool, 'default': False,
            'description': "Edge Logic: Enable neighbor count influencing edge target/rate (uses prev step counts). Overrides Group A target/rate.",
            "parameter_group": "Edge Group C: Node Effects on Edges"
        },
        "edge_survival_neighbor_min": {
            'type': int, 'default': 2,
            'description': "Edge Logic: Min active neighbors (prev step) for nodes to boost edge survival (target 1.0).",
            "min": 0, "max": 26, "parameter_group": "Edge Group C: Node Effects on Edges"
        },
        "edge_survival_neighbor_max": {
            'type': int, 'default': 4,
            'description': "Edge Logic: Max active neighbors (prev step) for nodes to boost edge survival (target 1.0).",
            "min": 0, "max": 26, "parameter_group": "Edge Group C: Node Effects on Edges"
        },
        "edge_survival_boost_rate": {
            'type': float, 'default': 0.1,
            'description': "Edge Logic: Rate edge moves towards 1 if node neighbor counts optimal.",
            "min": 0.0, "max": 1.0, "parameter_group": "Edge Group C: Node Effects on Edges"
        },
        "edge_death_neighbor_min": {
            'type': int, 'default': 1,
            'description': "Edge Logic: Edge death boost (target 0.0) if node neighbor count (prev step) < this.",
            "min": 0, "max": 26, "parameter_group": "Edge Group C: Node Effects on Edges"
        },
        "edge_death_neighbor_max": {
            'type': int, 'default': 5,
            'description': "Edge Logic: Edge death boost (target 0.0) if node neighbor count (prev step) > this.",
            "min": 0, "max": 26, "parameter_group": "Edge Group C: Node Effects on Edges"
        },
        "edge_death_boost_rate": {
            'type': float, 'default': 0.2,
            'description': "Edge Logic: Rate edge moves towards 0 if node neighbor counts non-optimal.",
            "min": 0.0, "max": 1.0, "parameter_group": "Edge Group C: Node Effects on Edges"
        },
        # --- END ADDED ---

        # === Visualization ===
        "use_state_coloring": {
            "type": bool, "description": "Color nodes based on state (0 or 1).",
            "default": False, "parameter_group": "Visualization: Nodes" # Grouped
        },
        "color_nodes_by_degree": { # Added
            "type": bool, "description": "If Use State Coloring is True, color nodes based on connection count (degree) in the current step.",
            "default": False, "parameter_group": "Visualization: Nodes"
        },
        "color_nodes_by_active_neighbors": { # Added
            "type": bool, "description": "If Use State Coloring is True, color nodes based on active neighbor count in the previous step.",
            "default": False, "parameter_group": "Visualization: Nodes"
        },
        "node_colormap": {
            "type": str, "description": "Colormap for node states.",
            "default": "prism", "parameter_group": "Visualization: Nodes",
            "allowed_values": _standard_colormaps
        },
        "node_color_norm_vmin": {
            "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..",
            "default": 0.0, "parameter_group": "Visualization: Nodes"
        },
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 8.0, "parameter_group": "Visualization: Nodes"},
        "use_state_coloring_edges": { # Added
            "type": bool, "description": "Color edges based on their state value (0-1).",
            "default": True, "parameter_group": "Visualization: Edges"
        },
        "edge_colormap": { # Added
            "type": str, "description": "Colormap for edge coloring.",
            "default": "prism", "parameter_group": "Visualization: Edges",
            "allowed_values": _standard_colormaps
        },
        "edge_color_norm_vmin": { # Added
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 8.0, "parameter_group": "Visualization: Edges"},

        # === Other ===
        "tiebreaker_type": {
            "type": str, "description": "Tiebreaker method (Not typically relevant for continuous edges).",
            "allowed_values": ["RANDOM"], "default": "RANDOM", "parameter_group": "Tiebreaker"
        },
        "node_history_depth": {
            'type': int, 'description': 'Number of previous states to store.',
            'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'
        }
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Life with Continuous Edges" # New Name
        metadata.description = "Standard GoL B/S node logic, optionally modified by edge sum. Continuous edges (0-1) strengthen based on mutual neighbor support (prev step), decay, randomness. Optional node effects on edges." # Updated description
        metadata.category = "Life-Like" # Keep category
        metadata.tags = ["Life", "Edges", "Continuous", "Dynamic", "Connectivity", "B3/S23", "Modular"] # Updated tags
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Life with Continuous Edges" # Set name explicitly
        self._params = {}
        # --- MODIFIED: Set requires_post_edge_state_update based on node coloring flags ---
        self.requires_post_edge_state_update = self.get_param('color_nodes_by_degree', False) or \
                                               self.get_param('color_nodes_by_active_neighbors', False)
        # ---
        # --- Set flags to request data ---
        self.needs_neighbor_degrees = False # Still not needed for core logic
        self.needs_neighbor_active_counts = True # Needed for Group D edge logic and potentially node coloring
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute new state using standard B3/S23 rules, optionally modified by edge sum effects."""
        logger = logging.getLogger(__name__)
        # detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING # Use passed arg
        node_idx = neighborhood.node_index

        # Fetch base GoL parameters
        birth_rules = self.get_param('birth_neighbor_counts', [3], neighborhood=neighborhood)
        survival_rules = self.get_param('survival_neighbor_counts', [2, 3], neighborhood=neighborhood)

        # Fetch edge effect parameters
        use_edge_effects = self.get_param('use_edge_effects_on_nodes_group', False, neighborhood=neighborhood)
        survival_boost_thr = self.get_param('node_survival_boost_edge_sum_min', 0.5, neighborhood=neighborhood)
        survival_boost_p = self.get_param('node_survival_boost_prob', 0.1, neighborhood=neighborhood)
        death_penalty_thr = self.get_param('node_death_penalty_edge_sum_max', 0.2, neighborhood=neighborhood)
        death_penalty_p = self.get_param('node_death_penalty_prob', 0.1, neighborhood=neighborhood)

        # Count *active neighbors* (state > 0 in previous step)
        num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
        # --- Store active neighbor count for potential use in _compute_final_state ---
        neighborhood.rule_params['_active_neighbor_count'] = num_active_neighbors
        # ---

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state (EdgeEffects={use_edge_effects}) ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State: {neighborhood.node_state:.1f}") # type: ignore [attr-defined]
            logger.detail(f"    Active Neighbors: {num_active_neighbors}") # type: ignore [attr-defined]

        # --- Determine Base Eligibility (GoL) ---
        base_eligible = False
        if neighborhood.node_state <= 0: # Birth check
            base_eligible = num_active_neighbors in birth_rules
            decision_reason = f"Base Birth Check (NeighborsOk={base_eligible})"
        else: # Survival check
            base_eligible = num_active_neighbors in survival_rules
            decision_reason = f"Base Survival Check (NeighborsOk={base_eligible})"

        if detailed_logging_enabled: logger.detail(f"    {decision_reason}") # type: ignore [attr-defined]

        # --- Apply Optional Edge Effects (Probabilistic Override) ---
        final_eligible = base_eligible
        override_reason = "None"
        if use_edge_effects:
            # Calculate incoming edge sum from PREVIOUS step
            incoming_edge_sum = sum(
                neighborhood.neighbor_edge_states.get(idx, 0.0)
                for idx in neighborhood.neighbor_indices if idx >= 0
            )
            if detailed_logging_enabled: logger.detail(f"    Edge Effect Check: Incoming Edge Sum (Prev): {incoming_edge_sum:.4f}") # type: ignore [attr-defined]

            if not base_eligible and neighborhood.node_state > 0: # Potential Survival Boost
                if incoming_edge_sum >= survival_boost_thr:
                    if np.random.random() < survival_boost_p:
                        final_eligible = True # Probabilistic survival
                        override_reason = f"SURVIVAL BOOST (EdgeSum={incoming_edge_sum:.2f}>={survival_boost_thr:.2f}, P={survival_boost_p:.2f})"
            elif base_eligible and neighborhood.node_state > 0: # Potential Death Penalty
                if incoming_edge_sum < death_penalty_thr:
                    if np.random.random() < death_penalty_p:
                        final_eligible = False # Probabilistic death
                        override_reason = f"DEATH PENALTY (EdgeSum={incoming_edge_sum:.2f}<{death_penalty_thr:.2f}, P={death_penalty_p:.2f})"

        if detailed_logging_enabled and override_reason != "None":
             logger.detail(f"    Override Applied: {override_reason} -> Final Eligible={final_eligible}") # type: ignore [attr-defined]

        new_state = 1.0 if final_eligible else 0.0

        if detailed_logging_enabled:
            logger.detail(f"    New State (Eligibility Proxy): {new_state:.1f}") # type: ignore [attr-defined]

        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> Dict[Tuple[int, int], float]:
        """Edges strengthen/form based on mutual support, decay otherwise. Optional node effects override."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # --- Get Toggle Parameters ---
        use_base_logic = True # Base logic based on support threshold is always active
        use_node_effects = self.get_param('use_node_effects_on_edges_group', False, neighborhood=neighborhood)
        use_decay = True # Base decay is always applied
        use_random = self.get_param('random_edge_change_prob', 0.0) > 0 and self.get_param('random_edge_change_amount', 0.0) > 0

        # --- Get Value Parameters ---
        # Base Logic (Group A)
        support_thr = self.get_param('edge_support_neighbor_threshold', 2, neighborhood=neighborhood)
        edge_change_rate_base = self.get_param('edge_change_rate', 0.4, neighborhood=neighborhood)
        # Node Effects (Group D)
        edge_survival_neighbor_min = self.get_param('edge_survival_neighbor_min', 2, neighborhood=neighborhood)
        edge_survival_neighbor_max = self.get_param('edge_survival_neighbor_max', 4, neighborhood=neighborhood)
        edge_survival_boost_rate = self.get_param('edge_survival_boost_rate', 0.1, neighborhood=neighborhood)
        edge_death_neighbor_min = self.get_param('edge_death_neighbor_min', 1, neighborhood=neighborhood)
        edge_death_neighbor_max = self.get_param('edge_death_neighbor_max', 5, neighborhood=neighborhood)
        edge_death_boost_rate = self.get_param('edge_death_boost_rate', 0.2, neighborhood=neighborhood)
        # Decay & Randomness (Groups E & F)
        edge_decay_rate = self.get_param('edge_decay_rate', 0.02, neighborhood=neighborhood)
        random_edge_prob = self.get_param('random_edge_change_prob', 0.01, neighborhood=neighborhood)
        random_edge_amount = self.get_param('random_edge_change_amount', 0.1, neighborhood=neighborhood)
        min_keep = self.get_param('min_edge_state_to_keep', 0.05, neighborhood=neighborhood)

        # Get current node's active neighbor count from *previous* step
        self_prev_active_count = -1
        if use_node_effects and neighborhood.neighbor_active_counts is not None:
            self_prev_active_count = neighborhood.neighbor_active_counts.get(node_idx, 0)
        elif use_node_effects:
            logger.warning(f"Node {node_idx}: neighbor_active_counts missing for Node Effects check.")

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]
            logger.detail(f"    Enabled Edge Groups: BaseSupport={use_base_logic}, NodeEffects={use_node_effects}, Decay={use_decay}, Random={use_random}") # type: ignore [attr-defined]
            logger.detail(f"    Self Prev Active Neighbors: {self_prev_active_count}") # type: ignore [attr-defined]

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            edge = (node_idx, neighbor_idx) # Canonical order
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0)

            # Get neighbor's active neighbor count from the *previous* step
            neighbor_prev_active_count = -1
            if use_node_effects and neighborhood.neighbor_active_counts is not None:
                 neighbor_prev_active_count = neighborhood.neighbor_active_counts.get(neighbor_idx, 0)
            elif use_node_effects:
                 logger.warning(f"Node {node_idx}: neighbor_active_counts missing for neighbor {neighbor_idx}.")

            # --- Determine Target Edge State & Change Rate based on Priority ---
            target_edge_state = 0.0 # Default target is decay/removal
            effective_change_rate = edge_change_rate_base # Default rate
            applied_group = "A:BaseSupport" # Default group

            # 1. Base Target from Mutual Support (Group A)
            self_supported = self_prev_active_count >= support_thr if self_prev_active_count != -1 else False
            neighbor_supported = neighbor_prev_active_count >= support_thr if neighbor_prev_active_count != -1 else False
            target_edge_state = 1.0 if (self_supported and neighbor_supported) else 0.0

            # 2. Node Effects Override (Group D - Highest Priority)
            if use_node_effects and self_prev_active_count != -1 and neighbor_prev_active_count != -1:
                 self_survival_boost = (edge_survival_neighbor_min <= self_prev_active_count <= edge_survival_neighbor_max)
                 neighbor_survival_boost = (edge_survival_neighbor_min <= neighbor_prev_active_count <= edge_survival_neighbor_max)
                 self_death_boost = (self_prev_active_count < edge_death_neighbor_min or self_prev_active_count > edge_death_neighbor_max)
                 neighbor_death_boost = (neighbor_prev_active_count < edge_death_neighbor_min or neighbor_prev_active_count > edge_death_neighbor_max)

                 if self_survival_boost and neighbor_survival_boost:
                      target_edge_state = 1.0 # Override target
                      effective_change_rate = edge_survival_boost_rate # Override rate
                      applied_group = "D:NodeEffects(Surv)"
                 elif self_death_boost or neighbor_death_boost:
                      target_edge_state = 0.0 # Override target
                      effective_change_rate = edge_death_boost_rate # Override rate
                      applied_group = "D:NodeEffects(Death)"
                 # Else: Keep target/rate from Group A

            # --- Apply Changes ---
            new_edge_state = current_edge_state + (target_edge_state - current_edge_state) * effective_change_rate

            # Apply Decay (Group E)
            if use_decay:
                new_edge_state -= edge_decay_rate

            # Apply Randomness (Group F)
            if use_random and np.random.random() < random_edge_prob:
                change = (np.random.random() * 2 - 1) * random_edge_amount
                new_edge_state += change

            # Clip and Prune
            clipped_new_edge_state = float(np.clip(new_edge_state, 0.0, 1.0))

            if detailed_logging_enabled:
                logger.detail(f"    Edge {edge}: Curr={current_edge_state:.3f}, Target={target_edge_state:.1f}, Rate={effective_change_rate:.3f} (Group={applied_group}) -> New(clipped)={clipped_new_edge_state:.3f}") # type: ignore [attr-defined]

            if clipped_new_edge_state >= min_keep:
                 new_edges[edge] = clipped_new_edge_state

        if detailed_logging_enabled:
            logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges

class WeightedEdgeInfluenceLife(Rule):
    """
    Node state (binary 0/1) depends on the sum of incoming edge weights from active neighbors.
    Edge state (continuous 0-1) evolves based on either node similarity or average node activity.
    (Round 9 Fix: Added missing visualization parameters)
    """
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'color_edges_by_neighbor_degree', # Edge state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edge state not based on this
        'tiebreaker_type', 'node_history_depth'
    }

    node_state_type: ClassVar[StateType] = StateType.BINARY # Nodes are 0/1
    edge_state_type: ClassVar[StateType] = StateType.REAL   # Edges are 0.0-1.0
    min_node_state: ClassVar[float] = 0.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": {
            'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"],
            'description': "Grid dimension.", "parameter_group": "Core"
        },
        "neighborhood_type": {
            'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"],
            'description': "Neighborhood type.", "parameter_group": "Core"
        },
        'grid_boundary': {
            'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap',
            'description': 'Grid boundary behavior.', "parameter_group": "Core"
        },
        # === Initialization ===
        "edge_initialization": {
            'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM',
            'description': 'Initial edge setup.', "parameter_group": "Initialization"
        },
        "initial_conditions": {
            "type": str, "description": "Initial grid state pattern.", "default": "Random",
            "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'],
            "parameter_group": "Initialization"
        },
        "initial_density": {
            "type": float, "description": "Initial density of active nodes (state=1) for 'Random' condition.",
            "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Initialization"
        },
        "connect_probability": {
            "type": float, "description": "Probability (0-1) for RANDOM edge initialization.",
            "min": 0.0, "max": 1.0, "default": 0.6, "parameter_group": "Initialization"
        },
        "min_edge_weight": {
            "type": float, "description": "Minimum initial random edge weight (for RANDOM init).",
            "min": 0.0, "max": 1.0, "default": 0.2, "parameter_group": "Initialization"
        },
        "max_edge_weight": {
            "type": float, "description": "Maximum initial random edge weight (for RANDOM init).",
            "min": 0.0, "max": 1.0, "default": 0.8, "parameter_group": "Initialization"
        },
        # === State Update (Based on Weighted Connection Sum) ===
        "birth_min_weight_sum": {
            'type': float, 'default': 2.5,
            'description': "Node Logic: Min sum of edge weights to active neighbors for birth.",
            "min": 0.0, "max": 26.0, "parameter_group": "Node Logic" # Max depends on neighborhood
        },
        "birth_max_weight_sum": {
            'type': float, 'default': 3.5,
            'description': "Node Logic: Max sum of edge weights to active neighbors for birth.",
            "min": 0.0, "max": 26.0, "parameter_group": "Node Logic"
        },
        "survival_min_weight_sum": {
            'type': float, 'default': 1.5,
            'description': "Node Logic: Min sum of edge weights to active neighbors for survival.",
            "min": 0.0, "max": 26.0, "parameter_group": "Node Logic"
        },
        "survival_max_weight_sum": {
            'type': float, 'default': 3.5,
            'description': "Node Logic: Max sum of edge weights to active neighbors for survival.",
            "min": 0.0, "max": 26.0, "parameter_group": "Node Logic"
        },
        # === Edge Update Logic ===
        "use_similarity_edge_logic": { # NEW Toggle
            'type': bool, 'default': False,
            'description': "Edge Logic: If True, use similarity/dissimilarity; If False, use average node state.",
            "parameter_group": "Edge Logic Mode"
        },
        # Parameters for Average State Mode (use_similarity_edge_logic = False)
        "edge_target_factor": {
            "type": float, "description": "Avg State Mode: Multiplier for avg node state to get target edge state.",
            "min": 0.0, "max": 2.0, "default": 1.0, "parameter_group": "Edge Logic: Average State Mode"
        },
        # Parameters for Similarity Mode (use_similarity_edge_logic = True)
        "similarity_threshold": {
            "type": float, "description": "Similarity Mode: Max state difference to consider nodes 'similar'.",
            "min": 0.0, "max": 1.0, "default": 0.3, "parameter_group": "Edge Logic: Similarity Mode"
        },
        "similarity_strengthen_factor": {
            "type": float, "description": "Similarity Mode: Factor influencing rate towards 1.0 when nodes are similar.",
            "min": 0.0, "max": 1.0, "default": 1.0, "parameter_group": "Edge Logic: Similarity Mode"
        },
        "dissimilarity_decay_factor": {
            "type": float, "description": "Similarity Mode: Factor influencing rate towards 0.0 when nodes are dissimilar.",
            "min": 0.0, "max": 1.0, "default": 1.0, "parameter_group": "Edge Logic: Similarity Mode"
        },
        # Common Edge Parameters
        "edge_change_rate": {
            "type": float, "description": "Edge Logic: Base rate (0-1) at which edge state moves towards its target.",
            "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Edge Logic: General"
        },
        "edge_decay_rate": {
            "type": float, "description": "Edge Logic: Constant decay subtracted from edge state each step.",
            "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Logic: General"
        },
        "random_edge_change_prob": {
            "type": float, "description": "Edge Logic: Probability (0-1) of applying a small random change to edge state.",
            "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Logic: General"
        },
         "random_edge_change_amount": {
            "type": float, "description": "Edge Logic: Max amount (+/-) of random change applied to edge state.",
            "min": 0.0, "max": 0.5, "default": 0.1, "parameter_group": "Edge Logic: General"
        },
        "min_edge_state_to_keep": {
            "type": float, "description": "Edge Logic: Minimum edge state required to be kept (non-zero). Edges below this are removed.",
            "min": 0.0, "max": 0.5, "default": 0.05, "parameter_group": "Edge Logic: General"
        },
        # === Visualization ===
        "use_state_coloring": {
            "type": bool, "description": "Color nodes based on state (0 or 1).",
            "default": False, "parameter_group": "Visualization: Nodes" # Changed group
        },
        # --- ADDED Node Visualization Params ---
        "node_colormap": {
            "type": str, "description": "Colormap for node states.",
            "default": "prism", "parameter_group": "Visualization: Nodes",
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "node_color_norm_vmin": {
            "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..",
            "default": 0.0, "parameter_group": "Visualization: Nodes"
        },
        "node_color_norm_vmax": {
            "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..",
            "default": 1.0, "parameter_group": "Visualization: Nodes" # Correct default for binary node state
        },
        # --- END ADDED ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on their state value (0-1).",
            "default": True, "parameter_group": "Visualization: Edges" # Changed group
        },
        # --- ADDED Edge Visualization Params ---
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring.",
            "default": "prism", "parameter_group": "Visualization: Edges",
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": {
            "type": float, "description": "Max value for edge color normalization.",
            "default": 1.0, "parameter_group": "Visualization: Edges" # Correct default for 0-1 edge state
        },
        # --- END ADDED ---
        # === Other ===
        "tiebreaker_type": {
            "type": str, "description": "Tiebreaker method (Not typically relevant).",
            "allowed_values": ["RANDOM"], "default": "RANDOM", "parameter_group": "Tiebreaker"
        },
        "node_history_depth": {
            'type': int, 'description': 'Number of previous states to store.',
            'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'
        }
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Weighted Edge Influence Life" # New Name
        metadata.description = "Node state (0/1) depends on sum of edge weights from active neighbors. Continuous edges (0-1) evolve based on either node similarity or average node activity, plus decay and randomness." # Updated description
        metadata.category = "Life-Like" # Keep category
        metadata.tags = ["Life", "Edges", "Weighted", "Continuous", "Dynamic", "Connectivity", "Similarity", "Experimental"] # Updated tags
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Weighted Edge Influence Life"
        self._params = {}
        self.requires_post_edge_state_update = False # State determined directly

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute new state based on sum of edge weights to active neighbors."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        birth_min_w = self.get_param('birth_min_weight_sum', 2.5, neighborhood=neighborhood)
        birth_max_w = self.get_param('birth_max_weight_sum', 3.5, neighborhood=neighborhood)
        survival_min_w = self.get_param('survival_min_weight_sum', 1.5, neighborhood=neighborhood)
        survival_max_w = self.get_param('survival_max_weight_sum', 3.5, neighborhood=neighborhood)

        # Sum edge strengths ONLY to active neighbors
        sum_edge_strength = sum(
            neighborhood.neighbor_edge_states.get(idx, 0.0)
            for idx, state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states)
            if idx >= 0 and state > 0 # Only edges to ACTIVE neighbors (state > 0)
        )

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State: {neighborhood.node_state:.1f}") # type: ignore [attr-defined]
            logger.detail(f"    Sum Edge Strength (to active neighbors): {sum_edge_strength:.4f}") # type: ignore [attr-defined]
            logger.detail(f"    Params: BirthW=[{birth_min_w:.2f}-{birth_max_w:.2f}], SurvivalW=[{survival_min_w:.2f}-{survival_max_w:.2f}]") # type: ignore [attr-defined]

        new_state = 0.0 # Default to death
        decision_reason = "Default (Death/Inactive)"

        if neighborhood.node_state <= 0: # --- Birth Condition ---
            if birth_min_w <= sum_edge_strength <= birth_max_w:
                new_state = 1.0
                decision_reason = f"Birth (EdgeSum={sum_edge_strength:.4f} in [{birth_min_w:.4f}-{birth_max_w:.4f}])"
            else:
                decision_reason = f"Remain Dead (EdgeSum={sum_edge_strength:.4f} OUT of [{birth_min_w:.4f}-{birth_max_w:.4f}])"
        else: # --- Survival Condition ---
            if survival_min_w <= sum_edge_strength <= survival_max_w:
                new_state = 1.0
                decision_reason = f"Survival (EdgeSum={sum_edge_strength:.4f} in [{survival_min_w:.4f}-{survival_max_w:.4f}])"
            else:
                decision_reason = f"Death (EdgeSum={sum_edge_strength:.4f} OUT of [{survival_min_w:.4f}-{survival_max_w:.4f}])"

        if detailed_logging_enabled:
            logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    New State: {new_state:.1f}") # type: ignore [attr-defined]

        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> Dict[Tuple[int, int], float]:
        """Compute new continuous edge states based on selected logic mode."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch common parameters
        edge_change_rate = self.get_param('edge_change_rate', 0.4, neighborhood=neighborhood)
        edge_decay_rate = self.get_param('edge_decay_rate', 0.01, neighborhood=neighborhood)
        random_edge_prob = self.get_param('random_edge_change_prob', 0.01, neighborhood=neighborhood)
        random_edge_amount = self.get_param('random_edge_change_amount', 0.1, neighborhood=neighborhood)
        min_keep = self.get_param('min_edge_state_to_keep', 0.05, neighborhood=neighborhood)
        use_similarity_logic = self.get_param('use_similarity_edge_logic', False, neighborhood=neighborhood)

        # Fetch mode-specific parameters
        if use_similarity_logic:
            sim_threshold = self.get_param('similarity_threshold', 0.3, neighborhood=neighborhood)
            sim_strengthen = self.get_param('similarity_strengthen_factor', 1.0, neighborhood=neighborhood)
            dissim_decay = self.get_param('dissimilarity_decay_factor', 1.0, neighborhood=neighborhood)
        else:
            avg_target_factor = self.get_param('edge_target_factor', 1.0, neighborhood=neighborhood)
            # formation_thr = self.get_param('edge_formation_threshold', 0.45, neighborhood=neighborhood) # Not used in this rule

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]
            logger.detail(f"    Using Similarity Logic: {use_similarity_logic}") # type: ignore [attr-defined]

        # Determine the *next* state of the current node (proxy 0 or 1)
        next_node_state = self._compute_new_state(neighborhood, detailed_logging_enabled)

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            edge = (node_idx, neighbor_idx) # Canonical order
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0) # Default 0 if no edge

            # Approximation: Use current neighbor state as proxy for neighbor's next state.
            neighbor_next_state_proxy = 1.0 if neighbor_state > 0 else 0.0

            # --- Determine Target Edge State ---
            target_edge_state = 0.0 # Default target is weak/removed
            mode_specific_rate_factor = 1.0 # Factor to potentially modify base change rate

            if use_similarity_logic:
                state_diff = abs(next_node_state - neighbor_next_state_proxy) # Difference between predicted next states
                if state_diff < sim_threshold: # Similar
                    target_edge_state = 1.0
                    mode_specific_rate_factor = sim_strengthen
                    # if detailed_logging_enabled: logger.detail(f"    Neighbor {neighbor_idx}: Similar (Diff={state_diff:.2f} < {sim_threshold:.2f}). Target=1.0, RateFactor={mode_specific_rate_factor:.2f}") # type: ignore [attr-defined] # Reduce noise
                else: # Dissimilar
                    target_edge_state = 0.0
                    mode_specific_rate_factor = dissim_decay
                    # if detailed_logging_enabled: logger.detail(f"    Neighbor {neighbor_idx}: Dissimilar (Diff={state_diff:.2f} >= {sim_threshold:.2f}). Target=0.0, RateFactor={mode_specific_rate_factor:.2f}") # type: ignore [attr-defined] # Reduce noise
            else: # Average State Logic
                # Target based on average state if BOTH nodes are active
                if next_node_state > 0 and neighbor_next_state_proxy > 0:
                    avg_node_state = (next_node_state + neighbor_next_state_proxy) / 2.0
                    target_edge_state = avg_node_state * avg_target_factor
                    # if detailed_logging_enabled: logger.detail(f"    Neighbor {neighbor_idx}: Avg State Logic. Avg={avg_node_state:.2f}, Factor={avg_target_factor:.2f} -> Target={target_edge_state:.4f}") # type: ignore [attr-defined] # Reduce noise
                else:
                    target_edge_state = 0.0 # Target 0 if one or both inactive
                    # if detailed_logging_enabled: logger.detail(f"    Neighbor {neighbor_idx}: Avg State Logic. Inactive -> Target=0.0") # type: ignore [attr-defined] # Reduce noise
                mode_specific_rate_factor = 1.0 # Use base change rate

            # Calculate effective change rate
            effective_change_rate = edge_change_rate * mode_specific_rate_factor

            # Move edge state towards target state
            new_edge_state = current_edge_state + (target_edge_state - current_edge_state) * effective_change_rate

            # Apply constant decay
            new_edge_state -= edge_decay_rate

            # Apply random change
            if np.random.random() < random_edge_prob:
                change = (np.random.random() * 2 - 1) * random_edge_amount
                new_edge_state += change

            clipped_new_edge_state = float(np.clip(new_edge_state, 0.0, 1.0))

            # if detailed_logging_enabled: # Reduce noise
                # logger.detail(f"      Edge ({edge[0]},{edge[1]}): Curr={current_edge_state:.4f}, Target={target_edge_state:.4f}, EffRate={effective_change_rate:.2f}, Decay={edge_decay_rate:.3f} -> New(clipped)={clipped_new_edge_state:.4f}") # type: ignore [attr-defined]

            # Only propose edges with a state greater than the minimum threshold
            if clipped_new_edge_state >= min_keep:
                 new_edges[edge] = clipped_new_edge_state

        # if detailed_logging_enabled: # Reduce noise
            # logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges
    
class ResourceCompetitionLife(Rule):
    """
    Game of Life variant where survival depends on resource intake (edge strength).
    Nodes follow standard GoL birth rules, but survival requires both GoL neighbor counts
    AND a minimum sum of incoming edge weights from active neighbors. Edges decay
    but are reinforced by connected active nodes.
    (Round 39: Added edge visualization parameters)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        # 'use_state_coloring_edges', # Keep
        # 'edge_colormap', # Keep
        # 'edge_color_norm_vmin', # Keep
        # 'edge_color_norm_vmax', # Keep
        'color_edges_by_neighbor_degree', # Edge state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edge state not based on this
        'tiebreaker_type', 'node_history_depth'
    }
    # --- END MODIFIED ---

    node_state_type: ClassVar[StateType] = StateType.REAL # Nodes are 0.0-1.0
    edge_state_type: ClassVar[StateType] = StateType.REAL # Edges are 0.0-1.0
    min_node_state: ClassVar[float] = 0.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern (Nodes initialized to 0).", "default": "Random", "allowed_values": ['Random', 'ShapeShifting'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Initial density of nodes with state > 0.", "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Initialization"},
        "initial_state_min": { "type": float, "description": "Min initial state value (0-1) for active nodes.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "initial_state_max": { "type": float, "description": "Max initial state value (0-1) for active nodes.", "min": 0.0, "max": 1.0, "default": 0.9, "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability for RANDOM edge initialization.", "min": 0.0, "max": 1.0, "default": 0.6, "parameter_group": "Initialization"},
        "min_edge_weight": { "type": float, "description": "Minimum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.3, "parameter_group": "Initialization"},
        "max_edge_weight": { "type": float, "description": "Maximum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.7, "parameter_group": "Initialization"},

        # === State Update (Resource Model) ===
        "consumption_rate": { 'type': float, 'default': 0.1, 'description': "Node Logic: Factor determining how much resource is transferred per edge strength unit.", "min": 0.0, "max": 0.5, "parameter_group": "Node Logic: Resource Dynamics"},
        "initial_birth_state": { 'type': float, 'default': 0.1, 'description': "Node Logic: Initial state value (resource level) assigned to a newly born node.", "min": 0.0, "max": 1.0, "parameter_group": "Node Logic: Resource Dynamics"},
        "birth_min_potential_intake": { 'type': float, 'default': 0.05, 'description': "Node Logic: Minimum potential resource intake (from neighbors prev state) required for birth.", "min": 0.0, "max": 5.0, "parameter_group": "Node Logic: Resource Dynamics"},
        "survival_min_self_state": { 'type': float, 'default': 0.01, 'description': "Node Logic: Minimum intermediate state (after consumption/gain) required for survival.", "min": 0.0, "max": 1.0, "parameter_group": "Node Logic: Resource Dynamics"},

        # === Edge Update (Competitive) ===
        "use_competitive_edge_formation": { 'type': bool, 'default': True, 'description': "Edge Logic: If True, edge formation probability depends on neighbor load and relative strength.", "parameter_group": "Edge Logic: Competitive Formation"},
        "base_connect_prob": { "type": float, "description": "Edge Logic: Base probability (0-1) to attempt connection between surviving nodes.", "min": 0.0, "max": 1.0, "default": 0.2, "parameter_group": "Edge Logic: Competitive Formation"},
        "max_load_factor": { "type": float, "description": "Edge Logic: Neighbor load divisor (higher = less competition effect).", "min": 0.1, "max": 26.0, "default": 5.0, "parameter_group": "Edge Logic: Competitive Formation"},
        "strength_diff_factor": { "type": float, "description": "Edge Logic: Multiplier for node state difference effect on connection probability.", "min": 0.0, "max": 2.0, "default": 0.5, "parameter_group": "Edge Logic: Competitive Formation"},
        "new_edge_initial_strength": { "type": float, "description": "Edge Logic: Initial strength (0-1) of a newly formed edge.", "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Edge Logic: General"},
        "growth_rate": { "type": float, "description": "Edge Logic: Amount edge strengthens if both nodes survive.", "min": 0.0, "max": 0.5, "default": 0.05, "parameter_group": "Edge Logic: General"},
        "decay_rate": { "type": float, "description": "Edge Logic: Amount edge weakens if one node dies.", "min": 0.0, "max": 0.2, "default": 0.1, "parameter_group": "Edge Logic: General"},
        "constant_decay_rate": { "type": float, "description": "Edge Logic: Constant decay subtracted from edge state each step.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Logic: General"},
        "min_edge_state_to_keep": { "type": float, "description": "Edge Logic: Minimum edge state required to be kept.", "min": 0.0, "max": 0.5, "default": 0.01, "parameter_group": "Edge Logic: General"},

        # === Visualization ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (resource level).", "default": True, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states.", "default": "YlGn", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization: Nodes"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on their state value (connection strength).",
            "default": True, "parameter_group": "Visualization: Edges" # Default True for continuous
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "prism", "parameter_group": "Visualization: Edges",
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # === Other ===
        "node_history_depth": { 'type': int, 'description': 'Grid Setting: History depth (not used by rule).', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Resource Competition Life"
        metadata.description = "Continuous node state (0-1) represents resources. Nodes consume/gain from neighbors. Birth/Survival depends on intake/self-state. Continuous edges (0-1) form competitively based on neighbor load/relative strength, and grow/decay." # Updated description
        metadata.category = "Continuous" # Changed category
        metadata.tags = ["Continuous", "Resource", "Competition", "Edges", "Weighted", "Connectivity", "Experimental"] # Updated tags
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Resource Competition Life"
        self._params = {}
        self.requires_post_edge_state_update = False # State determined directly
        self.needs_neighbor_degrees = True # Need neighbor state (degree proxy) for edge competition
        self.needs_neighbor_active_counts = False # Not directly used

        # --- ADDED: Ensure visualization params have defaults ---
        self._params.setdefault('use_state_coloring', True)
        self._params.setdefault('node_colormap', 'YlGn')
        self._params.setdefault('node_color_norm_vmin', 0.0)
        self._params.setdefault('node_color_norm_vmax', 1.0)
        self._params.setdefault('use_state_coloring_edges', True)
        self._params.setdefault('edge_colormap', 'prism')
        self._params.setdefault('edge_color_norm_vmin', 0.0)
        self._params.setdefault('edge_color_norm_vmax', 1.0)
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute new continuous node state based on resource dynamics."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index
        current_state = neighborhood.node_state

        # Fetch parameters
        consumption_rate = self.get_param('consumption_rate', 0.1, neighborhood=neighborhood)
        initial_birth_state = self.get_param('initial_birth_state', 0.1, neighborhood=neighborhood)
        birth_min_intake = self.get_param('birth_min_potential_intake', 0.05, neighborhood=neighborhood)
        survival_min_state = self.get_param('survival_min_self_state', 0.01, neighborhood=neighborhood)

        # --- Calculate Potential Intake and Output (based on PREVIOUS step) ---
        potential_intake = 0.0
        potential_output = 0.0
        active_neighbor_indices = [] # Store indices of active neighbors

        for neighbor_idx, neighbor_prev_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx >= 0:
                edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0)
                if neighbor_prev_state > 1e-6: # If neighbor was active
                    active_neighbor_indices.append(neighbor_idx)
                    # Calculate intake FROM this neighbor
                    potential_intake += edge_state * consumption_rate * neighbor_prev_state
                    # Calculate output TO this neighbor (if self was active)
                    if current_state > 1e-6:
                        potential_output += edge_state * consumption_rate * current_state

        # Calculate intermediate state
        intermediate_state = current_state + potential_intake - potential_output
        intermediate_state_clipped = np.clip(intermediate_state, 0.0, 1.0)

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State: {current_state:.4f}") # type: ignore [attr-defined]
            logger.detail(f"    Potential Intake: {potential_intake:.4f}, Potential Output: {potential_output:.4f}") # type: ignore [attr-defined]
            logger.detail(f"    Intermediate State: {intermediate_state:.4f} (Clipped: {intermediate_state_clipped:.4f})") # type: ignore [attr-defined]

        # --- Determine Final State ---
        new_state = 0.0 # Default to death
        decision_reason = "Default (Death/Inactive)"

        if current_state <= 1e-6: # --- Birth Condition ---
            if potential_intake >= birth_min_intake:
                new_state = initial_birth_state # Born with initial state
                decision_reason = f"Birth (Intake={potential_intake:.4f} >= {birth_min_intake:.4f})"
            else:
                decision_reason = f"Remain Dead (Intake={potential_intake:.4f} < {birth_min_intake:.4f})"
        else: # --- Survival Condition ---
            if intermediate_state_clipped >= survival_min_state:
                new_state = intermediate_state_clipped # Survive with updated state
                decision_reason = f"Survival (IntermediateState={intermediate_state_clipped:.4f} >= {survival_min_state:.4f})"
            else:
                decision_reason = f"Death (Starvation: IntermediateState={intermediate_state_clipped:.4f} < {survival_min_state:.4f})"

        # --- Store eligibility proxy for edge calculation ---
        # Eligibility is simply whether the node will be alive (state > 0)
        neighborhood.rule_params['_eligibility_proxies'] = {node_idx: 1.0 if new_state > 1e-6 else 0.0}
        # ---

        if detailed_logging_enabled:
            logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    New State: {new_state:.4f}") # type: ignore [attr-defined]

        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> Dict[Tuple[int, int], float]:
        """Compute new continuous edge states based on competition, growth, decay, and randomness."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        use_competitive_formation = self.get_param('use_competitive_edge_formation', True, neighborhood=neighborhood)
        base_connect_prob = self.get_param('base_connect_prob', 0.2, neighborhood=neighborhood)
        max_load_factor = self.get_param('max_load_factor', 5.0, neighborhood=neighborhood)
        strength_diff_factor = self.get_param('strength_diff_factor', 0.5, neighborhood=neighborhood)
        new_edge_strength = self.get_param('new_edge_initial_strength', 0.1, neighborhood=neighborhood)
        growth_rate = self.get_param('growth_rate', 0.05, neighborhood=neighborhood)
        decay_rate = self.get_param('decay_rate', 0.1, neighborhood=neighborhood)
        constant_decay_rate = self.get_param('constant_decay_rate', 0.01, neighborhood=neighborhood)
        min_keep = self.get_param('min_edge_state_to_keep', 0.01, neighborhood=neighborhood)
        # Randomness params not used in this simplified version yet

        # Get eligibility proxies calculated in _compute_new_state (passed via rule_params)
        eligibility_proxies = neighborhood.rule_params.get('_eligibility_proxies', {})
        self_is_eligible = eligibility_proxies.get(node_idx, 0.0) > 0.5
        self_prev_state = neighborhood.node_state # State from previous step

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges (Competitive={use_competitive_formation}) ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Eligible Next: {self_is_eligible}, Self Prev State: {self_prev_state:.3f}") # type: ignore [attr-defined]

        for neighbor_idx, neighbor_prev_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            edge = (node_idx, neighbor_idx) # Canonical order
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0)

            # Determine neighbor's eligibility
            neighbor_is_eligible = eligibility_proxies.get(neighbor_idx, 0.0) > 0.5

            new_edge_state = current_edge_state # Start with current
            decision_info = "Maintain (default)"

            if current_edge_state < min_keep: # --- Try to Form Edge ---
                if self_is_eligible and neighbor_is_eligible:
                    should_form = False
                    if use_competitive_formation:
                        # Calculate neighbor's load (sum of its *other* incoming edge weights - requires 2nd degree info)
                        # WORKAROUND: Use neighbor's previous degree as proxy for load
                        neighbor_prev_degree = 0
                        if neighborhood.neighbor_degrees is not None:
                            neighbor_prev_degree = neighborhood.neighbor_degrees.get(neighbor_idx, 0)
                        else: logger.warning(f"Node {node_idx}: neighbor_degrees missing for competitive check.")
                        neighbor_load_proxy = float(neighbor_prev_degree)

                        strength_diff = self_prev_state - neighbor_prev_state
                        # Calculate probability (adjust formula as needed)
                        prob = base_connect_prob * max(0, 1 - neighbor_load_proxy / max(1, max_load_factor)) * max(0, 1 + strength_diff * strength_diff_factor)
                        prob = np.clip(prob, 0.0, 1.0)

                        if np.random.random() < prob:
                            should_form = True
                            decision_info = f"Form (Competitive: LoadProxy={neighbor_load_proxy:.1f}, StrDiff={strength_diff:.2f}, P={prob:.3f})"
                        else:
                            decision_info = "No Form (Competitive Failed)"
                    else: # Simple formation based on eligibility
                        if np.random.random() < base_connect_prob: # Use base probability
                             should_form = True
                             decision_info = f"Form (Simple Eligibility, P={base_connect_prob:.3f})"
                        else:
                             decision_info = "No Form (Simple Failed)"

                    if should_form:
                        new_edge_state = new_edge_strength
            else: # --- Maintain/Decay Existing Edge ---
                if self_is_eligible and neighbor_is_eligible:
                    new_edge_state += growth_rate # Strengthen
                    decision_info = f"Strengthen (Rate={growth_rate:.3f})"
                else:
                    new_edge_state -= decay_rate # Weaken
                    decision_info = f"Weaken (Rate={decay_rate:.3f})"

            # Apply constant decay regardless
            new_edge_state -= constant_decay_rate

            # Clip and Prune
            clipped_new_edge_state = float(np.clip(new_edge_state, 0.0, 1.0))

            if detailed_logging_enabled:
                logger.detail(f"    Edge {edge}: Curr={current_edge_state:.3f}, Decision='{decision_info}', ConstDecay={constant_decay_rate:.3f} -> New(clipped)={clipped_new_edge_state:.3f}") # type: ignore [attr-defined]

            if clipped_new_edge_state >= min_keep:
                 new_edges[edge] = clipped_new_edge_state
            # else: edge is pruned (not added to new_edges)

        if detailed_logging_enabled:
            logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges

class LifeWithWeightedEdges(Rule):
    """
    Game of Life variant where birth/survival depends on BOTH active neighbor counts
    AND the sum of the connecting edge weights (states). Edges have continuous states (0-1).
    (Round 39: Added edge visualization parameters)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        # 'use_state_coloring_edges', # Keep
        # 'edge_colormap', # Keep
        # 'edge_color_norm_vmin', # Keep
        # 'edge_color_norm_vmax', # Keep
        'color_edges_by_neighbor_degree', # Edge state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edge state not based on this
        'tiebreaker_type', 'node_history_depth'
    }
    # --- END MODIFIED ---

    PARAMETER_METADATA = {
        # === Node Update Logic ===
        "birth_neighbor_counts": { 'type': list, 'element_type': int, 'default': [3], 'description': "Node Logic: List of exact active neighbor counts for potential birth.", "parameter_group": "Node Logic: GoL Counts"},
        "birth_min_edge_strength_sum": { 'type': float, 'default': 1.2, 'description': "Node Logic: Min sum of edge weights to active neighbors required for birth.", "min": 0.0, "max": 26.0, "parameter_group": "Node Logic: Edge Strength"},
        "survival_neighbor_counts": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Node Logic: List of exact active neighbor counts for potential survival.", "parameter_group": "Node Logic: GoL Counts"},
        "survival_min_edge_strength_sum": { 'type': float, 'default': 0.5, 'description': "Node Logic: Min sum of edge weights to active neighbors required for survival.", "min": 0.0, "max": 26.0, "parameter_group": "Node Logic: Edge Strength"},

        # === Edge Update Logic ===
        "edge_target_factor": { "type": float, "description": "Edge Logic: Multiplier for avg predicted node state to get target edge state.", "min": 0.0, "max": 2.0, "default": 1.0, "parameter_group": "Edge Logic"},
        "edge_change_rate": { "type": float, "description": "Edge Logic: Rate (0-1) edge state moves towards target.", "min": 0.0, "max": 1.0, "default": 0.3, "parameter_group": "Edge Logic"},
        "edge_decay_rate": { "type": float, "description": "Edge Logic: Constant decay subtracted each step.", "min": 0.0, "max": 0.1, "default": 0.02, "parameter_group": "Edge Logic"},
        "random_edge_change_prob": { "type": float, "description": "Edge Logic: Probability of random change.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Logic"},
        "random_edge_change_amount": { "type": float, "description": "Edge Logic: Max amount (+/-) of random change.", "min": 0.0, "max": 0.5, "default": 0.1, "parameter_group": "Edge Logic"},
        "min_edge_state_to_keep": { "type": float, "description": "Edge Logic: Minimum edge state required to be kept.", "min": 0.0, "max": 0.5, "default": 0.05, "parameter_group": "Edge Logic"},

        # === Core Parameters ===
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood definition.", "parameter_group": "Core"},
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Grid dimension.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},

        # === Initialization Parameters ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern.", "default": "Random", "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Initial density of active nodes (state=1).", "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability for RANDOM edge init.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "min_edge_weight": { "type": float, "description": "Minimum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.2, "parameter_group": "Initialization"},
        "max_edge_weight": { "type": float, "description": "Maximum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.8, "parameter_group": "Initialization"},

        # === Visualization ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (0 or 1).", "default": False, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states.", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization: Nodes"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on their state value (0-1).",
            "default": True, "parameter_group": "Visualization: Edges" # Default True for continuous
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "prism", "parameter_group": "Visualization: Edges",
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # === Other ===
        "node_history_depth": { 'type': int, 'description': 'Number of previous states to store.', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Life with Weighted Edges"
        metadata.description = "GoL B/S counts AND min edge strength sum required for birth/survival. Edges have continuous states (0-1) and evolve based on avg node activity, inertia, and randomness." # Updated description
        metadata.category = "Life-Like"
        metadata.tags = ["Life", "Edges", "Weighted", "Connectivity", "B3/S23"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Life with Weighted Edges"
        self._params = {}
        self.requires_post_edge_state_update = False # State determined directly

        # --- ADDED: Ensure visualization params have defaults ---
        self._params['use_state_coloring'] = False
        self._params['node_colormap'] = 'prism'
        self._params['node_color_norm_vmin'] = 0.0
        self._params['node_color_norm_vmax'] = 8.0
        self._params['use_state_coloring_edges'] = True
        self._params['edge_colormap'] = 'prism'
        self._params['edge_color_norm_vmin'] = 0.0
        self._params['edge_color_norm_vmax'] = 1.0
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute new state using GoL neighbor counts first, then edge strength sum."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        birth_neighbor_counts = self.get_param('birth_neighbor_counts', [3], neighborhood=neighborhood)
        survival_neighbor_counts = self.get_param('survival_neighbor_counts', [2, 3], neighborhood=neighborhood)
        birth_min_edge_sum = self.get_param('birth_min_edge_strength_sum', 1.2, neighborhood=neighborhood)
        survival_min_edge_sum = self.get_param('survival_min_edge_strength_sum', 0.5, neighborhood=neighborhood)

        # Count active neighbors
        num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
        # Sum edge strengths ONLY to active neighbors
        sum_edge_strength = sum(
            neighborhood.neighbor_edge_states.get(idx, 0.0)
            for idx, state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states)
            if idx >= 0 and state > 0 # Only edges to ACTIVE neighbors
        )

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State: {neighborhood.node_state:.1f}") # type: ignore [attr-defined]
            logger.detail(f"    Active Neighbors: {num_active_neighbors}") # type: ignore [attr-defined]
            logger.detail(f"    Sum Edge Strength (to active): {sum_edge_strength:.4f}") # type: ignore [attr-defined]
            logger.detail(f"    Params: BirthN={birth_neighbor_counts}, SurvN={survival_neighbor_counts}, BirthMinSum={birth_min_edge_sum:.2f}, SurvMinSum={survival_min_edge_sum:.2f}") # type: ignore [attr-defined]

        new_state = 0.0 # Default to death
        decision_reason = "Default (Death/Inactive)"

        if neighborhood.node_state <= 0: # --- Birth Condition ---
            passes_neighbor_check = num_active_neighbors in birth_neighbor_counts
            passes_edge_check = sum_edge_strength >= birth_min_edge_sum
            if passes_neighbor_check and passes_edge_check:
                new_state = 1.0
                decision_reason = f"Birth (NeighborsOk={passes_neighbor_check}, EdgeSumOk={passes_edge_check})"
            else:
                decision_reason = f"Remain Dead (NeighborsOk={passes_neighbor_check}, EdgeSumOk={passes_edge_check})"

        else: # --- Survival Condition ---
            passes_neighbor_check = num_active_neighbors in survival_neighbor_counts
            passes_edge_check = sum_edge_strength >= survival_min_edge_sum
            if passes_neighbor_check and passes_edge_check:
                new_state = 1.0
                decision_reason = f"Survival (NeighborsOk={passes_neighbor_check}, EdgeSumOk={passes_edge_check})"
            else:
                decision_reason = f"Death (NeighborsOk={passes_neighbor_check}, EdgeSumOk={passes_edge_check})"

        if detailed_logging_enabled:
            logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    New State: {new_state:.1f}") # type: ignore [attr-defined]

        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Compute new continuous edge states based on average node states, decay, and randomness."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        edge_target_factor = self.get_param('edge_target_factor', 1.0, neighborhood=neighborhood)
        edge_change_rate = self.get_param('edge_change_rate', 0.3, neighborhood=neighborhood)
        edge_decay_rate = self.get_param('edge_decay_rate', 0.02, neighborhood=neighborhood)
        random_edge_prob = self.get_param('random_edge_change_prob', 0.01, neighborhood=neighborhood)
        random_edge_amount = self.get_param('random_edge_change_amount', 0.1, neighborhood=neighborhood)
        min_keep = self.get_param('min_edge_state_to_keep', 0.05, neighborhood=neighborhood)

        # Determine the *next* state of the current node (0 or 1)
        next_node_state = self._compute_new_state(neighborhood, detailed_logging_enabled)

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]
            logger.detail(f"    Predicted Next State for Edge Calc: {next_node_state:.1f}") # type: ignore [attr-defined]
            logger.detail(f"    Params: TargetFactor={edge_target_factor:.2f}, ChangeRate={edge_change_rate:.2f}, DecayRate={edge_decay_rate:.3f}, RandProb={random_edge_prob:.3f}, RandAmt={random_edge_amount:.2f}, MinKeep={min_keep:.3f}") # type: ignore [attr-defined]

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            edge = (node_idx, neighbor_idx) # Canonical order
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0)

            # Approximation: Use current neighbor state as proxy for neighbor's next state.
            neighbor_next_state_proxy = 1.0 if neighbor_state > 0 else 0.0

            # --- Determine Target Edge State ---
            target_edge_state = 0.0 # Default target is decay/removal
            if next_node_state > 0 and neighbor_next_state_proxy > 0: # If both likely alive
                 avg_node_state = (next_node_state + neighbor_next_state_proxy) / 2.0
                 target_edge_state = avg_node_state * edge_target_factor
                 if detailed_logging_enabled: logger.detail(f"    Neighbor {neighbor_idx}: Active. AvgState={avg_node_state:.4f}. TargetEdgeState={target_edge_state:.4f}") # type: ignore [attr-defined]
            # else:
                 # if detailed_logging_enabled: logger.detail(f"    Neighbor {neighbor_idx}: Inactive or Self inactive. TargetEdgeState=0.0") # type: ignore [attr-defined] # Reduce noise

            # Apply random change
            if np.random.random() < random_edge_prob:
                change = (np.random.random() * 2 - 1) * random_edge_amount
                target_edge_state += change
                # if detailed_logging_enabled: logger.detail(f"      Applied Random Change: {change:.4f} -> New Target={target_edge_state:.4f}") # type: ignore [attr-defined] # Reduce noise

            # Move edge state towards target state by the change rate
            new_edge_state = current_edge_state + (target_edge_state - current_edge_state) * edge_change_rate

            # Apply constant decay
            new_edge_state -= edge_decay_rate
            # if detailed_logging_enabled: logger.detail(f"      Applied Decay ({edge_decay_rate:.4f}) -> State Before Clip={new_edge_state:.4f}") # type: ignore [attr-defined] # Reduce noise

            clipped_new_edge_state = float(np.clip(new_edge_state, 0.0, 1.0))

            # if detailed_logging_enabled:
                # logger.detail(f"      Edge ({edge[0]}, {edge[1]}): Current={current_edge_state:.4f}, Target={target_edge_state:.4f}, New(clipped)={clipped_new_edge_state:.4f}") # type: ignore [attr-defined] # Reduce noise

            # Only propose edges with a state greater than the minimum threshold
            if clipped_new_edge_state >= min_keep:
                 new_edges[edge] = clipped_new_edge_state
                 # if detailed_logging_enabled: logger.detail(f"      Proposed edge state {clipped_new_edge_state:.4f}") # type: ignore [attr-defined] # Reduce noise

        if detailed_logging_enabled:
            logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges

class MultiStateLifeWithEdges(Rule):
    """
    A rule with continuous node and edge states (0.0-1.0), where interactions
    are based on weighted influences and thresholds.
    (Round 39: Added edge visualization parameters)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        # 'use_state_coloring_edges', # Keep
        # 'edge_colormap', # Keep
        # 'edge_color_norm_vmin', # Keep
        # 'edge_color_norm_vmax', # Keep
        'color_edges_by_neighbor_degree', # Edge state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edge state not based on this
        'tiebreaker_type', 'node_history_depth'
    }
    # --- END MODIFIED ---

    node_state_type: ClassVar[StateType] = StateType.REAL # Nodes are -1.0 to 1.0
    edge_state_type: ClassVar[StateType] = StateType.REAL # Edges are -1.0 to 1.0
    min_node_state: ClassVar[float] = -1.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = -1.0
    max_edge_state: ClassVar[float] = 1.0

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Grid dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'bounded', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern.", "default": "Random", "allowed_values": ['Random', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', 'Glider Pattern', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Initial density of nodes with state > 0.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "initial_state_value": { "type": float, "description": "Initial state value (0-1) for initially active nodes.", "min": 0.0, "max": 1.0, "default": 0.7, "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability for RANDOM edge initialization.", "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Initialization"},
        "min_edge_weight": { "type": float, "description": "Minimum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.2, "parameter_group": "Initialization"},
        "max_edge_weight": { "type": float, "description": "Maximum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.8, "parameter_group": "Initialization"},

        # === Node Logic: Influence Weights ===
        "influence_neighbor_state_weight": { "type": float, "description": "Node Logic: Weight for avg neighbor state in combined influence calculation.", "min": 0.0, "max": 1.0, "default": 0.6, "parameter_group": "Node Logic: Influence Weights"},
        "influence_edge_state_weight": { "type": float, "description": "Node Logic: Weight for avg connecting edge state in combined influence calculation.", "min": 0.0, "max": 1.0, "default": 0.3, "parameter_group": "Node Logic: Influence Weights"},
        "influence_connection_count_weight": { "type": float, "description": "Node Logic: Weight for connection count factor in combined influence calculation.", "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Node Logic: Influence Weights"},
        "connection_count_saturation": { "type": int, "description": "Node Logic: Connection count at which its positive influence saturates (used in influence calc).", "min": 1, "max": 26, "default": 5, "parameter_group": "Node Logic: Influence Weights"},
        # === Node Logic: Thresholds ===
        "activation_threshold": { "type": float, "description": "Node Logic: Combined influence needed for inactive node activation target (target becomes influence value).", "min": 0.0, "max": 1.0, "default": 0.2, "parameter_group": "Node Logic: Thresholds"},
        "deactivation_threshold": { "type": float, "description": "Node Logic: Node target state becomes 0 if combined influence falls below this.", "min": 0.0, "max": 1.0, "default": 0.08, "parameter_group": "Node Logic: Thresholds"},
        "overcrowding_death_threshold": { "type": int, "description": "Node Logic: Node target state becomes 0 if active connection count exceeds this.", "min": 0, "max": 26, "default": 7, "parameter_group": "Node Logic: Thresholds"},
        "isolation_death_threshold": { "type": int, "description": "Node Logic: Node target state becomes 0 if active connection count is less than this.", "min": 0, "max": 26, "default": 1, "parameter_group": "Node Logic: Thresholds"},
        # === Node Logic: Rate & Oscillation ===
        "state_change_rate": { "type": float, "description": "Node Logic: Rate (0-1) at which node state moves towards its calculated target state.", "min": 0.0, "max": 1.0, "default": 0.65, "parameter_group": "Node Logic: Rate & Oscillation"},
        "random_state_boost": { "type": float, "description": "Node Logic: Small random boost (+/-) added to combined influence.", "min": 0.0, "max": 0.2, "default": 0.05, "parameter_group": "Node Logic: Randomness"},
        "oscillation_center": { "type": float, "description": "Node Logic: Center point for state oscillation (around 0).", "min": -1.0, "max": 1.0, "default": 0.0, "parameter_group": "Node Logic: Rate & Oscillation"},

        # === Edge Logic ===
        "use_similarity_edge_logic": { 'type': bool, 'default': False, 'description': "Edge Logic: If True, use similarity/dissimilarity; If False, use average node state.", "parameter_group": "Edge Logic: Mode"},
        "edge_formation_threshold": { "type": float, "description": "Avg State Mode: Avg node state needed to form a new edge.", "min": 0.0, "max": 1.0, "default": 0.45, "parameter_group": "Edge Logic: Average State Mode"},
        "edge_target_factor": { "type": float, "description": "Avg State Mode: Multiplier for avg node state to get target edge state.", "min": 0.0, "max": 2.0, "default": 1.0, "parameter_group": "Edge Logic: Average State Mode"},
        "similarity_threshold": { "type": float, "description": "Similarity Mode: Max state difference to consider nodes 'similar'.", "min": 0.0, "max": 1.0, "default": 0.3, "parameter_group": "Edge Logic: Similarity Mode"},
        "similarity_strengthen_factor": { "type": float, "description": "Similarity Mode: Factor modifying change rate towards 1.0 when nodes are similar.", "min": 0.0, "max": 1.0, "default": 1.0, "parameter_group": "Edge Logic: Similarity Mode"},
        "dissimilarity_decay_factor": { "type": float, "description": "Similarity Mode: Factor modifying change rate towards 0.0 when nodes are dissimilar.", "min": 0.0, "max": 1.0, "default": 1.0, "parameter_group": "Edge Logic: Similarity Mode"},
        "edge_change_rate": { "type": float, "description": "Edge Logic: Base rate (0-1) at which edge state moves towards its target.", "min": 0.0, "max": 1.0, "default": 0.75, "parameter_group": "Edge Logic: General"},
        "edge_decay_rate": { "type": float, "description": "Edge Logic: Constant decay subtracted from edge state each step.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Logic: General"},
        "random_edge_change_prob": { "type": float, "description": "Edge Logic: Probability (0-1) of applying a small random change to edge state.", "min": 0.0, "max": 0.1, "default": 0.06, "parameter_group": "Edge Logic: General"},
        "random_edge_change_amount": { "type": float, "description": "Edge Logic: Max amount (+/-) of random change applied to edge state.", "min": 0.0, "max": 0.5, "default": 0.25, "parameter_group": "Edge Logic: General"},
        "min_edge_state_to_keep": { "type": float, "description": "Edge Logic: Minimum edge state required to be kept (non-zero).", "min": 0.0, "max": 0.5, "default": 0.02, "parameter_group": "Edge Logic: General"},

        # === Visualization ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on their state value (0-1).", "default": True, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states.", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": -1.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on their state value (0-1).",
            "default": True, "parameter_group": "Visualization: Edges" # Default True for continuous
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "prism", "parameter_group": "Visualization: Edges",
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": -1.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization"},
        # --- END ADDED ---
        # === Other ===
        "node_history_depth": { 'type': int, 'description': 'Grid Setting: History depth (not used by rule).', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Multi-State Life with Edges"
        metadata.description = "Continuous state (0-1) nodes and edges. State influenced by neighbor/edge averages and connection count. Edges based on avg node state or similarity." # Keep concise
        metadata.category = "Continuous"
        metadata.tags = ["Continuous", "Edges", "Weighted", "Dynamic", "Configurable"] # Updated tags
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Multi-State Life with Edges"
        self._params = {}
        self.requires_post_edge_state_update = False # State determined directly

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute new state based on influence, connections, overcrowding, and isolation."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        w_neighbor = self.get_param('influence_neighbor_state_weight', 0.6, neighborhood=neighborhood)
        w_edge = self.get_param('influence_edge_state_weight', 0.3, neighborhood=neighborhood)
        w_connect = self.get_param('influence_connection_count_weight', 0.1, neighborhood=neighborhood)
        connect_sat = self.get_param('connection_count_saturation', 5, neighborhood=neighborhood)
        act_thr = self.get_param('activation_threshold', 0.25, neighborhood=neighborhood)
        deact_thr = self.get_param('deactivation_threshold', 0.1, neighborhood=neighborhood)
        overcrowd_thr = self.get_param('overcrowding_death_threshold', 7, neighborhood=neighborhood)
        isolation_thr = self.get_param('isolation_death_threshold', 1, neighborhood=neighborhood)
        state_rate = self.get_param('state_change_rate', 0.5, neighborhood=neighborhood)
        random_boost = self.get_param('random_state_boost', 0.02, neighborhood=neighborhood)
        oscillation_center = self.get_param('oscillation_center', 0.0, neighborhood=neighborhood) # ADDED

        # Calculate node degree (number of active edges)
        node_degree = sum(1 for idx, state in neighborhood.neighbor_edge_states.items() if idx >= 0 and state > 1e-6)

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State: {neighborhood.node_state:.4f}") # type: ignore [attr-defined]
            logger.detail(f"    Current Degree: {node_degree}") # type: ignore [attr-defined]
            logger.detail(f"    Params: W_N={w_neighbor:.2f}, W_E={w_edge:.2f}, W_C={w_connect:.2f}, Sat={connect_sat}, ActThr={act_thr:.2f}, DeactThr={deact_thr:.2f}, OverThr={overcrowd_thr}, IsoThr={isolation_thr}, Rate={state_rate:.2f}, Boost={rand_boost:.3f}") # type: ignore [attr-defined]

        # Calculate combined influence
        valid_neighbor_indices = neighborhood.neighbor_indices[neighborhood.neighbor_indices >= 0]
        num_valid_neighbors = len(valid_neighbor_indices)
        avg_neighbor_state = 0.0
        avg_neighbor_edge_state = 0.0
        if num_valid_neighbors > 0:
            valid_neighbor_states = neighborhood.neighbor_states[neighborhood.neighbor_indices >= 0]
            avg_neighbor_state = np.mean(valid_neighbor_states) if valid_neighbor_states.size > 0 else 0.0
            avg_neighbor_edge_state = neighborhood.neighborhood_metrics.get('avg_neighbor_edge_state', 0.0)

        connection_count_factor = min(1.0, node_degree / max(1, connect_sat))
        combined_influence = (avg_neighbor_state * w_neighbor) + \
                            (avg_neighbor_edge_state * w_edge) + \
                            (connection_count_factor * w_connect) + \
                            ((np.random.random() * 2 - 1) * random_boost)

        # Determine Target State based on Influence
        target_state = neighborhood.node_state # Default: stay same
        decision_reason = "Default (No change)"

        # Check for overcrowding and isolation
        if node_degree > overcrowd_thr:
            target_state = -1.0 # Death
            decision_reason = f"Overcrowding (Degree={node_degree} > {overcrowd_thr})"
        elif node_degree < isolation_thr:
            target_state = -1.0 # Death
            decision_reason = f"Isolation (Degree={node_degree} < {isolation_thr})"
        else:
            # Apply activation/deactivation thresholds
            if combined_influence >= act_thr:
                target_state = combined_influence # Target is influence value
                decision_reason = f"Target Influence (Influence >= ActivationThr)"
            elif neighborhood.node_state > 0 and combined_influence >= deact_thr:
                target_state = combined_influence # Target is influence (ramps down/sustains)
                decision_reason = f"Target Influence (Active, Influence >= DeactivationThr)"
            else:
                target_state = oscillation_center # Target center if influence too low
                decision_reason = f"Target Center (Influence < Thresholds)"

        # Apply inertia
        new_state = neighborhood.node_state + (target_state - neighborhood.node_state) * state_rate

        # Clip and return
        return float(np.clip(new_state, -1.0, 1.0))

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Compute new continuous edge states based on selected logic mode."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch common parameters
        edge_change_rate = self.get_param('edge_change_rate', 0.6, neighborhood=neighborhood)
        edge_decay_rate = self.get_param('edge_decay_rate', 0.005, neighborhood=neighborhood)
        random_edge_prob = self.get_param('random_edge_change_prob', 0.03, neighborhood=neighborhood)
        random_edge_amount = self.get_param('random_edge_change_amount', 0.2, neighborhood=neighborhood)
        min_keep = self.get_param('min_edge_state_to_keep', 0.02, neighborhood=neighborhood)
        use_similarity_logic = self.get_param('use_similarity_edge_logic', False, neighborhood=neighborhood)

        # Fetch mode-specific parameters
        if use_similarity_logic:
            sim_threshold = self.get_param('similarity_threshold', 0.3, neighborhood=neighborhood)
            sim_strengthen = self.get_param('similarity_strengthen_factor', 1.0, neighborhood=neighborhood)
            dissim_decay = self.get_param('dissimilarity_decay_factor', 1.0, neighborhood=neighborhood)
        else:
            avg_target_factor = self.get_param('edge_target_factor', 1.0, neighborhood=neighborhood)
            formation_thr = self.get_param('edge_formation_threshold', 0.6, neighborhood=neighborhood) # Needed for avg state mode

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]
            logger.detail(f"    Using Similarity Logic: {use_similarity_logic}") # type: ignore [attr-defined]

        # Use current node state as proxy for its next state in edge calculations
        node_state_proxy = neighborhood.node_state

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            edge = (node_idx, neighbor_idx) # Canonical order
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0)

            # Use current neighbor state as proxy for its next state
            neighbor_state_proxy = neighbor_state

            # --- Determine Target Edge State ---
            target_edge_state = 0.0 # Default target is weak/removed
            mode_specific_rate_factor = 1.0 # Factor to potentially modify base change rate

            if use_similarity_logic:
                state_diff = abs(node_state_proxy - neighbor_state_proxy) # Difference between current states
                if state_diff < sim_threshold: # Similar
                    target_edge_state = 1.0
                    mode_specific_rate_factor = sim_strengthen
                    if detailed_logging_enabled: logger.detail(f"    Neighbor {neighbor_idx}: Similar (Diff={state_diff:.2f} < {sim_threshold:.2f}). Target=1.0, RateFactor={mode_specific_rate_factor:.2f}") # type: ignore [attr-defined]
                else: # Dissimilar
                    target_edge_state = 0.0
                    mode_specific_rate_factor = dissim_decay
                    if detailed_logging_enabled: logger.detail(f"    Neighbor {neighbor_idx}: Dissimilar (Diff={state_diff:.2f} >= {sim_threshold:.2f}). Target=0.0, RateFactor={mode_specific_rate_factor:.2f}") # type: ignore [attr-defined]
            else: # Average State Logic
                # Check formation threshold for new edges
                if current_edge_state <= 1e-6: # If edge doesn't exist
                    if node_state_proxy > formation_thr and neighbor_state_proxy > formation_thr:
                        target_edge_state = 1.0 # Target formation
                        if detailed_logging_enabled: logger.detail(f"    Neighbor {neighbor_idx}: Forming Edge (Nodes > {formation_thr:.4f}) -> Target=1.0") # type: ignore [attr-defined]
                    else:
                        target_edge_state = 0.0 # Not active enough to form
                        if detailed_logging_enabled: logger.detail(f"    Neighbor {neighbor_idx}: No Edge Formation (Nodes <= {formation_thr:.4f}) -> Target=0.0") # type: ignore [attr-defined]
                else: # Existing edge: target based on average state
                    avg_node_state = (node_state_proxy + neighbor_state_proxy) / 2.0
                    target_edge_state = avg_node_state * avg_target_factor
                    if detailed_logging_enabled: logger.detail(f"    Neighbor {neighbor_idx}: Existing Edge. AvgState={avg_node_state:.4f}. Initial Target={target_edge_state:.4f}") # type: ignore [attr-defined]
                # Use base change rate for this mode
                mode_specific_rate_factor = 1.0

            # Calculate effective change rate
            effective_change_rate = edge_change_rate * mode_specific_rate_factor

            # Move edge state towards target state
            new_edge_state = current_edge_state + (target_edge_state - current_edge_state) * effective_change_rate

            # Apply constant decay
            new_edge_state -= edge_decay_rate

            # Apply random change
            if np.random.random() < random_edge_prob:
                change = (np.random.random() * 2 - 1) * random_edge_amount
                new_edge_state += change

            clipped_new_edge_state = float(np.clip(new_edge_state, 0.0, 1.0))

            if detailed_logging_enabled:
                logger.detail(f"      Edge ({edge[0]},{edge[1]}): Curr={current_edge_state:.4f}, Target={target_edge_state:.4f}, EffRate={effective_change_rate:.2f}, Decay={edge_decay_rate:.3f} -> New(clipped)={clipped_new_edge_state:.4f}") # type: ignore [attr-defined]

            # Only propose edges with a state greater than the minimum threshold
            if clipped_new_edge_state >= min_keep:
                 new_edges[edge] = clipped_new_edge_state

        if detailed_logging_enabled:
            logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges

class BidirectionalFeedbackLife(Rule):
    """
    Game of Life foundation with configurable bidirectional feedback loops
    between node state (binary 0/1) and edge state (continuous 0-1).
    Various interaction logic groups can be enabled or disabled via parameters.
    (Round 36: Added edge visualization parameters)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        # 'use_state_coloring_edges', # Keep
        # 'edge_colormap', # Keep
        # 'edge_color_norm_vmin', # Keep
        # 'edge_color_norm_vmax', # Keep
        'color_edges_by_neighbor_degree', # Edge state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edge state not based on this
        'tiebreaker_type', 'node_history_depth'
    }
    # --- END MODIFIED ---

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern (Nodes initialized to 0).", "default": "Random", "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Approx initial density of nodes with connections.", "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability for RANDOM edge init.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "min_edge_weight": { "type": float, "description": "Minimum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Initialization"},
        "max_edge_weight": { "type": float, "description": "Maximum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.8, "parameter_group": "Initialization"},

        # === Node Logic ===
        "birth_neighbor_counts": { 'type': list, 'element_type': int, 'default': [3], 'description': "Node Logic: List of active neighbor counts for birth.", "parameter_group": "Node Group A: GoL Counts"},
        "survival_neighbor_counts": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Node Logic: List of active neighbor counts for survival.", "parameter_group": "Node Group A: GoL Counts"},
        "use_edge_influence_group": {'type': bool, 'default': True, 'description': "Node Logic: Enable edge strength sum condition.", "parameter_group": "Node Group B: Edge Strength"},
        "birth_min_edge_strength_sum": { 'type': float, 'default': 0.0, 'description': "Node Logic: Min edge sum to active neighbors for birth.", "min": 0.0, "max": 26.0, "parameter_group": "Node Group B: Edge Strength"},
        "survival_min_edge_strength_sum": { 'type': float, 'default': 0.2, 'description': "Node Logic: Min edge sum to active neighbors for survival.", "min": 0.0, "max": 26.0, "parameter_group": "Node Group B: Edge Strength"},
        "use_node_randomness_group": {'type': bool, 'default': True, 'description': "Node Logic: Enable random node birth boost.", "parameter_group": "Node Group C: Randomness"},
        "random_birth_boost": { "type": float, "description": "Node Logic: Small random chance boost for inactive node activation.", "min": 0.0, "max": 0.1, "default": 0.005, "parameter_group": "Node Group C: Randomness"},

        # === Edge Logic ===
        "edge_target_factor": { "type": float, "description": "Edge Logic: Multiplier for avg node state to get target edge state.", "min": 0.0, "max": 2.0, "default": 1.0, "parameter_group": "Edge Group A: Node Activity"},
        "edge_change_rate": { "type": float, "description": "Edge Logic: Rate (0-1) edge state moves towards target.", "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Edge Group A: Node Activity"},
        "edge_decay_rate": { "type": float, "description": "Edge Logic: Constant decay subtracted each step.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Group B: Decay"},
        "random_edge_change_prob": { "type": float, "description": "Edge Logic: Probability of random change.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Group C: Randomness"},
        "random_edge_change_amount": { "type": float, "description": "Edge Logic: Max amount (+/-) of random change.", "min": 0.0, "max": 0.5, "default": 0.1, "parameter_group": "Edge Group C: Randomness"},
        "min_edge_state_to_keep": { "type": float, "description": "Edge Logic: Minimum edge state required to be kept.", "min": 0.0, "max": 0.5, "default": 0.02, "parameter_group": "Edge Pruning"},
        "use_probabilistic_degree_edges_group": { 'type': bool, 'default': False, 'description': "Edge Logic: Enable probabilistic edge creation/removal based on neighbor degree (prev step).", "parameter_group": "Edge Group D: Probabilistic Degree"},
        "prob_connect_degree_threshold": { 'type': int, 'default': 3, 'description': "Edge Logic: Connect probabilistically if neighbor degree (prev step) > this.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Probabilistic Degree"},
        "prob_connect_if_above_threshold": { 'type': float, 'default': 0.1, 'description': "Edge Logic: Probability (0-1) to form edge if neighbor degree above threshold.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Group D: Probabilistic Degree"},
        "prob_disconnect_degree_threshold": { 'type': int, 'default': 2, 'description': "Edge Logic: Disconnect probabilistically if neighbor degree (prev step) < this.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Probabilistic Degree"},
        "prob_disconnect_if_below_threshold": { 'type': float, 'default': 0.2, 'description': "Edge Logic: Probability (0-1) to remove edge if neighbor degree below threshold.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Group D: Probabilistic Degree"},

        # === Visualization ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (0 or 1).", "default": False, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states.", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": -0.1, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization"},# --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on their state value (0-1).",
            "default": True, "parameter_group": "Visualization: Edges" # Default True for continuous
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "prism", "parameter_group": "Visualization: Edges",
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": -1.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization"},# --- END ADDED ---
        # === Other ===
        "node_history_depth": { 'type': int, 'description': 'Number of previous states to store.', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Bidirectional Feedback Life"
        metadata.description = "GoL foundation with toggleable logic groups for node/edge interactions based on neighbor counts, edge strength, degree, avg states, decay, randomness. Node state 0/1, Edge state 0-1." # Updated description
        metadata.category = "Experimental"
        metadata.tags = ["Life", "Edges", "Feedback", "Modular", "Configurable", "Experimental"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Bidirectional Feedback Life"
        self._params = {}
        self.requires_post_edge_state_update = False # State determined directly
        # --- MODIFIED: Set needs_neighbor_degrees to True ---
        self.needs_neighbor_degrees = True # Now needed for probabilistic edge logic
        self.needs_neighbor_active_counts = False # Not needed
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute node state based on enabled logic groups."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Get toggle parameters
        use_gol = self.get_param('use_gol_logic_group', True, neighborhood=neighborhood)
        use_edge_infl = self.get_param('use_edge_influence_group', True, neighborhood=neighborhood)
        use_random = self.get_param('use_node_randomness_group', True, neighborhood=neighborhood)

        # Get specific thresholds/values needed based on enabled groups
        birth_counts = self.get_param('birth_neighbor_counts', [3], neighborhood=neighborhood)
        survival_counts = self.get_param('survival_neighbor_counts', [2, 3], neighborhood=neighborhood)
        birth_min_sum = self.get_param('birth_min_edge_strength_sum', 0.0, neighborhood=neighborhood)
        survival_min_sum = self.get_param('survival_min_edge_strength_sum', 0.2, neighborhood=neighborhood)
        rand_boost = self.get_param('random_birth_boost', 0.005, neighborhood=neighborhood)

        # Calculate base metrics needed by potentially active groups
        num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
        sum_edge_strength = 0.0
        if use_edge_infl:
            sum_edge_strength = sum(
                neighborhood.neighbor_edge_states.get(idx, 0.0)
                for idx, state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states)
                if idx >= 0 and state > 0 # Only edges to ACTIVE neighbors
            )

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State: {neighborhood.node_state:.1f}") # type: ignore [attr-defined]
            logger.detail(f"    Active Neighbors: {num_active_neighbors}, Edge Strength Sum: {sum_edge_strength:.4f}") # type: ignore [attr-defined]
            logger.detail(f"    Enabled Groups: GoL={use_gol}, EdgeInfl={use_edge_infl}, Random={use_random}") # type: ignore [attr-defined]

        # --- Determine Eligibility based on Enabled Logic ---
        final_eligible = True # Start assuming eligible, checks will invalidate

        if neighborhood.node_state <= 0: # --- Birth Logic ---
            eligible = False # Default to not born
            passes_gol = (not use_gol) or (num_active_neighbors in birth_counts)
            passes_edge_sum = (not use_edge_infl) or (sum_edge_strength >= birth_min_sum)

            if passes_gol and passes_edge_sum:
                eligible = True
                if detailed_logging_enabled: logger.detail(f"    Decision: Eligible for Birth (GoL:{passes_gol}, EdgeSum:{passes_edge_sum})") # type: ignore [attr-defined]
            else:
                 if detailed_logging_enabled: logger.detail(f"    Decision: Ineligible for Birth (GoL:{passes_gol}, EdgeSum:{passes_edge_sum})") # type: ignore [attr-defined]

            # Random birth boost
            if not eligible and use_random and np.random.random() < rand_boost:
                eligible = True
                if detailed_logging_enabled: logger.detail(f"    Decision Override: Random Birth!") # type: ignore [attr-defined]

            final_eligible = eligible

        else: # --- Survival Logic ---
            passes_gol = (not use_gol) or (num_active_neighbors in survival_counts)
            passes_edge_sum = (not use_edge_infl) or (sum_edge_strength >= survival_min_sum)

            if not (passes_gol and passes_edge_sum):
                final_eligible = False # Fails if any enabled check fails
                if detailed_logging_enabled: logger.detail(f"    Decision: Death (GoL:{passes_gol}, EdgeSum:{passes_edge_sum})") # type: ignore [attr-defined]
            else:
                if detailed_logging_enabled: logger.detail(f"    Decision: Survival (All enabled checks passed)") # type: ignore [attr-defined]

        new_state = 1.0 if final_eligible else 0.0

        if detailed_logging_enabled:
            logger.detail(f"    New State: {new_state:.1f}") # type: ignore [attr-defined]

        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Compute new continuous edge states based on node activity, decay, and randomness, with optional probabilistic degree effects."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch base parameters
        edge_target_factor = self.get_param('edge_target_factor', 1.0, neighborhood=neighborhood)
        edge_change_rate = self.get_param('edge_change_rate', 0.4, neighborhood=neighborhood)
        edge_decay_rate = self.get_param('edge_decay_rate', 0.01, neighborhood=neighborhood)
        random_edge_prob = self.get_param('random_edge_change_prob', 0.01, neighborhood=neighborhood)
        random_edge_amount = self.get_param('random_edge_change_amount', 0.1, neighborhood=neighborhood)
        min_keep = self.get_param('min_edge_state_to_keep', 0.02, neighborhood=neighborhood)

        # Fetch probabilistic parameters
        use_prob_logic = self.get_param('use_probabilistic_degree_edges_group', False, neighborhood=neighborhood)
        prob_conn_thr = self.get_param('prob_connect_degree_threshold', 3, neighborhood=neighborhood)
        prob_conn_p = self.get_param('prob_connect_if_above_threshold', 0.1, neighborhood=neighborhood)
        prob_disc_thr = self.get_param('prob_disconnect_degree_threshold', 2, neighborhood=neighborhood)
        prob_disc_p = self.get_param('prob_disconnect_if_below_threshold', 0.2, neighborhood=neighborhood)

        # Determine the *next* state of the current node
        next_node_state = self._compute_new_state(neighborhood, detailed_logging_enabled)
        self_is_active_next = next_node_state > 0.5

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges (Probabilistic={use_prob_logic}) ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Active Next: {self_is_active_next}") # type: ignore [attr-defined]

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            edge = (node_idx, neighbor_idx) # Canonical order
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0)

            # Approximation: Use current neighbor state as proxy for neighbor's next state.
            neighbor_is_active_next = neighbor_state > 0.5

            # --- Determine Target Edge State ---
            target_edge_state = 0.0 # Default target is decay/removal
            use_node_infl = self.get_param('use_node_influence_group', False, neighborhood=neighborhood)
            if use_node_infl:
                target_edge_state = (next_node_state + neighbor_state) / 2.0 * edge_target_factor # Use proxy state
            # ---

            # --- Probabilistic Override Logic ---
            if use_prob_logic:
                # Get neighbor's degree from previous step
                neighbor_prev_degree = 0
                if neighborhood.neighbor_degrees is not None:
                    neighbor_prev_degree = neighborhood.neighbor_degrees.get(neighbor_idx, 0)
                else:
                    logger.warning(f"Node {node_idx}: neighbor_degrees missing for neighbor {neighbor_idx}. Using 0.")

                has_current_edge = neighbor_idx in neighborhood.neighbor_edge_states
                if not has_current_edge: # Edge Birth Check
                    if neighbor_prev_degree > prob_conn_thr:
                        if np.random.random() < prob_conn_p:
                            target_edge_state = 1.0 # Force connect
                            if detailed_logging_enabled: logger.detail(f"    Neighbor {neighbor_idx}: Prob Connect (NeighDeg={neighbor_prev_degree}>{prob_conn_thr}, P={prob_conn_p:.2f})") # type: ignore [attr-defined]
                else: # Edge Survival Check
                    if neighbor_prev_degree < prob_disc_thr:
                        if np.random.random() < prob_disc_p:
                            target_edge_state = 0.0 # Force disconnect
                            if detailed_logging_enabled: logger.detail(f"    Neighbor {neighbor_idx}: Prob Disconnect (NeighDeg={neighbor_prev_degree}<{prob_disc_thr}, P={prob_disc_p:.2f})") # type: ignore [attr-defined]

            # Apply Inertia
            new_edge_state = current_edge_state + (target_edge_state - current_edge_state) * edge_change_rate

            # Apply Decay
            new_edge_state -= edge_decay_rate

            # Apply Randomness
            use_random = self.get_param('use_edge_randomness_group', False, neighborhood=neighborhood)
            if use_random and np.random.random() < random_edge_prob:
                change = (np.random.random() * 2 - 1) * random_edge_amount
                new_edge_state += change

            clipped_new_edge_state = float(np.clip(new_edge_state, 0.0, 1.0))

            if detailed_logging_enabled:
                logger.detail(f"    Edge {edge}: Curr={current_edge_state:.3f}, Target={target_edge_state:.3f}, New(clipped)={clipped_new_edge_state:.3f}") # type: ignore [attr-defined]

            # Only propose edges with a state greater than the minimum threshold
            if clipped_new_edge_state >= min_keep:
                 new_edges[edge] = clipped_new_edge_state

        if detailed_logging_enabled:
            logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges

class ConfigurableLifeWithEdges(Rule):
    """
    Highly configurable GoL variant with continuous edges (0-1) and binary nodes (0/1).
    Node and Edge behaviors are modular, controlled by boolean toggles for different logic groups.
    (Round 37: Added edge visualization parameters)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        # 'use_state_coloring_edges', # Keep
        # 'edge_colormap', # Keep
        # 'edge_color_norm_vmin', # Keep
        # 'edge_color_norm_vmax', # Keep
        'color_edges_by_neighbor_degree', # Edge state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edge state not based on this
        'tiebreaker_type', 'node_history_depth'
    }
    # --- END MODIFIED ---

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern (Nodes initialized to 0).", "default": "Random", "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Approx initial density of nodes with connections.", "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability for RANDOM edge init.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "min_edge_weight": { "type": float, "description": "Minimum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Initialization"},
        "max_edge_weight": { "type": float, "description": "Maximum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.8, "parameter_group": "Initialization"},

        # === Node Logic Group 1: GoL Counts ===
        "use_gol_counts_group": {'type': bool, 'default': True, 'description': "Node Logic: Enable GoL B/S neighbor counts.", "parameter_group": "Node Group 1: GoL Counts"},
        "birth_neighbor_counts": { 'type': list, 'element_type': int, 'default': [3], 'description': "Node Logic: Active neighbor counts for birth.", "parameter_group": "Node Group 1: GoL Counts"},
        "survival_neighbor_counts": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Node Logic: Active neighbor counts for survival.", "parameter_group": "Node Group 1: GoL Counts"},

        # === Node Logic Group 2: Edge Strength Sum ===
        "use_edge_strength_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable edge strength sum condition (uses prev step edges to active neighbors).", "parameter_group": "Node Group 2: Edge Strength Sum"},
        "birth_min_edge_strength_sum": { 'type': float, 'default': 1.0, 'description': "Node Logic: Min edge sum to active neighbors for birth.", "min": 0.0, "max": 26.0, "parameter_group": "Node Group 2: Edge Strength Sum"},
        "survival_min_edge_strength_sum": { 'type': float, 'default': 0.5, 'description': "Node Logic: Min edge sum to active neighbors for survival.", "min": 0.0, "max": 26.0, "parameter_group": "Node Group 2: Edge Strength Sum"},

        # === Node Logic Group 3: Degree Survival (Prev Step) ===
        "use_degree_survival_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable survival based on degree range (uses PREVIOUS step degree). Requires neighbor_degrees data.", "parameter_group": "Node Group 3: Degree Survival (Prev)"},
        "survival_degree_min": { 'type': int, 'default': 1, 'description': "Node Logic: Min connections (prev step) for survival eligibility.", "min": 0, "max": 26, "parameter_group": "Node Group 3: Degree Survival (Prev)"},
        "survival_degree_max": { 'type': int, 'default': 5, 'description': "Node Logic: Max connections (prev step) for survival eligibility.", "min": 0, "max": 26, "parameter_group": "Node Group 3: Degree Survival (Prev)"},

        # === Node Logic Group 4: Avg Edge State ===
        "use_avg_edge_influence_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable influence from avg neighbor edge state (uses PREVIOUS step edges).", "parameter_group": "Node Group 4: Avg Edge State"},
        "birth_min_avg_edge_state": { 'type': float, 'default': 0.4, 'description': "Node Logic: Min avg edge state (prev step) required for birth.", "min": 0.0, "max": 1.0, "parameter_group": "Node Group 4: Avg Edge State"},
        "survival_min_avg_edge_state": { 'type': float, 'default': 0.1, 'description': "Node Logic: Min avg edge state (prev step) required for survival.", "min": 0.0, "max": 1.0, "parameter_group": "Node Group 4: Avg Edge State"},

        # === Node Logic Group 5: Isolation/Overcrowding (Current Step) ===
        "use_isolation_overcrowding_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable death based on CURRENT degree (calculated from active edges). Applied after other checks.", "parameter_group": "Node Group 5: Isolation/Overcrowding (Current)"},
        "isolation_death_threshold": { 'type': int, 'default': 1, 'description': "Node Logic: Force death if CURRENT degree < this.", "min": 0, "max": 26, "parameter_group": "Node Group 5: Isolation/Overcrowding (Current)"},
        "overcrowding_death_threshold": { 'type': int, 'default': 7, 'description': "Node Logic: Force death if CURRENT degree > this.", "min": 0, "max": 26, "parameter_group": "Node Group 5: Isolation/Overcrowding (Current)"},

        # === Node Logic Group 6: Random Birth Boost ===
        "use_node_randomness_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable random node birth boost.", "parameter_group": "Node Group 6: Random Birth Boost"},
        "random_birth_boost": { "type": float, "description": "Node Logic: Small random chance boost for inactive node activation.", "min": 0.0, "max": 0.1, "default": 0.005, "parameter_group": "Node Group 6: Random Birth Boost"},

        # === Edge Logic Group A: Node Activity ===
        "use_node_activity_edge_group": {'type': bool, 'default': True, 'description': "Edge Logic (Priority 4): Enable edge target based on connected nodes' predicted activity.", "parameter_group": "Edge Group A: Node Activity"},
        "edge_target_factor": { "type": float, "description": "Edge Logic: Multiplier for avg predicted node state to get target edge state.", "min": 0.0, "max": 2.0, "default": 1.0, "parameter_group": "Edge Group A: Node Activity"},
        "edge_activity_change_rate": { "type": float, "description": "Edge Logic: Rate edge moves towards activity target.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Edge Group A: Node Activity"},

        # === Edge Logic Group B: Similarity ===
        "use_similarity_edge_group": {'type': bool, 'default': False, 'description': "Edge Logic (Priority 3): Enable edge target/rate based on similarity of predicted node states. Overrides Group A.", "parameter_group": "Edge Group B: Similarity"},
        "similarity_threshold": { "type": float, "description": "Edge Logic: Max state difference to consider nodes 'similar'.", "min": 0.0, "max": 1.0, "default": 0.3, "parameter_group": "Edge Group B: Similarity"},
        "similarity_strengthen_factor": { "type": float, "description": "Edge Logic: Factor modifying change rate towards 1.0 when nodes are similar.", "min": 0.0, "max": 1.0, "default": 1.0, "parameter_group": "Edge Group B: Similarity"},
        "dissimilarity_decay_factor": { "type": float, "description": "Edge Logic: Factor modifying change rate towards 0.0 when nodes are dissimilar.", "min": 0.0, "max": 1.0, "default": 1.0, "parameter_group": "Edge Group B: Similarity"},

        # === Edge Logic Group C: Neighbor Degree (Prev Step) ===
        "use_neighbor_degree_edge_group": {'type': bool, 'default': False, 'description': "Edge Logic (Priority 2): Enable edge target/rate based on neighbor degree thresholds (uses PREVIOUS step degree). Requires neighbor_degrees data. Overrides Groups A, B.", "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "connect_neighbor_degree_min": { 'type': int, 'default': 1, 'description': "Edge Logic: Connect if neighbor degree (prev step) >= this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "connect_neighbor_degree_max": { 'type': int, 'default': 4, 'description': "Edge Logic: Connect if neighbor degree (prev step) <= this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "disconnect_neighbor_degree_min": { 'type': int, 'default': 1, 'description': "Edge Logic: Disconnect if neighbor degree (prev step) < this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "disconnect_neighbor_degree_max": { 'type': int, 'default': 6, 'description': "Edge Logic: Disconnect if neighbor degree (prev step) > this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "neighbor_degree_change_rate": { "type": float, "description": "Edge Logic: Rate edge moves towards 0/1 based on neighbor degree.", "min": 0.0, "max": 1.0, "default": 0.6, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},

        # === Edge Logic Group D: Node Effects (Prev Step) ===
        "use_node_effects_on_edges_group": {'type': bool, 'default': False, 'description': "Edge Logic (Priority 1): Enable edge target/rate based on neighbor count (uses PREVIOUS step counts). Requires neighbor_active_counts data. Overrides Groups A, B, C.", "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_survival_neighbor_min": { 'type': int, 'default': 2, 'description': "Edge Logic: Min active neighbors (prev step) for nodes to boost edge survival.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_survival_neighbor_max": { 'type': int, 'default': 4, 'description': "Edge Logic: Max active neighbors (prev step) for nodes to boost edge survival.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_survival_boost_rate": { 'type': float, 'default': 0.1, 'description': "Edge Logic: Rate edge moves towards 1 if node neighbor counts optimal.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_death_neighbor_min": { 'type': int, 'default': 1, 'description': "Edge Logic: Edge death boost if node neighbor count (prev step) < this.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_death_neighbor_max": { 'type': int, 'default': 5, 'description': "Edge Logic: Edge death boost if node neighbor count (prev step) > this.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_death_boost_rate": { 'type': float, 'default': 0.2, 'description': "Edge Logic: Rate edge moves towards 0 if node neighbor counts non-optimal.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Group D: Node Effects (Prev)"},

        # === Edge Logic Group E: Decay ===
        "use_edge_decay_group": {'type': bool, 'default': True, 'description': "Edge Logic: Enable Constant Edge Decay.", "parameter_group": "Edge Group E: Decay"},
        "edge_decay_rate": { "type": float, "description": "Edge Logic: Constant decay applied each step.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Group E: Decay"},

        # === Edge Logic Group F: Randomness ===
        "use_edge_randomness_group": {'type': bool, 'default': False, 'description': "Edge Logic: Enable Random Edge Fluctuations.", "parameter_group": "Edge Group F: Randomness"},
        "random_edge_change_prob": { "type": float, "description": "Edge Logic: Probability of random change.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Group F: Randomness"},
        "random_edge_change_amount": { "type": float, "description": "Edge Logic: Max amount (+/-) of random change.", "min": 0.0, "max": 0.5, "default": 0.15, "parameter_group": "Edge Group F: Randomness"},

        # --- Edge Pruning ---
        "min_edge_state_to_keep": { "type": float, "description": "Edge Logic: Minimum edge state required to be kept.", "min": 0.0, "max": 0.5, "default": 0.02, "parameter_group": "Edge Pruning"},

        # --- Visualization ---
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (0 or 1).", "default": False, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states.", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization: Nodes"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on their state value (0-1).",
            "default": True, "parameter_group": "Visualization: Edges" # Default True for continuous
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "prism", "parameter_group": "Visualization: Edges",
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # --- Other ---
        "node_history_depth": {
            'type': int, 'default': 10, 'min': 0, 'max': 100,
            'description': "Grid Setting: Number of previous node states stored internally (not used by this rule's logic).",
            'parameter_group': 'History'
        }
    }
    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Configurable Life with Edges"
        metadata.description = "GoL foundation with toggleable logic groups for node/edge interactions. Node state 0/1, Edge state 0-1. Uses 2nd-order neighbor active counts if Group 5 enabled." # Updated description
        metadata.category = "Experimental"
        metadata.tags = ["Life", "Edges", "Feedback", "Modular", "Configurable", "Experimental"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Configurable Life with Edges"
        self._params = {}
        self.requires_post_edge_state_update = True # State determined directly
        # --- Set flags to request data ---
        self.needs_neighbor_degrees = False # No longer need neighbor degrees
        self.needs_neighbor_active_counts = False # No longer need neighbor active counts
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _count_active_neighbors(self, neighbor_states: np.ndarray) -> int:
        """Counts neighbors whose state > activity threshold (binary check)."""
        activity_threshold = self.get_param('node_activity_threshold', 0.1)
        return int(np.sum(neighbor_states > activity_threshold))

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute node state based on enabled logic groups, prioritizing GoL."""
        # logger.debug(f"--- Node {neighborhood.node_index}: {self.name} _compute_new_state ---") # Reduce noise

        # Get toggle parameters
        use_gol, use_edge_infl, use_degree, use_avg_edge, use_random = self._get_parameters(
            'use_gol_counts_group', 'use_edge_strength_group', 'use_degree_survival_group',
            'use_avg_edge_influence_group', 'use_node_randomness_group'
        )

        # Get specific thresholds/values needed based on enabled groups
        birth_counts = self.get_param('birth_neighbor_counts') if use_gol else []
        survival_counts = self.get_param('survival_neighbor_counts') if use_gol else []
        birth_min_sum = self.get_param('birth_min_edge_strength_sum') if use_edge_infl else 0.0
        survival_min_sum = self.get_param('survival_min_edge_strength_sum') if use_edge_infl else 0.0
        survival_min_deg = self.get_param('survival_degree_min') if use_degree else 0
        survival_max_deg = self.get_param('survival_degree_max') if use_degree else 100 # High default if unused
        birth_avg_edge_min = self.get_param('avg_edge_activation_threshold') if use_avg_edge else 0.0 # Corrected param name
        death_avg_edge_max = self.get_param('avg_edge_deactivation_threshold') if use_avg_edge else 1.0 # Corrected param name
        rand_boost = self.get_param('random_birth_boost') if use_random else 0.0
        activity_threshold = self.get_param('node_activity_threshold', 0.1)

        # Calculate base metrics needed by potentially active groups
        num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
        sum_edge_strength = 0.0
        if use_edge_infl or use_avg_edge:
            sum_edge_strength = sum(
                neighborhood.neighbor_edge_states.get(idx, 0.0)
                for idx, state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states)
                if idx >= 0 and state > activity_threshold # Use threshold
            )
        prev_degree = 0
        if use_degree and neighborhood.neighbor_degrees is not None:
             prev_degree = neighborhood.neighbor_degrees.get(neighborhood.node_index, 0)
        elif use_degree: logger.warning(f"Node {neighborhood.node_index}: neighbor_degrees missing for self!")
        avg_neighbor_edge_state = neighborhood.neighborhood_metrics.get('avg_neighbor_edge_state', 0.0)

        # --- Determine Eligibility based on Enabled Logic (AND logic) ---
        eligible = True # Start assuming eligible

        if neighborhood.node_state <= activity_threshold: # --- Birth Logic ---
            eligible = False # Default to not born
            passes_gol = (not use_gol) or (num_active_neighbors in birth_counts)
            passes_edge_sum = (not use_edge_infl) or (sum_edge_strength >= birth_min_sum)
            passes_avg_edge = (not use_avg_edge) or (avg_neighbor_edge_state >= birth_avg_edge_min)
            # Degree check doesn't apply to birth

            if passes_gol and passes_edge_sum and passes_avg_edge:
                eligible = True
                # logger.debug(f"    Decision: Eligible for Birth (GoL:{passes_gol}, EdgeSum:{passes_edge_sum}, AvgEdge:{passes_avg_edge})") # Reduce noise
            # else: logger.debug(f"    Decision: Ineligible for Birth (GoL:{passes_gol}, EdgeSum:{passes_edge_sum}, AvgEdge:{passes_avg_edge})") # Reduce noise

            # Random birth boost
            if not eligible and use_random and np.random.random() < rand_boost:
                eligible = True
                # logger.debug("    Decision Override: Random Birth!") # Reduce noise

        else: # --- Survival Logic ---
            passes_gol = (not use_gol) or (num_active_neighbors in survival_counts)
            passes_edge_sum = (not use_edge_infl) or (sum_edge_strength >= survival_min_sum)
            passes_degree = (not use_degree) or (survival_min_deg <= prev_degree <= survival_max_deg)
            passes_avg_edge = (not use_avg_edge) or (avg_neighbor_edge_state > death_avg_edge_max) # Dies if BELOW max threshold

            if not (passes_gol and passes_edge_sum and passes_degree and passes_avg_edge):
                eligible = False # Fails if any enabled check fails
                # logger.debug(f"    Decision: Death (GoL:{passes_gol}, EdgeSum:{passes_edge_sum}, Degree:{passes_degree}, AvgEdge:{passes_avg_edge})") # Reduce noise
            # else: logger.debug(f"    Decision: Survival (All enabled checks passed)") # Reduce noise

        new_state = 1.0 if eligible else 0.0
        # logger.debug(f"    New State: {new_state:.1f}") # Reduce noise
        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Compute new continuous edge states based on enabled logic groups with priority."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        edge_target_factor = self.get_param('edge_target_factor', 1.0, neighborhood=neighborhood)
        edge_change_rate = self.get_param('edge_change_rate', 0.4, neighborhood=neighborhood)
        edge_decay_rate = self.get_param('edge_decay_rate', 0.01, neighborhood=neighborhood)
        random_edge_prob = self.get_param('random_edge_change_prob', 0.01, neighborhood=neighborhood)
        random_edge_amount = self.get_param('random_edge_change_amount', 0.1, neighborhood=neighborhood)
        min_keep = self.get_param('min_edge_state_to_keep', 0.02, neighborhood=neighborhood)

        # --- Get Eligibility Proxies from rule_params ---
        eligibility_proxies_flat = neighborhood.rule_params.get('_eligibility_proxies')
        if eligibility_proxies_flat is None:
            logger.error(f"Node {node_idx}: Eligibility proxies missing in rule_params for edge calculation!")
            return new_edges # Cannot determine mutual eligibility
        # ---

        # Determine the *next* state of the current node
        self_is_eligible = False
        if 0 <= node_idx < eligibility_proxies_flat.size:
            self_is_eligible = eligibility_proxies_flat[node_idx] > 0.5
        else:
            logger.warning(f"Node {node_idx}: Index out of bounds for eligibility proxies.")
            return new_edges

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Eligible Next: {self_is_eligible}") # type: ignore [attr-defined]

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            edge = (node_idx, neighbor_idx) # Canonical order
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0)

            # Determine neighbor's eligibility from proxy array
            neighbor_is_eligible = False
            if 0 <= neighbor_idx < eligibility_proxies_flat.size:
                neighbor_is_eligible = eligibility_proxies_flat[neighbor_idx] > 0.5
            else:
                logger.warning(f"Node {node_idx}: Neighbor index {neighbor_idx} out of bounds for eligibility proxies.")
                continue

            # Only apply edge logic if BOTH nodes are eligible
            if self_is_eligible and neighbor_is_eligible:
                # Apply base logic (move towards target based on activity)
                target_edge_state = 1.0 if (self_is_eligible and neighbor_is_eligible) else 0.0
                new_edge_state = current_edge_state + (target_edge_state - current_edge_state) * edge_change_rate

                # Apply constant decay
                new_edge_state -= edge_decay_rate

                # Apply random change
                use_random = self.get_param('use_random', False)  # Define use_random
                if use_random and np.random.random() < random_edge_prob:
                    change = (np.random.random() * 2 - 1) * random_edge_amount
                    new_edge_state += change

                clipped_new_edge_state = float(np.clip(new_edge_state, 0.0, 1.0))

                if detailed_logging_enabled:
                    logger.detail(f"    Edge {edge}: Curr={current_edge_state:.3f}, Target={target_edge_state:.3f}, New(clipped)={clipped_new_edge_state:.3f}") # type: ignore [attr-defined]

                # Only propose edges with a state greater than the minimum threshold
                if clipped_new_edge_state >= min_keep:
                     new_edges[edge] = clipped_new_edge_state

        if detailed_logging_enabled:
            logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges

    def _compute_final_state(
            self, node_idx: int, current_proxy_state: float,
            final_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
            dimensions: Tuple[int,...],
            previous_node_states: npt.NDArray[np.float64],
            previous_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
            previous_edge_states: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float],
            previous_node_degrees: Optional[npt.NDArray[np.int32]],
            previous_active_neighbors: Optional[npt.NDArray[np.int32]],
            eligibility_proxies: Optional[np.ndarray] = None,
            detailed_logging_enabled: bool = False
            ) -> float:
        """
        Calculates the final node state (for visualization) based on eligibility and coloring parameters.
        (Round 20: Added eligibility_proxies parameter to signature)
        """
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_final_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Proxy State (Eligibility): {current_proxy_state:.1f}") # type: ignore [attr-defined]

        # If proxy state indicates ineligible (0.0), final state is 0
        if current_proxy_state < 0.5:
             if detailed_logging_enabled: logger.detail(f"    Node was ineligible based on prev step conditions, final state is 0.") # type: ignore [attr-defined]
             return 0.0

        # --- ADDED: Group 5 Isolation/Overcrowding Check (using FINAL degree) ---
        use_iso_over = self.get_param('use_isolation_overcrowding_group', False)
        if use_iso_over:
            isolation_thr = self.get_param('isolation_death_threshold', 1)
            overcrowding_thr = self.get_param('overcrowding_death_threshold', 7)
            # Calculate CURRENT degree based on final_edges
            node_coords = tuple(_unravel_index(node_idx, dimensions))
            current_degree = sum(1 for edge in final_edges if node_coords in edge)

            if current_degree < isolation_thr:
                if detailed_logging_enabled: logger.detail(f"    Death Override: Isolation (Current Degree={current_degree} < {isolation_thr})") # type: ignore [attr-defined]
                return 0.0 # Force death
            if current_degree > overcrowding_thr:
                if detailed_logging_enabled: logger.detail(f"    Death Override: Overcrowding (Current Degree={current_degree} > {overcrowding_thr})") # type: ignore [attr-defined]
                return 0.0 # Force death
        # --- END ADDED ---

        # If eligible and not overridden by isolation/overcrowding, determine final state for visualization
        color_by_degree = self.get_param('color_nodes_by_degree', False)
        color_by_neighbors = self.get_param('color_nodes_by_active_neighbors', False)

        final_state = 1.0 # Default to binary 1.0 if eligible and no other coloring
        decision_reason = "Default (Eligible, No Specific Coloring)"

        if color_by_degree:
            # Calculate final degree based on the FINAL edges for this step (already calculated if iso/over used)
            if not use_iso_over: # Recalculate if not done above
                 node_coords = tuple(_unravel_index(node_idx, dimensions))
                 final_degree = sum(1 for edge in final_edges if node_coords in edge)
            else: final_degree = current_degree # Reuse calculated degree
            final_state = float(final_degree)
            decision_reason = "Color by Degree"
            if detailed_logging_enabled: logger.detail(f"    Coloring by degree. Final Degree={final_degree}. Final State={final_state:.1f}") # type: ignore [attr-defined]
        elif color_by_neighbors:
            # Calculate active neighbor count from PREVIOUS step
            active_neighbor_count = 0
            if previous_active_neighbors is not None:
                if 0 <= node_idx < previous_active_neighbors.size:
                    active_neighbor_count = float(previous_active_neighbors[node_idx])
                else: logger.warning(f"Node {node_idx}: Index out of bounds for previous_active_neighbors array.")
            else: logger.warning("previous_active_neighbors array not provided. Cannot color by active neighbors.")
            final_state = float(active_neighbor_count)
            decision_reason = "Color by Active Neighbors"
            if detailed_logging_enabled: logger.detail(f"    Coloring by active neighbors. Count={active_neighbor_count}. Final State={final_state:.1f}") # type: ignore [attr-defined]
        else:
            # No specific coloring, use binary eligibility state
            decision_reason = "Binary Eligibility State"
            if detailed_logging_enabled: logger.detail(f"    No specific coloring, using binary eligibility state.") # type: ignore [attr-defined]

        if detailed_logging_enabled:
            logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    Final State: {final_state:.1f}") # type: ignore [attr-defined]

        return final_state

class EdgeDegreeLife(Rule):
    """
    Simulates a "Game of Life" for edges based on endpoint connectivity.
    Node state represents its degree (number of active edges connected).
    Edges are binary (0/1) and evolve based on GoL-like rules applied to their local edge density.
    Requires post-update step.
    (Round 34: Added edge visualization parameters)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'min_edge_weight', 'max_edge_weight', # Edges are binary
        'connect_probability', # Not used by core logic, only init
        # 'use_state_coloring_edges', # Keep
        # 'edge_colormap', # Keep
        # 'edge_color_norm_vmin', # Keep
        # 'edge_color_norm_vmax', # Keep
        'color_edges_by_neighbor_degree', # Edges are binary, state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edges are binary, state not based on this
        'birth_active_neighbor_counts', # Node eligibility based on degree sum
        'birth_avg_neighbor_degree_ranges', # Node eligibility based on degree sum
        'survival_self_degree_counts', # Node eligibility based on degree sum
        'survival_active_neighbor_counts', # Node eligibility based on degree sum
        'survival_avg_neighbor_degree_ranges', # Node eligibility based on degree sum
        'node_history_depth'
    }
    # --- END MODIFIED ---

    produces_binary_edges: ClassVar[bool] = True
    node_state_type: ClassVar[StateType] = StateType.INTEGER
    edge_state_type: ClassVar[StateType] = StateType.BINARY # Edges are 0/1
    min_node_state: ClassVar[float] = 0.0 # Degree cannot be negative
    max_node_state: ClassVar[float] = 26.0 # Theoretical max degree (Moore 3D)
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup (Edges are binary).', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern (Nodes initialized to 0, state becomes degree).", "default": "Random", "allowed_values": ['Random'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Approx initial density of nodes with connections (influences edge init).", "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Initialization"},

        # === Edge Update Logic ("Edge GoL" based ONLY on Neighbor Degree) ===
        "edge_birth_scores": { 'type': list, 'element_type': int, 'default': [0, 3], 'description': "Edge Logic: List of exact NEIGHBOR degrees (prev step) for edge birth.", "parameter_group": "Edge Logic (Neighbor Degree GoL)" },
        "edge_survival_scores": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Edge Logic: List of exact NEIGHBOR degrees (prev step) for edge survival.", "parameter_group": "Edge Logic (Neighbor Degree GoL)" },
        "random_edge_flip_prob": { "type": float, "description": "Edge Logic: Probability of randomly flipping a potential edge's state (0->1 or 1->0).", "min": 0.0, "max": 0.1, "default": 0.0, "parameter_group": "Edge Logic (Randomness)" },

        # === Final State Logic ===
        "final_death_degree_counts": { 'type': list, 'element_type': int, 'default': [0], 'description': "Final State: Node state becomes 0 if its *final* calculated degree is in this list.", "parameter_group": "Final State Death"},

        # === Visualization ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (degree).", "default": True, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states (degrees).", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 8.0, "parameter_group": "Visualization: Nodes"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on state (0 or 1).",
            "default": False, "parameter_group": "Visualization: Edges" # Default False for binary
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "binary", "parameter_group": "Visualization: Edges", # Default binary
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # --- Other ---
        "node_history_depth": {
            'type': int, 'default': 10, 'min': 0, 'max': 100,
            'description': "Grid Setting: Number of previous node states stored internally (not used by this rule's logic).",
            'parameter_group': 'History'
        }
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Edge Degree Life"
        metadata.description = "Node state = degree. Edges follow GoL-like rules based on combined degree of endpoints (prev step). Requires post-update step."
        metadata.category = "Experimental"
        metadata.tags = ["Life", "Edges", "GoL", "Dual", "Connectivity", "Degree", "Experimental"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Edge Degree Life" # Corrected name
        self._params = {}
        self.requires_post_edge_state_update = True
        self.needs_neighbor_degrees = True
        self.needs_neighbor_active_counts = False
        self.skip_standard_tiebreakers = True
        # --- ADDED: Ensure edge coloring defaults are set ---
        self._params.setdefault('use_state_coloring_edges', False)
        self._params.setdefault('edge_colormap', 'binary')
        self._params.setdefault('edge_color_norm_vmin', 0.0)
        self._params.setdefault('edge_color_norm_vmax', 1.0)
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    # --- Methods (_compute_new_state, _compute_new_edges, _compute_final_state) remain unchanged ---
    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """
        Calculates the node's ELIGIBILITY proxy state (0 or 1).
        Simplified: Returns 1.0 if the node had any connections (degree > 0)
        in the previous step, 0.0 otherwise.
        """
        eligibility_proxy = 1.0 if neighborhood.node_state > 1e-6 else 0.0
        return eligibility_proxy

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """
        Compute edge state using GoL-like rules based on a score derived from
        the degrees of BOTH endpoints in the previous step.
        Score = (Degree(Self) - 1) + (Degree(Neighbor) - 1).
        """
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        edge_birth_scores = self.get_param('edge_birth_scores', [0, 3], neighborhood=neighborhood) # List of scores for birth
        edge_survival_scores = self.get_param('edge_survival_scores', [2, 3], neighborhood=neighborhood) # List of scores for survival
        random_flip_prob = self.get_param('random_edge_flip_prob', 0.0, neighborhood=neighborhood)

        # Node's degree from previous step (is its state)
        self_prev_degree = int(neighborhood.node_state)

        if detailed_logging_enabled:
            log_func = logger.info # Use INFO level for this diagnostic
            log_func(f"--- Node {node_idx}: {self.name} _compute_new_edges (Symmetric Score Logic R35) ---") # type: ignore [attr-defined]
            log_func(f"    Self Prev Degree: {self_prev_degree}") # type: ignore [attr-defined]
            log_func(f"    Neighbor Degrees Dict Received: {neighborhood.neighbor_degrees}") # type: ignore [attr-defined]
            log_func(f"    Birth Scores: {edge_birth_scores}, Survival Scores: {edge_survival_scores}") # type: ignore [attr-defined]
        else:
            log_func = lambda *args, **kwargs: None # No-op if detailed logging off

        for neighbor_idx, neighbor_prev_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            # Skip invalid neighbors and avoid processing pairs twice
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            # Get neighbor's degree from previous step using the passed data
            neighbor_prev_degree = 0
            if neighborhood.neighbor_degrees is not None:
                 neighbor_prev_degree = neighborhood.neighbor_degrees.get(neighbor_idx, 0)
            else:
                 logger.warning(f"Node {node_idx}: neighbor_degrees missing for neighbor {neighbor_idx}. Falling back to state.")
                 neighbor_prev_degree = int(neighbor_prev_state) if neighbor_prev_state > 0 else 0 # Fallback

            # --- Calculate Symmetric Score ---
            score = (self_prev_degree - 1) + (neighbor_prev_degree - 1)
            # ---

            edge = (node_idx, neighbor_idx) # Canonical order
            has_current_edge = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0) > 1e-6

            new_edge_state = 0.0 # Default death/no birth
            decision_reason = "Default (No Edge)"

            # --- Logic Based on Symmetric Score ---
            if not has_current_edge: # Edge Birth Check
                if score in edge_birth_scores:
                    new_edge_state = 1.0
                    decision_reason = f"Birth (Score={score} in {edge_birth_scores})"
                else:
                     decision_reason = f"No Birth (Score={score} not in {edge_birth_scores})"
            else: # Edge Survival Check
                if score in edge_survival_scores:
                    new_edge_state = 1.0
                    decision_reason = f"Survival (Score={score} in {edge_survival_scores})"
                else:
                     decision_reason = f"Death (Score={score} not in {edge_survival_scores})"
            # ---

            # Apply random flip
            random_flip_applied = False
            if np.random.random() < random_flip_prob:
                new_edge_state = 1.0 - new_edge_state
                random_flip_applied = True
                decision_reason += " + Random Flip"

            log_func(f"    Edge ({node_idx}<->{neighbor_idx}): SelfDeg={self_prev_degree}, NeighDeg={neighbor_prev_degree} -> Score={score}. Decision: {decision_reason}. Final State={new_edge_state:.0f}") # type: ignore [attr-defined]

            if new_edge_state > 0.5:
                new_edges[edge] = 1.0 # Binary edges

        return new_edges

    def _compute_final_state(self,
                             node_idx: int,
                             current_proxy_state: float, # Eligibility/state from current step's computation
                             final_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]], # Edges calculated for *current* step
                             dimensions: Tuple[int,...],
                             # --- Accept all arguments even if unused ---
                             previous_node_states: npt.NDArray[np.float64],
                             previous_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
                             previous_edge_states: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float],
                             previous_node_degrees: Optional[npt.NDArray[np.int32]],
                             previous_active_neighbors: Optional[npt.NDArray[np.int32]],
                             eligibility_proxies: Optional[np.ndarray] = None,
                             detailed_logging_enabled: bool = False
                             ) -> float:
        """
        Calculates the final state (degree) based on eligibility and final edge count, applying death list.
        (Round 3 Fix: Remove final_death_degree_counts check)
        (Round 91: Updated signature)
        (Round 95: Use passed data for coloring)
        (Round 12: Accept all 10 args)
        """
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_final_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Proxy State (Eligibility): {current_proxy_state:.1f}") # type: ignore [attr-defined]

        # If proxy state indicates ineligible (0.0), final state is 0
        if current_proxy_state < 0.5:
             if detailed_logging_enabled: logger.detail(f"    Node was ineligible based on prev step conditions, final state is 0.") # type: ignore [attr-defined]
             return 0.0

        # If eligible (proxy was 1.0), calculate final degree based on the FINAL edges for this step
        node_coords = tuple(_unravel_index(node_idx, dimensions))
        final_degree = sum(1 for edge in final_edges if node_coords in edge)

        if detailed_logging_enabled:
            logger.detail(f"    Node was eligible. Final Calculated Degree: {final_degree}") # type: ignore [attr-defined]

        # Apply death list based on the FINAL degree
        final_death_degrees = self.get_param('final_death_degree_counts', [0]) # Using new default

        final_state = 0.0 # Default to death
        decision_reason = "Default (Death)"
        if final_degree in final_death_degrees:
            decision_reason = f"Final Death (Final Degree={final_degree} in death list {final_death_degrees})"
        else:
            # Survived final death checks, state is the final degree
            final_state = float(final_degree)
            decision_reason = f"Final Survival (Final Degree={final_degree} not in death list)"

        if detailed_logging_enabled:
            logger.detail(f"    Final Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    Final State: {final_state:.1f}") # type: ignore [attr-defined]
        return final_state
    
class ModularLife(Rule):
    """
    Modular GoL variant with configurable node/edge interactions.
    Node state is binary (0/1), Edge state is continuous (0-1).
    Different logic groups for node and edge updates can be toggled via parameters.
    (Round 35: Added edge visualization parameters)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        # 'use_state_coloring_edges', # Keep
        # 'edge_colormap', # Keep
        # 'edge_color_norm_vmin', # Keep
        # 'edge_color_norm_vmax', # Keep
        'color_edges_by_neighbor_degree', # Edge state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edge state not based on this
        'tiebreaker_type', # Not used by core logic
        'node_history_depth'
    }
    # --- END MODIFIED ---

    node_state_type: ClassVar[StateType] = StateType.BINARY # Nodes are 0/1
    edge_state_type: ClassVar[StateType] = StateType.REAL   # Edges are 0.0-1.0
    min_node_state: ClassVar[float] = 0.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0
    produces_binary_edges: ClassVar[bool] = True

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern (Nodes initialized to 0).", "default": "Random", "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Approx initial density of nodes with connections.", "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability for RANDOM edge init.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "min_edge_weight": { "type": float, "description": "Minimum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Initialization"},
        "max_edge_weight": { "type": float, "description": "Maximum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.8, "parameter_group": "Initialization"},

        # === Node Logic Group 1: GoL Counts ===
        "use_gol_counts_group": {'type': bool, 'default': True, 'description': "Node Logic: Enable GoL B/S neighbor counts.", "parameter_group": "Node Group 1: GoL Counts"},
        "birth_neighbor_counts": { 'type': list, 'element_type': int, 'default': [3], 'description': "Node Logic: Active neighbor counts for birth.", "parameter_group": "Node Group 1: GoL Counts"},
        "survival_neighbor_counts": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Node Logic: Active neighbor counts for survival.", "parameter_group": "Node Group 1: GoL Counts"},

        # === Node Logic Group 2: Edge Strength Sum ===
        "use_edge_strength_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable edge strength sum condition (uses prev step edges to active neighbors).", "parameter_group": "Node Group 2: Edge Strength Sum"},
        "birth_min_edge_strength_sum": { 'type': float, 'default': 1.0, 'description': "Node Logic: Min edge sum to active neighbors for birth.", "min": 0.0, "max": 26.0, "parameter_group": "Node Group 2: Edge Strength Sum"},
        "survival_min_edge_strength_sum": { 'type': float, 'default': 0.5, 'description': "Node Logic: Min edge sum to active neighbors for survival.", "min": 0.0, "max": 26.0, "parameter_group": "Node Group 2: Edge Strength Sum"},

        # === Node Logic Group 3: Degree Survival (Prev Step) ===
        "use_degree_survival_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable survival based on degree range (uses PREVIOUS step degree). Requires neighbor_degrees data.", "parameter_group": "Node Group 3: Degree Survival (Prev)"},
        "survival_degree_min": { 'type': int, 'default': 1, 'description': "Node Logic: Min connections (prev step) for survival eligibility.", "min": 0, "max": 26, "parameter_group": "Node Group 3: Degree Survival (Prev)"},
        "survival_degree_max": { 'type': int, 'default': 5, 'description': "Node Logic: Max connections (prev step) for survival eligibility.", "min": 0, "max": 26, "parameter_group": "Node Group 3: Degree Survival (Prev)"},

        # === Node Logic Group 4: Avg Edge State ===
        "use_avg_edge_influence_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable influence from avg neighbor edge state (uses PREVIOUS step edges).", "parameter_group": "Node Group 4: Avg Edge State"},
        "birth_min_avg_edge_state": { 'type': float, 'default': 0.4, 'description': "Node Logic: Min avg edge state (prev step) required for birth.", "min": 0.0, "max": 1.0, "parameter_group": "Node Group 4: Avg Edge State"},
        "survival_min_avg_edge_state": { 'type': float, 'default': 0.1, 'description': "Node Logic: Min avg edge state (prev step) required for survival.", "min": 0.0, "max": 1.0, "parameter_group": "Node Group 4: Avg Edge State"},

        # === Node Logic Group 5: Isolation/Overcrowding (Current Step) ===
        "use_isolation_overcrowding_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable death based on CURRENT degree (calculated from active edges). Applied after other checks.", "parameter_group": "Node Group 5: Isolation/Overcrowding (Current)"},
        "isolation_death_threshold": { 'type': int, 'default': 1, 'description': "Node Logic: Force death if CURRENT degree < this.", "min": 0, "max": 26, "parameter_group": "Node Group 5: Isolation/Overcrowding (Current)"},
        "overcrowding_death_threshold": { 'type': int, 'default': 7, 'description': "Node Logic: Force death if CURRENT degree > this.", "min": 0, "max": 26, "parameter_group": "Node Group 5: Isolation/Overcrowding (Current)"},

        # === Node Logic Group 6: Random Birth Boost ===
        "use_node_randomness_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable random node birth boost.", "parameter_group": "Node Group 6: Random Birth Boost"},
        "random_birth_boost": { "type": float, "description": "Node Logic: Small random chance boost for inactive node activation.", "min": 0.0, "max": 0.1, "default": 0.005, "parameter_group": "Node Group 6: Random Birth Boost"},

        # === Edge Logic Group A: Node Activity ===
        "use_node_activity_edge_group": {'type': bool, 'default': True, 'description': "Edge Logic (Priority 4): Enable edge target based on connected nodes' predicted activity.", "parameter_group": "Edge Group A: Node Activity"},
        "edge_target_factor": { "type": float, "description": "Edge Logic: Multiplier for avg predicted node state to get target edge state.", "min": 0.0, "max": 2.0, "default": 1.0, "parameter_group": "Edge Group A: Node Activity"},
        "edge_activity_change_rate": { "type": float, "description": "Edge Logic: Rate edge moves towards activity target.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Edge Group A: Node Activity"},

        # === Edge Logic Group B: Similarity ===
        "use_similarity_edge_group": {'type': bool, 'default': False, 'description': "Edge Logic (Priority 3): Enable edge target/rate based on similarity of predicted node states. Overrides Group A.", "parameter_group": "Edge Group B: Similarity"},
        "similarity_threshold": { "type": float, "description": "Edge Logic: Max state difference to consider nodes 'similar'.", "min": 0.0, "max": 1.0, "default": 0.3, "parameter_group": "Edge Group B: Similarity"},
        "similarity_strengthen_factor": { "type": float, "description": "Edge Logic: Factor modifying change rate towards 1.0 when nodes are similar.", "min": 0.0, "max": 1.0, "default": 1.0, "parameter_group": "Edge Group B: Similarity"},
        "dissimilarity_decay_factor": { "type": float, "description": "Edge Logic: Factor modifying change rate towards 0.0 when nodes are dissimilar.", "min": 0.0, "max": 1.0, "default": 1.0, "parameter_group": "Edge Group B: Similarity"},

        # === Edge Logic Group C: Neighbor Degree (Prev Step) ===
        "use_neighbor_degree_edge_group": {'type': bool, 'default': False, 'description': "Edge Logic (Priority 2): Enable edge target/rate based on neighbor degree thresholds (uses PREVIOUS step degree). Requires neighbor_degrees data. Overrides Groups A, B.", "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "connect_neighbor_degree_min": { 'type': int, 'default': 1, 'description': "Edge Logic: Connect if neighbor degree (prev step) >= this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "connect_neighbor_degree_max": { 'type': int, 'default': 4, 'description': "Edge Logic: Connect if neighbor degree (prev step) <= this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "disconnect_neighbor_degree_min": { 'type': int, 'default': 1, 'description': "Edge Logic: Disconnect if neighbor degree (prev step) < this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "disconnect_neighbor_degree_max": { 'type': int, 'default': 6, 'description': "Edge Logic: Disconnect if neighbor degree (prev step) > this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "neighbor_degree_change_rate": { "type": float, "description": "Edge Logic: Rate edge moves towards 0/1 based on neighbor degree.", "min": 0.0, "max": 1.0, "default": 0.6, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},

        # === Edge Logic Group D: Node Effects (Prev Step) ===
        "use_node_effects_on_edges_group": {'type': bool, 'default': False, 'description': "Edge Logic (Priority 1): Enable edge target/rate based on neighbor count (uses PREVIOUS step counts). Requires neighbor_active_counts data. Overrides Groups A, B, C.", "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_survival_neighbor_min": { 'type': int, 'default': 2, 'description': "Edge Logic: Min active neighbors (prev step) for nodes to boost edge survival.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_survival_neighbor_max": { 'type': int, 'default': 4, 'description': "Edge Logic: Max active neighbors (prev step) for nodes to boost edge survival.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_survival_boost_rate": { 'type': float, 'default': 0.1, 'description': "Edge Logic: Rate edge moves towards 1 if node neighbor counts optimal.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_death_neighbor_min": { 'type': int, 'default': 1, 'description': "Edge Logic: Edge death boost if node neighbor count (prev step) < this.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_death_neighbor_max": { 'type': int, 'default': 5, 'description': "Edge Logic: Edge death boost if node neighbor count (prev step) > this.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_death_boost_rate": { 'type': float, 'default': 0.2, 'description': "Edge Logic: Rate edge moves towards 0 if node neighbor counts non-optimal.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Group D: Node Effects (Prev)"},

        # === Edge Logic Group E: Decay ===
        "use_edge_decay_group": {'type': bool, 'default': True, 'description': "Edge Logic: Enable Constant Edge Decay.", "parameter_group": "Edge Group E: Decay"},
        "edge_decay_rate": { "type": float, "description": "Edge Logic: Constant decay applied each step.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Group E: Decay"},

        # === Edge Logic Group F: Randomness ===
        "use_edge_randomness_group": {'type': bool, 'default': False, 'description': "Edge Logic: Enable Random Edge Fluctuations.", "parameter_group": "Edge Group F: Randomness"},
        "random_edge_change_prob": { "type": float, "description": "Edge Logic: Probability of random change.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Group F: Randomness"},
        "random_edge_change_amount": { "type": float, "description": "Edge Logic: Max amount (+/-) of random change.", "min": 0.0, "max": 0.5, "default": 0.15, "parameter_group": "Edge Group F: Randomness"},

        # --- Edge Pruning ---
        "min_edge_state_to_keep": { "type": float, "description": "Edge Logic: Minimum edge state required to be kept.", "min": 0.0, "max": 0.5, "default": 0.02, "parameter_group": "Edge Pruning"},

        # --- Visualization ---
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (0 or 1).", "default": False, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states.", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization: Nodes"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on their state value (0-1).",
            "default": True, "parameter_group": "Visualization: Edges" # Default True for continuous
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "prism", "parameter_group": "Visualization: Edges",
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # --- Other ---
        "node_history_depth": {
            'type': int, 'default': 10, 'min': 0, 'max': 100,
            'description': "Grid Setting: Number of previous node states stored internally (not used by this rule's logic).",
            'parameter_group': 'History'
        }
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Modular Life"
        metadata.description = "GoL foundation with toggleable logic groups for node/edge interactions. Node state 0/1, Edge state 0-1. Uses 2nd-order neighbor active counts if Group 5 enabled." # Updated description
        metadata.category = "Experimental"
        metadata.tags = ["Life", "Edges", "Feedback", "Modular", "Configurable", "Experimental"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Modular Life"
        self._params = {}
        self.requires_post_edge_state_update = False # State determined directly
        # --- Set flags based on potential parameter usage ---
        self.needs_neighbor_degrees = False # Only need active counts for Group 5
        self.needs_neighbor_active_counts = True # Needed for Group 5
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute node state based on enabled logic groups, prioritizing GoL."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Get toggle parameters
        use_gol = self.get_param('use_gol_node_logic', True, neighborhood=neighborhood)
        use_edge_effects = self.get_param('use_edge_effects_on_nodes', False, neighborhood=neighborhood)

        # --- Determine Base Eligibility/Target based on GoL (if enabled) ---
        gol_eligible = True # Assume eligible if GoL group is off
        if use_gol:
            birth_counts = self.get_param('birth_neighbor_counts', [3], neighborhood=neighborhood)
            survival_counts = self.get_param('survival_neighbor_counts', [2, 3], neighborhood=neighborhood)
            num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
            if neighborhood.node_state <= 0: # Birth check
                gol_eligible = num_active_neighbors in birth_counts
            else: # Survival check
                gol_eligible = num_active_neighbors in survival_counts

        if detailed_logging_enabled:
            logger.detail(f"Node {node_idx}: GoL Check (Enabled={use_gol}): Eligible={gol_eligible}") # type: ignore [attr-defined]

        # --- Apply Edge Effects (if enabled AND GoL passed) ---
        final_eligible = gol_eligible
        if final_eligible and use_edge_effects:
            survival_min_deg = self.get_param('node_survival_degree_min', 1, neighborhood=neighborhood)
            survival_max_deg = self.get_param('node_survival_degree_max', 5, neighborhood=neighborhood)
            death_min_deg = self.get_param('node_death_degree_min', 1, neighborhood=neighborhood)
            death_max_deg = self.get_param('node_death_degree_max', 6, neighborhood=neighborhood)
            survival_boost_rate = self.get_param('node_survival_boost_rate', 0.1, neighborhood=neighborhood)
            death_boost_rate = self.get_param('node_death_boost_rate', 0.2, neighborhood=neighborhood)

            node_degree = sum(1 for idx, state in neighborhood.neighbor_edge_states.items() if idx >= 0 and state > 1e-6)

            if neighborhood.node_state > 0: # Only apply to currently active nodes
                if survival_min_deg <= node_degree <= survival_max_deg:
                    # Degree is optimal, potentially boost towards 1 (though it's already 1 if gol_eligible)
                    # This logic might be better for continuous states. For binary, it confirms survival.
                    if detailed_logging_enabled: logger.detail(f"    EdgeEffect: Survival confirmed by degree {node_degree} in [{survival_min_deg}-{survival_max_deg}]") # type: ignore [attr-defined]
                elif node_degree < death_min_deg or node_degree > death_max_deg:
                    # Degree is non-optimal, potentially boost towards 0
                    if np.random.random() < death_boost_rate: # Probabilistic death
                        final_eligible = False
                        if detailed_logging_enabled: logger.detail(f"    EdgeEffect: Death override by degree {node_degree} outside [{death_min_deg}-{death_max_deg}]") # type: ignore [attr-defined]
                    # else: # No death boost applied
                        # if detailed_logging_enabled: logger.detail(f"    EdgeEffect: Death boost failed random check.") # type: ignore [attr-defined]
            # else: # Node is inactive, edge effects don't apply to birth in this version

        new_state = 1.0 if final_eligible else 0.0

        if detailed_logging_enabled:
            logger.detail(f"    Final State: {new_state:.1f}") # type: ignore [attr-defined]

        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Compute new continuous edge states based on enabled logic groups."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Get toggle parameters
        use_basic, use_advanced, use_node_effects, use_random = self._get_parameters(
             'use_basic_edge_logic', 'use_advanced_edge_logic',
             'use_node_effects_on_edges', 'use_edge_randomness_group' # Implicitly enabled by prob/amount > 0
        )
        # Get specific parameters needed
        basic_target_factor, basic_change_rate, basic_decay = self._get_parameters(
            'basic_edge_target_factor', 'basic_edge_change_rate', 'basic_edge_decay_rate'
        )
        adv_birth_scores, adv_survival_scores, adv_change_rate = self._get_parameters(
            'adv_edge_birth_scores', 'adv_edge_survival_scores', 'adv_edge_change_rate'
        )
        edge_survival_neighbor_min, edge_survival_neighbor_max, edge_survival_boost_rate, \
        edge_death_neighbor_min, edge_death_neighbor_max, edge_death_boost_rate = (0,0,0,0,0,0)
        if use_node_effects:
            edge_survival_neighbor_min, edge_survival_neighbor_max, edge_survival_boost_rate, \
            edge_death_neighbor_min, edge_death_neighbor_max, edge_death_boost_rate = self._get_parameters(
                'edge_survival_neighbor_min', 'edge_survival_neighbor_max', 'edge_survival_boost_rate',
                'edge_death_neighbor_min', 'edge_death_neighbor_max', 'edge_death_boost_rate'
            )
        random_edge_prob, random_edge_amount, min_keep = self._get_parameters(
            'random_edge_change_prob', 'random_edge_change_amount', 'min_edge_state_to_keep'
        )
        # Check if randomness is effectively enabled
        use_random = use_random and random_edge_prob > 0 and random_edge_amount > 0

        # Determine the *next* state of the current node (proxy 0 or 1)
        next_node_state = self._compute_new_state(neighborhood, detailed_logging_enabled)
        self_is_active_next = next_node_state > 0.5

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue # Skip invalid and avoid double check

            edge = (node_idx, neighbor_idx) # Canonical order
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0)

            # Approximation: Use current neighbor state as proxy for neighbor's next state.
            neighbor_is_active_next = neighbor_state > 0.5

            # --- Get Neighbor's Previous Active Neighbor Count (for Group 5) ---
            neighbor_prev_active_count = 0
            if use_node_effects:
                if neighborhood.neighbor_active_counts is not None:
                     neighbor_prev_active_count = neighborhood.neighbor_active_counts.get(neighbor_idx, 0)
                else: logger.warning(f"Node {node_idx}: neighbor_active_counts missing for neighbor {neighbor_idx}")
            # ---

            # --- Determine Target Edge State & Change Rate ---
            target_edge_state = current_edge_state # Default: maintain
            effective_change_rate = 0.0 # Default: no change
            decay_to_apply = 0.0

            # 1. Basic Edge Logic (if enabled)
            if use_basic:
                if self_is_active_next and neighbor_is_active_next:
                    avg_node_state = (next_node_state + neighbor_state) / 2.0 # Use current neighbor state
                    target_edge_state = avg_node_state * basic_target_factor
                else:
                    target_edge_state = 0.0
                effective_change_rate = basic_change_rate
                decay_to_apply = basic_decay # Apply decay if basic logic is on

            # 2. Advanced Edge Logic (if enabled, overrides basic target/rate)
            if use_advanced:
                self_degree = sum(1 for s in neighborhood.neighbor_edge_states.values() if s > 1e-6)
                # Need neighbor's degree - NOT available directly. Approximate using its state? Risky.
                # Let's assume degree = state for this approximation (only works if node state = degree)
                # This group might need neighbor_degrees passed if node state isn't degree.
                # For now, using neighbor_state as proxy degree.
                neighbor_degree_proxy = int(neighbor_state) if neighbor_state > 0 else 0
                shared_score = float(self_degree + neighbor_degree_proxy)
                adv_target = 0.0
                if current_edge_state > 1e-6: # Survival
                     if any(mn <= shared_score <= mx for mn, mx in zip(adv_survival_scores[::2], adv_survival_scores[1::2])): adv_target = 1.0
                else: # Birth
                     if any(mn <= shared_score <= mx for mn, mx in zip(adv_birth_scores[::2], adv_birth_scores[1::2])): adv_target = 1.0
                target_edge_state = adv_target # Override target
                effective_change_rate = adv_change_rate # Override rate
                # Decay is still applied based on basic_decay if use_basic is also true

            # 3. Node Effects on Edges (if enabled, overrides target/rate)
            if use_node_effects:
                 self_neighbors_count = self._count_active_neighbors(neighborhood.neighbor_states) # Use binary count
                 neighbor_neighbors_count = neighbor_prev_active_count # Use pre-calculated value

                 self_survival_boost = (edge_survival_neighbor_min <= self_neighbors_count <= edge_survival_neighbor_max)
                 neighbor_survival_boost = (edge_survival_neighbor_min <= neighbor_neighbors_count <= edge_survival_neighbor_max)
                 self_death_boost = (self_neighbors_count < edge_death_neighbor_min or self_neighbors_count > edge_death_neighbor_max)
                 neighbor_death_boost = (neighbor_neighbors_count < edge_death_neighbor_min or neighbor_neighbors_count > edge_death_neighbor_max)

                 if self_survival_boost and neighbor_survival_boost:
                      target_edge_state = 1.0 # Override target
                      effective_change_rate = edge_survival_boost_rate # Override rate
                 elif self_death_boost or neighbor_death_boost:
                      target_edge_state = 0.0 # Override target
                      effective_change_rate = edge_death_boost_rate # Override rate
                 # Else: Keep target/rate from previous groups

            # --- Apply Changes ---
            new_edge_state = current_edge_state + (target_edge_state - current_edge_state) * effective_change_rate

            # Apply Decay (only if basic group enabled it)
            new_edge_state -= decay_to_apply

            # Apply Randomness
            if use_random and np.random.random() < random_edge_prob:
                change = (np.random.random() * 2 - 1) * random_edge_amount
                new_edge_state += change

            # Clip and Prune
            clipped_new_edge_state = float(np.clip(new_edge_state, 0.0, 1.0))
            if clipped_new_edge_state >= min_keep:
                 new_edges[edge] = clipped_new_edge_state

        return new_edges

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
                             eligibility_proxies: Optional[np.ndarray] = None,
                             detailed_logging_enabled: bool = False
                             # ---
                             ) -> float:
        """Calculates the final node state based on the eligibility proxy."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_final_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Proxy State (Eligibility): {current_proxy_state:.1f}") # type: ignore [attr-defined]

        # The final state is simply the eligibility proxy (0 or 1)
        final_state = current_proxy_state

        if detailed_logging_enabled:
            logger.detail(f"    Final State: {final_state:.1f}") # type: ignore [attr-defined]

        return final_state

class ConfigurableContinuousLife(Rule):
    """
    Highly configurable rule with continuous node/edge states (-1 to 1), aiming for dynamic interactions.
    Node state influenced by weighted averages of neighbors/edges/connections and thresholds.
    Edge state evolves based on node similarity or average activity, with inertia, decay, and randomness.
    (Round 15: Expose visualization parameters)
    """
    # --- MODIFIED: Ensure visualization params are NOT excluded ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        # 'use_state_coloring', # Keep
        # 'color_nodes_by_degree', # Keep
        # 'color_nodes_by_active_neighbors', # Keep
        # 'use_state_coloring_edges', # Keep
        # 'edge_coloring_mode', # Keep
        'color_edges_by_neighbor_degree', # Not applicable/used
        'color_edges_by_neighbor_active_neighbors', # Not applicable/used
        'tiebreaker_type', # Not used by core logic
        'node_history_depth' # Grid setting, not rule logic
    }
    # --- END MODIFIED ---

    node_state_type: ClassVar[StateType] = StateType.REAL # Nodes are -1.0 to 1.0
    edge_state_type: ClassVar[StateType] = StateType.REAL # Edges are -1.0 to 1.0
    min_node_state: ClassVar[float] = -1.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = -1.0
    max_edge_state: ClassVar[float] = 1.0

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Grid dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'bounded', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern.", "default": "Random", "allowed_values": ['Random', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', 'Glider Pattern', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Initial density of nodes with non-zero state.", "min": 0.0, "max": 1.0, "default": 0.6, "parameter_group": "Initialization"},
        "initial_state_min": { "type": float, "description": "Min initial state value (-1 to 1) for active nodes.", "min": -1.0, "max": 1.0, "default": -0.8, "parameter_group": "Initialization"},
        "initial_state_max": { "type": float, "description": "Max initial state value (-1 to 1) for active nodes.", "min": -1.0, "max": 1.0, "default": 0.8, "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability for RANDOM edge initialization.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "min_edge_weight": { "type": float, "description": "Min initial random edge weight (-1 to 1).", "min": -1.0, "max": 1.0, "default": -0.5, "parameter_group": "Initialization"},
        "max_edge_weight": { "type": float, "description": "Max initial random edge weight (-1 to 1).", "min": -1.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},

        # === Node Logic: Influence Weights ===
        "influence_neighbor_state_weight": { "type": float, "description": "Node Logic: Weight for avg neighbor state in combined influence calculation.", "min": 0.0, "max": 1.0, "default": 0.6, "parameter_group": "Node Logic: Influence Weights"},
        "influence_edge_state_weight": { "type": float, "description": "Node Logic: Weight for avg connecting edge state in combined influence calculation.", "min": 0.0, "max": 1.0, "default": 0.3, "parameter_group": "Node Logic: Influence Weights"},
        "influence_connection_count_weight": { "type": float, "description": "Node Logic: Weight for connection count factor in combined influence calculation.", "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Node Logic: Influence Weights"},
        "connection_count_saturation": { "type": int, "description": "Node Logic: Connection count at which its positive influence saturates (used in influence calc).", "min": 1, "max": 26, "default": 5, "parameter_group": "Node Logic: Influence Weights"},
        # === Node Logic: Thresholds ===
        "activation_threshold": { "type": float, "description": "Node Logic: Combined influence needed for inactive node activation target (target becomes influence value).", "min": 0.0, "max": 1.0, "default": 0.2, "parameter_group": "Node Logic: Thresholds"},
        "deactivation_threshold": { "type": float, "description": "Node Logic: Node target state becomes 0 if combined influence falls below this.", "min": 0.0, "max": 1.0, "default": 0.08, "parameter_group": "Node Logic: Thresholds"},
        "overcrowding_death_threshold": { "type": int, "description": "Node Logic: Node target state becomes 0 if active connection count exceeds this.", "min": 0, "max": 26, "default": 7, "parameter_group": "Node Logic: Thresholds"},
        "isolation_death_threshold": { "type": int, "description": "Node Logic: Node target state becomes 0 if active connection count is less than this.", "min": 0, "max": 26, "default": 1, "parameter_group": "Node Logic: Thresholds"},
        # === Node Logic: Rate & Oscillation ===
        "state_change_rate": { "type": float, "description": "Node Logic: Rate (0-1) at which node state moves towards its calculated target state.", "min": 0.0, "max": 1.0, "default": 0.65, "parameter_group": "Node Logic: Rate & Oscillation"},
        "random_state_boost": { "type": float, "description": "Node Logic: Small random boost (+/-) added to combined influence.", "min": 0.0, "max": 0.2, "default": 0.05, "parameter_group": "Node Logic: Randomness"},
        "oscillation_center": { "type": float, "description": "Node Logic: Center point for state oscillation (around 0).", "min": -1.0, "max": 1.0, "default": 0.0, "parameter_group": "Node Logic: Rate & Oscillation"},

        # === Edge Logic ===
        "use_similarity_edge_logic": { 'type': bool, 'default': False, 'description': "Edge Logic: If True, use similarity/dissimilarity; If False, use average node state.", "parameter_group": "Edge Logic: Mode"},
        "edge_formation_threshold": { "type": float, "description": "Avg State Mode: Avg node state needed to form a new edge.", "min": 0.0, "max": 1.0, "default": 0.45, "parameter_group": "Edge Logic: Average State Mode"},
        "edge_target_factor": { "type": float, "description": "Avg State Mode: Multiplier for avg node state to get target edge state.", "min": 0.0, "max": 2.0, "default": 1.0, "parameter_group": "Edge Logic: Average State Mode"},
        "similarity_threshold": { "type": float, "description": "Similarity Mode: Max state difference to consider nodes 'similar'.", "min": 0.0, "max": 1.0, "default": 0.3, "parameter_group": "Edge Logic: Similarity Mode"},
        "similarity_strengthen_factor": { "type": float, "description": "Similarity Mode: Factor modifying change rate towards 1.0 when nodes are similar.", "min": 0.0, "max": 1.0, "default": 1.0, "parameter_group": "Edge Logic: Similarity Mode"},
        "dissimilarity_decay_factor": { "type": float, "description": "Similarity Mode: Factor modifying change rate towards 0.0 when nodes are dissimilar.", "min": 0.0, "max": 1.0, "default": 1.0, "parameter_group": "Edge Logic: Similarity Mode"},
        "edge_change_rate": { "type": float, "description": "Edge Logic: Base rate (0-1) at which edge state moves towards its target.", "min": 0.0, "max": 1.0, "default": 0.75, "parameter_group": "Edge Logic: General"},
        "edge_decay_rate": { "type": float, "description": "Edge Logic: Constant decay subtracted from edge state each step.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Logic: General"},
        "random_edge_change_prob": { "type": float, "description": "Edge Logic: Probability (0-1) of applying a small random change to edge state.", "min": 0.0, "max": 0.1, "default": 0.06, "parameter_group": "Edge Logic: General"},
        "random_edge_change_amount": { "type": float, "description": "Edge Logic: Max amount (+/-) of random change applied to edge state.", "min": 0.0, "max": 0.5, "default": 0.25, "parameter_group": "Edge Logic: General"},
        "min_edge_state_to_keep": { "type": float, "description": "Edge Logic: Minimum edge state required to be kept (non-zero).", "min": 0.0, "max": 0.5, "default": 0.02, "parameter_group": "Edge Logic: General"},

        # --- ADDED Visualization Parameters ---
        "use_state_coloring": { "type": bool, "description": "Color nodes based on their state value (-1 to 1).", "default": True, "parameter_group": "Visualization: Nodes"},
        "color_nodes_by_degree": {"type": bool, "description": "If Use State Coloring is True, color nodes based on connection count (degree) in the current step.", "default": False, "parameter_group": "Visualization: Nodes"},
        "color_nodes_by_active_neighbors": {"type": bool, "description": "If Use State Coloring is True, color nodes based on active neighbor count in the previous step.", "default": False, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states.", "default": "coolwarm", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization.", "default": -1.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization.", "default": 1.0, "parameter_group": "Visualization: Nodes"},
        "use_state_coloring_edges": { "type": bool, "description": "Color edges based on their state value (-1 to 1).", "default": True, "parameter_group": "Visualization: Edges"},
        "edge_coloring_mode": { 'type': str, 'default': 'Default', 'allowed_values': ['Default', 'ActiveNeighbors', 'DegreeSum'], 'description': "Edge Color: 'Default' (edge state), 'ActiveNeighbors' (avg active neighbors of endpoints), 'DegreeSum' (sum of endpoint degrees). Uses prev step data.", "parameter_group": "Visualization: Edges"},
        "edge_colormap": { "type": str, "description": "Colormap for edge coloring.", "default": "prism", "parameter_group": "Visualization: Edges", "allowed_values": ["(None)"] + _standard_colormaps},
        "edge_color_norm_vmin": { "type": float, "description": "Min value for edge color normalization.", "default": -1.0, "parameter_group": "Visualization: Edges"},
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # === Other ===
        "node_history_depth": { 'type': int, 'description': 'Grid Setting: History depth (not used by rule).', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Configurable Continuous Life"
        metadata.description = "Continuous state (-1 to 1) nodes/edges. State influenced by neighbor/edge averages and connection count. Edges based on avg node state or similarity. Tuned for more dynamic behavior."
        metadata.category = "Continuous"
        metadata.tags = ["Continuous", "Edges", "Weighted", "Dynamic", "Configurable"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Configurable Continuous Life"
        # --- REMOVED: Hardcoded _params setting ---
        # self._params = {} # Base class handles initialization now
        # ---
        self.requires_post_edge_state_update = False
        self.needs_neighbor_degrees = False # Not needed by core logic
        self.needs_neighbor_active_counts = False # Not needed by core logic

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    # --- Methods (_compute_new_state, _compute_new_edges) remain unchanged ---
    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute new continuous node state based on influence, connections, overcrowding, and isolation."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        w_neighbor = self.get_param('influence_neighbor_state_weight', 0.6, neighborhood=neighborhood)
        w_edge = self.get_param('influence_edge_state_weight', 0.3, neighborhood=neighborhood)
        w_connect = self.get_param('influence_connection_count_weight', 0.1, neighborhood=neighborhood)
        connect_sat = self.get_param('connection_count_saturation', 5, neighborhood=neighborhood)
        act_thr = self.get_param('activation_threshold', 0.2, neighborhood=neighborhood) # Using new default
        deact_thr = self.get_param('deactivation_threshold', 0.08, neighborhood=neighborhood) # Using new default
        overcrowd_thr = self.get_param('overcrowding_death_threshold', 7, neighborhood=neighborhood)
        isolation_thr = self.get_param('isolation_death_threshold', 1, neighborhood=neighborhood)
        state_rate = self.get_param('state_change_rate', 0.65, neighborhood=neighborhood) # Using new default
        rand_boost = self.get_param('random_state_boost', 0.05, neighborhood=neighborhood) # Using new default
        oscillation_center = self.get_param('oscillation_center', 0.0, neighborhood=neighborhood) # ADDED

        # Calculate node degree (number of active edges)
        node_degree = sum(1 for idx, state in neighborhood.neighbor_edge_states.items() if idx >= 0 and abs(state) > 1e-6) # Use abs for continuous edges

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State: {neighborhood.node_state:.4f}") # type: ignore [attr-defined]
            logger.detail(f"    Current Degree: {node_degree}") # type: ignore [attr-defined]
            logger.detail(f"    Params: W_N={w_neighbor:.2f}, W_E={w_edge:.2f}, W_C={w_connect:.2f}, Sat={connect_sat}, ActThr={act_thr:.2f}, DeactThr={deact_thr:.2f}, OverThr={overcrowd_thr}, IsoThr={isolation_thr}, Rate={state_rate:.2f}, Boost={rand_boost:.3f}, OscCenter={oscillation_center:.2f}") # type: ignore [attr-defined]

        target_state = neighborhood.node_state # Default: stay same
        decision_reason = "Default (No change)"

        # --- Priority Death Checks ---
        if neighborhood.node_state > 1e-6: # Only check for active nodes (state > 0)
            if node_degree > overcrowd_thr:
                target_state = oscillation_center # Target center (death)
                decision_reason = f"Target Center (Overcrowding: Degree={node_degree} > Thr={overcrowd_thr})"
            elif node_degree < isolation_thr:
                target_state = oscillation_center # Target center (death)
                decision_reason = f"Target Center (Isolation: Degree={node_degree} < Thr={isolation_thr})"
            else: decision_reason = "Calculate Influence (Survived Priority Death Checks)"
        else: decision_reason = "Calculate Influence (Node Inactive)"

        # --- Influence Calculation (Only if not already forced to center by priority checks) ---
        if target_state == neighborhood.node_state: # Check if target wasn't already set to center
            valid_neighbor_indices = neighborhood.neighbor_indices[neighborhood.neighbor_indices >= 0]
            num_valid_neighbors = len(valid_neighbor_indices)
            avg_neighbor_state = 0.0
            avg_neighbor_edge_state = 0.0
            if num_valid_neighbors > 0:
                valid_neighbor_states = neighborhood.neighbor_states[neighborhood.neighbor_indices >= 0]
                avg_neighbor_state = np.mean(valid_neighbor_states) if valid_neighbor_states.size > 0 else 0.0
                avg_neighbor_edge_state = neighborhood.neighborhood_metrics.get('avg_neighbor_edge_state', 0.0)
            connection_count_factor = min(1.0, node_degree / max(1, connect_sat))
            combined_influence = (avg_neighbor_state * w_neighbor) + \
                                (avg_neighbor_edge_state * w_edge) + \
                                (connection_count_factor * w_connect) + \
                                ((np.random.random() * 2 - 1) * rand_boost)

            if detailed_logging_enabled: logger.detail(f"    Influence Calc: AvgNState={avg_neighbor_state:.4f}, AvgEState={avg_neighbor_edge_state:.4f}, ConnFactor={connection_count_factor:.4f} -> Influence={combined_influence:.4f}") # type: ignore [attr-defined]

            # Determine Target State based on Influence
            if combined_influence >= act_thr: target_state = combined_influence; decision_reason = f"Target Influence (Influence >= ActivationThr)"
            elif neighborhood.node_state > 1e-6 and combined_influence >= deact_thr: target_state = combined_influence; decision_reason = f"Target Influence (Active, Influence >= DeactivationThr)"
            else: target_state = oscillation_center; decision_reason = f"Target Center (Influence < Thresholds)"

        # --- Final State Calculation ---
        new_state = neighborhood.node_state + (target_state - neighborhood.node_state) * state_rate
        clipped_new_state = float(np.clip(new_state, -1.0, 1.0)) # Clip to -1 to 1

        if detailed_logging_enabled:
            logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    Target State: {target_state:.4f}") # type: ignore [attr-defined]
            logger.detail(f"    New State (clipped): {clipped_new_state:.4f}") # type: ignore [attr-defined]

        return clipped_new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Compute new continuous edge states based on selected logic mode."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch common parameters
        edge_change_rate = self.get_param('edge_change_rate', 0.75, neighborhood=neighborhood)
        edge_decay_rate = self.get_param('edge_decay_rate', 0.01, neighborhood=neighborhood)
        random_edge_prob = self.get_param('random_edge_change_prob', 0.06, neighborhood=neighborhood)
        random_edge_amount = self.get_param('random_edge_change_amount', 0.25, neighborhood=neighborhood)
        min_keep = self.get_param('min_edge_state_to_keep', 0.02, neighborhood=neighborhood)
        use_similarity_logic = self.get_param('use_similarity_edge_logic', False, neighborhood=neighborhood)

        # Fetch mode-specific parameters
        if use_similarity_logic:
            sim_threshold = self.get_param('similarity_threshold', 0.3, neighborhood=neighborhood)
            sim_strengthen = self.get_param('similarity_strengthen_factor', 1.0, neighborhood=neighborhood)
            dissim_decay = self.get_param('dissimilarity_decay_factor', 1.0, neighborhood=neighborhood)
        else:
            avg_target_factor = self.get_param('edge_target_factor', 1.0, neighborhood=neighborhood)
            formation_thr = self.get_param('edge_formation_threshold', 0.45, neighborhood=neighborhood)

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]
            logger.detail(f"    Using Similarity Logic: {use_similarity_logic}") # type: ignore [attr-defined]

        # Use current node state as proxy for its next state in edge calculations
        node_state_proxy = neighborhood.node_state

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            edge = (node_idx, neighbor_idx) # Canonical order
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0)

            # Use current neighbor state as proxy for its next state
            neighbor_state_proxy = neighbor_state

            # --- Determine Target Edge State ---
            target_edge_state = 0.0 # Default target is weak/removed
            mode_specific_rate_factor = 1.0 # Factor to potentially modify base change rate

            if use_similarity_logic:
                state_diff = abs(node_state_proxy - neighbor_state_proxy)
                if state_diff < sim_threshold: # Similar
                    target_edge_state = 1.0
                    mode_specific_rate_factor = sim_strengthen
                else: # Dissimilar
                    target_edge_state = -1.0 # Target repulsion for dissimilar
                    mode_specific_rate_factor = dissim_decay
            else: # Average State Logic
                avg_node_state = (node_state_proxy + neighbor_state_proxy) / 2.0
                if current_edge_state == 0.0: # If edge doesn't exist
                    if abs(avg_node_state) > formation_thr: # Check formation threshold using magnitude
                        target_edge_state = avg_node_state * avg_target_factor # Target formation based on avg
                    # else: target remains 0.0
                else: # Existing edge: target based on average state
                    target_edge_state = avg_node_state * avg_target_factor
                mode_specific_rate_factor = 1.0 # Use base change rate

            # Calculate effective change rate
            effective_change_rate = edge_change_rate * mode_specific_rate_factor

            # Move edge state towards target state
            new_edge_state = current_edge_state + (target_edge_state - current_edge_state) * effective_change_rate

            # Apply constant decay (towards zero)
            decay_amount = edge_decay_rate * np.sign(new_edge_state)
            if abs(new_edge_state) > abs(decay_amount): new_edge_state -= decay_amount
            else: new_edge_state = 0.0

            # Apply random change
            if np.random.random() < random_edge_prob:
                change = (np.random.random() * 2 - 1) * random_edge_amount
                new_edge_state += change

            clipped_new_edge_state = float(np.clip(new_edge_state, -1.0, 1.0)) # Clip to -1 to 1

            # Only propose edges with a state magnitude greater than the minimum threshold
            if abs(clipped_new_edge_state) >= min_keep:
                 new_edges[edge] = clipped_new_edge_state

        return new_edges
    
class GoLNodesGoLEdges(Rule):
    """
    Applies Game of Life rules independently to nodes and edges.
    Node state is binary (0/1), determined by active neighbor counts (standard GoL).
    Edge state is binary (0/1), determined by applying GoL-like rules to the degree
    of the connected neighbor node from the *previous* step.
    (Round 36: Added edge visualization parameters)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'min_edge_weight', 'max_edge_weight', # Edges are binary
        'connect_probability', # Not used by core logic, only init
        # 'use_state_coloring_edges', # Keep
        # 'edge_colormap', # Keep
        # 'edge_color_norm_vmin', # Keep
        # 'edge_color_norm_vmax', # Keep
        'color_edges_by_neighbor_degree', # Edges are binary, state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edges are binary, state not based on this
        'tiebreaker_type',
        'node_history_depth'
    }
    # --- END MODIFIED ---

    produces_binary_edges: ClassVar[bool] = True
    node_state_type: ClassVar[StateType] = StateType.BINARY
    edge_state_type: ClassVar[StateType] = StateType.BINARY
    min_node_state: ClassVar[float] = 0.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup (Edges are binary).', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern.", "default": "Random", "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Initial density of active nodes (state=1).", "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Initialization"},

        # === Node Update Logic (GoL) ===
        "node_birth_neighbor_counts": { 'type': list, 'element_type': int, 'default': [3], 'description': "Node Logic: Active neighbor counts for node birth.", "parameter_group": "Node Logic (GoL)"},
        "node_survival_neighbor_counts": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Node Logic: Active neighbor counts for node survival.", "parameter_group": "Node Logic (GoL)"},

        # === Edge Update Logic ("Edge GoL" based on Neighbor Degree) ===
        "edge_birth_degrees": { 'type': list, 'element_type': int, 'default': [3], 'description': "Edge Logic: List of neighbor degrees (prev step) for edge birth.", "parameter_group": "Edge Logic (Degree GoL)"},
        "edge_survival_degrees": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Edge Logic: List of neighbor degrees (prev step) for edge survival.", "parameter_group": "Edge Logic (Degree GoL)"},

        # === Visualization ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (0 or 1).", "default": False, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states.", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization: Nodes"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on state (0 or 1).",
            "default": False, "parameter_group": "Visualization: Edges" # Default False for binary
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "binary", "parameter_group": "Visualization: Edges", # Default binary
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # --- Other ---
        "node_history_depth": { 'type': int, 'description': 'Grid Setting: History depth (not used by rule).', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "GoL Nodes GoL Edges"
        # --- UPDATED DESCRIPTION ---
        metadata.description = "Nodes follow GoL B/S rules based on active neighbor counts. Edges follow GoL B/S rules based on the neighbor's degree from the previous step. Edges only form/survive if both nodes are predicted active."
        # ---
        metadata.category = "Experimental"
        metadata.tags = ["Life", "Edges", "GoL", "Dual", "Connectivity", "Degree", "Experimental"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "GoL Nodes GoL Edges"
        self._params = {}
        self.requires_post_edge_state_update = False # State determined directly
        # --- Set flags to request data ---
        self.needs_neighbor_degrees = True # Need neighbor degree for edge logic
        self.needs_neighbor_active_counts = False # Not needed
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute node state using standard B/S rules based on *active neighbor count*."""
        # This is pure GoL logic based on active neighbors from previous step
        birth_rules = self.get_param('node_birth_neighbor_counts', [3], neighborhood=neighborhood)
        survival_rules = self.get_param('node_survival_neighbor_counts', [2, 3], neighborhood=neighborhood)
        num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
        new_state = 0.0
        if neighborhood.node_state <= 0: # Birth
            if num_active_neighbors in birth_rules: new_state = 1.0
        else: # Survival
            if num_active_neighbors in survival_rules: new_state = 1.0
        # logger.debug(f"Node {neighborhood.node_index}: State={neighborhood.node_state:.0f}, ActiveN={num_active_neighbors} -> NewState={new_state:.0f}") # Reduce noise
        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Compute edge state using GoL-like rules based on neighbor's previous degree."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        edge_birth_degrees = self.get_param('edge_birth_degrees', [3], neighborhood=neighborhood)
        edge_survival_degrees = self.get_param('edge_survival_degrees', [2, 3], neighborhood=neighborhood)

        # Determine eligibility of the current node for the *next* step
        self_is_eligible = self._compute_new_state(neighborhood, detailed_logging_enabled) > 0.5

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Eligible Next: {self_is_eligible}") # type: ignore [attr-defined]

        # If self is ineligible, no edges can form/survive from its perspective
        if not self_is_eligible:
             # if detailed_logging_enabled: logger.detail("    Self ineligible, proposing no edges.") # type: ignore [attr-defined] # Reduce noise
             return new_edges

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue # Skip invalid and avoid double check

            # Determine neighbor's eligibility (proxy: was active in previous step?)
            neighbor_is_eligible = neighbor_state > 0.5

            # Get neighbor's degree from previous step using the passed data
            neighbor_prev_degree = 0
            if neighborhood.neighbor_degrees is not None:
                 neighbor_prev_degree = neighborhood.neighbor_degrees.get(neighbor_idx, 0)
            else:
                 logger.warning(f"Node {node_idx}: neighbor_degrees missing for neighbor {neighbor_idx}")
                 # Fallback: Approximate degree based on current state (less accurate)
                 neighbor_prev_degree = int(neighbor_state) if neighbor_state > 0 else 0

            edge = (node_idx, neighbor_idx) # Canonical order
            # Check if edge existed in previous step
            has_current_edge = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0) > 1e-6

            new_edge_state = 0.0 # Default death
            decision_reason = "Default (No Edge)"

            # --- Apply Edge Logic ONLY if BOTH nodes are eligible ---
            if self_is_eligible and neighbor_is_eligible:
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
            elif has_current_edge: # If edge exists but one node is ineligible, it dies
                decision_reason = "Death (One node ineligible)"
            # else: # Edge doesn't exist and one node ineligible -> remains dead

            if detailed_logging_enabled:
                logger.detail(f"  Edge {edge}: NeighEligible(Proxy)={neighbor_is_eligible}, NeighPrevDeg={neighbor_prev_degree}. Decision: {decision_reason}. Final State={new_edge_state:.0f}") # type: ignore [attr-defined]

            # Add to proposed edges if it should exist
            if new_edge_state > 0.5:
                new_edges[edge] = 1.0 # Binary edges

        # if detailed_logging_enabled: # Reduce noise
            # logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges

    # No _compute_final_state needed

class EdgeFeedbackLife(Rule):
    """
    Rule with feedback between node state (binary 0/1) and edge state (continuous 0-1).
    Node birth depends on the average degree of active neighbors (prev step).
    Node survival depends on the sum of its incoming edge weights (prev step).
    Edges strengthen towards active neighbors and decay otherwise.
    (Round 8 Fix: Exclude incompatible node coloring modes)
    """
    # --- MODIFIED: Exclude degree/neighbor count coloring modes ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'color_edges_by_neighbor_degree', # Edge state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edge state not based on this
        'tiebreaker_type', # Not used by core logic
        'color_nodes_by_degree', # Exclude: Node state is binary
        'color_nodes_by_active_neighbors,' # Exclude: Node state is binary
        'node_history_depth'
    }
    # --- END MODIFIED ---

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup (continuous 0-1).', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern.", "default": "Random", "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Initial density of active nodes (state=1).", "min": 0.0, "max": 1.0, "default": 0.35, "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability for RANDOM edge init.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "min_edge_weight": { "type": float, "description": "Minimum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Initialization"},
        "max_edge_weight": { "type": float, "description": "Maximum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.8, "parameter_group": "Initialization"},

        # === State Update Logic ===
        "birth_min_avg_neighbor_degree": { 'type': float, 'default': 1.5, 'description': "Node Logic: Min avg degree of active neighbors (prev step) for birth.", "min": 0.0, "max": 26.0, "parameter_group": "Node Logic"},
        "birth_max_avg_neighbor_degree": { 'type': float, 'default': 4.0, 'description': "Node Logic: Max avg degree of active neighbors (prev step) for birth.", "min": 0.0, "max": 26.0, "parameter_group": "Node Logic"},
        "survival_min_edge_sum": { 'type': float, 'default': 0.8, 'description': "Node Logic: Min sum of edge weights from active neighbors (prev step) for survival.", "min": 0.0, "max": 26.0, "parameter_group": "Node Logic"},
        "survival_max_edge_sum": { 'type': float, 'default': 3.5, 'description': "Node Logic: Max sum of edge weights from active neighbors (prev step) for survival.", "min": 0.0, "max": 26.0, "parameter_group": "Node Logic"},

        # === Edge Update Logic ===
        "edge_change_rate": { "type": float, "description": "Edge Logic: Rate (0-1) edge state moves towards target (0 or 1).", "min": 0.0, "max": 1.0, "default": 0.3, "parameter_group": "Edge Logic"},
        "edge_decay_rate": { "type": float, "description": "Edge Logic: Constant decay subtracted from edge state each step.", "min": 0.0, "max": 0.1, "default": 0.02, "parameter_group": "Edge Logic"},
        "random_edge_change_prob": { "type": float, "description": "Edge Logic: Probability (0-1) of applying random change to edge state.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Logic"},
        "random_edge_change_amount": { "type": float, "description": "Edge Logic: Max amount (+/-) of random change applied to edge state.", "min": 0.0, "max": 0.5, "default": 0.1, "parameter_group": "Edge Logic"},
        "min_edge_state_to_keep": { "type": float, "description": "Edge Logic: Minimum edge state required to be kept (non-zero).", "min": 0.0, "max": 0.5, "default": 0.02, "parameter_group": "Edge Logic"},

        # === Visualization ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (0 or 1).", "default": False, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states.", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization: Nodes"},
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "prism", "parameter_group": "Visualization: Edges",
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # === Other ===
        "node_history_depth": { 'type': int, 'description': 'Number of previous states to store.', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Edge Feedback Life"
        metadata.description = "Node birth depends on avg neighbor degree (prev step). Node survival depends on incoming edge weight sum (prev step). Continuous edges strengthen towards active neighbors."
        metadata.category = "Life-Like"
        metadata.tags = ["Life", "Edges", "Feedback", "Weighted", "Connectivity", "Degree", "Experimental"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Edge Feedback Life"
        self._params = {}
        self.requires_post_edge_state_update = False
        self.needs_neighbor_degrees = True # Needed for birth logic
        self.needs_neighbor_active_counts = False # Not needed

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute new state based on avg neighbor degree (birth) or edge sum (survival)."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        birth_min_avg_deg = self.get_param('birth_min_avg_neighbor_degree', 1.5, neighborhood=neighborhood)
        birth_max_avg_deg = self.get_param('birth_max_avg_neighbor_degree', 4.0, neighborhood=neighborhood)
        survival_min_sum = self.get_param('survival_min_edge_sum', 0.8, neighborhood=neighborhood)
        survival_max_sum = self.get_param('survival_max_edge_sum', 3.5, neighborhood=neighborhood)

        new_state = 0.0 # Default to death
        decision_reason = "Default (Death/Inactive)"

        if neighborhood.node_state <= 0: # --- Birth Condition ---
            # Calculate avg degree of *active* neighbors using pre-calculated degrees
            sum_neighbor_degree = 0
            active_neighbor_count_for_avg = 0
            avg_neighbor_degree = 0.0
            if neighborhood.neighbor_degrees is not None:
                for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
                     if neighbor_idx >= 0 and neighbor_state > 0: # If neighbor was active
                          sum_neighbor_degree += neighborhood.neighbor_degrees.get(neighbor_idx, 0)
                          active_neighbor_count_for_avg += 1
                if active_neighbor_count_for_avg > 0:
                    avg_neighbor_degree = sum_neighbor_degree / active_neighbor_count_for_avg
            else: logger.warning(f"Node {node_idx}: neighbor_degrees missing for birth check!")

            if detailed_logging_enabled: logger.detail(f"    Birth Check: Avg Active Neighbor Degree (Prev Step) = {avg_neighbor_degree:.2f}") # type: ignore [attr-defined]

            if birth_min_avg_deg <= avg_neighbor_degree <= birth_max_avg_deg:
                new_state = 1.0
                decision_reason = f"Birth (AvgNeighDeg={avg_neighbor_degree:.2f} in [{birth_min_avg_deg:.2f}-{birth_max_avg_deg:.2f}])"
            else:
                decision_reason = f"Remain Dead (AvgNeighDeg={avg_neighbor_degree:.2f} OUT of [{birth_min_avg_deg:.2f}-{birth_max_avg_deg:.2f}])"

        else: # --- Survival Condition ---
            # Sum edge strengths ONLY to active neighbors
            sum_edge_strength = sum(
                neighborhood.neighbor_edge_states.get(idx, 0.0)
                for idx, state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states)
                if idx >= 0 and state > 0 # Only edges to ACTIVE neighbors
            )
            if detailed_logging_enabled: logger.detail(f"    Survival Check: Edge Strength Sum (to active) = {sum_edge_strength:.4f}") # type: ignore [attr-defined]

            if survival_min_sum <= sum_edge_strength <= survival_max_sum:
                new_state = 1.0
                decision_reason = f"Survival (EdgeSum={sum_edge_strength:.4f} in [{survival_min_sum:.4f}-{survival_max_sum:.4f}])"
            else:
                decision_reason = f"Death (EdgeSum={sum_edge_strength:.4f} OUT of [{survival_min_sum:.4f}-{survival_max_sum:.4f}])"

        if detailed_logging_enabled:
            logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    New State: {new_state:.1f}") # type: ignore [attr-defined]

        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Edges strengthen towards active neighbors, decay otherwise."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        edge_change_rate = self.get_param('edge_change_rate', 0.3, neighborhood=neighborhood)
        edge_decay_rate = self.get_param('edge_decay_rate', 0.02, neighborhood=neighborhood)
        random_edge_prob = self.get_param('random_edge_change_prob', 0.01, neighborhood=neighborhood)
        random_edge_amount = self.get_param('random_edge_change_amount', 0.1, neighborhood=neighborhood)
        min_keep = self.get_param('min_edge_state_to_keep', 0.02, neighborhood=neighborhood)

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            edge = (node_idx, neighbor_idx) # Canonical order
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0)

            # Target state based on neighbor's *previous* state
            neighbor_was_active = neighbor_state > 0.5 # Use 0.5 threshold for binary check
            target_edge_state = 1.0 if neighbor_was_active else 0.0

            if detailed_logging_enabled:
                logger.detail(f"  Neighbor {neighbor_idx}: Prev State={neighbor_state:.1f} -> Target={target_edge_state:.1f}") # type: ignore [attr-defined]

            # Move edge state towards target state by the change rate
            new_edge_state = current_edge_state + (target_edge_state - current_edge_state) * edge_change_rate

            # Apply constant decay
            new_edge_state -= edge_decay_rate

            # Apply random change
            if np.random.random() < random_edge_prob:
                change = (np.random.random() * 2 - 1) * random_edge_amount
                new_edge_state += change

            clipped_new_edge_state = float(np.clip(new_edge_state, 0.0, 1.0))

            if detailed_logging_enabled:
                logger.detail(f"    Edge ({edge[0]},{edge[1]}): Curr={current_edge_state:.4f}, Target={target_edge_state:.4f}, Rate={edge_change_rate:.2f}, Decay={edge_decay_rate:.3f} -> New(clipped)={clipped_new_edge_state:.4f}") # type: ignore [attr-defined]

            # Only propose edges with a state greater than the minimum threshold
            if clipped_new_edge_state >= min_keep:
                 new_edges[edge] = clipped_new_edge_state

        # if detailed_logging_enabled: # Reduce noise
            # logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges
    
class NetworkTopologyLife(Rule):
    """
    Rule where node state represents its degree (connection count).
    Birth/Survival eligibility depends on neighbor counts and degree criteria from the previous step.
    Edges (binary 0/1) change based on comparing the degrees of connected nodes from the previous step,
    aiming for more dynamic network structures. Requires post-update step.
    (Round 35: Added edge visualization parameters)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'min_edge_weight', 'max_edge_weight', # Edges are binary
        'connect_probability', # Not used by core logic, only init
        # 'use_state_coloring_edges', # Keep
        # 'edge_colormap', # Keep
        # 'edge_color_norm_vmin', # Keep
        # 'edge_color_norm_vmax', # Keep
        'color_edges_by_neighbor_degree', # Edges are binary, state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edges are binary, state not based on this
        'node_history_depth'
    }
    # --- END MODIFIED ---

    node_state_type: ClassVar[StateType] = StateType.INTEGER
    edge_state_type: ClassVar[StateType] = StateType.BINARY # Edges are 0/1
    min_node_state: ClassVar[float] = 0.0 # Degree cannot be negative
    max_node_state: ClassVar[float] = 26.0 # Theoretical max degree (Moore 3D)
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0
    produces_binary_edges: ClassVar[bool] = True

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'bounded', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern (Nodes initialized to 0).", "default": "Random", "allowed_values": ['Random'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Approx initial density of nodes with connections.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},

        # --- Node Birth Eligibility Logic ---
        "birth_active_neighbor_counts": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Node Birth: List of exact active neighbor counts (prev step) for birth eligibility.", "parameter_group": "Node Birth Eligibility"},
        "birth_avg_neighbor_degree_ranges": { 'type': list, 'element_type': tuple, 'default': [[0.0, 5.0]], 'description': "Node Birth: List of (min_avg, max_avg) degree ranges for active neighbors (prev step) for birth eligibility. Empty list skips check.", "parameter_group": "Node Birth Eligibility"},
        # --- Node Survival Eligibility Logic ---
        "survival_self_degree_counts": { 'type': list, 'element_type': int, 'default': [3, 4, 5], 'description': "Node Survival: List of exact own degrees (prev step state) for survival eligibility.", "parameter_group": "Node Survival Eligibility"},
        "survival_active_neighbor_counts": { 'type': list, 'element_type': int, 'default': [1, 2, 3, 4], 'description': "Node Survival: List of exact active neighbor counts (prev step) for survival eligibility.", "parameter_group": "Node Survival Eligibility"},
        "survival_avg_neighbor_degree_ranges": { 'type': list, 'element_type': tuple, 'default': [], 'description': "Node Survival: List of (min_avg, max_avg) degree ranges for active neighbors (prev step) for survival eligibility. Empty list skips check.", "parameter_group": "Node Survival Eligibility"},

        # --- Edge Formation Logic (Based on Prev Step Degrees) ---
        "connect_degree_pairs": { 'type': list, 'element_type': tuple, 'default': [[1, 3], [1, 4], [2, 4], [2, 5], [3, 5], [3, 6], [4, 6], [4, 1], [3, 1], [4, 2], [5, 2], [5, 3], [6, 3], [6, 4]], 'description': "Edge Formation: List of exact (self_deg, neighbor_deg) pairs (prev step) to trigger connection if no edge exists. Order matters if asymmetric.", "parameter_group": "Edge Formation"},
        "connect_degree_ranges": { 'type': list, 'element_type': tuple, 'default': [], 'description': "Edge Formation: List of ((min_s, max_s), (min_n, max_n)) range pairs (prev step) to trigger connection if no edge exists.", "parameter_group": "Edge Formation"},

        # --- Edge Destruction Logic (Based on Prev Step Degrees) ---
        "disconnect_degree_pairs": { 'type': list, 'element_type': tuple, 'default': [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], 'description': "Edge Destruction: List of exact (self_deg, neighbor_deg) pairs (prev step) to trigger disconnection if edge exists. Order matters if asymmetric.", "parameter_group": "Edge Destruction"},
        "disconnect_degree_ranges": { 'type': list, 'element_type': tuple, 'default': [[[7, 26], [7, 26]]], 'description': "Edge Destruction: List of ((min_s, max_s), (min_n, max_n)) range pairs (prev step) to trigger disconnection if edge exists.", "parameter_group": "Edge Destruction"},

        # --- Edge Randomness ---
        "random_edge_flip_prob": { "type": float, "description": "Edge Randomness: Probability of randomly flipping a potential edge's state.", "min": 0.0, "max": 0.1, "default": 0.002, "parameter_group": "Edge Randomness"},

        # --- Final State Death Conditions ---
        "final_death_degree_counts": { 'type': list, 'element_type': int, 'default': [0], 'description': "Final State: Node state becomes 0 if its *final* calculated degree is in this list.", "parameter_group": "Final State Death"},

        # --- Visualization ---
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (degree).", "default": True, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states (degrees).", "default": "coolwarm", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 8.0, "parameter_group": "Visualization: Nodes"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on state (0 or 1).",
            "default": False, "parameter_group": "Visualization: Edges" # Default False for binary
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "binary", "parameter_group": "Visualization: Edges", # Default binary
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # --- Other ---
        "tiebreaker_type": { "type": str, "description": "Tiebreaker for edge conflicts.", "allowed_values": ["RANDOM", "AGREEMENT"], "default": "RANDOM", "parameter_group": "Tiebreaker"},
        "node_history_depth": { 'type': int, 'description': 'Grid Setting: History depth (not used by rule).', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Network Topology Life"
        metadata.description = "Node state = degree. Birth/Survival eligibility depends on neighbor counts, own degree, avg neighbor degree (prev step). Edges change based on relative degrees (prev step), favoring connections between nodes with different degrees and pruning overly similar/connected nodes. Aims for dynamic topological structures." # Updated description
        metadata.category = "Connectivity-Based"
        metadata.tags = ["Connectivity", "Degree", "Discrete State", "Edges", "Topology", "Dynamic"] # Added Dynamic
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Network Topology Life"
        self._params = {}
        self.requires_post_edge_state_update = True # Requires final degree calculation
        # --- Set flags to request data ---
        self.needs_neighbor_degrees = True
        self.needs_neighbor_active_counts = True # Need for neighbor count checks
        # ---
        self.skip_standard_tiebreakers = True # Uses its own edge logic

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _count_active_neighbors(self, neighbor_states: np.ndarray) -> int:
        """Counts neighbors whose state (degree) > 0."""
        activity_threshold = 1e-6
        return int(np.sum(neighbor_states > activity_threshold))

    def _check_ranges(self, value: float, ranges: List[Tuple[float, float]]) -> bool:
        """Checks if a value falls within any of the specified (min, max) ranges."""
        if not ranges: return True # Pass if no ranges specified
        return any(min_val <= value <= max_val for min_val, max_val in ranges)

    def _check_degree_pairs(self, self_deg: int, neighbor_deg: int,
                            pairs: List[Tuple[int, int]],
                            ranges: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> bool:
        """Checks if a degree pair matches any exact pair or range."""
        # Check exact pairs (order matters now for asymmetric rules)
        if [self_deg, neighbor_deg] in pairs: # Check list of lists
            return True
        # Check ranges
        for (min_s, max_s), (min_n, max_n) in ranges:
            if (min_s <= self_deg <= max_s) and (min_n <= neighbor_deg <= max_n):
                return True
        return False

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Calculates the node's ELIGIBILITY proxy state (0 or 1) based on previous step data."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        birth_neighbor_counts = self.get_param('birth_active_neighbor_counts', [3], neighborhood=neighborhood)
        birth_avg_deg_ranges = self.get_param('birth_avg_neighbor_degree_ranges', [[1.0, 4.0]], neighborhood=neighborhood) # Using new default
        survival_self_degree_counts = self.get_param('survival_self_degree_counts', [3, 4, 5], neighborhood=neighborhood) # Using new default
        survival_neighbor_counts = self.get_param('survival_active_neighbor_counts', [1, 2, 3, 4, 5, 6], neighborhood=neighborhood) # Using new default
        survival_avg_deg_ranges = self.get_param('survival_avg_neighbor_degree_ranges', [], neighborhood=neighborhood)

        # Calculate metrics from previous step
        num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
        current_degree = int(neighborhood.node_state) # Degree from prev step
        sum_neighbor_degree = 0
        active_neighbor_count_for_avg = 0
        avg_neighbor_degree = 0.0
        if neighborhood.neighbor_degrees is not None:
            for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
                 if neighbor_idx >= 0 and neighbor_state > 0: # If neighbor was active
                      sum_neighbor_degree += neighborhood.neighbor_degrees.get(neighbor_idx, 0)
                      active_neighbor_count_for_avg += 1
            if active_neighbor_count_for_avg > 0:
                avg_neighbor_degree = sum_neighbor_degree / active_neighbor_count_for_avg
        else: logger.warning(f"Node {node_idx}: neighbor_degrees missing!")

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state (Eligibility Proxy) ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State (Degree): {current_degree}") # type: ignore [attr-defined]
            logger.detail(f"    Prev Active Neighbors: {num_active_neighbors}") # type: ignore [attr-defined]
            logger.detail(f"    Prev Avg Active Neighbor Degree: {avg_neighbor_degree:.2f}") # type: ignore [attr-defined]

        eligibility_proxy = 0.0
        decision_reason = "Default (Ineligible)"

        if neighborhood.node_state <= 0: # --- Check Birth Eligibility ---
            passes_neighbor_count = (num_active_neighbors in birth_neighbor_counts)
            passes_neighbor_degree = self._check_ranges(avg_neighbor_degree, birth_avg_deg_ranges)
            if passes_neighbor_count and passes_neighbor_degree:
                eligibility_proxy = 1.0
                decision_reason = f"Eligible for Birth (NeighborsOk={passes_neighbor_count}, AvgDegreeOk={passes_neighbor_degree})"
            else:
                decision_reason = f"Ineligible (Birth conditions not met: NeighborsOk={passes_neighbor_count}, AvgDegreeOk={passes_neighbor_degree})"

        else: # --- Check Survival Eligibility ---
            passes_self_degree = (current_degree in survival_self_degree_counts)
            survival_active_neighbor_counts = self.get_param('survival_active_neighbor_counts', [1, 2, 3, 4, 5, 6], neighborhood=neighborhood)
            passes_neighbor_count = (num_active_neighbors in survival_active_neighbor_counts)
            passes_neighbor_degree = self._check_ranges(avg_neighbor_degree, survival_avg_deg_ranges)

            if passes_self_degree and passes_neighbor_count and passes_neighbor_degree:
                eligibility_proxy = 1.0
                decision_reason = f"Eligible for Survival (DegreeOk={passes_self_degree}, NeighborsOk={passes_neighbor_count}, AvgDegreeOk={passes_neighbor_degree})"
            else:
                decision_reason = f"Ineligible (Death: Survival conditions not met: DegreeOk={passes_self_degree}, NeighborsOk={passes_neighbor_count}, AvgDegreeOk={passes_neighbor_degree})"

        if detailed_logging_enabled:
            logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    Returning Eligibility Proxy: {eligibility_proxy:.1f}") # type: ignore [attr-defined]

        return eligibility_proxy

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Determine edge existence based on relative degrees from previous step AND mutual eligibility."""

        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        connect_pairs = self.get_param('connect_degree_pairs', [[1, 3], [1, 4], [2, 4], [2, 5], [3, 5], [3, 6], [4, 6], [4, 1], [3, 1], [4, 2], [5, 2], [5, 3], [6, 3], [6, 4]], neighborhood=neighborhood)
        connect_ranges = self.get_param('connect_degree_ranges', [], neighborhood=neighborhood)
        disconnect_pairs = self.get_param('disconnect_degree_pairs', [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8]], neighborhood=neighborhood)
        disconnect_ranges = self.get_param('disconnect_degree_ranges', [[[7, 26], [7, 26]]], neighborhood=neighborhood)
        random_flip_prob = self.get_param('random_edge_flip_prob', 0.001, neighborhood=neighborhood)

        # Convert list pairs/ranges from params to tuples
        connect_pairs_tuples = [tuple(p) for p in connect_pairs]
        connect_ranges_tuples = [(tuple(r1), tuple(r2)) for r1, r2 in connect_ranges]
        disconnect_pairs_tuples = [tuple(p) for p in disconnect_pairs]
        disconnect_ranges_tuples = [(tuple(r1), tuple(r2)) for r1, r2 in disconnect_ranges]

        # Get Eligibility Proxies from rule_params
        eligibility_proxies_flat = neighborhood.rule_params.get('_eligibility_proxies')
        if eligibility_proxies_flat is None:
            logger.error(f"Node {node_idx}: Eligibility proxies missing in rule_params!")
            return new_edges

        # Determine eligibility of the current node from proxy
        self_is_eligible = False
        if 0 <= node_idx < eligibility_proxies_flat.size:
            self_is_eligible = eligibility_proxies_flat[node_idx] > 0.5
        else: logger.warning(f"Node {node_idx}: Index out of bounds for eligibility proxies.")

        # Node's degree from previous step (is its state)
        self_prev_degree = int(neighborhood.node_state)

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Prev Degree: {self_prev_degree}, Eligible Next (from proxy): {self_is_eligible}") # type: ignore [attr-defined]

        for neighbor_idx, neighbor_prev_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            # Get Neighbor Eligibility from Proxy Array
            neighbor_is_eligible = False
            if 0 <= neighbor_idx < eligibility_proxies_flat.size:
                neighbor_is_eligible = eligibility_proxies_flat[neighbor_idx] > 0.5
            else: logger.warning(f"Node {node_idx}: Neighbor index {neighbor_idx} out of bounds for eligibility proxies."); continue

            # Get neighbor's degree from previous step
            neighbor_prev_degree = 0
            if neighborhood.neighbor_degrees is not None:
                 neighbor_prev_degree = neighborhood.neighbor_degrees.get(neighbor_idx, 0)
            else: logger.warning(f"Node {node_idx}: neighbor_degrees missing for neighbor {neighbor_idx}")

            edge = (node_idx, neighbor_idx) # Canonical order
            has_current_edge = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0) > 1e-6

            propose_edge = has_current_edge # Default: maintain
            decision_reason = "Default (Maintain)"

            # Apply Edge Logic ONLY if BOTH nodes are eligible
            if self_is_eligible and neighbor_is_eligible:
                # --- MODIFIED: Correct call to _check_degree_pairs ---
                should_connect = self._check_degree_pairs(self_prev_degree, neighbor_prev_degree, connect_pairs_tuples, connect_ranges_tuples)
                should_disconnect = self._check_degree_pairs(self_prev_degree, neighbor_prev_degree, disconnect_pairs_tuples, disconnect_ranges_tuples)
                # --- END MODIFIED ---

                if not has_current_edge and should_connect and not should_disconnect:
                    propose_edge = True
                    decision_reason = f"Form edge (Connect rule met: Self={self_prev_degree}, Neigh={neighbor_prev_degree})"
                elif has_current_edge and should_disconnect and not should_connect:
                    propose_edge = False
                    decision_reason = f"Break edge (Disconnect rule met: Self={self_prev_degree}, Neigh={neighbor_prev_degree})"
                elif should_connect and should_disconnect: # Conflict
                    propose_edge = random.choice([True, False]) # Randomly resolve conflict
                    decision_reason = f"Conflict (Connect AND Disconnect rules met) -> Randomly chose {propose_edge}"
                elif has_current_edge:
                     decision_reason = "Maintain existing edge (No disconnect rule, both eligible)"
                else: # Not has_current_edge and not should_connect
                     decision_reason = "Do not form edge (No connect rule, both eligible)"

            elif has_current_edge: # If edge exists but one node is ineligible, break it
                propose_edge = False
                decision_reason = "Break edge (One or both nodes ineligible next step)"
            # else: # No edge exists and one node ineligible -> remains no edge

            # Apply random flip
            random_flip_applied = False
            if np.random.random() < random_flip_prob:
                propose_edge = not propose_edge
                random_flip_applied = True
                decision_reason += " + Random Flip"

            if detailed_logging_enabled:
                logger.detail(f"  Neighbor {neighbor_idx}: PrevDeg={neighbor_prev_degree}, EligibleNext(Proxy)={neighbor_is_eligible}. Decision: {decision_reason}. RandomFlip={random_flip_applied}. Final Propose={propose_edge}") # type: ignore [attr-defined]

            # Add to proposed edges if it should exist
            if propose_edge:
                new_edges[edge] = 1.0 # Binary edges

        return new_edges

    def _compute_final_state(
            self, node_idx: int,
            current_proxy_state: float,
            final_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
            dimensions: Tuple[int,...],
            # --- ADDED: Missing base class arguments ---
            previous_node_states: npt.NDArray[np.float64],
            previous_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
            previous_edge_states: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float],
            previous_node_degrees: Optional[npt.NDArray[np.int32]],
            previous_active_neighbors: Optional[npt.NDArray[np.int32]],
            eligibility_proxies: Optional[np.ndarray] = None,
            detailed_logging_enabled: bool = False
            ) -> float:
        """Calculates the final state (degree) based on eligibility and final edge count, applying death list."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_final_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Proxy State (Eligibility): {current_proxy_state:.1f}") # type: ignore [attr-defined]

        # If proxy state indicates ineligible (0.0), final state is 0
        if current_proxy_state < 0.5:
             if detailed_logging_enabled: logger.detail(f"    Node was ineligible based on prev step conditions, final state is 0.") # type: ignore [attr-defined]
             return 0.0

        # If eligible (proxy was 1.0), calculate final degree based on the FINAL edges for this step
        node_coords = tuple(_unravel_index(node_idx, dimensions))
        final_degree = sum(1 for edge in final_edges if node_coords in edge)

        if detailed_logging_enabled:
            logger.detail(f"    Node was eligible. Final Calculated Degree: {final_degree}") # type: ignore [attr-defined]

        # Apply death list based on the FINAL degree
        final_death_degrees = self.get_param('final_death_degree_counts', [0]) # Using new default

        final_state = 0.0 # Default to death
        decision_reason = "Default (Death)"
        if final_degree in final_death_degrees:
            decision_reason = f"Final Death (Final Degree={final_degree} in death list {final_death_degrees})"
        else:
            # Survived final death checks, state is the final degree
            final_state = float(final_degree)
            decision_reason = f"Final Survival (Final Degree={final_degree} not in death list)"

        if detailed_logging_enabled:
            logger.detail(f"    Final Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    Final State: {final_state:.1f}") # type: ignore [attr-defined]
        return final_state

class TopologicalRule(Rule):
    """
    Rule where node state represents its degree (connection count). Birth/Survival eligibility
    depends on neighbor counts, own degree, and avg neighbor degree lists/ranges (from prev step).
    Edges (binary 0/1) change based on relative degree pair lists/ranges (from prev step),
    aiming for dynamic topological structures. Requires post-update step.
    (Round 35: Added edge visualization parameters)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'min_edge_weight', 'max_edge_weight', # Edges are binary
        'connect_probability', # Not used by core logic, only init
        # 'use_state_coloring_edges', # Keep
        # 'edge_colormap', # Keep
        # 'edge_color_norm_vmin', # Keep
        # 'edge_color_norm_vmax', # Keep
        'color_edges_by_neighbor_degree', # Edges are binary, state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edges are binary, state not based on this
        'node_history_depth'
    }
    # --- END MODIFIED ---

    produces_binary_edges: ClassVar[bool] = True

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'bounded', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern (Nodes initialized to 0).", "default": "Random", "allowed_values": ['Random'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Approx initial density of nodes with connections.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},

        # --- Node Birth Eligibility Logic ---
        "birth_active_neighbor_counts": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Node Birth: List of exact active neighbor counts (prev step) for birth eligibility.", "parameter_group": "Node Birth Eligibility"},
        "birth_avg_neighbor_degree_ranges": { 'type': list, 'element_type': tuple, 'default': [[0.0, 5.0]], 'description': "Node Birth: List of (min_avg, max_avg) degree ranges for active neighbors (prev step) for birth eligibility. Empty list skips check.", "parameter_group": "Node Birth Eligibility"},
        # --- Node Survival Eligibility Logic ---
        "survival_self_degree_counts": { 'type': list, 'element_type': int, 'default': [3, 4, 5], 'description': "Node Survival: List of exact own degrees (prev step state) for survival eligibility.", "parameter_group": "Node Survival Eligibility"},
        "survival_active_neighbor_counts": { 'type': list, 'element_type': int, 'default': [1, 2, 3, 4], 'description': "Node Survival: List of exact active neighbor counts (prev step) for survival eligibility.", "parameter_group": "Node Survival Eligibility"},
        "survival_avg_neighbor_degree_ranges": { 'type': list, 'element_type': tuple, 'default': [], 'description': "Node Survival: List of (min_avg, max_avg) degree ranges for active neighbors (prev step) for survival eligibility. Empty list skips check.", "parameter_group": "Node Survival Eligibility"},

        # --- Edge Formation Logic (Based on Prev Step Degrees) ---
        "connect_degree_pairs": { 'type': list, 'element_type': tuple, 'default': [[1, 3], [1, 4], [2, 4], [2, 5], [3, 5], [3, 6], [4, 6], [4, 1], [3, 1], [4, 2], [5, 2], [5, 3], [6, 3], [6, 4]], 'description': "Edge Formation: List of exact (self_deg, neighbor_deg) pairs (prev step) to trigger connection if no edge exists. Order matters if asymmetric.", "parameter_group": "Edge Formation"},
        "connect_degree_ranges": { 'type': list, 'element_type': tuple, 'default': [], 'description': "Edge Formation: List of ((min_s, max_s), (min_n, max_n)) range pairs (prev step) to trigger connection if no edge exists.", "parameter_group": "Edge Formation"},

        # --- Edge Destruction Logic (Based on Prev Step Degrees) ---
        "disconnect_degree_pairs": { 'type': list, 'element_type': tuple, 'default': [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], 'description': "Edge Destruction: List of exact (self_deg, neighbor_deg) pairs (prev step) to trigger disconnection if edge exists. Order matters if asymmetric.", "parameter_group": "Edge Destruction"},
        "disconnect_degree_ranges": { 'type': list, 'element_type': tuple, 'default': [[[7, 26], [7, 26]]], 'description': "Edge Destruction: List of ((min_s, max_s), (min_n, max_n)) range pairs (prev step) to trigger disconnection if edge exists.", "parameter_group": "Edge Destruction"},

        # --- Edge Randomness ---
        "random_edge_flip_prob": { "type": float, "description": "Edge Randomness: Probability of randomly flipping a potential edge's state.", "min": 0.0, "max": 0.1, "default": 0.002, "parameter_group": "Edge Randomness"},

        # --- Final State Death Conditions ---
        "final_death_degree_counts": { 'type': list, 'element_type': int, 'default': [0], 'description': "Final State: Node state becomes 0 if its *final* calculated degree is in this list.", "parameter_group": "Final State Death"},

        # --- Visualization ---
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (degree).", "default": True, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states (degrees).", "default": "coolwarm", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization.", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 8.0, "parameter_group": "Visualization: Nodes"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on state (0 or 1).",
            "default": False, "parameter_group": "Visualization: Edges" # Default False for binary
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "binary", "parameter_group": "Visualization: Edges", # Default binary
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # --- Other ---
        "tiebreaker_type": { "type": str, "description": "Tiebreaker for edge conflicts.", "allowed_values": ["RANDOM", "AGREEMENT"], "default": "RANDOM", "parameter_group": "Tiebreaker"},
        "node_history_depth": { 'type': int, 'description': 'Grid Setting: History depth (not used by rule).', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Topological Rule"
        metadata.description = "Connects rules based on topological properties of local neighborhodd" # TODO: Write a better description
        metadata.category = "Connectivity-Based"
        metadata.tags = ["Connectivity", "Degree", "Discrete State", "Edges", "Topology", "Dynamic"] 
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Topological Rule"
        self._params = {}
        self.requires_post_edge_state_update = True
        self.needs_neighbor_degrees = True
        self.needs_neighbor_active_counts = True
        self.skip_standard_tiebreakers = True
        # --- ADDED: Set coloring flags ---
        self._params['use_state_coloring'] = True # Use the stored state (degree)
        self._params['color_nodes_by_degree'] = False
        self._params['color_nodes_by_active_neighbors'] = False
        # Set appropriate vmax/cmap for coloring
        self._params['node_color_norm_vmax'] = 8.0 # Default for Moore 2D
        self._params['node_colormap'] = 'coolwarm' # Default cmap
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.
    
    def _count_active_neighbors(self, neighbor_states: np.ndarray) -> int:
        """Counts neighbors whose state (degree) > 0."""
        activity_threshold = 1e-6
        return int(np.sum(neighbor_states > activity_threshold))

    def _check_ranges(self, value: float, ranges: List[Tuple[float, float]]) -> bool:
        """Checks if a value falls within any of the specified (min, max) ranges."""
        if not ranges: return True # Pass if no ranges specified
        return any(min_val <= value <= max_val for min_val, max_val in ranges)

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Calculates the node's ELIGIBILITY proxy state (0 or 1) based on previous step data."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        birth_neighbor_counts = self.get_param('birth_active_neighbor_counts', [2, 3], neighborhood=neighborhood) # Using new default
        birth_avg_deg_ranges = self.get_param('birth_avg_neighbor_degree_ranges', [[0.0, 5.0]], neighborhood=neighborhood) # Using new default
        survival_self_degree_counts = self.get_param('survival_self_degree_counts', [3, 4, 5], neighborhood=neighborhood) # Using new default
        survival_neighbor_counts = self.get_param('survival_active_neighbor_counts', [1, 2, 3, 4], neighborhood=neighborhood) # Using new default
        survival_avg_deg_ranges = self.get_param('survival_avg_neighbor_degree_ranges', [], neighborhood=neighborhood) # Using new default

        # Calculate metrics from previous step
        num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
        current_degree = int(neighborhood.node_state) # Degree from prev step
        sum_neighbor_degree = 0
        active_neighbor_count_for_avg = 0
        avg_neighbor_degree = 0.0
        if neighborhood.neighbor_degrees is not None:
            for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
                 if neighbor_idx >= 0 and neighbor_state > 0: # If neighbor was active
                      sum_neighbor_degree += neighborhood.neighbor_degrees.get(neighbor_idx, 0)
                      active_neighbor_count_for_avg += 1
            if active_neighbor_count_for_avg > 0:
                avg_neighbor_degree = sum_neighbor_degree / active_neighbor_count_for_avg
        else: logger.warning(f"Node {node_idx}: neighbor_degrees missing!")

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state (Eligibility Proxy) ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State (Degree): {current_degree}") # type: ignore [attr-defined]
            logger.detail(f"    Prev Active Neighbors: {num_active_neighbors}") # type: ignore [attr-defined]
            logger.detail(f"    Prev Avg Active Neighbor Degree: {avg_neighbor_degree:.2f}") # type: ignore [attr-defined]

        eligibility_proxy = 0.0
        decision_reason = "Default (Ineligible)"

        if neighborhood.node_state <= 0: # --- Check Birth Eligibility ---
            passes_neighbor_count = (num_active_neighbors in birth_neighbor_counts)
            passes_neighbor_degree = self._check_ranges(avg_neighbor_degree, birth_avg_deg_ranges)
            if passes_neighbor_count and passes_neighbor_degree:
                eligibility_proxy = 1.0
                decision_reason = f"Eligible for Birth (NeighborsOk={passes_neighbor_count}, AvgDegreeOk={passes_neighbor_degree})"
            else:
                decision_reason = f"Ineligible (Birth conditions not met: NeighborsOk={passes_neighbor_count}, AvgDegreeOk={passes_neighbor_degree})"

        else: # --- Check Survival Eligibility ---
            passes_self_degree = (current_degree in survival_self_degree_counts)
            survival_active_neighbor_counts = self.get_param('survival_active_neighbor_counts', [2, 3], neighborhood=neighborhood)
            passes_neighbor_count = (num_active_neighbors in survival_active_neighbor_counts)
            passes_neighbor_degree = self._check_ranges(avg_neighbor_degree, survival_avg_deg_ranges)

            if passes_self_degree and passes_neighbor_count and passes_neighbor_degree:
                eligibility_proxy = 1.0
                decision_reason = f"Eligible for Survival (DegreeOk={passes_self_degree}, NeighborsOk={passes_neighbor_count}, AvgDegreeOk={passes_neighbor_degree})"
            else:
                decision_reason = f"Ineligible (Death: Survival conditions not met: DegreeOk={passes_self_degree}, NeighborsOk={passes_neighbor_count}, AvgDegreeOk={passes_neighbor_degree})"

        if detailed_logging_enabled:
            logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    Returning Eligibility Proxy: {eligibility_proxy:.1f}") # type: ignore [attr-defined]

        return eligibility_proxy

    def _check_degree_pairs(self, self_deg: int, neighbor_deg: int,
                            pairs: List[Tuple[int, int]],
                            ranges: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> bool:
        """Checks if a degree pair matches any exact pair or range."""
        # Check exact pairs (order matters for asymmetric rules)
        # Ensure pairs contains tuples, not lists, for comparison if needed
        pairs_as_tuples = [tuple(p) for p in pairs]
        if (self_deg, neighbor_deg) in pairs_as_tuples: # Check tuple against list of tuples
            return True
        # Check ranges
        for range_pair in ranges:
             # Ensure inner elements are tuples before unpacking
             if isinstance(range_pair, (list, tuple)) and len(range_pair) == 2:
                 range1, range2 = range_pair
                 if isinstance(range1, (list, tuple)) and len(range1) == 2 and \
                    isinstance(range2, (list, tuple)) and len(range2) == 2:
                     min_s, max_s = range1
                     min_n, max_n = range2
                     if (min_s <= self_deg <= max_s) and (min_n <= neighbor_deg <= max_n):
                         return True
                 else:
                      # Log or handle error for invalid inner range format
                      logger = logging.getLogger(__name__)
                      logger.warning(f"Skipping invalid range format inside ranges list: {range_pair}")
             else:
                 # Log or handle error for invalid outer range format
                 logger = logging.getLogger(__name__)
                 logger.warning(f"Skipping invalid range pair format in ranges list: {range_pair}")

        return False

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Determine edge existence based on relative degrees from previous step AND mutual eligibility."""

        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        connect_pairs = self.get_param('connect_degree_pairs', [[1, 3], [1, 4], [2, 4], [2, 5], [3, 5], [3, 6], [4, 6], [4, 1], [3, 1], [4, 2], [5, 2], [5, 3], [6, 3], [6, 4]], neighborhood=neighborhood)
        connect_ranges = self.get_param('connect_degree_ranges', [], neighborhood=neighborhood)
        disconnect_pairs = self.get_param('disconnect_degree_pairs', [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], neighborhood=neighborhood)
        disconnect_ranges = self.get_param('disconnect_degree_ranges', [[[7, 26], [7, 26]]], neighborhood=neighborhood)
        random_flip_prob = self.get_param('random_edge_flip_prob', 0.002, neighborhood=neighborhood)

        # Convert list pairs/ranges from params to tuples
        connect_pairs_tuples = [tuple(p) for p in connect_pairs]
        connect_ranges_tuples = [(tuple(r1), tuple(r2)) for r1, r2 in connect_ranges]
        disconnect_pairs_tuples = [tuple(p) for p in disconnect_pairs]
        disconnect_ranges_tuples = [(tuple(r1), tuple(r2)) for r1, r2 in disconnect_ranges]

        # Get Eligibility Proxies from rule_params
        eligibility_proxies_flat = neighborhood.rule_params.get('_eligibility_proxies')
        if eligibility_proxies_flat is None:
            logger.error(f"Node {node_idx}: Eligibility proxies missing in rule_params!")
            return new_edges

        # Determine eligibility of the current node from proxy
        self_is_eligible = False
        if 0 <= node_idx < eligibility_proxies_flat.size:
            self_is_eligible = eligibility_proxies_flat[node_idx] > 0.5
        else: logger.warning(f"Node {node_idx}: Index out of bounds for eligibility proxies.")

        # Node's degree from previous step (is its state)
        self_prev_degree = int(neighborhood.node_state)

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Prev Degree: {self_prev_degree}, Eligible Next (from proxy): {self_is_eligible}") # type: ignore [attr-defined]

        for neighbor_idx, neighbor_prev_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            # Get Neighbor Eligibility from Proxy Array
            neighbor_is_eligible = False
            if 0 <= neighbor_idx < eligibility_proxies_flat.size:
                neighbor_is_eligible = eligibility_proxies_flat[neighbor_idx] > 0.5
            else: logger.warning(f"Node {node_idx}: Neighbor index {neighbor_idx} out of bounds for eligibility proxies."); continue

            # Get neighbor's degree from previous step
            neighbor_prev_degree = 0
            if neighborhood.neighbor_degrees is not None:
                 neighbor_prev_degree = neighborhood.neighbor_degrees.get(neighbor_idx, 0)
            else: logger.warning(f"Node {node_idx}: neighbor_degrees missing for neighbor {neighbor_idx}")

            edge = (node_idx, neighbor_idx) # Canonical order
            has_current_edge = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0) > 1e-6

            propose_edge = has_current_edge # Default: maintain
            decision_reason = "Default (Maintain)"

            # Apply Edge Logic ONLY if BOTH nodes are eligible
            if self_is_eligible and neighbor_is_eligible:
                # --- MODIFIED: Correct call to _check_degree_pairs ---
                should_connect = self._check_degree_pairs(self_prev_degree, neighbor_prev_degree, connect_pairs_tuples, connect_ranges_tuples)
                should_disconnect = self._check_degree_pairs(self_prev_degree, neighbor_prev_degree, disconnect_pairs_tuples, disconnect_ranges_tuples)
                # --- END MODIFIED ---

                if not has_current_edge and should_connect and not should_disconnect:
                    propose_edge = True
                    decision_reason = f"Form edge (Connect rule met: Self={self_prev_degree}, Neigh={neighbor_prev_degree})"
                elif has_current_edge and should_disconnect and not should_connect:
                    propose_edge = False
                    decision_reason = f"Break edge (Disconnect rule met: Self={self_prev_degree}, Neigh={neighbor_prev_degree})"
                elif should_connect and should_disconnect: # Conflict
                    propose_edge = random.choice([True, False]) # Randomly resolve conflict
                    decision_reason = f"Conflict (Connect AND Disconnect rules met) -> Randomly chose {propose_edge}"
                elif has_current_edge:
                     decision_reason = "Maintain existing edge (No disconnect rule, both eligible)"
                else: # Not has_current_edge and not should_connect
                     decision_reason = "Do not form edge (No connect rule, both eligible)"

            elif has_current_edge: # If edge exists but one node is ineligible, break it
                propose_edge = False
                decision_reason = "Break edge (One or both nodes ineligible next step)"
            # else: # No edge exists and one node ineligible -> remains no edge

            # Apply random flip
            random_flip_applied = False
            if np.random.random() < random_flip_prob:
                propose_edge = not propose_edge
                random_flip_applied = True
                decision_reason += " + Random Flip"

            if detailed_logging_enabled:
                logger.detail(f"  Neighbor {neighbor_idx}: PrevDeg={neighbor_prev_degree}, EligibleNext(Proxy)={neighbor_is_eligible}. Decision: {decision_reason}. RandomFlip={random_flip_applied}. Final Propose={propose_edge}") # type: ignore [attr-defined]

            # Add to proposed edges if it should exist
            if propose_edge:
                new_edges[edge] = 1.0 # Binary edges

        return new_edges
    
    def _compute_final_state(
            self, node_idx: int, 
            current_proxy_state: float, 
            final_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]], 
            dimensions: Tuple[int,...],
            detailed_logging_enabled: bool = False,
            ) -> float:
        """Calculates the final state (degree) based on eligibility and final edge count."""

        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_final_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Proxy State (Eligibility): {current_proxy_state:.1f}") # type: ignore [attr-defined]

        # If proxy state indicates ineligible (0.0), final state is 0
        if current_proxy_state < 0.5:
             if detailed_logging_enabled: logger.detail(f"    Node was ineligible based on prev step conditions, final state is 0.") # type: ignore [attr-defined]
             return 0.0

        # If eligible (proxy was 1.0), calculate final degree based on the FINAL edges for this step
        node_coords = tuple(_unravel_index(node_idx, dimensions))
        final_degree = sum(1 for edge in final_edges if node_coords in edge)

        if detailed_logging_enabled:
            logger.detail(f"    Node was eligible. Final Calculated Degree: {final_degree}") # type: ignore [attr-defined]
            logger.detail(f"    Setting Final State to: {float(final_degree):.1f}") # type: ignore [attr-defined]

        # The state IS the degree if eligible
        return float(final_degree)

class MasterConfigurableRule(Rule):
    """
    Highly configurable 'master' rule combining multiple optional logic blocks.
    Node state is binary (0/1), Edge state is continuous (0-1).
    Behavior is determined by enabling/disabling various logic groups via parameters.
    (Round 38: Added edge visualization parameters)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        # 'use_state_coloring_edges', # Keep
        # 'edge_colormap', # Keep
        # 'edge_color_norm_vmin', # Keep
        # 'edge_color_norm_vmax', # Keep
        'color_edges_by_neighbor_degree', # Edge state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edge state not based on this
        'tiebreaker_type', # Not used by core logic
        'node_history_depth'
    }
    # --- END MODIFIED ---

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern (Nodes initialized to 0).", "default": "Random", "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Approx initial density of nodes with connections.", "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability for RANDOM edge init.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "min_edge_weight": { "type": float, "description": "Minimum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.1, "parameter_group": "Initialization"},
        "max_edge_weight": { "type": float, "description": "Maximum initial random edge weight.", "min": 0.0, "max": 1.0, "default": 0.8, "parameter_group": "Initialization"},

        # === Node Logic Group 1: GoL Counts ===
        "use_gol_counts_group": {'type': bool, 'default': True, 'description': "Node Logic: Enable GoL B/S neighbor counts.", "parameter_group": "Node Group 1: GoL Counts"},
        "birth_neighbor_counts": { 'type': list, 'element_type': int, 'default': [3], 'description': "Node Logic: Active neighbor counts for birth.", "parameter_group": "Node Group 1: GoL Counts"},
        "survival_neighbor_counts": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Node Logic: Active neighbor counts for survival.", "parameter_group": "Node Group 1: GoL Counts"},

        # === Node Logic Group 2: Edge Strength Sum ===
        "use_edge_strength_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable edge strength sum condition (uses prev step edges to active neighbors).", "parameter_group": "Node Group 2: Edge Strength Sum"},
        "birth_min_edge_strength_sum": { 'type': float, 'default': 1.0, 'description': "Node Logic: Min edge sum to active neighbors for birth.", "min": 0.0, "max": 26.0, "parameter_group": "Node Group 2: Edge Strength Sum"},
        "survival_min_edge_strength_sum": { 'type': float, 'default': 0.5, 'description': "Node Logic: Min edge sum to active neighbors for survival.", "min": 0.0, "max": 26.0, "parameter_group": "Node Group 2: Edge Strength Sum"},

        # === Node Logic Group 3: Degree Survival (Prev Step) ===
        "use_degree_survival_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable survival based on degree range (uses PREVIOUS step degree). Requires neighbor_degrees data.", "parameter_group": "Node Group 3: Degree Survival (Prev)"},
        "survival_degree_min": { 'type': int, 'default': 1, 'description': "Node Logic: Min connections (prev step) for survival eligibility.", "min": 0, "max": 26, "parameter_group": "Node Group 3: Degree Survival (Prev)"},
        "survival_degree_max": { 'type': int, 'default': 5, 'description': "Node Logic: Max connections (prev step) for survival eligibility.", "min": 0, "max": 26, "parameter_group": "Node Group 3: Degree Survival (Prev)"},

        # === Node Logic Group 4: Avg Edge State ===
        "use_avg_edge_influence_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable influence from avg neighbor edge state (uses PREVIOUS step edges).", "parameter_group": "Node Group 4: Avg Edge State"},
        "birth_min_avg_edge_state": { 'type': float, 'default': 0.4, 'description': "Node Logic: Min avg edge state (prev step) required for birth.", "min": 0.0, "max": 1.0, "parameter_group": "Node Group 4: Avg Edge State"},
        "survival_min_avg_edge_state": { 'type': float, 'default': 0.1, 'description': "Node Logic: Min avg edge state (prev step) required for survival.", "min": 0.0, "max": 1.0, "parameter_group": "Node Group 4: Avg Edge State"},

        # === Node Logic Group 5: Isolation/Overcrowding (Current Step) ===
        "use_isolation_overcrowding_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable death based on CURRENT degree (calculated from active edges). Applied after other checks.", "parameter_group": "Node Group 5: Isolation/Overcrowding (Current)"},
        "isolation_death_threshold": { 'type': int, 'default': 1, 'description': "Node Logic: Force death if CURRENT degree < this.", "min": 0, "max": 26, "parameter_group": "Node Group 5: Isolation/Overcrowding (Current)"},
        "overcrowding_death_threshold": { 'type': int, 'default': 7, 'description': "Node Logic: Force death if CURRENT degree > this.", "min": 0, "max": 26, "parameter_group": "Node Group 5: Isolation/Overcrowding (Current)"},

        # === Node Logic Group 6: Random Birth Boost ===
        "use_node_randomness_group": {'type': bool, 'default': False, 'description': "Node Logic: Enable random node birth boost.", "parameter_group": "Node Group 6: Random Birth Boost"},
        "random_birth_boost": { "type": float, "description": "Node Logic: Small random chance boost for inactive node activation.", "min": 0.0, "max": 0.1, "default": 0.005, "parameter_group": "Node Group 6: Random Birth Boost"},

        # === Edge Logic Group A: Node Activity ===
        "use_node_activity_edge_group": {'type': bool, 'default': True, 'description': "Edge Logic (Priority 4): Enable edge target based on connected nodes' predicted activity.", "parameter_group": "Edge Group A: Node Activity"},
        "edge_target_factor": { "type": float, "description": "Edge Logic: Multiplier for avg predicted node state to get target edge state.", "min": 0.0, "max": 2.0, "default": 1.0, "parameter_group": "Edge Group A: Node Activity"},
        "edge_activity_change_rate": { "type": float, "description": "Edge Logic: Rate edge moves towards activity target.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Edge Group A: Node Activity"},

        # === Edge Logic Group B: Similarity ===
        "use_similarity_edge_group": {'type': bool, 'default': False, 'description': "Edge Logic (Priority 3): Enable edge target/rate based on similarity of predicted node states. Overrides Group A.", "parameter_group": "Edge Group B: Similarity"},
        "similarity_threshold": { "type": float, "description": "Edge Logic: Max state difference to consider nodes 'similar'.", "min": 0.0, "max": 1.0, "default": 0.3, "parameter_group": "Edge Group B: Similarity"},
        "similarity_strengthen_factor": { "type": float, "description": "Edge Logic: Factor modifying change rate towards 1.0 when nodes are similar.", "min": 0.0, "max": 1.0, "default": 1.0, "parameter_group": "Edge Group B: Similarity"},
        "dissimilarity_decay_factor": { "type": float, "description": "Edge Logic: Factor modifying change rate towards 0.0 when nodes are dissimilar.", "min": 0.0, "max": 1.0, "default": 1.0, "parameter_group": "Edge Group B: Similarity"},

        # === Edge Logic Group C: Neighbor Degree (Prev Step) ===
        "use_neighbor_degree_edge_group": {'type': bool, 'default': False, 'description': "Edge Logic (Priority 2): Enable edge target/rate based on neighbor degree thresholds (uses PREVIOUS step degree). Requires neighbor_degrees data. Overrides Groups A, B.", "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "connect_neighbor_degree_min": { 'type': int, 'default': 1, 'description': "Edge Logic: Connect if neighbor degree (prev step) >= this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "connect_neighbor_degree_max": { 'type': int, 'default': 4, 'description': "Edge Logic: Connect if neighbor degree (prev step) <= this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "disconnect_neighbor_degree_min": { 'type': int, 'default': 1, 'description': "Edge Logic: Disconnect if neighbor degree (prev step) < this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "disconnect_neighbor_degree_max": { 'type': int, 'default': 6, 'description': "Edge Logic: Disconnect if neighbor degree (prev step) > this.", "min": 0, "max": 26, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},
        "neighbor_degree_change_rate": { "type": float, "description": "Edge Logic: Rate edge moves towards 0/1 based on neighbor degree.", "min": 0.0, "max": 1.0, "default": 0.6, "parameter_group": "Edge Group C: Neighbor Degree (Prev)"},

        # === Edge Logic Group D: Node Effects (Prev Step) ===
        "use_node_effects_on_edges_group": {'type': bool, 'default': False, 'description': "Edge Logic (Priority 1): Enable edge target/rate based on neighbor count (uses PREVIOUS step counts). Requires neighbor_active_counts data. Overrides Groups A, B, C.", "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_survival_neighbor_min": { 'type': int, 'default': 2, 'description': "Edge Logic: Min active neighbors (prev step) for nodes to boost edge survival.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_survival_neighbor_max": { 'type': int, 'default': 4, 'description': "Edge Logic: Max active neighbors (prev step) for nodes to boost edge survival.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_survival_boost_rate": { 'type': float, 'default': 0.1, 'description': "Edge Logic: Rate edge moves towards 1 if node neighbor counts optimal.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_death_neighbor_min": { 'type': int, 'default': 1, 'description': "Edge Logic: Edge death boost if node neighbor count (prev step) < this.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_death_neighbor_max": { 'type': int, 'default': 5, 'description': "Edge Logic: Edge death boost if node neighbor count (prev step) > this.", "min": 0, "max": 26, "parameter_group": "Edge Group D: Node Effects (Prev)"},
        "edge_death_boost_rate": { 'type': float, 'default': 0.2, 'description': "Edge Logic: Rate edge moves towards 0 if node neighbor counts non-optimal.", "min": 0.0, "max": 1.0, "parameter_group": "Edge Group D: Node Effects (Prev)"},

        # === Edge Logic Group E: Decay ===
        "use_edge_decay_group": {'type': bool, 'default': True, 'description': "Edge Logic: Enable Constant Edge Decay.", "parameter_group": "Edge Group E: Decay"},
        "edge_decay_rate": { "type": float, "description": "Edge Logic: Constant decay applied each step.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Group E: Decay"},

        # === Edge Logic Group F: Randomness ===
        "use_edge_randomness_group": {'type': bool, 'default': False, 'description': "Edge Logic: Enable Random Edge Fluctuations.", "parameter_group": "Edge Group F: Randomness"},
        "random_edge_change_prob": { "type": float, "description": "Edge Logic: Probability of random change.", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Group F: Randomness"},
        "random_edge_change_amount": { "type": float, "description": "Edge Logic: Max amount (+/-) of random change.", "min": 0.0, "max": 0.5, "default": 0.15, "parameter_group": "Edge Group F: Randomness"},

        # --- Edge Pruning ---
        "min_edge_state_to_keep": { "type": float, "description": "Edge Logic: Minimum edge state required to be kept.", "min": 0.0, "max": 0.5, "default": 0.02, "parameter_group": "Edge Pruning"},

        # --- Visualization ---
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (0 or 1).", "default": False, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states.", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization: Nodes"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on their state value (0-1).",
            "default": True, "parameter_group": "Visualization: Edges" # Default True for continuous
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "prism", "parameter_group": "Visualization: Edges",
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # --- Other ---
        "node_history_depth": {
            'type': int, 'default': 10, 'min': 0, 'max': 100,
            'description': "Grid Setting: Number of previous node states stored internally (not used by this rule's logic).",
            'parameter_group': 'History'
        }
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Master Configurable Rule"
        metadata.description = "Highly configurable rule combining multiple optional logic blocks for node (0/1) and edge (0-1) updates. Node state via AND logic across enabled groups (GoL, Edge Sum, Prev Degree, Avg Edge, Current Degree). Edge state via priority override (Node Effects > Neigh Degree > Similarity > Node Activity) plus Decay/Randomness." # Updated description
        metadata.category = "Experimental"
        metadata.tags = ["Life", "Edges", "Feedback", "Modular", "Configurable", "Experimental", "Master"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN", "HEX", "HEX_PRISM"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Master Configurable Rule"
        self._params = {}
        self.requires_post_edge_state_update = False # State determined directly
        # --- Set flags based on potential parameter usage ---
        # Check which groups might need 2nd order data
        self.needs_neighbor_degrees = True # Needed for Group 3 and Group C
        self.needs_neighbor_active_counts = True # Needed for Group D
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute node state based on enabled logic groups using AND logic."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Get toggle parameters for node logic
        use_gol, use_edge_sum, use_degree, use_avg_edge, use_iso_over, use_random = self._get_parameters(
            'use_gol_counts_group', 'use_edge_strength_group', 'use_degree_survival_group',
            'use_avg_edge_influence_group', 'use_isolation_overcrowding_group', 'use_node_randomness_group'
        )

        # --- Calculate Intermediate Eligibility (before Isolation/Overcrowding) ---
        eligible_intermediate = True # Start assuming eligible
        reasons = []

        # Group 1: GoL Counts
        if use_gol:
            birth_counts = self.get_param('birth_neighbor_counts', [3], neighborhood=neighborhood)
            survival_counts = self.get_param('survival_neighbor_counts', [2, 3], neighborhood=neighborhood)
            num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
            passes_gol = False
            if neighborhood.node_state <= 0: # Birth check
                passes_gol = num_active_neighbors in birth_counts
            else: # Survival check
                passes_gol = num_active_neighbors in survival_counts
            if not passes_gol: eligible_intermediate = False
            reasons.append(f"GoL={passes_gol}")

        # Group 2: Edge Strength Sum
        if use_edge_sum and eligible_intermediate:
            birth_min_sum = self.get_param('birth_min_edge_strength_sum', 1.0, neighborhood=neighborhood)
            survival_min_sum = self.get_param('survival_min_edge_strength_sum', 0.5, neighborhood=neighborhood)
            sum_edge_strength = sum(
                neighborhood.neighbor_edge_states.get(idx, 0.0)
                for idx, state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states)
                if idx >= 0 and state > 0 # Only edges to ACTIVE neighbors
            )
            passes_edge_sum = False
            if neighborhood.node_state <= 0: # Birth check
                passes_edge_sum = sum_edge_strength >= birth_min_sum
            else: # Survival check
                passes_edge_sum = sum_edge_strength >= survival_min_sum
            if not passes_edge_sum: eligible_intermediate = False
            reasons.append(f"EdgeSum={passes_edge_sum}")

        # Group 3: Degree Survival (Prev Step)
        if use_degree and eligible_intermediate and neighborhood.node_state > 0: # Only applies to survival
            survival_min_deg = self.get_param('survival_degree_min', 1, neighborhood=neighborhood)
            survival_max_deg = self.get_param('survival_degree_max', 5, neighborhood=neighborhood)
            prev_degree = 0
            if neighborhood.neighbor_degrees is not None:
                 prev_degree = neighborhood.neighbor_degrees.get(node_idx, 0)
            else: logger.warning(f"Node {node_idx}: neighbor_degrees missing for Group 3!")
            passes_degree = (survival_min_deg <= prev_degree <= survival_max_deg)
            if not passes_degree: eligible_intermediate = False
            reasons.append(f"PrevDeg={passes_degree}")

        # Group 4: Avg Edge State
        if use_avg_edge and eligible_intermediate:
            birth_min_avg = self.get_param('birth_min_avg_edge_state', 0.4, neighborhood=neighborhood)
            survival_min_avg = self.get_param('survival_min_avg_edge_state', 0.1, neighborhood=neighborhood)
            avg_neighbor_edge_state = neighborhood.neighborhood_metrics.get('avg_neighbor_edge_state', 0.0)
            passes_avg_edge = False
            if neighborhood.node_state <= 0: # Birth check
                passes_avg_edge = avg_neighbor_edge_state >= birth_min_avg
            else: # Survival check
                passes_avg_edge = avg_neighbor_edge_state >= survival_min_avg
            if not passes_avg_edge: eligible_intermediate = False
            reasons.append(f"AvgEdge={passes_avg_edge}")

        # --- Determine Final Eligibility (after Isolation/Overcrowding) ---
        final_eligible = eligible_intermediate
        death_override_reason = None

        # Group 5: Isolation/Overcrowding (Current Step)
        if use_iso_over and final_eligible: # Only apply if still eligible
            isolation_thr = self.get_param('isolation_death_threshold', 1, neighborhood=neighborhood)
            overcrowding_thr = self.get_param('overcrowding_death_threshold', 7, neighborhood=neighborhood)
            # Calculate CURRENT degree based on active edges
            current_degree = sum(1 for idx, state in neighborhood.neighbor_edge_states.items() if idx >= 0 and state > 1e-6)
            if current_degree < isolation_thr:
                final_eligible = False
                death_override_reason = f"IsoDeath (CurrDeg={current_degree}<{isolation_thr})"
            elif current_degree > overcrowding_thr:
                final_eligible = False
                death_override_reason = f"OverDeath (CurrDeg={current_degree}>{overcrowding_thr})"
            reasons.append(f"IsoOver={final_eligible}") # Reflects outcome after check

        # Group 6: Random Birth Boost
        if not final_eligible and use_random and neighborhood.node_state <= 0: # Only boost inactive nodes
            rand_boost = self.get_param('random_birth_boost', 0.005, neighborhood=neighborhood)
            if np.random.random() < rand_boost:
                final_eligible = True
                reasons.append("RandBirth=True")

        new_state = 1.0 if final_eligible else 0.0

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Checks: {', '.join(reasons)}") # type: ignore [attr-defined]
            if death_override_reason: logger.detail(f"    Override: {death_override_reason}") # type: ignore [attr-defined]
            logger.detail(f"    Final State: {new_state:.1f}") # type: ignore [attr-defined]

        return new_state

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Compute new continuous edge states based on enabled logic groups with priority."""
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Get toggle parameters for edge logic
        use_node_act, use_sim, use_neigh_deg, use_node_eff, use_decay, use_random = self._get_parameters(
             'use_node_activity_edge_group', 'use_similarity_edge_group',
             'use_neighbor_degree_edge_group', 'use_node_effects_on_edges_group',
             'use_edge_decay_group', 'use_edge_randomness_group'
        )

        # Determine the *next* state of the current node (proxy 0 or 1)
        next_node_state = self._compute_new_state(neighborhood, detailed_logging_enabled)
        self_is_active_next = next_node_state > 0.5

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            edge = (node_idx, neighbor_idx) # Canonical order
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0)

            # Approximation: Use current neighbor state as proxy for neighbor's next state.
            neighbor_is_active_next = neighbor_state > 0.5
            next_neighbor_state_proxy = 1.0 if neighbor_is_active_next else 0.0

            # --- Get Data Needed by Groups (Prev Step Data) ---
            neighbor_prev_degree = 0
            if use_neigh_deg and neighborhood.neighbor_degrees is not None:
                 neighbor_prev_degree = neighborhood.neighbor_degrees.get(neighbor_idx, 0)
            elif use_neigh_deg: logger.warning(f"Node {node_idx}: neighbor_degrees missing for Group C!")

            neighbor_prev_active_count = 0
            self_prev_active_count = 0
            if use_node_eff and neighborhood.neighbor_active_counts is not None:
                 neighbor_prev_active_count = neighborhood.neighbor_active_counts.get(neighbor_idx, 0)
                 self_prev_active_count = neighborhood.neighbor_active_counts.get(node_idx, 0) # Need self count too
            elif use_node_eff: logger.warning(f"Node {node_idx}: neighbor_active_counts missing for Group D!")
            # ---

            # --- Determine Target Edge State & Change Rate based on Priority ---
            target_edge_state = current_edge_state # Default: maintain
            effective_change_rate = 0.0 # Default: no change
            applied_group = "None"

            # Group D: Node Effects (Highest Priority)
            if use_node_eff:
                min_n, max_n, surv_rate, min_d, max_d, death_rate = self._get_parameters(
                    'edge_survival_neighbor_min', 'edge_survival_neighbor_max', 'edge_survival_boost_rate',
                    'edge_death_neighbor_min', 'edge_death_neighbor_max', 'edge_death_boost_rate'
                )
                self_surv_boost = (min_n <= self_prev_active_count <= max_n)
                neigh_surv_boost = (min_n <= neighbor_prev_active_count <= max_n)
                self_death_boost = (self_prev_active_count < min_d or self_prev_active_count > max_d)
                neigh_death_boost = (neighbor_prev_active_count < min_d or neighbor_prev_active_count > max_d)

                if self_surv_boost and neigh_surv_boost:
                    target_edge_state = 1.0
                    effective_change_rate = surv_rate
                    applied_group = "D:NodeEffects(Surv)"
                elif self_death_boost or neigh_death_boost:
                    target_edge_state = 0.0
                    effective_change_rate = death_rate
                    applied_group = "D:NodeEffects(Death)"

            # Group C: Neighbor Degree (If Group D didn't apply)
            if applied_group == "None" and use_neigh_deg:
                min_c, max_c, min_d, max_d, rate = self._get_parameters(
                    'connect_neighbor_degree_min', 'connect_neighbor_degree_max',
                    'disconnect_neighbor_degree_min', 'disconnect_neighbor_degree_max',
                    'neighbor_degree_change_rate'
                )
                should_connect = (min_c <= neighbor_prev_degree <= max_c)
                should_disconnect = (neighbor_prev_degree < min_d or neighbor_prev_degree > max_d)

                if should_connect and not should_disconnect:
                    target_edge_state = 1.0
                    effective_change_rate = rate
                    applied_group = "C:NeighDeg(Connect)"
                elif should_disconnect and not should_connect:
                    target_edge_state = 0.0
                    effective_change_rate = rate
                    applied_group = "C:NeighDeg(Disconnect)"
                # If conflict or neither applies, fall through

            # Group B: Similarity (If Groups C, D didn't apply)
            if applied_group == "None" and use_sim:
                sim_thr, sim_f, dissim_f = self._get_parameters(
                    'similarity_threshold', 'similarity_strengthen_factor', 'dissimilarity_decay_factor'
                )
                # Use predicted next states
                state_diff = abs(next_node_state - next_neighbor_state_proxy)
                base_rate = self.get_param('edge_activity_change_rate', 0.5) # Need a base rate if Group A is off

                if state_diff < sim_thr: # Similar
                    target_edge_state = 1.0
                    effective_change_rate = base_rate * sim_f
                    applied_group = "B:Similarity(Similar)"
                else: # Dissimilar
                    target_edge_state = 0.0
                    effective_change_rate = base_rate * dissim_f
                    applied_group = "B:Similarity(Dissimilar)"

            # Group A: Node Activity (Lowest Priority, if others didn't apply)
            if applied_group == "None" and use_node_act:
                factor, rate = self._get_parameters('edge_target_factor', 'edge_activity_change_rate')
                if self_is_active_next and neighbor_is_active_next:
                    avg_node_state = (next_node_state + next_neighbor_state_proxy) / 2.0
                    target_edge_state = avg_node_state * factor
                else:
                    target_edge_state = 0.0
                effective_change_rate = rate
                applied_group = "A:NodeActivity"

            # --- Apply Changes ---
            new_edge_state = current_edge_state + (target_edge_state - current_edge_state) * effective_change_rate

            # Group E: Apply Decay
            if use_decay:
                decay_rate = self.get_param('edge_decay_rate', 0.01)
                new_edge_state -= decay_rate

            # Group F: Apply Randomness
            if use_random:
                prob, amount = self._get_parameters('random_edge_change_prob', 'random_edge_change_amount')
                if np.random.random() < prob:
                    change = (np.random.random() * 2 - 1) * amount
                    new_edge_state += change

            # Clip and Prune
            clipped_new_edge_state = float(np.clip(new_edge_state, 0.0, 1.0))
            min_keep = self.get_param('min_edge_state_to_keep', 0.02)

            if detailed_logging_enabled:
                logger.detail(f"  Edge {edge}: Curr={current_edge_state:.3f}, Target={target_edge_state:.3f}, Rate={effective_change_rate:.3f} (Group={applied_group}) -> New(clipped)={clipped_new_edge_state:.3f}") # type: ignore [attr-defined]

            if clipped_new_edge_state >= min_keep:
                 new_edges[edge] = clipped_new_edge_state

        return new_edges

class DynamicOscillatorNetwork(Rule):
    """
    Continuous node/edge states (-1 to 1) with oscillations, attraction/repulsion,
    and similarity-based edge dynamics. Logic controlled by boolean groups.
    """

    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'node_history_depth'
    }
    PARAMETER_METADATA = { # Copy the revised parameter metadata here
        # ... (parameters as defined above) ...
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Grid dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern.", "default": "Random", "allowed_values": ['Random', 'ShapeShifting'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Initial density of nodes with non-zero state.", "min": 0.0, "max": 1.0, "default": 0.6, "parameter_group": "Initialization"},
        "initial_state_min": { "type": float, "description": "Min initial state value (-1 to 1) for active nodes.", "min": -1.0, "max": 1.0, "default": -0.8, "parameter_group": "Initialization"},
        "initial_state_max": { "type": float, "description": "Max initial state value (-1 to 1) for active nodes.", "min": -1.0, "max": 1.0, "default": 0.8, "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability for RANDOM edge initialization.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "min_edge_weight": { "type": float, "description": "Min initial random edge weight (-1 to 1).", "min": -1.0, "max": 1.0, "default": -0.5, "parameter_group": "Initialization"},
        "max_edge_weight": { "type": float, "description": "Max initial random edge weight (-1 to 1).", "min": -1.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        "use_node_attraction_repulsion_group": {'type': bool, 'default': True, 'description': "Node Logic: Enable attraction/repulsion based on connected neighbor states.", "parameter_group": "Node Group A: Attraction/Repulsion"},
        "node_attraction_factor": { "type": float, "description": "Node Logic: Strength of attraction towards same-sign connected neighbors.", "min": 0.0, "max": 2.0, "default": 0.4, "parameter_group": "Node Group A: Attraction/Repulsion"},
        "node_repulsion_factor": { "type": float, "description": "Node Logic: Strength of repulsion from opposite-sign connected neighbors.", "min": 0.0, "max": 2.0, "default": 0.2, "parameter_group": "Node Group A: Attraction/Repulsion"},
        "use_node_oscillation_group": {'type': bool, 'default': True, 'description': "Node Logic: Enable state oscillation based on thresholds.", "parameter_group": "Node Group B: Oscillation"},
        "node_oscillation_low_threshold": { "type": float, "description": "Node Logic: State below this targets +1.0.", "min": -1.0, "max": 1.0, "default": -0.7, "parameter_group": "Node Group B: Oscillation"},
        "node_oscillation_high_threshold": { "type": float, "description": "Node Logic: State above this targets -1.0.", "min": -1.0, "max": 1.0, "default": 0.7, "parameter_group": "Node Group B: Oscillation"},
        "node_state_change_rate": { "type": float, "description": "Node Logic: Rate (0-1) node state moves towards target.", "min": 0.0, "max": 1.0, "default": 0.35, "parameter_group": "Node Logic: General"},
        "use_state_distance_connect_group": {'type': bool, 'default': True, 'description': "Edge Logic: Enable edge target based on node state similarity/dissimilarity.", "parameter_group": "Edge Group C: State Distance Connect"},
        "connect_similarity_threshold": { "type": float, "description": "Edge Logic: Max node state difference to target +1.0 (attraction).", "min": 0.0, "max": 2.0, "default": 0.3, "parameter_group": "Edge Group C: State Distance Connect"},
        "disconnect_dissimilarity_threshold": { "type": float, "description": "Edge Logic: Min node state difference to target -1.0 (repulsion).", "min": 0.0, "max": 2.0, "default": 1.0, "parameter_group": "Edge Group C: State Distance Connect"},
        "edge_attraction_rate": { "type": float, "description": "Edge Logic: Change rate when nodes are similar.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Edge Group C: State Distance Connect"},
        "edge_repulsion_rate": { "type": float, "description": "Edge Logic: Change rate when nodes are dissimilar or neutral.", "min": 0.0, "max": 1.0, "default": 0.2, "parameter_group": "Edge Group C: State Distance Connect"},
        "formation_node_threshold": { "type": float, "description": "Edge Logic: Min state magnitude BOTH nodes need to form a new edge.", "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Edge Group C: State Distance Connect"},
        "new_edge_initial_strength": { "type": float, "description": "Edge Logic: Initial strength (-1 to 1) of a newly formed edge.", "min": -1.0, "max": 1.0, "default": 0.1, "parameter_group": "Edge Group C: State Distance Connect"},
        "use_edge_oscillation_group": {'type': bool, 'default': True, 'description': "Edge Logic: Enable edge state oscillation based on thresholds.", "parameter_group": "Edge Group D: Oscillation"},
        "edge_oscillation_low_threshold": { "type": float, "description": "Edge Logic: State below this targets +1.0.", "min": -1.0, "max": 1.0, "default": -0.8, "parameter_group": "Edge Group D: Oscillation"},
        "edge_oscillation_high_threshold": { "type": float, "description": "Edge Logic: State above this targets -1.0.", "min": -1.0, "max": 1.0, "default": 0.8, "parameter_group": "Edge Group D: Oscillation"},
        "use_edge_decay_group": {'type': bool, 'default': True, 'description': "Edge Logic: Enable Constant Edge Decay.", "parameter_group": "Edge Group E: Decay"},
        "edge_decay_rate": { "type": float, "description": "Edge Logic: Constant decay subtracted each step (magnitude reduced towards 0).", "min": 0.0, "max": 0.1, "default": 0.01, "parameter_group": "Edge Group E: Decay"},
        "use_edge_randomness_group": {'type': bool, 'default': True, 'description': "Edge Logic: Enable Random Edge Fluctuations.", "parameter_group": "Edge Group F: Randomness"},
        "random_edge_change_prob": { "type": float, "description": "Edge Logic: Probability of random change.", "min": 0.0, "max": 0.1, "default": 0.02, "parameter_group": "Edge Group F: Randomness"},
        "random_edge_change_amount": { "type": float, "description": "Edge Logic: Max amount (+/-) of random change.", "min": 0.0, "max": 0.5, "default": 0.1, "parameter_group": "Edge Group F: Randomness"},
        "min_edge_state_to_keep": { "type": float, "description": "Edge Logic: Minimum state *magnitude* for edge to persist.", "min": 0.0, "max": 0.5, "default": 0.05, "parameter_group": "Edge Logic: General"},
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state value.", "default": True, "parameter_group": "Visualization"},
        "node_colormap": { "type": str, "description": "Colormap for node states (-1 to 1).", "default": "coolwarm", "parameter_group": "Visualization"},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": -1.0, "parameter_group": "Visualization"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 1.0, "parameter_group": "Visualization"},
        "use_state_coloring_edges": { "type": bool, "description": "Color edges based on state value.", "default": True, "parameter_group": "Visualization"},
        "edge_colormap": { "type": str, "description": "Colormap for edge states (-1 to 1).", "default": "prism", "parameter_group": "Visualization"},
        "edge_color_norm_vmin": { "type": float, "description": "Min value for edge color normalization.", "default": -1.0, "parameter_group": "Visualization"},
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization"},
        "tiebreaker_type": { "type": str, "description": "Tiebreaker (Not used).", "allowed_values": ["RANDOM"], "default": "RANDOM", "parameter_group": "Tiebreaker"},
        "node_history_depth": { 'type': int, 'description': 'Grid Setting: History depth (not used by rule).', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Dynamic Oscillator Network" # New Name
        metadata.description = "Continuous node/edge states (-1 to 1). Nodes influenced by attraction/repulsion from connected neighbors and oscillation thresholds. Edges change based on node state distance and oscillation. Connection/disconnection based on state distance."
        metadata.category = "Continuous" # Changed Category
        metadata.tags = ["Continuous", "Oscillation", "Gradient", "Similarity", "Attraction", "Repulsion", "Edges", "Experimental", "Dynamic Oscillator Network"] # Added tags
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Dynamic Oscillator Network"
        self._params = {}
        self.requires_post_edge_state_update = False
        self.needs_neighbor_degrees = False
        self.needs_neighbor_active_counts = False
    
    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """Compute new continuous node state based on enabled logic groups."""
        # Get toggle parameters
        use_att_rep = self.get_param('use_node_attraction_repulsion_group', True, neighborhood=neighborhood)
        use_osc = self.get_param('use_node_oscillation_group', True, neighborhood=neighborhood)

        # Get value parameters
        att_factor = self.get_param('node_attraction_factor', 0.4, neighborhood=neighborhood)
        rep_factor = self.get_param('node_repulsion_factor', 0.2, neighborhood=neighborhood)
        state_rate = self.get_param('node_state_change_rate', 0.35, neighborhood=neighborhood)
        low_thr = self.get_param('node_oscillation_low_threshold', -0.7, neighborhood=neighborhood)
        high_thr = self.get_param('node_oscillation_high_threshold', 0.7, neighborhood=neighborhood)

        current_state = neighborhood.node_state
        target_state = current_state # Default: inertia

        # --- Group A: Attraction/Repulsion ---
        if use_att_rep:
            sum_same_sign_state = 0.0
            count_same_sign = 0
            sum_opp_sign_state = 0.0
            count_opp_sign = 0

            for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
                if neighbor_idx >= 0 and neighborhood.neighbor_edge_states.get(neighbor_idx, -1.0) > -1.0 + 1e-6: # Check if edge exists (state > -1)
                    if np.sign(neighbor_state) == np.sign(current_state) or np.sign(current_state) == 0: # Treat 0 as attracting both
                        sum_same_sign_state += neighbor_state
                        count_same_sign += 1
                    else:
                        sum_opp_sign_state += neighbor_state
                        count_opp_sign += 1

            attraction_influence = 0.0
            if count_same_sign > 0:
                avg_same_sign = sum_same_sign_state / count_same_sign
                attraction_influence = (avg_same_sign - current_state) * att_factor # Move towards average

            repulsion_influence = 0.0
            if count_opp_sign > 0:
                avg_opp_sign = sum_opp_sign_state / count_opp_sign
                # Push away from the average opposite sign state
                repulsion_influence = (current_state - avg_opp_sign) * rep_factor

            combined_influence = attraction_influence + repulsion_influence
            target_state = current_state + combined_influence # Influence modifies current state direction

        # --- Group B: Oscillation (Overrides influence target) ---
        if use_osc:
            if current_state > high_thr:
                target_state = -1.0 # Force target down
            elif current_state < low_thr:
                target_state = 1.0 # Force target up
            # Else: keep target from influence (or initial state if influence off)

        # Apply inertia
        new_state = current_state + (target_state - current_state) * state_rate

        # Clip and return
        return float(np.clip(new_state, -1.0, 1.0))

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Compute new continuous edge states based on enabled logic groups."""
        new_edges: Dict[Tuple[int, int], float] = {}
        node_idx = neighborhood.node_index

        # Get toggle parameters
        use_connect = self.get_param('use_state_distance_connect_group', True, neighborhood=neighborhood)
        use_osc = self.get_param('use_edge_oscillation_group', True, neighborhood=neighborhood)
        use_decay = self.get_param('use_edge_decay_group', True, neighborhood=neighborhood)
        use_random = self.get_param('use_edge_randomness_group', True, neighborhood=neighborhood)

        # Get value parameters
        connect_thr = self.get_param('connect_similarity_threshold', 0.3, neighborhood=neighborhood)
        disconnect_thr = self.get_param('disconnect_dissimilarity_threshold', 1.0, neighborhood=neighborhood)
        rate_sim = self.get_param('edge_attraction_rate', 0.5, neighborhood=neighborhood)
        rate_dissim = self.get_param('edge_repulsion_rate', 0.2, neighborhood=neighborhood)
        formation_node_thr = self.get_param('formation_node_threshold', 0.4, neighborhood=neighborhood)
        new_edge_strength = self.get_param('new_edge_initial_strength', 0.1, neighborhood=neighborhood)
        low_thr = self.get_param('edge_oscillation_low_threshold', -0.8, neighborhood=neighborhood)
        high_thr = self.get_param('edge_oscillation_high_threshold', 0.8, neighborhood=neighborhood)
        decay_rate = self.get_param('edge_decay_rate', 0.01, neighborhood=neighborhood)
        rand_prob = self.get_param('random_edge_change_prob', 0.02, neighborhood=neighborhood)
        rand_amount = self.get_param('random_edge_change_amount', 0.1, neighborhood=neighborhood)
        min_keep = self.get_param('min_edge_state_to_keep', 0.05, neighborhood=neighborhood)

        # Use current node state as proxy for its next state
        node_state_proxy = neighborhood.node_state

        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            edge = (node_idx, neighbor_idx)
            current_edge_state = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0) # Default 0 if no edge

            # Use current neighbor state as proxy for its next state
            neighbor_state_proxy = neighbor_state

            state_diff = abs(node_state_proxy - neighbor_state_proxy)

            # --- Edge Formation Logic ---
            if current_edge_state == 0.0: # Check if edge needs forming
                if use_connect:
                    is_similar_for_connect = state_diff < connect_thr
                    # Check magnitude for activity threshold
                    nodes_active_enough = abs(node_state_proxy) > formation_node_thr and abs(neighbor_state_proxy) > formation_node_thr
                    if is_similar_for_connect and nodes_active_enough:
                        # Form new edge
                        new_edge_state = new_edge_strength
                        # Apply randomness immediately
                        if use_random and np.random.random() < rand_prob:
                            new_edge_state += (np.random.random() * 2 - 1) * rand_amount
                        clipped_state = float(np.clip(new_edge_state, -1.0, 1.0))
                        # Check magnitude against min_keep
                        if abs(clipped_state) >= min_keep:
                            new_edges[edge] = clipped_state
                # If formation conditions not met, skip to next neighbor
                continue
            # --- End Edge Formation ---

            # --- Edge Update Logic (for existing edges) ---
            target_state = 0.0 # Default target
            effective_change_rate = rate_dissim # Default rate

            # Group C: State Distance Target/Rate
            if use_connect:
                if state_diff < connect_thr:
                    target_state = 1.0
                    effective_change_rate = rate_sim
                elif state_diff > disconnect_thr:
                    target_state = -1.0
                    effective_change_rate = rate_dissim
                # Else: target stays 0.0, rate stays rate_dissim

            # Group D: Oscillation (Overrides target)
            if use_osc:
                if current_edge_state > high_thr:
                    target_state = -1.0
                elif current_edge_state < low_thr:
                    target_state = 1.0

            # Apply Inertia
            new_edge_state = current_edge_state + (target_state - current_edge_state) * effective_change_rate

            # Group E: Decay (Applied towards zero)
            if use_decay:
                decay_amount = decay_rate * np.sign(new_edge_state) # Decay towards zero
                if abs(new_edge_state) > abs(decay_amount):
                    new_edge_state -= decay_amount
                else:
                    new_edge_state = 0.0 # Don't overshoot zero

            # Group F: Randomness
            if use_random and np.random.random() < rand_prob:
                new_edge_state += (np.random.random() * 2 - 1) * rand_amount

            # Clip and Prune
            clipped_state = float(np.clip(new_edge_state, -1.0, 1.0))
            # Check magnitude against min_keep
            if abs(clipped_state) >= min_keep:
                new_edges[edge] = clipped_state
            # --- End Edge Update ---

        return new_edges

class AngularEdgeRule(Rule):
    """
    Node state = degree. Edges (0/1) form/survive/die based on the angular pattern
    of existing edges connected to endpoints. Requires post-update step.
    (Round 38: Added edge visualization parameters)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'min_edge_weight', 'max_edge_weight', # Edges are binary
        'connect_probability', # Not used by core logic, only init
        'color_edges_by_neighbor_degree', # Edges are binary, state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edges are binary, state not based on this
        'node_history_depth'
    }

    node_state_type: ClassVar[StateType] = StateType.INTEGER
    edge_state_type: ClassVar[StateType] = StateType.BINARY # Edges are 0/1
    min_node_state: ClassVar[float] = 0.0 # Degree cannot be negative
    max_node_state: ClassVar[float] = 26.0 # Theoretical max degree (Moore 3D)
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0

    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup.', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern (Nodes start at 0).", "default": "Random", "allowed_values": ['Random'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Approx initial density of nodes with connections.", "min": 0.0, "max": 1.0, "default": 0.4, "parameter_group": "Initialization"},

        # === Node Eligibility Logic (Based on Own Angular Pattern - Prev Step) ===
        "use_node_angle_logic": {'type': bool, 'default': True, 'description': "Node Eligibility: Enable checks based on node's edge angle pattern (prev step).", "parameter_group": "Node Eligibility (Angle Patterns)"},
        "node_birth_angle_patterns": { 'type': list, 'element_type': list, 'default': [['N', 'S'], ['E', 'W'], ['NE', 'SW'], ['NW', 'SE']], 'description': "Node Birth Eligibility: List of direction lists. Node eligible if inactive AND its (empty) pattern implicitly matches OR if patterns match (if logic extended).", "parameter_group": "Node Eligibility (Angle Patterns)"},
        "node_survival_angle_patterns": { 'type': list, 'element_type': list, 'default': [], 'description': "Node Survival Eligibility: List of direction lists. Node eligible if active AND its pattern matches ANY list (if list not empty).", "parameter_group": "Node Eligibility (Angle Patterns)"},
        "node_death_angle_patterns": { 'type': list, 'element_type': list, 'default': [['N','NE','E','SE','S','SW','W','NW']], 'description': "Node Death Eligibility: List of direction lists. Node ineligible if active AND its pattern matches ANY list (overrides survival).", "parameter_group": "Node Eligibility (Angle Patterns)"},

        # === Edge Formation/Destruction Logic (Based on Prev Step Degrees) ===
        "use_edge_degree_logic": {'type': bool, 'default': True, 'description': "Edge Logic: Enable edge formation/destruction based on endpoint degrees (prev step).", "parameter_group": "Edge Logic (Degree Pairs/Ranges)"},
        "edge_connect_degree_pairs": { 'type': list, 'element_type': tuple, 'default': [[1, 3], [1, 4], [2, 4], [2, 5], [3, 5], [3, 6], [4, 6], [4, 1], [3, 1], [4, 2], [5, 2], [5, 3], [6, 3], [6, 4]], 'description': "Edge Formation: List of exact (self_deg, neighbor_deg) pairs (prev step) to trigger connection if no edge exists.", "parameter_group": "Edge Logic (Degree Pairs/Ranges)"},
        "edge_connect_degree_ranges": { 'type': list, 'element_type': tuple, 'default': [], 'description': "Edge Formation: List of ((min_s, max_s), (min_n, max_n)) range pairs (prev step) to trigger connection if no edge exists.", "parameter_group": "Edge Logic (Degree Pairs/Ranges)"},
        "edge_disconnect_degree_pairs": { 'type': list, 'element_type': tuple, 'default': [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], 'description': "Edge Destruction: List of exact (self_deg, neighbor_deg) pairs (prev step) to trigger disconnection if edge exists.", "parameter_group": "Edge Logic (Degree Pairs/Ranges)"},
        "edge_disconnect_degree_ranges": { 'type': list, 'element_type': tuple, 'default': [[[7, 26], [7, 26]]], 'description': "Edge Destruction: List of ((min_s, max_s), (min_n, max_n)) range pairs (prev step) to trigger disconnection if edge exists.", "parameter_group": "Edge Logic (Degree Pairs/Ranges)"},

        # === Edge Logic (Randomness) ===
        "use_edge_randomness_group": {'type': bool, 'default': False, 'description': "Edge Logic: Enable Random Edge Flips.", "parameter_group": "Edge Logic (Randomness)"},
        "random_edge_flip_prob": { "type": float, "description": "Edge Randomness: Probability of randomly flipping a potential edge's state.", "min": 0.0, "max": 0.1, "default": 0.001, "parameter_group": "Edge Logic (Randomness)"},

        # === Final State Death Conditions ===
        "final_death_degree_counts": { 'type': list, 'element_type': int, 'default': [0], 'description': "Final State: Node state becomes 0 if its *final* calculated degree is in this list.", "parameter_group": "Final State Death"},

        # === Visualization ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (degree).", "default": True, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states (degrees).", "default": "rainbow", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 8.0, "parameter_group": "Visualization: Nodes"},
        # --- ADDED Edge Visualization Params ---
        "use_state_coloring_edges": {
            "type": bool, "description": "Color edges based on state (0 or 1).",
            "default": False, "parameter_group": "Visualization: Edges" # Default False for binary
        },
        "edge_colormap": {
            "type": str, "description": "Colormap for edge coloring (if enabled).",
            "default": "binary", "parameter_group": "Visualization: Edges", # Default binary
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "edge_color_norm_vmin": {
            "type": float, "description": "Min value for edge color normalization.",
            "default": 0.0, "parameter_group": "Visualization: Edges"
        },
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # --- END ADDED ---
        # === Other ===
        "tiebreaker_type": { "type": str, "description": "Tiebreaker (Not used).", "allowed_values": ["RANDOM"], "default": "RANDOM", "parameter_group": "Tiebreaker"},
        "node_history_depth": { 'type': int, 'description': 'Grid Setting: History depth (not used by rule).', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Angular Edge Rule"
        # --- UPDATED DESCRIPTION ---
        metadata.description = "Node state = degree. Node eligibility based on angular patterns of its edges (prev step). Edges (0/1) change based on endpoint degrees (prev step). Requires post-update step."
        # ---
        metadata.category = "Experimental"
        metadata.tags = ["Connectivity", "Degree", "Edges", "Angles", "Topology", "Experimental", "Angular Edge Rule"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"] # Simpler neighborhoods easier for angle patterns
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Angular Edge Rule"
        self._params = {}
        self.requires_post_edge_state_update = True # Requires final degree calculation
        # --- Set flags to request data ---
        self.needs_neighbor_degrees = True # Needed for edge logic
        self.needs_neighbor_active_counts = False # Not needed
        # ---
        self.skip_standard_tiebreakers = True # Uses its own edge logic
        # --- Set coloring flags ---
        self._params['use_state_coloring'] = True
        self._params['node_colormap'] = 'rainbow' # Default cmap
        self._params['node_color_norm_vmax'] = 8.0 # Default for Moore 2D
        self._params['use_state_coloring_edges'] = True # Color edges by avg degree
        self._params['edge_colormap'] = 'prism'
        self._params['edge_color_norm_vmax'] = 8.0
        # ---

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _get_edge_directions(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Set[str]:
        """Calculates the set of canonical direction strings for existing edges using the helper."""
        # This helper remains the same.
        directions = set()
        node_coords = np.array(neighborhood.node_coords) # Central node coords

        for neighbor_idx, edge_state in neighborhood.neighbor_edge_states.items():
            if neighbor_idx >= 0 and edge_state > 1e-6: # If edge exists
                try:
                    neighbor_coords = np.array(_unravel_index(neighbor_idx, neighborhood.dimensions))
                    delta = neighbor_coords - node_coords
                    direction_str = Rule._get_direction_from_delta(delta)
                    if not direction_str.startswith("Unknown") and direction_str != "Unsupported_Dim" and direction_str != "Center":
                        directions.add(direction_str)
                    elif direction_str != "Center":
                        logger.warning(f"Node {neighborhood.node_index}: Could not determine valid direction for neighbor {neighbor_idx} (delta: {delta}), got '{direction_str}'")
                except IndexError:
                     logger.warning(f"Node {neighborhood.node_index}: Index error getting coords for neighbor {neighbor_idx}")
                except Exception as e:
                     logger.error(f"Node {neighborhood.node_index}: Error getting direction for neighbor {neighbor_idx}: {e}")

        return directions

    def _check_angle_pattern(self, current_directions: Set[str], patterns: List[List[str]]) -> bool:
        """Checks if the current directions exactly match any of the target patterns."""
        # This helper remains the same.
        if not patterns: return False
        current_directions_sorted_tuple = tuple(sorted(list(current_directions)))
        for pattern_list in patterns:
            target_pattern_sorted_tuple = tuple(sorted(pattern_list))
            if current_directions_sorted_tuple == target_pattern_sorted_tuple:
                return True
        return False

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """
        Calculates the node's ELIGIBILITY proxy state (0 or 1) based on its
        angular edge pattern from the previous step.
        (Round 112: Refactored logic)
        """
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Get parameters
        use_angle_logic = self.get_param('use_node_angle_logic', True, neighborhood=neighborhood)
        birth_patterns = self.get_param('node_birth_angle_patterns', [], neighborhood=neighborhood)
        survival_patterns = self.get_param('node_survival_angle_patterns', [], neighborhood=neighborhood)
        death_patterns = self.get_param('node_death_angle_patterns', [], neighborhood=neighborhood)

        # Calculate current angular pattern from previous step's edges
        current_directions = self._get_edge_directions(neighborhood, detailed_logging_enabled)
        current_degree = len(current_directions) # Degree is based on existing edges

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_state (Eligibility Proxy) ---") # type: ignore [attr-defined]
            logger.detail(f"    Prev State (Degree): {neighborhood.node_state:.0f}") # type: ignore [attr-defined]
            logger.detail(f"    Current Edge Directions (Prev Step): {current_directions}") # type: ignore [attr-defined]
            logger.detail(f"    Using Angle Logic: {use_angle_logic}") # type: ignore [attr-defined]

        eligibility_proxy = 1.0 # Default to eligible if angle logic is off
        decision_reason = "Default (Angle Logic Disabled or No Conditions Met)"

        if use_angle_logic:
            eligibility_proxy = 0.0 # Default to ineligible if angle logic is on
            decision_reason = "Default (Ineligible - Angle Logic On)"
            is_currently_active = current_degree > 0

            if not is_currently_active: # --- Check Birth Eligibility ---
                # Birth only happens if the (empty) pattern matches a birth pattern
                # (or if birth patterns list is empty, meaning any inactive node can be born)
                # Current implementation: Birth requires matching a specific pattern (e.g., [])
                # Let's adjust: If birth_patterns is empty, inactive nodes are NOT eligible by default.
                # If birth_patterns contains [], then degree 0 nodes are eligible.
                # If birth_patterns contains other patterns, those are checked (though unlikely for degree 0).
                matches_birth = self._check_angle_pattern(current_directions, birth_patterns)
                if matches_birth:
                    eligibility_proxy = 1.0
                    decision_reason = f"Eligible for Birth (Pattern {current_directions} matches {birth_patterns})"
                else:
                    decision_reason = f"Ineligible (Birth pattern {current_directions} not in {birth_patterns})"

            else: # --- Check Survival Eligibility ---
                matches_death = self._check_angle_pattern(current_directions, death_patterns)
                if matches_death:
                    eligibility_proxy = 0.0 # Dies due to death pattern
                    decision_reason = f"Ineligible (Death pattern {current_directions} matches {death_patterns})"
                else:
                    # Survive if pattern matches survival list OR if survival list is empty (meaning any non-death pattern survives)
                    matches_survival = self._check_angle_pattern(current_directions, survival_patterns)
                    if not survival_patterns or matches_survival:
                        eligibility_proxy = 1.0 # Survives
                        decision_reason = f"Eligible for Survival (Pattern {current_directions} matches {survival_patterns} or survival list empty)"
                    else:
                        eligibility_proxy = 0.0 # Dies (doesn't match survival or death)
                        decision_reason = f"Ineligible (Death: Pattern {current_directions} not in survival {survival_patterns} or death {death_patterns})"

        if detailed_logging_enabled:
            logger.detail(f"    Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    Returning Eligibility Proxy: {eligibility_proxy:.1f}") # type: ignore [attr-defined]

        return eligibility_proxy

    def _check_degree_pairs(self, self_deg: int, neighbor_deg: int,
                            pairs: List[Tuple[int, int]],
                            ranges: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> bool:
        """Checks if a degree pair matches any exact pair or range."""
        # Check exact pairs (order matters for asymmetric rules)
        # Ensure pairs contains tuples, not lists, for comparison if needed
        pairs_as_tuples = [tuple(p) for p in pairs]
        if (self_deg, neighbor_deg) in pairs_as_tuples: # Check tuple against list of tuples
            return True
        # Check ranges
        for range_pair in ranges:
             # Ensure inner elements are tuples before unpacking
             if isinstance(range_pair, (list, tuple)) and len(range_pair) == 2:
                 range1, range2 = range_pair
                 if isinstance(range1, (list, tuple)) and len(range1) == 2 and \
                    isinstance(range2, (list, tuple)) and len(range2) == 2:
                     min_s, max_s = range1
                     min_n, max_n = range2
                     if (min_s <= self_deg <= max_s) and (min_n <= neighbor_deg <= max_n):
                         return True
                 else:
                      # Log or handle error for invalid inner range format
                      logger = logging.getLogger(__name__)
                      logger.warning(f"Skipping invalid range format inside ranges list: {range_pair}")
             else:
                 # Log or handle error for invalid outer range format
                 logger = logging.getLogger(__name__)
                 logger.warning(f"Skipping invalid range pair format in ranges list: {range_pair}")

        return False

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Determine edge existence based on endpoint degrees (prev step) AND mutual eligibility."""

        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        use_degree_logic = self.get_param('use_edge_degree_logic', True, neighborhood=neighborhood)
        connect_pairs = self.get_param('edge_connect_degree_pairs', [], neighborhood=neighborhood)
        connect_ranges = self.get_param('edge_connect_degree_ranges', [], neighborhood=neighborhood)
        disconnect_pairs = self.get_param('edge_disconnect_degree_pairs', [], neighborhood=neighborhood)
        disconnect_ranges = self.get_param('edge_disconnect_degree_ranges', [], neighborhood=neighborhood)
        use_randomness = self.get_param('use_edge_randomness_group', False, neighborhood=neighborhood)
        random_flip_prob = self.get_param('random_edge_flip_prob', 0.0, neighborhood=neighborhood)

        # Convert list pairs/ranges from params to tuples
        connect_pairs_tuples = [tuple(p) for p in connect_pairs]
        connect_ranges_tuples = [(tuple(r1), tuple(r2)) for r1, r2 in connect_ranges]
        disconnect_pairs_tuples = [tuple(p) for p in disconnect_pairs]
        disconnect_ranges_tuples = [(tuple(r1), tuple(r2)) for r1, r2 in disconnect_ranges]

        # Get Eligibility Proxies from rule_params
        eligibility_proxies_flat = neighborhood.rule_params.get('_eligibility_proxies')
        if eligibility_proxies_flat is None:
            logger.error(f"Node {node_idx}: Eligibility proxies missing in rule_params!")
            return new_edges

        # Determine eligibility of the current node from proxy
        self_is_eligible = False
        if 0 <= node_idx < eligibility_proxies_flat.size:
            self_is_eligible = eligibility_proxies_flat[node_idx] > 0.5
        else: logger.warning(f"Node {node_idx}: Index out of bounds for eligibility proxies.")

        # Node's degree from previous step (is its state)
        self_prev_degree = int(neighborhood.node_state)

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Prev Degree: {self_prev_degree}, Eligible Next (from proxy): {self_is_eligible}") # type: ignore [attr-defined]
            logger.detail(f"    Using Degree Logic: {use_degree_logic}, Using Randomness: {use_randomness}") # type: ignore [attr-defined]

        for neighbor_idx, neighbor_prev_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0 or neighbor_idx <= node_idx: continue

            # Get Neighbor Eligibility from Proxy Array
            neighbor_is_eligible = False
            if 0 <= neighbor_idx < eligibility_proxies_flat.size:
                neighbor_is_eligible = eligibility_proxies_flat[neighbor_idx] > 0.5
            else: logger.warning(f"Node {node_idx}: Neighbor index {neighbor_idx} out of bounds for eligibility proxies."); continue

            # Get neighbor's degree from previous step
            neighbor_prev_degree = 0
            if neighborhood.neighbor_degrees is not None:
                 neighbor_prev_degree = neighborhood.neighbor_degrees.get(neighbor_idx, 0)
            else: logger.warning(f"Node {node_idx}: neighbor_degrees missing for neighbor {neighbor_idx}")

            edge = (node_idx, neighbor_idx) # Canonical order
            has_current_edge = neighborhood.neighbor_edge_states.get(neighbor_idx, 0.0) > 1e-6

            propose_edge = has_current_edge # Default: maintain
            decision_reason = "Default (Maintain)"

            # Apply Edge Logic ONLY if BOTH nodes are eligible
            if self_is_eligible and neighbor_is_eligible:
                if use_degree_logic:
                    # --- MODIFIED: Correct call to _check_degree_pairs ---
                    should_connect = self._check_degree_pairs(self_prev_degree, neighbor_prev_degree, connect_pairs_tuples, connect_ranges_tuples)
                    should_disconnect = self._check_degree_pairs(self_prev_degree, neighbor_prev_degree, disconnect_pairs_tuples, disconnect_ranges_tuples)
                    # --- END MODIFIED ---

                    if not has_current_edge and should_connect and not should_disconnect:
                        propose_edge = True
                        decision_reason = f"Form (Degree rule met: Self={self_prev_degree}, Neigh={neighbor_prev_degree})"
                    elif has_current_edge and should_disconnect and not should_connect:
                        propose_edge = False
                        decision_reason = f"Break (Degree rule met: Self={self_prev_degree}, Neigh={neighbor_prev_degree})"
                    elif should_connect and should_disconnect: # Conflict
                        propose_edge = random.choice([True, False]) # Randomly resolve conflict
                        decision_reason = f"Conflict (Connect AND Disconnect rules met) -> Randomly chose {propose_edge}"
                    elif has_current_edge:
                         decision_reason = "Maintain (No disconnect rule, both eligible)"
                    else: # Not has_current_edge and not should_connect
                         decision_reason = "No Form (No connect rule, both eligible)"
                else: # Degree logic disabled, just connect if both eligible
                    propose_edge = True
                    decision_reason = "Maintain/Form (Degree Logic Disabled, Both Eligible)"

            elif has_current_edge: # If edge exists but one node is ineligible, break it
                propose_edge = False
                decision_reason = "Break (One or both nodes ineligible next step)"
            # else: # No edge exists and one node ineligible -> remains no edge

            # Apply random flip
            random_flip_applied = False
            if use_randomness and np.random.random() < random_flip_prob:
                propose_edge = not propose_edge
                random_flip_applied = True
                decision_reason += " + Random Flip"

            if detailed_logging_enabled:
                logger.detail(f"  Neighbor {neighbor_idx}: PrevDeg={neighbor_prev_degree}, EligibleNext(Proxy)={neighbor_is_eligible}. Decision: {decision_reason}. RandomFlip={random_flip_applied}. Final Propose={propose_edge}") # type: ignore [attr-defined]

            # Add to proposed edges if it should exist
            if propose_edge:
                new_edges[edge] = 1.0 # Binary edges

        return new_edges
    
    def _compute_final_state(self,
                             node_idx: int,
                             current_proxy_state: float, # Eligibility/state from current step's computation
                             final_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]], # Edges calculated for *current* step
                             dimensions: Tuple[int,...],
                             # --- CORRECTED: Accept all 10 arguments ---
                             previous_node_states: npt.NDArray[np.float64],
                             previous_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
                             previous_edge_states: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float],
                             previous_node_degrees: Optional[npt.NDArray[np.int32]],
                             previous_active_neighbors: Optional[npt.NDArray[np.int32]],
                             eligibility_proxies: Optional[np.ndarray] = None,
                             detailed_logging_enabled: bool = False
                             # --- END CORRECTION ---
                             ) -> float:
        """Calculates the final state (degree) based on eligibility and final edge count, applying death list."""
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_final_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Proxy State (Eligibility): {current_proxy_state:.1f}") # type: ignore [attr-defined]

        # If proxy state indicates ineligible (0.0), final state is 0
        if current_proxy_state < 0.5:
             if detailed_logging_enabled: logger.detail(f"    Node was ineligible based on prev step conditions, final state is 0.") # type: ignore [attr-defined]
             return 0.0

        # If eligible (proxy was 1.0), calculate final degree based on the FINAL edges for this step
        node_coords = tuple(_unravel_index(node_idx, dimensions))
        final_degree = sum(1 for edge in final_edges if node_coords in edge)
        if detailed_logging_enabled: logger.detail(f"    Node was eligible. Final Calculated Degree: {final_degree}") # type: ignore [attr-defined]

        # Apply death list based on the FINAL degree
        final_death_degrees = self.get_param('final_death_degree_counts', [0]) # Using new default

        final_state = 0.0 # Default to death
        decision_reason = "Default (Death)"
        if final_degree in final_death_degrees:
            decision_reason = f"Final Death (Final Degree={final_degree} in death list {final_death_degrees})"
        else:
            # Survived final death checks, state is the final degree
            final_state = float(final_degree)
            decision_reason = f"Final Survival (Final Degree={final_degree} not in death list)"

        if detailed_logging_enabled:
            logger.detail(f"    Final Decision: {decision_reason}") # type: ignore [attr-defined]
            logger.detail(f"    Final State: {final_state:.1f}") # type: ignore [attr-defined]
        return final_state

class ColoredLifeWithEdges(Rule):
    """
    Game of Life variant where node color represents its degree (connection count).
    Node state logic uses GoL B/S counts, optionally modified by edge connectivity.
    Edge logic is binary, based on node activity/support.
    Requires post-update step to set node state = degree for visualization.
    (Round 15: Added missing PARAMETER_METADATA)
    """
    # --- MODIFIED: Remove edge coloring params from exclusion ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'state_rule_table', 'edge_rule_table', # No rule tables
        'min_edge_weight', 'max_edge_weight', # Edges are binary
        'color_edges_by_neighbor_degree', # Edges are binary, state not based on this
        'color_edges_by_neighbor_active_neighbors', # Edges are binary, state not based on this
        'tiebreaker_type' # Not used by core logic
        'node_history_depth'
    }
    # --- END MODIFIED ---
    
    produces_binary_edges: ClassVar[bool] = True # Edges are 0 or 1
    node_state_type: ClassVar[StateType] = StateType.BINARY
    edge_state_type: ClassVar[StateType] = StateType.BINARY
    min_node_state: ClassVar[float] = 0.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0

    # --- ADDED PARAMETER_METADATA ---
    PARAMETER_METADATA = {
        # === Core ===
        "dimension_type": { 'type': str, 'default': "TWO_D", 'allowed_values': ["TWO_D", "THREE_D"], 'description': "Dimension.", "parameter_group": "Core"},
        "neighborhood_type": { 'type': str, 'default': "MOORE", 'allowed_values': ["MOORE", "VON_NEUMANN"], 'description': "Neighborhood type.", "parameter_group": "Core"},
        'grid_boundary': { 'type': str, 'allowed_values': ['bounded', 'wrap'], 'default': 'wrap', 'description': 'Grid boundary behavior.', "parameter_group": "Core"},
        # === Initialization ===
        "edge_initialization": { 'type': str, 'allowed_values': ['RANDOM', 'FULL', 'NONE'], 'default': 'RANDOM', 'description': 'Initial edge setup (Binary Edges).', "parameter_group": "Initialization"},
        "initial_conditions": { "type": str, "description": "Initial grid state pattern.", "default": "Random", "allowed_values": ['Random', 'Glider Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'], "parameter_group": "Initialization"},
        "initial_density": { "type": float, "description": "Initial density of active nodes (state=1).", "min": 0.0, "max": 1.0, "default": 0.35, "parameter_group": "Initialization"},
        "connect_probability": { "type": float, "description": "Probability for RANDOM edge init.", "min": 0.0, "max": 1.0, "default": 0.5, "parameter_group": "Initialization"},
        # === Node Update Logic ===
        "birth_neighbor_counts": { 'type': list, 'element_type': int, 'default': [3], 'description': "Node Logic: List of active neighbor counts for birth.", "parameter_group": "Node Logic: GoL Counts"},
        "survival_neighbor_counts": { 'type': list, 'element_type': int, 'default': [2, 3], 'description': "Node Logic: List of active neighbor counts for survival.", "parameter_group": "Node Logic: GoL Counts"},
        "use_edge_effects_on_nodes": { 'type': bool, 'default': False, 'description': "Node Logic: Enable probabilistic boost/penalty to node survival based on edge connectivity.", "parameter_group": "Node Logic: Edge Effects"},
        "survival_boost_factor": { 'type': float, 'default': 0.1, 'description': "Node Logic: Probability (0-1) to survive death if connected to active neighbors.", "min": 0.0, "max": 1.0, "parameter_group": "Node Logic: Edge Effects"},
        "death_boost_factor": { 'type': float, 'default': 0.1, 'description': "Node Logic: Probability (0-1) to die despite survival conditions if NOT connected to active neighbors.", "min": 0.0, "max": 1.0, "parameter_group": "Node Logic: Edge Effects"},
        # === Edge Update Logic ===
        "use_dynamic_edge_logic": { 'type': bool, 'default': False, 'description': "Edge Logic: Enable dynamic edge formation/breakage based on neighbor support.", "parameter_group": "Edge Logic"},
        "edge_support_neighbor_threshold": { 'type': int, 'default': 1, 'description': "Edge Logic: Min active neighbors (prev step) BOTH nodes need to form/maintain edge.", "min": 0, "max": 26, "parameter_group": "Edge Logic"},
        # === Visualization ===
        "use_state_coloring": { "type": bool, "description": "Color nodes based on state (degree).", "default": True, "parameter_group": "Visualization: Nodes"},
        "color_nodes_by_degree": { "type": bool, "description": "Color nodes based on connection count (degree). (Implicitly True if use_state_coloring is True)", "default": True, "parameter_group": "Visualization: Nodes"}, # Default True
        "color_nodes_by_active_neighbors": { "type": bool, "description": "Color nodes based on active neighbor count (previous step).", "default": False, "parameter_group": "Visualization: Nodes"},
        "node_colormap": { "type": str, "description": "Colormap for node states (degrees).", "default": "prism", "parameter_group": "Visualization: Nodes", "allowed_values": ["(None)"] + _standard_colormaps},
        "node_color_norm_vmin": { "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..", "default": 0.0, "parameter_group": "Visualization: Nodes"},
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 8.0, "parameter_group": "Visualization: Nodes"},
        "use_state_coloring_edges": { "type": bool, "description": "Color edges based on state (0 or 1).", "default": False, "parameter_group": "Visualization: Edges"},
        "edge_colormap": { "type": str, "description": "Colormap for edge coloring (if enabled).", "default": "binary", "parameter_group": "Visualization: Edges", "allowed_values": ["(None)"] + _standard_colormaps},
        "edge_color_norm_vmin": { "type": float, "description": "Min value for edge color normalization.", "default": 0.0, "parameter_group": "Visualization: Edges"},
        "edge_color_norm_vmax": { "type": float, "description": "Max value for edge color normalization.", "default": 1.0, "parameter_group": "Visualization: Edges"},
        # === Other ===
        "node_history_depth": { 'type': int, 'description': 'Number of previous states to store.', 'default': 10, 'min': 0, 'max': 100, 'parameter_group': 'History'}
    }
    # --- END ADDED ---

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Colored Life with Edges"
        metadata.description = "GoL B/S node logic (binary eligibility), optionally modified by edge connectivity. Binary edges based on node activity/support. Node state/color = degree (connection count in current step)."
        metadata.category = "Life-Like"
        metadata.tags = ["Life", "Edges", "Binary", "Configurable", "B3/S23", "Connectivity", "Degree Color"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        super().__init__(metadata)
        self.name = "Colored Life with Edges"
        # --- REMOVED: _params initialization here, handled by base class ---
        self.requires_post_edge_state_update = True # Final state IS degree
        self.needs_neighbor_degrees = False
        self.needs_neighbor_active_counts = True # Needed if dynamic edge logic is used
        # --- Set coloring flags AFTER super().__init__ ---
        self._params.setdefault('use_state_coloring', True)
        self._params.setdefault('color_nodes_by_degree', True) # Default to degree coloring
        self._params.setdefault('color_nodes_by_active_neighbors', False)
        self._params.setdefault('node_colormap', 'prism')
        self._params.setdefault('node_color_norm_vmax', 8.0)
        self._params.setdefault('use_state_coloring_edges', False) # Default False for binary edges
        self._params.setdefault('edge_colormap', 'binary')
        self._params.setdefault('edge_color_norm_vmin', 0.0)
        self._params.setdefault('edge_color_norm_vmax', 1.0)

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    # --- Methods (_compute_new_state, _compute_new_edges, _compute_final_state) remain unchanged ---
    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """
        Compute ELIGIBILITY proxy state (0 or 1) using GoL counts,
        optionally modified by edge effects.
        The final state (degree) is calculated in _compute_final_state.
        """
        # This logic is identical to LaceLifeWithEdges._compute_new_state
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        birth_rules = self.get_param('birth_neighbor_counts', [3], neighborhood=neighborhood)
        survival_rules = self.get_param('survival_neighbor_counts', [2, 3], neighborhood=neighborhood)
        use_edge_effects = self.get_param('use_edge_effects_on_nodes', False, neighborhood=neighborhood)
        survival_boost = self.get_param('survival_boost_factor', 0.1, neighborhood=neighborhood)
        death_boost = self.get_param('death_boost_factor', 0.1, neighborhood=neighborhood)

        num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
        neighborhood.rule_params['_active_neighbor_count'] = num_active_neighbors

        # --- Step 1: Determine Base GoL State ---
        gol_state = 0.0
        if neighborhood.node_state > 0: # Check if node was active (degree > 0)
            if num_active_neighbors in survival_rules: gol_state = 1.0
        else: # Birth check
            if num_active_neighbors in birth_rules: gol_state = 1.0

        if detailed_logging_enabled:
            logger.detail(f"Node {node_idx}: PrevState(Degree)={neighborhood.node_state:.0f}, ActiveN={num_active_neighbors} -> BaseGoLState={gol_state:.0f}") # type: ignore [attr-defined]

        # --- Step 2: Apply Optional Edge Effects ---
        final_eligibility_proxy = gol_state # Start with GoL state
        if use_edge_effects and neighborhood.node_state > 0: # Only apply to nodes that were active
            connected_active_neighbors = sum(1 for idx, state in neighborhood.neighbor_edge_states.items()
                                             if idx >= 0 and state > 1e-6 and neighborhood.neighbor_states[neighborhood.neighbor_indices == idx][0] > 0)

            if gol_state <= 0: # GoL predicts death
                if connected_active_neighbors > 0:
                    if np.random.random() < survival_boost:
                        final_eligibility_proxy = 1.0 # Probabilistic survival boost
                        if detailed_logging_enabled: logger.detail(f"    EdgeEffect: SURVIVAL BOOST applied (ConnectedActiveN={connected_active_neighbors})") # type: ignore [attr-defined]
            elif gol_state > 0: # GoL predicts survival
                if connected_active_neighbors == 0:
                    if np.random.random() < death_boost:
                        final_eligibility_proxy = 0.0 # Probabilistic death penalty
                        if detailed_logging_enabled: logger.detail(f"    EdgeEffect: DEATH PENALTY applied (ConnectedActiveN=0)") # type: ignore [attr-defined]

        if detailed_logging_enabled:
            logger.detail(f"    Final Eligibility Proxy: {final_eligibility_proxy:.1f}") # type: ignore [attr-defined]

        return final_eligibility_proxy

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Propose binary edges based on selected logic (simple or dynamic)."""
        # This logic is identical to LaceLifeWithEdges._compute_new_edges
        new_edges: Dict[Tuple[int, int], float] = {}
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        use_dynamic_logic = self.get_param('use_dynamic_edge_logic', False, neighborhood=neighborhood)
        support_thr = self.get_param('edge_support_neighbor_threshold', 1, neighborhood=neighborhood)

        # Determine eligibility of the current node for the *next* step
        self_is_eligible = self._compute_new_state(neighborhood, detailed_logging_enabled) > 0.5
        # Get current node's active neighbor count from *previous* step (needed for dynamic logic)
        self_prev_active_count = self._count_active_neighbors(neighborhood.neighbor_states)

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_new_edges (Dynamic={use_dynamic_logic}) ---") # type: ignore [attr-defined]
            logger.detail(f"    Self Eligible Next: {self_is_eligible}, Self Prev Active Neighbors: {self_prev_active_count}") # type: ignore [attr-defined]

        if not self_is_eligible:
            if detailed_logging_enabled: logger.detail("    Self ineligible, proposing no edges.") # type: ignore [attr-defined]
            return new_edges # Propose no edges if self will be dead

        # If current node will be alive, check neighbors
        for neighbor_idx, neighbor_state in zip(neighborhood.neighbor_indices, neighborhood.neighbor_states):
            if neighbor_idx < 0: continue

            # Determine neighbor's eligibility for the *next* step (approximation)
            neighbor_is_eligible = neighbor_state > 0 # Check if neighbor was active (degree > 0)

            propose_edge = False # Default: no edge
            decision_reason = "Default (No Edge)"

            if self_is_eligible and neighbor_is_eligible:
                if use_dynamic_logic:
                    # Dynamic Logic: Check support threshold using PREVIOUS active counts
                    neighbor_prev_active_count = 0
                    if neighborhood.neighbor_active_counts is not None:
                         neighbor_prev_active_count = neighborhood.neighbor_active_counts.get(neighbor_idx, 0)
                    else: logger.warning(f"Node {node_idx}: neighbor_active_counts missing for neighbor {neighbor_idx}")

                    if self_prev_active_count >= support_thr and neighbor_prev_active_count >= support_thr:
                        propose_edge = True
                        decision_reason = f"Propose/Maintain (Dynamic: Eligible & Support Met >= {support_thr})"
                    else:
                        decision_reason = f"Break/No Form (Dynamic: Support Low < {support_thr})"
                else:
                    # Simple Logic: Just connect if both are eligible
                    propose_edge = True
                    decision_reason = "Propose/Maintain (Simple: Both Eligible)"

            else: # One or both ineligible
                decision_reason = "Break/No Form (One Ineligible)"

            if detailed_logging_enabled:
                logger.detail(f"  Neighbor {neighbor_idx}: Eligible Next(Proxy)={neighbor_is_eligible}. Decision: {decision_reason}") # type: ignore [attr-defined]

            if propose_edge:
                edge = (node_idx, neighbor_idx) if node_idx < neighbor_idx else (neighbor_idx, node_idx)
                new_edges[edge] = 1.0 # Binary edges

        if detailed_logging_enabled:
            logger.detail(f"    Proposed Edges Count: {len(new_edges)}") # type: ignore [attr-defined]
        return new_edges

    def _compute_final_state(self,
                             node_idx: int,
                             current_proxy_state: float, # Eligibility/state from current step's computation
                             final_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]], # Edges calculated for *current* step
                             dimensions: Tuple[int,...],
                             # --- Accept all arguments even if unused ---
                             previous_node_states: npt.NDArray[np.float64],
                             previous_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
                             previous_edge_states: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float],
                             previous_node_degrees: Optional[npt.NDArray[np.int32]],
                             previous_active_neighbors: Optional[npt.NDArray[np.int32]],
                             eligibility_proxies: Optional[np.ndarray] = None,
                             detailed_logging_enabled: bool = False
                             ) -> float:
        """
        Calculates the final node state (degree) based on eligibility and final edge count.
        """
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING

        if detailed_logging_enabled:
            logger.detail(f"--- Node {node_idx}: {self.name} _compute_final_state ---") # type: ignore [attr-defined]
            logger.detail(f"    Proxy State (Eligibility): {current_proxy_state:.1f}") # type: ignore [attr-defined]

        # If proxy state indicates ineligible (0.0), final state is 0
        if current_proxy_state < 0.5:
             if detailed_logging_enabled: logger.detail(f"    Node was ineligible based on prev step conditions, final state is 0.") # type: ignore [attr-defined]
             return 0.0

        # If eligible (proxy was 1.0), calculate final degree based on the FINAL edges for this step
        node_coords = tuple(_unravel_index(node_idx, dimensions))
        final_degree = sum(1 for edge in final_edges if node_coords in edge)

        if detailed_logging_enabled:
            logger.detail(f"    Node was eligible. Final Calculated Degree: {final_degree}") # type: ignore [attr-defined]
            logger.detail(f"    Setting Final State to: {float(final_degree):.1f}") # type: ignore [attr-defined]

        # The state IS the degree if eligible
        return float(final_degree)
      
class LifeWithColor(Rule):
    """
    Standard Game of Life (B3/S23) node logic. Edges are not used.
    Node color represents the number of active neighbors in the previous step.
    Requires post-update step to set node state = active neighbor count for visualization.

    Node Logic (Binary State 0/1 for computation):
    - Follows standard Conway's Game of Life rules based on the count of active neighbors (state > 0) in the previous step.
    - Birth: Inactive node becomes active if active neighbor count is in 'birth_neighbor_counts' (Default: [3]).
    - Survival: Active node remains active if active neighbor count is in 'survival_neighbor_counts' (Default: [2, 3]).
    - Death: Otherwise, node becomes or remains inactive (state 0).

    Edge Logic: None.

    Visualization:
    - Node color is determined by the number of active neighbors the node had in the *previous* step.
    - Node state stored in the grid array *is* this active neighbor count, but the rule logic uses binary interpretation (count > 0 means active for next step's neighbor count).
    """
    # --- ADDED: Exclude unused base parameters from editor ---
    EXCLUDE_EDITOR_PARAMS: ClassVar[Set[str]] = {
        'edge_initialization', 'connect_probability', 'min_edge_weight', 'max_edge_weight',
        'tiebreaker_type', 'node_history_depth', 'state_rule_table', 'edge_rule_table',
        'use_state_coloring_edges', 'color_edges_by_neighbor_degree',
        'color_edges_by_neighbor_active_neighbors', 'edge_colormap',
        'edge_color_norm_vmin', 'edge_color_norm_vmax',
        'color_nodes_by_degree', # Degree coloring not applicable
        'node_history_depth'
    }
    # ---
    produces_binary_edges: ClassVar[bool] = True
    node_state_type: ClassVar[StateType] = StateType.BINARY
    edge_state_type: ClassVar[StateType] = StateType.BINARY
    min_node_state: ClassVar[float] = 0.0
    max_node_state: ClassVar[float] = 1.0
    min_edge_state: ClassVar[float] = 0.0
    max_edge_state: ClassVar[float] = 1.0

    PARAMETER_METADATA = {
        # --- State Update ---
        "birth_neighbor_counts": {
            'type': list, 'element_type': int, 'default': [3],
            'description': "List of exact active neighbor counts required for an inactive cell to become active (birth). Example: [3]",
            "parameter_group": "State Update (GoL Counts)"
        },
        "survival_neighbor_counts": {
            'type': list, 'element_type': int, 'default': [2, 3],
            'description': "List of exact active neighbor counts required for an active cell to remain active (survival). Example: [2, 3]",
            "parameter_group": "State Update (GoL Counts)"
        },
        # --- Core Parameters ---
        "neighborhood_type": {
            'type': str, 'default': "MOORE",
            'allowed_values': ["MOORE", "VON_NEUMANN"],
            'description': "Neighborhood definition (Adjacent cells).",
            "parameter_group": "Core"
        },
        "dimension_type": {
            'type': str, 'default': "TWO_D",
            'allowed_values': ["TWO_D", "THREE_D"],
            'description': "Grid dimension.",
            "parameter_group": "Core"
        },
        'grid_boundary': {
            'type': str, 'default': 'wrap',
            'allowed_values': ['bounded', 'wrap'],
            'description': 'Grid boundary behavior.',
            "parameter_group": "Core"
        },
        # --- Initialization Parameters ---
        "initial_density": {
            "type": float, 'default': GlobalSettings.Simulation.INITIAL_NODE_DENSITY,
            "description": "Initial density of active nodes (0.0 to 1.0) when using 'Random' initial conditions.",
            "min": 0.0, "max": 1.0,
            "parameter_group": "Initialization"
        },
        "initial_conditions": {
            "type": str, 'default': "Random",
            "description": "Method for setting the initial grid state.",
            "allowed_values": ['Random', 'Glider Pattern', 'Gosper Glider Gun Pattern', '2D - Circle', '2D - Square', '3D - Sphere', '3D - Cube', 'ShapeShifting', '2D - Square Tessellation', '2D - Triangle Tessellation'],
            "parameter_group": "Initialization"
        },
        # --- Visualization Parameters ---
        "use_state_coloring": {
            "type": bool, "description": "Color nodes based on state (active neighbor count).",
            "default": True, "parameter_group": "Visualization: Nodes" # Default True
        },
        "color_nodes_by_active_neighbors": {
            "type": bool, "description": "Color nodes based on active neighbor count (previous step). (Implicitly True if use_state_coloring is True)",
            "default": True, "parameter_group": "Visualization: Nodes" # Default True
        },
        "node_colormap": {
            "type": str, "description": "Colormap for node states (neighbor counts).",
            "default": "prism", "parameter_group": "Visualization: Nodes", # Changed default
            "allowed_values": ["(None)"] + _standard_colormaps
        },
        "node_color_norm_vmin": {
            "type": float, "description": "Min value for node color normalization. Examples: 0.0 (for binary/degree/0-1 state), -1.0 (for -1 to 1 state)..",
            "default": 0.0, "parameter_group": "Visualization: Nodes"
        },
        "node_color_norm_vmax": { "type": float, "description": "Max value for node color normalization. Examples: 1.0 (for binary/0-1/-1 to 1 state), 8.0 (for degree/neighbor count in Moore 2D)..", "default": 8.0, "parameter_group": "Visualization: Nodes"},
        }

    def __init__(self, metadata: 'RuleMetadata'):
        metadata.name = "Life with Color"
        metadata.description = "Standard GoL B/S node logic (computes binary eligibility 0/1). Edges not used. Node state stored is binary eligibility, but node color represents the number of active neighbors in the *previous* step (if coloring enabled)."
        metadata.category = "Life-Like"
        metadata.tags = ["Life", "Conway", "B3/S23", "Standard", "Color", "Neighbor Count"]
        metadata.neighborhood_compatibility = ["MOORE", "VON_NEUMANN"]
        metadata.dimension_compatibility = ["TWO_D", "THREE_D"]
        metadata.allow_rule_tables = False # No tables
        super().__init__(metadata)
        self.name = "Life with Color"
        # Set fixed core param AFTER super().__init__
        self._params['edge_initialization'] = "NONE"
        # Set visualization flags AFTER super().__init__
        self._params['use_state_coloring'] = True
        self._params['color_nodes_by_degree'] = False
        self._params['color_nodes_by_active_neighbors'] = True
        self._params['node_colormap'] = 'prism' # Ensure default
        self._params['node_color_norm_vmax'] = 8.0 # Ensure default

    def initialize_grid_state(self, grid: 'Grid'):
        """
        Handles rule-specific grid state initialization.
        For most standard rules, the primary initialization (Random, Empty, Patterns)
        is handled externally by InitialConditionManager based on the
        'initial_conditions' parameter selected for the rule.
        This method can be overridden by subclasses if they require truly unique
        initialization logic beyond the standard patterns managed externally,
        or if they need to perform actions *after* the InitialConditionManager runs
        (though the manager handles degree-based state setting).
        """
        # Default implementation: Assumes external handling via InitialConditionManager.
        logger.debug(f"Rule '{self.name}': initialize_grid_state called. Relying on external InitialConditionManager based on 'initial_conditions' param: '{self.get_param('initial_conditions', 'Unknown')}'.")
        pass # No specific action needed here for most rules.

    def _compute_new_state(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool) -> float:
        """
        Compute ELIGIBILITY proxy state (0 or 1) using standard GoL B/S rules.
        The final state (neighbor count) is calculated in _compute_final_state.
        """
        logger = logging.getLogger(__name__)
        detailed_logging_enabled = LogSettings.Performance.ENABLE_DETAILED_LOGGING
        node_idx = neighborhood.node_index

        # Fetch parameters
        birth_rules = self.get_param('birth_neighbor_counts', [3], neighborhood=neighborhood)
        survival_rules = self.get_param('survival_neighbor_counts', [2, 3], neighborhood=neighborhood)

        num_active_neighbors = self._count_active_neighbors(neighborhood.neighbor_states)
        # --- Store active neighbor count for potential use in _compute_final_state ---
        # Although not strictly needed now, keep for potential future use or consistency
        neighborhood.rule_params['_active_neighbor_count'] = num_active_neighbors
        # ---

        eligibility_proxy = 0.0
        decision_reason = "Default (Death/Inactive)"
        # --- MODIFIED: Check node_state > 0 (since state might be neighbor count) ---
        if neighborhood.node_state > 0: # Active node (check survival)
        # ---
            if num_active_neighbors in survival_rules:
                eligibility_proxy = 1.0
                decision_reason = f"Survival (Neighbors={num_active_neighbors} in {survival_rules})"
            else:
                decision_reason = f"Death (Neighbors={num_active_neighbors} not in {survival_rules})"
        else: # Inactive node (check birth)
            if num_active_neighbors in birth_rules:
                eligibility_proxy = 1.0
                decision_reason = f"Birth (Neighbors={num_active_neighbors} in {birth_rules})"
            else:
                 decision_reason = f"Remain Dead (Neighbors={num_active_neighbors} not in {birth_rules})"

        if detailed_logging_enabled:
            logger.detail(f"Node {node_idx}: {self.name} _compute_new_state (Eligibility Proxy)") # type: ignore [attr-defined]
            logger.detail(f"    Prev State: {neighborhood.node_state:.1f}, ActiveN: {num_active_neighbors}") # type: ignore [attr-defined]
            logger.detail(f"    Decision: {decision_reason} -> Eligibility Proxy: {eligibility_proxy:.1f}") # type: ignore [attr-defined]

        return eligibility_proxy

    def _compute_new_edges(self, neighborhood: NeighborhoodData, detailed_logging_enabled: bool)-> Dict[Tuple[int, int], float]:
        """Life with Color does not use or modify edges."""
        return {}

    def _compute_final_state(self,
                             node_idx: int,
                             current_proxy_state: float, # Eligibility/state from current step's computation
                             final_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]], # Edges calculated for *current* step
                             dimensions: Tuple[int,...],
                             # --- Accept all arguments even if unused ---
                             previous_node_states: npt.NDArray[np.float64],
                             previous_edges: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]],
                             previous_edge_states: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float],
                             previous_node_degrees: Optional[npt.NDArray[np.int32]],
                             previous_active_neighbors: Optional[npt.NDArray[np.int32]],
                             eligibility_proxies: Optional[np.ndarray] = None,
                             detailed_logging_enabled: bool = False
                             ) -> float:
        """
        Returns the eligibility proxy state (0 or 1). The actual coloring based on
        previous neighbor count happens in the visualizer.
        """
        # The final state stored in the grid is just the eligibility proxy.
        return current_proxy_state


################################################
#                 RULE LIBRARY                 #
################################################

class RuleLibrary:
    """Comprehensive library of balanced cellular automaton rules"""

    # --- MODIFIED: Store class types directly instead of lambdas ---
    # --- MODIFIED: Use names with spaces as keys to match rule names ---
    RULES: Dict[str, Type[Rule]] = {
        # Basic Rules
        "Realm of Lace": RealmOfLace,
        "Realm of Lace Unified": RealmOfLaceUnified,
        "Game of Life": GameOfLife,
        "Life with Color": LifeWithColor, # Added space
        "Lace Life with Edges": LaceLifeWithEdges,
        "Colored Life with Edges": ColoredLifeWithEdges, # Added space
        "Life with Edge Growth": LifeWithEdgeGrowth,
        "Life with Connectivity Weighting": LifeWithConnectivityWeighting,
        "Life with Dynamic Edges": LifeWithDynamicEdges, # Added space
        "Edge Feedback Life": EdgeFeedbackLife,
        "Edge Degree Life": EdgeDegreeLife,
        "Network Topology Life": NetworkTopologyLife,
        "Topological Rule": TopologicalRule,
        "Modular Life": ModularLife,
        "GoL Nodes GoL Edges": GoLNodesGoLEdges,
        "Configurable Continuous Life": ConfigurableContinuousLife,
        "Bidirectional Feedback Life": BidirectionalFeedbackLife,
        "Configurable Life with Edges": ConfigurableLifeWithEdges,
        "Life with Continuous Edges": LifeWithContinuousEdges,
        "Weighted Edge Influence Life": WeightedEdgeInfluenceLife,
        "Resource Competition Life": ResourceCompetitionLife,
        "Life with Weighted Edges": LifeWithWeightedEdges,
        "Multi-State Life with Edges": MultiStateLifeWithEdges,
        "2D CA Master Rule": TwoDCAMasterRule,
        "Master Configurable Rule": MasterConfigurableRule,
        "Angular Edge Rule": AngularEdgeRule
        # Add other rules here using ClassName directly
    }
    # --- END MODIFIED ---

    @classmethod
    def get_rule_names(cls) -> List[str]:
        """Get list of available rule names"""
        # Get rule names from the JSON file via RuleLibraryManager
        try:
            # rule_data = RuleLibraryManager.get_rule("") # Get all rules - NO, this is wrong
            # return list(RuleLibraryManager.get_instance().rules.keys()) # This is also wrong
            return list(RuleLibraryManager.get_instance().rules.keys()) # Get rule names from the instance
        except:
            return list(cls.RULES.keys())

    @classmethod
    def get_rule_categories(cls) -> Dict[str, List[str]]:
        """Get dictionary of rule categories and their rules, ensuring normalization, adding Favorites, and excluding Orphaned."""

        categories: Dict[str, List[str]] = defaultdict(list)
        favorites: List[str] = []
        logger.debug("--- Entering RuleLibrary.get_rule_categories (R5 Orphan Exclude) ---")
        try:
            manager = RuleLibraryManager.get_instance()
            all_rule_names = manager.rules.keys()
            logger.debug(f"  Processing {len(all_rule_names)} rule names from manager.")

            for rule_name in all_rule_names:
                try:
                    rule_data = manager.get_rule(rule_name)
                    raw_category = rule_data.get('category', 'Unknown')
                    is_favorite = rule_data.get('favorite', False)

                    # Normalize category string
                    normalized_category = raw_category.strip().title()

                    # --- ADDED: Skip Orphaned Rules ---
                    if normalized_category.startswith("Orphaned Rule"):
                        logger.debug(f"    Skipping orphaned rule: '{rule_name}' (Category: '{normalized_category}')")
                        continue
                    # --- END ADDED ---

                    if raw_category != normalized_category:
                        logger.debug(f"    Rule: '{rule_name}', Raw Category: '{raw_category}', Normalized Category: '{normalized_category}'")
                    else:
                        logger.debug(f"    Rule: '{rule_name}', Category: '{normalized_category}' (No normalization needed)")

                    categories[normalized_category].append(rule_name)

                    if is_favorite:
                        favorites.append(rule_name)
                        logger.debug(f"    Rule: '{rule_name}' marked as favorite.")

                except Exception as e:
                    logger.error(f"  Error processing rule '{rule_name}' in get_rule_categories: {e}")
                    categories["Error"].append(rule_name)

            # Sort names within each category alphabetically
            for name_list in categories.values():
                name_list.sort()

            if favorites:
                favorites.sort()
                categories["Favorites"] = favorites
                logger.debug(f"  Added 'Favorites' category with {len(favorites)} rules.")

            sorted_categories = dict(sorted(categories.items()))
            logger.debug(f"  Returning {len(sorted_categories)} non-orphaned categories: {list(sorted_categories.keys())}")
            logger.debug("--- Exiting RuleLibrary.get_rule_categories ---")
            return sorted_categories

        except Exception as e:
            logger.error(f"Fatal error in get_rule_categories: {e}")
            logger.error(traceback.format_exc())
            return {"Error": []}

    @classmethod
    def get_rules_in_category(cls, category: str) -> List[str]:
        """Get list of rules in a specific category, normalizing category names and excluding Orphaned."""

        rules_in_category = []
        normalized_target_category = category.strip().title()
        logger.debug(f"get_rules_in_category: Looking for rules in normalized category '{normalized_target_category}'")

        try:
            manager = RuleLibraryManager.get_instance()
            all_rules = manager.rules
            logger.debug(f"get_rules_in_category: Checking {len(all_rules)} rules from manager.")

            for rule_name, rule_data in all_rules.items():
                raw_rule_category = rule_data.get('category', 'Unknown')
                normalized_rule_category = raw_rule_category.strip().title()

                # --- ADDED: Skip Orphaned Rules ---
                if normalized_rule_category.startswith("Orphaned Rule"):
                    continue
                # --- END ADDED ---

                if normalized_rule_category == normalized_target_category:
                    rules_in_category.append(rule_name)
                    logger.debug(f"    MATCH! Added '{rule_name}' to rules_in_category.")

            logger.debug(f"get_rules_in_category: Found {len(rules_in_category)} rules in category '{normalized_target_category}': {rules_in_category}")
            return sorted(rules_in_category)
        except Exception as e:
             logger.error(f"Error in get_rules_in_category: {e}")
             logger.error(traceback.format_exc())
             return []

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

            # --- MODIFIED: Get class directly ---
            rule_class = cls.RULES[rule_name]
            # ---

            # Create a temporary instance of the rule to access its metadata
            metadata = RuleMetadata(**rule_data)
            rule = rule_class(metadata) # Instantiate using the class

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

    @classmethod
    def create_rule(cls, rule_name: str, metadata: 'RuleMetadata') -> 'Rule':
        """Create rule instance by name"""
        try:
            # --- MODIFIED: Get class directly and instantiate ---
            if rule_name in cls.RULES:
                rule_class = cls.RULES[rule_name]
                return rule_class(metadata) # Instantiate directly
            # ---
            else:
                # Dynamic loading attempt (keep as fallback, though less likely needed now)
                try:
                    rule_data = RuleLibraryManager.get_rule(rule_name)
                    rule_type = rule_data['type']
                    rule_class = globals()[rule_type]
                    return rule_class(metadata)
                except Exception as e:
                    logger.error(f"Error creating dynamic rule {rule_name}: {e}")
                    raise ValueError(f"Could not create rule {rule_name}: {e}")
        except Exception as e:
            logger.error(f"Error creating rule {rule_name}: {e}")
            raise ValueError(f"Could not create rule {rule_name}: {e}")
                                 
class RuleLibraryManager:
    """Manages rule library loading, saving, and updates"""
    _instance: Optional['RuleLibraryManager'] = None
    _rules_cache: Dict[str, Dict[str, Any]] = {}  # Add cache to prevent recursion
    _initialized: bool = False
    _is_validating = False  # ADDED: Class-level flag

    def __init__(self, library_path: Optional[str] = None, app_paths: Optional[Dict[str, str]] = None):

        if not RuleLibraryManager._initialized:
            if app_paths is None:
                logger.warning("RuleLibraryManager initialized without app_paths! Attempting fallback.")
                try:
                    global APP_PATHS
                    self.app_paths = APP_PATHS
                except NameError:
                     logger.error("APP_PATHS not defined globally. Cannot determine paths.")
                     script_dir = os.path.dirname(os.path.abspath(__file__))
                     self.app_paths = {
                         'config_rules': os.path.join(script_dir, 'config', 'rules'),
                         'config_rules_backups': os.path.join(script_dir, 'config', 'rules_backups')
                     }
                     os.makedirs(self.app_paths['config_rules'], exist_ok=True)
                     os.makedirs(self.app_paths['config_rules_backups'], exist_ok=True)
            else:
                self.app_paths = app_paths

            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(self.app_paths['config_rules'], 'rules.json')

            self.library_path = library_path or default_path
            print(f"Rule library path: {self.library_path}")

            self.rules: Dict[str, Dict[str, Any]] = {}
            self.rule_metadata: Dict[str, RuleMetadata] = {}

            # --- Ensure base Rule metadata is populated ONCE if needed, but rely on validation fetch ---
            if not Rule.PARAMETER_METADATA:
                 Rule._populate_base_metadata()
            # ---

            try:
                self.load_library()
                if not self.rules:
                    logger.warning("Rules dictionary is empty after loading, creating full default library")
                    library_data = self._create_default_library()
                    self.rules = {rule['name']: rule for rule in library_data['rules']}
                    self.save_library()
            except Exception as e:
                logger.error(f"Error loading library: {e}")
                if not self.rules:
                    logger.warning("Rules dictionary is empty after load attempt, creating full default library")
                    library_data = self._create_default_library()
                    self.rules = {rule['name']: rule for rule in library_data['rules']}
                    self.save_library()

            RuleLibraryManager._initialized = True

    def _initialize_rule_metadata(self):
        """Initialize metadata for all default rules"""
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Use the RuleLibrary class to create metadata
        for rule_name in RuleLibrary.RULES.keys():
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

    def _create_rule_library_backup(self, filepath):
        """Create a backup of the rule library file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(APP_PATHS['config_rules_backups'])  # Use APP_PATHS
        os.makedirs(backup_dir, exist_ok=True)  # Ensure directory exists
        backup_path = os.path.join(backup_dir, f"rules_backup_{timestamp}.json")
        try:
            shutil.copyfile(filepath, backup_path)
            logger.info(f"Created backup of rule library at {backup_path}")
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            # Proceed without backup, but log the error

    def _check_for_missing_default_rules(self, library_data):
        """Ensure all default rules from RuleLibrary.RULES are present in the loaded library_data."""
        log_prefix = "_check_for_missing_default_rules: "
        logger.debug(f"{log_prefix}Entering.")

        default_rule_names = set(RuleLibrary.RULES.keys())
        # --- CORRECTED: Get names from the passed library_data ---
        if 'rules' not in library_data or not isinstance(library_data['rules'], list):
             logger.error(f"{log_prefix}Invalid library_data structure: 'rules' key missing or not a list.")
             current_rule_names_in_data = set() # Assume no rules loaded
        else:
             # Get names from the list of rule dictionaries within library_data
             current_rule_names_in_data = {rule.get('name') for rule in library_data['rules'] if isinstance(rule, dict) and rule.get('name')}
             logger.debug(f"{log_prefix}Found {len(current_rule_names_in_data)} rules in loaded library_data: {current_rule_names_in_data}")
        # --- END CORRECTION ---

        missing_rules = default_rule_names - current_rule_names_in_data # Compare against data from file
        rules_needing_parameter_update = []
        rules_modified_in_check = False # Flag if we modify library_data

        if missing_rules:
            logger.info(f"{log_prefix}Found {len(missing_rules)} missing default rules in loaded data: {missing_rules}")
            rules_modified_in_check = True
            for rule_name in missing_rules:
                try:
                    rule_class = RuleLibrary.RULES[rule_name]
                    default_metadata = DefaultRuleMetadata.get_default_metadata(
                        rule_type=rule_class.__name__, rule_name=rule_name,
                        category="General Purpose", description="A Lace link automata rule."
                    )
                    temp_rule_for_meta = rule_class(default_metadata)

                    new_rule_data = {
                        'name': rule_name, 'type': rule_class.__name__,
                        'position': default_metadata.position, 'category': default_metadata.category,
                        'author': default_metadata.author, 'url': default_metadata.url,
                        'email': default_metadata.email, 'date_created': default_metadata.date_created,
                        'date_modified': datetime.now().strftime("%Y-%m-%d"), 'version': default_metadata.version,
                        'description': default_metadata.description, 'tags': default_metadata.tags + [rule_name, "default"],
                        'dimension_compatibility': default_metadata.dimension_compatibility,
                        'neighborhood_compatibility': default_metadata.neighborhood_compatibility,
                        'parent_rule': default_metadata.parent_rule, 'rating': default_metadata.rating,
                        'notes': default_metadata.notes, 'allowed_initial_conditions': default_metadata.allowed_initial_conditions,
                        'allow_rule_tables': default_metadata.allow_rule_tables, 'favorite': False,
                        'params': {}
                    }

                    # [ Corrected Parameter Population - Unchanged from R22 ]
                    base_metadata = getattr(Rule, 'PARAMETER_METADATA', {})
                    subclass_metadata = getattr(temp_rule_for_meta, 'PARAMETER_METADATA', {})
                    merged_metadata = {**base_metadata, **subclass_metadata}
                    exclude_set = getattr(rule_class, 'EXCLUDE_EDITOR_PARAMS', set())
                    final_expected_param_names = set(merged_metadata.keys()) - exclude_set
                    for param_name in final_expected_param_names:
                        param_info = merged_metadata.get(param_name)
                        if param_info and 'default' in param_info:
                            new_rule_data['params'][param_name] = param_info['default']

                    # Add to the library_data['rules'] list being processed
                    library_data['rules'].append(new_rule_data)
                    logger.debug(f"{log_prefix}Added missing default rule '{rule_name}' to library_data list.")
                except Exception as add_err:
                    logger.error(f"{log_prefix}Error adding missing rule '{rule_name}': {add_err}")
                    logger.error(traceback.format_exc())
                    continue

        # --- Check for missing parameters in EXISTING rules ---
        # --- MODIFIED: Iterate through library_data['rules'] ---
        for rule_data in library_data['rules']:
        # ---
            rule_name = rule_data.get('name')
            # --- MODIFIED: Skip rules just added OR unnamed rules ---
            if not rule_name or rule_name in missing_rules:
                continue
            # ---

            if 'params' not in rule_data: rule_data['params'] = {}
            rule_type = rule_data.get('type', '')
            rule_class_existing = None
            try:
                rule_class_existing = globals()[rule_type]
            except KeyError:
                if rule_name in RuleLibrary.RULES:
                    try:
                        temp_metadata = RuleMetadata(**{k: v for k, v in rule_data.items() if k != 'params' and k != '_ignored_params'})
                        temp_metadata.position = rule_data.get('position', 1)
                        temp_rule = RuleLibrary.RULES[rule_name](temp_metadata)
                        rule_class_existing = temp_rule.__class__
                        if rule_data['type'] != rule_class_existing.__name__:
                            rule_data['type'] = rule_class_existing.__name__; logger.info(f"Updated rule type for {rule_name}")
                    except Exception as factory_err: logger.warning(f"Could not check params for rule {rule_name} (factory error): {factory_err}"); continue
                else: logger.warning(f"Could not check params for rule {rule_name} (class not found)"); continue

            if rule_class_existing:
                base_meta = getattr(Rule, 'PARAMETER_METADATA', {})
                sub_meta = getattr(rule_class_existing, 'PARAMETER_METADATA', {})
                merged_meta = {**base_meta, **sub_meta}
                exclude_set_existing = getattr(rule_class_existing, 'EXCLUDE_EDITOR_PARAMS', set())
                final_expected_existing = set(merged_meta.keys()) - exclude_set_existing

                missing_params_found = False
                for param_name in final_expected_existing:
                    param_info = merged_meta.get(param_name)
                    if param_name not in rule_data['params'] and param_info and 'default' in param_info:
                        rule_data['params'][param_name] = param_info['default']
                        missing_params_found = True
                        logger.warning(f"Rule {rule_name}: Added missing param '{param_name}' with default: {param_info['default']}")

                if missing_params_found:
                    rules_needing_parameter_update.append(rule_name)
                    rules_modified_in_check = True

        # --- Save only if changes were made ---
        if rules_modified_in_check:
            logger.info(f"{log_prefix}Modifications made (missing rules added or params updated). Saving changes.")
            global _CREATING_DEFAULT_LIBRARY
            _CREATING_DEFAULT_LIBRARY = True
            try:
                # --- MODIFIED: Save the modified library_data by updating self.rules first ---
                # Update self.rules from the potentially modified library_data list
                self.rules = {rule['name']: rule for rule in library_data['rules'] if rule.get('name')}
                self.save_library()
                # ---
                logger.info(f"{log_prefix}Saved changes after checking for missing rules/params.")
            finally:
                _CREATING_DEFAULT_LIBRARY = False
        else:
            logger.debug(f"{log_prefix}No missing rules or parameters found in existing rules.")
        logger.debug(f"{log_prefix}Exiting.")

    def _load_rules_from_library_data(self, library_data):
        """Load rules from the provided library data using a loop for robustness."""
        log_prefix = "_load_rules_from_library_data: "
        logger.debug(f"{log_prefix}Loading rules from provided data structure.")
        loaded_rules_dict: Dict[str, Dict[str, Any]] = {}
        duplicate_names = []
        rules_list = library_data.get('rules', [])

        if not isinstance(rules_list, list):
            logger.error(f"{log_prefix}Invalid format: 'rules' key is not a list.")
            self.rules = {} # Set to empty on error
            return

        logger.debug(f"{log_prefix}Processing {len(rules_list)} rule entries from data.")
        for i, rule_data in enumerate(rules_list):
            # --- ADDED: Log the raw rule_data being processed ---
            # logger.debug(f"{log_prefix}  Processing Index {i}: Raw rule_data keys: {list(rule_data.keys()) if isinstance(rule_data, dict) else 'Not a dict'}")
            # ---
            if not isinstance(rule_data, dict):
                logger.warning(f"{log_prefix}Skipping item at index {i}: not a dictionary.")
                continue
            rule_name = rule_data.get('name')
            if not rule_name or not isinstance(rule_name, str):
                logger.warning(f"{log_prefix}Skipping rule at index {i}: missing or invalid 'name'. Found name: '{rule_name}' (Type: {type(rule_name)}).")
                continue

            logger.debug(f"{log_prefix}  Processing rule '{rule_name}' (Index {i}).")

            if rule_name in loaded_rules_dict:
                logger.warning(f"{log_prefix}Duplicate rule name found: '{rule_name}'. Overwriting previous entry.")
                duplicate_names.append(rule_name)

            if 'params' not in rule_data or not isinstance(rule_data.get('params'), dict):
                logger.warning(f"{log_prefix}Rule '{rule_name}' missing or invalid 'params' dict. Initializing empty.")
                rule_data['params'] = {}

            # --- ADDED: Log BEFORE assignment ---
            # logger.debug(f"{log_prefix}  Attempting to assign rule '{rule_name}' to loaded_rules_dict.")
            # ---
            loaded_rules_dict[rule_name] = rule_data # Add or overwrite
            # logger.debug(f"{log_prefix}  Successfully added/updated '{rule_name}' in loaded_rules_dict.")

        self.rules = loaded_rules_dict # Assign the processed dictionary
        num_loaded = len(self.rules)
        logger.info(f"{log_prefix}Successfully loaded {num_loaded} rules into self.rules.")
        logger.debug(f"{log_prefix}Final keys in self.rules: {list(self.rules.keys())}")
        if duplicate_names:
            logger.warning(f"{log_prefix}Encountered {len(duplicate_names)} duplicate rule names: {duplicate_names}")
        if num_loaded != len(rules_list):
             logger.warning(f"{log_prefix}Number of loaded rules ({num_loaded}) differs from input list length ({len(rules_list)}). Check logs for skipped rules.")

    def get_favorite_rule_names(self) -> List[str]:
        """Returns a sorted list of names for rules marked as favorite."""
        favorite_names = []
        # --- MODIFIED: Access self.rules directly ---
        if not hasattr(self, 'rules') or not self.rules:
             logger.error("RuleLibraryManager.rules not initialized when getting favorites.")
             return [] # Return empty list if rules aren't loaded

        for rule_name, rule_data in self.rules.items():
            if rule_data.get('favorite', False): # Check the 'favorite' flag in the rule data
                favorite_names.append(rule_name)
        # --- END MODIFIED ---
        return sorted(favorite_names) # Return the sorted list

    @classmethod
    def get_instance(cls, app_paths: Optional[Dict[str, str]] = None) -> 'RuleLibraryManager':
        if cls._instance is None:
            # Store app_paths in a class variable so it's available even before initialization
            cls._app_paths = app_paths

            # Create the instance with app_paths
            cls._instance = RuleLibraryManager(app_paths=app_paths)

            # Verify the rules were loaded correctly
            if not hasattr(cls._instance, 'rules') or not cls._instance.rules:
                logger.warning("Rules not loaded correctly in get_instance, attempting to reload")
                cls._instance.reload_library()

        # If app_paths is provided and different from stored app_paths, update it
        elif app_paths and cls._app_paths != app_paths:
            cls._app_paths = app_paths
            # Update the library path
            if hasattr(cls._instance, 'library_path'):
                cls._instance.library_path = os.path.join(app_paths['config_rules'], 'rules.json')
                logger.info(f"Updated library path to {cls._instance.library_path}")
                # Reload the library with the new path
                cls._instance.reload_library()

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

                # Preserve existing tags and add new ones if they don't exist
                existing_tags = self.rules[rule_name].get('tags', [])
                new_tags = rule_data.get('tags', [])

                # Add new tags only if they don't already exist
                for tag in new_tags:
                    if tag not in existing_tags:
                        existing_tags.append(tag)

                # Update the tags in the rule data
                rule_data['tags'] = existing_tags

                self.rules[rule_name] = rule_data
            else:
                logger.info(f"Adding new rule: {rule_name}")

                # Add default tags if it's a default rule
                if rule_data.get('default', False):
                    if 'tags' not in rule_data:
                        rule_data['tags'] = []
                    if rule_data['name'] not in rule_data['tags']:
                        rule_data['tags'].append(rule_data['name'])
                    if 'default' not in rule_data['tags']:
                        rule_data['tags'].append('default')

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
                return copy.deepcopy(cls._rules_cache[rule_name])

            # CRITICAL FIX: Check if we're in the process of creating a default library
            global _CREATING_DEFAULT_LIBRARY
            if _CREATING_DEFAULT_LIBRARY:
                # Return a minimal rule structure during default library creation
                logger.debug(f"Returning minimal rule structure for {rule_name} during default library creation")
                return {
                    'name': rule_name,
                    'type': rule_name,
                    'category': 'Uncategorized',
                    'description': 'A network cellular automata rule.',
                    'params': {}
                }

            # If not in cache, load from instance
            instance = cls.get_instance()

            # CRITICAL FIX: Check if instance has rules attribute or if rules is empty
            if not hasattr(instance, 'rules') or not instance.rules:
                logger.error(f"RuleLibraryManager instance has no 'rules' attribute or rules is empty. Creating full default library.")

                # CRITICAL FIX: Ensure library_path is set
                if not hasattr(instance, 'library_path') or instance.library_path is None:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    instance.library_path = os.path.join(script_dir, 'rules.json')
                    logger.info(f"Set library_path to {instance.library_path}")

                # Create a FULL default library with all rules
                library_data = instance._create_default_library()
                instance.rules = {rule['name']: rule for rule in library_data['rules']}

                try:
                    instance.save_library()
                except Exception as save_error:
                    logger.error(f"Error saving default library: {save_error}")

                # Check if the requested rule is now available
                if rule_name in instance.rules:
                    rule_data = copy.deepcopy(instance.rules[rule_name])
                    if 'params' not in rule_data:
                        rule_data['params'] = {}
                    cls._rules_cache[rule_name] = copy.deepcopy(rule_data)
                    return copy.deepcopy(rule_data)
                else:
                    # Return a default rule if the requested one isn't available
                    default_rule = next(iter(instance.rules.values())) if instance.rules else None
                    if default_rule:
                        logger.warning(f"Rule '{rule_name}' not found, returning default rule '{default_rule['name']}'")
                        cls._rules_cache[rule_name] = copy.deepcopy(default_rule)
                        return copy.deepcopy(default_rule)
                    else:
                        raise ValueError(f"No rules available in library")

            # Normal flow - check if rule exists in instance
            if rule_name in instance.rules:
                # Get a DEEP copy of the rule data
                rule_data = copy.deepcopy(instance.rules[rule_name])

                # Ensure 'params' key exists, even if it's empty
                if 'params' not in rule_data:
                    rule_data['params'] = {}

                # Update cache and return a DEEP copy
                cls._rules_cache[rule_name] = copy.deepcopy(rule_data) # Store a deep copy in the cache
                logger.debug(f"Rule data loaded for {rule_name} from instance: {rule_data}")
                return copy.deepcopy(rule_data) # Return a deep copy

            logger.error(f"Rule '{rule_name}' not found in instance.rules. Available rules: {list(instance.rules.keys())}")
            raise ValueError(f"Rule '{rule_name}' not found in library")

        except Exception as e:
            logger.error(f"Error getting rule {rule_name}: {e}")
            raise

    def ensure_default_rules_present(self):
        """Ensure all default rules from RuleLibrary.RULES are present in the library."""
        try:
            logger.debug("Checking for missing default rules")

            # Get all default rule names
            default_rule_names = set(RuleLibrary.RULES.keys())

            # Get current rule names
            current_rule_names = set(self.rules.keys())

            # Find missing default rules
            missing_rules = default_rule_names - current_rule_names

            # Check for missing parameters in existing rules
            rules_needing_parameter_update = []
            for rule_name, rule_data in self.rules.items():
                if 'params' not in rule_data:
                    rule_data['params'] = {}

                # CRITICAL FIX: Only check for parameters that are actually defined in the rule's PARAMETER_METADATA
                rule_type = rule_data.get('type', '')

                try:
                    # Try to get the rule class directly
                    try:
                        rule_class = globals()[rule_type]
                    except KeyError:
                        # If the class doesn't exist with that name, try to find it in RuleLibrary.RULES
                        if rule_name in RuleLibrary.RULES:
                            # Create a temporary instance to get the class
                            temp_metadata = RuleMetadata(**{k: v for k, v in rule_data.items() if k != 'params'})
                            temp_rule = RuleLibrary.RULES[rule_name](temp_metadata)
                            rule_class = temp_rule.__class__

                            # Update the type field to match the actual class name
                            if rule_data['type'] != rule_class.__name__:
                                rule_data['type'] = rule_class.__name__
                                logger.info(f"Updated rule type from '{rule_type}' to '{rule_class.__name__}'")
                        else:
                            raise KeyError(f"Could not find rule class for {rule_name}")

                    if hasattr(rule_class, 'PARAMETER_METADATA'):
                        missing_params = False
                        for param_name, param_info in rule_class.PARAMETER_METADATA.items():
                            if param_name not in rule_data['params'] and 'default' in param_info:
                                rule_data['params'][param_name] = param_info['default']
                                missing_params = True
                                logger.debug(f"Added missing parameter {param_name} to rule {rule_name}")

                        if missing_params:
                            rules_needing_parameter_update.append(rule_name)
                except (KeyError, AttributeError) as e:
                    logger.warning(f"Could not check parameters for rule {rule_name}: {e}")

            if missing_rules or rules_needing_parameter_update:
                logger.info(f"Found {len(missing_rules)} missing default rules and {len(rules_needing_parameter_update)} rules needing parameter updates")

                # Set flag to prevent recursion
                global _CREATING_DEFAULT_LIBRARY
                _CREATING_DEFAULT_LIBRARY = True

                try:
                    # Create default metadata for each missing rule
                    for rule_name in missing_rules:
                        try:
                            # Create a default metadata object
                            default_metadata = DefaultRuleMetadata.get_default_metadata(
                                rule_type=rule_name,
                                rule_name=rule_name,
                                category="Basic",
                                description="A cellular automata rule."
                            )

                            # Create a temporary instance to get the actual class name
                            temp_rule = RuleLibrary.RULES[rule_name](default_metadata)
                            rule_class = temp_rule.__class__

                            # Create a dictionary for the rule
                            new_rule_data = {
                                'name': rule_name,
                                'type': rule_class.__name__,  # Use the actual class name
                                'position': default_metadata.position,
                                'category': default_metadata.category,
                                'author': default_metadata.author,
                                'url': default_metadata.url,
                                'email': default_metadata.email,
                                'date_created': default_metadata.date_created,
                                'date_modified': datetime.now().strftime("%Y-%m-%d"),
                                'version': default_metadata.version,
                                'description': default_metadata.description,
                                'tags': default_metadata.tags + [rule_name, "default"],  # Add rule name and default tag
                                'dimension_compatibility': default_metadata.dimension_compatibility,
                                'neighborhood_compatibility': default_metadata.neighborhood_compatibility,
                                'parent_rule': default_metadata.parent_rule,
                                'rating': default_metadata.rating,
                                'notes': default_metadata.notes,
                                'allowed_initial_conditions': default_metadata.allowed_initial_conditions,
                                'allow_rule_tables': default_metadata.allow_rule_tables,
                                'favorite': False, # ADDED: Default favorite to False
                                'params': {}
                            }

                            # Get the rule class to populate parameters
                            if hasattr(rule_class, 'PARAMETER_METADATA'):
                                for param_name, param_info in rule_class.PARAMETER_METADATA.items():
                                    if 'default' in param_info:
                                        new_rule_data['params'][param_name] = param_info['default']

                            # Add the rule to the library
                            self.rules[rule_name] = new_rule_data
                            logger.debug(f"Added missing default rule: {rule_name} with type {rule_class.__name__}")
                        except Exception as e:
                            logger.error(f"Error adding missing default rule {rule_name}: {e}")
                            continue

                    # Save the updated library
                    self.save_library()
                    logger.info(f"Added {len(missing_rules)} missing default rules and updated {len(rules_needing_parameter_update)} rules with missing parameters")

                finally:
                    # Clear the flag
                    _CREATING_DEFAULT_LIBRARY = False
            else:
                logger.debug("All default rules are present in the library with all required parameters")

        except Exception as e:
            logger.error(f"Error ensuring default rules are present: {e}")

    def load_library(self, filepath=None, validate=True):
        """
        Loads the rule library from a JSON file, creating defaults if missing,
        validating, and saving modifications if necessary.
        (Round 11 Fix: Corrected validation and loading flow)
        """
        log_prefix = "RuleLibraryManager.load_library: "
        try:
            # Prevent recursive validation calls
            if RuleLibraryManager._is_validating:
                logger.debug(f"{log_prefix}Already validating, skipping recursive call.")
                return True
            RuleLibraryManager._is_validating = True
            logger.debug(f"{log_prefix}Set _is_validating = True.")

            if filepath is None:
                filepath = self.library_path

            logger.debug(f"{log_prefix}Attempting to load rule library from {filepath}")

            library_data = None # Initialize library_data

            # Check if the file exists and has content
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                logger.warning(f"{log_prefix}Rule library file not found at {filepath} or is empty. Creating default library.")
                library_data = self._create_default_library() # Generate default data
                self.save_library(data_to_save=library_data) # Save the generated library
                # Load self.rules from the newly created default data
                self._load_rules_from_library_data(library_data)
                logger.info(f"{log_prefix}Default library created and loaded.")
                RuleLibraryManager._is_validating = False # Reset flag
                logger.debug(f"{log_prefix}Reset _is_validating = False.")
                return True

            # Load the rule library from the existing JSON file
            logger.debug(f"{log_prefix}Loading existing file: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                library_data = json.load(f)
            logger.debug(f"{log_prefix}Successfully loaded data from JSON.")

            # Check for missing default rules and update parameters if needed
            # This might modify library_data in place and save if changes were made
            logger.debug(f"{log_prefix}Calling _check_for_missing_default_rules...")
            self._check_for_missing_default_rules(library_data)
            logger.debug(f"{log_prefix}Finished _check_for_missing_default_rules.")

            # Validate the rule library metadata (if requested)
            if validate:
                logger.info(f"{log_prefix}Starting library validation...")
                is_valid, modified, validated_data = self.validate_library_metadata(library_data)
                logger.info(f"{log_prefix}Validation complete. IsValid={is_valid}, Modified={modified}")

                if not is_valid:
                    logger.error(f"{log_prefix}Rule library metadata validation failed. Errors may exist.")
                    # Decide whether to proceed or stop? For now, proceed with potentially invalid data.
                if modified:
                    logger.info(f"{log_prefix}Rule library metadata was modified during validation. Saving changes.")
                    self._create_rule_library_backup(filepath)
                    # Save the validated data
                    self.save_library(data_to_save=validated_data)
                    # Use the validated data for loading self.rules
                    library_data = validated_data
                else:
                    logger.info(f"{log_prefix}Rule library metadata validation successful, no modifications needed.")
            else:
                 logger.info(f"{log_prefix}Skipping validation.")

            # Load the rules into the rule library manager instance using the final (potentially validated) data
            logger.debug(f"{log_prefix}Loading final rule data into self.rules...")
            self._load_rules_from_library_data(library_data)

            logger.info(f"{log_prefix}Library loaded successfully.")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"{log_prefix}Invalid JSON in {filepath}: {e}. Attempting to create default library.")
            try:
                self._create_rule_library_backup(filepath) # Backup corrupted file
                library_data = self._create_default_library()
                self.save_library(data_to_save=library_data)
                self._load_rules_from_library_data(library_data)
                return True
            except Exception as e_create:
                 logger.critical(f"{log_prefix}Failed to create default library after JSON error: {e_create}")
                 return False
        except Exception as e:
            logger.error(f"{log_prefix}Unexpected error loading rule library: {e}")
            logger.error(traceback.format_exc())
            # Attempt to create a default library as a fallback
            try:
                logger.warning(f"{log_prefix}Attempting to create default library due to loading error.")
                library_data = self._create_default_library()
                self.save_library(data_to_save=library_data)
                self._load_rules_from_library_data(library_data)
                return True
            except Exception as e_create:
                 logger.critical(f"{log_prefix}Failed to create default library after loading error: {e_create}")
                 return False
        finally:
            # Reset the validation flag
            RuleLibraryManager._is_validating = False
            logger.debug(f"{log_prefix}Reset _is_validating = False.")

    def _create_basic_default_library(self) -> Dict[str, Any]:
        """Create a basic default rule library with a hardcoded rule and all default rules."""
        logger.info("Creating basic default rule library")

        # Start with a basic structure
        library_data = {
            'library_metadata': {
                'version': '1.0',
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'description': 'Lace Rule Library'
            },
            'rules': []
        }

        # Add a hardcoded Default Rule (NOT Test Rule)
        new_rule_data = {
            'name': 'Default Rule',  # CHANGED: Name to Default Rule
            'type': 'TestRule',
            'position': 1,
            'category': 'Uncategorized',
            'author': "Nova Spivack",
            'url': "https://novaspivack.com/network_automata",
            'email': "novaspivackrelay @ gmail . com",
            'date_created': datetime.now().strftime("%Y-%m-%d"),
            'date_modified': datetime.now().strftime("%Y-%m-%d"),
            'version': "1.0",
            'description': "A simple rule for debugging neighbor identification.",
            'tags': ["test", "debug"],
            'dimension_compatibility': ["TWO_D"],
            'neighborhood_compatibility': ["MOORE", "VON_NEUMANN"],
            'parent_rule': None,
            'rating': None,
            'notes': "This rule is for debugging purposes only.",
            'allowed_initial_conditions': ["Random"],
            'allow_rule_tables': True,
            'params': {
                'neighborhood_type': "MOORE",
                'dimension_type': "TWO_D",
                'edge_initialization': "FULL",
                'tiebreaker_type': "RANDOM",
                'initial_conditions': "Random",
                'initial_density': 0.5,
                'grid_boundary': "bounded",
                'use_state_coloring': True,
                'use_state_coloring_edges': True,
                'chunk_size': 100,
                'connect_probability': 0.5,
                'birth_threshold': 3,
                'survival_threshold': 2,
                'death_threshold': 4,
                'min_neighbors_birth': 3,
                'max_neighbors_birth': 3,
                'min_neighbors_survival': 2,
                'max_neighbors_survival': 3
            }
        }
        library_data['rules'].append(new_rule_data)

        logger.info("Created basic default rule library with Default Rule")
        return library_data

    def _create_default_library(self) -> Dict[str, Any]:
        """
        Create a default rule library with all known rules, ensuring all metadata
        and default parameters (base + subclass), respecting exclusions and ensuring
        correct types (e.g., list of tuples) are written.
        (Round 12 Fix: Correct metadata defaults & list-of-tuple param types)
        """
        logger.info("Creating default rule library (R12 Fix: Metadata defaults & list-of-tuple types)")
        library_data = {
            'library_metadata': {
                'version': '1.0',
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'description': 'Lace Rule Library'
            },
            'rules': []
        }

        metadata_field_names = [f.name for f in fields(RuleMetadata) if f.name != '_ignored_params']

        for rule_name, rule_class in RuleLibrary.RULES.items():
            temp_rule = None
            try:
                logger.debug(f"Processing default rule: {rule_name} (Class: {rule_class.__name__})")
                # Create temporary metadata just to instantiate the rule
                temp_metadata_for_instantiation = RuleMetadata(
                    name=rule_name, type=rule_class.__name__, position=1, category="Temp",
                    author="", url="", email="", date_created="", date_modified="", version="",
                    description="", tags=[], dimension_compatibility=[], neighborhood_compatibility=[],
                    parent_rule=None, rating=None, notes=None, allowed_initial_conditions=[],
                    allow_rule_tables=True, favorite=False
                )
                temp_rule = rule_class(temp_metadata_for_instantiation)
                rule_type = temp_rule.__class__.__name__

                new_rule_data: Dict[str, Any] = {}

                # 1. Populate Metadata Fields (Use GlobalSettings explicitly)
                for field_name in metadata_field_names:
                    default_value = None
                    # --- Explicitly use GlobalSettings for these ---
                    if field_name == 'author': default_value = GlobalSettings.Defaults.DEFAULT_AUTHOR
                    elif field_name == 'url': default_value = GlobalSettings.Defaults.DEFAULT_URL
                    elif field_name == 'email': default_value = GlobalSettings.Defaults.DEFAULT_EMAIL
                    # --- Fallback to getattr or other defaults ---
                    elif field_name == 'name': default_value = rule_name
                    elif field_name == 'type': default_value = rule_type
                    elif field_name == 'category': default_value = getattr(temp_rule.metadata, 'category', "General Purpose")
                    elif field_name == 'date_created': default_value = getattr(temp_rule.metadata, 'date_created', datetime.now().strftime("%Y-%m-%d"))
                    elif field_name == 'date_modified': default_value = datetime.now().strftime("%Y-%m-%d") # Always set to now
                    elif field_name == 'version': default_value = getattr(temp_rule.metadata, 'version', "1.0")
                    elif field_name == 'description': default_value = getattr(temp_rule.metadata, 'description', "A LACE link automata rule.")
                    elif field_name == 'tags': default_value = list(set(getattr(temp_rule.metadata, 'tags', []) + [rule_name, "default"]))
                    elif field_name == 'dimension_compatibility': default_value = getattr(temp_rule.metadata, 'dimension_compatibility', ["TWO_D", "THREE_D"])
                    elif field_name == 'neighborhood_compatibility': default_value = getattr(temp_rule.metadata, 'neighborhood_compatibility', [])
                    elif field_name == 'allowed_initial_conditions': default_value = getattr(temp_rule.metadata, 'allowed_initial_conditions', ["Random"])
                    elif field_name == 'allow_rule_tables': default_value = getattr(temp_rule.metadata, 'allow_rule_tables', True)
                    elif field_name == 'favorite': default_value = getattr(temp_rule.metadata, 'favorite', False)
                    elif field_name == 'position': default_value = getattr(temp_rule.metadata, 'position', 1)
                    # Optional fields default to None implicitly

                    value = default_value # Start with the determined default

                    # --- Special overrides (Name, Type, Date Modified) ---
                    if field_name == 'name': value = rule_name
                    if field_name == 'type': value = rule_type
                    if field_name == 'date_modified': value = datetime.now().strftime("%Y-%m-%d")
                    # Ensure tags are handled correctly
                    if field_name == 'tags':
                         existing_tags = getattr(temp_rule.metadata, 'tags', [])
                         value = list(set(existing_tags + [rule_name, "default"]))

                    # Assign value
                    optional_none_fields = {'parent_rule', 'rating', 'notes', 'neighborhood_compatibility'}
                    optional_empty_list_fields = {'tags', 'dimension_compatibility', 'neighborhood_compatibility', 'allowed_initial_conditions'}
                    if value is not None or field_name in optional_none_fields or field_name in optional_empty_list_fields:
                         new_rule_data[field_name] = value

                # 2. Populate 'params' dictionary, respecting EXCLUDE_EDITOR_PARAMS and types
                params_dict: Dict[str, Any] = {}
                base_metadata_params = getattr(Rule, 'PARAMETER_METADATA', {})
                subclass_metadata_params = getattr(rule_class, 'PARAMETER_METADATA', {})
                merged_metadata_params = {**base_metadata_params, **subclass_metadata_params}
                exclude_set = getattr(rule_class, 'EXCLUDE_EDITOR_PARAMS', set())
                logger.debug(f"  Rule {rule_name}: Exclude set = {exclude_set}")

                for param_name, param_info in merged_metadata_params.items():
                    if param_name in exclude_set:
                        logger.debug(f"  Skipping excluded parameter '{param_name}' for {rule_name} during default creation.")
                        continue
                    if 'default' in param_info:
                        default_val = param_info['default']
                        # --- Convert list-of-lists to list-of-tuples if needed ---
                        if isinstance(default_val, list) and param_info.get('element_type') == tuple:
                            try:
                                converted_val = [tuple(item) if isinstance(item, list) else item for item in default_val]
                                if all(isinstance(item, tuple) for item in converted_val):
                                    logger.debug(f"  Converting default for '{param_name}' to list of tuples.")
                                    params_dict[param_name] = converted_val
                                else:
                                     logger.warning(f"  Could not convert default for '{param_name}' to list of tuples, using original.")
                                     params_dict[param_name] = default_val # Fallback
                            except Exception as conv_err:
                                logger.warning(f"  Error converting default for '{param_name}' to list of tuples: {conv_err}. Using original.")
                                params_dict[param_name] = default_val # Fallback
                        else:
                            params_dict[param_name] = default_val
                        # ---
                    # else: logger.debug(f"  Skipping param '{param_name}' (no default)") # Reduce noise

                new_rule_data['params'] = params_dict

                library_data['rules'].append(new_rule_data)
                logger.debug(f"Added default rule: {rule_name} with type {rule_type}")

            except Exception as e:
                logger.error(f"!!! FAILED to create default entry for rule {rule_name} !!! Error: {e}")
                logger.error(traceback.format_exc())
                continue

        logger.info(f"Created default rule library with {len(library_data['rules'])} rules")
        return library_data

    @staticmethod
    def _get_merged_parameter_metadata(rule_class: Type[Rule]) -> Dict[str, Dict[str, Any]]:
        """
        Statically gets the merged and filtered parameter metadata for a given rule class.
        Handles merging base and subclass metadata and applying exclusions.
        (Moved from RuleEditorWindow in Round 2)
        """
        logger = logging.getLogger(__name__) # Get logger instance
        log_prefix = "RuleLibraryManager._get_merged_parameter_metadata: "
        logger.debug(f"{log_prefix}Merging metadata for class {rule_class.__name__}")

        # Ensure base Rule metadata is populated if needed
        if not Rule.PARAMETER_METADATA:
            logger.debug(f"{log_prefix}Base Rule.PARAMETER_METADATA empty, populating...")
            Rule._populate_base_metadata()

        base_meta = getattr(Rule, 'PARAMETER_METADATA', {})
        sub_meta = getattr(rule_class, 'PARAMETER_METADATA', {}) # Get from class object

        # Perform the merge (same logic as before, now static)
        merged = copy.deepcopy(base_meta)
        for param_name, sub_info in sub_meta.items():
            if param_name in merged:
                base_info = merged[param_name]
                if 'type' in sub_info and 'type' in base_info and sub_info['type'] != base_info['type']:
                    logger.warning(f"{log_prefix}Type mismatch for '{param_name}' in {rule_class.__name__}. Base: {base_info['type']}, Subclass: {sub_info['type']}. Using subclass type.")
                    base_info['type'] = sub_info['type']
                for key in ['default', 'allowed_values', 'min', 'max', 'parameter_group', 'description', 'element_type', 'editor_sort_key']:
                    if key in sub_info:
                        base_info[key] = sub_info[key]
                merged[param_name] = base_info
            else:
                merged[param_name] = copy.deepcopy(sub_info)

        # Filter excluded parameters
        exclude_set = getattr(rule_class, 'EXCLUDE_EDITOR_PARAMS', set())
        final_meta = {k: v for k, v in merged.items() if k not in exclude_set}
        logger.debug(f"{log_prefix}Finished merging. Final metadata keys: {list(final_meta.keys())}")
        return final_meta

    def validate_library_metadata(self, library_data: Dict[str, Any]) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        Validates and fixes rule metadata and parameters in the library.
        Includes migration step and handles dynamic parameters correctly.
        (Round 14: Ensure dictionary isolation during validation)
        (Round 1: Exclude _BASE params from missing check)
        """
        log_prefix = "validate_library_metadata (R1 Fix): " # Updated round
        logger.debug(f"{log_prefix}Entering.")

        is_valid_library = True
        modified = False # Overall library modification flag
        validated_data = copy.deepcopy(library_data)

        required_metadata = { # Unchanged
            'name': None, 'type': None, 'category': "General Purpose",
            'author': GlobalSettings.Defaults.DEFAULT_AUTHOR, 'url': GlobalSettings.Defaults.DEFAULT_URL,
            'email': GlobalSettings.Defaults.DEFAULT_EMAIL, 'date_created': datetime.now().strftime("%Y-%m-%d"),
            'date_modified': datetime.now().strftime("%Y-%m-%d"), 'version': "1.0",
            'description': "A cellular automata rule.", 'tags': [],
            'dimension_compatibility': ["TWO_D", "THREE_D"], 'neighborhood_compatibility': None,
            'parent_rule': None, 'rating': None, 'notes': None,
            'allowed_initial_conditions': ["Random"], 'allow_rule_tables': True,
            'favorite': False, 'position': 1
        }
        optional_none_fields = {'parent_rule', 'rating', 'notes', 'neighborhood_compatibility'}
        optional_empty_list_fields = {'tags', 'dimension_compatibility', 'neighborhood_compatibility', 'allowed_initial_conditions'}

        rules_to_process = validated_data.get('rules', [])
        if not isinstance(rules_to_process, list):
            logger.error("Invalid library_data['rules'] format, expected list.")
            return False, False, validated_data

        if not Rule.PARAMETER_METADATA:
            Rule._populate_base_metadata()

        for rule_idx, rule_data in enumerate(rules_to_process):
            # [ Initial checks and metadata validation - Unchanged ]
            if not isinstance(rule_data, dict): logger.warning(f"Skipping invalid rule entry at index {rule_idx} (not a dict)."); continue
            rule_name = rule_data.get('name', f"Unknown Rule {rule_idx}"); rule_type_from_json = rule_data.get('type', 'Unknown')
            logger.debug(f"\nValidating Rule {rule_idx + 1}: {rule_name} (Type from JSON: {rule_type_from_json})")
            fixed_metadata = rule_data; rule_modified_this_iter = False; is_orphaned = False
            for field, default_value in required_metadata.items():
                current_value = fixed_metadata.get(field); original_value = copy.deepcopy(current_value)
                field_missing = field not in fixed_metadata; field_is_none = current_value is None
                field_is_empty_str = isinstance(current_value, str) and not current_value.strip() and field in ['name', 'type', 'category', 'description', 'author', 'url', 'email']
                needs_default = field_missing or field_is_none or field_is_empty_str
                if needs_default:
                    if default_value is None and field in ['name', 'type']: logger.error(f"Rule {rule_name}: Required field '{field}' missing/empty. Marking as Orphaned."); fixed_metadata['category'] = "Orphaned Rule - Missing Core Info"; is_orphaned = True; rule_modified_this_iter = True; break
                    effective_default = copy.deepcopy(default_value)
                    if field == 'category' and (effective_default is None or not str(effective_default).strip()): effective_default = "General Purpose"
                    fixed_metadata[field] = effective_default
                    if (field_missing or field_is_none or field_is_empty_str) and (effective_default is not None or field in optional_none_fields or field in optional_empty_list_fields):
                        if original_value != effective_default:
                             log_level = logging.WARNING if field not in optional_none_fields.union(optional_empty_list_fields) else logging.DEBUG
                             logger.log(log_level, f"Rule {rule_name}: Missing/None/empty field '{field}'. Applied default: '{effective_default}'")
                             rule_modified_this_iter = True
            if is_orphaned: modified = True; continue
            rule_class: Optional[Type[Rule]] = None
            if not is_orphaned:
                try: rule_class = globals()[fixed_metadata['type']]
                except KeyError:
                    if rule_name in RuleLibrary.RULES:
                        try:
                            temp_meta_dict = {k: v for k, v in fixed_metadata.items() if k != 'params' and k != '_ignored_params'}; temp_meta_dict.setdefault('position', 1); temp_metadata = RuleMetadata(**temp_meta_dict)
                            temp_rule = RuleLibrary.RULES[rule_name](temp_metadata); rule_class = temp_rule.__class__
                            if fixed_metadata['type'] != rule_class.__name__: logger.warning(f"  Correcting 'type' field for '{rule_name}' from '{fixed_metadata['type']}' to '{rule_class.__name__}'."); fixed_metadata['type'] = rule_class.__name__; rule_modified_this_iter = True
                        except Exception as factory_err: logger.error(f"  Error creating temp rule '{rule_name}': {factory_err}. Marking Orphaned."); is_orphaned = True
                    else: logger.error(f"  Rule '{rule_name}' (Type: '{fixed_metadata['type']}') not found. Marking Orphaned."); is_orphaned = True
                if is_orphaned: fixed_metadata['category'] = "Orphaned Rule - Type Mismatch"; logger.warning(f"  Changed category to 'Orphaned Rule - Type Mismatch'."); rule_modified_this_iter = True; continue
            if not is_orphaned:
                for field in ['author', 'url', 'email']:
                    default_val = required_metadata[field]; current_val = fixed_metadata.get(field)
                    if field not in fixed_metadata or (isinstance(current_val, str) and not current_val.strip()):
                        if current_val != default_val: logger.info(f"Fixed {field.capitalize()} for {rule_name}"); fixed_metadata[field] = default_val; rule_modified_this_iter = True
                    elif not isinstance(current_val, str): logger.warning(f"Corrected non-string {field} for {rule_name}"); fixed_metadata[field] = default_val; rule_modified_this_iter = True
                if 'favorite' not in fixed_metadata or not isinstance(fixed_metadata.get('favorite'), bool):
                    if fixed_metadata.get('favorite') != False: rule_modified_this_iter = True; logger.warning(f"Fixed non-boolean 'favorite' for {rule_name}")
                    fixed_metadata['favorite'] = False
                if 'position' not in fixed_metadata or not isinstance(fixed_metadata.get('position'), int):
                     if fixed_metadata.get('position') != 1: rule_modified_this_iter = True; logger.warning(f"Fixed non-integer 'position' for {rule_name}")
                     fixed_metadata['position'] = 1
                for field in optional_none_fields:
                    if field not in fixed_metadata: fixed_metadata[field] = None
                for field in optional_empty_list_fields:
                     if field not in fixed_metadata: fixed_metadata[field] = []
            if 'params' not in fixed_metadata: fixed_metadata['params'] = {}; rule_modified_this_iter = True; logger.warning(f"Rule {rule_name}: Added missing params dictionary")
            elif not isinstance(fixed_metadata['params'], dict): logger.error(f"Rule {rule_name}: 'params' not dict. Resetting."); fixed_metadata['params'] = {}; rule_modified_this_iter = True; is_valid_library = False

            # --- Step 2.5: Apply Migration (if applicable) ---
            params_from_json = fixed_metadata['params']
            if not isinstance(params_from_json, dict): logger.error(f"Rule {rule_name}: 'params' field is not a dictionary (type: {type(params_from_json)}). Resetting to empty before migration."); params_from_json = {}; rule_modified_this_iter = True
            params_after_migration = params_from_json # Default to original if no migration
            if not is_orphaned and rule_class and hasattr(rule_class, 'migrate_params_vX_to_vY') and callable(getattr(rule_class, 'migrate_params_vX_to_vY', None)):
                logger.info(f"{log_prefix}Applying migration for rule '{rule_name}' (Type: {rule_class.__name__})...")
                params_before_migration = copy.deepcopy(params_from_json)
                try:
                    migration_method = getattr(rule_class, 'migrate_params_vX_to_vY'); params_after_migration = copy.deepcopy(migration_method(params_from_json)) # Use deepcopy
                    if not isinstance(params_after_migration, dict): logger.error(f"{log_prefix}Migration for rule '{rule_name}' did not return a dictionary (returned {type(params_after_migration)}). Reverting."); params_after_migration = params_before_migration
                    elif params_after_migration != params_before_migration: logger.info(f"{log_prefix}Migration applied changes for rule '{rule_name}'."); rule_modified_this_iter = True
                    else: logger.debug(f"{log_prefix}Migration function ran for '{rule_name}', but no changes were made.")
                except Exception as migrate_err: logger.error(f"{log_prefix}Error during parameter migration for rule '{rule_name}': {migrate_err}"); logger.error(traceback.format_exc()); params_after_migration = params_before_migration
            # --- END Migration Step ---

            # --- Step 3: Validate Parameters ---
            if not is_orphaned and rule_class:
                params_to_validate = params_after_migration
                if not isinstance(params_to_validate, dict): logger.error(f"Rule {rule_name}: 'params' field is not a dictionary after migration attempt (type: {type(params_to_validate)}). Skipping parameter validation."); is_valid_library = False
                else:
                    try:
                        static_merged_metadata = RuleLibraryManager._get_merged_parameter_metadata(rule_class)
                        expected_dynamic_names = set()
                        if hasattr(rule_class, '_get_expected_dynamic_param_names_from_selectors') and callable(getattr(rule_class, '_get_expected_dynamic_param_names_from_selectors', None)):
                            try:
                                get_dynamic_names_method = getattr(rule_class, '_get_expected_dynamic_param_names_from_selectors')
                                expected_dynamic_names = get_dynamic_names_method(params_to_validate)
                                logger.debug(f"Rule {rule_name}: Expected dynamic params based on selectors in migrated params: {expected_dynamic_names}")
                            except Exception as dyn_err: logger.error(f"Error getting expected dynamic param names for {rule_name}: {dyn_err}")

                        exclude_set = getattr(rule_class, 'EXCLUDE_EDITOR_PARAMS', set())
                        all_expected_param_names = set(static_merged_metadata.keys()) | expected_dynamic_names
                        logger.debug(f"Rule {rule_name}: All expected param names (static + dynamic): {all_expected_param_names}")

                        # --- 3a. Remove UNKNOWN parameters ---
                        params_to_remove = []
                        keys_in_current_params = list(params_to_validate.keys())
                        logger.debug(f"  Keys present before removing unknown: {keys_in_current_params}")
                        for param_name_in_current in keys_in_current_params:
                            if param_name_in_current not in all_expected_param_names:
                                params_to_remove.append(param_name_in_current)
                                logger.warning(f"Rule {rule_name}: Found unknown parameter '{param_name_in_current}' after migration/dynamic check. Removing it.")
                                rule_modified_this_iter = True
                        for param_to_remove in params_to_remove:
                            if param_to_remove in params_to_validate: del params_to_validate[param_to_remove]
                        logger.debug(f"  Keys remaining after removing unknown: {list(params_to_validate.keys())}")

                        # --- 3b. Add Missing DEFINED parameters (Static AND Dynamic) ---
                        logger.debug(f"{log_prefix}Checking for missing parameters...")
                        params_to_add = {}
                        for param_name in all_expected_param_names:
                            # --- MODIFIED: Skip _BASE parameters ---
                            if param_name.endswith('_BASE'):
                                logger.debug(f"  Skipping check for template parameter: {param_name}")
                                continue
                            # --- END MODIFIED ---
                            if param_name not in params_to_validate or params_to_validate[param_name] is None:
                                param_info = static_merged_metadata.get(param_name)
                                default_val = None; has_default = False
                                if param_info and 'default' in param_info and param_name not in exclude_set:
                                    default_val = param_info['default']; has_default = True
                                elif param_name in expected_dynamic_names:
                                    if hasattr(rule_class, '_get_permutation_default') and isinstance(getattr(rule_class, '_get_permutation_default', None), staticmethod):
                                        try: get_default_method = getattr(rule_class, '_get_permutation_default'); default_val = get_default_method(param_name); has_default = True
                                        except Exception as dyn_def_err: logger.error(f"Error getting dynamic default for {param_name}: {dyn_def_err}")
                                    else: logger.warning(f"Rule {rule_name}: Missing or non-static dynamic default getter for '{param_name}'.")
                                if has_default:
                                    if isinstance(default_val, list) and param_info and param_info.get('element_type') == tuple:
                                        try: converted_default = [tuple(item) if isinstance(item, list) else item for item in default_val]; params_to_add[param_name] = converted_default
                                        except Exception: params_to_add[param_name] = default_val
                                    else: params_to_add[param_name] = default_val
                                    logger.warning(f"Rule {rule_name}: Adding missing defined param '{param_name}' with default: {params_to_add[param_name]}")
                                    rule_modified_this_iter = True
                                elif param_name not in exclude_set:
                                    logger.warning(f"Rule {rule_name}: Missing expected param '{param_name}' with no default value available.")
                        params_to_validate.update(params_to_add)
                        logger.debug(f"  Keys remaining after adding missing: {list(params_to_validate.keys())}")

                        # --- 3c. Validate Types/Values of ALL parameters now present ---
                        logger.debug(f"{log_prefix}Validating types and values for {len(params_to_validate)} parameters...")
                        for param_name in list(params_to_validate.keys()):
                             param_info = static_merged_metadata.get(param_name)
                             dynamic_default_val = None; has_dynamic_default_info = False
                             if param_info is None and param_name in expected_dynamic_names:
                                 param_info = {}
                                 if hasattr(rule_class, '_get_permutation_default'):
                                     try:
                                         dynamic_default_val = getattr(rule_class, '_get_permutation_default')(param_name); has_dynamic_default_info = True
                                         if dynamic_default_val is not None:
                                             param_info['type'] = type(dynamic_default_val)
                                             if isinstance(dynamic_default_val, list):
                                                 if dynamic_default_val and all(isinstance(i, tuple) for i in dynamic_default_val): param_info['element_type'] = tuple
                                                 elif dynamic_default_val and all(isinstance(i, int) for i in dynamic_default_val): param_info['element_type'] = int
                                                 elif dynamic_default_val and all(isinstance(i, float) for i in dynamic_default_val): param_info['element_type'] = float
                                                 elif dynamic_default_val and all(isinstance(i, object) for i in dynamic_default_val): param_info['element_type'] = object
                                     except: pass
                             if param_info is not None:
                                 value = params_to_validate[param_name]; expected_type = param_info.get('type'); converted_value = value; type_valid = True; value_valid = True
                                 original_value_before_conv = copy.deepcopy(value); param_reset_to_default = False
                                 if value is None and (expected_type == str or expected_type is None): converted_value = None; type_valid = True;
                                 elif expected_type:
                                     try:
                                         if expected_type == int: converted_value = int(float(value))
                                         elif expected_type == float: converted_value = float(value)
                                         elif expected_type == bool: converted_value = str(value).lower() in ('true', '1', 'yes', 'on')
                                         elif expected_type == list:
                                             if isinstance(value, str): parsed = ast.literal_eval(value)
                                             else: parsed = value
                                             if not isinstance(parsed, list): raise TypeError("Not a list")
                                             element_type = param_info.get('element_type')
                                             if element_type == tuple:
                                                 converted_value = [tuple(item) if isinstance(item, list) else item for item in parsed]
                                                 if not all(isinstance(item, tuple) for item in converted_value): raise TypeError("Inner elements not tuples")
                                             elif element_type == object:
                                                 if not all(isinstance(item, (int, float)) for item in parsed): raise TypeError(f"List elements must be int or float for element_type object")
                                                 converted_value = parsed
                                             elif element_type and not all(isinstance(item, element_type) for item in parsed): raise TypeError(f"List elements must be of type {element_type}")
                                             else: converted_value = parsed
                                         elif expected_type == tuple:
                                              if isinstance(value, str): parsed = ast.literal_eval(value)
                                              else: parsed = value
                                              if not isinstance(parsed, tuple): raise TypeError("Not a tuple")
                                              converted_value = parsed
                                         elif expected_type == dict:
                                             if isinstance(value, str): parsed = ast.literal_eval(value)
                                             else: parsed = value
                                             if not isinstance(parsed, dict): raise TypeError("Not a dict")
                                             converted_value = parsed
                                         else:
                                             if converted_value is not None and not isinstance(converted_value, expected_type):
                                                  try: converted_value = expected_type(value)
                                                  except (ValueError, TypeError): raise ValueError(f"Cannot convert '{value}' to type {expected_type}")
                                         if converted_value is not None and not isinstance(converted_value, expected_type): type_valid = False
                                     except (ValueError, TypeError, SyntaxError) as conv_err: logger.warning(f"Rule {rule_name}: Type/Conversion error for '{param_name}': {conv_err}"); type_valid = False
                                 if not type_valid:
                                     default_val = None; has_default_info = False
                                     if 'default' in param_info: default_val = param_info['default']; has_default_info = True
                                     elif has_dynamic_default_info: default_val = dynamic_default_val;
                                     if has_default_info: logger.warning(f"  Using default '{default_val}' for '{param_name}' due to type error."); params_to_validate[param_name] = default_val; rule_modified_this_iter = True; param_reset_to_default = True
                                     else: logger.error(f"  Cannot fix type for '{param_name}' - no default value!"); is_valid_library = False; continue
                                 else:
                                     params_to_validate[param_name] = converted_value
                                     min_val = param_info.get('min'); max_val = param_info.get('max'); allowed = param_info.get('allowed_values')
                                     if value is None and (param_name in ('node_colormap', 'edge_colormap') or param_name.startswith('rule_')): value_valid = True
                                     elif min_val is not None and isinstance(converted_value, (int, float)) and converted_value < min_val: value_valid = False;
                                     elif max_val is not None and isinstance(converted_value, (int, float)) and converted_value > max_val: value_valid = False;
                                     elif allowed is not None and converted_value not in allowed: value_valid = False;
                                     if not value_valid:
                                         default_val = None; has_default_info = False
                                         if 'default' in param_info: default_val = param_info['default']; has_default_info = True
                                         elif has_dynamic_default_info: default_val = dynamic_default_val;
                                         if has_default_info: logger.warning(f"Rule {rule_name}: Invalid value '{converted_value}' for param '{param_name}'. Using default."); params_to_validate[param_name] = default_val; rule_modified_this_iter = True; param_reset_to_default = True
                                         else: logger.error(f"  Cannot fix value for '{param_name}' - no default!"); is_valid_library = False
                                     elif converted_value != original_value_before_conv:
                                         is_list_tuple_conv = (expected_type == list and param_info.get('element_type') == tuple and isinstance(original_value_before_conv, list) and all(isinstance(x, list) for x in original_value_before_conv) and isinstance(converted_value, list) and all(isinstance(x, tuple) for x in converted_value))
                                         if not is_list_tuple_conv: logger.debug(f"Rule {rule_name}: Param '{param_name}' value changed during type conversion from '{original_value_before_conv}' to '{converted_value}'.")
                             else: logger.warning(f"Rule {rule_name}: No static or dynamic metadata found for parameter '{param_name}' during type/value validation. Skipping.")

                        # --- Remove _ignored_params field ---
                        if '_ignored_params' in fixed_metadata: del fixed_metadata['_ignored_params']; logger.info(f"Rule {rule_name}: Removed obsolete _ignored_params field."); rule_modified_this_iter = True
                        # ---

                    except Exception as param_val_err: logger.error(f"Could not validate parameters for rule {rule_name}: {param_val_err}")

            fixed_metadata['params'] = params_to_validate

            if rule_modified_this_iter: modified = True; logger.debug(f"Rule '{rule_name}' was modified during validation.")

        validated_data['rules'] = rules_to_process

        logger.debug("===================================================================")
        logger.debug("===================== $$$ VALIDATION COMPLETE =====================")
        logger.debug("===================================================================")
        return is_valid_library, modified, validated_data

    def _validate_rule_table(self, table_data: Dict[str, Any], table_type: str, rule_data: Dict[str, Any]):
        """Validate rule table data, dynamically handling different neighborhood sizes."""
        rule_name = rule_data.get('name', 'Unknown Rule') # Get name from rule_data
        if not isinstance(table_data, dict):
            raise ValueError(f"Rule {rule_name}: Rule table must be a dictionary")

        if "default" not in table_data:
            raise ValueError(f"Rule {rule_name}: Rule table must have a default value")

        # Get the expected pattern length based on the rule type and neighborhood.
        # Access rule data directly from the passed rule_data dictionary.
        try:
            rule_type_str = rule_data['type']

            # Try to get the rule class directly
            try:
                rule_class = globals()[rule_type_str]
            except KeyError:
                # If the class doesn't exist with that name, try to find it in RuleLibrary.RULES
                if rule_name in RuleLibrary.RULES:
                    # Create a temporary instance to get the class
                    temp_metadata = RuleMetadata(**{k: v for k, v in rule_data.items() if k != 'params'})
                    temp_rule = RuleLibrary.RULES[rule_name](temp_metadata)
                    rule_class = temp_rule.__class__
                else:
                    raise KeyError(f"Could not find rule class for {rule_name}")

            # Create metadata dictionary *without* 'params'
            metadata_dict = {k: v for k, v in rule_data.items() if k != 'params'}
            metadata = RuleMetadata(**metadata_dict)  # Create metadata *without* params
            temp_rule = rule_class(metadata)  # Create a temporary instance

            # Get neighborhood type from parameters, use Moore as default
            neighborhood_type_str = temp_rule.get_param("neighborhood_type", "MOORE") # Get from params, default to MOORE
            neighborhood_type = NeighborhoodType[neighborhood_type_str]

            # Get dimension type from parameters, use 2D as default
            dimension_type_str = temp_rule.get_param("dimension_type", "TWO_D") # Get from params, default to TWO_D
            dimension_type = Dimension[dimension_type_str]

            # Get neighborhood size
            if neighborhood_type == NeighborhoodType.MOORE:
                if dimension_type == Dimension.TWO_D:
                    neighborhood_size = 8
                elif dimension_type == Dimension.THREE_D:
                    neighborhood_size = 26
                else:
                    raise ValueError(f"Unsupported dimension type: {dimension_type_str}")
            elif neighborhood_type == NeighborhoodType.VON_NEUMANN:
                if dimension_type == Dimension.TWO_D:
                    neighborhood_size = 4
                elif dimension_type == Dimension.THREE_D:
                    neighborhood_size = 6
                else:
                    raise ValueError(f"Unsupported dimension type: {dimension_type_str}")
            elif neighborhood_type == NeighborhoodType.HEX:
                if dimension_type == Dimension.TWO_D:
                    neighborhood_size = 6
                else:
                    raise ValueError(f"Unsupported dimension type: {dimension_type_str}")
            elif neighborhood_type == NeighborhoodType.HEX_PRISM:
                if dimension_type == Dimension.THREE_D:
                    neighborhood_size = 12  # 6 on the same plane, 3 above, 3 below
                else:
                    raise ValueError(f"Unsupported dimension type: {dimension_type_str}")
            else:
                raise ValueError(f"Unsupported neighborhood type: {neighborhood_type_str}")

        except Exception as e:
            logger.error(f"Error getting neighborhood size for rule {rule_name}: {e}")
            raise ValueError(f"Could not determine neighborhood size for rule {rule_name}") from e

        for key, value in table_data.items():
            if key == "default":
                continue

            # Validate key format
            try:
                components = key.strip("()").split(",")
                if table_type == 'state':
                    if len(components) != 3:
                        raise ValueError(f"Rule {rule_name}: Invalid state rule key format: {key}")
                    current_state = int(components[0].strip())
                    neighbor_pattern = components[1].strip()
                    connection_pattern = components[2].strip()

                    if current_state not in [-1, 0, 1]:
                        raise ValueError(f"Rule {rule_name}: Current state must be -1, 0, or 1, but is {current_state}")
                    # Dynamically check pattern length
                    if len(neighbor_pattern) != neighborhood_size or not all(c in '01IN' for c in neighbor_pattern):
                        raise ValueError(f"Rule {rule_name}: Neighbor pattern must be {neighborhood_size} bits of 0, 1, I, or N, but is {neighbor_pattern}")
                    if len(connection_pattern) != neighborhood_size or not all(c in '01I' for c in connection_pattern):
                        raise ValueError(f"Rule {rule_name}: Connection pattern must be {neighborhood_size} bits of 0, 1, or I, but is {connection_pattern}")
                    if value not in [-1, 0, 1]:
                        raise ValueError(f"Rule {rule_name}: Invalid value for state rule key {key}: {value}")

                elif table_type == 'edge':
                    if len(components) != 3:
                        raise ValueError(f"Rule {rule_name}: Invalid edge rule key format: {key}")

                    self_state = int(components[0].strip())
                    neighbor_state = int(components[1].strip())
                    connection_pattern = components[2].strip()

                    if self_state not in [0, 1] or neighbor_state not in [0, 1]:
                        raise ValueError(f"Rule {rule_name}: Self state and neighbor state must be 0 or 1, but are {self_state} and {neighbor_state}")
                    # Dynamically check pattern length
                    if len(connection_pattern) != neighborhood_size or not all(c in '01I' for c in connection_pattern):
                        raise ValueError(f"Rule {rule_name}: Connection pattern must be {neighborhood_size} bits of 0, 1, or I, but is {connection_pattern}")
                    if value not in ["add", "remove", "maintain"]:
                        raise ValueError(f"Rule {rule_name}: Invalid value for edge rule key {key}: {value}")
            except (ValueError, IndexError) as e:
                raise ValueError(f"Rule {rule_name}: Invalid rule table format: {e}")

    def _validate_rule_table_completeness(self, table_data: dict, table_type: str) -> bool:
        """Validate that the rule table covers all necessary cases"""
        try:
            if table_type == 'state':
                # Check for required state transitions
                required_states = [-1, 0, 1]
                for current_state in required_states:
                    # Check basic transitions
                    for neighbor_pattern in itertools.product(['0', '1'], repeat=8):
                        for connection_pattern in itertools.product(['0', '1'], repeat=8):
                            key = f"({current_state}, {''.join(neighbor_pattern)}, {''.join(connection_pattern)})"
                            if key not in table_data and 'default' not in table_data:
                                raise ValueError(f"Missing rule for case: {key}")

            else:  # edge table
                # Check for required edge rules
                required_states = [0, 1]
                for self_state in required_states:
                    for neighbor_state in required_states:
                        for connection_pattern in itertools.product(['0', '1'], repeat=8):
                            key = f"({self_state}, {neighbor_state}, {''.join(connection_pattern)})"
                            if key not in table_data and 'default' not in table_data:
                                raise ValueError(f"Missing rule for case: {key}")

            return True

        except ValueError as e:
            logger.error(f"Completeness error: {e}")
            return False

    def _validate_rule_table_consistency(self, table_data: dict, table_type: str) -> bool:
        """Validate consistency across all rule table entries"""
        try:
            if table_type == 'state':
                # Check for required state transitions
                required_states = [-1, 0, 1]
                for current_state in required_states:
                    # Check basic transitions
                    for neighbor_pattern in itertools.product(['0', '1'], repeat=8):
                        for connection_pattern in itertools.product(['0', '1'], repeat=8):
                            key = f"({current_state}, {''.join(neighbor_pattern)}, {''.join(connection_pattern)})"
                            if key not in table_data and 'default' not in table_data:
                                raise ValueError(f"Missing rule for case: {key}")

            else:  # edge table
                # Check for required edge rules
                required_states = [0, 1]
                for self_state in required_states:
                    for neighbor_state in required_states:
                        for connection_pattern in itertools.product(['0', '1'], repeat=8):
                            key = f"({self_state}, {neighbor_state}, {''.join(connection_pattern)})"
                            if key not in table_data and 'default' not in table_data:
                                raise ValueError(f"Missing rule for case: {key}")

            return True

        except ValueError as e:
            logger.error(f"Consistency error: {e}")
            return False

    def _initialize_default_metadata(self):
        """Initialize metadata for all default rules"""
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Use the RuleLibrary class to create metadata
        for rule_name in RuleLibrary.RULES.keys():
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

    def get_all_categories(self) -> List[str]:
        """Returns a sorted list of unique, normalized category names, excluding 'Orphaned Rule'."""
        categories = set()
        for rule_data in self.rules.values():
            raw_category = rule_data.get('category', 'Unknown')
            normalized = raw_category.strip().title()
            if not normalized.startswith("Orphaned Rule"):
                categories.add(normalized)
        return sorted(list(categories))

    def get_rules_in_category(self, category_name: str) -> List[str]:
        """Returns a sorted list of rule names within a specific normalized category."""
        rules_in_cat = []
        normalized_target = category_name.strip().title()
        for rule_name, rule_data in self.rules.items():
            raw_category = rule_data.get('category', 'Unknown')
            normalized_rule_cat = raw_category.strip().title()
            if normalized_rule_cat == normalized_target:
                rules_in_cat.append(rule_name)
        return sorted(rules_in_cat)

    def update_rule_category(self, rule_name: str, new_category: str) -> bool:
        """Updates the category for a specific rule and saves the library."""
        log_prefix = f"RuleLibraryManager.update_rule_category(Rule='{rule_name}', NewCat='{new_category}'): "
        if rule_name not in self.rules:
            logger.error(f"{log_prefix}Rule '{rule_name}' not found.")
            return False
        try:
            normalized_new_category = new_category.strip().title()
            if not normalized_new_category:
                logger.error(f"{log_prefix}New category cannot be empty.")
                return False

            # --- Check if category changed ---
            old_category = self.rules[rule_name].get('category', 'Unknown')
            if old_category.strip().title() == normalized_new_category:
                logger.debug(f"{log_prefix}Category is already '{normalized_new_category}'. No change needed.")
                return True # No change, but considered success
            # ---

            logger.info(f"{log_prefix}Updating category from '{old_category}' to '{normalized_new_category}'.")
            self.rules[rule_name]['category'] = normalized_new_category
            self.rules[rule_name]['date_modified'] = datetime.now().strftime("%Y-%m-%d") # Update modified date
            self.save_library() # Save the entire library after modification
            # Clear cache for this rule
            if rule_name in RuleLibraryManager._rules_cache:
                del RuleLibraryManager._rules_cache[rule_name]
            return True
        except Exception as e:
            logger.error(f"{log_prefix}Error updating category: {e}")
            return False

    def rename_category(self, old_category_name: str, new_category_name: str) -> int:
        """Renames a category for all rules within it."""
        log_prefix = f"RuleLibraryManager.rename_category(Old='{old_category_name}', New='{new_category_name}'): "
        normalized_old = old_category_name.strip().title()
        normalized_new = new_category_name.strip().title()
        if not normalized_old or not normalized_new or normalized_old == normalized_new:
            logger.warning(f"{log_prefix}Invalid rename request (empty or same names).")
            return 0

        logger.info(f"{log_prefix}Renaming category '{normalized_old}' to '{normalized_new}'.")
        count = 0
        modified = False
        for rule_name, rule_data in self.rules.items():
            current_cat_norm = rule_data.get('category', 'Unknown').strip().title()
            if current_cat_norm == normalized_old:
                rule_data['category'] = normalized_new
                rule_data['date_modified'] = datetime.now().strftime("%Y-%m-%d")
                count += 1
                modified = True
                # Clear cache for modified rule
                if rule_name in RuleLibraryManager._rules_cache:
                    del RuleLibraryManager._rules_cache[rule_name]

        if modified:
            self.save_library()
            logger.info(f"{log_prefix}Renamed category for {count} rules.")
        else:
            logger.info(f"{log_prefix}No rules found in category '{normalized_old}'.")
        return count

    def delete_category(self, category_name: str, move_to_category: str = "Uncategorized") -> int:
        """Deletes a category by moving its rules to another category (default: Uncategorized)."""
        log_prefix = f"RuleLibraryManager.delete_category(Category='{category_name}', MoveTo='{move_to_category}'): "
        normalized_target = category_name.strip().title()
        normalized_move_to = move_to_category.strip().title()
        if not normalized_target or normalized_target == normalized_move_to:
            logger.warning(f"{log_prefix}Invalid delete request (empty target or same as move-to).")
            return 0

        logger.info(f"{log_prefix}Deleting category '{normalized_target}' by moving rules to '{normalized_move_to}'.")
        count = 0
        modified = False
        for rule_name, rule_data in self.rules.items():
            current_cat_norm = rule_data.get('category', 'Unknown').strip().title()
            if current_cat_norm == normalized_target:
                rule_data['category'] = normalized_move_to
                rule_data['date_modified'] = datetime.now().strftime("%Y-%m-%d")
                count += 1
                modified = True
                # Clear cache for modified rule
                if rule_name in RuleLibraryManager._rules_cache:
                    del RuleLibraryManager._rules_cache[rule_name]

        if modified:
            self.save_library()
            logger.info(f"{log_prefix}Moved {count} rules from '{normalized_target}' to '{normalized_move_to}'.")
        else:
            logger.info(f"{log_prefix}No rules found in category '{normalized_target}'.")
        return count

    def save_library(self, data_to_save: Optional[Dict[str, Any]] = None):
        """
        Save current library state to JSON, creating a backup first.
        Optionally saves the provided data structure instead of self.rules.
        (Round 12 Fix: Correctly add data_to_save parameter)
        """
        log_prefix = "RuleLibraryManager.save_library: "
        logger.debug(f"{log_prefix}Attempting to save rule library to: {self.library_path}")
        try:
            # [ Backup logic remains the same ]
            backup_created = False
            if os.path.exists(self.library_path) and os.path.getsize(self.library_path) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # --- CORRECTED: Use config_rules_backups ---
                backup_dir = self.app_paths.get('config_rules_backups')
                if not backup_dir:
                    logger.error(f"{log_prefix}Backup directory path ('config_rules_backups') not found in app_paths. Cannot backup.")
                else:
                    os.makedirs(backup_dir, exist_ok=True) # Ensure directory exists
                    backup_path = os.path.join(backup_dir, f"rules_backup_{timestamp}.json")
                    logger.info(f"{log_prefix}Creating backup of rules.json at: {backup_path}")
                    try:
                        with open(self.library_path, 'r', encoding='utf-8') as original_file:
                            backup_content = original_file.read()
                        if backup_content.strip():
                            with open(backup_path, 'w', encoding='utf-8') as backup_file:
                                backup_file.write(backup_content)
                            logger.info(f"{log_prefix}Backup created successfully at {backup_path}")
                            backup_created = True
                        else: logger.info(f"{log_prefix}Original file is empty, skipping backup")
                    except FileNotFoundError: logger.info(f"{log_prefix}Original file not found, skipping backup")
                    except Exception as e: logger.error(f"{log_prefix}Error reading original file for backup: {e}")
            else: logger.info(f"{log_prefix}Original file doesn't exist or is empty, skipping backup")
            # ---

            # --- Prepare data for saving ---
            # --- CORRECTED: Use data_to_save if provided, otherwise use self.rules ---
            if data_to_save is not None:
                logger.debug(f"{log_prefix}Saving provided data_to_save structure.")
                library_data_for_json = data_to_save
            else:
                logger.debug(f"{log_prefix}Saving current self.rules.")
                rules_list_to_save = list(self.rules.values()) # Convert dict values to list
                library_data_for_json = {
                    'library_metadata': {
                        'version': '1.0',
                        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'description': 'Lace Rule Library'
                    },
                    'rules': rules_list_to_save
                }
            # --- END CORRECTED ---

            num_rules_to_save = len(library_data_for_json.get('rules', []))
            logger.info(f"{log_prefix}Preparing to save {num_rules_to_save} rules.")
            logger.debug(f"{log_prefix}Data structure check before dump:")
            logger.debug(f"  len(rules): {num_rules_to_save}")

            # --- Write to temporary file with error handling ---
            temp_path = f"{self.library_path}.tmp"
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(library_data_for_json, f, indent=2, ensure_ascii=False)
                logger.debug(f"{log_prefix}Successfully wrote {num_rules_to_save} rules to temporary file: {temp_path}")
            except TypeError as te:
                 logger.error(f"{log_prefix}JSON SERIALIZATION ERROR: {te}")
                 for i, rule_item in enumerate(library_data_for_json.get('rules', [])):
                     try: json.dumps(rule_item)
                     except TypeError: logger.error(f"Serialization failed for rule at index {i}: {rule_item.get('name', 'N/A')}")
                 raise
            except Exception as dump_err:
                 logger.error(f"{log_prefix}Error during json.dump: {dump_err}")
                 raise
            # ---

            # [ Verification and Move - Unchanged ]
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                logger.debug(f"{log_prefix}Temporary file {temp_path} created successfully.")
                shutil.move(temp_path, self.library_path)
                logger.info(f"{log_prefix}Rule library saved successfully to {self.library_path}")
            else:
                logger.error(f"{log_prefix}Failed to write to temporary save file or file is empty.")
                if os.path.exists(temp_path): os.remove(temp_path)

        except Exception as e:
            logger.error(f"{log_prefix}Error saving rule library: {e}")
            logger.error(traceback.format_exc())

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

    def reload_library(self):
        """Reload the rule library from the JSON file"""
        try:
            logger.info("Reloading rule library")
            RuleLibraryManager._rules_cache.clear()  # Clear the cache
            self.load_library()  # Reload from JSON
            logger.info("Rule library reloaded successfully")
        except Exception as e:
            logger.error(f"Error reloading rule library: {e}")

    def regenerate_library(self):
        """Regenerate the rules.json file using RuleLibrary data."""
        try:
            logger.info("Starting regeneration of rules.json")

            # Create a backup of the existing rules.json file
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(APP_PATHS['rules_backups'], f"rules_backup_{backup_timestamp}.json")
            logger.info(f"Creating backup at {backup_path}")
            try:
                shutil.copyfile(self.library_path, backup_path)
            except FileNotFoundError:
                logger.warning(f"No existing rules.json found at {self.library_path}, skipping backup.")
            except Exception as e:
                logger.error(f"Error creating backup: {e}")
                raise

            # Create a new rules dictionary
            new_rules = []

            # Iterate through all rules in RuleLibrary
            for rule_name in RuleLibrary.get_rule_names():
                try:
                    # Get rule data from library
                    rule_data = RuleLibraryManager.get_rule(rule_name)

                    # Get the factory function
                    rule_type = rule_data['type']
                    rule_class = globals()[rule_type]

                    # Create metadata dictionary *without* 'params'
                    metadata_dict = {k: v for k, v in rule_data.items() if k != 'params'}
                    metadata = RuleMetadata(**metadata_dict)

                    # Create a temporary instance of the rule to access its metadata
                    rule = rule_class(metadata)

                    # Get parameters from the rule
                    params = copy.deepcopy(rule.params)

                    # Create a dictionary for the rule
                    new_rule_data = {
                        'name': rule.name,
                        'type': rule_type,
                        'position': rule_data['position'],
                        'category': rule.metadata.category,
                        'author': rule.metadata.author,
                        'url': rule.metadata.url,
                        'email': rule.metadata.email,
                        'date_created': rule.metadata.date_created,
                        'date_modified': datetime.now().strftime("%Y-%m-%d"),
                        'version': rule.metadata.version,
                        'description': rule.metadata.description,
                        'tags': rule.metadata.tags,
                        'dimension_compatibility': rule.metadata.dimension_compatibility,
                        'neighborhood_compatibility': rule.metadata.neighborhood_compatibility,
                        'parent_rule': rule.metadata.parent_rule,
                        'rating': rule.metadata.rating,
                        'notes': rule.metadata.notes,
                        'allowed_initial_conditions': rule.metadata.allowed_initial_conditions,
                        'allow_rule_tables': rule.metadata.allow_rule_tables,
                        'params': params
                    }
                    new_rules.append(new_rule_data)

                    logger.debug(f"Successfully processed rule: {rule_name}")

                except Exception as e:
                    logger.error(f"Error processing rule {rule_name}: {e}")
                    logger.error(traceback.format_exc())
                    continue

            # Create the library data
            library_data = {
                'library_metadata': {
                    'version': '1.0',
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'description': 'Lace Rule Library'
                },
                'rules': new_rules
            }

            # Save the library data to the JSON file
            try:
                with open(self.library_path, 'w') as f:
                    json.dump(library_data, f, indent=2)
                logger.info(f"Successfully regenerated rules.json at {self.library_path}")
            except Exception as e:
                logger.error(f"Error saving regenerated rules.json: {e}")
                raise

        except Exception as e:
            logger.error(f"Error regenerating rule library: {e}")
            raise

# =========== END of rules.py ===========