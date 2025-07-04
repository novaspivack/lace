# =========== START of presets.py ===========
from __future__ import annotations
import json
import ast
import shutil
import threading
import setproctitle
from datetime import datetime
import dataclasses
from tkinter import simpledialog
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from typing import Dict, List, Tuple, Optional, Union, Any, cast, TypeVar, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
from tkinter import messagebox, simpledialog
import matplotlib.pyplot as plt
plt.ioff()
from enum import Enum
from dataclasses import dataclass, field
import os
from datetime import datetime
import traceback
import numpy as np
import warnings
import cProfile
import pstats
import queue

 
from .logging_config import logger, APP_PATHS, APP_DIR
from .enums import Dimension, NeighborhoodType
from .settings import GlobalSettings
from .shapes import (
    ShapeLibraryManager
    )
# from .rules import RuleLibrary
from .initial_conditions import InitialConditionManager


# --- ADDED: TYPE_CHECKING block ---
if TYPE_CHECKING:
    # Import SimulationGUI only for type checking purposes
    from lace_app import SimulationGUI
# --- END ADDED ---

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


@dataclass
class GridPreset:
    """Represents a pre-calculated grid state or initialization method."""
    name: str
    dimensions: Tuple[int, ...]
    neighborhood_type: str
    rule_name: str # Rule associated with this preset

    # --- Initialization Control ---
    initialization_mode: str = "SAVED_STATE" # Options: SAVED_STATE, RULE_DEFAULT, SPECIFIC_CONDITION, LIBRARY_SHAPE
    initialization_data: Optional[str] = None # Name of condition or shape, if applicable

    # --- State/Structure Data (Only used if mode is SAVED_STATE) ---
    initial_state: Optional[npt.NDArray[np.float64]] = None # Store initial state ONLY if mode is SAVED_STATE
    edges: Optional[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = None # Store edges ONLY if mode is SAVED_STATE
    edge_states: Optional[Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float]] = field(default_factory=dict) # Store edge states ONLY if mode is SAVED_STATE

    # --- Metadata ---
    description: str = ""
    initial_conditions: str = "Pattern" # DEPRECATED - Use initialization_mode instead, kept for loading old files
    node_density: float = 0.5 # Store for info/fallback
    edge_density: float = 0.5 # Store for info/fallback

    # --- ADDED: InitVar for sparse state loading ---
    initial_state_sparse: dataclasses.InitVar[Optional[Dict[str, float]]] = None
    # ---

    def to_dict(self) -> dict:
        """Convert the grid preset to a dictionary for saving, using sparse representation for initial_state."""
        def convert_value(value):
            """Recursively convert values to JSON serializable types."""
            # (Keep existing convert_value logic for other fields)
            if isinstance(value, np.ndarray):
                return value.tolist() # Keep for non-initial_state arrays if any
            elif isinstance(value, (list, tuple)):
                if all(isinstance(x, (int, float, str, bool, type(None))) for x in value):
                    return list(value)
                else:
                    return type(value)(convert_value(x) for x in value)
            elif isinstance(value, dict):
                new_dict = {}
                for k, v in value.items():
                    key_str = str(k)
                    if isinstance(k, tuple) and all(isinstance(i, tuple) for i in k): # Edge tuple key
                        key_str = str(tuple(list(n) for n in k))
                    elif isinstance(k, tuple) and all(isinstance(i, int) for i in k): # Coordinate tuple key
                         key_str = str(list(k)) # Convert coord tuple key to list string "[r, c, ...]"
                    new_dict[key_str] = convert_value(v)
                return new_dict
            elif isinstance(value, np.integer): return int(value)
            elif isinstance(value, (np.floating, float)): return float(value)
            elif isinstance(value, Enum): return value.name
            else: return value

        # --- Sparse State Conversion ---
        initial_state_for_json = None
        initial_state_sparse_for_json = None
        if self.initialization_mode == "SAVED_STATE" and self.initial_state is not None:
            logger.debug(f"Preset '{self.name}': Converting initial_state to sparse format for saving.")
            sparse_dict = {}
            # Use a small threshold to identify non-zero states reliably
            activity_threshold = 1e-6
            non_zero_indices = np.argwhere(np.abs(self.initial_state) > activity_threshold)
            for coord_array in non_zero_indices:
                coord_tuple = tuple(coord_array)
                state_value = float(self.initial_state[coord_tuple])
                sparse_dict[coord_tuple] = state_value

            # Convert coordinate tuple keys to list strings for JSON
            initial_state_sparse_for_json = {str(list(k)): v for k, v in sparse_dict.items()}
            logger.debug(f"  Created sparse dict with {len(initial_state_sparse_for_json)} non-zero entries.")
            # Set initial_state to None for the JSON output as sparse dict replaces it
            initial_state_for_json = None
        elif self.initial_state is not None:
             # If mode is not SAVED_STATE but state exists (shouldn't happen after post_init), store None
             initial_state_for_json = None
        # ---

        # Convert edges (remains the same)
        edges_for_json = None
        if self.edges is not None:
            edges_for_json = []
            for edge in self.edges:
                node1 = list(edge[0]) if isinstance(edge[0], tuple) else edge[0]
                node2 = list(edge[1]) if isinstance(edge[1], tuple) else edge[1]
                edges_for_json.append([node1, node2])

        # Convert edge_states (remains the same)
        edge_states_for_json = None
        if self.edge_states is not None:
            edge_states_for_json = {}
            for edge, state in self.edge_states.items():
                node1 = list(edge[0]) if isinstance(edge[0], tuple) else edge[0]
                node2 = list(edge[1]) if isinstance(edge[1], tuple) else edge[1]
                edge_str = str((node1, node2))
                edge_states_for_json[edge_str] = state

        return {
            'name': self.name,
            'dimensions': convert_value(self.dimensions),
            'neighborhood_type': self.neighborhood_type,
            'rule_name': self.rule_name,
            'initialization_mode': self.initialization_mode,
            'initialization_data': self.initialization_data,
            # --- Use new sparse dict key ---
            'initial_state': initial_state_for_json, # Will be None if sparse is used
            'initial_state_sparse': initial_state_sparse_for_json, # Store the sparse dict
            # ---
            'edges': convert_value(edges_for_json) if self.initialization_mode == "SAVED_STATE" and edges_for_json is not None else None,
            'edge_states': convert_value(edge_states_for_json) if self.initialization_mode == "SAVED_STATE" and edge_states_for_json is not None else None,
            'description': self.description,
            'initial_conditions': self.initial_conditions,
            'node_density': convert_value(self.node_density),
            'edge_density': convert_value(self.edge_density)
        }

    # --- MODIFIED: Accept initial_state_sparse InitVar ---
    def __post_init__(self, initial_state_sparse: Optional[Dict[str, float]] = None):
    # ---
        """Post-initialization to ensure data consistency and reconstruct sparse state."""
        log_prefix = f"GridPreset.__post_init__(Name='{self.name}'): "
        logger.debug(f"{log_prefix}START. Initial Mode='{self.initialization_mode}', Initial Conditions='{self.initial_conditions}'")

        # --- MODIFIED: Sparse State Reconstruction using InitVar ---
        # Check if sparse data exists (passed via InitVar) and full state is missing (or None)
        if self.initialization_mode == "SAVED_STATE" and initial_state_sparse is not None and self.initial_state is None:
            logger.info(f"{log_prefix}Found sparse state data, reconstructing full initial_state array.")
            try:
                reconstructed_state = np.zeros(tuple(self.dimensions), dtype=np.float64)
                count = 0
                if isinstance(initial_state_sparse, dict):
                    for key_str, state_val in initial_state_sparse.items():
                        try:
                            # Key is string representation of list "[r, c, ...]"
                            coord_list = ast.literal_eval(key_str)
                            coord_tuple = tuple(map(int, coord_list))
                            if len(coord_tuple) == len(self.dimensions):
                                # Check bounds before assignment
                                if all(0 <= c < d for c, d in zip(coord_tuple, self.dimensions)):
                                    reconstructed_state[coord_tuple] = float(state_val)
                                    count += 1
                                else:
                                    logger.warning(f"  Skipping sparse coord out of bounds: {coord_tuple} for dims {self.dimensions}")
                            else:
                                logger.warning(f"  Skipping sparse coord with wrong dimension: {coord_tuple} vs {self.dimensions}")
                        except (ValueError, SyntaxError, TypeError) as parse_err:
                            logger.warning(f"  Error parsing sparse state key '{key_str}': {parse_err}")
                else:
                     logger.warning(f"  initial_state_sparse is not a dict ({type(initial_state_sparse)}), cannot reconstruct.")

                self.initial_state = reconstructed_state # Assign the reconstructed array
                logger.info(f"{log_prefix}Reconstructed initial_state from sparse data ({count} cells set). Shape: {self.initial_state.shape}")
                # No need to delete the attribute as InitVar is not stored

            except Exception as e:
                logger.error(f"{log_prefix}Error reconstructing initial_state from sparse data: {e}")
                self.initial_state = None # Set to None on error
        # --- END MODIFIED ---

        # --- Handle Deprecated 'initial_conditions' (Keep existing logic) ---
        mode_changed_by_migration = False
        if self.initialization_mode == "SAVED_STATE" and self.initial_conditions != "Pattern":
            original_mode = self.initialization_mode
            logger.info(f"{log_prefix}Detected potential deprecated 'initial_conditions' ('{self.initial_conditions}') with mode '{original_mode}'. Attempting migration.")
            if self.initial_conditions == "Random":
                self.initialization_mode = "RULE_DEFAULT"
                self.initialization_data = None
                logger.info(f"  Migrated 'Random' to RULE_DEFAULT.")
            elif self.initial_conditions:
                shape_manager = ShapeLibraryManager.get_instance()
                if shape_manager.get_shape(self.initial_conditions):
                    self.initialization_mode = "LIBRARY_SHAPE"
                    self.initialization_data = self.initial_conditions
                    logger.info(f"  Migrated '{self.initial_conditions}' to LIBRARY_SHAPE.")
                else:
                    self.initialization_mode = "SPECIFIC_CONDITION"
                    self.initialization_data = self.initial_conditions
                    logger.info(f"  Migrated '{self.initial_conditions}' to SPECIFIC_CONDITION.")
            else:
                 logger.warning(f"  Deprecated 'initial_conditions' is empty/None, keeping mode as {original_mode} but clearing state/edge data.")
                 self.initialization_mode = original_mode

            if self.initialization_mode != "SAVED_STATE":
                self.initial_state = None # Clear state if mode changed away
                self.edges = None
                self.edge_states = None
                mode_changed_by_migration = True
                logger.info(f"  Cleared state/edge data due to mode change from '{original_mode}' to '{self.initialization_mode}'.")
            else:
                 logger.info(f"  Mode remained '{self.initialization_mode}', state/edge data preserved (if present).")
        # ---

        # --- Ensure state/edge data is None if mode is not SAVED_STATE (Keep existing logic) ---
        if self.initialization_mode != "SAVED_STATE":
            if self.initial_state is not None or self.edges is not None or (self.edge_states and len(self.edge_states) > 0):
                logger.debug(f"{log_prefix}Clearing state/edge data because initialization_mode is '{self.initialization_mode}' (not SAVED_STATE).")
                self.initial_state = None
                self.edges = None
                self.edge_states = {}
        # ---

        # --- Validate SAVED_STATE data if mode is SAVED_STATE (Keep existing logic, now works on potentially reconstructed state) ---
        if self.initialization_mode == "SAVED_STATE":
            logger.debug(f"{log_prefix}Mode is SAVED_STATE. Validating saved data.")
            if self.initial_state is not None and not isinstance(self.initial_state, np.ndarray):
                try:
                    self.initial_state = np.array(self.initial_state, dtype=np.float64)
                    logger.debug(f"  Converted initial_state list to ndarray (Shape: {self.initial_state.shape})")
                    if self.initial_state.shape != tuple(self.dimensions):
                        logger.warning(f"  Saved state shape {self.initial_state.shape} mismatch dimensions {self.dimensions}. Clearing state.")
                        self.initial_state = None
                except Exception as e:
                    logger.error(f"  Error converting saved initial_state to numpy array: {e}. Clearing state.")
                    self.initial_state = None
            elif self.initial_state is not None and self.initial_state.shape != tuple(self.dimensions):
                 logger.warning(f"  Reconstructed state shape {self.initial_state.shape} mismatch dimensions {self.dimensions}. Clearing state.")
                 self.initial_state = None

            # [ Edge and Edge State validation remains the same ]
            if self.edges is not None:
                converted_edges = []
                if isinstance(self.edges, (list, tuple)):
                    for i, edge in enumerate(self.edges):
                        try:
                            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                                node1, node2 = edge
                                node1_tuple = tuple(map(int, node1)) if isinstance(node1, (list, tuple)) else None
                                node2_tuple = tuple(map(int, node2)) if isinstance(node2, (list, tuple)) else None
                                if node1_tuple and node2_tuple and len(node1_tuple) == len(self.dimensions) and len(node2_tuple) == len(self.dimensions):
                                    ordered_edge = (node1_tuple, node2_tuple) if node1_tuple < node2_tuple else (node2_tuple, node1_tuple)
                                    converted_edges.append(ordered_edge)
                                else: logger.warning(f"  Skipping invalid edge format/dimension at index {i}: {edge}")
                            else: logger.warning(f"  Skipping invalid edge format at index {i}: {edge}")
                        except (TypeError, ValueError) as e: logger.warning(f"  Error converting edge at index {i}: {edge} - {e}")
                else: logger.warning(f"  Edges data is not list/tuple. Clearing edges.")
                self.edges = converted_edges
                logger.debug(f"  Validated/Converted {len(self.edges)} edges.")

            if self.edge_states is not None:
                converted_edge_states = {}
                if isinstance(self.edge_states, dict):
                    for edge_key, state in self.edge_states.items():
                        try:
                            # Handle both string representation "([r1,c1,...], [r2,c2,...])" and actual tuple keys
                            edge_tuple_repr = edge_key
                            if isinstance(edge_key, str):
                                # Safely evaluate the string representation
                                edge_tuple_repr = ast.literal_eval(edge_key)

                            if isinstance(edge_tuple_repr, (list, tuple)) and len(edge_tuple_repr) == 2:
                                node1, node2 = edge_tuple_repr
                                # Ensure inner elements are tuples of ints
                                node1_tuple = tuple(map(int, node1)) if isinstance(node1, (list, tuple)) else None
                                node2_tuple = tuple(map(int, node2)) if isinstance(node2, (list, tuple)) else None

                                if node1_tuple and node2_tuple and len(node1_tuple) == len(self.dimensions) and len(node2_tuple) == len(self.dimensions):
                                    ordered_edge = (node1_tuple, node2_tuple) if node1_tuple < node2_tuple else (node2_tuple, node1_tuple)
                                    converted_edge_states[ordered_edge] = float(state)
                                else: logger.warning(f"  Skipping invalid edge state key format/dimension: {edge_key}")
                            else: logger.warning(f"  Skipping invalid edge state key format: {edge_key}")
                        except (SyntaxError, ValueError, TypeError) as e: logger.warning(f"  Error converting edge state key: {edge_key} - {e}")
                else: logger.warning(f"  Edge states data is not dict. Clearing edge states.")
                self.edge_states = converted_edge_states
                logger.debug(f"  Validated/Converted {len(self.edge_states)} edge states.")
        # ---

        logger.debug(f"{log_prefix}END. Final Mode: '{self.initialization_mode}', State is None: {self.initial_state is None}, Edges is None: {self.edges is None}")

class GridPresetManager:
    """Manages loading, storing, and accessing grid presets."""

    _instance: Optional['GridPresetManager'] = None

    def __init__(self, app_paths: Dict[str, str]):
        """Initializes the GridPresetManager.
           (Round 38: Store app_paths)"""
        # Use config_presets for the default path
        self.app_paths = app_paths # STORE app_paths
        if 'config_presets' in app_paths:
            self.presets_path = os.path.join(app_paths['config_presets'], 'grid_presets.json')
        else:
            # Fallback to using the base path with a config/presets subdirectory
            if 'config' in app_paths:
                config_presets_dir = os.path.join(app_paths['config'], 'presets')
                os.makedirs(config_presets_dir, exist_ok=True)  # Create config/presets directory if it doesn't exist
                self.presets_path = os.path.join(config_presets_dir, 'grid_presets.json')
            elif 'logs' in app_paths:  # Use any existing key to get the base path
                base_dir = os.path.dirname(app_paths['logs'])
                config_presets_dir = os.path.join(base_dir, 'config', 'presets')
                os.makedirs(config_presets_dir, exist_ok=True)  # Create config/presets directory if it doesn't exist
                self.presets_path = os.path.join(config_presets_dir, 'grid_presets.json')
            else:
                # Last resort fallback
                self.presets_path = os.path.join(os.getcwd(), APP_DIR, 'Resources', 'config', 'presets', 'grid_presets.json')
                os.makedirs(os.path.dirname(self.presets_path), exist_ok=True)

        self.presets: Dict[str, GridPreset] = {}
        self.default_preset_name: Optional[str] = None
        self.load_presets()

    def _backup_presets_file(self) -> bool:
        """Create a backup of the grid presets file before modifying it."""
        try:
            if not os.path.exists(self.presets_path):
                logger.info("No grid presets file to backup")
                return True # Nothing to backup, proceed

            # Determine backup directory (using app_paths if available)
            backup_dir = None
            if hasattr(self, 'app_paths') and 'config_rules_backups' in self.app_paths:
                # Use the rules_backups directory for preset backups too
                backup_dir = self.app_paths['config_rules_backups']
            elif hasattr(self, 'app_paths') and 'config' in self.app_paths:
                # Create a backups directory under config
                backup_dir = os.path.join(self.app_paths['config'], 'backups')
            else: # Fallback if app_paths not available or missing keys
                backup_dir = os.path.join(os.path.dirname(self.presets_path), 'backups')

            if backup_dir:
                os.makedirs(backup_dir, exist_ok=True)

                # Create backup filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(backup_dir, f"grid_presets_backup_{timestamp}.json")

                # Copy the file
                shutil.copyfile(self.presets_path, backup_path)
                logger.info(f"Created backup of grid presets at {backup_path}")
                return True
            else:
                logger.warning("Could not determine backup directory for presets, skipping backup")
                return False

        except Exception as e:
            logger.error(f"Error creating backup of grid presets file: {e}")
            return False
        
    def load_presets(self):
        """Load grid presets from the JSON file, handling sparse state representation.
           (Round 13: Handle initial_state_sparse key)"""
        try:
            self._backup_presets_file()

            with open(self.presets_path, 'r') as f:
                data = json.load(f)

            presets = {}
            if 'presets' not in data or not isinstance(data['presets'], list):
                 logger.error(f"Invalid preset file format: 'presets' key missing or not a list in {self.presets_path}")
                 data['presets'] = []

            for preset_data in data['presets']:
                if not isinstance(preset_data, dict):
                    logger.warning(f"Skipping invalid preset entry (not a dictionary): {preset_data}")
                    continue
                if 'name' not in preset_data:
                     logger.warning(f"Skipping preset entry missing 'name': {preset_data}")
                     continue

                # --- MODIFIED: Pass initial_state_sparse if present ---
                # The GridPreset.__post_init__ will handle the reconstruction.
                # We just need to ensure the dictionary from JSON is passed.
                # No specific handling needed here for initial_state_sparse,
                # as **preset_data will pass it along if it exists.
                # We still need to handle the OLD initial_state list format for backward compatibility.
                initial_state_data = preset_data.get('initial_state')
                if initial_state_data is not None and isinstance(initial_state_data, list):
                     try:
                         # Convert old list format to numpy array here before passing
                         preset_data['initial_state'] = np.array(initial_state_data, dtype=np.float64)
                     except ValueError:
                          logger.warning(f"Could not convert old initial_state list to numpy array for preset '{preset_data['name']}'. Setting to None.")
                          preset_data['initial_state'] = None
                elif initial_state_data is not None and not isinstance(initial_state_data, np.ndarray):
                     logger.warning(f"Unexpected type for old initial_state in preset '{preset_data['name']}': {type(initial_state_data)}. Setting to None.")
                     preset_data['initial_state'] = None
                # If initial_state_sparse exists, initial_state should ideally be None or absent
                # If both exist, __post_init__ will prioritize sparse if initial_state is None.
                # --- END MODIFIED ---

                # --- Edge/Edge State loading remains the same ---
                edge_states = preset_data.get('edge_states')
                if edge_states is None: edge_states = {}
                converted_edge_states = {}
                if isinstance(edge_states, dict):
                    for edge_str, state in edge_states.items():
                        try:
                            edge = ast.literal_eval(edge_str)
                            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                                node1, node2 = edge
                                node1_tuple = tuple(map(int, node1)) if isinstance(node1, (list, tuple)) else None
                                node2_tuple = tuple(map(int, node2)) if isinstance(node2, (list, tuple)) else None
                                if node1_tuple and node2_tuple:
                                    ordered_edge = tuple(sorted((node1_tuple, node2_tuple)))
                                    converted_edge_states[ordered_edge] = float(state)
                                else: logger.warning(f"Skipping invalid edge state format (nodes not tuples/ints): {edge}")
                            else: logger.warning(f"Skipping invalid edge state format (edge not pair): {edge}")
                        except (SyntaxError, ValueError, TypeError) as e: logger.warning(f"Skipping invalid edge state key format: {edge_str} - {e}")
                preset_data['edge_states'] = converted_edge_states

                edges_list = preset_data.get('edges')
                if edges_list is None: edges_list = []
                converted_edges = []
                if isinstance(edges_list, list):
                    for edge in edges_list:
                         if isinstance(edge, list) and len(edge) == 2:
                             node1, node2 = edge
                             node1_tuple = tuple(map(int, node1)) if isinstance(node1, (list, tuple)) else None
                             node2_tuple = tuple(map(int, node2)) if isinstance(node2, (list, tuple)) else None
                             if node1_tuple and node2_tuple:
                                 converted_edges.append(tuple(sorted((node1_tuple, node2_tuple))))
                             else: logger.warning(f"Skipping invalid edge format in preset '{preset_data.get('name', 'Unknown')}': {edge}")
                         else: logger.warning(f"Skipping invalid edge format in preset '{preset_data.get('name', 'Unknown')}': {edge}")
                preset_data['edges'] = converted_edges
                # ---

                preset_data['node_density'] = preset_data.get('node_density', 0.5)
                preset_data['edge_density'] = preset_data.get('edge_density', 0.5)

                try:
                    required_fields = ['name', 'dimensions', 'neighborhood_type', 'rule_name']
                    if not all(field in preset_data for field in required_fields):
                        logger.error(f"Preset '{preset_data.get('name', 'MISSING_NAME')}' is missing required fields. Skipping.")
                        continue
                    # Pass the potentially modified preset_data dictionary
                    presets[preset_data['name']] = GridPreset(**preset_data)
                except TypeError as te:
                     logger.error(f"Error creating GridPreset '{preset_data.get('name', 'Unknown')}' due to missing/invalid fields: {te}")
                     logger.error(f"Preset data causing error: {preset_data}")
                     continue

            self.presets = presets
            self.default_preset_name = data.get('default_preset', None)
            logger.info(f"Loaded {len(self.presets)} grid presets from {self.presets_path}")
        except FileNotFoundError:
            logger.warning(f"Grid presets file not found at {self.presets_path}")
            self._create_default_preset_file()
            self.load_presets()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {self.presets_path}: {e}")
            messagebox.showerror("Error", f"Invalid JSON format in grid presets file: {e}")
            self.presets = {}
            self.default_preset_name = None
        except Exception as e:
            logger.error(f"Error loading grid presets: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to load grid presets: {e}")
            self.presets = {}
            self.default_preset_name = None

    def _create_default_preset_file(self):
        """Create a default grid_presets.json file with 65% node density."""
        try:
            # --- MODIFIED: Set default densities ---
            default_node_density = 0.65 # Set desired node density
            default_edge_density = 0.0   # Keep edge density at 0 for GOL default
            # --- END MODIFIED ---

            preset_dimensions = (5, 5)
            initial_state_array = np.zeros(preset_dimensions, dtype=np.float64)

            # --- ADDED: Initialize initial_state_array based on density ---
            total_nodes = np.prod(preset_dimensions)
            active_cells = int(int(total_nodes) * default_node_density)
            logger.debug(f"Default Preset: Target active cells = {active_cells} (Density: {default_node_density})") # Log target

            if active_cells > 0:
                active_cells = min(active_cells, total_nodes) # Ensure not more than total nodes
                active_indices = np.random.choice(total_nodes, size=active_cells, replace=False)
                initial_state_array.ravel()[active_indices] = 1.0
                # --- ADDED: Log actual active count in generated array ---
                actual_active_count = np.sum(initial_state_array > 0)
                logger.debug(f"Default Preset: Generated initial_state_array with {actual_active_count} active nodes.")
                # --- END ADDED ---
            # --- END ADDED ---

            default_preset = GridPreset(
                name="5 X 5 Moore 2D - Default",
                dimensions=preset_dimensions,
                neighborhood_type="MOORE",
                # --- MODIFIED: Use the generated initial_state_array ---
                initial_state=initial_state_array,
                # --- END MODIFIED ---
                edges=[],
                description="Default 2D 5x5 grid with Moore neighborhood and Game of Life rule (65% density).", # Updated description
                rule_name="Game of Life",
                initial_conditions="Random", # Keep as Random, even though we set state here
                # --- MODIFIED: Use the defined densities ---
                node_density=default_node_density,
                edge_density=default_edge_density
                # --- END MODIFIED ---
            )

            # Add the default preset to the presets dictionary
            self.presets[default_preset.name] = default_preset

            # Set the default preset name
            self.default_preset_name = default_preset.name # Use the actual name

            # Save the presets to the file
            self._save_presets_to_file()

            logger.info("Created default grid_presets.json file with 65% node density.")
        except Exception as e:
            logger.error(f"Error creating default grid presets file: {e}")
            messagebox.showerror("Error", f"Failed to create default grid presets file: {e}")

    def save_preset(self, preset: GridPreset):
        """Save a grid preset to the JSON file."""
        try:
            self.presets[preset.name] = preset
            self._save_presets_to_file()
            messagebox.showinfo("Success", f"Grid preset '{preset.name}' saved successfully.")
            logger.info(f"Saved grid preset: {preset.name}")
        except Exception as e:
            logger.error(f"Error saving grid preset: {e}")
            messagebox.showerror("Error", f"Failed to save grid preset: {e}")

    def delete_preset(self, preset_name: str):
        """Delete a grid preset from the JSON file."""
        try:
            if preset_name in self.presets:
                del self.presets[preset_name]
                self._save_presets_to_file()
                messagebox.showinfo("Success", f"Grid preset '{preset_name}' deleted successfully.")
                logger.info(f"Deleted grid preset: {preset_name}")
            else:
                logger.warning(f"Grid preset not found: {preset_name}")
                messagebox.showerror("Error", f"Grid preset '{preset_name}' not found.")
        except Exception as e:
            logger.error(f"Error deleting grid preset: {e}")
            messagebox.showerror("Error", f"Failed to delete grid preset: {e}")

    def _save_presets_to_file(self):
        """Save all grid presets to the JSON file."""
        try:
            data = {'presets': [preset.to_dict() for preset in self.presets.values()], 'default_preset': self.default_preset_name}
            with open(self.presets_path, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Saved grid presets to {self.presets_path}")
        except Exception as e:
            logger.error(f"Error saving grid presets to file: {e}")
            messagebox.showerror("Error", f"Failed to save grid presets to file: {e}")

    def get_preset(self, name: str) -> Optional[GridPreset]:
        """Get a grid preset by name."""
        return self.presets.get(name)

    def get_preset_by_dimensions(self, dimensions: Tuple[int, ...], neighborhood_type: NeighborhoodType) -> Optional[GridPreset]:
        """Get a grid preset by dimensions and neighborhood type."""
        for preset in self.presets.values():
            if (tuple(preset.dimensions) == dimensions and
                NeighborhoodType[preset.neighborhood_type] == neighborhood_type):
                return preset
        return None

    def set_default_preset(self, preset_name: str):
        """Set the default grid preset name."""
        if preset_name in self.presets:
            self.default_preset_name = preset_name
            self._save_presets_to_file()
            logger.info(f"Set default grid preset to: {preset_name}")
        else:
            logger.warning(f"Grid preset not found: {preset_name}")
            messagebox.showerror("Error", f"Grid preset '{preset_name}' not found.")

    @classmethod
    def get_instance(cls, app_paths: Dict[str, str]) -> 'GridPresetManager':
        """Get the singleton instance of the GridPresetManager."""
        if cls._instance is None:
            cls._instance = GridPresetManager(app_paths)
        return cls._instance
         
###########  PRESETS AND GRID SIZE  ########### 

class ResizePromptDialog(tk.Toplevel):
    """Custom dialog for handling grid resize options when placing shapes."""
    
    def __init__(self, parent, required_dims: Tuple[int, ...], current_dims: Tuple[int, ...], grid_is_empty: bool):
        super().__init__(parent)
        self.title("Resize Grid?")
        self.transient(parent)
        self.grab_set()
        self.result: Optional[Dict[str, Any]] = None # To store user choice {'action': 'resize'/'cancel', 'dimensions': tuple, 'clear_action': 'copy'/'clear'}

        self.required_dims = required_dims
        self.current_dims = current_dims
        self.grid_is_empty = grid_is_empty

        # Calculate padded dimensions
        # --- MODIFIED: Changed padding factor to 2.5 (150%) ---
        padding_factor = 2.5
        # ---
        min_padding = 5
        # Calculate individual padded dimensions first
        individual_padded_dims = [max(rd, int(rd * padding_factor), cd + min_padding) for rd, cd in zip(required_dims, current_dims)]
        # --- MODIFIED: Make padded dimensions symmetrical using max ---
        max_padded_dim = max(individual_padded_dims)
        self.padded_dims = tuple(max_padded_dim for _ in individual_padded_dims)
        # ---

        # --- Widgets ---
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Message
        req_str = "x".join(map(str, required_dims))
        curr_str = "x".join(map(str, current_dims))
        padded_str = "x".join(map(str, self.padded_dims)) # Use calculated padded dims
        message = f"The shape requires a grid of at least {req_str}.\n"
        message += f"The current grid size is {curr_str}.\n\n"
        message += "Choose an action:"
        ttk.Label(main_frame, text=message, justify=tk.LEFT).pack(pady=(0, 15))

        # Resize Options
        self.resize_choice = tk.StringVar(value="minimum") # Default choice
        ttk.Radiobutton(main_frame, text=f"Resize to Minimum ({req_str})",
                        variable=self.resize_choice, value="minimum").pack(anchor=tk.W)
        # --- MODIFIED: Use calculated padded_str in label ---
        ttk.Radiobutton(main_frame, text=f"Resize with Padding ({padded_str})",
                        variable=self.resize_choice, value="padded").pack(anchor=tk.W)
        # ---
        ttk.Radiobutton(main_frame, text="Enter Custom Size...",
                        variable=self.resize_choice, value="custom", command=self._ask_custom_size).pack(anchor=tk.W)
        ttk.Radiobutton(main_frame, text="Cancel Placement",
                        variable=self.resize_choice, value="cancel").pack(anchor=tk.W)
        self.custom_dims: Optional[Tuple[int,...]] = None # To store custom size if entered

        # Clear/Copy Options (only if grid is not empty)
        self.clear_copy_choice = tk.StringVar(value="copy") # Default to copy
        if not grid_is_empty:
            ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
            ttk.Label(main_frame, text="Existing grid content:").pack(anchor=tk.W)
            ttk.Radiobutton(main_frame, text="Attempt to Copy Content",
                            variable=self.clear_copy_choice, value="copy").pack(anchor=tk.W)
            ttk.Radiobutton(main_frame, text="Clear Grid Before Resizing",
                            variable=self.clear_copy_choice, value="clear").pack(anchor=tk.W)
        else:
             self.clear_copy_choice.set("clear") # Force clear if empty

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        ttk.Button(button_frame, text="OK", command=self._on_ok).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT)

        self.wait_window(self) # Make it modal

    def _ask_custom_size(self):
        """Prompt for custom size when 'Custom' radio is selected."""
        # Only ask if the custom radio is actually selected
        if self.resize_choice.get() == "custom":
            custom_size_str = simpledialog.askstring(
                "Custom Grid Size",
                "Enter custom size (e.g., 42,17 or 15,15,15):",
                parent=self,
                initialvalue=",".join(map(str, self.padded_dims)) # Suggest padded size
            )
            if custom_size_str:
                custom_size_str = custom_size_str.strip()
                try:
                    target_dims = tuple(int(d.strip()) for d in custom_size_str.split(','))
                    if not (2 <= len(target_dims) <= 3): raise ValueError("Invalid dimensions (must be 2 or 3)")
                    if any(d <= 0 for d in target_dims): raise ValueError("Dimensions must be positive")
                    if len(target_dims) != len(self.required_dims): raise ValueError(f"Dimension mismatch ({len(target_dims)}D vs {len(self.required_dims)}D)")
                    if any(td < rd for td, rd in zip(target_dims, self.required_dims)):
                         raise ValueError(f"Custom size must be at least {'x'.join(map(str, self.required_dims))}.")
                    self.custom_dims = target_dims # Store valid custom dims
                except Exception as e:
                    messagebox.showerror("Invalid Input", f"Invalid custom size: {e}", parent=self)
                    self.resize_choice.set("minimum") # Revert choice if custom input fails
                    self.custom_dims = None
            else:
                # User cancelled custom input, revert choice
                self.resize_choice.set("minimum")
                self.custom_dims = None

    def _on_ok(self):
        """Handle OK button click."""
        choice = self.resize_choice.get()
        clear_action = self.clear_copy_choice.get()

        if choice == "cancel":
            self.result = {"action": "cancel"}
        elif choice == "custom":
            if self.custom_dims: # Use stored custom dims if valid
                self.result = {"action": "resize", "dimensions": self.custom_dims, "clear_action": clear_action}
            else:
                # If custom was selected but no valid dims stored (e.g., user cancelled prompt), ask again
                self._ask_custom_size()
                if self.custom_dims: # Check again if they entered valid dims now
                     self.result = {"action": "resize", "dimensions": self.custom_dims, "clear_action": clear_action}
                else:
                     return # Keep dialog open if custom size still not provided/valid
        elif choice == "minimum":
            self.result = {"action": "resize", "dimensions": self.required_dims, "clear_action": clear_action}
        elif choice == "padded":
            self.result = {"action": "resize", "dimensions": self.padded_dims, "clear_action": clear_action}

        if self.result is not None:
             self.destroy()

    def _on_cancel(self):
        """Handle Cancel button click."""
        self.result = {"action": "cancel"}
        self.destroy()

class GridPresetManagementModal(tk.Toplevel):
    """Modal dialog for managing grid presets."""

    # --- ADDED: Type hints for attributes ---
    parent: tk.Tk
    gui: 'SimulationGUI'
    preset_manager: GridPresetManager
    category_listbox: tk.Listbox
    rules_listbox: tk.Listbox
    rename_button: tk.Button
    delete_button: tk.Button
    move_button: tk.Button
    # ---

    def __init__(self, parent, gui):
        super().__init__(parent)
        self.parent = parent
        self.gui = gui
        self.title("Manage Grid Presets")

        # --- Get parent height ---
        parent_height = parent.winfo_height()
        # Set a fixed width and use the parent's height
        modal_width = 750
        self.geometry(f"{modal_width}x{parent_height}")
        # ---

        self.transient(parent)
        self.grab_set()

        # --- CORRECTED: Use correct path key ---
        self.preset_manager = GridPresetManager.get_instance(self.gui.app_paths) # Use GUI's app_paths
        # ---

        self._create_widgets()

        # Bind window closing event
        self.protocol("WM_DELETE_WINDOW", self.on_close) # Use the correct method name

    def _create_widgets(self):
        """Create widgets for the modal dialog with scrollable list and Edit button.
           (Round 3 Fix: Correct method references and attribute name)"""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Create a frame to hold the listbox and scrollbar ---
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Create Scrollbar ---
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Create Listbox and configure scrollbar ---
        # --- CORRECTED: Use self.preset_listbox ---
        self.preset_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            exportselection=False,
            selectbackground="#0078D7",
            selectforeground="white"
        )
        self.preset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.preset_listbox.yview)
        # --- END CORRECTION ---

        self._populate_preset_list() # Populate with sorted list

        # --- Bind double-click to load ---
        # --- CORRECTED: Use self.preset_listbox ---
        self.preset_listbox.bind("<Double-Button-1>", lambda event: self._load_preset())
        # --- Bind selection change ---
        self.preset_listbox.bind("<<ListboxSelect>>", self._on_preset_select) # Bind selection change
        # --- END CORRECTION ---

        # Button frame (Packed at the bottom)
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(10, 0)) # Pack at the bottom

        # Create buttons (Commands now reference methods that will be added)
        ttk.Button(button_frame, text="Create New", command=self._open_create_preset_modal).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load", command=self._load_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Edit Selected", command=self._edit_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete", command=self._delete_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Set as Default", command=self._set_as_default).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=self.on_close).pack(side=tk.RIGHT, padx=5) # Use correct method name

    # --- ADDED Missing Methods ---
    def _populate_preset_list(self):
        """Populate the preset listbox with available presets, sorted alphabetically."""
        self.preset_listbox.delete(0, tk.END)
        sorted_preset_names = sorted(self.preset_manager.presets.keys())
        for preset_name in sorted_preset_names:
            self.preset_listbox.insert(tk.END, preset_name)
            if preset_name == self.preset_manager.default_preset_name:
                 self.preset_listbox.itemconfig(tk.END, {'bg':'#d9ead3'})

    def _on_preset_select(self, event=None):
        """Placeholder for handling preset selection if needed (e.g., enabling buttons)."""
        # You can add logic here if buttons need enabling/disabling based on selection
        pass

    def _load_preset(self):
        """Load the selected grid preset."""
        selected_indices = self.preset_listbox.curselection() # Use correct attribute
        if not selected_indices:
            messagebox.showerror("Error", "No grid preset selected.", parent=self)
            return

        selected_name = self.preset_listbox.get(selected_indices[0]) # Use correct attribute
        preset = self.preset_manager.get_preset(selected_name)
        if preset:
            self.gui.apply_grid_preset(preset) # Call GUI method
            self.on_close() # Close after loading
        else:
            messagebox.showerror("Error", f"Grid preset '{selected_name}' not found.", parent=self)

    def _edit_preset(self):
        """Open the Create/Edit modal pre-filled with the selected preset's data."""
        selected_indices = self.preset_listbox.curselection() # Use correct attribute
        if not selected_indices:
            messagebox.showerror("Error", "No preset selected to edit.", parent=self)
            return

        selected_name = self.preset_listbox.get(selected_indices[0]) # Use correct attribute
        preset_to_edit = self.preset_manager.get_preset(selected_name)

        if preset_to_edit:
            logger.info(f"Editing preset: {selected_name}")
            # Open the create/edit window, passing the preset object
            # Also pass a reference back to this management window for refreshing
            self.gui._open_create_grid_preset_modal(preset_to_edit=preset_to_edit, manager_window=self)
            # Keep this management window open while editing
        else:
            messagebox.showerror("Error", f"Could not load preset '{selected_name}' for editing.", parent=self)

    def _delete_preset(self):
        """Delete the selected grid preset."""
        selected_indices = self.preset_listbox.curselection() # Use correct attribute
        if not selected_indices:
            messagebox.showerror("Error", "No grid preset selected.", parent=self)
            return

        selected_name = self.preset_listbox.get(selected_indices[0]) # Use correct attribute

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete preset '{selected_name}'?", icon='warning', parent=self):
            try:
                self.preset_manager.delete_preset(selected_name)
                logger.info(f"Preset '{selected_name}' deleted.")
                self._populate_preset_list() # Refresh the list
                self.gui._update_grid_preset_selector() # Update main GUI dropdown
            except Exception as e:
                 logger.error(f"Error deleting preset '{selected_name}': {e}")
                 messagebox.showerror("Error", f"Failed to delete preset: {e}", parent=self)

    def _set_as_default(self):
        """Set the selected preset as the default."""
        selected_indices = self.preset_listbox.curselection() # Use correct attribute
        if not selected_indices:
            messagebox.showerror("Error", "No grid preset selected.", parent=self)
            return

        selected_name = self.preset_listbox.get(selected_indices[0]) # Use correct attribute
        self.preset_manager.set_default_preset(selected_name)
        self._populate_preset_list() # Refresh list to show new default highlighting
        messagebox.showinfo("Success", f"Grid preset '{selected_name}' set as default.", parent=self)

    def _open_create_preset_modal(self):
        """Open the create preset modal dialog."""
        # Keep this management window open
        self.gui._open_create_grid_preset_modal(manager_window=self) # Pass self

    def on_close(self): # Renamed from _on_close for consistency
        """Handle window close."""
        self.grab_release()
        self.destroy()
    # --- END ADDED Missing Methods ---
    
class CreateGridPresetModal(tk.Toplevel):
    """Modal dialog for creating new grid presets."""

    def __init__(self, parent, gui, preset_to_edit: Optional[GridPreset] = None, manager_window: Optional['GridPresetManagementModal'] = None): # Added type hint
        """Initialize the Create/Edit Preset modal.
           Defaults init_mode to RULE_DEFAULT for new presets.
           (Round 17: Default new presets to RULE_DEFAULT)
           (Round 9 Fix: Use shape_manager.get_shape_names())"""
        super().__init__(parent)
        self.parent = parent
        self.gui = gui
        self.preset_manager = GridPresetManager.get_instance(self.gui.app_paths) # Use GUI's app_paths
        self.preset_to_edit = preset_to_edit
        self.manager_window: Optional['GridPresetManagementModal'] = manager_window # Store reference with type hint

        # Set Title Based on Mode
        if preset_to_edit:
            self.title(f"Edit Preset - {preset_to_edit.name}")
            self.original_preset_name = preset_to_edit.name # Store original name for overwrite check
        else:
            self.title("Create New Grid Preset")
            self.original_preset_name = None

        self.geometry("550x900") # Increased width
        self.transient(parent)
        self.grab_set()

        # Initialize Variables
        self.name_var = tk.StringVar(value=preset_to_edit.name if preset_to_edit else "")
        dims_str = ",".join(map(str, preset_to_edit.dimensions)) if preset_to_edit else ",".join(map(str, gui.dimensions))
        self.grid_size_var = tk.StringVar(value=dims_str)
        rule_name = preset_to_edit.rule_name if preset_to_edit else (gui.controller.rule_name if gui.controller else "")
        self.rule_var = tk.StringVar(value=rule_name)
        dim_type_name = (Dimension.TWO_D if len(preset_to_edit.dimensions) == 2 else Dimension.THREE_D).name if preset_to_edit else (gui.grid.dimension_type.name if gui.grid else "TWO_D")
        self.dimension_var = tk.StringVar(value=dim_type_name)
        neigh_name = preset_to_edit.neighborhood_type if preset_to_edit else (gui.grid.neighborhood_type.name if gui.grid else "MOORE")
        self.neighborhood_var = tk.StringVar(value=neigh_name)
        self.boundary_var = tk.StringVar(value="bounded") # Default, might be overridden by rule params later if needed
        init_dens = preset_to_edit.node_density if preset_to_edit else GlobalSettings.Simulation.INITIAL_NODE_DENSITY
        edge_dens = preset_to_edit.edge_density if preset_to_edit else GlobalSettings.Simulation.INITIAL_EDGE_DENSITY
        self.initial_density_var = tk.StringVar(value=f"{init_dens:.2f}")
        self.edge_density_var = tk.StringVar(value=f"{edge_dens:.2f}")

        # --- MODIFIED: Default init_mode based on edit/create ---
        if preset_to_edit:
            init_mode = preset_to_edit.initialization_mode
            # Handle the case where an old preset might still have "Pattern"
            if init_mode == "Pattern":
                logger.warning(f"Preset '{preset_to_edit.name}' uses deprecated 'Pattern' mode. Assuming 'SAVED_STATE'.")
                init_mode = "SAVED_STATE"
        else:
            # Default NEW presets to RULE_DEFAULT
            init_mode = "RULE_DEFAULT"
        self.init_mode_var = tk.StringVar(value=init_mode)
        # --- END MODIFIED ---

        self.description_text = preset_to_edit.description if preset_to_edit else ""

        manager = InitialConditionManager.get_instance()
        condition_names = manager.get_all_names()
        if not condition_names: condition_names = ["(None)"]
        shape_manager = ShapeLibraryManager.get_instance()
        shape_names = shape_manager.get_shape_names()
        if not shape_names: shape_names = ["(None)"]

        initial_cond_data = condition_names[0]
        initial_shape_data = shape_names[0]
        if preset_to_edit:
            if preset_to_edit.initialization_mode == "SPECIFIC_CONDITION" and preset_to_edit.initialization_data in condition_names:
                initial_cond_data = preset_to_edit.initialization_data
            elif preset_to_edit.initialization_mode == "LIBRARY_SHAPE" and preset_to_edit.initialization_data in shape_names:
                initial_shape_data = preset_to_edit.initialization_data

        self.init_condition_var = tk.StringVar(value=initial_cond_data)
        self.init_shape_var = tk.StringVar(value=initial_shape_data)

        self._create_widgets()
        self._select_rule_in_listbox(self.rule_var.get())

        if hasattr(self, 'description_entry') and self.description_text:
             self.description_entry.insert("1.0", self.description_text)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._update_dimension_display()
        self._update_init_data_widgets() # Ensure correct widgets shown initially

    def _create_widgets(self):
        """Create the widgets for the modal dialog, including initialization mode selection
           and the option to keep the original saved state when editing.
           (Round 4 Fix: Correct packing order and ensure rule list population)"""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True) # Main frame expands

        # --- Buttons (Pack at the very bottom FIRST) ---
        button_frame = ttk.Frame(main_frame)
        # Pack button_frame at the BOTTOM of main_frame
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(10, 0))

        save_button_text = "Update Preset" if self.preset_to_edit else "Save Preset"
        ttk.Button(button_frame, text=save_button_text, command=self._start_save_preset).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save and Set Default", command=self._save_and_set_default).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_close).pack(side=tk.RIGHT, padx=5)
        # --- Buttons packed at bottom ---

        # --- Top Frame for Basic Info ---
        top_info_frame = ttk.Frame(main_frame)
        top_info_frame.pack(fill=tk.X, pady=(0, 10), side=tk.TOP) # Pack at top

        # --- Basic Info (Name, Size, Dimension, Neighborhood) ---
        ttk.Label(top_info_frame, text="Preset Name:").pack(fill=tk.X, padx=5, pady=2)
        self.name_entry = ttk.Entry(top_info_frame, textvariable=self.name_var)
        self.name_entry.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(top_info_frame, text="Grid Size (e.g., 30,30 or 10,10,10):").pack(fill=tk.X, padx=5, pady=2)
        self.grid_size_entry = ttk.Entry(top_info_frame, textvariable=self.grid_size_var)
        self.grid_size_entry.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(top_info_frame, text="Dimension:").pack(fill=tk.X, padx=5, pady=2)
        self.dimension_menu = ttk.Combobox(top_info_frame, textvariable=self.dimension_var, values=["TWO_D", "THREE_D"], state="readonly")
        self.dimension_menu.pack(fill=tk.X, padx=5, pady=2)
        self.grid_size_var.trace_add("write", self._update_dimension_display)

        ttk.Label(top_info_frame, text="Neighborhood Type:").pack(fill=tk.X, padx=5, pady=2)
        neighborhood_choices = sorted([n.name for n in NeighborhoodType])
        if self.neighborhood_var.get() not in neighborhood_choices and neighborhood_choices: self.neighborhood_var.set(neighborhood_choices[0])
        self.neighborhood_menu = ttk.OptionMenu(top_info_frame, self.neighborhood_var, self.neighborhood_var.get(), *neighborhood_choices)
        self.neighborhood_menu.pack(fill=tk.X, padx=5, pady=2)

        # --- Rule Selection (Ensure it expands vertically) ---
        rule_section_frame = ttk.LabelFrame(main_frame, text="Rule", padding=5)
        # Pack rule section AFTER top_info_frame
        rule_section_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5, side=tk.TOP) # Fill and Expand

        rule_search_frame = ttk.Frame(rule_section_frame)
        rule_search_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(0, 2))
        self.rule_search_var = tk.StringVar()
        search_entry = ttk.Entry(rule_search_frame, textvariable=self.rule_search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        search_entry.bind("<KeyRelease>", self._filter_rule_list)

        rule_list_frame = ttk.Frame(rule_section_frame)
        rule_list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True) # Fill and Expand
        rule_scrollbar = tk.Scrollbar(rule_list_frame, orient=tk.VERTICAL)
        rule_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.rule_listbox = tk.Listbox(rule_list_frame, yscrollcommand=rule_scrollbar.set, exportselection=False, height=6) # Keep height reasonable
        rule_scrollbar.config(command=self.rule_listbox.yview)
        self.rule_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) # Fill and Expand
        # --- MODIFIED: Populate listbox AFTER packing ---
        self._populate_rule_listbox()
        # ---

        # --- Initialization Mode ---
        init_frame = ttk.LabelFrame(main_frame, text="Initialization Mode", padding=5)
        # Pack init_frame AFTER rule section
        init_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.TOP) # Pack below rule section

        initial_mode = "SAVED_STATE"
        if self.preset_to_edit: initial_mode = self.preset_to_edit.initialization_mode
        self.init_mode_var = tk.StringVar(value=initial_mode)

        ttk.Radiobutton(init_frame, text="Save Current Grid State/Edges", variable=self.init_mode_var, value="SAVED_STATE", command=self._update_init_data_widgets).pack(anchor=tk.W)
        self.keep_original_rb = None
        if self.preset_to_edit and self.preset_to_edit.initialization_mode == "SAVED_STATE":
            self.keep_original_rb = ttk.Radiobutton(init_frame, text="Keep Original Saved State/Edges", variable=self.init_mode_var, value="KEEP_ORIGINAL_SAVED_STATE", command=self._update_init_data_widgets)
            self.keep_original_rb.pack(anchor=tk.W)
            self.init_mode_var.set("KEEP_ORIGINAL_SAVED_STATE")

        ttk.Radiobutton(init_frame, text="Use Rule's Default Initial Condition", variable=self.init_mode_var, value="RULE_DEFAULT", command=self._update_init_data_widgets).pack(anchor=tk.W)
        ttk.Radiobutton(init_frame, text="Apply Specific Initial Condition:", variable=self.init_mode_var, value="SPECIFIC_CONDITION", command=self._update_init_data_widgets).pack(anchor=tk.W)
        ttk.Radiobutton(init_frame, text="Place Library Shape:", variable=self.init_mode_var, value="LIBRARY_SHAPE", command=self._update_init_data_widgets).pack(anchor=tk.W)

        # --- Initialization Data (Dropdowns) ---
        self.init_data_frame = ttk.Frame(main_frame, padding=(20, 0, 0, 0))
        # Pack init_data_frame AFTER init_frame
        self.init_data_frame.pack(fill=tk.X, padx=5, pady=0, side=tk.TOP) # Pack below init mode

        self.init_condition_label = ttk.Label(self.init_data_frame, text="Condition:")
        manager = InitialConditionManager.get_instance(); condition_names = manager.get_all_names()
        if not condition_names: condition_names = ["(None)"]
        initial_cond_data = condition_names[0]
        if self.preset_to_edit and self.preset_to_edit.initialization_mode == "SPECIFIC_CONDITION" and self.preset_to_edit.initialization_data in condition_names: initial_cond_data = self.preset_to_edit.initialization_data
        self.init_condition_var = tk.StringVar(value=initial_cond_data)
        self.init_condition_menu = ttk.Combobox(self.init_data_frame, textvariable=self.init_condition_var, values=condition_names, state="readonly", width=30)

        self.init_shape_label = ttk.Label(self.init_data_frame, text="Shape:")
        shape_manager = ShapeLibraryManager.get_instance(); shape_names = shape_manager.get_shape_names()
        if not shape_names: shape_names = ["(None)"]
        initial_shape_data = shape_names[0]
        if self.preset_to_edit and self.preset_to_edit.initialization_mode == "LIBRARY_SHAPE" and self.preset_to_edit.initialization_data in shape_names: initial_shape_data = self.preset_to_edit.initialization_data
        self.init_shape_var = tk.StringVar(value=initial_shape_data)
        self.init_shape_menu = ttk.Combobox(self.init_data_frame, textvariable=self.init_shape_var, values=shape_names, state="readonly", width=30)

        self._update_init_data_widgets() # Set initial visibility

        # --- Density (Informational/Fallback) ---
        density_frame = ttk.LabelFrame(main_frame, text="Density (Informational/Fallback)", padding=5)
        # Pack density_frame AFTER init_data_frame
        density_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.TOP) # Pack below init data
        ttk.Label(density_frame, text="Node Density (0.0-1.0):").pack(anchor=tk.W)
        self.initial_density_entry = ttk.Entry(density_frame, textvariable=self.initial_density_var)
        self.initial_density_entry.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(density_frame, text="Edge Density (0.0-1.0):").pack(anchor=tk.W)
        self.edge_density_entry = ttk.Entry(density_frame, textvariable=self.edge_density_var)
        self.edge_density_entry.pack(fill=tk.X, padx=5, pady=2)

        # --- Description ---
        ttk.Label(main_frame, text="Description:").pack(fill=tk.X, padx=5, pady=2, side=tk.TOP) # Pack below density
        desc_frame = ttk.Frame(main_frame)
        # Pack desc_frame AFTER density_frame
        desc_frame.pack(fill=tk.X, padx=5, pady=2, side=tk.TOP) # Pack below label
        desc_scrollbar = tk.Scrollbar(desc_frame, orient=tk.VERTICAL)
        self.description_entry = tk.Text(desc_frame, height=3, wrap=tk.WORD, yscrollcommand=desc_scrollbar.set)
        desc_scrollbar.config(command=self.description_entry.yview)
        desc_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.description_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Progress Bar ---
        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=300, mode="determinate")
        # Pack progress_bar AFTER description
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5, side=tk.TOP) # Pack below description

    def _update_init_data_widgets(self):
        """Show/hide the appropriate dropdown based on initialization_mode."""
        mode = self.init_mode_var.get()

        # Hide all data widgets first
        self.init_condition_label.pack_forget()
        self.init_condition_menu.pack_forget()
        self.init_shape_label.pack_forget()
        self.init_shape_menu.pack_forget()

        # Show the relevant widget
        if mode == "SPECIFIC_CONDITION":
            self.init_condition_label.pack(side=tk.LEFT, padx=(0, 5))
            self.init_condition_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        elif mode == "LIBRARY_SHAPE":
            self.init_shape_label.pack(side=tk.LEFT, padx=(0, 5))
            self.init_shape_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _filter_rule_list(self, event=None):
        """Filter the rule listbox based on the search entry."""
        filter_term = self.rule_search_var.get()
        self._populate_rule_listbox(filter_term)

    def _select_rule_in_listbox(self, rule_name_to_select: str):
        """Selects the specified rule name in the rule listbox."""
        if not hasattr(self, 'rule_listbox') or not self.rule_listbox:
            logger.warning("Rule listbox not available for selection.")
            return

        list_items = self.rule_listbox.get(0, tk.END)
        if rule_name_to_select in list_items:
            try:
                index_to_select = list_items.index(rule_name_to_select)
                self.rule_listbox.selection_clear(0, tk.END)
                self.rule_listbox.selection_set(index_to_select)
                self.rule_listbox.see(index_to_select)
                self.rule_listbox.activate(index_to_select)
                logger.debug(f"Selected rule '{rule_name_to_select}' in listbox at index {index_to_select}.")
            except ValueError:
                logger.warning(f"Rule '{rule_name_to_select}' was in list items but index lookup failed.")
            except tk.TclError:
                 logger.warning(f"TclError selecting rule '{rule_name_to_select}' in listbox.")
        else:
            logger.warning(f"Rule '{rule_name_to_select}' not found in the visible listbox items. Cannot select.")
            # Optionally select the first item if nothing is selected
            if not self.rule_listbox.curselection():
                self.rule_listbox.selection_set(0)

    def _populate_rule_listbox(self, filter_term: str = ""):
        """Populate the rule listbox, applying an optional filter."""
        # --- ADDED: Import RuleLibrary ---
        from .rules import RuleLibrary
        # ---
        if not hasattr(self, 'rule_listbox') or not self.rule_listbox:
            logger.warning("_populate_rule_listbox: Listbox not initialized.")
            return

        self.rule_listbox.delete(0, tk.END)
        # --- MODIFIED: Get names from RuleLibrary ---
        rule_names = sorted(RuleLibrary.get_rule_names())
        # ---
        filter_term_lower = filter_term.lower()

        initial_rule_name = self.rule_var.get() # Get the currently set rule name
        selected_index = -1
        current_index = 0

        for name in rule_names:
            if filter_term_lower in name.lower():
                self.rule_listbox.insert(tk.END, name)
                if name == initial_rule_name:
                    selected_index = current_index # Track index if it matches initial rule
                current_index += 1

        # Reselect the previously selected/initial rule if it's still visible
        if selected_index != -1:
            try:
                self.rule_listbox.selection_clear(0, tk.END)
                self.rule_listbox.selection_set(selected_index)
                self.rule_listbox.see(selected_index)
                self.rule_listbox.activate(selected_index)
            except tk.TclError:
                logger.warning("TclError setting initial rule selection after filter.")
        elif self.rule_listbox.size() > 0: # Select first item if previous selection gone
            self.rule_listbox.selection_set(0)
            # --- ADDED: Update rule_var if selection changed ---
            first_item = self.rule_listbox.get(0)
            if first_item != initial_rule_name:
                self.rule_var.set(first_item)
            # ---

    def _update_dimension_display(self, *args):
        """Update the Dimension dropdown based on the grid size entry."""
        dimensions_str = self.grid_size_var.get()
        try:
            dimensions = tuple(int(d) for d in dimensions_str.split(','))
            if len(dimensions) == 2:
                self.dimension_var.set("TWO_D")
            elif len(dimensions) == 3:
                self.dimension_var.set("THREE_D")
            else:
                # Handle invalid number of dimensions if needed, maybe clear selection
                self.dimension_var.set("") # Or set to a default/error state
        except ValueError:
            # Handle invalid format (non-integer, wrong separator)
             self.dimension_var.set("") # Clear if format is wrong

    def _save_preset(self) -> Dict[str, Any]: # Removed queue argument, added return type
        """
        Save the new or updated grid preset with initialization mode.
        Performs validation and saving logic directly.
        Returns a dictionary indicating success, final name, and error.
        Sets mode to SAVED_STATE if saving current grid state.
        (Round 17: Set mode to SAVED_STATE when saving current grid)
        (Round 5 Refactor: Made synchronous, returns result dict)
        """
        save_success = False
        saved_preset_name = "Unknown"
        error_message = None
        save_mode_used = "overwrite" if self.preset_to_edit else "new" # Track save mode
        final_preset_data = None # Store the data that was actually saved

        try:
            # --- Get common values ---
            name = self.name_var.get().strip()
            dimensions_str = self.grid_size_var.get()
            neighborhood_type = self.neighborhood_var.get()
            initial_density_str = self.initial_density_var.get()
            edge_density_str = self.edge_density_var.get()
            selected_indices = self.rule_listbox.curselection()
            if not selected_indices: raise ValueError("Please select a rule.")
            rule_name = self.rule_listbox.get(selected_indices[0])
            description = self.description_entry.get("1.0", tk.END).strip()
            initialization_mode_selected = self.init_mode_var.get() # Get the selected mode
            initialization_data = None

            # --- Get initialization data based on mode ---
            if initialization_mode_selected == "SPECIFIC_CONDITION":
                initialization_data = self.init_condition_var.get()
                if not initialization_data or initialization_data == "(None)":
                    raise ValueError("Please select a specific initial condition.")
            elif initialization_mode_selected == "LIBRARY_SHAPE":
                initialization_data = self.init_shape_var.get()
                if not initialization_data or initialization_data == "(None)":
                    raise ValueError("Please select a library shape.")

            # --- Overwrite/New Check ---
            is_overwriting = False
            if self.preset_to_edit and name == self.original_preset_name:
                is_overwriting = True
                save_mode_used = "overwrite" # Confirm overwrite mode

            # --- Validation ---
            if not name: raise ValueError("Preset name cannot be empty.")
            try:
                dimensions = tuple(int(d.strip()) for d in dimensions_str.split(','))
                if len(dimensions) not in (2, 3): raise ValueError("Dimensions must have 2 or 3 integers.")
                if any(d <= 0 for d in dimensions): raise ValueError("Dimensions must be positive.")
            except ValueError as e: raise ValueError(f"Invalid dimensions format: {e}. Use comma-separated positive integers (e.g., 30,30 or 10,10,10).")
            try: initial_density = float(initial_density_str); assert 0.0 <= initial_density <= 1.0
            except (ValueError, AssertionError): raise ValueError("Invalid initial node density (0.0-1.0).")
            try: edge_density = float(edge_density_str); assert 0.0 <= edge_density <= 1.0
            except (ValueError, AssertionError): raise ValueError("Invalid initial edge density (0.0-1.0).")

            # --- Prepare Preset Data based on selected mode ---
            initial_state_to_save = None
            edges_to_save = None
            edge_states_to_save = None
            final_init_mode_for_preset = initialization_mode_selected # Start with selected mode

            # --- MODIFIED: Handle SAVED_STATE and KEEP_ORIGINAL ---
            if initialization_mode_selected == "SAVED_STATE":
                logger.info(f"Saving current grid state for preset '{name}'...")
                if self.gui.grid is None: raise ValueError("Current grid state is not available to save.")
                # Check if dimensions match the ones entered in the dialog
                if tuple(self.gui.grid.dimensions) != dimensions:
                    logger.warning(f"Saving preset '{name}' with dimensions {dimensions} but current grid state is {self.gui.grid.dimensions}. Saved state will be empty.")
                    # Set state/edges to None if dimensions don't match
                    initial_state_to_save = None
                    edges_to_save = None
                    edge_states_to_save = None
                else:
                    initial_state_to_save = self.gui.grid.grid_array.copy()
                    edges_to_save = list(self.gui.grid.edges)
                    edge_states_to_save = self.gui.grid.edge_states.copy()
                    logger.debug(f"Captured current grid state: {initial_state_to_save.shape}, {len(edges_to_save)} edges.")
                final_init_mode_for_preset = "SAVED_STATE" # Ensure mode is correct

            elif initialization_mode_selected == "KEEP_ORIGINAL_SAVED_STATE":
                if self.preset_to_edit:
                    logger.info(f"Keeping original saved state/edges for preset '{name}'.")
                    # Check if original dimensions match the potentially edited dimensions
                    if tuple(self.preset_to_edit.dimensions) != dimensions:
                         logger.warning(f"Keeping original state/edges but applying NEW dimensions {dimensions}. Original state might be incompatible.")
                    # Use original data regardless of dimension match (user was warned)
                    initial_state_to_save = self.preset_to_edit.initial_state.copy() if self.preset_to_edit.initial_state is not None else None
                    edges_to_save = list(self.preset_to_edit.edges) if self.preset_to_edit.edges is not None else None
                    edge_states_to_save = self.preset_to_edit.edge_states.copy() if self.preset_to_edit.edge_states is not None else None
                    final_init_mode_for_preset = "SAVED_STATE" # Save mode as SAVED_STATE
                else:
                    raise ValueError("Cannot keep original state when creating a new preset.")
            # --- END MODIFIED ---

            # --- Delete original if name changed ---
            if self.preset_to_edit and name != self.original_preset_name:
                 try:
                      if self.original_preset_name is not None:
                          self.preset_manager.delete_preset(self.original_preset_name)
                      logger.info(f"Deleted original preset '{self.original_preset_name}' after name change.")
                      save_mode_used = "rename" # Indicate rename happened
                 except Exception as del_err:
                      logger.error(f"Error deleting original preset '{self.original_preset_name}': {del_err}")

            # --- Create and Save Preset ---
            preset = GridPreset(
                name=name, dimensions=dimensions, neighborhood_type=neighborhood_type,
                rule_name=rule_name,
                initialization_mode=final_init_mode_for_preset, # Use the determined mode
                initialization_data=initialization_data,
                initial_state=initial_state_to_save,
                edges=edges_to_save,
                edge_states=edge_states_to_save,
                description=description,
                node_density=initial_density, edge_density=edge_density
            )

            self.preset_manager.save_preset(preset)
            save_success = True
            saved_preset_name = name # Store the name used for saving
            final_preset_data = preset.to_dict() # Get the dict representation of the saved preset

        except ValueError as e:
            logger.error(f"Validation error creating/updating grid preset: {e}")
            error_message = str(e)
        except Exception as e:
            logger.error(f"Error creating/updating grid preset: {e}")
            logger.error(traceback.format_exc())
            error_message = str(e)

        # Return result dictionary
        return {
            'success': save_success,
            'preset_name': saved_preset_name,
            'preset_data': final_preset_data, # Include the saved data
            'error': error_message,
            'save_mode': save_mode_used
        }

    def _start_save_preset(self):
        """Start the standard save preset process in a separate thread."""
        log_prefix = "CreateGridPresetModal._start_save_preset: "
        logger.debug(f"{log_prefix}Starting standard save thread.")
        save_result_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=1)

        def save_thread_target():
            result = self._save_preset() # Call the synchronous save method
            save_result_queue.put(result) # Put the result dict onto the queue

        threading.Thread(target=save_thread_target, daemon=True).start()
        # Schedule check for the standard save result
        self.after(100, self._check_standard_save_result, save_result_queue)

    def _save_and_set_default(self):
        """Saves the preset and then sets it as the default, using a thread."""
        log_prefix = "CreateGridPresetModal._save_and_set_default: "
        logger.info(f"{log_prefix}Starting 'Save and Set Default' thread.")
        save_result_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=1)

        def save_and_set_default_thread_target():
            result = self._save_preset() # Call the synchronous save method
            save_result_queue.put(result) # Put the result dict onto the queue

        threading.Thread(target=save_and_set_default_thread_target, daemon=True).start()
        # Schedule check for the save AND set default result
        self.after(100, self._check_save_and_set_default_result, save_result_queue)

    def _check_standard_save_result(self, result_queue: queue.Queue):
        """Checks the result queue from the standard save thread and updates UI."""
        log_prefix = "CreateGridPresetModal._check_standard_save_result: "
        try:
            result = result_queue.get_nowait()
            logger.debug(f"{log_prefix}Received result from standard save thread: {result}")
            if result['success']:
                preset_name = result['preset_name']
                # Refresh manager list if open
                if self.manager_window and self.manager_window.winfo_exists():
                    manager_modal = cast('GridPresetManagementModal', self.manager_window)
                    if hasattr(manager_modal, '_populate_preset_list'):
                        manager_modal._populate_preset_list()
                    manager_modal.lift()
                # Update main GUI selector
                self.gui._update_grid_preset_selector()
                # --- MODIFIED: Set active preset in GUI ---
                self.gui._set_active_preset(preset_name)
                # ---
                messagebox.showinfo("Save Successful", f"Preset '{preset_name}' saved.", parent=self.parent) # Show on parent
                self.destroy() # Close on successful standard save
            else:
                error_msg = result.get('error', 'Unknown save error')
                logger.error(f"{log_prefix}Save failed: {error_msg}")
                messagebox.showerror("Save Failed", f"Could not save preset:\n{error_msg}", parent=self)
                # Keep dialog open on failure
        except queue.Empty:
            if self.winfo_exists():
                self.after(100, self._check_standard_save_result, result_queue)
        except Exception as e:
            logger.error(f"{log_prefix}Error checking standard save result: {e}")
            messagebox.showerror("Error", f"Error checking save result: {e}", parent=self)
            self.destroy()

    def _check_save_and_set_default_result(self, result_queue: queue.Queue):
        """Checks the result from the save thread and sets default if successful."""
        log_prefix = "CreateGridPresetModal._check_save_and_set_default_result: "
        try:
            result = result_queue.get_nowait()
            logger.debug(f"{log_prefix}Received result from save thread: {result}")

            if result['success']:
                preset_name = result['preset_name']
                logger.info(f"{log_prefix}Save successful for '{preset_name}'. Setting as default.")
                try:
                    self.preset_manager.set_default_preset(preset_name)
                    # Update the management modal list if it's open
                    if self.manager_window and self.manager_window.winfo_exists():
                        manager_modal = cast('GridPresetManagementModal', self.manager_window)
                        if hasattr(manager_modal, '_populate_preset_list'):
                            manager_modal._populate_preset_list()
                        manager_modal.lift()
                    # Update main GUI selector
                    self.gui._update_grid_preset_selector()
                    # --- MODIFIED: Set active preset in GUI ---
                    self.gui._set_active_preset(preset_name)
                    # ---
                    logger.info(f"{log_prefix}Preset '{preset_name}' set as default.")
                    messagebox.showinfo("Success", f"Preset '{preset_name}' saved and set as default.", parent=self.parent) # Show on parent
                    # Close this modal AFTER setting default
                    self.destroy()
                except Exception as e_set:
                    logger.error(f"{log_prefix}Error setting preset '{preset_name}' as default: {e_set}")
                    messagebox.showerror("Error", f"Preset saved, but failed to set as default: {e_set}", parent=self.parent) # Show error on parent
                    self.destroy() # Still close the editor
            else:
                error_msg = result.get('error', 'Unknown save error')
                logger.error(f"{log_prefix}Save failed: {error_msg}")
                messagebox.showerror("Save Failed", f"Could not save preset:\n{error_msg}", parent=self)
                # Don't close the dialog on save failure

        except queue.Empty:
            # Result not ready yet, check again
            if self.winfo_exists(): # Check if window still exists
                self.after(100, self._check_save_and_set_default_result, result_queue)
        except Exception as e:
            logger.error(f"{log_prefix}Error checking save result: {e}")
            messagebox.showerror("Error", f"Error checking save result: {e}", parent=self)
            self.destroy() # Close on unexpected error

    def on_close(self):
        """Handle window close."""
        self.grab_release()
        self.destroy()

class ResizeProgressDialog(tk.Toplevel):
    """Custom dialog for handling grid resize with progress and options."""

    def __init__(self, parent, gui: 'SimulationGUI', new_dimensions: Tuple[int, ...], old_grid_array: Optional[np.ndarray]):
        super().__init__(parent)
        self.parent_gui = gui
        self.new_dimensions = new_dimensions
        self.old_grid_array = old_grid_array # Can be None if grid was empty or clearing
        self.title("Grid Resize Options")
        self.geometry("450x350") # Adjusted size
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        # --- Initialized Attributes ---
        # --- Default to rule_default ---
        self.initialization_choice = tk.StringVar(value="rule_default") # Default choice
        # ---
        self.progress_bar: Optional[ttk.Progressbar] = None
        self.progress_label: Optional[tk.Label] = None
        self.result: Optional[Dict[str, Any]] = None # Store result for caller
        # ---

        self._create_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel) # Handle closing via 'X'

        # --- REMOVED Logic that overrides default based on old_grid_array ---
        # if self.old_grid_array is not None and self.old_grid_array.size > 0:
        #     self.initialization_choice.set("copy")
        # else:
        #     self.initialization_choice.set("rule_default")
        # ---

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Options Frame ---
        options_frame = ttk.LabelFrame(main_frame, text="Initialization After Resize", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 10))

        # Use self.initialization_choice here
        ttk.Radiobutton(options_frame, text="Initialize Empty Grid",
                        variable=self.initialization_choice, value="empty").pack(anchor=tk.W)
        ttk.Radiobutton(options_frame, text="Initialize Random (using current densities)",
                        variable=self.initialization_choice, value="random").pack(anchor=tk.W)
        ttk.Radiobutton(options_frame, text="Initialize using Rule's Default Condition",
                        variable=self.initialization_choice, value="rule_default").pack(anchor=tk.W)

        copy_state = tk.NORMAL if self.old_grid_array is not None and self.old_grid_array.size > 0 else tk.DISABLED
        self.copy_radio = ttk.Radiobutton(options_frame, text="Attempt to Copy Content (Top-Left)",
                        variable=self.initialization_choice, value="copy", state=copy_state)
        self.copy_radio.pack(anchor=tk.W)

        # --- Button Frame ---
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0)) # Adjusted padding

        self.start_button = ttk.Button(button_frame, text="Start Resize", command=self._start_resize_sync)
        self.start_button.pack(side=tk.RIGHT, padx=5)
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self._on_cancel)
        self.cancel_button.pack(side=tk.RIGHT)

    def _start_resize_sync(self):
        """Handles the 'Start Resize' button click for synchronous operation."""
        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.DISABLED) # Disable cancel during sync operation
        self.update_idletasks() # Ensure UI updates

        init_choice = self.initialization_choice.get()
        old_array_to_pass = self.old_grid_array if init_choice == "copy" else None

        # Set result for the caller (_apply_new_grid_size)
        self.result = {
            "action": "resize",
            "dimensions": self.new_dimensions,
            "init_choice": init_choice,
            "old_grid_array": old_array_to_pass # Pass old array if needed
        }
        # --- ADDED LOGGING ---
        logger.debug(f"ResizeProgressDialog: Setting self.result = {self.result} before destroying.")
        # ---
        # Close the dialog immediately, the caller will perform the resize
        self._safe_destroy()

    def close_dialog(self, success: bool = True):
        """Closes the dialog (called by main thread)."""
        if not self.winfo_exists(): return
        if success: logger.info("Resize operation reported success.")
        else: logger.warning("Resize operation reported cancellation or failure.")
        self._safe_destroy()

    def _on_cancel(self):
        """Handles Cancel button click or window close."""
        logger.info("Resize operation cancelled by user.")
        self.result = {"action": "cancel"} # Set result to cancel
        self._safe_destroy()

    def _safe_destroy(self):
        """Safely destroy the window."""
        try:
            if self.winfo_exists():
                self.grab_release()
                self.destroy()
                logger.debug("ResizeProgressDialog destroyed safely.")
        except Exception as e:
            logger.warning(f"Error during _safe_destroy: {e}")

class PatternFitResizeDialog(tk.Toplevel):
    """Dialog prompting user to resize further if pattern doesn't fit."""
    def __init__(self, parent, pattern_dims: Tuple[int, ...], current_target_dims: Tuple[int, ...]):
        super().__init__(parent)
        self.title("Pattern Too Large")
        self.transient(parent)
        self.grab_set()
        self.result: Optional[Dict[str, Any]] = None # {'action': 'resize_fit'/'cancel', 'dimensions': tuple}

        self.pattern_dims = pattern_dims
        # Calculate minimum required size with 20% padding
        padding_factor = 1.20
        min_padding = 4 # Minimum 2 cells padding on each side
        self.required_dims = tuple(max(pd, int(pd * padding_factor), min_padding) for pd in pattern_dims)

        # --- Widgets ---
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Message
        pattern_str = "x".join(map(str, pattern_dims))
        target_str = "x".join(map(str, current_target_dims))
        required_str = "x".join(map(str, self.required_dims))

        message = f"The active pattern ({pattern_str}) from the current grid\n"
        message += f"does not fit within the selected target size ({target_str}).\n\n"
        message += f"The minimum size required to fit the pattern (with padding) is {required_str}.\n\n"
        message += "Choose an action:"
        ttk.Label(main_frame, text=message, justify=tk.LEFT).pack(pady=(0, 15))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        ttk.Button(button_frame, text=f"Resize to Fit ({required_str})", command=self._on_resize_fit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel Resize", command=self._on_cancel).pack(side=tk.RIGHT, padx=5)

        self.wait_window(self) # Make it modal

    def _on_resize_fit(self):
        """Set result to resize to the calculated required dimensions."""
        self.result = {"action": "resize_fit", "dimensions": self.required_dims}
        self.destroy()

    def _on_cancel(self):
        """Set result to cancel."""
        self.result = {"action": "cancel"}
        self.destroy()
    
# =========== END of presets.py ===========
