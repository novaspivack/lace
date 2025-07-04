# =========== START of initial_conditions.py ===========
from __future__ import annotations
import random
import threading
import setproctitle
import matplotlib
import matplotlib.pyplot as plt
from sympy import Line
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from typing import Dict, List, Set, Tuple, Optional, Union, Callable, TypeVar, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import warnings
import cProfile
import pstats


from .logging_config import logger
from .enums import Dimension, ShapeType
from .settings import GlobalSettings
from .utils import (
    _ravel_multi_index, _unravel_index
    )   
from .shapes import (
    ShapeDefinition, ShapeGenerator, Square, Circle, Sphere, Line
    )


# --- ADDED: TYPE_CHECKING block ---
if TYPE_CHECKING:
    from lace_app import Grid


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
#              INITIAL CONDITIONS            #
################################################

class InitialConditionManager:
    """Manages and applies different initial grid conditions."""
    _instance: Optional['InitialConditionManager'] = None
    _initial_conditions: Dict[str, Callable[['InitialConditionManager', 'Grid'], None]] = {}

    @classmethod
    def get_instance(cls) -> 'InitialConditionManager':
        """Get the singleton instance of the InitialConditionManager."""
        if cls._instance is None:
            # --- Ensure registration happens only once ---
            if not cls._initial_conditions:
                 cls._instance = InitialConditionManager() # Calls __init__ which calls _register_defaults
            else:
                 # If called again after registration, just create instance
                 cls._instance = super().__new__(cls)
                 cls._instance._initial_conditions = cls._initial_conditions # Ensure instance has the dict
        return cls._instance

    def __init__(self):
        # --- Prevent re-initialization ---
        if InitialConditionManager._instance is not None:
            # logger.warning("InitialConditionManager already initialized.") # Reduce noise
            return
        # ---
        self._register_defaults()
        InitialConditionManager._instance = self

    def register(self, name: str, func: Callable[['InitialConditionManager', 'Grid'], None]):
        """Register an initial condition function."""
        self._initial_conditions[name] = func
        logger.debug(f"Registered initial condition: {name}")

    def get(self, name: str) -> Optional[Callable[['InitialConditionManager', 'Grid'], None]]:
        """Get an initial condition function by name."""
        return self._initial_conditions.get(name)

    def get_all_names(self) -> List[str]:
        """Get a list of all registered initial condition names."""
        # Ensure Random is first if it exists
        names = list(self._initial_conditions.keys())
        if "Random" in names:
            names.remove("Random")
            names.insert(0, "Random")
        return names

    def apply(self, name: str, grid: Grid,
              progress_callback: Optional[Callable[[int, str], None]] = None,
              progress_offset: int = 0, progress_total: int = 1,
              cancel_event: Optional[threading.Event] = None):
        """
        Apply the specified initial condition to the grid.
        If the grid's rule requires degree-as-state, performs an additional
        step after the base initialization to set node states to their degree.
        """
        log_prefix = f"InitialConditionManager.apply(Name='{name}'): "

        # --- Handle Preset/Shape Placeholders (No change needed here) ---
        if name == "Pattern" or name.startswith("Shape: "):
            logger.info(f"{log_prefix}Condition name '{name}' indicates state already set by preset/shape. Skipping initialization function, but ensuring grid state.")
            grid.update_active_nodes()
            grid.previous_active_nodes_set = grid.active_nodes.copy()
            if grid.rule and (grid.rule.needs_neighbor_degrees or grid.rule.needs_neighbor_active_counts):
                 # Recalculate previous degree/active counts based on loaded state
                 grid.previous_degree_array = np.zeros(grid.total_nodes, dtype=np.int32)
                 grid.previous_active_neighbor_array = np.zeros(grid.total_nodes, dtype=np.int32)
                 for edge_coords in grid.edges:
                     try:
                         idx1 = _ravel_multi_index(np.array(edge_coords[0]), grid.dimensions)
                         idx2 = _ravel_multi_index(np.array(edge_coords[1]), grid.dimensions)
                         if 0 <= idx1 < grid.total_nodes: grid.previous_degree_array[idx1] += 1
                         if 0 <= idx2 < grid.total_nodes: grid.previous_degree_array[idx2] += 1
                     except Exception as degree_err: logger.warning(f"{log_prefix}Error calculating initial degree for edge {edge_coords}: {degree_err}")
                 activity_threshold = 1e-6
                 for node_idx in range(grid.total_nodes):
                     count = 0; neighbors = grid.get_neighbors(node_idx, grid.coord_system)
                     for neighbor_idx in neighbors:
                          if neighbor_idx != -1 and 0 <= neighbor_idx < grid.grid_array.size and grid.grid_array.ravel()[neighbor_idx] > activity_threshold:
                              count += 1
                     grid.previous_active_neighbor_array[node_idx] = count
            logger.debug(f"{log_prefix}Updated grid state/history based on existing preset/shape state.")
            return
        # ---

        # --- Get the standard initialization function ---
        func = self.get(name)
        if not func:
            logger.warning(f"{log_prefix}Initial condition '{name}' not found. Applying Random fallback.")
            name = "Random" # Set name for logging/progress
            func = self.get(name)
            if not func: # Should not happen if Random is registered
                 logger.error(f"{log_prefix}Fallback 'Random' condition not found! Cannot initialize.")
                 return

        # --- Apply the standard initialization function ---
        logger.info(f"{log_prefix}Applying BASE initial condition function: '{name}'...")
        if grid.shape_placer is None:
             # Assuming ShapePlacer is defined/imported correctly
             from .shapes import ShapePlacer
             grid.shape_placer = ShapePlacer(grid)

        if cancel_event and cancel_event.is_set(): raise InterruptedError("Initial condition application cancelled")
        if progress_callback: progress_callback(progress_offset, f"Applying '{name}'...")

        func(self, grid) # Call the specific init method (e.g., initialize_random)

        try:
            active_count = np.sum(grid.grid_array > 1e-6) if grid.grid_array is not None else -1
            edge_count = len(grid.edges) if grid.edges is not None else -1
            logger.info(f"{log_prefix}Grid state AFTER applying BASE '{name}' function: ActiveNodes={active_count}, Edges={edge_count}")
        except Exception as log_err: logger.warning(f"{log_prefix}Error logging grid state after base apply: {log_err}")

        # --- Check if the rule requires degree-as-state ---
        # We'll add a flag 'sets_state_to_degree' to the Rule class later
        rule_sets_degree_as_state = False
        if grid.rule and hasattr(grid.rule, 'sets_state_to_degree'):
            rule_sets_degree_as_state = grid.rule.sets_states_to_degree
            logger.debug(f"{log_prefix}Rule '{grid.rule.name}' sets_state_to_degree flag is {rule_sets_degree_as_state}.")
        elif grid.rule:
             logger.debug(f"{log_prefix}Rule '{grid.rule.name}' does not have 'sets_state_to_degree' flag (assuming False).")
        else:
             logger.debug(f"{log_prefix}No rule set on grid, cannot check for degree-as-state.")

        # --- Perform Degree Calculation Step if Required ---
        if rule_sets_degree_as_state:
            logger.info(f"{log_prefix}Rule requires degree-as-state. Calculating degrees and updating grid_array...")
            if progress_callback: progress_callback(progress_offset + 1, "Calculating Degrees...")

            if cancel_event and cancel_event.is_set(): raise InterruptedError("Initial condition application cancelled")

            initial_degrees = np.zeros(grid.total_nodes, dtype=np.int32)
            for edge_coords in grid.edges: # Use edges created by the base initializer
                try:
                    idx1 = _ravel_multi_index(np.array(edge_coords[0]), grid.dimensions)
                    idx2 = _ravel_multi_index(np.array(edge_coords[1]), grid.dimensions)
                    if 0 <= idx1 < grid.total_nodes: initial_degrees[idx1] += 1
                    if 0 <= idx2 < grid.total_nodes: initial_degrees[idx2] += 1
                except Exception as degree_err:
                    logger.warning(f"{log_prefix}Error calculating initial degree for edge {edge_coords}: {degree_err}")

            # Set grid_array state to the calculated degree
            np.copyto(grid.grid_array.ravel(), initial_degrees.astype(np.float64))
            logger.info(f"{log_prefix}Set node states to calculated degrees.")

            # Update previous degree array for the *next* step's calculations
            grid.previous_degree_array = initial_degrees.copy()
            logger.debug(f"{log_prefix}Updated previous_degree_array.")

            if progress_callback: progress_callback(progress_offset + 2, "Degrees Calculated.")
        elif progress_callback:
             # If not degree-based, still increment progress to match total steps
             progress_callback(progress_offset + 2, f"'{name}' Applied.")

        # --- Final Updates (Common to both paths) ---
        if cancel_event and cancel_event.is_set(): raise InterruptedError("Initial condition application cancelled")

        grid.update_active_nodes() # Update based on final state (binary or degree)
        grid.previous_active_nodes_set = grid.active_nodes.copy()

        # Calculate previous_active_neighbor_array based on the FINAL grid state
        if grid.rule and (grid.rule.needs_neighbor_degrees or grid.rule.needs_neighbor_active_counts):
            grid.previous_active_neighbor_array = np.zeros(grid.total_nodes, dtype=np.int32)
            activity_threshold = 1e-6
            for node_idx in range(grid.total_nodes):
                 count = 0; neighbors = grid.get_neighbors(node_idx, grid.coord_system)
                 for neighbor_idx in neighbors:
                      # Check FINAL state in grid_array
                      if neighbor_idx != -1 and 0 <= neighbor_idx < grid.grid_array.size and grid.grid_array.ravel()[neighbor_idx] > activity_threshold:
                          count += 1
                 grid.previous_active_neighbor_array[node_idx] = count
            logger.debug(f"{log_prefix}Calculated previous_active_neighbor_array based on final state.")

        # Ensure spatial hash is populated AFTER final state is set
        if not grid.populate_spatial_hash():
             logger.error(f"{log_prefix}Failed to populate spatial hash after applying initial condition '{name}'.")

        logger.debug(f"{log_prefix}Applied initial condition '{name}' and updated grid state/history.")

    # --- Initialization Methods (Moved from global scope) ---

    def initialize_random(self, grid: 'Grid'):
        """Initializes the grid with a random distribution of active nodes.
           (Round 4 Fix: Added logging after setting state)"""
        log_prefix = "InitialConditionManager.initialize_random: " # Added prefix
        logger.debug(f"{log_prefix}Applying 'Random' initial condition.")
        grid.clear_grid() # Start fresh
        initial_density = grid.rule.get_param('initial_density', GlobalSettings.Simulation.INITIAL_NODE_DENSITY) if grid.rule else GlobalSettings.Simulation.INITIAL_NODE_DENSITY
        total_nodes = np.prod(grid.dimensions)
        active_cells = int(total_nodes * initial_density)
        active_indices_set = set() # Keep track of indices set

        if active_cells > 0:
            grid_size = grid.grid_array.size
            active_cells = min(active_cells, grid_size)
            active_indices = np.random.choice(grid_size, size=active_cells, replace=False)
            grid.grid_array.ravel()[active_indices] = 1.0 # Use 1.0 for active state
            active_indices_set = set(active_indices) # Store the indices we set

        # --- ADDED LOGGING ---
        active_count_after_set = np.sum(grid.grid_array > 1e-6)
        logger.info(f"{log_prefix}Set {len(active_indices_set)} nodes randomly based on density {initial_density:.2f}. Active count in grid_array NOW: {active_count_after_set}")
        # ---

        # Initialize edges based on rule parameter AFTER nodes are set
        edge_init_type = grid.rule.get_param('edge_initialization', 'RANDOM') if grid.rule else 'RANDOM'
        if edge_init_type == 'RANDOM' and grid.rule:
            grid.rule.params.setdefault('connect_probability', GlobalSettings.Simulation.INITIAL_EDGE_DENSITY)
        grid.initialize_edges_after_nodes(edge_init_type) # Use the method that handles density/type

    def initialize_empty(self, grid: 'Grid'):
        """Initializes the grid with no active nodes or edges."""
        logger.debug(f"Applying 'Empty' initial condition.")
        grid.clear_grid()

    def initialize_glider_pattern(self, grid: Grid):
        """Initializes the grid with a standard 5-node glider pattern using ShapePlacer."""
        log_prefix = "InitialConditionManager.initialize_glider_pattern: "
        logger.debug(f"{log_prefix}Applying 'Glider Pattern' initial condition.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}Glider pattern requires 2D grid. Applying Random instead.")
            self.initialize_random(grid) # Random handles its own edges
            return

        grid.clear_grid() # Start fresh

        # Define glider relative coordinates
        glider_coords_rel = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

        # Check if grid is large enough
        max_r = max(p[0] for p in glider_coords_rel)
        max_c = max(p[1] for p in glider_coords_rel)
        if grid.dimensions[0] <= max_r or grid.dimensions[1] <= max_c:
             logger.warning(f"{log_prefix}Grid dimensions {grid.dimensions} too small for Glider. Applying Random.")
             self.initialize_random(grid) # Random handles its own edges
             return

        # Create the shape definition
        # Use connectivity="none" initially, as edges will be added later if needed
        glider_shape_def = ShapeDefinition(
            name="Glider_Internal", category="Internal", # Internal use name
            relative_coords=glider_coords_rel, connectivity="none",
            shape_type=ShapeType.CUSTOM
        )

        # Determine placement origin (e.g., near top-left)
        origin = (1, 1)

        # Place the shape using ShapePlacer
        placed_indices: Optional[Set[int]] = None
        if grid.shape_placer:
            # place_shape_definition handles setting node states
            placed_indices = grid.shape_placer.place_shape_definition(glider_shape_def, origin)
            if placed_indices is None:
                logger.error(f"{log_prefix}Shape placer failed to place glider definition.")
                return # Stop if placement failed
            logger.debug(f"{log_prefix}Placed {len(placed_indices)} glider nodes.")
        else:
            logger.error(f"{log_prefix}Shape placer not available on grid.")
            return

        # Update active nodes AFTER placement
        grid.update_active_nodes()
        logger.debug(f"{log_prefix}Updated active nodes. Count: {len(grid.active_nodes)}")

        # Add default edges IF the rule requires edges
        rule_uses_edges = False
        if grid.rule:
            edge_init_type = grid.rule.get_param('edge_initialization', 'NONE')
            rule_uses_edges = edge_init_type != 'NONE'
            logger.debug(f"{log_prefix}Rule '{grid.rule.name}' uses edges: {rule_uses_edges} (Init type: {edge_init_type})")
        else:
            logger.warning(f"{log_prefix}No rule found on grid, cannot determine if edges are needed.")

        if rule_uses_edges and placed_indices and grid.shape_placer:
            logger.debug(f"{log_prefix}Adding default 'full' edges for the placed glider nodes.")
            # Use the ShapePlacer's method to add edges between the placed nodes
            grid.shape_placer.add_default_edges(placed_indices, "full")
        elif not rule_uses_edges:
             logger.debug(f"{log_prefix}Skipping default edge addition as rule does not use edges.")

    def initialize_gosper_glider_gun_pattern(self, grid: Grid):
        """Initializes the grid with a Gosper Glider Gun pattern using ShapePlacer."""
        log_prefix = "InitialConditionManager.initialize_gosper_glider_gun_pattern: "
        logger.debug(f"{log_prefix}Applying 'Gosper Glider Gun Pattern' initial condition.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}Gosper Glider Gun pattern requires 2D grid. Applying Random instead.")
            self.initialize_random(grid) # Random handles its own edges
            return

        grid.clear_grid() # Start fresh

        # Define Gosper Glider Gun relative coordinates
        glider_gun_coords = [
            (5, 1), (5, 2), (6, 1), (6, 2), (3, 13), (3, 14), (4, 12), (4, 16),
            (5, 11), (5, 17), (6, 11), (6, 15), (6, 17), (6, 18), (7, 11), (7, 17),
            (8, 12), (8, 16), (9, 13), (9, 14), (1, 25), (2, 23), (2, 25),
            (3, 21), (3, 22), (4, 21), (4, 22), (5, 21), (5, 22), (6, 23), (6, 25),
            (7, 25), (3, 35), (3, 36), (4, 35), (4, 36)
        ]

        # Check if grid is large enough
        max_r = max(p[0] for p in glider_gun_coords)
        max_c = max(p[1] for p in glider_gun_coords)
        if grid.dimensions[0] <= max_r or grid.dimensions[1] <= max_c:
             logger.warning(f"{log_prefix}Grid dimensions {grid.dimensions} too small for Gosper Glider Gun. Applying Random.")
             self.initialize_random(grid) # Random handles its own edges
             return

        # Create the shape definition
        glider_gun_shape_def = ShapeDefinition(
            name="GosperGun_Internal", category="Internal",
            relative_coords=glider_gun_coords, connectivity="none",
            shape_type=ShapeType.CUSTOM
        )

        # Determine placement origin (e.g., near top-left)
        origin = (1, 1)

        # Place the shape using ShapePlacer
        placed_indices: Optional[Set[int]] = None
        if grid.shape_placer:
            placed_indices = grid.shape_placer.place_shape_definition(glider_gun_shape_def, origin)
            if placed_indices is None:
                logger.error(f"{log_prefix}Shape placer failed to place Gosper Glider Gun definition.")
                return
            logger.debug(f"{log_prefix}Placed {len(placed_indices)} Gosper Glider Gun nodes.")
        else:
            logger.error(f"{log_prefix}Shape placer not available on grid.")
            return

        # Update active nodes AFTER placement
        grid.update_active_nodes()
        logger.debug(f"{log_prefix}Updated active nodes. Count: {len(grid.active_nodes)}")

        # Add default edges IF the rule requires edges
        rule_uses_edges = False
        if grid.rule:
            edge_init_type = grid.rule.get_param('edge_initialization', 'NONE')
            rule_uses_edges = edge_init_type != 'NONE'
            logger.debug(f"{log_prefix}Rule '{grid.rule.name}' uses edges: {rule_uses_edges} (Init type: {edge_init_type})")
        else:
            logger.warning(f"{log_prefix}No rule found on grid, cannot determine if edges are needed.")

        if rule_uses_edges and placed_indices and grid.shape_placer:
            logger.debug(f"{log_prefix}Adding default 'full' edges for the placed Gosper Glider Gun nodes.")
            grid.shape_placer.add_default_edges(placed_indices, "full")
        elif not rule_uses_edges:
             logger.debug(f"{log_prefix}Skipping default edge addition as rule does not use edges.")

    def initialize_2d_square(self, grid: Grid):
        """Initializes the grid with a square in the center using ShapePlacer."""
        log_prefix = "InitialConditionManager.initialize_2d_square: "
        logger.debug(f"{log_prefix}Applying '2D - Square' initial condition.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}2D Square requires 2D grid. Applying Random instead.")
            self.initialize_random(grid) # Random handles its own edges
            return

        grid.clear_grid() # Start fresh

        # Determine size and origin
        size = min(grid.dimensions) // 4
        size = max(2, size) # Ensure minimum size
        origin = ((grid.dimensions[0] // 2) - size // 2, (grid.dimensions[1] // 2) - size // 2)

        # Create the shape definition
        square_shape_def = ShapeDefinition(
            name="Square_Internal", category="Internal",
            relative_coords=Square(size, filled=True).get_relative_coordinates(), # Get coords from Square class
            connectivity="none", # Edges handled later
            shape_type=ShapeType.SQUARE
        )

        # Place the shape using ShapePlacer
        placed_indices: Optional[Set[int]] = None
        if grid.shape_placer:
            placed_indices = grid.shape_placer.place_shape_definition(square_shape_def, origin)
            if placed_indices is None:
                logger.error(f"{log_prefix}Shape placer failed to place square definition.")
                return
            logger.debug(f"{log_prefix}Placed {len(placed_indices)} square nodes.")
        else:
            logger.error(f"{log_prefix}Shape placer not available on grid.")
            return

        # Update active nodes AFTER placement
        grid.update_active_nodes()
        logger.debug(f"{log_prefix}Updated active nodes. Count: {len(grid.active_nodes)}")

        # Add default edges IF the rule requires edges
        rule_uses_edges = False
        if grid.rule:
            edge_init_type = grid.rule.get_param('edge_initialization', 'NONE')
            rule_uses_edges = edge_init_type != 'NONE'
            logger.debug(f"{log_prefix}Rule '{grid.rule.name}' uses edges: {rule_uses_edges} (Init type: {edge_init_type})")
        else:
            logger.warning(f"{log_prefix}No rule found on grid, cannot determine if edges are needed.")

        if rule_uses_edges and placed_indices and grid.shape_placer:
            logger.debug(f"{log_prefix}Adding default 'full' edges for the placed square nodes.")
            grid.shape_placer.add_default_edges(placed_indices, "full")
        elif not rule_uses_edges:
             logger.debug(f"{log_prefix}Skipping default edge addition as rule does not use edges.")

    def initialize_2d_circle(self, grid: Grid):
        """Initializes the grid with a circle in the center using ShapePlacer."""
        log_prefix = "InitialConditionManager.initialize_2d_circle: "
        logger.debug(f"{log_prefix}Applying '2D - Circle' initial condition.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}2D Circle requires 2D grid. Applying Random instead.")
            self.initialize_random(grid) # Random handles its own edges
            return

        grid.clear_grid() # Start fresh

        # Determine radius and center
        center = (grid.dimensions[0] // 2, grid.dimensions[1] // 2)
        radius = min(grid.dimensions) // 4
        radius = max(1, radius) # Ensure minimum radius

        # Create the shape definition
        circle_shape_def = ShapeDefinition(
            name="Circle_Internal", category="Internal",
            relative_coords=Circle(radius, filled=True).get_relative_coordinates(), # Get coords from Circle class
            connectivity="none", # Edges handled later
            shape_type=ShapeType.CIRCLE
        )

        # Place the shape using ShapePlacer (origin is the center for Circle)
        placed_indices: Optional[Set[int]] = None
        if grid.shape_placer:
            # Note: place_shape_definition uses origin, but Circle coords are relative to 0,0.
            # We need to adjust the origin passed to place_shape_definition.
            # Since Circle coords range from -radius to +radius, the effective origin
            # to center it should be the grid center.
            placed_indices = grid.shape_placer.place_shape_definition(circle_shape_def, center)
            if placed_indices is None:
                logger.error(f"{log_prefix}Shape placer failed to place circle definition.")
                return
            logger.debug(f"{log_prefix}Placed {len(placed_indices)} circle nodes.")
        else:
            logger.error(f"{log_prefix}Shape placer not available on grid.")
            return

        # Update active nodes AFTER placement
        grid.update_active_nodes()
        logger.debug(f"{log_prefix}Updated active nodes. Count: {len(grid.active_nodes)}")

        # Add default edges IF the rule requires edges
        rule_uses_edges = False
        if grid.rule:
            edge_init_type = grid.rule.get_param('edge_initialization', 'NONE')
            rule_uses_edges = edge_init_type != 'NONE'
            logger.debug(f"{log_prefix}Rule '{grid.rule.name}' uses edges: {rule_uses_edges} (Init type: {edge_init_type})")
        else:
            logger.warning(f"{log_prefix}No rule found on grid, cannot determine if edges are needed.")

        if rule_uses_edges and placed_indices and grid.shape_placer:
            logger.debug(f"{log_prefix}Adding default 'full' edges for the placed circle nodes.")
            grid.shape_placer.add_default_edges(placed_indices, "full")
        elif not rule_uses_edges:
             logger.debug(f"{log_prefix}Skipping default edge addition as rule does not use edges.")

    # TODO: Implement the missing Cube method and then re-enable this
    # def initialize_3d_cube(self, grid: Grid):
    #     """Initializes the grid with a cube in the center (3D only) using ShapePlacer."""
    #     log_prefix = "InitialConditionManager.initialize_3d_cube: "
    #     logger.debug(f"{log_prefix}Applying '3D - Cube' initial condition.")

    #     if grid.dimension_type != Dimension.THREE_D:
    #         logger.warning(f"{log_prefix}3D Cube requires 3D grid. Applying Random instead.")
    #         self.initialize_random(grid) # Random handles its own edges
    #         return

    #     grid.clear_grid() # Start fresh

    #     # Determine size and origin
    #     size = min(grid.dimensions) // 4
    #     size = max(2, size) # Ensure minimum size
    #     origin = tuple((d // 2) - size // 2 for d in grid.dimensions)

    #     # Create the shape definition
    #     cube_shape_def = ShapeDefinition(
    #         name="Cube_Internal", category="Internal",
    #         relative_coords=Cube(size, filled=True).get_relative_coordinates(), # Get coords from Cube class
    #         connectivity="none", # Edges handled later
    #         shape_type=ShapeType.CUBE
    #     )

    #     # Place the shape using ShapePlacer
    #     placed_indices: Optional[Set[int]] = None
    #     if grid.shape_placer:
    #         placed_indices = grid.shape_placer.place_shape_definition(cube_shape_def, origin)
    #         if placed_indices is None:
    #             logger.error(f"{log_prefix}Shape placer failed to place cube definition.")
    #             return
    #         logger.debug(f"{log_prefix}Placed {len(placed_indices)} cube nodes.")
    #     else:
    #         logger.error(f"{log_prefix}Shape placer not available on grid.")
    #         return

    #     # Update active nodes AFTER placement
    #     grid.update_active_nodes()
    #     logger.debug(f"{log_prefix}Updated active nodes. Count: {len(grid.active_nodes)}")

    #     # Add default edges IF the rule requires edges
    #     rule_uses_edges = False
    #     if grid.rule:
    #         edge_init_type = grid.rule.get_param('edge_initialization', 'NONE')
    #         rule_uses_edges = edge_init_type != 'NONE'
    #         logger.debug(f"{log_prefix}Rule '{grid.rule.name}' uses edges: {rule_uses_edges} (Init type: {edge_init_type})")
    #     else:
    #         logger.warning(f"{log_prefix}No rule found on grid, cannot determine if edges are needed.")

    #     if rule_uses_edges and placed_indices and grid.shape_placer:
    #         logger.debug(f"{log_prefix}Adding default 'full' edges for the placed cube nodes.")
    #         grid.shape_placer.add_default_edges(placed_indices, "full")
    #     elif not rule_uses_edges:
    #          logger.debug(f"{log_prefix}Skipping default edge addition as rule does not use edges.")

    def initialize_3d_sphere(self, grid: Grid):
        """Initializes the grid with a sphere in the center (3D only) using ShapePlacer."""
        log_prefix = "InitialConditionManager.initialize_3d_sphere: "
        logger.debug(f"{log_prefix}Applying '3D - Sphere' initial condition.")

        if grid.dimension_type != Dimension.THREE_D:
            logger.warning(f"{log_prefix}3D Sphere requires 3D grid. Applying Random instead.")
            self.initialize_random(grid) # Random handles its own edges
            return

        grid.clear_grid() # Start fresh

        # Determine radius and center
        center = tuple(d // 2 for d in grid.dimensions)
        radius = min(grid.dimensions) // 4
        radius = max(1, radius) # Ensure minimum radius

        # Create the shape definition
        # Sphere class needs center for calculation, but coords should be relative to (0,0,0)
        sphere_shape_def = ShapeDefinition(
            name="Sphere_Internal", category="Internal",
            relative_coords=Sphere(center=(0,0,0), radius=radius, filled=True).get_relative_coordinates(),
            connectivity="none", # Edges handled later
            shape_type=ShapeType.SPHERE
        )

        # Place the shape using ShapePlacer (origin is the grid center)
        placed_indices: Optional[Set[int]] = None
        if grid.shape_placer:
            placed_indices = grid.shape_placer.place_shape_definition(sphere_shape_def, center)
            if placed_indices is None:
                logger.error(f"{log_prefix}Shape placer failed to place sphere definition.")
                return
            logger.debug(f"{log_prefix}Placed {len(placed_indices)} sphere nodes.")
        else:
            logger.error(f"{log_prefix}Shape placer not available on grid.")
            return

        # Update active nodes AFTER placement
        grid.update_active_nodes()
        logger.debug(f"{log_prefix}Updated active nodes. Count: {len(grid.active_nodes)}")

        # Add default edges IF the rule requires edges
        rule_uses_edges = False
        if grid.rule:
            edge_init_type = grid.rule.get_param('edge_initialization', 'NONE')
            rule_uses_edges = edge_init_type != 'NONE'
            logger.debug(f"{log_prefix}Rule '{grid.rule.name}' uses edges: {rule_uses_edges} (Init type: {edge_init_type})")
        else:
            logger.warning(f"{log_prefix}No rule found on grid, cannot determine if edges are needed.")

        if rule_uses_edges and placed_indices and grid.shape_placer:
            logger.debug(f"{log_prefix}Adding default 'full' edges for the placed sphere nodes.")
            grid.shape_placer.add_default_edges(placed_indices, "full")
        elif not rule_uses_edges:
             logger.debug(f"{log_prefix}Skipping default edge addition as rule does not use edges.")

    def initialize_square_tessellation(self, grid: Grid):
        """Initializes the grid with a square tessellation using ShapePlacer."""
        log_prefix = "InitialConditionManager.initialize_square_tessellation: "
        logger.debug(f"{log_prefix}Applying '2D - Square Tessellation' initial condition.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}Square tessellation requires 2D grid. Applying Random instead.")
            self.initialize_random(grid) # Random handles its own edges
            return

        grid.clear_grid() # Start fresh

        # Get parameters (consider making these rule params later if needed)
        square_size = 5
        spacing = 2

        if grid.shape_placer is None:
            logger.error(f"{log_prefix}Shape placer not available on grid.")
            return

        all_placed_indices: Set[int] = set() # Accumulate indices from all squares

        # Iterate and place squares
        for row in range(0, grid.dimensions[0] - square_size + 1, square_size + spacing):
            for col in range(0, grid.dimensions[1] - square_size + 1, square_size + spacing):
                # Create the square shape definition for this instance
                square_shape = ShapeGenerator.create_square(square_size, filled=True, connectivity="none")
                origin = (row, col)

                # Place the nodes using ShapePlacer
                placed_indices_this_square = grid.shape_placer.place_shape(square_shape, origin) # place_shape is sufficient
                if placed_indices_this_square:
                    all_placed_indices.update(placed_indices_this_square)

        logger.debug(f"{log_prefix}Placed {len(all_placed_indices)} total nodes for tessellation.")

        # Update active nodes AFTER all placements
        grid.update_active_nodes()
        logger.debug(f"{log_prefix}Updated active nodes. Count: {len(grid.active_nodes)}")

        # Add default edges IF the rule requires edges, using the accumulated indices
        rule_uses_edges = False
        if grid.rule:
            edge_init_type = grid.rule.get_param('edge_initialization', 'NONE')
            rule_uses_edges = edge_init_type != 'NONE'
            logger.debug(f"{log_prefix}Rule '{grid.rule.name}' uses edges: {rule_uses_edges} (Init type: {edge_init_type})")
        else:
            logger.warning(f"{log_prefix}No rule found on grid, cannot determine if edges are needed.")

        if rule_uses_edges and all_placed_indices and grid.shape_placer:
            logger.debug(f"{log_prefix}Adding default 'full' edges for the tessellation nodes.")
            # Use the ShapePlacer's method to add edges between ALL placed nodes
            grid.shape_placer.add_default_edges(all_placed_indices, "full")
        elif not rule_uses_edges:
             logger.debug(f"{log_prefix}Skipping default edge addition as rule does not use edges.")

    def initialize_triangle_tessellation(self, grid: Grid):
        """Initializes the grid with a triangle tessellation using ShapePlacer."""
        log_prefix = "InitialConditionManager.initialize_triangle_tessellation: "
        logger.debug(f"{log_prefix}Applying '2D - Triangle Tessellation' initial condition.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}Triangle tessellation requires 2D grid. Applying Random instead.")
            self.initialize_random(grid) # Random handles its own edges
            return

        grid.clear_grid() # Start fresh

        # Get parameters (consider making these rule params later if needed)
        triangle_size = 5
        spacing = 1
        row_step = triangle_size + spacing
        # Approximate column step for equilateral triangles
        col_step = int((triangle_size * 0.866) + spacing) # 0.866 is approx sqrt(3)/2

        if grid.shape_placer is None:
            logger.error(f"{log_prefix}Shape placer not available on grid.")
            return

        all_placed_indices: Set[int] = set() # Accumulate indices from all triangles

        # Iterate and place triangles
        for row in range(0, grid.dimensions[0] - triangle_size + 1, row_step):
            for col_idx, col in enumerate(range(0, grid.dimensions[1] - triangle_size + 1, col_step)):
                # Create the triangle shape definition for this instance
                # Note: Triangle shape origin might need adjustment depending on its definition
                # Assuming create_triangle places it relative to a corner or center
                triangle_shape = ShapeGenerator.create_triangle(corner=(0,0), side_length=triangle_size, connectivity="none")

                # Adjust origin based on how create_triangle works.
                # If it uses bottom-left corner:
                # origin = (row + triangle_size - 1, col)
                # If it uses top corner:
                origin = (row, col + triangle_size // 2) # Example if origin is top point

                # Place the nodes using ShapePlacer
                placed_indices_this_triangle = grid.shape_placer.place_shape(triangle_shape, origin)
                if placed_indices_this_triangle:
                    all_placed_indices.update(placed_indices_this_triangle)

        logger.debug(f"{log_prefix}Placed {len(all_placed_indices)} total nodes for tessellation.")

        # Update active nodes AFTER all placements
        grid.update_active_nodes()
        logger.debug(f"{log_prefix}Updated active nodes. Count: {len(grid.active_nodes)}")

        # Add default edges IF the rule requires edges, using the accumulated indices
        rule_uses_edges = False
        if grid.rule:
            edge_init_type = grid.rule.get_param('edge_initialization', 'NONE')
            rule_uses_edges = edge_init_type != 'NONE'
            logger.debug(f"{log_prefix}Rule '{grid.rule.name}' uses edges: {rule_uses_edges} (Init type: {edge_init_type})")
        else:
            logger.warning(f"{log_prefix}No rule found on grid, cannot determine if edges are needed.")

        if rule_uses_edges and all_placed_indices and grid.shape_placer:
            logger.debug(f"{log_prefix}Adding default 'full' edges for the tessellation nodes.")
            # Use the ShapePlacer's method to add edges between ALL placed nodes
            grid.shape_placer.add_default_edges(all_placed_indices, "full")
        elif not rule_uses_edges:
             logger.debug(f"{log_prefix}Skipping default edge addition as rule does not use edges.")

    def initialize_2d_square_with_edges(self, grid: Grid):
        """Initializes the grid with a square in the center and adds edges based on rule."""
        log_prefix = "InitialConditionManager.initialize_2d_square_with_edges: "
        logger.debug(f"{log_prefix}Applying '2D - Square with Edges' initial condition.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}2D Square requires 2D grid. Applying Random instead.")
            self.initialize_random(grid)
            return

        grid.clear_grid() # Start fresh

        # Determine size and origin
        size = min(grid.dimensions) // 4
        size = max(2, size) # Ensure minimum size
        origin = ((grid.dimensions[0] // 2) - size // 2, (grid.dimensions[1] // 2) - size // 2)

        # Create the shape definition (connectivity='none' initially)
        square_shape_def = ShapeDefinition(
            name="Square_Internal_Edges", category="Internal",
            relative_coords=Square(size, filled=True).get_relative_coordinates(),
            connectivity="none", # Edges handled separately
            shape_type=ShapeType.SQUARE
        )

        # Place the shape using ShapePlacer
        placed_indices: Optional[Set[int]] = None
        if grid.shape_placer:
            placed_indices = grid.shape_placer.place_shape_definition(square_shape_def, origin)
            if placed_indices is None:
                logger.error(f"{log_prefix}Shape placer failed to place square definition.")
                return
            logger.debug(f"{log_prefix}Placed {len(placed_indices)} square nodes.")
        else:
            logger.error(f"{log_prefix}Shape placer not available on grid.")
            return

        # Update active nodes AFTER placement
        grid.update_active_nodes()
        logger.debug(f"{log_prefix}Updated active nodes. Count: {len(grid.active_nodes)}")

        # Add default edges IF the rule requires edges
        rule_uses_edges = False
        if grid.rule:
            edge_init_type = grid.rule.get_param('edge_initialization', 'NONE')
            rule_uses_edges = edge_init_type != 'NONE'
            logger.debug(f"{log_prefix}Rule '{grid.rule.name}' uses edges: {rule_uses_edges} (Init type: {edge_init_type})")
        else:
            logger.warning(f"{log_prefix}No rule found on grid, cannot determine if edges are needed.")

        if rule_uses_edges and placed_indices and grid.shape_placer:
            logger.debug(f"{log_prefix}Adding default 'full' edges for the placed square nodes.")
            grid.shape_placer.add_default_edges(placed_indices, "full") # Use 'full' connectivity
        elif not rule_uses_edges:
             logger.debug(f"{log_prefix}Skipping default edge addition as rule does not use edges.")

    def initialize_2d_circle_with_edges(self, grid: Grid):
        """Initializes the grid with a circle in the center and adds edges based on rule."""
        log_prefix = "InitialConditionManager.initialize_2d_circle_with_edges: "
        logger.debug(f"{log_prefix}Applying '2D - Circle with Edges' initial condition.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}2D Circle requires 2D grid. Applying Random instead.")
            self.initialize_random(grid)
            return

        grid.clear_grid() # Start fresh

        # Determine radius and center
        center = (grid.dimensions[0] // 2, grid.dimensions[1] // 2)
        radius = min(grid.dimensions) // 4
        radius = max(1, radius) # Ensure minimum radius

        # Create the shape definition
        circle_shape_def = ShapeDefinition(
            name="Circle_Internal_Edges", category="Internal",
            relative_coords=Circle(radius, filled=True).get_relative_coordinates(),
            connectivity="none", # Edges handled separately
            shape_type=ShapeType.CIRCLE
        )

        # Place the shape using ShapePlacer
        placed_indices: Optional[Set[int]] = None
        if grid.shape_placer:
            placed_indices = grid.shape_placer.place_shape_definition(circle_shape_def, center)
            if placed_indices is None:
                logger.error(f"{log_prefix}Shape placer failed to place circle definition.")
                return
            logger.debug(f"{log_prefix}Placed {len(placed_indices)} circle nodes.")
        else:
            logger.error(f"{log_prefix}Shape placer not available on grid.")
            return

        # Update active nodes AFTER placement
        grid.update_active_nodes()
        logger.debug(f"{log_prefix}Updated active nodes. Count: {len(grid.active_nodes)}")

        # Add default edges IF the rule requires edges
        rule_uses_edges = False
        if grid.rule:
            edge_init_type = grid.rule.get_param('edge_initialization', 'NONE')
            rule_uses_edges = edge_init_type != 'NONE'
            logger.debug(f"{log_prefix}Rule '{grid.rule.name}' uses edges: {rule_uses_edges} (Init type: {edge_init_type})")
        else:
            logger.warning(f"{log_prefix}No rule found on grid, cannot determine if edges are needed.")

        if rule_uses_edges and placed_indices and grid.shape_placer:
            logger.debug(f"{log_prefix}Adding default 'full' edges for the placed circle nodes.")
            grid.shape_placer.add_default_edges(placed_indices, "full")
        elif not rule_uses_edges:
             logger.debug(f"{log_prefix}Skipping default edge addition as rule does not use edges.")

    def initialize_symmetry_regions(self, grid: Grid):
        """Initializes grid with symmetric squares in some regions and random noise in others."""
        log_prefix = "InitialConditionManager.initialize_symmetry_regions: "
        logger.debug(f"{log_prefix}Applying 'Symmetry Regions' initial condition.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}Symmetry Regions requires 2D grid. Applying Random instead.")
            self.initialize_random(grid)
            return

        grid.clear_grid() # Start fresh
        rows, cols = grid.dimensions
        center_r, center_c = rows // 2, cols // 2
        quadrant_size_r, quadrant_size_c = rows // 2, cols // 2
        square_size = max(2, min(quadrant_size_r, quadrant_size_c) // 2) # Size relative to quadrant

        # Quadrant Origins (Top-Left)
        origins = {
            "TL": (center_r // 2 - square_size // 2, center_c // 2 - square_size // 2), # Top-Left Quadrant Center
            "TR": (center_r // 2 - square_size // 2, center_c + quadrant_size_c // 2 - square_size // 2), # Top-Right
            "BL": (center_r + quadrant_size_r // 2 - square_size // 2, center_c // 2 - square_size // 2), # Bottom-Left
            "BR": (center_r + quadrant_size_r // 2 - square_size // 2, center_c + quadrant_size_c // 2 - square_size // 2) # Bottom-Right
        }

        # Place Squares in TL and BR
        square_shape = Square(square_size, filled=True)
        if grid.shape_placer:
            grid.shape_placer.place_shape(square_shape, origins["TL"])
            grid.shape_placer.place_shape(square_shape, origins["BR"])
            logger.debug(f"{log_prefix}Placed squares in TL and BR quadrants.")
        else:
            logger.error(f"{log_prefix}Shape placer not available.")
            return

        # Place Random Noise in TR and BL
        random_density = grid.rule.get_param('initial_density', 0.3) if grid.rule else 0.3
        # Top-Right Quadrant
        for r in range(0, center_r):
            for c in range(center_c, cols):
                if random.random() < random_density:
                    if grid.is_valid_coord((r, c)): grid.grid_array[r, c] = 1.0
        # Bottom-Left Quadrant
        for r in range(center_r, rows):
            for c in range(0, center_c):
                if random.random() < random_density:
                     if grid.is_valid_coord((r, c)): grid.grid_array[r, c] = 1.0
        logger.debug(f"{log_prefix}Placed random noise (density ~{random_density:.2f}) in TR and BL quadrants.")

        # Update active nodes AFTER all placements
        grid.update_active_nodes()
        logger.debug(f"{log_prefix}Updated active nodes. Count: {len(grid.active_nodes)}")

        # Initialize edges globally based on rule parameter
        edge_init_type = grid.rule.get_param('edge_initialization', 'RANDOM') if grid.rule else 'RANDOM'
        if edge_init_type == 'RANDOM' and grid.rule:
            grid.rule.params.setdefault('connect_probability', GlobalSettings.Simulation.INITIAL_EDGE_DENSITY)
        grid.initialize_edges_after_nodes(edge_init_type)
        logger.debug(f"{log_prefix}Initialized global edges using type: {edge_init_type}")

    def initialize_symmetric_core_random_shell(self, grid: Grid):
        """Initializes grid with a symmetric shape (e.g., square) in the center and random noise around it."""
        log_prefix = "InitialConditionManager.initialize_symmetric_core_random_shell: "
        logger.debug(f"{log_prefix}Applying 'Symmetric Core, Random Shell' initial condition.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}Symmetric Core requires 2D grid. Applying Random instead.")
            self.initialize_random(grid)
            return

        grid.clear_grid() # Start fresh
        rows, cols = grid.dimensions

        # Place Core Shape (e.g., Square)
        core_size = min(rows, cols) // 3
        core_size = max(2, core_size)
        core_origin = ((rows // 2) - core_size // 2, (cols // 2) - core_size // 2)
        core_shape = Square(core_size, filled=True)
        core_placed_indices: Set[int] = set()
        core_coords_set: Set[Tuple[int,int]] = set()

        if grid.shape_placer:
            temp_indices = grid.shape_placer.place_shape(core_shape, core_origin)
            if temp_indices:
                core_placed_indices = temp_indices
                core_coords_set = {tuple(_unravel_index(idx, grid.dimensions)) for idx in core_placed_indices} # type: ignore
            logger.debug(f"{log_prefix}Placed core square ({core_size}x{core_size}) at {core_origin}.")
        else:
            logger.error(f"{log_prefix}Shape placer not available.")
            return

        # Place Random Noise in Shell (avoiding core)
        random_density = grid.rule.get_param('initial_density', 0.2) if grid.rule else 0.2 # Lower density for shell
        shell_node_count = 0
        for r in range(rows):
            for c in range(cols):
                coord = (r, c)
                if coord not in core_coords_set: # Check if outside the core
                    if random.random() < random_density:
                        if grid.is_valid_coord(tuple(map(int, coord))):
                            grid.grid_array[r, c] = 1.0
                            shell_node_count += 1
        logger.debug(f"{log_prefix}Placed {shell_node_count} random noise nodes in shell (density ~{random_density:.2f}).")

        # Update active nodes AFTER all placements
        grid.update_active_nodes()
        logger.debug(f"{log_prefix}Updated active nodes. Count: {len(grid.active_nodes)}")

        # Initialize edges globally based on rule parameter
        edge_init_type = grid.rule.get_param('edge_initialization', 'RANDOM') if grid.rule else 'RANDOM'
        if edge_init_type == 'RANDOM' and grid.rule:
            grid.rule.params.setdefault('connect_probability', GlobalSettings.Simulation.INITIAL_EDGE_DENSITY)
        grid.initialize_edges_after_nodes(edge_init_type)
        logger.debug(f"{log_prefix}Initialized global edges using type: {edge_init_type}")

    def initialize_hollow_square_with_edges(self, grid: Grid):
        """Initializes grid with a hollow square in the center and perimeter edges."""
        log_prefix = "ICM.initialize_hollow_square_with_edges: "
        logger.debug(f"{log_prefix}Applying '2D - Hollow Square with Edges'.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}Requires 2D grid. Applying Random.")
            self.initialize_random(grid); return

        grid.clear_grid()
        size = max(3, min(grid.dimensions) // 3) # Ensure size is at least 3
        origin = ((grid.dimensions[0] // 2) - size // 2, (grid.dimensions[1] // 2) - size // 2)

        square_shape = Square(size, filled=False) # Hollow square
        placed_indices: Optional[Set[int]] = None
        if grid.shape_placer:
            placed_indices = grid.shape_placer.place_shape(square_shape, origin)
            if placed_indices is None: logger.error(f"{log_prefix}Placement failed."); return
            logger.debug(f"{log_prefix}Placed {len(placed_indices)} hollow square nodes.")
        else: logger.error(f"{log_prefix}Shape placer not available."); return

        grid.update_active_nodes()

        if grid.rule and grid.rule.get_param('edge_initialization', 'NONE') != 'NONE' and placed_indices and grid.shape_placer:
            logger.debug(f"{log_prefix}Adding 'perimeter' edges.")
            grid.shape_placer.add_default_edges(placed_indices, "perimeter")
        elif not (grid.rule and grid.rule.get_param('edge_initialization', 'NONE') != 'NONE'):
             logger.debug(f"{log_prefix}Skipping edge addition as rule does not use edges.")

        # REMOVED: self.parent_gui._calculate_initial_previous_arrays()

    def initialize_hollow_circle_with_edges(self, grid: Grid):
        """Initializes grid with a hollow circle in the center and perimeter edges."""
        log_prefix = "ICM.initialize_hollow_circle_with_edges: "
        logger.debug(f"{log_prefix}Applying '2D - Hollow Circle with Edges'.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}Requires 2D grid. Applying Random.")
            self.initialize_random(grid); return

        grid.clear_grid()
        center = (grid.dimensions[0] // 2, grid.dimensions[1] // 2)
        radius = max(2, min(grid.dimensions) // 3) # Ensure radius is at least 2

        circle_shape = Circle(radius, filled=False) # Hollow circle
        placed_indices: Optional[Set[int]] = None
        if grid.shape_placer:
            placed_indices = grid.shape_placer.place_shape(circle_shape, center)
            if placed_indices is None: logger.error(f"{log_prefix}Placement failed."); return
            logger.debug(f"{log_prefix}Placed {len(placed_indices)} hollow circle nodes.")
        else: logger.error(f"{log_prefix}Shape placer not available."); return

        grid.update_active_nodes()

        if grid.rule and grid.rule.get_param('edge_initialization', 'NONE') != 'NONE' and placed_indices and grid.shape_placer:
            logger.debug(f"{log_prefix}Adding 'perimeter' edges.")
            grid.shape_placer.add_default_edges(placed_indices, "perimeter")
        elif not (grid.rule and grid.rule.get_param('edge_initialization', 'NONE') != 'NONE'):
             logger.debug(f"{log_prefix}Skipping edge addition as rule does not use edges.")

        # REMOVED: self.parent_gui._calculate_initial_previous_arrays()

    def initialize_3d_sphere_with_edges(self, grid: Grid):
        """Initializes the grid with a sphere in the center and adds edges based on rule (3D only)."""
        log_prefix = "ICM.initialize_3d_sphere_with_edges: "
        logger.debug(f"{log_prefix}Applying '3D - Sphere with Edges' initial condition.")

        if grid.dimension_type != Dimension.THREE_D:
            logger.warning(f"{log_prefix}3D Sphere requires 3D grid. Applying Random instead.")
            self.initialize_random(grid)
            return

        grid.clear_grid() # Start fresh

        # Determine radius and center
        center = tuple(d // 2 for d in grid.dimensions)
        radius = min(grid.dimensions) // 4
        radius = max(1, radius) # Ensure minimum radius

        # Create the shape definition
        sphere_shape_def = ShapeDefinition(
            name="Sphere_Internal_Edges", category="Internal",
            relative_coords=Sphere(center=(0,0,0), radius=radius, filled=True).get_relative_coordinates(),
            connectivity="none", # Edges handled separately
            shape_type=ShapeType.SPHERE
        )

        # Place the shape using ShapePlacer
        placed_indices: Optional[Set[int]] = None
        if grid.shape_placer:
            placed_indices = grid.shape_placer.place_shape_definition(sphere_shape_def, center)
            if placed_indices is None:
                logger.error(f"{log_prefix}Shape placer failed to place sphere definition.")
                return
            logger.debug(f"{log_prefix}Placed {len(placed_indices)} sphere nodes.")
        else:
            logger.error(f"{log_prefix}Shape placer not available on grid.")
            return

        # Update active nodes AFTER placement
        grid.update_active_nodes()
        logger.debug(f"{log_prefix}Updated active nodes. Count: {len(grid.active_nodes)}")

        # Add default edges IF the rule requires edges
        rule_uses_edges = False
        if grid.rule:
            edge_init_type = grid.rule.get_param('edge_initialization', 'NONE')
            rule_uses_edges = edge_init_type != 'NONE'
            logger.debug(f"{log_prefix}Rule '{grid.rule.name}' uses edges: {rule_uses_edges} (Init type: {edge_init_type})")
        else:
            logger.warning(f"{log_prefix}No rule found on grid, cannot determine if edges are needed.")

        if rule_uses_edges and placed_indices and grid.shape_placer:
            logger.debug(f"{log_prefix}Adding default 'full' edges for the placed sphere nodes.")
            grid.shape_placer.add_default_edges(placed_indices, "full")
        elif not rule_uses_edges:
             logger.debug(f"{log_prefix}Skipping default edge addition as rule does not use edges.")

        # REMOVED: self.parent_gui._calculate_initial_previous_arrays()

    def initialize_grid_lattice_von_neumann(self, grid: Grid):
        """Initializes grid with nodes on a lattice and Von Neumann edges between them."""
        log_prefix = "ICM.initialize_grid_lattice_von_neumann: "
        logger.debug(f"{log_prefix}Applying 'Grid Lattice (Von Neumann Edges)'.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}Requires 2D grid. Applying Random.")
            self.initialize_random(grid); return

        grid.clear_grid()
        spacing = 3 # Place node every 3 cells (adjust as needed)
        rows, cols = grid.dimensions
        placed_indices: Set[int] = set()
        placed_coords: Set[Tuple[int,int]] = set()

        # Place nodes
        for r in range(0, rows, spacing):
            for c in range(0, cols, spacing):
                coord = (r, c)
                if grid.is_valid_coord(tuple(map(int, coord))):
                    idx = _ravel_multi_index(np.array(coord), grid.dimensions)
                    grid.grid_array[r, c] = 1.0
                    placed_indices.add(idx)
                    placed_coords.add(coord)
        logger.debug(f"{log_prefix}Placed {len(placed_indices)} lattice nodes.")

        # Add Von Neumann edges between lattice nodes
        edges_added_count = 0
        if grid.rule and grid.rule.get_param('edge_initialization', 'NONE') != 'NONE':
            logger.debug(f"{log_prefix}Adding Von Neumann edges...")
            vn_offsets = [(-spacing, 0), (spacing, 0), (0, -spacing), (0, spacing)]
            for r_node, c_node in placed_coords:
                idx1 = _ravel_multi_index(np.array((r_node, c_node)), grid.dimensions)
                for dr, dc in vn_offsets:
                    neighbor_coord = (r_node + dr, c_node + dc)
                    if neighbor_coord in placed_coords: # Check if neighbor is also a lattice node
                        idx2 = _ravel_multi_index(np.array(neighbor_coord), grid.dimensions)
                        # Add edge (add_edge handles ordering and duplicates)
                        grid.add_edge(idx1, idx2, edge_state=1.0)
                        edges_added_count += 1 # Count each potential connection attempt
            # Divide by 2 because we check each connection twice
            logger.debug(f"{log_prefix}Added {edges_added_count // 2} Von Neumann edges.")
        elif not (grid.rule and grid.rule.get_param('edge_initialization', 'NONE') != 'NONE'):
             logger.debug(f"{log_prefix}Skipping edge addition as rule does not use edges.")

        grid.update_active_nodes()
        # REMOVED: self.parent_gui._calculate_initial_previous_arrays()

    def initialize_grid_lattice_moore(self, grid: Grid):
        """Initializes grid with nodes on a lattice and Moore edges between them."""
        log_prefix = "ICM.initialize_grid_lattice_moore: "
        logger.debug(f"{log_prefix}Applying 'Grid Lattice (Moore Edges)'.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}Requires 2D grid. Applying Random.")
            self.initialize_random(grid); return

        grid.clear_grid()
        spacing = 3 # Place node every 3 cells
        rows, cols = grid.dimensions
        placed_indices: Set[int] = set()
        placed_coords: Set[Tuple[int,int]] = set()

        # Place nodes
        for r in range(0, rows, spacing):
            for c in range(0, cols, spacing):
                coord = (r, c)
                if grid.is_valid_coord(coord):
                    idx = _ravel_multi_index(np.array(coord), grid.dimensions)
                    grid.grid_array[r, c] = 1.0
                    placed_indices.add(idx)
                    placed_coords.add(coord)
        logger.debug(f"{log_prefix}Placed {len(placed_indices)} lattice nodes.")

        # Add Moore edges between lattice nodes
        edges_added_count = 0
        if grid.rule and grid.rule.get_param('edge_initialization', 'NONE') != 'NONE':
            logger.debug(f"{log_prefix}Adding Moore edges...")
            moore_offsets = [(-spacing, -spacing), (-spacing, 0), (-spacing, spacing),
                             (0, -spacing),                 (0, spacing),
                             (spacing, -spacing), (spacing, 0), (spacing, spacing)]
            for r_node, c_node in placed_coords:
                idx1 = _ravel_multi_index(np.array((r_node, c_node)), grid.dimensions)
                for dr, dc in moore_offsets:
                    neighbor_coord = (r_node + dr, c_node + dc)
                    if neighbor_coord in placed_coords: # Check if neighbor is also a lattice node
                        idx2 = _ravel_multi_index(np.array(neighbor_coord), grid.dimensions)
                        grid.add_edge(idx1, idx2, edge_state=1.0)
                        edges_added_count += 1
            logger.debug(f"{log_prefix}Added {edges_added_count // 2} Moore edges.")
        elif not (grid.rule and grid.rule.get_param('edge_initialization', 'NONE') != 'NONE'):
             logger.debug(f"{log_prefix}Skipping edge addition as rule does not use edges.")

        grid.update_active_nodes()
        # REMOVED: self.parent_gui._calculate_initial_previous_arrays()

    def initialize_radial_spokes(self, grid: Grid):
        """Initializes grid with a center node and radiating spokes with edges."""
        log_prefix = "ICM.initialize_radial_spokes (R45 Type Hint Fix): " # Updated round
        logger.debug(f"{log_prefix}Applying 'Radial Spokes'.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}Requires 2D grid. Applying Random.")
            self.initialize_random(grid); return

        grid.clear_grid()
        rows, cols = grid.dimensions
        center_r, center_c = rows // 2, cols // 2
        center_coord = (center_r, center_c)
        num_spokes = 8
        spoke_length = min(rows, cols) // 3
        spoke_length = max(1, spoke_length)
        placed_indices: Set[int] = set()

        # Place center node
        center_idx = -1 # Initialize to invalid index
        if grid.is_valid_coord(center_coord):
            center_idx = _ravel_multi_index(np.array(center_coord), grid.dimensions)
            grid.grid_array[center_coord] = 1.0
            placed_indices.add(center_idx)
            logger.debug(f"{log_prefix}Placed center node at {center_coord}.")
        else:
            logger.error(f"{log_prefix}Center coordinate {center_coord} invalid. Cannot place spokes."); return

        # Place spokes
        spoke_node_indices: List[List[int]] = [] # Store indices for each spoke
        for i in range(num_spokes):
            angle = 2 * np.pi * i / num_spokes
            end_r_rel = int(round(spoke_length * np.sin(angle)))
            end_c_rel = int(round(spoke_length * np.cos(angle)))
            # Create Line object from our shapes module
            line_shape: Line = Line(start=center_coord, end=(center_r + end_r_rel, center_c + end_c_rel)) # Explicit Type Hint

            spoke_indices_this: List[int] = []
            # Call the method on the correctly typed object
            line_coords = line_shape.get_relative_coordinates()
            logger.debug(f"{log_prefix}Spoke {i}: Angle={np.degrees(angle):.1f}, EndRel=({end_r_rel},{end_c_rel}), NumCoords={len(line_coords)}")
            for coord in line_coords:
                if isinstance(coord, tuple):
                    if grid.is_valid_coord(coord): # Pass the tuple coord
                        idx = _ravel_multi_index(np.array(coord), grid.dimensions)
                        if idx != center_idx: # Don't overwrite center
                            grid.grid_array[coord] = 1.0
                            placed_indices.add(idx)
                            spoke_indices_this.append(idx)
                else:
                    logger.error(f"{log_prefix}Coordinate '{coord}' from line algorithm is not a tuple (Type: {type(coord)}). Skipping.")

            if spoke_indices_this:
                spoke_node_indices.append(spoke_indices_this)
                logger.debug(f"  Added {len(spoke_indices_this)} nodes for spoke {i}.")
            else:
                 logger.debug(f"  No valid nodes added for spoke {i}.")

        logger.debug(f"{log_prefix}Placed {len(placed_indices)} total nodes for spokes.")

        # Add edges along spokes and to center
        edges_added_count = 0
        if grid.rule and grid.rule.get_param('edge_initialization', 'NONE') != 'NONE':
            logger.debug(f"{log_prefix}Adding edges along spokes...")
            for spoke in spoke_node_indices:
                # Connect center to first node of spoke
                if spoke and center_idx != -1: # Ensure center_idx is valid
                    grid.add_edge(center_idx, spoke[0], edge_state=1.0)
                    edges_added_count += 1
                # Connect consecutive nodes along spoke
                for i in range(len(spoke) - 1):
                    grid.add_edge(spoke[i], spoke[i+1], edge_state=1.0)
                    edges_added_count += 1
            logger.debug(f"{log_prefix}Added {edges_added_count} spoke edges.")
        elif not (grid.rule and grid.rule.get_param('edge_initialization', 'NONE') != 'NONE'):
             logger.debug(f"{log_prefix}Skipping edge addition as rule does not use edges.")

        grid.update_active_nodes()
        # Previous array calculation is handled by the caller (SimulationGUI)

    def initialize_checkerboard_with_edges(self, grid: Grid):
        """Initializes grid with a checkerboard pattern and Moore edges."""
        log_prefix = "ICM.initialize_checkerboard_with_edges: "
        logger.debug(f"{log_prefix}Applying 'Checkerboard with Edges'.")

        if grid.dimension_type != Dimension.TWO_D:
            logger.warning(f"{log_prefix}Requires 2D grid. Applying Random.")
            self.initialize_random(grid); return

        grid.clear_grid()
        rows, cols = grid.dimensions
        placed_indices: Set[int] = set()

        # Place nodes
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0: # Place on even sum of coords
                    coord = (r, c)
                    if grid.is_valid_coord(coord):
                        idx = _ravel_multi_index(np.array(coord), grid.dimensions)
                        grid.grid_array[r, c] = 1.0
                        placed_indices.add(idx)
        logger.debug(f"{log_prefix}Placed {len(placed_indices)} checkerboard nodes.")

        grid.update_active_nodes() # Update active set before adding edges

        # Add Moore edges between placed nodes
        if grid.rule and grid.rule.get_param('edge_initialization', 'NONE') != 'NONE' and placed_indices and grid.shape_placer:
            logger.debug(f"{log_prefix}Adding 'full' (Moore) edges between checkerboard nodes.")
            grid.shape_placer.add_default_edges(placed_indices, "full")
        elif not (grid.rule and grid.rule.get_param('edge_initialization', 'NONE') != 'NONE'):
             logger.debug(f"{log_prefix}Skipping edge addition as rule does not use edges.")

        # REMOVED: self.parent_gui._calculate_initial_previous_arrays()

    def _register_defaults(self):
        """Register all default initial condition methods."""
        self.register("Random", InitialConditionManager.initialize_random)
        self.register("Empty", InitialConditionManager.initialize_empty)
        # Basic Shapes (Edges handled by caller based on rule)
        self.register("2D - Square", InitialConditionManager.initialize_2d_square)
        self.register("2D - Circle", InitialConditionManager.initialize_2d_circle)
        # self.register("3D - Cube", InitialConditionManager.initialize_3d_cube) # Re-enable when Cube is fixed
        self.register("3D - Sphere", InitialConditionManager.initialize_3d_sphere)
        # Patterns (Edges handled by caller based on rule)
        self.register("Glider Pattern", InitialConditionManager.initialize_glider_pattern)
        self.register("Gosper Glider Gun Pattern", InitialConditionManager.initialize_gosper_glider_gun_pattern)
        self.register("2D - Square Tessellation", InitialConditionManager.initialize_square_tessellation)
        self.register("2D - Triangle Tessellation", InitialConditionManager.initialize_triangle_tessellation) # Original
        # Explicit "with Edges" versions
        self.register("2D - Square with Edges", InitialConditionManager.initialize_2d_square_with_edges)
        self.register("2D - Circle with Edges", InitialConditionManager.initialize_2d_circle_with_edges)
        self.register("3D - Sphere with Edges", InitialConditionManager.initialize_3d_sphere_with_edges)
        self.register("Grid Lattice (Von Neumann Edges)", InitialConditionManager.initialize_grid_lattice_von_neumann)
        self.register("Grid Lattice (Moore Edges)", InitialConditionManager.initialize_grid_lattice_moore)
        self.register("Radial Spokes", InitialConditionManager.initialize_radial_spokes)
        self.register("Symmetry Regions", InitialConditionManager.initialize_symmetry_regions)
        self.register("Symmetric Core, Random Shell", InitialConditionManager.initialize_symmetric_core_random_shell)
 
# =========== END of initial_conditions.py ===========