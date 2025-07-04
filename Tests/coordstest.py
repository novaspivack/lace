import sys
import numpy as np
from typing import Tuple
from enum import Enum, auto
import logging
import multiprocessing as mp
from datetime import datetime
import os


class GlobalSettings:

    class Defaults:
        DEFAULT_RULE = "Test Rule"  # Initial rule displayed on startup.
        NUM_SIMILAR_RULES = 3 # what does this do and are we using it?

        # Default rule author info (used for new rules)
        DEFAULT_AUTHOR = "Nova Spivack"
        DEFAULT_URL = "https://novaspivack.com/network_automata"
        DEFAULT_EMAIL = "novaspivackrelay @ gmail . com" # (remove spaces)

    ENABLE_BLITTING: bool = True  # Use matplotlib's blitting for faster updates.
    ENABLE_TIEBREAKERS: bool = True  # Use tiebreaker rules when multiple state transitions are possible.
    USE_PARALLEL_PROCESSING: bool = False  # Use multiprocessing for faster updates.

    class Simulation:


        NUM_STEPS = 100  # Number of simulation steps to run in non-continuous mode.

        INITIAL_NODE_DENSITY = 0.5 # Initial proportion of active nodes.
        INITIAL_EDGE_DENSITY = 0.4 # Initial proportion of edges, relative to maximum possible.

        # Milliseconds between simulation steps - lower values make the simulation run faster
        STEP_DELAY = 10 # Initial step delay value
        MIN_STEP_DELAY = 10 # Fastest the simulation can run
        MAX_STEP_DELAY = 1000 # Slowest the simulation can run
        TARGET_FPS = 30 # Target frames per second for dynamic step delay adjustment.
        STEP_DELAY_ADJUSTMENT_RATE = 0.1  # How quickly to adjust step delay (10% per frame).

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
        MIN_GENERATIONS = 100

        # Number of recent generations to consider when checking for stability
        # Larger window means more stable detection but slower response
        STABILITY_WINDOW = 20

        # Maximum allowed variation in activity over stability window
        # Lower values require more stable patterns before stopping
        STABILITY_THRESHOLD = 0.01

        LOG_LEVEL = "DEBUG"

        # Size of chunks for parallel processing - larger chunks mean fewer
        # communication overhead but less even distribution of work
        # CHUNK_SIZE = 1000 # MOVED UP

        ###### $$ PERFORMANNCE SETTINGS ######

        # Number of parallel processes to use - leaves one CPU core free
        # for system and GUI operations
        NUM_PROCESSES = max(1, mp.cpu_count() - 1)

        SCROLL_SPEED = 0.2 # controls the scroll speed in mouse scrolling withing scrollable frames; lower is slower

        # Size of LRU cache for computational results
        # Larger cache means more memory use but faster repeat calculations
        CACHE_SIZE = 2048

    class Visualization:
        EDGE_SCALE = 15.0  # Base scaling factor for edge lengths.
        NODE_SIZE = 1.0  # Base size of nodes.  This is scaled by zoom.
        NODE_SHAPE = 'sphere'  # Default shape for nodes ('o', 's', '^', etc.)
        NODE_SPACING = 5.0  # Default spacing between nodes (0.0 = no extra spacing).
        MAX_NODE_SPACING = 20.0  # Maximum allowed spacing between nodes.
        MIN_NODE_SPACING = 0.0  # Minimum allowed spacing (can be 0, but not negative).
        NODE_OPACITY = 1.0  # Node opacity (1.0 = fully opaque, 0.0 = fully transparent).
        RESOLUTION = 20  # Resolution for drawing circles/spheres.
        EDGE_WIDTH = 1.0  # Base width of edges.
        NODE_OUTLINE_WIDTH = 1.0  # Base width of node outlines.
        EDGE_OPACITY = 1.0  # Edge opacity (1.0 = fully opaque, 0.0 = fully transparent).
        USE_CUSTOM_EDGE_COLORS = True  # Use custom edge colors (defined in rules).
        NODE_VISIBILITY_THRESHOLD = 0.0  # Minimum node state to be considered visible.
        SHOW_INACTIVE_NODES = False  # Show nodes with state <= NODE_VISIBILITY_THRESHOLD.
        SHOW_INACTIVE_EDGES = False  # Show edges connected to inactive nodes.
        HIGHLIGHT_DURATION = 1  # Duration (in seconds) for highlighting changes.
        ROTATION_SPEED = 0.5  # Speed of 3D rotation.

        ZOOM_FACTOR = 2.0  # Base zoom factor (unused, zoom is handled dynamically).
        ANIMATION_INTERVAL = 50  # Interval between animation frames (ms).
        ANIMATION_BLIT = True  # Use blitting for faster animation.
        ANIMATION_CACHE = True  # Cache animation frames (not currently used).
        ANIMATION_REPEAT = False  # Repeat animation (not currently used).

        WINDOW_SIZE = (1500, 1200)  # Main application window size.

        RULE_EDITOR_WINDOW_SIZE = (750, 1300)  # Rule Editor window size.
        RULE_EDITOR_HEADING_FONT_SIZE = 16 # Font size for column headings
        RULE_EDITOR_FIELD_FONT_SIZE = 15 # Font size for field names
        RULE_EDITOR_FONT_SIZE = 13  # Font size for parameter descriptions.

        FIGURE_SIZE = (12, 10)  # Size of the matplotlib figure.
        CONTROL_HEIGHT = 80  # Height of the control panel (not currently used).
        WINDOW_POSITION = None  # Initial window position (None = centered).
        WINDOW_PADDING = 20  # Padding around the main window content.
        CONTROL_PADDING = 10  # Padding within the control panel.
        MOUSE_ROTATION_SENSITIVITY = 0.5  # Sensitivity of mouse rotation (3D).
        MOUSE_ZOOM_SENSITIVITY = 0.1  # Sensitivity of mouse zoom.

        USE_HARDWARE_ACCELERATION = True  # Use hardware acceleration if available.
        MAX_VISIBLE_NODES = 10000  # Limit the number of visible nodes for performance.
        DYNAMIC_RESOLUTION = True  # Adjust resolution based on performance (not implemented).

        NODE_COLORMAP = 'viridis'  # Default colormap for node states.

        # --- ZOOM-RELATED SETTINGS ---

        # Controls the exponent for node diameter increase on zoom-in.
        # Node diameter is scaled by: base_size * (zoom_factor ** NODE_DIAMETER_ZOOM_IN_AMP).
        #   - zoom_factor < 1.0 when zooming in.
        #   - Values > 1 will cause exponential growth with zoom.  3 is cubic growth.
        #   - Value of 1 will cause linear growth with 1/zoom_factor (diameter doubles when zoom_factor halves).
        #   - Value of 0 will cause no change in size on zoom.
        #   - Negative values are not recommended and will lead to undefined behavior.
        NODE_DIAMETER_ZOOM_IN_AMP = 2.0

        # Controls the exponent for node diameter decrease on zoom-out.
        # Node diameter is scaled by: base_size * (zoom_factor ** NODE_DIAMETER_ZOOM_OUT_AMP).
        #   - zoom_factor > 1.0 when zooming out.
        #   - Values > 1 will cause exponential decrease with zoom. 3 is cubic decrease.
        #   - Value of 1 will cause linear decrease with zoom (diameter halves when zoom_factor doubles).
        #   - Value of 0 will cause no change in size on zoom.
        #   - Negative values are not recommended and will lead to undefined behavior.
        NODE_DIAMETER_ZOOM_OUT_AMP = 0.7

        # Limits the maximum increase in node outline thickness during zoom-in.
        # The outline width will be clamped to: min(base_outline_width * zoom_factor,
        #                                         NODE_OUTLINE_ZOOM_IN_LIMIT * base_outline_width)
        #   - Values > 0 will limit the outline width.  2.0 means the outline can be at most
        #     twice the base outline width.
        #   - Value of 0 will disable the limit (not recommended, as outlines can become too thick).
        NODE_OUTLINE_ZOOM_IN_LIMIT = 3.0

        # Limits the *minimum* node outline thickness during zoom-out.
        # The outline width will be clamped to: max(base_outline_width * zoom_factor,
        #                                         base_outline_width * NODE_OUTLINE_ZOOM_OUT_LIMIT)
        #   - Values > 0 will limit the minimum thickness.  0.5 means the outline can be at most
        #     half the base outline width.
        #   - Value of 0 will disable the limit (outlines can become invisible).
        NODE_OUTLINE_ZOOM_OUT_LIMIT = 0.4  # 0 means no limit

        # Limits the maximum increase in edge thickness during zoom-in.
        # The edge width will be clamped to: min(base_edge_width * zoom_factor,
        #                                      EDGE_THICKNESS_ZOOM_IN_LIMIT * base_edge_width)
        #   - Values > 0 will limit the edge width.  3.0 means the edge can be at most
        #     three times the base edge width.
        #   - Value of 0 will disable the limit (not recommended, as edges can become too thick).
        EDGE_THICKNESS_ZOOM_IN_LIMIT = 5.0

        # Limits the *minimum* edge thickness during zoom-out.
        # The edge width will be clamped to: max(base_edge_width * zoom_factor,
        #                                      base_edge_width * EDGE_THICKNESS_ZOOM_OUT_LIMIT)
        #   - Values > 0 will limit the minimum thickness.  0.5 means the edge can be at most
        #     half the base edge width.
        #   - Value of 0 will disable the limit (edges can become invisible).
        EDGE_THICKNESS_ZOOM_OUT_LIMIT = 0.5  # 0 means no limit

        # Controls how much the node spacing *increases* with each zoom-in step.
        # The spacing is multiplied by (1/zoom_factor) ** NODE_SPACING_ZOOM_IN_AMP.
        # We multiply by the inverse of the zoom factor because we want the spacing to *increase* as we zoom *in* 
        # (and zoom_factor is < 1 when zooming in).  Values > 1 cause exponential increase in spacing.
        # A value of 1 causes linear increase. A value of 0 causes no change. Negative values are not recommended.
        NODE_SPACING_ZOOM_IN_AMP = 2.0
        
        # Controls how much the node spacing *decreases* with each zoom-out step.
        # The spacing is divided by zoom_factor ** NODE_SPACING_ZOOM_OUT_AMP.
        # We divide because we want the spacing to *decrease* as we zoom *out* 
        # (and zoom_factor is > 1 when zooming out). Values > 1 cause exponential decrease in spacing.
        # A value of 1 causes linear decrease. A value of 0 causes no change. Negative values are not recommended.
        NODE_SPACING_ZOOM_OUT_AMP = 2.7

        # Limits the maximum increase in node diameter during zoom-in.
        # The diameter will be clamped to: min(calculated_diameter, base_diameter * NODE_DIAMETER_ZOOM_IN_LIMIT)
        # Set to 0 for no limit.
        NODE_DIAMETER_ZOOM_IN_LIMIT = 0  # 0 means no limit

        # Limits the maximum decrease in node diameter during zoom-out.
        # The diameter will be clamped to: max(calculated_diameter, base_diameter * NODE_DIAMETER_ZOOM_OUT_LIMIT)
        # Set to 0 for no limit.  Values between 0 and 1 will limit how small the nodes can get.
        NODE_DIAMETER_ZOOM_OUT_LIMIT = 0.2  # 0 means no limit

        # Setting this to -1.0 disables the maximum node spacing limit.
        DISABLE_MAX_NODE_SPACING = -1.0

        @classmethod
        def set_node_spacing(cls, spacing: float):
            """Set node spacing within allowed range"""
            cls.NODE_SPACING = max(0.0, min(spacing, cls.MAX_NODE_SPACING))

    class Colors:
        NODE_INACTIVE: str = '#ffffff'  # White 
        NODE_ACTIVE: str = '#f77b4f'  # Fixed: Added missing 'f' at the end
        NODE_HIGHLIGHT: str = '#8B0000'  # Dark Red
        NODE_EDGE_NEW: str = '#ff0000'  # Red for new nodes
        NODE_EDGE_OLD: str = '#0000ff'  # Blue for existing nodes
        BACKGROUND: str = '#ffffff'  # White background for main area
        CONTROL_BACKGROUND: str = '#f0f0f0'  # Light grey for control panel
        CONTROL_BUTTON: str = '#e0e0e0'  # Slightly darker grey for buttons
        
    class Cache:
        ENABLE_CACHING = True
        METRIC_CACHE_SIZE = 1000
        NEIGHBOR_CACHE_SIZE = 1000
        STATE_CACHE_SIZE = 1000
        CACHE_INVALIDATION_THRESHOLD = 0.1  # 10% change triggers invalidation
        FULL_CACHE_THRESHOLD = 0.9  # Hit rate threshold for switching to full caching
        PARTIAL_CACHE_THRESHOLD = 0.7  # Hit rate threshold for switching back to partial
        FULL_CACHE_MEMORY_THRESHOLD = 0.5  # Memory usage threshold for full caching
        PARTIAL_CACHE_MEMORY_THRESHOLD = 0.8  # Memory usage threshold for partial caching
        CACHE_CHECK_INTERVAL = 100 # How many steps to check
        
    class SpatialHash:
        MIN_CELLS_PER_DIMENSION = 10  # Minimum number of cells per dimension.
        MAX_CELLS_PER_DIMENSION = 1000  # Maximum number of cells per dimension.
        TARGET_CELLS_PER_NODE = 8  # Target number of cells per node (initial value).
        MEMORY_THRESHOLD = 0.8  # Use up to 80% of available memory.
        ADAPTATION_INTERVAL = 100  # Steps between adaptations.
        MIN_CELL_SIZE = 1.0  # Minimum cell size.
        MAX_CELL_SIZE = 100.0  # Maximum cell size.
        PERFORMANCE_WINDOW = 50  # Steps to average performance over.
        TARGET_QUERY_TIME = 0.001  # Target time for neighbor queries (seconds).
        
        # Auto-tuning parameters
        ENABLE_AUTO_TUNING = True  # Enable/disable automatic grid adaptation.
        GROWTH_FACTOR = 1.2  # Factor to increase cell size by.
        SHRINK_FACTOR = 0.8  # Factor to decrease cell size by.
        MIN_ADAPTATION_THRESHOLD = 0.1  # Minimum performance change to trigger adaptation.



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
        # CRITICAL FIX: Use LACE_{timestamp}.log as filename
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'LACE_{timestamp}.log')
        )
        file_handler.setFormatter(formatter)
        
        # Console handler for INFO and above
        console_handler = logging.StreamHandler(sys.stdout)
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


        logger = logging.getLogger(__name__)
        logger.debug("Logging initialized with single log file: " + 
                    os.path.join(log_dir, f'LACE_{timestamp}.log')) # Use the correct filename here too
        return logger

    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise

APP_DIR = "LACE"
SUBDIRS = {
    'logs': 'logs',
    'saves': 'saves',
    'rules_backups': 'rules_backups', 
    'data': 'data',
    'cache': 'cache',
    'profiles': 'profiles',
    'config': 'config'  
}

def setup_directories() -> Tuple[dict, str]:
    try:
        base_path = os.path.join(os.getcwd(), APP_DIR)
        
        # Check if Resources directory exists
        resources_path = os.path.join(base_path, "Resources")
        if not os.path.exists(resources_path):
            # Create Resources directory
            os.makedirs(resources_path, exist_ok=True)
            
        # Define subdirectories with new structure
        SUBDIRS = {
            'logs': 'logs',
            'saves': 'saves',
            'data': 'data',
            'cache': 'cache',
            'profiles': 'profiles',
            'config': 'config',  # Main config directory
            'config_colors': 'config/colors',  # New subdirectory for color schemes
            'config_rules': 'config/rules',  # New subdirectory for rules
            'config_rules_backups': 'config/rules_backups',  # New subdirectory for rule backups
        }
            
        # Create subdirectories inside Resources
        paths = {}
        for key, subdir in SUBDIRS.items():
            path = os.path.join(resources_path, subdir)
            os.makedirs(path, exist_ok=True)
            paths[key] = path
                
        return paths, base_path
    except Exception as e:
        print(f"Fatal error in directory setup: {str(e)}")
        raise SystemExit(1)
       
# Initialize directory structure and logger
APP_PATHS, BASE_PATH = setup_directories() # This needs to happen BEFORE ColorManager
logger = setup_logging(APP_PATHS['logs'])




# Assuming Dimension and NeighborhoodType are defined as Enums as in your original code
class Dimension(Enum):
    """Enum for dimension types"""
    TWO_D = auto()
    THREE_D = auto()

class NeighborhoodType(Enum):
    """Defines types of neighborhood relationships"""
    VON_NEUMANN = auto()  # 4 neighbors in 2D (N,S,E,W), 6 neighbors in 3D (N,S,E,W,Up,Down)
    MOORE = auto()        # 8 neighbors in 2D (N,S,E,W,NE,NW,SE,SW), 26 neighbors in 3D (all adjacent cells)
    HEX = auto()         # 6 neighbors in 2D (hexagonal grid), not valid for 3D
    HEX_PRISM = auto()   # 12 neighbors in 3D (6 in hexagonal plane + 3 above + 3 below), only valid for 3D

# Assuming CoordinateSystem class is defined as in your original code
# (I'm including a simplified version here for completeness)
class CoordinateSystem:
    """Unified coordinate system handler for grid visualization"""

    def __init__(self, grid_dimensions, edge_scale=1.0, node_spacing=0.0, dimension_type=Dimension.TWO_D):
        """Initialize the coordinate system with grid parameters"""
        self.grid_dimensions = grid_dimensions
        self.edge_scale = edge_scale
        self.node_spacing = node_spacing
        self.dimension_type = dimension_type
        self.scale_factor = self.edge_scale * (1.0 + self.node_spacing)
        self.display_bounds = self._calculate_default_display_bounds()
        logger.debug(f"CoordinateSystem initialized with dimensions: {grid_dimensions}, "
                    f"scale: {edge_scale}, spacing: {node_spacing}")

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
        # logger.debug(f"display_to_grid called with display_coords: {display_coords}, scale_factor: {self.scale_factor}") # Added logging
        if self.dimension_type == Dimension.THREE_D:
            assert len(display_coords) == 3, "3D display requires 3 coordinates"
            x, y, z = display_coords
            # For 3D, we use (y, x, z) for (i, j, k) in grid
            return (round(y / self.scale_factor), round(x / self.scale_factor), round(z / self.scale_factor))
        else:  # TWO_D
            assert len(display_coords) == 2, "2D display requires 2 coordinates"
            x, y = display_coords
            # For 2D, we use (y, x) for (i, j) in grid
            return (round(y / self.scale_factor), round(x / self.scale_factor))

    def update_parameters(self, edge_scale=None, node_spacing=None, dimension_type=None, grid_dimensions=None):
        """Update coordinate system parameters."""
        if edge_scale is not None:
            self.edge_scale = edge_scale
        if node_spacing is not None:
            self.node_spacing = node_spacing
        if dimension_type is not None:
            self.dimension_type = dimension_type
        if grid_dimensions is not None:
            self.grid_dimensions = grid_dimensions

        # Recalculate scale factor
        self.scale_factor = self.edge_scale * (1.0 + self.node_spacing)

        # Recalculate default display bounds (not actively used, but kept for consistency)
        self.display_bounds = self._calculate_default_display_bounds()

        logger.debug(f"CoordinateSystem parameters updated: scale_factor={self.scale_factor}")

def test_coordinate_transformations():
    """Test function for CoordinateSystem transformations."""

    test_cases = [
        {
            'grid_dimensions': (10, 10),
            'initial_scale_factor': 1.0,
            'node_spacing': 0.0,
            'dimension_type': Dimension.TWO_D,
            'grid_coords': [(0, 0), (1, 1), (9, 9), (5, 3), (0, 9), (9,0)],
            'zoom_factors': [1.0, 0.5, 2.0, 0.1, 5.0]
        },
        {
            'grid_dimensions': (5, 5, 5),
            'initial_scale_factor': 1.0,
            'node_spacing': 0.0,
            'dimension_type': Dimension.THREE_D,
            'grid_coords': [(0, 0, 0), (1, 1, 1), (4, 4, 4), (2, 3, 1)],
            'zoom_factors': [1.0, 0.5, 2.0, 0.1, 5.0]
        },
        { # Added test case with node spacing
            'grid_dimensions': (10, 10),
            'initial_scale_factor': 1.0,
            'node_spacing': 2.0,
            'dimension_type': Dimension.TWO_D,
            'grid_coords': [(0, 0), (1, 1), (9, 9), (5, 3)],
            'zoom_factors': [1.0, 0.5, 2.0]
        },
        { # Added test case with negative and out of bounds
            'grid_dimensions': (5, 5),
            'initial_scale_factor': 1.0,
            'node_spacing': 0.0,
            'dimension_type': Dimension.TWO_D,
            'grid_coords': [(-1, -1), (5, 5), (10, 10)],
            'zoom_factors': [1.0]
        }
    ]

    for case in test_cases:
        print(f"Testing case: {case}")
        coord_system = CoordinateSystem(case['grid_dimensions'], case['initial_scale_factor'], case['node_spacing'], case['dimension_type'])

        for grid_coord in case['grid_coords']:
            print(f"  Testing grid coordinate: {grid_coord}")
            display_coord = coord_system.grid_to_display(grid_coord)
            print(f"    Initial display coord: {display_coord}")

            for zoom_factor in case['zoom_factors']:
                print(f"    Zoom factor: {zoom_factor}")
                # Update scale_factor based on zoom (simplified for testing)
                new_scale_factor = coord_system.edge_scale * (1.0 + coord_system.node_spacing) / zoom_factor
                coord_system.scale_factor = new_scale_factor
                
                # Transform with new scale_factor
                zoomed_display_coord = coord_system.grid_to_display(grid_coord)
                print(f"      Zoomed display coord: {zoomed_display_coord}")
                
                returned_grid_coord = coord_system.display_to_grid(zoomed_display_coord)
                print(f"      Returned grid coord: {returned_grid_coord}")

                # Assert that the returned grid coordinates match the original
                assert returned_grid_coord == grid_coord, f"Failed for {grid_coord} with zoom {zoom_factor}. Expected {grid_coord}, got {returned_grid_coord}"
                print(f"      Assertion passed for {grid_coord} with zoom {zoom_factor}")

# To run the tests, simply call the function:
if __name__ == "__main__":
    test_coordinate_transformations()