# =========== START of settings.py ===========
from __future__ import annotations
import multiprocessing as mp
import setproctitle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from typing import Tuple, Optional, Union, TypeVar, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
plt.ioff()
from numba import cuda
import numpy as np
import warnings
import cProfile
import pstats

from .enums import Dimension, NeighborhoodType
from .logging_config import logger

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
#                 GLOBAL SETTINGS              #
################################################


class GlobalSettings:

    class Defaults:
        DEFAULT_RULE = "Realm of Lace with Metrics_Betweenness_Amazing_Dragons_Wow"  # Initial rule displayed on startup.
        NUM_SIMILAR_RULES = 3 # what does this do and are we using it? # Kept as is

        # Default rule author info (used for new rules)
        DEFAULT_AUTHOR = "Nova Spivack"
        DEFAULT_URL = "https://novaspivack.com/network_automata"
        DEFAULT_EMAIL = "novaspivackrelay @ gmail . com" # (remove spaces)

    ENABLE_BLITTING: bool = True
    ENABLE_TIEBREAKERS: bool = True
    USE_PARALLEL_PROCESSING: bool = True

    class Simulation:

        ADMIN_MODE = bool = True

        DIMENSION_TYPE: Dimension = Dimension.TWO_D
        NEIGHBORHOOD_TYPE = NeighborhoodType.MOORE

        GRID_SIZE_2D = 50 # Default, will be overridden by preset or GUI
        GRID_SIZE_3D = 50 # Default, will be overridden by preset or GUI

        @staticmethod
        def set_grid_size(size: int, dimension_type: Dimension):
            """Set the grid size based on dimension type"""
            if dimension_type == Dimension.TWO_D:
                GlobalSettings.Simulation.GRID_SIZE_2D = size
            elif dimension_type == Dimension.THREE_D:
                GlobalSettings.Simulation.GRID_SIZE_3D = size
            else:
                raise ValueError(f"Invalid dimension type: {dimension_type}")

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

        NUM_STEPS = 100 # Max steps if not running continuously
        INITIAL_NODE_DENSITY = 0.4
        INITIAL_EDGE_DENSITY = 0.5

        # --- Target FPS for Rendering Loop ---
        TARGET_FPS: int = 30 # Target rendering frame rate

        MIN_GENERATIONS = 10
        STABILITY_WINDOW = 20
        STABILITY_THRESHOLD = 0.01

        # --- Performance Settings ---
        NUM_PROCESSES = max(1, mp.cpu_count() - 1) # Keep dynamic based on CPU
        SCROLL_SPEED = 0.2
        USE_GPU = cuda.is_available()
        CACHE_SIZE = 2048

        # --- ADDED: Chunk Size Setting (Round 11) ---
        # Stores the *currently selected* chunk size. 0 means 'Auto'. When set to 0 chunk size will auto-scale. For reference, a chunk size of 512 seems to perform optimally for a 50,50 grid.
        CHUNK_SIZE: int = 0
        # --- END ADDED ---

        # --- ADDED: Queue Size Settings (Round 5) ---
        # Max number of steps computation thread can get ahead of preparation thread
        MAX_QUEUE_AHEAD: int = 30
        # Max number of fully prepared frames the render queue can hold
        RENDER_QUEUE_SIZE: int = 60
        # --- END ADDED ---

    class Visualization:
        DEFAULT_NODE_COLORMAP: str = 'rainbow' # Default colormap for node state visualization
        DEFAULT_EDGE_COLORMAP: str = 'prism' # Default colormap for edge state visualization
        EDGE_SCALE = 15.0
        NODE_SIZE = 1.0
        BASE_NODE_SIZE_REFERENCE: int = 30
        NODE_SHAPE = 'sphere'
        NODE_SPACING = 5.0
        MAX_NODE_SPACING = 20.0
        MIN_NODE_SPACING = 0.0
        DYNAMIC_MIN_NODE_SPACING_FACTOR: float = 1.05 # Factor for dynamic min spacing (NODE_SIZE * factor - 1)
        MAX_NODE_SIZE: float = 5.0 # Maximum node size multiplie
        NODE_OPACITY = 1.0
        RESOLUTION = 20
        EDGE_WIDTH = 1.0
        NODE_OUTLINE_WIDTH = 1.0
        EDGE_OPACITY = 1.0
        USE_CUSTOM_EDGE_COLORS = True
        NODE_VISIBILITY_THRESHOLD = 0.0
        SHOW_INACTIVE_NODES = False
        SHOW_INACTIVE_EDGES = False
        HIGHLIGHT_DURATION = 1
        ROTATION_SPEED = 0.5
        ZOOM_FACTOR = 2.0
        ANIMATION_INTERVAL = 50
        ANIMATION_BLIT = True
        ANIMATION_CACHE = True
        ANIMATION_REPEAT = False
        PAN_DECELERATION = 0.80
        PAN_SENSITIVITY = 0.15
        WINDOW_SIZE = (1500, 1200)
        RULE_EDITOR_WINDOW_SIZE = (750, 1300)
        RULE_EDITOR_HEADING_FONT_SIZE = 16
        RULE_EDITOR_FIELD_FONT_SIZE = 15
        RULE_EDITOR_FONT_SIZE = 13
        FIGURE_SIZE = (12, 10)
        CONTROL_HEIGHT = 80
        WINDOW_POSITION = None
        WINDOW_PADDING = 20
        CONTROL_PADDING = 10
        MOUSE_ROTATION_SENSITIVITY = 0.5
        MOUSE_ZOOM_SENSITIVITY = 0.1
        USE_HARDWARE_ACCELERATION = True
        MAX_VISIBLE_NODES = 10000
        DYNAMIC_RESOLUTION = True
        NODE_COLORMAP = 'viridis'
        NODE_DIAMETER_ZOOM_IN_AMP = 2.0
        NODE_DIAMETER_ZOOM_OUT_AMP = 0.8
        NODE_OUTLINE_ZOOM_IN_LIMIT = 4.0
        NODE_OUTLINE_ZOOM_OUT_LIMIT = 0.4
        EDGE_THICKNESS_ZOOM_IN_LIMIT = 7.0
        EDGE_THICKNESS_ZOOM_OUT_LIMIT = 0.5
        NODE_SPACING_ZOOM_IN_AMP = 2.0
        NODE_SPACING_ZOOM_OUT_AMP = 0.65
        NODE_DIAMETER_ZOOM_IN_LIMIT = 0
        NODE_DIAMETER_ZOOM_OUT_LIMIT = 0.025
        DISABLE_MAX_NODE_SPACING = -1.0

        @classmethod
        def set_node_spacing(cls, spacing: float):
            """Set node spacing, applying a dynamic minimum based on NODE_SIZE to prevent overlap."""
            # --- REVERTED: Remove dynamic minimum calculation, keep clamping ---
            logger.debug(f"Setting node spacing: Input={spacing:.3f}, Min={cls.MIN_NODE_SPACING:.3f}, Max={cls.MAX_NODE_SPACING:.3f}")

            # Clamp the requested spacing between the absolute minimum and the absolute maximum
            clamped_spacing = max(cls.MIN_NODE_SPACING, min(spacing, cls.MAX_NODE_SPACING))

            # Store the clamped value
            cls.NODE_SPACING = clamped_spacing
            logger.debug(f"Node spacing setting stored: {cls.NODE_SPACING:.3f}")

        @classmethod
        def calculate_max_node_size_for_spacing(cls, node_spacing: float) -> float:
            """Calculate the maximum allowed node size for a given spacing to prevent overlap."""
            # Invert the dynamic minimum spacing calculation to find max NODE_SIZE
            # EDGE_SCALE * (1 + min_spacing) = EDGE_SCALE * sqrt(NODE_SIZE) * 1.05
            # 1 + min_spacing = sqrt(NODE_SIZE) * 1.05
            # sqrt(NODE_SIZE) = (1 + min_spacing) / 1.05
            # NODE_SIZE = ((1 + min_spacing) / 1.05)**2
            max_node_size = ((1.0 + node_spacing) / GlobalSettings.Visualization.DYNAMIC_MIN_NODE_SPACING_FACTOR)**2
            # Clamp to the absolute maximum
            return max(0.1, min(max_node_size, cls.MAX_NODE_SIZE)) # Ensure a tiny minimum and also clamp to MAX_NODE_SIZE
        
    class Colors:
        NODE_INACTIVE: str = '#ffffff'
        # --- CORRECTED: NODE_ACTIVE color ---
        NODE_ACTIVE: str = '#f77b4f' # Was missing the last 'f'
        # ---
        NODE_HIGHLIGHT: str = '#8B0000'
        NODE_EDGE_NEW: str = '#ff0000'
        NODE_EDGE_OLD: str = '#0000ff'
        BACKGROUND: str = '#ffffff'
        CONTROL_BACKGROUND: str = '#f0f0f0'
        CONTROL_BUTTON: str = '#e0e0e0'

    class Analytics:
        DEFAULT_ENABLED: bool = False # Analytics disabled by default
        DEFAULT_SAVE_REPORT: bool = False # Report saving disabled by default
        DEFAULT_HISTORY_LENGTH: int = 1000 # Max steps of history per metric
        DEFAULT_UPDATE_INTERVAL_MS: int = 1000 # How often AnalyticsWindow UI updates
        DEFAULT_REPORT_DIR: str = "reports" # Default subdirectory name

        # Default calculation frequencies (steps)
        FREQ_BASIC: int = 1
        FREQ_STATE_AVG: int = 5
        FREQ_ENTROPY: int = 10
        FREQ_TOPOLOGY: int = 20 # Example for later
        FREQ_COMPLEXITY: int = 30 # Example for later
        FREQ_FRACTAL: int = 50 # Example for later

        # Default analysis frequencies (steps)
        FREQ_STABILITY: int = 10
        FREQ_TREND: int = 5
        FREQ_PATTERN: int = 50 # Example for later

        # Default group enable states
        ENABLE_BASIC: bool = True
        ENABLE_ENTROPY: bool = True
        ENABLE_STABILITY: bool = True
        ENABLE_TREND: bool = True
        ENABLE_TOPOLOGY: bool = False # Disabled by default
        ENABLE_COMPLEXITY: bool = False # Disabled by default
        ENABLE_FRACTAL: bool = False # Disabled by default
        ENABLE_PATTERN: bool = False # Disabled by default

    class Cache:
        ENABLE_CACHING = True
        METRIC_CACHE_SIZE = 1000
        NEIGHBOR_CACHE_SIZE = 1000 # Note: This is unused now
        STATE_CACHE_SIZE = 1000
        CACHE_INVALIDATION_THRESHOLD = 0.1
        FULL_CACHE_THRESHOLD = 0.9
        PARTIAL_CACHE_THRESHOLD = 0.7
        FULL_CACHE_MEMORY_THRESHOLD = 0.5
        PARTIAL_CACHE_MEMORY_THRESHOLD = 0.8
        CACHE_CHECK_INTERVAL = 100

    class SpatialHash:
        MIN_CELLS_PER_DIMENSION = 10
        MAX_CELLS_PER_DIMENSION = 1000
        TARGET_CELLS_PER_NODE = 8
        MEMORY_THRESHOLD = 0.8
        ADAPTATION_INTERVAL = 10 # Check adaptation more frequently
        MIN_CELL_SIZE = 1.0
        MAX_CELL_SIZE = 100.0
        PERFORMANCE_WINDOW = 10
        TARGET_QUERY_TIME = 0.001

        # Auto-tuning parameters
        # --- MODIFIED: Ensure Auto-tuning is ON for Round 41/42 logic ---
        ENABLE_AUTO_TUNING = False # Default OFF - leave this as False for now - system is tuned for this as is
        # ---
        GROWTH_FACTOR = 1.2
        SHRINK_FACTOR = 0.8
        MIN_ADAPTATION_THRESHOLD = 0.1

# =========== END of settings.py ===========