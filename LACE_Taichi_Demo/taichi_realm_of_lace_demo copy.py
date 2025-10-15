# =========== START of taichi_realm_of_lace_demo.py ===========
# --- Imports ---
import taichi as ti
import math 
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import cv2  # pip install opencv-python
import json
import os
import sys
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Union, TypeVar, Dict, Set # Added Dict, Set



# --- Call ti.init FIRST ---
ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_fraction=0.9)

# --- Logging Configuration (Integrated from logging_config.py) ---
class LogSettings:
    class Logging:
        LOG_LEVEL: str = "DEBUG" # Set desired level (DEBUG, INFO, DETAIL, WARNING, ERROR)

# --- Custom Log Level ---
DETAIL_LEVEL_NUM = 15  # Between DEBUG (10) and INFO (20)
logging.addLevelName(DETAIL_LEVEL_NUM, "DETAIL")

def detail(self, message, *args, **kws):
    """Logs a message with level DETAIL on this logger."""
    if self.isEnabledFor(DETAIL_LEVEL_NUM):
        self._log(DETAIL_LEVEL_NUM, message, args, **kws)

logging.Logger.detail = detail # type: ignore [attr-defined]
# --- End Custom Log Level ---

APP_DIR = "LACE_Taichi_Demo" # Specific name for this app's logs/dirs
SUBDIRS = {
    'logs': 'logs',
    'config': 'config', # Simplified for this script
}

def setup_directories() -> Tuple[dict, str]:
    """Sets up the necessary directories for the application."""
    try:
        # Use current working directory for simplicity here
        base_path = os.getcwd()
        app_base_path = os.path.join(base_path, APP_DIR)
        os.makedirs(app_base_path, exist_ok=True)

        paths = {}
        for key, subdir in SUBDIRS.items():
            path = os.path.join(app_base_path, subdir)
            os.makedirs(path, exist_ok=True)
            paths[key] = path
        return paths, app_base_path
    except Exception as e:
        print(f"Fatal error in directory setup: {str(e)}")
        raise SystemExit(1)

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup basic logging for the Taichi CA Demo."""
    try:
        logger_name = "taichi_ca_logger"
        logger = logging.getLogger(logger_name)
        if logger.handlers:
             return logger

        timestamp_24hr = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_level_str = LogSettings.Logging.LOG_LEVEL.upper()
        file_log_level = getattr(logging, log_level_str, logging.INFO)
        if log_level_str == "DETAIL":
            file_log_level = DETAIL_LEVEL_NUM

        # --- MODIFIED: Set console level to DEBUG ---
        console_log_level = logging.DEBUG
        # --- END MODIFIED ---
        main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

        main_log_filename = f'taichi_ca_demo_{timestamp_24hr}.log'
        main_file_handler = logging.FileHandler(os.path.join(log_dir, main_log_filename))
        main_file_handler.setFormatter(main_formatter)
        main_file_handler.setLevel(file_log_level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(main_formatter)
        console_handler.setLevel(console_log_level) # Use modified level

        logger.setLevel(min(file_log_level, console_log_level))
        logger.addHandler(main_file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False

        file_log_level_name = logging.getLevelName(file_log_level)
        console_log_level_name = logging.getLevelName(console_log_level)
        logger.info(f"Logging initialized. Logger Level: {logging.getLevelName(logger.level)}, File Handler Level: {file_log_level_name}, Console Handler Level: {console_log_level_name}")
        logger.info(f"Log file: {os.path.join(log_dir, main_log_filename)}")
        return logger

    except Exception as e:
        print(f"Error setting up logging: {e}")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("taichi_ca_logger_fallback")
        logger.error(f"Logging setup failed: {e}. Using basicConfig.")
        return logger
    
# Initialize directory structure and logger
APP_PATHS, BASE_PATH = setup_directories()
logger = setup_logging(APP_PATHS['logs'])
# --- End Logging Configuration ---

# --- JSON Loading and Rule Variant Processing ---
# Construct path dynamically relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
JSON_FILE_PATH = os.path.join(parent_dir, 'LACE', 'Resources', 'config', 'rules', 'rules.json')

def load_rules_from_json(filepath):
    """Loads rule definitions from a JSON file."""
    if not os.path.exists(filepath):
        logger.error(f"JSON file not found at {filepath}")
        return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON from {filepath}")
        return data.get("rules", [])
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        return []
    except Exception as e:
        logger.error(f"An error occurred while reading {filepath}: {e}")
        return []

def parse_rule_variants(json_rules):
    """Parses JSON rules into the format needed for the Taichi script."""
    variants = []
    logger.info(f"--- Starting JSON Rule Parsing ({len(json_rules)} rules found) ---")

    # Add standard Game of Life first
    variants.append(
        ("Game of Life (B3/S23)", "game_of_life", {
            "GOL_BIRTH": [3],
            "GOL_SURVIVAL": [2, 3],
            "NODE_COLORMAP": 'gray',
            "NODE_COLOR_NORM_VMIN": 0.0,
            "NODE_COLOR_NORM_VMAX": 1.0,
            "COLOR_BY_DEGREE": False,
            "COLOR_BY_ACTIVE_NEIGHBORS": False,
        })
    )

    for rule_idx, rule in enumerate(json_rules):
        rule_type = rule.get("type")
        rule_name = rule.get("name", f"Unnamed Rule {rule_idx}")
        params = rule.get("params", {})
        logger.debug(f"\nProcessing Rule Index {rule_idx}: Name='{rule_name}', Type='{rule_type}'")

        # Process RealmOfLace variants
        if rule_type == "RealmOfLace":
            logger.detail(f"  Raw RoL Params Dict (first 80 chars): {str(params)[:80]}...") # type: ignore [attr-defined]

            birth_ranges_raw = params.get("birth_neighbor_degree_sum_range", '---MISSING---')
            survival_ranges_raw = params.get("survival_neighbor_degree_sum_range", '---MISSING---')
            death_degrees_raw = params.get("final_death_degree_counts", '---MISSING---')
            death_ranges_raw = params.get("final_death_degree_range", '---MISSING---')

            logger.detail(f"    Raw Birth Sum Ranges: {birth_ranges_raw} (Type: {type(birth_ranges_raw).__name__})") # type: ignore [attr-defined]
            logger.detail(f"    Raw Survival Sum Ranges: {survival_ranges_raw} (Type: {type(survival_ranges_raw).__name__})") # type: ignore [attr-defined]
            logger.detail(f"    Raw Death Degrees: {death_degrees_raw} (Type: {type(death_degrees_raw).__name__})") # type: ignore [attr-defined]
            logger.detail(f"    Raw Death Ranges: {death_ranges_raw} (Type: {type(death_ranges_raw).__name__})") # type: ignore [attr-defined]

            birth_ranges, survival_ranges, death_degrees, death_ranges = [], [], [], []
            try:
                if isinstance(birth_ranges_raw, list):
                    birth_ranges = [tuple(map(float, r)) for r in birth_ranges_raw if isinstance(r, list) and len(r) == 2]
                if isinstance(survival_ranges_raw, list):
                    survival_ranges = [tuple(map(float, r)) for r in survival_ranges_raw if isinstance(r, list) and len(r) == 2]
                if isinstance(death_degrees_raw, list):
                    death_degrees = [int(d) for d in death_degrees_raw]
                if isinstance(death_ranges_raw, list):
                    death_ranges = [tuple(map(float, r)) for r in death_ranges_raw if isinstance(r, list) and len(r) == 2]
            except Exception as e:
                 logger.error(f"    ERROR converting RoL params for '{rule_name}': {e}. Using empty defaults.")

            logger.detail(f"    ---> Parsed Birth Sum Ranges: {birth_ranges}") # type: ignore [attr-defined]
            logger.detail(f"    ---> Parsed Survival Sum Ranges: {survival_ranges}") # type: ignore [attr-defined]
            logger.detail(f"    ---> Parsed Death Degrees: {death_degrees}") # type: ignore [attr-defined]
            logger.detail(f"    ---> Parsed Death Ranges: {death_ranges}") # type: ignore [attr-defined]

            variant_params = {
                "BIRTH_SUM_RANGES": birth_ranges,
                "SURVIVAL_SUM_RANGES": survival_ranges,
                "FINAL_DEATH_DEGREES": death_degrees,
                "FINAL_DEATH_DEGREE_RANGES": death_ranges,
                "NODE_COLORMAP": params.get("node_colormap", 'viridis'),
                "NODE_COLOR_NORM_VMIN": float(params.get("node_color_norm_vmin", 0.0)),
                "NODE_COLOR_NORM_VMAX": float(params.get("node_color_norm_vmax", 8.0)),
                "COLOR_BY_DEGREE": params.get("color_nodes_by_degree", True),
                "COLOR_BY_ACTIVE_NEIGHBORS": params.get("color_nodes_by_active_neighbors", False),
            }
            variants.append((rule_name, "realm_of_lace", variant_params))

        # Process LifeWithColor variant
        elif rule_type == "LifeWithColor":
            logger.debug(f"  Processing LifeWithColor rule.")
            variant_params = {
                "GOL_BIRTH": params.get("birth_neighbor_counts", [3]),
                "GOL_SURVIVAL": params.get("survival_neighbor_counts", [2, 3]),
                "NODE_COLORMAP": params.get("node_colormap", 'plasma'),
                "NODE_COLOR_NORM_VMIN": float(params.get("node_color_norm_vmin", 0.0)),
                "NODE_COLOR_NORM_VMAX": float(params.get("node_color_norm_vmax", 8.0)),
                "COLOR_BY_DEGREE": False,
                "COLOR_BY_ACTIVE_NEIGHBORS": True,
            }
            variants.append((rule_name, "colored_life", variant_params))

    logger.info(f"--- Finished JSON Rule Parsing ---")
    return variants

# Load rules and create variants list
all_json_rules = load_rules_from_json(JSON_FILE_PATH)
RULE_VARIANTS = parse_rule_variants(all_json_rules)

if not RULE_VARIANTS or len(RULE_VARIANTS) <= 1:
    logger.warning("No rule variants loaded or parsed from JSON. Using default fallback rules.")
    RULE_VARIANTS = [
        ("Game of Life (Fallback)", "game_of_life", {
            "GOL_BIRTH": [3], "GOL_SURVIVAL": [2, 3], "NODE_COLORMAP": 'gray',
            "NODE_COLOR_NORM_VMIN": 0.0, "NODE_COLOR_NORM_VMAX": 1.0,
            "COLOR_BY_DEGREE": False, "COLOR_BY_ACTIVE_NEIGHBORS": False,
        }),
        ("Realm of Lace (Fallback)", "realm_of_lace", {
            "BIRTH_SUM_RANGES": [(5.0, 6.0), (8.0, 9.0), (15.0, 16.0)],
            "SURVIVAL_SUM_RANGES": [(3.0, 6.0), (8.0, 11.0), (15.0, 16.0)],
            "FINAL_DEATH_DEGREES": [0, 11, 12, 13, 14], "FINAL_DEATH_DEGREE_RANGES": [],
            "NODE_COLORMAP": 'viridis', "NODE_COLOR_NORM_VMIN": 0.0, "NODE_COLOR_NORM_VMAX": 8.0,
            "COLOR_BY_DEGREE": True, "COLOR_BY_ACTIVE_NEIGHBORS": False,
        }),
         ("Colored Life (Fallback)", "colored_life", {
            "GOL_BIRTH": [3], "GOL_SURVIVAL": [2, 3], "NODE_COLORMAP": 'plasma',
            "NODE_COLOR_NORM_VMIN": 0.0, "NODE_COLOR_NORM_VMAX": 8.0,
            "COLOR_BY_DEGREE": False, "COLOR_BY_ACTIVE_NEIGHBORS": True,
        }),
    ]

# --- Global Settings ---
GRID_SIZE = 1000
WINDOW_SIZE = 1000

# --- Shared Parameters ---
WRAP = True
INITIAL_DENSITY = 0.10
STEP_DELAY = 1/60
paused = False
speed_slider_value = 60.0

# --- Rule Selection State ---
rule_names = [r[0] for r in RULE_VARIANTS]
rule_keys = [r[1] for r in RULE_VARIANTS]
rule_params = [r[2] for r in RULE_VARIANTS]

# --- Find the index of the rule matching the original hardcoded params ---
# The original hardcoded params match "Realm of Lace_Fancy Interesting Shapes"
initial_rule_name_to_find = "Realm of Lace_Fancy Interesting Shapes"
selected_rule_idx = 0 # Default to 0
try:
    # Find the first rule whose name EXACTLY matches the target string
    selected_rule_idx = next(i for i, name in enumerate(rule_names) if name == initial_rule_name_to_find)
    # --- Corrected Log Message ---
    logger.info(f"Found initial rule '{rule_names[selected_rule_idx]}' at index {selected_rule_idx} in RULE_VARIANTS list.")
except StopIteration:
    logger.warning(f"Could not find rule named '{initial_rule_name_to_find}'. Defaulting to index 0 ({rule_names[0]}).")
    selected_rule_idx = 0 # Fallback to the first rule if not found



# --- Rule Parameters (Global Python Vars - mainly for display/reference) ---
BIRTH_SUM_RANGES = []
SURVIVAL_SUM_RANGES = []
FINAL_DEATH_DEGREES = []
FINAL_DEATH_DEGREE_RANGES = []
GOL_BIRTH = []
GOL_SURVIVAL = []
NODE_COLORMAP = 'viridis'
NODE_COLOR_NORM_VMIN = 0.0
NODE_COLOR_NORM_VMAX = 8.0
COLOR_BY_DEGREE = True
COLOR_BY_ACTIVE_NEIGHBORS = False

# --- Visualization Style Settings ---
RENDER_EDGES = False
BASE_NODE_RADIUS = 5.0 # Max radius in pixels when fully zoomed for scaling
BASE_EDGE_WIDTH = 1.0  # Keep edge width constant for now
EDGE_COLOR = (0.6, 0.6, 0.6)

# --- Settings for conditional/scaled rendering ---
# Min separation based on MAX radius to ensure space when fully scaled
MIN_NODE_SEPARATION_PIXELS = BASE_NODE_RADIUS * 2 * 1.2 # e.g., 120% of max diameter
# Separation at which nodes reach BASE_NODE_RADIUS. Higher value means slower scaling.
MAX_SEP_FOR_NODE_SCALING = MIN_NODE_SEPARATION_PIXELS * 3.0 # e.g., scales up over a 3x increase in separation

# --- New settings for conditional/scaled rendering ---
MIN_ZOOM_FOR_NODE_EDGE_RENDER = 3.0 # Minimum zoom level to switch to node/edge drawing
BASE_NODE_RADIUS = 5.0 # Base node radius in pixels *at zoom level 1*
BASE_EDGE_WIDTH = 1.5  # Base edge width in pixels *at zoom level 1*

# --- Fields for Drawing Primitives ---
# Choose a max size large enough for typical zoomed-out views, but not excessively large
# Adjust based on performance and typical node counts
MAX_DRAW_NODES = 50000
MAX_DRAW_LINES = 100000 # Each line needs 2 endpoints

# Fields to hold data copied from Python lists before drawing
# Use vec fields for positions (2D) and colors (3D)
draw_node_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_DRAW_NODES)
draw_node_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_DRAW_NODES)
draw_edge_endpoints = ti.Vector.field(2, dtype=ti.f32, shape=MAX_DRAW_LINES * 2)

# Scalar fields to track actual number of items to draw
num_nodes_to_draw = ti.field(dtype=ti.i32, shape=())
num_edges_to_draw = ti.field(dtype=ti.i32, shape=())

# --- Taichi Fields ---
node_state = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
node_next_state = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
node_active_neighbors = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
node_eligible = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
node_next_eligible = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
node_degree = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
node_next_degree = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))

# --- Fields for Rule Parameters ---
MAX_RANGES = 10
MAX_DEGREES = 20
rol_birth_ranges_field = ti.field(dtype=ti.f32, shape=(MAX_RANGES, 2))
rol_survival_ranges_field = ti.field(dtype=ti.f32, shape=(MAX_RANGES, 2))
rol_death_degrees_field = ti.field(dtype=ti.i32, shape=(MAX_DEGREES))
rol_death_ranges_field = ti.field(dtype=ti.f32, shape=(MAX_RANGES, 2))
rol_num_birth_ranges = ti.field(dtype=ti.i32, shape=())
rol_num_survival_ranges = ti.field(dtype=ti.i32, shape=())
rol_num_death_degrees = ti.field(dtype=ti.i32, shape=())
rol_num_death_ranges = ti.field(dtype=ti.i32, shape=())
gol_birth_field = ti.field(dtype=ti.i32, shape=(MAX_DEGREES))
gol_survival_field = ti.field(dtype=ti.i32, shape=(MAX_DEGREES))
gol_num_birth = ti.field(dtype=ti.i32, shape=())
gol_num_survival = ti.field(dtype=ti.i32, shape=())

# --- Helper Kernel to Clear Parameter Fields ---
@ti.kernel
def clear_param_fields():
    rol_num_birth_ranges[None] = 0
    rol_num_survival_ranges[None] = 0
    rol_num_death_degrees[None] = 0
    rol_num_death_ranges[None] = 0
    gol_num_birth[None] = 0
    gol_num_survival[None] = 0
    for i in range(MAX_RANGES):
        rol_birth_ranges_field[i, 0] = -1.0
        rol_birth_ranges_field[i, 1] = -1.0
        rol_survival_ranges_field[i, 0] = -1.0
        rol_survival_ranges_field[i, 1] = -1.0
        rol_death_ranges_field[i, 0] = -1.0
        rol_death_ranges_field[i, 1] = -1.0
    for i in range(MAX_DEGREES):
        rol_death_degrees_field[i] = -1
        gol_birth_field[i] = -1
        gol_survival_field[i] = -1

# --- Set Rule Variant Function ---
def set_rule_variant(idx):
    """Sets the global parameters AND copies them into Taichi fields."""
    global BIRTH_SUM_RANGES, SURVIVAL_SUM_RANGES, FINAL_DEATH_DEGREES, FINAL_DEATH_DEGREE_RANGES
    global GOL_BIRTH, GOL_SURVIVAL
    global NODE_COLORMAP, NODE_COLOR_NORM_VMIN, NODE_COLOR_NORM_VMAX
    global COLOR_BY_DEGREE, COLOR_BY_ACTIVE_NEIGHBORS

    if idx < 0 or idx >= len(rule_params):
        logger.error(f"Invalid rule index {idx}")
        idx = 0

    params = rule_params[idx]
    rule_key = rule_keys[idx]
    rule_name = rule_names[idx]

    logger.info(f"--- Setting Variant {idx}: {rule_name} (Key: {rule_key}) ---")

    clear_param_fields()

    # Reset Global Python Vars
    BIRTH_SUM_RANGES, SURVIVAL_SUM_RANGES, FINAL_DEATH_DEGREES, FINAL_DEATH_DEGREE_RANGES = [], [], [], []
    GOL_BIRTH, GOL_SURVIVAL = [], []
    NODE_COLORMAP = 'gray'
    NODE_COLOR_NORM_VMIN = 0.0
    NODE_COLOR_NORM_VMAX = 1.0
    COLOR_BY_DEGREE = False
    COLOR_BY_ACTIVE_NEIGHBORS = False

    # Apply specific parameters AND copy to Taichi fields
    if rule_key == "realm_of_lace":
        BIRTH_SUM_RANGES = params.get("BIRTH_SUM_RANGES", [])
        SURVIVAL_SUM_RANGES = params.get("SURVIVAL_SUM_RANGES", [])
        FINAL_DEATH_DEGREES = params.get("FINAL_DEATH_DEGREES", [])
        FINAL_DEATH_DEGREE_RANGES = params.get("FINAL_DEATH_DEGREE_RANGES", [])
        NODE_COLORMAP = params.get("NODE_COLORMAP", 'viridis')
        NODE_COLOR_NORM_VMIN = params.get("NODE_COLOR_NORM_VMIN", 0.0)
        NODE_COLOR_NORM_VMAX = params.get("NODE_COLOR_NORM_VMAX", 8.0)
        COLOR_BY_DEGREE = params.get("COLOR_BY_DEGREE", True)
        COLOR_BY_ACTIVE_NEIGHBORS = params.get("COLOR_BY_ACTIVE_NEIGHBORS", False)

        logger.debug(f"  Retrieved RoL Params for '{rule_name}':")
        logger.debug(f"    Birth Sum Ranges: {BIRTH_SUM_RANGES}")
        logger.debug(f"    Survival Sum Ranges: {SURVIVAL_SUM_RANGES}")
        logger.debug(f"    Death Degrees: {FINAL_DEATH_DEGREES}")
        logger.debug(f"    Death Ranges: {FINAL_DEATH_DEGREE_RANGES}")

        try:
            n_birth = min(len(BIRTH_SUM_RANGES), MAX_RANGES)
            rol_num_birth_ranges[None] = n_birth
            for i in range(n_birth):
                rol_birth_ranges_field[i, 0] = BIRTH_SUM_RANGES[i][0]
                rol_birth_ranges_field[i, 1] = BIRTH_SUM_RANGES[i][1]

            n_survival = min(len(SURVIVAL_SUM_RANGES), MAX_RANGES)
            rol_num_survival_ranges[None] = n_survival
            for i in range(n_survival):
                rol_survival_ranges_field[i, 0] = SURVIVAL_SUM_RANGES[i][0]
                rol_survival_ranges_field[i, 1] = SURVIVAL_SUM_RANGES[i][1]

            n_death_deg = min(len(FINAL_DEATH_DEGREES), MAX_DEGREES)
            rol_num_death_degrees[None] = n_death_deg
            for i in range(n_death_deg):
                rol_death_degrees_field[i] = FINAL_DEATH_DEGREES[i]

            n_death_range = min(len(FINAL_DEATH_DEGREE_RANGES), MAX_RANGES)
            rol_num_death_ranges[None] = n_death_range
            for i in range(n_death_range):
                rol_death_ranges_field[i, 0] = FINAL_DEATH_DEGREE_RANGES[i][0]
                rol_death_ranges_field[i, 1] = FINAL_DEATH_DEGREE_RANGES[i][1]

            logger.info(f"  ---> Copied to Fields: BirthRanges={n_birth}, SurvivalRanges={n_survival}, DeathDeg={n_death_deg}, DeathRanges={n_death_range}")
        except Exception as e:
            logger.error(f"    ERROR copying RoL params to Taichi fields for '{rule_name}': {e}")

    elif rule_key == "game_of_life":
        GOL_BIRTH = params.get("GOL_BIRTH", [3])
        GOL_SURVIVAL = params.get("GOL_SURVIVAL", [2, 3])
        NODE_COLORMAP = params.get("NODE_COLORMAP", 'gray')
        NODE_COLOR_NORM_VMIN = params.get("NODE_COLOR_NORM_VMIN", 0.0)
        NODE_COLOR_NORM_VMAX = params.get("NODE_COLOR_NORM_VMAX", 1.0)
        logger.debug(f"  Retrieved GoL Params: Birth={GOL_BIRTH}, Survival={GOL_SURVIVAL}")

        try:
            n_birth = min(len(GOL_BIRTH), MAX_DEGREES)
            gol_num_birth[None] = n_birth
            for i in range(n_birth):
                gol_birth_field[i] = GOL_BIRTH[i]

            n_survival = min(len(GOL_SURVIVAL), MAX_DEGREES)
            gol_num_survival[None] = n_survival
            for i in range(n_survival):
                gol_survival_field[i] = GOL_SURVIVAL[i]
            logger.info(f"  ---> Copied to Fields: Birth={n_birth}, Survival={n_survival}")
        except Exception as e:
            logger.error(f"    ERROR copying GoL params to Taichi fields for '{rule_name}': {e}")

    elif rule_key == "colored_life":
        GOL_BIRTH = params.get("GOL_BIRTH", [3])
        GOL_SURVIVAL = params.get("GOL_SURVIVAL", [2, 3])
        NODE_COLORMAP = params.get("NODE_COLORMAP", 'plasma')
        NODE_COLOR_NORM_VMIN = params.get("NODE_COLOR_NORM_VMIN", 0.0)
        NODE_COLOR_NORM_VMAX = params.get("NODE_COLOR_NORM_VMAX", 8.0)
        COLOR_BY_DEGREE = False
        COLOR_BY_ACTIVE_NEIGHBORS = True
        logger.debug(f"  Retrieved Colored Life Params: Birth={GOL_BIRTH}, Survival={GOL_SURVIVAL}")

        try:
            n_birth = min(len(GOL_BIRTH), MAX_DEGREES)
            gol_num_birth[None] = n_birth
            for i in range(n_birth):
                gol_birth_field[i] = GOL_BIRTH[i]

            n_survival = min(len(GOL_SURVIVAL), MAX_DEGREES)
            gol_num_survival[None] = n_survival
            for i in range(n_survival):
                gol_survival_field[i] = GOL_SURVIVAL[i]
            logger.info(f"  ---> Copied to Fields: Birth={n_birth}, Survival={n_survival}")
        except Exception as e:
            logger.error(f"    ERROR copying Colored Life params to Taichi fields for '{rule_name}': {e}")

# Initialize with the selected (or default) rule
set_rule_variant(selected_rule_idx)

# --- Neighborhood Definition ---
NUM_NEIGHBORS = 8
neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]

# --- Helper Functions ---
@ti.func
def wrap_idx(idx, N):
    """Wraps or clamps index based on WRAP setting."""
    result = 0
    if WRAP:
        result = idx % N
    else:
        result = min(max(idx, 0), N - 1)
    return result

# --- Initialization ---
@ti.kernel
def initialize_kernel(density: float):
    """Initializes all relevant fields based on random density."""
    for i, j in node_state:
        if ti.random() < density:
            node_state[i, j] = 1
        else:
            node_state[i, j] = 0
        node_next_state[i, j] = 0
        node_active_neighbors[i, j] = 0
        node_eligible[i, j] = 0
        node_next_eligible[i, j] = 0
        node_degree[i, j] = 0
        node_next_degree[i, j] = 0

@ti.kernel
def copy_node_data_to_field(pos_arr: ti.types.ndarray(), color_arr: ti.types.ndarray(), count: int): # type: ignore
    """Copies node position and color data from NumPy arrays to Taichi fields and clears unused."""
    num_nodes = ti.cast(min(count, MAX_DRAW_NODES), ti.i32)
    num_nodes_to_draw[None] = num_nodes
    # Copy valid data
    for i in range(num_nodes): # type: ignore
        draw_node_pos[i][0] = pos_arr[i, 0]
        draw_node_pos[i][1] = pos_arr[i, 1]
        draw_node_colors[i][0] = color_arr[i, 0]
        draw_node_colors[i][1] = color_arr[i, 1]
        draw_node_colors[i][2] = color_arr[i, 2]
    # Clear unused portion
    for i in range(num_nodes, MAX_DRAW_NODES): # type: ignore
        draw_node_pos[i] = ti.Vector([-10.0, -10.0])
        draw_node_colors[i] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def copy_edge_data_to_field(endpoint_arr: ti.types.ndarray(), count: int): # type: ignore
    """Copies edge endpoint data from NumPy array to Taichi field and clears unused."""
    num_endpoints_to_copy = ti.cast(min(count * 2, MAX_DRAW_LINES * 2), ti.i32)
    num_edges_to_draw[None] = num_endpoints_to_copy // 2
    # Copy valid data
    for i in range(num_endpoints_to_copy): # type: ignore
        draw_edge_endpoints[i][0] = endpoint_arr[i, 0]
        draw_edge_endpoints[i][1] = endpoint_arr[i, 1]
    # Clear unused portion
    for i in range(num_endpoints_to_copy, MAX_DRAW_LINES * 2): # type: ignore
        draw_edge_endpoints[i] = ti.Vector([-10.0, -10.0])

def initialize():
    """Calls the initialization kernel and prepares for the selected rule."""
    logger.info("Initializing grid...")
    initialize_kernel(INITIAL_DENSITY)
    if rule_keys[selected_rule_idx] == "realm_of_lace":
         logger.info("RoL rule selected, copying initial state to degree.")
         node_degree.copy_from(node_state)
    logger.info("Initialization complete.")

# --- RealmOfLace Rule ---
@ti.kernel
def rol_compute_eligibility(): # Reads params from fields
    """Computes eligibility based on neighbor degree sum (reads params from fields)."""
    for i, j in node_degree:
        sum_neighbor_degree = 0.0
        for k in ti.static(range(NUM_NEIGHBORS)):
            di, dj = neighbor_offsets[k]
            ni = wrap_idx(i + di, GRID_SIZE)
            nj = wrap_idx(j + dj, GRID_SIZE)
            sum_neighbor_degree += node_degree[ni, nj]

        eligible = 0
        current_degree = node_degree[i, j]

        if current_degree <= 0:
            is_eligible_for_birth = False
            num_ranges = ti.cast(rol_num_birth_ranges[None], ti.i32)
            for r_idx in range(num_ranges): # type: ignore
                min_val = rol_birth_ranges_field[r_idx, 0]
                max_val = rol_birth_ranges_field[r_idx, 1]
                if min_val <= sum_neighbor_degree <= max_val:
                    is_eligible_for_birth = True
                    break
            if is_eligible_for_birth:
                eligible = 1
        else:
            is_eligible_for_survival = False
            num_ranges = ti.cast(rol_num_survival_ranges[None], ti.i32)
            for r_idx in range(num_ranges): # type: ignore
                min_val = rol_survival_ranges_field[r_idx, 0]
                max_val = rol_survival_ranges_field[r_idx, 1]
                if min_val <= sum_neighbor_degree <= max_val:
                    is_eligible_for_survival = True
                    break
            if is_eligible_for_survival:
                eligible = 1
        node_next_eligible[i, j] = eligible

@ti.kernel
def rol_compute_edges_and_degree():
    """Computes next degree based on mutually eligible neighbors."""
    for i, j in node_degree:
        degree = 0
        if node_next_eligible[i, j] > 0:
            for k in ti.static(range(NUM_NEIGHBORS)):
                di, dj = neighbor_offsets[k]
                ni = wrap_idx(i + di, GRID_SIZE)
                nj = wrap_idx(j + dj, GRID_SIZE)
                if node_next_eligible[ni, nj] > 0:
                    degree += 1
        node_next_degree[i, j] = degree

@ti.kernel
def rol_finalize_state(): # Reads params from fields
    """Finalizes the node degree based on eligibility and death rules (reads params from fields)."""
    for i, j in node_degree:
        calculated_degree = node_next_degree[i, j]
        is_eligible = node_next_eligible[i, j] > 0

        final_state = 0
        if is_eligible:
            triggers_death = False
            # 1. Check discrete death degrees from field
            num_degrees = ti.cast(rol_num_death_degrees[None], ti.i32)
            for d_idx in range(num_degrees): # type: ignore
                if calculated_degree == rol_death_degrees_field[d_idx]:
                    triggers_death = True
                    break

            # 2. Check death degree ranges from field
            # --- Restore Range Check ---
            if not triggers_death:
                num_ranges = ti.cast(rol_num_death_ranges[None], ti.i32)
                for r_idx in range(num_ranges): # type: ignore
                    min_val = rol_death_ranges_field[r_idx, 0]
                    max_val = rol_death_ranges_field[r_idx, 1]
                    if min_val <= calculated_degree <= max_val:
                        triggers_death = True
                        break
            # --- End Restore Range Check ---

            if not triggers_death:
                final_state = calculated_degree

        node_degree[i, j] = final_state
        
def rol_step():
    """Performs one step of the Realm of Lace simulation."""
    rol_compute_eligibility()
    rol_compute_edges_and_degree()
    rol_finalize_state()

# --- Game of Life / Colored Life Rule ---
@ti.kernel
def gol_compute_neighbors():
    """Computes the number of active neighbors for each cell based on node_state."""
    for i, j in node_state:
        active_neighbors = 0
        for k in ti.static(range(NUM_NEIGHBORS)):
            di, dj = neighbor_offsets[k]
            ni = wrap_idx(i + di, GRID_SIZE)
            nj = wrap_idx(j + dj, GRID_SIZE)
            if node_state[ni, nj] > 0:
                active_neighbors += 1
        node_active_neighbors[i, j] = active_neighbors

@ti.kernel
def gol_update_state(): # Reads params from fields
    """Updates the next state based on GoL rules using pre-computed neighbor counts (reads params from fields)."""
    for i, j in node_state:
        active_neighbors = node_active_neighbors[i, j]
        current_state = node_state[i, j]
        next_state = 0

        survives = False
        born = False

        if current_state > 0:
            num_survival = ti.cast(gol_num_survival[None], ti.i32)
            for s_idx in range(num_survival): # type: ignore
                if active_neighbors == gol_survival_field[s_idx]:
                    survives = True
                    break
        else:
            num_birth = ti.cast(gol_num_birth[None], ti.i32)
            for b_idx in range(num_birth): # type: ignore
                if active_neighbors == gol_birth_field[b_idx]:
                    born = True
                    break

        if survives or born:
            next_state = 1

        node_next_state[i, j] = next_state

def gol_step():
    """Performs one step of the Game of Life simulation."""
    gol_compute_neighbors()
    gol_update_state()
    node_state.copy_from(node_next_state)

def colored_life_step():
    """Performs one step of the Colored Life simulation."""
    gol_compute_neighbors()
    gol_update_state()
    node_state.copy_from(node_next_state)

# --- Visualization: Node grid as image ---

# NODE_COLORMAP = 'viridis'  # Default: 0 is black
# Other good colormaps where 0 is black (or very dark):
#   'plasma'    # 0 is dark purple/black
#   'inferno'   # 0 is black
#   'magma'     # 0 is black
#   'cividis'   # 0 is black
#   'twilight'  # 0 is black
#   'cubehelix' # 0 is black
#   'hot'       # 0 is black (black-red-yellow-white)
#   'gray'      # 0 is black (grayscale)
#   'Greys'     # 0 is black (grayscale)
# See: https://matplotlib.org/stable/users/explain/colors/colormaps.html

def get_image_window(x0, x1, y0, y1):
    """Generates the image for the current view window based on the selected rule."""
    rule_key = rule_keys[selected_rule_idx]
    img = None
    h = y1 - y0
    w = x1 - x0

    if w <= 0 or h <= 0:
        logger.warning(f"Invalid view window size: w={w}, h={h}. Returning black.")
        return np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)

    try:
        current_cmap = NODE_COLORMAP
        current_vmin = NODE_COLOR_NORM_VMIN
        current_vmax = NODE_COLOR_NORM_VMAX

        if rule_key == "realm_of_lace":
            data_np = node_degree.to_numpy()[y0:y1, x0:x1]
            # Avoid division by zero if vmax <= vmin
            if current_vmax <= current_vmin: current_vmax = current_vmin + 1e-6
            normed = np.clip((data_np - current_vmin) / (current_vmax - current_vmin), 0, 1)
            cmap = cm.get_cmap(current_cmap)
            rgb = (cmap(normed)[..., :3] * 255).astype(np.uint8)
            mask = (data_np == 0)
            rgb[mask] = [0, 0, 0]
            img = rgb

        elif rule_key == "game_of_life":
            state_np = node_state.to_numpy()[y0:y1, x0:x1]
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[state_np > 0] = [255, 255, 255]

        elif rule_key == "colored_life":
            state_np = node_state.to_numpy()[y0:y1, x0:x1]
            neighbors_np = node_active_neighbors.to_numpy()[y0:y1, x0:x1]
            if current_vmax <= current_vmin: current_vmax = current_vmin + 1e-6
            normed = np.clip((neighbors_np - current_vmin) / (current_vmax - current_vmin), 0, 1)
            cmap = cm.get_cmap(current_cmap)
            rgb = (cmap(normed)[..., :3] * 255).astype(np.uint8)
            mask = (state_np == 0)
            rgb[mask] = [0, 0, 0]
            img = rgb

        if img is None:
             logger.warning(f"Unhandled rule key '{rule_key}' in get_image_window. Returning black.")
             img = np.zeros((h, w, 3), dtype=np.uint8)

    except Exception as e:
        logger.error(f"Error during get_image_window for rule {rule_key}: {e}", exc_info=True)
        img = np.zeros((h, w, 3), dtype=np.uint8)

    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
         img_resized = cv2.resize(img, (WINDOW_SIZE, WINDOW_SIZE), interpolation=cv2.INTER_NEAREST)
         return img_resized
    else:
         logger.warning("Image generation resulted in invalid shape or None. Returning black.")
         return np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)

# --- GGUI Setup ---
window = ti.ui.Window("Taichi CA Demo", res=(WINDOW_SIZE, WINDOW_SIZE), vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()

# --- GUI State ---
zoom = 1.0
pan_x = 0.0
pan_y = 0.0
zoom_slider = 1.0
pan_x_slider = 0.0
pan_y_slider = 0.0

# --- Mouse Panning State ---
mouse_dragging = False
last_mouse_x = 0.0
last_mouse_y = 0.0

def clamp_pan(pan, zoom):
    """Clamps panning value based on zoom level. Prevents panning at zoom=1."""
    # Use original max(zoom, 1.0) - this correctly results in max_pan=0 if zoom is 1.0
    safe_zoom = max(zoom, 1.0)
    max_pan = max(0.0, 1.0 - 1.0 / safe_zoom)
    # Log the clamping values for debugging
    logger.debug(f"Clamping pan={pan:.6f} with max_pan={max_pan:.6f} (zoom={zoom:.2f})")
    clamped_pan = np.clip(pan, -max_pan, max_pan)
    logger.debug(f"  -> Clamped pan={clamped_pan:.6f}")
    return clamped_pan

def handle_gui():
    """Manages the GUI elements and updates corresponding global variables."""
    global zoom_slider, zoom, pan_x, pan_y # pan_x/y needed for update
    global INITIAL_DENSITY, STEP_DELAY, paused, frame, speed_slider_value
    global selected_rule_idx
    global RENDER_EDGES

    control_height = 0.36 # Adjusted height

    gui.begin("Controls", 0.01, 0.01, 0.3, control_height)

    # --- Rule Selection ---
    gui.text("Rule Selection")
    gui.text(f"Current: {rule_names[selected_rule_idx]}")
    rule_changed = False
    if gui.button("Previous Rule"):
        selected_rule_idx = (selected_rule_idx - 1 + len(rule_names)) % len(rule_names)
        rule_changed = True
    if gui.button("Next Rule"):
        selected_rule_idx = (selected_rule_idx + 1) % len(rule_names)
        rule_changed = True
    if rule_changed:
        logger.info(f"--- Rule changed via GUI to index {selected_rule_idx} ---")
        set_rule_variant(selected_rule_idx)
        initialize() # Re-initialize grid for the new rule
        clear_render_data_kernel() # <<<--- Call kernel to clear drawing buffers
        frame = 0
        # Clear canvas or force redraw if needed (main loop redraw should handle it)
        logger.info("Rule changed, grid re-initialized, render data cleared.")

    # --- Visualization Style ---
    gui.text("Visualization Style")
    render_edges_checkbox_value = gui.checkbox("Render Nodes & Edges", RENDER_EDGES)
    if render_edges_checkbox_value != RENDER_EDGES:
        logger.info(f"Checkbox value changed! Old RENDER_EDGES={RENDER_EDGES}, New Checkbox Value={render_edges_checkbox_value}")
        RENDER_EDGES = render_edges_checkbox_value
        logger.info(f"Global RENDER_EDGES updated to: {RENDER_EDGES}")

    # --- Zoom Buttons ---
    if gui.button("Zoom to Edges"):
        target_zoom = MIN_NODE_SEPARATION_PIXELS * GRID_SIZE / WINDOW_SIZE
        target_zoom = max(3.0, target_zoom) # Ensure minimum zoom for node visibility
        logger.info(f"Zoom to Edges pressed. Setting Zoom: {target_zoom:.2f}. Resetting pan.")
        zoom = target_zoom
        zoom_slider = zoom
        # --- Pan Reset ---
        pan_x = 0.0 # Center pan
        pan_y = 0.0 # Center pan
        # --- End Pan Reset ---
        if not RENDER_EDGES:
            RENDER_EDGES = True
            logger.info("Enabled Render Nodes & Edges.")

    if gui.button("Zoom Out Full"):
        logger.info("Zoom Out Full pressed.")
        zoom = 1.0
        zoom_slider = zoom
        pan_x = 0.0 # Reset pan when zooming fully out
        pan_y = 0.0
        # Optionally turn off edge rendering when zooming out fully?
        # if RENDER_EDGES:
        #     RENDER_EDGES = False
        #     logger.info("Disabled Render Nodes & Edges.")

    # --- Zoom Control ---
    gui.text("Zoom Control")
    zoom_slider = gui.slider_float("Zoom", zoom_slider, 1.0, 40.0)

    # --- Simulation Controls ---
    gui.text("Simulation Controls")
    INITIAL_DENSITY = gui.slider_float("Init Density", INITIAL_DENSITY, 0.01, 1.0)
    speed_slider_value = gui.slider_float("Speed (steps/sec)", speed_slider_value, 1.0, 240.0)
    STEP_DELAY = 1.0 / max(speed_slider_value, 1.0)
    if gui.button("Pause" if not paused else "Resume"):
        paused = not paused
    if gui.button("Reset"):
        logger.info("--- Reset button pressed ---")
        initialize()
        clear_render_data_kernel() # Also clear render data on manual reset
        frame = 0
    gui.end()

    # --- Update zoom state AFTER GUI ---
    zoom = zoom_slider
    # Pan state is now only updated by handle_mouse or the zoom buttons
    
# Modify function signature to accept the rendering mode flag
def handle_mouse(window, use_node_edge_mode: bool):
    """Handles mouse input events for panning, applying different logic based on render mode."""
    global mouse_dragging, last_mouse_x, last_mouse_y
    global pan_x, pan_y

    # --- Process event queue first to update internal states ---
    while window.get_event():
        pass # Consume all events

    # --- Get current state AFTER processing events ---
    is_lmb_pressed_now = window.is_pressed(ti.ui.LMB)
    current_mouse_x, current_mouse_y = window.get_cursor_pos()

    # --- State transition logic ---
    if is_lmb_pressed_now and not mouse_dragging:
        mouse_dragging = True
        last_mouse_x, last_mouse_y = current_mouse_x, current_mouse_y
        logger.info(f"Mouse drag started (State Change) at ({last_mouse_x:.3f}, {last_mouse_y:.3f})")
    elif not is_lmb_pressed_now and mouse_dragging:
        mouse_dragging = False
        logger.info("Mouse drag ended (State Change).")

    # --- Motion Handling ---
    if mouse_dragging:
        dx_norm = current_mouse_x - last_mouse_x # Positive = Right
        dy_norm = current_mouse_y - last_mouse_y # Positive = Down

        if abs(dx_norm) > 1e-6 or abs(dy_norm) > 1e-6:
            if zoom > 1.0: # Only allow panning if zoom > 1.0

                delta_pan_x = 0.0
                delta_pan_y = 0.0

                # --- Conditional Panning Logic ---
                if use_node_edge_mode:
                    # Logic for Node/Edge Rendering (Above Threshold - Round 2 Fix)
                    # Horizontal mouse movement (dx_norm) affects pan_x. Negative sign for content-follows-mouse.
                    delta_pan_x = -dx_norm / zoom
                    # Vertical mouse movement (dy_norm) affects pan_y. Positive sign for content-follows-mouse (due to Y-flip in mapping).
                    delta_pan_y = +dy_norm / zoom
                    logger.debug("Using Node/Edge panning logic.")
                else:
                    # Logic for Image Rendering (Below Threshold - Round 85 version)
                    # Vertical mouse movement (dy_norm) affects pan_x. Invert sign for content-follows-mouse.
                    delta_pan_x = -dy_norm / zoom
                    # Horizontal mouse movement (dx_norm) affects pan_y. Invert sign for content-follows-mouse.
                    delta_pan_y = -dx_norm / zoom
                    logger.debug("Using Image panning logic.")
                # --- End Conditional Panning Logic ---

                logger.debug(f"  Calculated Delta Pan: ({delta_pan_x:.6f}, {delta_pan_y:.6f}) (Zoom: {zoom:.2f})")
                logger.debug(f"  Before Update: pan_x={pan_x:.6f}, pan_y={pan_y:.6f}")

                new_pan_x = pan_x + delta_pan_x
                new_pan_y = pan_y + delta_pan_y

                logger.debug(f"  Before Clamp: new_pan_x={new_pan_x:.6f}, new_pan_y={new_pan_y:.6f}")

                # Clamp pan values based on zoom level
                pan_x = clamp_pan(new_pan_x, zoom)
                pan_y = clamp_pan(new_pan_y, zoom)

                logger.debug(f"  After Clamp:  pan_x={pan_x:.6f}, pan_y={pan_y:.6f}")

                logger.info(f"Mouse drag motion update: dx={dx_norm:.3f}, dy={dy_norm:.3f} -> Pan=({pan_x:.3f}, {pan_y:.3f})")

            # Update last mouse position regardless of zoom, but only if dragging
            last_mouse_x = current_mouse_x
            last_mouse_y = current_mouse_y

    # Fallback check
    elif mouse_dragging and not is_lmb_pressed_now:
         mouse_dragging = False
         logger.debug("Mouse drag ended (detected button up between frames).")

@ti.func
def map_grid_to_canvas_kernel(i, j, current_zoom, current_pan_x, current_pan_y):
    """Taichi func to map grid coordinates (i, j) to canvas pixel coordinates (cx, cy)."""
    # Using the alternative mapping logic from Round 102
    # 1. Normalize grid coordinate (0 to 1), centering the cell
    norm_x = (ti.cast(i, ti.f32) + 0.5) / GRID_SIZE
    norm_y = (ti.cast(j, ti.f32) + 0.5) / GRID_SIZE

    # 2. Apply pan
    panned_norm_x = norm_x - current_pan_x
    panned_norm_y = norm_y - current_pan_y

    # 3. Apply zoom relative to the panned center (0.5, 0.5)
    zoomed_norm_x = 0.5 + (panned_norm_x - 0.5) * current_zoom
    zoomed_norm_y = 0.5 + (panned_norm_y - 0.5) * current_zoom

    # 4. Scale to window size and flip Y for canvas coordinates (0,0 is top-left)
    canvas_x = zoomed_norm_x * WINDOW_SIZE
    canvas_y = (1.0 - zoomed_norm_y) * WINDOW_SIZE

    return canvas_x, canvas_y

@ti.kernel
def collect_node_render_data_kernel(current_zoom: float, current_pan_x: float, current_pan_y: float,
                                    node_radius_px: float, rule_key_static: ti.template()): # type: ignore
    """Collects visible node data directly into Taichi fields."""
    num_nodes_to_draw[None] = 0

    for j, i in node_state:
        is_active = False
        node_val = 0.0

        if ti.static(rule_key_static == "realm_of_lace"):
            node_val = ti.cast(node_degree[j, i], ti.f32)
            if node_degree[j, i] > 0: is_active = True
        elif ti.static(rule_key_static == "game_of_life"):
            node_val = ti.cast(node_state[j, i], ti.f32)
            if node_state[j, i] > 0: is_active = True
        elif ti.static(rule_key_static == "colored_life"):
            node_val = ti.cast(node_active_neighbors[j, i], ti.f32)
            if node_state[j, i] > 0: is_active = True

        if is_active:
            cx, cy = map_grid_to_canvas_kernel(i, j, current_zoom, current_pan_x, current_pan_y)
            margin = node_radius_px * 1.5
            if -margin < cx < WINDOW_SIZE + margin and -margin < cy < WINDOW_SIZE + margin:
                current_count = ti.cast(num_nodes_to_draw[None], ti.i32)
                # Ignore Pylance error for comparing Taichi expr with Python int
                if current_count < MAX_DRAW_NODES: # type: ignore
                    draw_idx = ti.atomic_add(num_nodes_to_draw[None], 1)
                    # Ignore Pylance error for comparing Taichi expr with Python int
                    if draw_idx < MAX_DRAW_NODES: # type: ignore
                        draw_node_pos[draw_idx] = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                        node_color = ti.Vector([0.8, 0.8, 0.8])
                        if ti.static(rule_key_static == "game_of_life"):
                            node_color = ti.Vector([1.0, 1.0, 1.0])
                        else:
                            norm_val = 0.0
                            v_min = NODE_COLOR_NORM_VMIN
                            v_max = NODE_COLOR_NORM_VMAX
                            if (v_max - v_min) > 1e-6:
                                norm_val = clamp((node_val - v_min) / (v_max - v_min), 0.0, 1.0)
                            node_color = ti.Vector([norm_val, norm_val, norm_val])
                        draw_node_colors[draw_idx] = node_color

@ti.kernel
def collect_edge_render_data_kernel(current_zoom: float, current_pan_x: float, current_pan_y: float,
                                    node_radius_px: float, rule_key_static: ti.template()): # type: ignore
    """Collects visible edge data directly into Taichi fields."""
    num_edges_to_draw[None] = 0

    for j, i in node_state:
        is_active1 = False
        if ti.static(rule_key_static == "realm_of_lace"):
            if node_degree[j, i] > 0: is_active1 = True
        elif ti.static(rule_key_static == "game_of_life"):
            if node_state[j, i] > 0: is_active1 = True
        elif ti.static(rule_key_static == "colored_life"):
            if node_state[j, i] > 0: is_active1 = True

        if is_active1:
            cx1, cy1 = map_grid_to_canvas_kernel(i, j, current_zoom, current_pan_x, current_pan_y)
            margin = node_radius_px * 1.5
            is_visible1 = -margin < cx1 < WINDOW_SIZE + margin and -margin < cy1 < WINDOW_SIZE + margin

            if is_visible1:
                for k in ti.static(range(NUM_NEIGHBORS)):
                    di, dj = neighbor_offsets[k]
                    ni_raw, nj_raw = i + di, j + dj
                    neighbor_out_of_bounds = not (0 <= ni_raw < GRID_SIZE and 0 <= nj_raw < GRID_SIZE)
                    if WRAP and neighbor_out_of_bounds: continue
                    ni = wrap_idx(ni_raw, GRID_SIZE)
                    nj = wrap_idx(nj_raw, GRID_SIZE)

                    is_active2 = False
                    if ti.static(rule_key_static == "realm_of_lace"):
                        if node_degree[nj, ni] > 0: is_active2 = True
                    elif ti.static(rule_key_static == "game_of_life"):
                        if node_state[nj, ni] > 0: is_active2 = True
                    elif ti.static(rule_key_static == "colored_life"):
                        if node_state[nj, ni] > 0: is_active2 = True

                    if is_active2:
                        idx1 = j * GRID_SIZE + i
                        idx2 = nj * GRID_SIZE + ni
                        if idx1 < idx2:
                            current_edge_count = ti.cast(num_edges_to_draw[None], ti.i32)
                            if current_edge_count < MAX_DRAW_LINES: # type: ignore
                                draw_idx_base = ti.atomic_add(num_edges_to_draw[None], 1) * 2 # type: ignore
                                if ti.cast(draw_idx_base, ti.i32) + ti.i32(1) < MAX_DRAW_LINES * 2: # type: ignore
                                    cx2_raw, cy2_raw = map_grid_to_canvas_kernel(ni_raw, nj_raw, current_zoom, current_pan_x, current_pan_y)
                                    draw_edge_endpoints[draw_idx_base] = ti.Vector([cx1 / WINDOW_SIZE, cy1 / WINDOW_SIZE])
                                    # --- Cast Python literal 1 to ti.i32 for indexing ---
                                    draw_edge_endpoints[draw_idx_base + ti.i32(1)] = ti.Vector([cx2_raw / WINDOW_SIZE, cy2_raw / WINDOW_SIZE])
                                    # ---

@ti.kernel
def clear_render_data_kernel():
    """Clears the Taichi fields used for drawing nodes and edges."""
    num_nodes_to_draw[None] = 0
    num_edges_to_draw[None] = 0
    # Setting the counts to zero is usually sufficient, as the drawing loops
    # iterate based on these counts. Explicitly clearing the arrays might
    # be slightly safer but adds overhead. Let's rely on the counts for now.
    # for i in range(MAX_DRAW_NODES):
    #     draw_node_pos[i] = ti.Vector([-10.0, -10.0]) # Off-screen
    #     draw_node_colors[i] = ti.Vector([0.0, 0.0, 0.0])
    # for i in range(MAX_DRAW_LINES * 2):
    #     draw_edge_endpoints[i] = ti.Vector([-10.0, -10.0]) # Off-screen

def map_canvas_to_grid_approx(px, py):
    """Approximates the grid coordinates (i, j) corresponding to canvas pixel coordinates (px, py)."""
    # Inverse of the map_grid_to_canvas logic
    # 1. Normalize canvas coordinates (Y flipped)
    norm_canvas_x = px / WINDOW_SIZE
    norm_canvas_y = (WINDOW_SIZE - py) / WINDOW_SIZE # Flip Y back

    # 2. Reverse zoom relative to canvas center (0.5, 0.5)
    offset_x_norm = (norm_canvas_x - 0.5) / zoom
    offset_y_norm = (norm_canvas_y - 0.5) / zoom

    # 3. Reverse pan to find original grid normalized coordinates
    # canvas_norm = 0.5 + (grid_norm - (0.5 + pan)) * zoom  <- Original forward mapping idea (less clear)
    # Let's use the kernel's forward mapping logic and invert it:
    # zoomed_norm_x = 0.5 + (panned_norm_x - 0.5) * current_zoom
    # zoomed_norm_y = 0.5 + (panned_norm_y - 0.5) * current_zoom
    # panned_norm_x = norm_x - current_pan_x
    # panned_norm_y = norm_y - current_pan_y

    # From step 2, we have offset_x_norm = (panned_norm_x - 0.5)
    # And offset_y_norm = (panned_norm_y - 0.5)
    # So, panned_norm_x = offset_x_norm + 0.5
    # And panned_norm_y = offset_y_norm + 0.5

    panned_norm_x = (norm_canvas_x - 0.5) / zoom + 0.5
    panned_norm_y = (norm_canvas_y - 0.5) / zoom + 0.5

    # Now reverse pan: norm_x = panned_norm_x + current_pan_x
    norm_x_grid = panned_norm_x + pan_x
    norm_y_grid = panned_norm_y + pan_y

    # 4. Scale to grid size
    grid_i = norm_x_grid * GRID_SIZE
    grid_j = norm_y_grid * GRID_SIZE

    # logger.debug(f"map_canvas_to_grid({px}, {py}) -> norm_canvas=({norm_canvas_x:.3f}, {norm_canvas_y:.3f}) -> offset=({offset_x_norm:.3f}, {offset_y_norm:.3f}) -> panned_norm=({panned_norm_x:.3f}, {panned_norm_y:.3f}) -> grid_norm=({norm_x_grid:.3f}, {norm_y_grid:.3f}) -> grid=({grid_i:.1f}, {grid_j:.1f})")

    return grid_i, grid_j

# --- Specialized Kernels for Render Data Collection ---

@ti.func
def get_rol_color(degree_val): # No type hint
    """Calculates grayscale color based on RoL degree."""
    # Simplified grayscale based on normalized value for RoL
    # Access globals directly (captured at compile time of kernel using this func)
    norm_val = 0.0
    v_min = NODE_COLOR_NORM_VMIN
    v_max = NODE_COLOR_NORM_VMAX
    if (v_max - v_min) > 1e-6:
        # Ensure clamp function is available or defined as ti.func
        norm_val = clamp((degree_val - v_min) / (v_max - v_min), 0.0, 1.0)
    return ti.Vector([norm_val, norm_val, norm_val])

@ti.func
def get_colored_life_color(neighbor_val): # No type hint
    """Calculates grayscale color based on Colored Life neighbor count."""
    # Simplified grayscale based on normalized value for Colored Life
    norm_val = 0.0
    v_min = NODE_COLOR_NORM_VMIN
    v_max = NODE_COLOR_NORM_VMAX
    if (v_max - v_min) > 1e-6:
        # Ensure clamp function is available or defined as ti.func
        norm_val = clamp((neighbor_val - v_min) / (v_max - v_min), 0.0, 1.0)
    return ti.Vector([norm_val, norm_val, norm_val])

                   
@ti.kernel
def collect_gol_render_data(current_zoom: float, current_pan_x: float, current_pan_y: float, node_radius_px: float):
    """Collects visible node data for Game of Life (no edges)."""
    num_nodes_to_draw[None] = 0
    num_edges_to_draw[None] = 0 # GoL has no edges here

    for j, i in node_state: # Iterate using GoL's primary state field
        if node_state[j, i] > 0: # Check if active
            cx, cy = map_grid_to_canvas_kernel(i, j, current_zoom, current_pan_x, current_pan_y)
            margin = node_radius_px * 1.5
            is_visible = -margin < cx < WINDOW_SIZE + margin and -margin < cy < WINDOW_SIZE + margin

            if is_visible:
                current_count = ti.cast(num_nodes_to_draw[None], ti.i32)
                if current_count < MAX_DRAW_NODES: # type: ignore
                    draw_idx = ti.atomic_add(num_nodes_to_draw[None], 1)
                    if draw_idx < MAX_DRAW_NODES: # type: ignore
                        draw_node_pos[draw_idx] = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                        draw_node_colors[draw_idx] = ti.Vector([1.0, 1.0, 1.0]) # White

@ti.kernel
def collect_rol_render_data(current_zoom: float, current_pan_x: float, current_pan_y: float, node_radius_px: float):
    """Collects visible node and edge data for Realm of Lace."""
    num_nodes_to_draw[None] = 0
    num_edges_to_draw[None] = 0

    for j, i in node_degree: # Iterate using RoL's primary state field
        if node_degree[j, i] > 0: # Check if active
            cx, cy = map_grid_to_canvas_kernel(i, j, current_zoom, current_pan_x, current_pan_y)
            margin = node_radius_px * 1.5
            is_visible1 = -margin < cx < WINDOW_SIZE + margin and -margin < cy < WINDOW_SIZE + margin

            if is_visible1:
                # Add Node
                current_node_count = ti.cast(num_nodes_to_draw[None], ti.i32)
                if current_node_count < MAX_DRAW_NODES: # type: ignore
                    draw_idx = ti.atomic_add(num_nodes_to_draw[None], 1)
                    if draw_idx < MAX_DRAW_NODES: # type: ignore
                        draw_node_pos[draw_idx] = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                        node_val = ti.cast(node_degree[j, i], ti.f32)
                        draw_node_colors[draw_idx] = get_rol_color(node_val) # Use helper func

                # Check Neighbors for Edges
                for k in ti.static(range(NUM_NEIGHBORS)):
                    di, dj = neighbor_offsets[k]
                    ni_raw, nj_raw = i + di, j + dj

                    # Calculate bounds check result (runtime variable)
                    neighbor_out_of_bounds = not (0 <= ni_raw < GRID_SIZE and 0 <= nj_raw < GRID_SIZE)

                    # --- MODIFIED: Avoid continue in static loop ---
                    # Process neighbor only if it's not a wrapped edge when WRAP is True
                    if not (WRAP and neighbor_out_of_bounds):
                        ni = wrap_idx(ni_raw, GRID_SIZE) # wrap_idx handles clamping if WRAP is False
                        nj = wrap_idx(nj_raw, GRID_SIZE)

                        if node_degree[nj, ni] > 0: # Check neighbor activity (RoL)
                            idx1 = j * GRID_SIZE + i
                            idx2 = nj * GRID_SIZE + ni
                            if idx1 < idx2: # Avoid duplicate edges
                                current_edge_count = ti.cast(num_edges_to_draw[None], ti.i32)
                                if current_edge_count < MAX_DRAW_LINES: # type: ignore
                                    draw_idx_base = ti.atomic_add(num_edges_to_draw[None], 1) * 2 # type: ignore
                                    if ti.cast(draw_idx_base, ti.i32) + ti.i32(1) < MAX_DRAW_LINES * 2: # type: ignore
                                        cx2_raw, cy2_raw = map_grid_to_canvas_kernel(ni_raw, nj_raw, current_zoom, current_pan_x, current_pan_y)
                                        draw_edge_endpoints[draw_idx_base] = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                                        draw_edge_endpoints[draw_idx_base + ti.i32(1)] = ti.Vector([cx2_raw / WINDOW_SIZE, cy2_raw / WINDOW_SIZE])
                    # --- END MODIFIED ---

@ti.kernel
def collect_colored_life_render_data(current_zoom: float, current_pan_x: float, current_pan_y: float, node_radius_px: float):
    """Collects visible node and edge data for Colored Life."""
    num_nodes_to_draw[None] = 0
    num_edges_to_draw[None] = 0

    for j, i in node_state: # Iterate using node_state for activity
        if node_state[j, i] > 0: # Check if active
            cx, cy = map_grid_to_canvas_kernel(i, j, current_zoom, current_pan_x, current_pan_y)
            margin = node_radius_px * 1.5
            is_visible1 = -margin < cx < WINDOW_SIZE + margin and -margin < cy < WINDOW_SIZE + margin

            if is_visible1:
                # Add Node
                current_node_count = ti.cast(num_nodes_to_draw[None], ti.i32)
                if current_node_count < MAX_DRAW_NODES: # type: ignore
                    draw_idx = ti.atomic_add(num_nodes_to_draw[None], 1)
                    if draw_idx < MAX_DRAW_NODES: # type: ignore
                        draw_node_pos[draw_idx] = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                        node_val = ti.cast(node_active_neighbors[j, i], ti.f32) # Color based on neighbors
                        draw_node_colors[draw_idx] = get_colored_life_color(node_val) # Use helper func

                # Check Neighbors for Edges (Edges between active nodes)
                for k in ti.static(range(NUM_NEIGHBORS)):
                    di, dj = neighbor_offsets[k]
                    ni_raw, nj_raw = i + di, j + dj

                    # Calculate bounds check result (runtime variable)
                    neighbor_out_of_bounds = not (0 <= ni_raw < GRID_SIZE and 0 <= nj_raw < GRID_SIZE)

                    # --- MODIFIED: Avoid continue in static loop ---
                    # Process neighbor only if it's not a wrapped edge when WRAP is True
                    if not (WRAP and neighbor_out_of_bounds):
                        ni = wrap_idx(ni_raw, GRID_SIZE)
                        nj = wrap_idx(nj_raw, GRID_SIZE)

                        if node_state[nj, ni] > 0: # Check neighbor activity (GoL state)
                            idx1 = j * GRID_SIZE + i
                            idx2 = nj * GRID_SIZE + ni
                            if idx1 < idx2: # Avoid duplicate edges
                                current_edge_count = ti.cast(num_edges_to_draw[None], ti.i32)
                                if current_edge_count < MAX_DRAW_LINES: # type: ignore
                                    draw_idx_base = ti.atomic_add(num_edges_to_draw[None], 1) * 2 # type: ignore
                                    if ti.cast(draw_idx_base, ti.i32) + ti.i32(1) < MAX_DRAW_LINES * 2: # type: ignore
                                        cx2_raw, cy2_raw = map_grid_to_canvas_kernel(ni_raw, nj_raw, current_zoom, current_pan_x, current_pan_y)
                                        draw_edge_endpoints[draw_idx_base] = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                                        draw_edge_endpoints[draw_idx_base + ti.i32(1)] = ti.Vector([cx2_raw / WINDOW_SIZE, cy2_raw / WINDOW_SIZE])
                    # --- END MODIFIED ---

# --- Ensure clamp function is defined ---
@ti.func
def clamp(v, v_min, v_max):
    """Clamps value v between v_min and v_max."""
    return ti.max(v_min, ti.min(v_max, v))

def map_grid_to_canvas(i, j):
    """Maps grid coordinates (i, j) to canvas pixel coordinates (cx, cy). (Original Logic)"""
    # This mapping treats the grid as centered on (0.5, 0.5) in normalized coords
    # It applies zoom relative to the center and then pan, scaling to window size.
    norm_x = i / GRID_SIZE
    norm_y = j / GRID_SIZE
    # Apply zoom relative to center (0.5, 0.5), then apply pan, then scale to window
    # Note: Pan shifts the *center* of the view. A positive pan_x means the grid's 0.5 moves left relative to the canvas's 0.5.
    # So, to find the canvas coord for a grid point, we find its offset from the panned center (0.5+pan) and scale by zoom.
    # canvas_norm_x = 0.5 + (norm_x - (0.5 + pan_x)) * zoom # Incorrect interpretation
    # canvas_norm_y = 0.5 + (norm_y - (0.5 + pan_y)) * zoom # Incorrect interpretation

    # Corrected interpretation: Find the grid point's position relative to the view center (panned grid center), scale by zoom, place relative to canvas center.
    grid_center_x = 0.5 + pan_x
    grid_center_y = 0.5 + pan_y
    offset_x_norm = norm_x - grid_center_x
    offset_y_norm = norm_y - grid_center_y
    canvas_norm_x = 0.5 + offset_x_norm * zoom
    canvas_norm_y = 0.5 + offset_y_norm * zoom

    canvas_x = canvas_norm_x * WINDOW_SIZE
    # Flip Y for canvas coordinates (0,0 is top-left)
    canvas_y = (1.0 - canvas_norm_y) * WINDOW_SIZE

    return canvas_x, canvas_y

# Modify function signature to accept calculated sizes
def render_nodes_and_edges(current_node_radius_px, current_edge_width_px):
    """Renders nodes as circles and edges as lines directly onto the canvas using Taichi fields."""
    # logger.debug(f"Starting render_nodes_and_edges with radius={current_node_radius_px}, width={current_edge_width_px}")
    rule_key = rule_keys[selected_rule_idx]

    # --- Log Current Colormap ---
    logger.debug(f"Render func using NODE_COLORMAP='{NODE_COLORMAP}', VMIN={NODE_COLOR_NORM_VMIN}, VMAX={NODE_COLOR_NORM_VMAX}")
    # ---

    # --- Data Preparation ---
    # (Keep this section as is)
    # ... rest of the function remains the same as Round 105 ...
    active_map = None
    color_data = None
    if rule_key == "realm_of_lace":
        degree_np = node_degree.to_numpy()
        active_map = degree_np > 0
        color_data = degree_np
    elif rule_key == "game_of_life":
        state_np = node_state.to_numpy()
        active_map = state_np > 0
        color_data = state_np
    elif rule_key == "colored_life":
        state_np = node_state.to_numpy()
        active_map = state_np > 0
        color_data = node_active_neighbors.to_numpy()
    else: return
    if active_map is None or color_data is None:
        logger.error("Failed to get necessary numpy arrays for rendering.")
        return

    # --- Determine Visible Grid Range ---
    def map_canvas_to_grid_approx(px, py):
        py_norm_canvas = (WINDOW_SIZE - py) / WINDOW_SIZE
        norm_x_grid = (px / WINDOW_SIZE - 0.5) / zoom + 0.5 + pan_x
        norm_y_grid = (py_norm_canvas - 0.5) / zoom + 0.5 + pan_y
        grid_i = norm_x_grid * GRID_SIZE
        grid_j = norm_y_grid * GRID_SIZE
        return grid_i, grid_j
    g_i00, g_j00 = map_canvas_to_grid_approx(0, 0)
    g_i11, g_j11 = map_canvas_to_grid_approx(WINDOW_SIZE, WINDOW_SIZE)
    min_i_visible = math.floor(min(g_i00, g_i11))
    max_i_visible = math.ceil(max(g_i00, g_i11))
    min_j_visible = math.floor(min(g_j00, g_j11))
    max_j_visible = math.ceil(max(g_j00, g_j11))
    buffer_grid_cells = 10
    view_x0 = max(0, min_i_visible - buffer_grid_cells)
    view_y0 = max(0, min_j_visible - buffer_grid_cells)
    view_x1 = min(GRID_SIZE, max_i_visible + buffer_grid_cells)
    view_y1 = min(GRID_SIZE, max_j_visible + buffer_grid_cells)
    if view_x1 <= view_x0: view_x1 = view_x0 + 1
    if view_y1 <= view_y0: view_y1 = view_y0 + 1
    view_x1 = min(GRID_SIZE, view_x1)
    view_y1 = min(GRID_SIZE, view_y1)
    # logger.debug(f"Final Iteration Range: i={view_x0}-{view_x1}, j={view_y0}-{view_y1}")

    # --- Check if any active nodes are in the calculated range BEFORE iterating ---
    active_nodes_in_slice = 0
    if view_x1 > view_x0 and view_y1 > view_y0:
        try:
            visible_slice = active_map[view_y0:view_y1, view_x0:view_x1]
            active_nodes_in_slice = np.count_nonzero(visible_slice)
            if active_nodes_in_slice > 0:
                 logger.debug(f"Slice check found {active_nodes_in_slice} active nodes in [{view_y0}:{view_y1}, {view_x0}:{view_x1}]. Proceeding.")
                 # logger.debug(f"Final Iteration Range: i={view_x0}-{view_x1}, j={view_y0}-{view_y1}") # Log range only if iterating
            # else:
                 # logger.debug(f"No active nodes found in slice [{view_y0}:{view_y1}, {view_x0}:{view_x1}]. Skipping loop.")
        except IndexError as e:
            logger.error(f"IndexError accessing active_map slice [{view_y0}:{view_y1}, {view_x0}:{view_x1}]: {e}")
            active_nodes_in_slice = -1
    else:
        active_nodes_in_slice = 0

    if active_nodes_in_slice <= 0:
        num_nodes_to_draw[None] = 0
        num_edges_to_draw[None] = 0
        return

    # --- Prepare Drawing Data in Python Lists ---
    node_positions_list = []
    node_colors_list = []
    edge_endpoints_list = []
    processed_edges = set()

    try:
        cmap = cm.get_cmap(NODE_COLORMAP) # Use the global variable
    except ValueError:
        logger.warning(f"Invalid colormap '{NODE_COLORMAP}', defaulting to 'viridis'.")
        cmap = cm.get_cmap('viridis')

    def python_wrap_idx(idx, N):
        if WRAP: return idx % N
        else: return min(max(idx, 0), N - 1)

    # --- Iterate over POTENTIALLY VISIBLE grid and Collect Data ---
    start_collect_time = time.perf_counter()
    nodes_collected_count = 0
    first_active_logged = False

    for j in range(view_y0, view_y1):
        for i in range(view_x0, view_x1):
            if not (0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE): continue
            is_active = active_map[j, i]
            if not is_active: continue

            if not first_active_logged:
                # logger.debug(f"  Iteration loop found first active node at grid({i},{j})") # Reduce noise
                first_active_logged = True

            nodes_collected_count += 1
            cx, cy = map_grid_to_canvas(i, j)
            margin = current_node_radius_px * 1.5
            if -margin <= cx <= WINDOW_SIZE + margin and \
               -margin <= cy <= WINDOW_SIZE + margin:

                norm_cx = cx / WINDOW_SIZE
                norm_cy = cy / WINDOW_SIZE
                node_positions_list.append((norm_cx, norm_cy))

                node_val = color_data[j, i]
                if rule_key == "game_of_life":
                    node_color = (1.0, 1.0, 1.0)
                else:
                    # Use global VMIN/VMAX here
                    norm_val = 0.0
                    if (NODE_COLOR_NORM_VMAX - NODE_COLOR_NORM_VMIN) > 1e-6:
                         norm_val = np.clip((node_val - NODE_COLOR_NORM_VMIN) / (NODE_COLOR_NORM_VMAX - NODE_COLOR_NORM_VMIN), 0, 1)
                    node_color = cmap(norm_val)[:3]
                node_colors_list.append(node_color)

                # Calculate Edges
                for k in range(NUM_NEIGHBORS):
                    di, dj = neighbor_offsets[k]
                    ni_raw, nj_raw = i + di, j + dj
                    neighbor_out_of_bounds = not (0 <= ni_raw < GRID_SIZE and 0 <= nj_raw < GRID_SIZE)
                    edge_wraps = WRAP and neighbor_out_of_bounds
                    if edge_wraps: continue
                    ni_idx = python_wrap_idx(ni_raw, GRID_SIZE)
                    nj_idx = python_wrap_idx(nj_raw, GRID_SIZE)
                    if not (0 <= ni_idx < GRID_SIZE and 0 <= nj_idx < GRID_SIZE): continue
                    if active_map[nj_idx, ni_idx]:
                        node1_idx = j * GRID_SIZE + i
                        node2_idx = nj_idx * GRID_SIZE + ni_idx
                        edge_tuple = tuple(sorted((node1_idx, node2_idx)))
                        if edge_tuple not in processed_edges:
                            ncx, ncy = map_grid_to_canvas(ni_raw, nj_raw)
                            edge_endpoints_list.append((norm_cx, norm_cy))
                            edge_endpoints_list.append((ncx / WINDOW_SIZE, ncy / WINDOW_SIZE))
                            processed_edges.add(edge_tuple)

    collect_duration = time.perf_counter() - start_collect_time
    num_nodes_in_list = len(node_positions_list)
    num_edges_collected = len(edge_endpoints_list) // 2

    logger.debug(f"Data collection took {collect_duration:.4f}s. Active in Loop: {nodes_collected_count}, Nodes in List: {num_nodes_in_list}, Edges in List: {num_edges_collected}")

    if num_nodes_in_list == 0 and num_edges_collected == 0:
        num_nodes_to_draw[None] = 0
        num_edges_to_draw[None] = 0
        return

    # --- Copy Data to Taichi Fields ---
    # (Keep this section as is)
    num_nodes = num_nodes_in_list
    num_edges = num_edges_collected
    if num_nodes > 0:
        if num_nodes > MAX_DRAW_NODES: logger.warning(f"Truncating {num_nodes} nodes to {MAX_DRAW_NODES}."); num_nodes = MAX_DRAW_NODES
        node_pos_np = np.array(node_positions_list[:num_nodes], dtype=np.float32)
        node_colors_np = np.array(node_colors_list[:num_nodes], dtype=np.float32)
        copy_node_data_to_field(node_pos_np, node_colors_np, num_nodes)
    else: num_nodes_to_draw[None] = 0
    if num_edges > 0:
        if num_edges > MAX_DRAW_LINES: logger.warning(f"Truncating {num_edges} edges to {MAX_DRAW_LINES}."); num_edges = MAX_DRAW_LINES
        edge_endpoints_np = np.array(edge_endpoints_list[:num_edges * 2], dtype=np.float32)
        copy_edge_data_to_field(edge_endpoints_np, num_edges)
    else: num_edges_to_draw[None] = 0

    # --- Draw using Taichi GGUI with Taichi Fields ---
    # (Keep this section as is)
    actual_edges = num_edges_to_draw[None]
    if actual_edges > 0:
        width_norm = max(0.0, current_edge_width_px / max(WINDOW_SIZE, 1))
        canvas.lines(draw_edge_endpoints, width=width_norm, color=EDGE_COLOR)

    actual_nodes = num_nodes_to_draw[None]
    if actual_nodes > 0:
        draw_radius_px = max(1.0, current_node_radius_px + current_edge_width_px * 0.5)
        radius_norm = max(0.0, draw_radius_px / max(WINDOW_SIZE, 1))
        default_color_tuple = (0.8, 0.8, 0.8)
        canvas.circles(draw_node_pos,
                       radius=radius_norm,
                       color=default_color_tuple,
                       per_vertex_color=draw_node_colors)

    # logger.debug("Finished render_nodes_and_edges")
    
def handle_keyboard(window):
    """Handles keyboard input events."""
    global paused
    if window.get_event(ti.ui.PRESS):
        if window.event.key == ti.ui.SPACE:
             paused = not paused
             logger.debug(f"Pause toggled via keyboard: {paused}")

# --- Main Loop ---
initialize()
frame = 0
logger.info("--- Entering Main Loop ---")
while window.running:
    # Process inputs first
    handle_keyboard(window)

    # --- Determine Rendering Mode *Before* Handling Mouse ---
    # Calculate projected separation
    px0, py0 = map_grid_to_canvas(0, 0)
    px1, py1 = map_grid_to_canvas(1, 0) # Adjacent grid points horizontally
    dist_sq = (px1 - px0)**2 + (py1 - py0)**2
    projected_separation_pixels = np.sqrt(dist_sq) if dist_sq > 0 else 0

    # Decide rendering mode based on toggle AND calculated separation
    separation_check = projected_separation_pixels >= MIN_NODE_SEPARATION_PIXELS
    use_node_edge_render_mode = RENDER_EDGES and separation_check
    # --- End Determine Rendering Mode ---

    # Pass the calculated mode to handle_mouse
    handle_mouse(window, use_node_edge_render_mode) # Handle mouse events for panning
    handle_gui()         # Update GUI widgets and handle their interactions

    rule_key = rule_keys[selected_rule_idx]

    # --- Simulation Step ---
    if not paused:
        if rule_key == "realm_of_lace":
            rol_step()
        elif rule_key == "game_of_life":
            gol_step()
        elif rule_key == "colored_life":
            colored_life_step()
        frame += 1

    # --- Visualization ---
    canvas.set_background_color((0, 0, 0)) # Clear canvas

    # --- Use the pre-calculated rendering mode ---
    if use_node_edge_render_mode:
        # Calculate dynamic radius/width based on zoom
        min_radius_at_threshold = max(1.0, MIN_NODE_SEPARATION_PIXELS / (2.0 * 1.2))
        scaling_range = max(1e-6, MAX_SEP_FOR_NODE_SCALING - MIN_NODE_SEPARATION_PIXELS)
        scale_factor = np.clip((projected_separation_pixels - MIN_NODE_SEPARATION_PIXELS) / scaling_range, 0.0, 1.0)
        current_node_radius_pixels = max(1.0, min_radius_at_threshold + scale_factor * (BASE_NODE_RADIUS - min_radius_at_threshold))
        current_edge_width_pixels = BASE_EDGE_WIDTH

        render_nodes_and_edges(current_node_radius_pixels, current_edge_width_pixels)
    else:
        # Use original image-based rendering (faster, good for zoomed out)
        if RENDER_EDGES and not separation_check:
             logger.debug(f"RENDER_EDGES is True, but separation ({projected_separation_pixels:.2f}) < threshold ({MIN_NODE_SEPARATION_PIXELS:.2f}). Using image rendering.")

        # --- Image Rendering View Calculation ---
        # This calculation seems different from map_grid_to_canvas and might be the source
        # of the discrepancy if the pan interpretation differs.
        # Let's keep it for now, but be aware it might need adjustment if panning
        # still feels off in image mode after this fix.
        view_size_pixels = GRID_SIZE / zoom
        view_size_half_pixels = view_size_pixels / 2
        # Map pan (-1 to 1 relative to center) to grid coordinates
        # Center point in grid coords: GRID_SIZE/2
        # Panned center point: GRID_SIZE/2 + pan * (GRID_SIZE/2) ? Let's re-evaluate this.
        # If pan_x = 0, center is GRID_SIZE/2. If pan_x = 1 (max right), center should be GRID_SIZE? No, max pan is less than 1.
        # Let's use the inverse of map_canvas_to_grid_approx from render_nodes_and_edges
        # to find the grid coordinates corresponding to the canvas center (0.5, 0.5)
        center_canvas_x, center_canvas_y = 0.5 * WINDOW_SIZE, 0.5 * WINDOW_SIZE
        center_grid_i, center_grid_j = map_canvas_to_grid_approx(center_canvas_x, center_canvas_y)

        # Use the calculated grid center for the view window
        cx_pixels = center_grid_i # Use calculated grid center i
        cy_pixels = center_grid_j # Use calculated grid center j

        x0 = int(np.clip(cx_pixels - view_size_half_pixels, 0, GRID_SIZE))
        y0 = int(np.clip(cy_pixels - view_size_half_pixels, 0, GRID_SIZE))
        x1 = int(np.clip(x0 + view_size_pixels, 0, GRID_SIZE))
        y1 = int(np.clip(y0 + view_size_pixels, 0, GRID_SIZE))
        # --- End Image Rendering View Calculation ---

        view_width = x1 - x0
        view_height = y1 - y0

        if view_width > 0 and view_height > 0:
            img_resized = get_image_window(x0, x1, y0, y1)
            canvas.set_image(img_resized)
        else:
            # Avoid division by zero or invalid slice if view is too small
            logger.warning(f"Calculated view window for image rendering is invalid: w={view_width}, h={view_height}. Setting black.")
            canvas.set_image(np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8))

    # --- Info Display ---
    control_height = 0.38 # Match height in handle_gui
    info_y = 0.01 + control_height + 0.01
    gui.begin("Info", 0.01, info_y, 0.3, 0.18)
    gui.text(f"Frame: {frame}")
    gui.text(f"Rule: {rule_names[selected_rule_idx]}")
    gui.text(f"Zoom: {zoom:.2f}")
    gui.text(f"Pan: ({pan_x:.2f}, {pan_y:.2f})")
    gui.text(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    gui.text(f"Paused: {paused}")
    gui.text(f"Init Density: {INITIAL_DENSITY:.2f}")
    gui.text(f"Speed (steps/sec): {speed_slider_value:.1f}")
    gui.text(f"Render Edges: {RENDER_EDGES}")
    gui.end()

    window.show()
    ti.sync()
    time.sleep(STEP_DELAY)

logger.info("--- Exited Main Loop ---")
# =========== END of taichi_realm_of_lace_demo.py ===========