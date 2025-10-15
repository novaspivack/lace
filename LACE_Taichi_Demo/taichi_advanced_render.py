
# --- Imports ---
import taichi as ti
import math
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import to_rgb # For color conversion
import time
import cv2  # pip install opencv-python
import json
import os
import sys
import logging # Keep logging import for local setup
from datetime import datetime
from typing import List, Tuple, Optional, Union, TypeVar, Dict, Set

# --- Adjust Python Path to Find LACE modules ---
# Get the directory of the current script (LACE_Taichi_Demo)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (where the main LACE directory resides)
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to sys.path so Python can find the LACE package
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"Added {parent_dir} to sys.path") # Optional: confirm path addition

# --- Attempt to import ColorManager (Needs LACE path) ---
try:
    # ColorManager should now find the correct logger via its own import from LACE.logging_config
    from LACE.colors import ColorManager, ColorScheme
    COLOR_MODULE_IMPORTED = True
except ImportError as e:
    # Use print here as local logger isn't set up yet
    print(f"ERROR: Could not import ColorManager from LACE.colors: {e}. Highlighting will be disabled.", file=sys.stderr)
    ColorManager = None # type: ignore
    ColorScheme = None # type: ignore
    COLOR_MODULE_IMPORTED = False

# --- Taichi Script Specific Settings ---
APP_NAME = "taichi_advanced_render" # Specific App Name for logs
RULE_DENSITY_CONFIG_FILENAME = "rule_densities.json" # Config file for per-rule densities

# --- Logging Configuration (Local to this script) ---
class LogSettings:
    class Logging:
        LOG_LEVEL: str = "DEBUG" # Set desired level (DEBUG, INFO, DETAIL, WARNING, ERROR)

# --- Custom Log Level ---
DETAIL_LEVEL_NUM = 15
logging.addLevelName(DETAIL_LEVEL_NUM, "DETAIL")
def detail(self, message, *args, **kws):
    if self.isEnabledFor(DETAIL_LEVEL_NUM):
        self._log(DETAIL_LEVEL_NUM, message, args, **kws)
logging.Logger.detail = detail # type: ignore [attr-defined]

# --- Directory Setup (Local to this script) ---
TAICHI_APP_DIR = "LACE_Taichi_Demo" # Directory name for this app's files
TAICHI_SUBDIRS = {
    'logs': 'logs',
    'config': 'config',
    # Add other subdirs if this script needs them (e.g., 'saves')
}

def setup_taichi_directories() -> Tuple[dict, str]:
    """Sets up the necessary directories for THIS Taichi application."""
    try:
        # Base path is the directory containing this script
        base_path = os.path.dirname(os.path.abspath(__file__))
        # Assuming script is inside TAICHI_APP_DIR
        app_base_path = base_path

        paths = {}
        for key, subdir in TAICHI_SUBDIRS.items():
            path = os.path.join(app_base_path, subdir)
            os.makedirs(path, exist_ok=True)
            paths[key] = path
        # Add app_base_path to the returned dict for convenience
        paths['base'] = app_base_path
        return paths, app_base_path
    except Exception as e:
        print(f"Fatal error in Taichi directory setup: {str(e)}")
        raise SystemExit(1)

def setup_taichi_logging(log_dir: str, app_name: str) -> logging.Logger:
    """Setup basic logging for THIS Taichi script."""
    try:
        logger_name = app_name # Use the app name for the logger instance
        logger = logging.getLogger(logger_name)
        if logger.handlers:
             print(f"Logger '{logger_name}' already has handlers. Skipping setup.")
             return logger # Avoid adding duplicate handlers

        timestamp_24hr = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_level_str = LogSettings.Logging.LOG_LEVEL.upper()
        file_log_level = getattr(logging, log_level_str, logging.INFO)
        if log_level_str == "DETAIL":
            file_log_level = DETAIL_LEVEL_NUM

        console_log_level = logging.DEBUG # Keep console verbose for debugging Taichi
        main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

        # --- Use app_name in the filename ---
        main_log_filename = f'{app_name}_{timestamp_24hr}.log'
        # ---
        main_file_handler = logging.FileHandler(os.path.join(log_dir, main_log_filename))
        main_file_handler.setFormatter(main_formatter)
        main_file_handler.setLevel(file_log_level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(main_formatter)
        console_handler.setLevel(console_log_level)

        # Configure THIS logger instance
        logger.setLevel(min(file_log_level, console_log_level))
        logger.addHandler(main_file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False # Prevent messages going to root logger

        file_log_level_name = logging.getLevelName(file_log_level)
        console_log_level_name = logging.getLevelName(console_log_level)
        # Use logger directly now that it's configured
        logger.info(f"Taichi script logging initialized. Logger Level: {logging.getLevelName(logger.level)}, File Handler Level: {file_log_level_name}, Console Handler Level: {console_log_level_name}")
        logger.info(f"Log file: {os.path.join(log_dir, main_log_filename)}")
        return logger

    except Exception as e:
        # Fallback if specific setup fails
        print(f"Error setting up Taichi logging: {e}")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(app_name + "_fallback")
        logger.error(f"Logging setup failed: {e}. Using basicConfig.")
        return logger

# --- Initialize Taichi Script Directories and Logger ---
TAICHI_APP_PATHS, TAICHI_BASE_PATH = setup_taichi_directories()
logger = setup_taichi_logging(TAICHI_APP_PATHS['logs'], APP_NAME) # Pass APP_NAME
# --- End Taichi Logging Setup ---

# --- Call ti.init AFTER local logging setup ---
try:
    ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_fraction=0.9)
    logger.info("Taichi initialized successfully.")
except Exception as e:
    logger.critical(f"Taichi initialization failed: {e}", exc_info=True)
    raise SystemExit("Taichi failed to initialize. Exiting.")

# --- JSON Loading Function Definitions (Defined BEFORE use) ---
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

def load_rule_densities():
    """Loads saved initial densities for rules from config file."""
    config_path = os.path.join(TAICHI_APP_PATHS.get('config', '.'), RULE_DENSITY_CONFIG_FILENAME)
    if not os.path.exists(config_path):
        logger.info(f"Rule density config file not found at {config_path}. Using defaults.")
        return {}
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} rule density settings from {config_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding rule density config from {config_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading rule density config from {config_path}: {e}")
        return {}

def save_rule_densities(density_dict):
    """Saves initial densities for rules to config file."""
    config_path = os.path.join(TAICHI_APP_PATHS.get('config', '.'), RULE_DENSITY_CONFIG_FILENAME)
    try:
        with open(config_path, 'w') as f:
            json.dump(density_dict, f, indent=2)
        logger.info(f"Saved {len(density_dict)} rule density settings to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving rule density config to {config_path}: {e}")
        return False

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
        # Add elif blocks here for any other rule types from rules.json
        # elif rule_type == "SomeOtherRuleType":
        #     # Parse params for SomeOtherRuleType
        #     pass

    logger.info(f"--- Finished JSON Rule Parsing ---")
    return variants

# --- JSON Loading and Rule Variant Processing (Execution) ---
# Construct path using THIS script's TAICHI_APP_PATHS
try:
    rules_dir = TAICHI_APP_PATHS.get('config')
    if not rules_dir:
        logger.error("Local config directory path not found in TAICHI_APP_PATHS. Using fallback.")
        rules_dir = os.path.join(TAICHI_BASE_PATH, 'config')

    # Check if a 'rules.json' exists in the local config directory
    local_rules_path = os.path.join(rules_dir, 'rules.json')
    if os.path.exists(local_rules_path):
        JSON_FILE_PATH = local_rules_path
        logger.info(f"Using local rules file path: {JSON_FILE_PATH}")
    else:
        # If not found locally, fall back to the main LACE rules file
        logger.warning(f"Local rules file not found at {local_rules_path}. Falling back to main LACE rules file.")
        # Construct path to main LACE rules using BASE_PATH from logging_config import (if successful)
        try:
            from LACE.logging_config import BASE_PATH as LACE_BASE_PATH, APP_PATHS as LACE_APP_PATHS
            main_rules_dir_key = 'config_rules'
            if main_rules_dir_key not in LACE_APP_PATHS:
                main_rules_dir_key = 'config'
            main_rules_dir_path = LACE_APP_PATHS.get(main_rules_dir_key)
            if main_rules_dir_path and not os.path.isabs(main_rules_dir_path):
                 resources_dir = os.path.join(LACE_BASE_PATH, "Resources")
                 main_rules_dir = os.path.join(resources_dir, main_rules_dir_path)
            elif main_rules_dir_path:
                 main_rules_dir = main_rules_dir_path
            else:
                 main_rules_dir = os.path.join(LACE_BASE_PATH, 'Resources', 'config', 'rules')
            JSON_FILE_PATH = os.path.join(main_rules_dir, 'rules.json')
            logger.info(f"Using main LACE rules file path: {JSON_FILE_PATH}")
        except Exception as lace_path_err:
            logger.error(f"Error constructing main LACE rules path: {lace_path_err}. Using hardcoded fallback.")
            JSON_FILE_PATH = '/Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/LACE - NEW - In progress/LACE/Resources/config/rules/rules.json'

except Exception as e:
    logger.error(f"Error determining JSON_FILE_PATH: {e}. Using hardcoded fallback path.")
    JSON_FILE_PATH = '/Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/LACE - NEW - In progress/LACE/Resources/config/rules/rules.json'

# Initialize empty lists first
rule_names = []
rule_keys = []
rule_params = []
RULE_VARIANTS = []

# Load rules and create variants list
all_json_rules = load_rules_from_json(JSON_FILE_PATH) # Functions are now defined above
RULE_VARIANTS = parse_rule_variants(all_json_rules) # Functions are now defined above

# Load saved density settings for rules
rule_density_map = load_rule_densities()  # Load saved densities

# --- Instantiate Color Manager ---
# NOTE: ColorManager will still log to the main LACE log file
# because it imports LACE.logging_config internally.
color_manager = None # Initialize
if COLOR_MODULE_IMPORTED and ColorManager:
    try:
        # We need APP_PATHS from the main app for ColorManager config path finding
        try:
            from LACE.logging_config import APP_PATHS as LACE_APP_PATHS
            color_manager = ColorManager(LACE_APP_PATHS) # Pass main app's paths
            logger.info("ColorManager initialized successfully (using its internal logger).")
        except ImportError as lace_path_e:
             logger.error(f"Failed to import LACE_APP_PATHS for ColorManager: {lace_path_e}. ColorManager may fail.")
             COLOR_MODULE_IMPORTED = False # Treat as failure if paths missing
        except Exception as cm_e:
             logger.error(f"Failed to initialize ColorManager: {cm_e}", exc_info=True)
             color_manager = None
             COLOR_MODULE_IMPORTED = False

    except Exception as e: # General catch just in case
        logger.error(f"Unexpected error during ColorManager setup: {e}", exc_info=True)
        color_manager = None
        COLOR_MODULE_IMPORTED = False
# Ensure color_manager is None if import/init failed
if not COLOR_MODULE_IMPORTED:
    color_manager = None

# --- Highlighting Settings ---
HIGHLIGHT_CHANGES = False # Toggle for highlighting

# --- Global Settings ---
GRID_SIZE = 1000
WINDOW_SIZE = 1000

# --- Shared Parameters ---
WRAP = True
INITIAL_DENSITY = 0.10
DEFAULT_INITIAL_DENSITY = 0.10  # Fallback default
STEP_DELAY = 1/60
paused = False
speed_slider_value = 60.0

# --- Per-Rule Density Storage ---
rule_density_map: Dict[str, float] = {}  # Maps rule name -> initial density

# --- Rule Selection State ---
# Fallback definition moved here, after RULE_VARIANTS is populated or fails
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
    # Repopulate lists if fallback was used
    rule_names = [r[0] for r in RULE_VARIANTS]
    rule_keys = [r[1] for r in RULE_VARIANTS]
    rule_params = [r[2] for r in RULE_VARIANTS]
else:
    # Populate lists from successfully loaded RULE_VARIANTS
    rule_names = [r[0] for r in RULE_VARIANTS]
    rule_keys = [r[1] for r in RULE_VARIANTS]
    rule_params = [r[2] for r in RULE_VARIANTS]

# --- Find the index of the rule matching the original hardcoded params ---
initial_rule_name_to_find = "Realm of Lace_Fancy Interesting Shapes"
selected_rule_idx = 0 # Default to 0
try:
    # Ensure rule_names is populated before searching
    if rule_names:
        selected_rule_idx = next(i for i, name in enumerate(rule_names) if name == initial_rule_name_to_find)
        logger.info(f"Found initial rule '{rule_names[selected_rule_idx]}' at index {selected_rule_idx} in RULE_VARIANTS list.")
    else:
        logger.warning("Rule names list is empty. Cannot find initial rule. Defaulting to index 0.")
        selected_rule_idx = 0
except StopIteration:
    # Check if rule_names is not empty before logging the default
    default_name = rule_names[0] if rule_names else "Unknown Fallback"
    logger.warning(f"Could not find rule named '{initial_rule_name_to_find}'. Defaulting to index 0 ({default_name}).")
    selected_rule_idx = 0 # Fallback to the first rule if not found
except IndexError:
     logger.warning(f"RULE_VARIANTS list seems empty or invalid after loading. Defaulting rule index to 0.")
     selected_rule_idx = 0

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
BASE_EDGE_WIDTH = 1.5  # Base edge width in pixels *at zoom level 1*
EDGE_COLOR = (0.6, 0.6, 0.6) # Default edge color if not using ColorManager

# --- Settings for conditional/scaled rendering ---
MIN_NODE_SEPARATION_PIXELS = BASE_NODE_RADIUS * 2 * 1.2
MAX_SEP_FOR_NODE_SCALING = MIN_NODE_SEPARATION_PIXELS * 3.0

# --- Fields for Drawing Primitives ---
MAX_DRAW_NODES = 50000
MAX_DRAW_LINES = 100000 # Max for EACH category (default/highlight)

# --- Fields for Direct Kernel Collection ---
draw_node_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_DRAW_NODES)
draw_node_outline_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_DRAW_NODES) # For outline color

# Separate fields for default and highlighted edges
draw_default_edge_endpoints = ti.Vector.field(2, dtype=ti.f32, shape=MAX_DRAW_LINES * 2)
draw_highlight_edge_endpoints = ti.Vector.field(2, dtype=ti.f32, shape=MAX_DRAW_LINES * 2)

# Scalar fields to track actual number of items to draw
num_nodes_to_draw = ti.field(dtype=ti.i32, shape=())
num_default_edges_to_draw = ti.field(dtype=ti.i32, shape=())
num_highlight_edges_to_draw = ti.field(dtype=ti.i32, shape=())

# --- Taichi Fields ---
node_state = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
node_next_state = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
node_active_neighbors = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
node_eligible = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
node_next_eligible = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
node_degree = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
node_next_degree = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))

# --- Added Fields for State Tracking (Highlighting) ---
previous_node_state = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
previous_node_degree = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))

# --- Debugging Field ---
debug_node_changed_flag = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))

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

# --- Debug Print for Field Shape ---
logger.info(f"DEBUG: node_state field shape after definition: {node_state.shape}")
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
    """Sets the global parameters AND copies them into Taichi fields. Also loads saved density if available."""
    global BIRTH_SUM_RANGES, SURVIVAL_SUM_RANGES, FINAL_DEATH_DEGREES, FINAL_DEATH_DEGREE_RANGES
    global GOL_BIRTH, GOL_SURVIVAL
    global NODE_COLORMAP, NODE_COLOR_NORM_VMIN, NODE_COLOR_NORM_VMAX
    global COLOR_BY_DEGREE, COLOR_BY_ACTIVE_NEIGHBORS
    global INITIAL_DENSITY

    if idx < 0 or idx >= len(rule_params):
        logger.error(f"Invalid rule index {idx}")
        idx = 0

    params = rule_params[idx]
    rule_key = rule_keys[idx]
    rule_name = rule_names[idx]

    logger.info(f"--- Setting Variant {idx}: {rule_name} (Key: {rule_key}) ---")

    clear_param_fields()
    
    # Load saved density for this rule if available
    if rule_name in rule_density_map:
        INITIAL_DENSITY = rule_density_map[rule_name]
        logger.info(f"  Loaded saved initial density for '{rule_name}': {INITIAL_DENSITY:.3f}")
    else:
        INITIAL_DENSITY = DEFAULT_INITIAL_DENSITY
        logger.info(f"  No saved density for '{rule_name}', using default: {INITIAL_DENSITY:.3f}")

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

# Initialize with the selected (or default) rule (this will also load saved density)
set_rule_variant(selected_rule_idx)
logger.info(f"Initial rule set to: {rule_names[selected_rule_idx]} with density: {INITIAL_DENSITY:.3f}")

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

@ti.kernel
def initialize_kernel(density: float):
    """Initializes all relevant fields based on random density."""
    for i, j in node_state:
        is_active = ti.random() < density
        current_state = 1 if is_active else 0
        node_state[i, j] = current_state
        node_next_state[i, j] = 0
        node_active_neighbors[i, j] = 0
        node_eligible[i, j] = 0
        node_next_eligible[i, j] = 0
        node_degree[i, j] = current_state
        node_next_degree[i, j] = 0
        previous_node_state[i, j] = current_state
        previous_node_degree[i, j] = current_state


def initialize():
    """Calls the initialization kernel and prepares for the selected rule."""
    logger.info("Initializing grid...")
    initialize_kernel(INITIAL_DENSITY)
    if rule_keys[selected_rule_idx] == "realm_of_lace":
         logger.info("RoL rule selected, copying initial state to degree.")
         node_degree.copy_from(node_state)
    logger.info("Initialization complete.")

@ti.kernel # test kernel for debugging grid size
def populate_draw_buffers_for_debug(num_to_populate: int, color_vec: ti.types.vector(3, ti.f32)): # type: ignore
    """Populates the first 'num_to_populate' elements of draw buffers with fixed test data."""
    # Ensure we don't exceed buffer limits
    actual_num = ti.min(num_to_populate, MAX_DRAW_NODES)
    num_nodes_to_draw[None] = actual_num

    # Simple grid layout within the window for testing
    cols = ti.cast(ti.sqrt(ti.cast(actual_num, ti.f32)), ti.i32)
    if cols == 0:
        cols = 1
    rows = (actual_num + cols - 1) // cols

    spacing_x = 1.0 / ti.cast(cols + 1, ti.f32)
    spacing_y = 1.0 / ti.cast(rows + 1, ti.f32)

    for i in range(actual_num):
        col_idx = i % cols
        row_idx = i // cols
        pos_x = (ti.cast(col_idx, ti.f32) + 1.0) * spacing_x
        pos_y = (ti.cast(row_idx, ti.f32) + 1.0) * spacing_y
        draw_node_pos[i] = ti.Vector([pos_x, pos_y])
        draw_node_outline_colors[i] = color_vec

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
    global zoom_slider, zoom, pan_x, pan_y
    global INITIAL_DENSITY, STEP_DELAY, paused, frame, speed_slider_value
    global selected_rule_idx
    global RENDER_EDGES
    global HIGHLIGHT_CHANGES # Add global toggle
    global color_manager # Access color manager

    control_height = 0.40 # Increased height slightly for new checkbox

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
        clear_render_data_kernel() # Clear drawing buffers
        frame = 0
        logger.info("Rule changed, grid re-initialized, render data cleared.")

    # --- Visualization Style ---
    gui.text("Visualization Style")
    render_edges_checkbox_value = gui.checkbox("Render Nodes & Edges", RENDER_EDGES)
    if render_edges_checkbox_value != RENDER_EDGES:
        logger.info(f"Checkbox value changed! Old RENDER_EDGES={RENDER_EDGES}, New Checkbox Value={render_edges_checkbox_value}")
        RENDER_EDGES = render_edges_checkbox_value
        logger.info(f"Global RENDER_EDGES updated to: {RENDER_EDGES}")

    # --- Highlighting Toggle ---
    # Check if color module loaded AND color_manager was successfully initialized
    if COLOR_MODULE_IMPORTED and color_manager:
        highlight_checkbox_value = gui.checkbox("Highlight Changes", HIGHLIGHT_CHANGES)
        if highlight_checkbox_value != HIGHLIGHT_CHANGES:
            HIGHLIGHT_CHANGES = highlight_checkbox_value
            logger.info(f"Highlight Changes toggled to: {HIGHLIGHT_CHANGES}")
    else:
        # Optionally display a disabled checkbox or just text
        # gui.checkbox("Highlight Changes", False) # Disabled appearance
        gui.text("Highlighting Disabled")

    # --- Zoom Buttons ---
    if gui.button("Zoom to Edges"):
        target_zoom = MIN_NODE_SEPARATION_PIXELS * GRID_SIZE / WINDOW_SIZE
        target_zoom = max(3.0, target_zoom) # Ensure minimum zoom for node visibility
        logger.info(f"Zoom to Edges pressed. Setting Zoom: {target_zoom:.2f}. Resetting pan.")
        zoom = target_zoom
        zoom_slider = zoom
        pan_x = 0.0 # Center pan
        pan_y = 0.0 # Center pan
        if not RENDER_EDGES:
            RENDER_EDGES = True
            logger.info("Enabled Render Nodes & Edges.")

    if gui.button("Zoom Out Full"):
        logger.info("Zoom Out Full pressed.")
        zoom = 1.0
        zoom_slider = zoom
        pan_x = 0.0 # Reset pan when zooming fully out
        pan_y = 0.0

    # --- Zoom Control ---
    gui.text("Zoom Control")
    zoom_slider = gui.slider_float("Zoom", zoom_slider, 1.0, 40.0)

    # --- Simulation Controls ---
    gui.text("Simulation Controls")
    INITIAL_DENSITY = gui.slider_float("Init Density", INITIAL_DENSITY, 0.01, 1.0)
    
    # Save Init Density button
    if gui.button("Save Init Density"):
        current_rule_name = rule_names[selected_rule_idx]
        rule_density_map[current_rule_name] = INITIAL_DENSITY
        if save_rule_densities(rule_density_map):
            logger.info(f"Saved density {INITIAL_DENSITY:.3f} for rule '{current_rule_name}'")
        else:
            logger.error(f"Failed to save density for rule '{current_rule_name}'")
    
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
    # Check if zoom slider value changed significantly
    if abs(zoom - zoom_slider) > 1e-4: # Add a tolerance
        logger.debug(f"Zoom slider changed: {zoom:.4f} -> {zoom_slider:.4f}")
        zoom = zoom_slider
        # Potential place to force clear/redraw if needed, but let's try clearing in render func first
    # Pan state is now only updated by handle_mouse or the zoom buttons
    #  
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
def clear_render_data_kernel():
    """Clears the Taichi fields used for drawing nodes and edges."""
    num_nodes_to_draw[None] = 0
    num_default_edges_to_draw[None] = 0
    num_highlight_edges_to_draw[None] = 0

    # --- Explicitly clear arrays to off-screen/default values ---
    off_screen = ti.Vector([-10.0, -10.0]) # Use a value far off-screen
    default_color = ti.Vector([0.0, 0.0, 0.0]) # Black or any default

    for i in range(MAX_DRAW_NODES):
        draw_node_pos[i] = off_screen
        draw_node_outline_colors[i] = default_color
    for i in range(MAX_DRAW_LINES * 2):
        draw_default_edge_endpoints[i] = off_screen
        draw_highlight_edge_endpoints[i] = off_screen

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
def collect_gol_render_data(
    current_zoom: float, current_pan_x: float, current_pan_y: float,
    node_radius_px: float,
    highlight_changes_flag: bool,
    node_base_color_v: ti.types.vector(3, ti.f32), # type: ignore
    node_outline_old_color_v: ti.types.vector(3, ti.f32), # type: ignore
    node_outline_new_color_v: ti.types.vector(3, ti.f32), # type: ignore
):
    """Collects visible node and edge data for Game of Life, conditionally handling highlighting."""
    num_nodes_to_draw[None] = 0
    num_default_edges_to_draw[None] = 0
    num_highlight_edges_to_draw[None] = 0

    highlight_changes = ti.cast(highlight_changes_flag, ti.i32) # Cast flag once

    for j, i in node_state:
        current_state = node_state[j, i]
        is_active = current_state > 0

        if is_active:
            cx, cy = map_grid_to_canvas_kernel(i, j, current_zoom, current_pan_x, current_pan_y)
            margin = node_radius_px * 1.5
            is_visible1 = -margin < cx < WINDOW_SIZE + margin and -margin < cy < WINDOW_SIZE + margin

            if is_visible1:
                # --- Determine Node Highlight Status (Conditional) ---
                node_is_highlighted = False # Default
                node_state_changed = False # Default
                if highlight_changes > 0: # Only check state if highlighting is ON
                    previous_state = previous_node_state[j, i]
                    node_state_changed = (previous_state != current_state)
                    if node_state_changed:
                        node_is_highlighted = True

                # --- Add Node ---
                current_node_count = ti.cast(num_nodes_to_draw[None], ti.i32)
                if current_node_count < MAX_DRAW_NODES: # type: ignore
                    draw_idx = ti.atomic_add(num_nodes_to_draw[None], 1)
                    if draw_idx < MAX_DRAW_NODES: # type: ignore
                        draw_node_pos[draw_idx] = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                        outline_color = node_outline_old_color_v
                        if node_is_highlighted:
                            outline_color = node_outline_new_color_v
                        draw_node_outline_colors[draw_idx] = outline_color

                # --- Check Neighbors for Edges (Conditional Highlighting) ---
                for k in ti.static(range(NUM_NEIGHBORS)):
                    di, dj = neighbor_offsets[k]
                    ni_raw, nj_raw = i + di, j + dj
                    neighbor_out_of_bounds = not (0 <= ni_raw < GRID_SIZE and 0 <= nj_raw < GRID_SIZE)

                    if not (WRAP and neighbor_out_of_bounds):
                        ni = wrap_idx(ni_raw, GRID_SIZE)
                        nj = wrap_idx(nj_raw, GRID_SIZE)

                        neighbor_state = node_state[nj, ni]
                        if neighbor_state > 0: # Check neighbor activity
                            idx1 = j * GRID_SIZE + i
                            idx2 = nj * GRID_SIZE + ni
                            if idx1 < idx2: # Avoid duplicate edges
                                # --- Determine Edge Highlight Status (Conditional) ---
                                edge_is_highlighted = False # Default
                                if highlight_changes > 0: # Only check state if highlighting is ON
                                    prev_neighbor_state = previous_node_state[nj, ni]
                                    neighbor_state_changed = (prev_neighbor_state != neighbor_state)
                                    if node_state_changed or neighbor_state_changed:
                                        edge_is_highlighted = True

                                # --- Add Edge to Appropriate List ---
                                cx2_raw, cy2_raw = map_grid_to_canvas_kernel(ni_raw, nj_raw, current_zoom, current_pan_x, current_pan_y)
                                endpoint1 = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                                endpoint2 = ti.Vector([cx2_raw / WINDOW_SIZE, cy2_raw / WINDOW_SIZE])

                                if edge_is_highlighted:
                                    current_edge_count = ti.cast(num_highlight_edges_to_draw[None], ti.i32)
                                    if current_edge_count < MAX_DRAW_LINES: # type: ignore
                                        draw_idx_base = ti.atomic_add(num_highlight_edges_to_draw[None], 1) * 2 # type: ignore
                                        if ti.cast(draw_idx_base, ti.i32) + ti.i32(1) < MAX_DRAW_LINES * 2: # type: ignore
                                            draw_highlight_edge_endpoints[draw_idx_base] = endpoint1
                                            draw_highlight_edge_endpoints[draw_idx_base + ti.i32(1)] = endpoint2
                                else: # Default edge
                                    current_edge_count = ti.cast(num_default_edges_to_draw[None], ti.i32)
                                    if current_edge_count < MAX_DRAW_LINES: # type: ignore
                                        draw_idx_base = ti.atomic_add(num_default_edges_to_draw[None], 1) * 2 # type: ignore
                                        if ti.cast(draw_idx_base, ti.i32) + ti.i32(1) < MAX_DRAW_LINES * 2: # type: ignore
                                            draw_default_edge_endpoints[draw_idx_base] = endpoint1
                                            draw_default_edge_endpoints[draw_idx_base + ti.i32(1)] = endpoint2

@ti.kernel
def collect_rol_render_data(
    current_zoom: float, current_pan_x: float, current_pan_y: float,
    node_radius_px: float,
    highlight_changes_flag: bool,
    node_base_color_v: ti.types.vector(3, ti.f32), # type: ignore
    node_outline_old_color_v: ti.types.vector(3, ti.f32), # type: ignore
    node_outline_new_color_v: ti.types.vector(3, ti.f32), # type: ignore
):
    """Collects visible node and edge data for Realm of Lace, conditionally handling highlighting."""
    num_nodes_to_draw[None] = 0
    num_default_edges_to_draw[None] = 0
    num_highlight_edges_to_draw[None] = 0

    highlight_changes = ti.cast(highlight_changes_flag, ti.i32) # Cast flag once

    for j, i in node_degree:
        current_degree = node_degree[j, i]
        is_active = current_degree > 0

        if is_active:
            cx, cy = map_grid_to_canvas_kernel(i, j, current_zoom, current_pan_x, current_pan_y)
            margin = node_radius_px * 1.5
            is_visible1 = -margin < cx < WINDOW_SIZE + margin and -margin < cy < WINDOW_SIZE + margin

            if is_visible1:
                # --- Determine Node Highlight Status (Conditional) ---
                node_is_highlighted = False # Default to not highlighted
                node_state_changed = False # Default state change flag
                if highlight_changes > 0: # Only check state if highlighting is ON
                    previous_degree = previous_node_degree[j, i]
                    node_state_changed = (previous_degree != current_degree)
                    if node_state_changed:
                        node_is_highlighted = True # Set flag if state actually changed

                # --- Add Node ---
                current_node_count = ti.cast(num_nodes_to_draw[None], ti.i32)
                if current_node_count < MAX_DRAW_NODES: # type: ignore
                    draw_idx = ti.atomic_add(num_nodes_to_draw[None], 1)
                    if draw_idx < MAX_DRAW_NODES: # type: ignore
                        draw_node_pos[draw_idx] = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                        # Select outline color based on calculated highlight status
                        outline_color = node_outline_old_color_v
                        if node_is_highlighted: # Check the flag set above
                            outline_color = node_outline_new_color_v
                        draw_node_outline_colors[draw_idx] = outline_color

                # --- Check Neighbors for Edges (Conditional Highlighting) ---
                for k in ti.static(range(NUM_NEIGHBORS)):
                    di, dj = neighbor_offsets[k]
                    ni_raw, nj_raw = i + di, j + dj
                    neighbor_out_of_bounds = not (0 <= ni_raw < GRID_SIZE and 0 <= nj_raw < GRID_SIZE)

                    if not (WRAP and neighbor_out_of_bounds):
                        ni = wrap_idx(ni_raw, GRID_SIZE)
                        nj = wrap_idx(nj_raw, GRID_SIZE)

                        neighbor_degree = node_degree[nj, ni]
                        if neighbor_degree > 0: # Check neighbor activity
                            idx1 = j * GRID_SIZE + i
                            idx2 = nj * GRID_SIZE + ni
                            if idx1 < idx2: # Avoid duplicate edges
                                # --- Determine Edge Highlight Status (Conditional) ---
                                edge_is_highlighted = False # Default to not highlighted
                                if highlight_changes > 0: # Only check state if highlighting is ON
                                    prev_neighbor_degree = previous_node_degree[nj, ni]
                                    neighbor_state_changed = (prev_neighbor_degree != neighbor_degree)
                                    # Highlight if highlight is ON AND (this node changed OR neighbor changed)
                                    if node_state_changed or neighbor_state_changed:
                                        edge_is_highlighted = True

                                # --- Add Edge to Appropriate List ---
                                cx2_raw, cy2_raw = map_grid_to_canvas_kernel(ni_raw, nj_raw, current_zoom, current_pan_x, current_pan_y)
                                endpoint1 = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                                endpoint2 = ti.Vector([cx2_raw / WINDOW_SIZE, cy2_raw / WINDOW_SIZE])

                                if edge_is_highlighted: # Check the flag set above
                                    current_edge_count = ti.cast(num_highlight_edges_to_draw[None], ti.i32)
                                    if current_edge_count < MAX_DRAW_LINES: # type: ignore
                                        draw_idx_base = ti.atomic_add(num_highlight_edges_to_draw[None], 1) * 2 # type: ignore
                                        if ti.cast(draw_idx_base, ti.i32) + ti.i32(1) < MAX_DRAW_LINES * 2: # type: ignore
                                            draw_highlight_edge_endpoints[draw_idx_base] = endpoint1
                                            draw_highlight_edge_endpoints[draw_idx_base + ti.i32(1)] = endpoint2
                                else: # Default edge (if highlighting is OFF or edge didn't change)
                                    current_edge_count = ti.cast(num_default_edges_to_draw[None], ti.i32)
                                    if current_edge_count < MAX_DRAW_LINES: # type: ignore
                                        draw_idx_base = ti.atomic_add(num_default_edges_to_draw[None], 1) * 2 # type: ignore
                                        if ti.cast(draw_idx_base, ti.i32) + ti.i32(1) < MAX_DRAW_LINES * 2: # type: ignore
                                            draw_default_edge_endpoints[draw_idx_base] = endpoint1
                                            draw_default_edge_endpoints[draw_idx_base + ti.i32(1)] = endpoint2

@ti.kernel
def collect_colored_life_render_data(
    current_zoom: float, current_pan_x: float, current_pan_y: float,
    node_radius_px: float,
    highlight_changes_flag: bool,
    node_base_color_v: ti.types.vector(3, ti.f32), # type: ignore
    node_outline_old_color_v: ti.types.vector(3, ti.f32), # type: ignore
    node_outline_new_color_v: ti.types.vector(3, ti.f32), # type: ignore
):
    """Collects visible node and edge data for Colored Life, conditionally handling highlighting."""
    num_nodes_to_draw[None] = 0
    num_default_edges_to_draw[None] = 0
    num_highlight_edges_to_draw[None] = 0

    highlight_changes = ti.cast(highlight_changes_flag, ti.i32) # Cast flag once

    for j, i in node_state:
        current_state = node_state[j, i]
        is_active = current_state > 0

        if is_active:
            cx, cy = map_grid_to_canvas_kernel(i, j, current_zoom, current_pan_x, current_pan_y)
            margin = node_radius_px * 1.5
            is_visible1 = -margin < cx < WINDOW_SIZE + margin and -margin < cy < WINDOW_SIZE + margin

            if is_visible1:
                # --- Determine Node Highlight Status (Conditional) ---
                node_is_highlighted = False # Default
                node_state_changed = False # Default
                if highlight_changes > 0: # Only check state if highlighting is ON
                    previous_state = previous_node_state[j, i]
                    node_state_changed = (previous_state != current_state)
                    if node_state_changed:
                        node_is_highlighted = True

                # --- Add Node ---
                current_node_count = ti.cast(num_nodes_to_draw[None], ti.i32)
                if current_node_count < MAX_DRAW_NODES: # type: ignore
                    draw_idx = ti.atomic_add(num_nodes_to_draw[None], 1)
                    if draw_idx < MAX_DRAW_NODES: # type: ignore
                        draw_node_pos[draw_idx] = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                        outline_color = node_outline_old_color_v
                        if node_is_highlighted:
                            outline_color = node_outline_new_color_v
                        draw_node_outline_colors[draw_idx] = outline_color

                # --- Check Neighbors for Edges (Conditional Highlighting) ---
                for k in ti.static(range(NUM_NEIGHBORS)):
                    di, dj = neighbor_offsets[k]
                    ni_raw, nj_raw = i + di, j + dj
                    neighbor_out_of_bounds = not (0 <= ni_raw < GRID_SIZE and 0 <= nj_raw < GRID_SIZE)

                    if not (WRAP and neighbor_out_of_bounds):
                        ni = wrap_idx(ni_raw, GRID_SIZE)
                        nj = wrap_idx(nj_raw, GRID_SIZE)

                        neighbor_state = node_state[nj, ni]
                        if neighbor_state > 0: # Check neighbor activity
                            idx1 = j * GRID_SIZE + i
                            idx2 = nj * GRID_SIZE + ni
                            if idx1 < idx2: # Avoid duplicate edges
                                # --- Determine Edge Highlight Status (Conditional) ---
                                edge_is_highlighted = False # Default
                                if highlight_changes > 0: # Only check state if highlighting is ON
                                    prev_neighbor_state = previous_node_state[nj, ni]
                                    neighbor_state_changed = (prev_neighbor_state != neighbor_state)
                                    if node_state_changed or neighbor_state_changed:
                                        edge_is_highlighted = True

                                # --- Add Edge to Appropriate List ---
                                cx2_raw, cy2_raw = map_grid_to_canvas_kernel(ni_raw, nj_raw, current_zoom, current_pan_x, current_pan_y)
                                endpoint1 = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                                endpoint2 = ti.Vector([cx2_raw / WINDOW_SIZE, cy2_raw / WINDOW_SIZE])

                                if edge_is_highlighted:
                                    current_edge_count = ti.cast(num_highlight_edges_to_draw[None], ti.i32)
                                    if current_edge_count < MAX_DRAW_LINES: # type: ignore
                                        draw_idx_base = ti.atomic_add(num_highlight_edges_to_draw[None], 1) * 2 # type: ignore
                                        if ti.cast(draw_idx_base, ti.i32) + ti.i32(1) < MAX_DRAW_LINES * 2: # type: ignore
                                            draw_highlight_edge_endpoints[draw_idx_base] = endpoint1
                                            draw_highlight_edge_endpoints[draw_idx_base + ti.i32(1)] = endpoint2
                                else: # Default edge
                                    current_edge_count = ti.cast(num_default_edges_to_draw[None], ti.i32)
                                    if current_edge_count < MAX_DRAW_LINES: # type: ignore
                                        draw_idx_base = ti.atomic_add(num_default_edges_to_draw[None], 1) * 2 # type: ignore
                                        if ti.cast(draw_idx_base, ti.i32) + ti.i32(1) < MAX_DRAW_LINES * 2: # type: ignore
                                            draw_default_edge_endpoints[draw_idx_base] = endpoint1
                                            draw_default_edge_endpoints[draw_idx_base + ti.i32(1)] = endpoint2
                                            
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

# # Modify function signature to accept calculated sizes
# def render_nodes_and_edges(current_node_radius_px, current_edge_width_px):
#     """Renders nodes as circles with outlines and edges as lines directly onto the canvas using Taichi fields, supporting highlighting."""
#     global color_manager # Ensure access to the color manager

#     # --- Force Clear Render Buffers ---
#     # Call this at the beginning to prevent artifacts from previous frames/zoom levels
#     clear_render_data_kernel()
#     # ---

#     rule_key = rule_keys[selected_rule_idx]

#     # --- Get Current Color Scheme ---
#     if not (COLOR_MODULE_IMPORTED and color_manager and color_manager.current_scheme):
#         logger.warning("ColorManager not available or no scheme set. Skipping node/edge render.")
#         # Buffers already cleared above
#         return

#     scheme = color_manager.current_scheme
#     try:
#         # Convert hex colors to Taichi-compatible RGB tuples (0-1)
#         bg_color_rgb = to_rgb(scheme.background) # Not used directly here, but good to have
#         node_base_rgb = to_rgb(scheme.node_base)
#         node_outline_old_rgb = to_rgb(scheme.node)
#         node_outline_new_rgb = to_rgb(scheme.new_node)
#         default_edge_rgb = to_rgb(scheme.default_edge)
#         new_edge_rgb = to_rgb(scheme.new_edge)

#         # Convert tuples to Taichi vectors
#         node_base_color_v = ti.Vector(node_base_rgb)
#         node_outline_old_color_v = ti.Vector(node_outline_old_rgb)
#         node_outline_new_color_v = ti.Vector(node_outline_new_rgb)

#     except Exception as e:
#         logger.error(f"Error processing colors from scheme '{scheme.name}': {e}. Skipping render.")
#         # Buffers already cleared above
#         return

#     # --- Call Appropriate Data Collection Kernel ---
#     # Pass necessary parameters including colors and highlight flag
#     # Kernels will now use the most up-to-date zoom/pan values for this frame
#     if rule_key == "realm_of_lace":
#         collect_rol_render_data(zoom, pan_x, pan_y, current_node_radius_px,
#                                 HIGHLIGHT_CHANGES, node_base_color_v,
#                                 node_outline_old_color_v, node_outline_new_color_v)
#     elif rule_key == "game_of_life":
#         collect_gol_render_data(zoom, pan_x, pan_y, current_node_radius_px,
#                                 HIGHLIGHT_CHANGES, node_base_color_v,
#                                 node_outline_old_color_v, node_outline_new_color_v)
#     elif rule_key == "colored_life":
#         collect_colored_life_render_data(zoom, pan_x, pan_y, current_node_radius_px,
#                                          HIGHLIGHT_CHANGES, node_base_color_v,
#                                          node_outline_old_color_v, node_outline_new_color_v)
#     else:
#         logger.warning(f"Unsupported rule key '{rule_key}' for node/edge rendering.")
#         # Buffers already cleared above
#         return

#     # --- Draw using Taichi GGUI with Taichi Fields ---

#     # Calculate radii and widths in normalized coordinates
#     outline_radius_norm = max(0.0, current_node_radius_px / max(WINDOW_SIZE, 1))
#     # Make fill slightly smaller than outline
#     fill_radius_norm = max(0.0, (current_node_radius_px - current_edge_width_px * 0.5) / max(WINDOW_SIZE, 1))
#     fill_radius_norm = max(0.0, min(fill_radius_norm, outline_radius_norm * 0.85)) # Ensure fill is visibly smaller

#     edge_width_norm = max(0.0, current_edge_width_px / max(WINDOW_SIZE, 1))

#     # 1. Draw Default Edges
#     actual_default_edges = num_default_edges_to_draw[None]
#     if actual_default_edges > 0:
#         canvas.lines(draw_default_edge_endpoints, width=edge_width_norm, color=default_edge_rgb)

#     # 2. Draw Highlighted Edges
#     actual_highlight_edges = num_highlight_edges_to_draw[None]
#     if actual_highlight_edges > 0:
#         canvas.lines(draw_highlight_edge_endpoints, width=edge_width_norm, color=new_edge_rgb)

#     # 3. Draw Nodes (Outline then Fill)
#     actual_nodes = num_nodes_to_draw[None]
#     if actual_nodes > 0:
#         # Draw outlines first (larger radius, color from field)
#         canvas.circles(draw_node_pos,
#                        radius=outline_radius_norm,
#                        per_vertex_color=draw_node_outline_colors)

#         # Draw fills second (smaller radius, node_base color)
#         if fill_radius_norm > 0: # Only draw fill if it's visible
#              canvas.circles(draw_node_pos,
#                            radius=fill_radius_norm,
#                            color=node_base_rgb) # Use node_base color for fill

# Modify function signature to accept calculated sizes
def render_nodes_and_edges(current_node_radius_px, current_edge_width_px):
    """Renders nodes as circles with outlines and edges as lines directly onto the canvas using Taichi fields, supporting highlighting."""
    global color_manager # Ensure access to the color manager

    # --- Force Clear Render Buffers ---
    clear_render_data_kernel()
    # ---

    rule_key = rule_keys[selected_rule_idx]

    # --- Get Current Color Scheme ---
    if not (COLOR_MODULE_IMPORTED and color_manager and color_manager.current_scheme):
        logger.warning("ColorManager not available or no scheme set. Skipping node/edge render.")
        return

    scheme = color_manager.current_scheme
    try:
        # Convert hex colors to Taichi-compatible RGB tuples (0-1)
        bg_color_rgb = to_rgb(scheme.background) # Not used directly here, but good to have
        node_base_rgb = to_rgb(scheme.node_base)
        node_outline_old_rgb = to_rgb(scheme.node)
        node_outline_new_rgb = to_rgb(scheme.new_node)
        default_edge_rgb = to_rgb(scheme.default_edge)
        new_edge_rgb = to_rgb(scheme.new_edge)

        # Convert tuples to Taichi vectors
        node_base_color_v = ti.Vector(node_base_rgb)
        node_outline_old_color_v = ti.Vector(node_outline_old_rgb)
        node_outline_new_color_v = ti.Vector(node_outline_new_rgb)

    except Exception as e:
        logger.error(f"Error processing colors from scheme '{scheme.name}': {e}. Skipping render.")
        return

    # --- Call Appropriate Data Collection Kernel --- <<< RESTORED >>>
    # Pass necessary parameters including colors and highlight flag
    if rule_key == "realm_of_lace":
        collect_rol_render_data(zoom, pan_x, pan_y, current_node_radius_px,
                                HIGHLIGHT_CHANGES, node_base_color_v,
                                node_outline_old_color_v, node_outline_new_color_v)
    elif rule_key == "game_of_life":
        collect_gol_render_data(zoom, pan_x, pan_y, current_node_radius_px,
                                HIGHLIGHT_CHANGES, node_base_color_v,
                                node_outline_old_color_v, node_outline_new_color_v)
    elif rule_key == "colored_life":
        collect_colored_life_render_data(zoom, pan_x, pan_y, current_node_radius_px,
                                         HIGHLIGHT_CHANGES, node_base_color_v,
                                         node_outline_old_color_v, node_outline_new_color_v)
    else:
        logger.warning(f"Unsupported rule key '{rule_key}' for node/edge rendering.")
        return
    # --- <<< END RESTORED >>> ---

    # --- Draw using Taichi GGUI with Taichi Fields ---

    # Calculate radii and widths in normalized coordinates
    outline_radius_norm = max(0.0, current_node_radius_px / max(WINDOW_SIZE, 1))
    fill_radius_norm = max(0.0, (current_node_radius_px - current_edge_width_px * 0.5) / max(WINDOW_SIZE, 1))
    fill_radius_norm = max(0.0, min(fill_radius_norm, outline_radius_norm * 0.85))
    edge_width_norm = max(0.0, current_edge_width_px / max(WINDOW_SIZE, 1))

    # 1. Draw Default Edges
    actual_default_edges = num_default_edges_to_draw[None]
    if actual_default_edges > 0:
        canvas.lines(draw_default_edge_endpoints, width=edge_width_norm, color=default_edge_rgb)

    # 2. Draw Highlighted Edges
    actual_highlight_edges = num_highlight_edges_to_draw[None]
    if actual_highlight_edges > 0:
        canvas.lines(draw_highlight_edge_endpoints, width=edge_width_norm, color=new_edge_rgb)

    # 3. Draw Nodes (Outline then Fill)
    actual_nodes = num_nodes_to_draw[None]
    if actual_nodes > 0:
        # Draw outlines first (larger radius, color from field)
        canvas.circles(draw_node_pos,
                       radius=outline_radius_norm,
                       per_vertex_color=draw_node_outline_colors)

        # Draw fills second (smaller radius, node_base color)
        if fill_radius_norm > 0:
             canvas.circles(draw_node_pos,
                           radius=fill_radius_norm,
                           color=node_base_rgb)
                                                                               
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
    px0, py0 = map_grid_to_canvas(0, 0)
    px1, py1 = map_grid_to_canvas(1, 0)
    dist_sq = (px1 - px0)**2 + (py1 - py0)**2
    projected_separation_pixels = np.sqrt(dist_sq) if dist_sq > 0 else 0
    separation_check = projected_separation_pixels >= MIN_NODE_SEPARATION_PIXELS
    use_node_edge_render_mode = RENDER_EDGES and separation_check

    # Pass the calculated mode to handle_mouse
    handle_mouse(window, use_node_edge_render_mode)
    handle_gui()

    rule_key = rule_keys[selected_rule_idx]

    # --- Simulation Step ---
    if not paused:
        # --- Copy state BEFORE simulation step --- (Correct Timing)
        logger.debug(f"Frame {frame}: Copying state before simulation step.")
        if rule_key == "realm_of_lace":
            previous_node_degree.copy_from(node_degree)
        elif rule_key == "game_of_life" or rule_key == "colored_life":
            previous_node_state.copy_from(node_state)
        # ---

        # --- Run the simulation step ---
        if rule_key == "realm_of_lace":
            rol_step()
        elif rule_key == "game_of_life":
            gol_step()
        elif rule_key == "colored_life":
            colored_life_step()
        frame += 1

        # --- Add Synchronization Point --- ADDED HERE ---
        # Ensure simulation kernels finish writing to node_state/node_degree
        # before the rendering kernels read them.
        ti.sync()
        logger.debug(f"Frame {frame}: ti.sync() after simulation step.")
        # --- END ADDED SECTION ---

    # --- Visualization ---
    canvas.set_background_color((0, 0, 0)) # Clear canvas

    # --- Use the pre-calculated rendering mode ---
    if use_node_edge_render_mode:
        # ... (calculate radii/width - code remains the same) ...
        min_radius_at_threshold = max(1.0, MIN_NODE_SEPARATION_PIXELS / (2.0 * 1.2))
        scaling_range = max(1e-6, MAX_SEP_FOR_NODE_SCALING - MIN_NODE_SEPARATION_PIXELS)
        scale_factor = np.clip((projected_separation_pixels - MIN_NODE_SEPARATION_PIXELS) / scaling_range, 0.0, 1.0)
        current_node_radius_pixels = max(1.0, min_radius_at_threshold + scale_factor * (BASE_NODE_RADIUS - min_radius_at_threshold))
        current_edge_width_pixels = BASE_EDGE_WIDTH

        # Render function reads state AFTER simulation step and sync
        render_nodes_and_edges(current_node_radius_pixels, current_edge_width_pixels)
    else:
        # ... (image rendering code remains the same) ...
        if RENDER_EDGES and not separation_check:
             logger.debug(f"RENDER_EDGES is True, but separation ({projected_separation_pixels:.2f}) < threshold ({MIN_NODE_SEPARATION_PIXELS:.2f}). Using image rendering.")
        view_size_pixels = GRID_SIZE / zoom
        view_size_half_pixels = view_size_pixels / 2
        center_canvas_x, center_canvas_y = 0.5 * WINDOW_SIZE, 0.5 * WINDOW_SIZE
        center_grid_i, center_grid_j = map_canvas_to_grid_approx(center_canvas_x, center_canvas_y)
        cx_pixels = center_grid_i
        cy_pixels = center_grid_j
        x0 = int(np.clip(cx_pixels - view_size_half_pixels, 0, GRID_SIZE))
        y0 = int(np.clip(cy_pixels - view_size_half_pixels, 0, GRID_SIZE))
        x1 = int(np.clip(x0 + view_size_pixels, 0, GRID_SIZE))
        y1 = int(np.clip(y0 + view_size_pixels, 0, GRID_SIZE))
        view_width = x1 - x0
        view_height = y1 - y0
        if view_width > 0 and view_height > 0:
            img_resized = get_image_window(x0, x1, y0, y1)
            canvas.set_image(img_resized)
        else:
            logger.warning(f"Calculated view window for image rendering is invalid: w={view_width}, h={view_height}. Setting black.")
            canvas.set_image(np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8))

    # --- Info Display ---
    control_height = 0.38
    info_y = 0.01 + control_height + 0.01
    gui.begin("Info", 0.01, info_y, 0.3, 0.18)
    gui.text(f"Frame: {frame}")
    gui.text(f"Rule: {rule_names[selected_rule_idx]}")
    gui.text(f"Zoom: {zoom:.2f}")
    gui.text(f"Pan: ({pan_x:.2f}, {pan_y:.2f})")
    gui.text(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    gui.text(f"Paused: {paused}")
    
    # Show if density is saved for current rule
    current_rule_name = rule_names[selected_rule_idx]
    density_status = "*" if current_rule_name in rule_density_map else ""
    gui.text(f"Init Density: {INITIAL_DENSITY:.2f}{density_status}")
    
    gui.text(f"Speed (steps/sec): {speed_slider_value:.1f}")
    gui.text(f"Render Edges: {RENDER_EDGES}")
    gui.text(f"Highlight Changes: {HIGHLIGHT_CHANGES}")
    gui.end()

    window.show()
    time.sleep(STEP_DELAY)

logger.info("--- Exited Main Loop ---")
# --- End of Script ---