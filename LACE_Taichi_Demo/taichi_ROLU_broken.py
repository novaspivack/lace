# taichi_realm_of_lace_demo.py
# Combined and Refactored Taichi Demo for RoL, GoL, Colored Life, and ROL-U

# ==============================================================================
# --- IMPORTS ---
# ==============================================================================
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
from typing import List, Tuple, Optional, Union, TypeVar, Dict, Set, Any # Added Any
import dataclasses # For RuleMetadata fields check
# ==============================================================================

# ==============================================================================
# --- TAICHI INITIALIZATION ---
# ==============================================================================
try:
    ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_fraction=0.9)
    print("Taichi initialized successfully on GPU.")
except Exception as e_gpu:
    print(f"GPU initialization failed: {e_gpu}. Falling back to CPU.")
    try:
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        print("Taichi initialized successfully on CPU.")
    except Exception as e_cpu:
        print(f"CPU initialization failed: {e_cpu}. Exiting.")
        sys.exit(1)
# ==============================================================================

# ==============================================================================
# --- LOGGING CONFIGURATION ---
# ==============================================================================
class LogSettings:
    class Logging:
        LOG_LEVEL: str = "DEBUG" # Set desired level (DEBUG, INFO, DETAIL, WARNING, ERROR)

# --- Custom Log Level ---
DETAIL_LEVEL_NUM = 15
logging.addLevelName(DETAIL_LEVEL_NUM, "DETAIL")

def detail(self, message, *args, **kws):
    if self.isEnabledFor(DETAIL_LEVEL_NUM): self._log(DETAIL_LEVEL_NUM, message, args, **kws)
logging.Logger.detail = detail # type: ignore [attr-defined]
# --- End Custom Log Level ---

APP_DIR = "LACE_Taichi_Demo"
SUBDIRS = {'logs': 'logs', 'config': 'config'}

def setup_directories() -> Tuple[dict, str]:
    try:
        base_path = os.getcwd(); app_base_path = os.path.join(base_path, APP_DIR)
        os.makedirs(app_base_path, exist_ok=True); paths = {}
        for key, subdir in SUBDIRS.items():
            path = os.path.join(app_base_path, subdir); os.makedirs(path, exist_ok=True); paths[key] = path
        return paths, app_base_path
    except Exception as e: print(f"Fatal error in directory setup: {str(e)}"); raise SystemExit(1)

def setup_logging(log_dir: str) -> logging.Logger:
    try:
        logger_name = "taichi_ca_logger"; logger = logging.getLogger(logger_name)
        if logger.handlers: return logger
        timestamp_24hr = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_level_str = LogSettings.Logging.LOG_LEVEL.upper()
        file_log_level = getattr(logging, log_level_str, logging.INFO)
        if log_level_str == "DETAIL": file_log_level = DETAIL_LEVEL_NUM
        console_log_level = logging.DEBUG
        main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        main_log_filename = f'taichi_ca_demo_{timestamp_24hr}.log'
        main_file_handler = logging.FileHandler(os.path.join(log_dir, main_log_filename))
        main_file_handler.setFormatter(main_formatter); main_file_handler.setLevel(file_log_level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(main_formatter); console_handler.setLevel(console_log_level)
        logger.setLevel(min(file_log_level, console_log_level))
        logger.addHandler(main_file_handler); logger.addHandler(console_handler); logger.propagate = False
        file_log_level_name = logging.getLevelName(file_log_level); console_log_level_name = logging.getLevelName(console_log_level)
        logger.info(f"Logging initialized. Logger Level: {logging.getLevelName(logger.level)}, File Handler Level: {file_log_level_name}, Console Handler Level: {console_log_level_name}")
        logger.info(f"Log file: {os.path.join(log_dir, main_log_filename)}")
        return logger
    except Exception as e:
        print(f"Error setting up logging: {e}")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("taichi_ca_logger_fallback"); logger.error(f"Logging setup failed: {e}. Using basicConfig."); return logger

# --- Initialize Logger FIRST ---
APP_PATHS, BASE_PATH = setup_directories()
logger = setup_logging(APP_PATHS['logs'])
# ==============================================================================

# ==============================================================================
# --- JSON LOADING & RULE VARIANT PARSING ---
# ==============================================================================
# <<< IMPORTANT: Set the correct path to your rules.json file here >>>
JSON_FILE_PATH = '/Users/nova/My Drive (nova@novaspivack.com)/Works in Progress/Python/LACE - NEW - In progress/LACE/Resources/config/rules/rules.json'
# <<< --- >>>

def load_rules_from_json(filepath):
    """Loads rule definitions from a JSON file."""
    if not os.path.exists(filepath): logger.error(f"JSON file not found at {filepath}"); return []
    try:
        with open(filepath, 'r') as f: data = json.load(f)
        logger.info(f"Successfully loaded JSON from {filepath}"); return data.get("rules", [])
    except json.JSONDecodeError as e: logger.error(f"Error decoding JSON from {filepath}: {e}"); return []
    except Exception as e: logger.error(f"An error occurred while reading {filepath}: {e}"); return []

def parse_rule_variants(json_rules):
    """Parses JSON rules into the format needed for the Taichi script, including ROL-U."""
    variants = []
    logger.info(f"--- Starting JSON Rule Parsing ({len(json_rules)} rules found) ---")
    variants.append(
        ("Game of Life (B3/S23)", "game_of_life", {
            "GOL_BIRTH": [3], "GOL_SURVIVAL": [2, 3], "NODE_COLORMAP": 'gray',
            "NODE_COLOR_NORM_VMIN": 0.0, "NODE_COLOR_NORM_VMAX": 1.0,
            "COLOR_BY_DEGREE": False, "COLOR_BY_ACTIVE_NEIGHBORS": False,
        })
    )
    for rule_idx, rule in enumerate(json_rules):
        rule_type = rule.get("type"); rule_name = rule.get("name", f"Unnamed Rule {rule_idx}"); params = rule.get("params", {})
        logger.debug(f"\nProcessing Rule Index {rule_idx}: Name='{rule_name}', Type='{rule_type}'")
        if rule_type == "RealmOfLace":
            logger.detail(f"  Processing as RealmOfLace (Original)") # type: ignore [attr-defined]
            birth_ranges_raw = params.get("birth_neighbor_degree_sum_range", []); survival_ranges_raw = params.get("survival_neighbor_degree_sum_range", [])
            death_degrees_raw = params.get("final_death_degree_counts", []); death_ranges_raw = params.get("final_death_degree_range", [])
            birth_ranges, survival_ranges, death_degrees, death_ranges = [], [], [], []
            try:
                if isinstance(birth_ranges_raw, list): birth_ranges = [tuple(map(float, r)) for r in birth_ranges_raw if isinstance(r, (list, tuple)) and len(r) == 2]
                if isinstance(survival_ranges_raw, list): survival_ranges = [tuple(map(float, r)) for r in survival_ranges_raw if isinstance(r, (list, tuple)) and len(r) == 2]
                if isinstance(death_degrees_raw, list): death_degrees = [int(d) for d in death_degrees_raw if isinstance(d, (int, float))]
                if isinstance(death_ranges_raw, list): death_ranges = [tuple(map(float, r)) for r in death_ranges_raw if isinstance(r, (list, tuple)) and len(r) == 2]
            except Exception as e: logger.error(f"    ERROR converting RoL params for '{rule_name}': {e}. Using empty defaults.")
            variant_params = {
                "BIRTH_SUM_RANGES": birth_ranges, "SURVIVAL_SUM_RANGES": survival_ranges,
                "FINAL_DEATH_DEGREES": death_degrees, "FINAL_DEATH_DEGREE_RANGES": death_ranges,
                "NODE_COLORMAP": params.get("node_colormap", 'viridis'), "NODE_COLOR_NORM_VMIN": float(params.get("node_color_norm_vmin", 0.0)),
                "NODE_COLOR_NORM_VMAX": float(params.get("node_color_norm_vmax", 8.0)), "COLOR_BY_DEGREE": params.get("color_nodes_by_degree", True),
                "COLOR_BY_ACTIVE_NEIGHBORS": params.get("color_nodes_by_active_neighbors", False),
            }
            variants.append((rule_name, "realm_of_lace", variant_params))
        elif rule_type == "RealmOfLaceUnified":
            logger.detail(f"  Processing as RealmOfLaceUnified") # type: ignore [attr-defined]
            birth_metric = params.get("birth_metric_type", "DEGREE"); birth_agg = params.get("birth_metric_aggregation", "SUM")
            survival_metric = params.get("survival_metric_type", "DEGREE"); survival_agg = params.get("survival_metric_aggregation", "SUM")
            final_metric = params.get("final_check_metric", "DEGREE"); clustering_denom = params.get("clustering_denominator_type", "ACTUAL")
            def get_dynamic_param_name(base, metric, agg):
                is_non_agg = metric in ["SYMMETRY_STATE", "SYMMETRY_DEGREE", "NEIGHBOR_DEGREE_VARIANCE", "NEIGHBOR_DEGREE_STDDEV"]
                agg_suffix = f"_{agg}" if agg and not is_non_agg else ""; return f"{base}_{metric}{agg_suffix}"
            birth_range_key = get_dynamic_param_name("birth_eligibility_range", birth_metric, birth_agg); birth_values_key = get_dynamic_param_name("birth_eligibility_values", birth_metric, birth_agg)
            survival_range_key = get_dynamic_param_name("survival_eligibility_range", survival_metric, survival_agg); survival_values_key = get_dynamic_param_name("survival_eligibility_values", survival_metric, survival_agg)
            final_death_range_key = get_dynamic_param_name("final_death_metric_range", final_metric, None); final_death_values_key = get_dynamic_param_name("final_death_metric_values", final_metric, None)
            final_life_range_key = get_dynamic_param_name("final_life_metric_range", final_metric, None); final_life_values_key = get_dynamic_param_name("final_life_metric_values", final_metric, None)
            birth_ranges_raw = params.get(birth_range_key, []); birth_values_raw = params.get(birth_values_key, [])
            survival_ranges_raw = params.get(survival_range_key, []); survival_values_raw = params.get(survival_values_key, [])
            death_ranges_raw = params.get(final_death_range_key, []); death_values_raw = params.get(final_death_values_key, [])
            life_ranges_raw = params.get(final_life_range_key, []); life_values_raw = params.get(final_life_values_key, [])
            birth_ranges, survival_ranges, death_ranges, life_ranges = [], [], [], []
            birth_values, survival_values, death_values, life_values = [], [], [], []
            try:
                if isinstance(birth_ranges_raw, list): birth_ranges = [tuple(map(float, r)) for r in birth_ranges_raw if isinstance(r, (list, tuple)) and len(r) == 2]
                if isinstance(survival_ranges_raw, list): survival_ranges = [tuple(map(float, r)) for r in survival_ranges_raw if isinstance(r, (list, tuple)) and len(r) == 2]
                if isinstance(death_ranges_raw, list): death_ranges = [tuple(map(float, r)) for r in death_ranges_raw if isinstance(r, (list, tuple)) and len(r) == 2]
                if isinstance(life_ranges_raw, list): life_ranges = [tuple(map(float, r)) for r in life_ranges_raw if isinstance(r, (list, tuple)) and len(r) == 2]
                if isinstance(birth_values_raw, list): birth_values = [float(v) if isinstance(v, (int, float)) else 0.0 for v in birth_values_raw]
                if isinstance(survival_values_raw, list): survival_values = [float(v) if isinstance(v, (int, float)) else 0.0 for v in survival_values_raw]
                if isinstance(death_values_raw, list): death_values = [float(v) if isinstance(v, (int, float)) else 0.0 for v in death_values_raw]
                if isinstance(life_values_raw, list): life_values = [float(v) if isinstance(v, (int, float)) else 0.0 for v in life_values_raw]
            except Exception as e: logger.error(f"    ERROR converting ROL-U params for '{rule_name}': {e}. Using empty defaults.")
            variant_params = {
                "ACTIVE_BIRTH_RANGES": birth_ranges, "ACTIVE_BIRTH_VALUES": birth_values,
                "ACTIVE_SURVIVAL_RANGES": survival_ranges, "ACTIVE_SURVIVAL_VALUES": survival_values,
                "ACTIVE_DEATH_RANGES": death_ranges, "ACTIVE_DEATH_VALUES": death_values,
                "ACTIVE_LIFE_RANGES": life_ranges, "ACTIVE_LIFE_VALUES": life_values,
                "BIRTH_METRIC": birth_metric, "BIRTH_AGG": birth_agg, "SURVIVAL_METRIC": survival_metric,
                "SURVIVAL_AGG": survival_agg, "FINAL_METRIC": final_metric, "CLUSTERING_DENOM": clustering_denom,
                "NODE_COLORMAP": params.get("node_colormap", 'viridis'), "NODE_COLOR_NORM_VMIN": float(params.get("node_color_norm_vmin", 0.0)),
                "NODE_COLOR_NORM_VMAX": float(params.get("node_color_norm_vmax", 8.0)), "COLOR_BY_DEGREE": params.get("color_nodes_by_degree", True),
                "COLOR_BY_ACTIVE_NEIGHBORS": params.get("color_nodes_by_active_neighbors", False),
            }
            variants.append((rule_name, "realm_of_lace_unified", variant_params))
        elif rule_type == "LifeWithColor":
            logger.detail(f"  Processing as LifeWithColor") # type: ignore [attr-defined]
            variant_params = {
                "GOL_BIRTH": params.get("birth_neighbor_counts", [3]), "GOL_SURVIVAL": params.get("survival_neighbor_counts", [2, 3]),
                "NODE_COLORMAP": params.get("node_colormap", 'plasma'), "NODE_COLOR_NORM_VMIN": float(params.get("node_color_norm_vmin", 0.0)),
                "NODE_COLOR_NORM_VMAX": float(params.get("node_color_norm_vmax", 8.0)), "COLOR_BY_DEGREE": False, "COLOR_BY_ACTIVE_NEIGHBORS": True,
            }
            variants.append((rule_name, "colored_life", variant_params))
    logger.info(f"--- Finished JSON Rule Parsing ({len(variants)} variants created) ---")
    return variants

# --- Load rules and create variants list BEFORE using RULE_VARIANTS ---
all_json_rules = load_rules_from_json(JSON_FILE_PATH)
RULE_VARIANTS = parse_rule_variants(all_json_rules)

# --- Add Fallback Rules AFTER parsing, if RULE_VARIANTS is still too short ---
if not RULE_VARIANTS or len(RULE_VARIANTS) <= 1:
    logger.warning("No rule variants loaded or parsed from JSON. Using default fallback rules.")
    # Define fallbacks directly here
    fallback_rules = [
        ("Game of Life (Fallback)", "game_of_life", {"GOL_BIRTH": [3], "GOL_SURVIVAL": [2, 3], "NODE_COLORMAP": 'gray', "NODE_COLOR_NORM_VMIN": 0.0, "NODE_COLOR_NORM_VMAX": 1.0, "COLOR_BY_DEGREE": False, "COLOR_BY_ACTIVE_NEIGHBORS": False}),
        ("Realm of Lace (Fallback)", "realm_of_lace", {"BIRTH_SUM_RANGES": [(5.0, 6.0), (8.0, 9.0), (15.0, 16.0)], "SURVIVAL_SUM_RANGES": [(3.0, 6.0), (8.0, 11.0), (15.0, 16.0)], "FINAL_DEATH_DEGREES": [0, 11, 12, 13, 14], "FINAL_DEATH_DEGREE_RANGES": [], "NODE_COLORMAP": 'viridis', "NODE_COLOR_NORM_VMIN": 0.0, "NODE_COLOR_NORM_VMAX": 8.0, "COLOR_BY_DEGREE": True, "COLOR_BY_ACTIVE_NEIGHBORS": False}),
        ("Colored Life (Fallback)", "colored_life", {"GOL_BIRTH": [3], "GOL_SURVIVAL": [2, 3], "NODE_COLORMAP": 'plasma', "NODE_COLOR_NORM_VMIN": 0.0, "NODE_COLOR_NORM_VMAX": 8.0, "COLOR_BY_DEGREE": False, "COLOR_BY_ACTIVE_NEIGHBORS": True}),
        ("Realm of Lace Unified (Fallback)", "realm_of_lace_unified", {"ACTIVE_BIRTH_RANGES": [(3.0, 5.0)], "ACTIVE_BIRTH_VALUES": [], "ACTIVE_SURVIVAL_RANGES": [(2.0, 4.0)], "ACTIVE_SURVIVAL_VALUES": [], "ACTIVE_DEATH_RANGES": [], "ACTIVE_DEATH_VALUES": [0.0, 1.0, 7.0, 8.0], "ACTIVE_LIFE_RANGES": [], "ACTIVE_LIFE_VALUES": [], "BIRTH_METRIC": "DEGREE", "BIRTH_AGG": "SUM", "SURVIVAL_METRIC": "DEGREE", "SURVIVAL_AGG": "SUM", "FINAL_METRIC": "DEGREE", "CLUSTERING_DENOM": "ACTUAL", "NODE_COLORMAP": 'viridis', "NODE_COLOR_NORM_VMIN": 0.0, "NODE_COLOR_NORM_VMAX": 8.0, "COLOR_BY_DEGREE": True, "COLOR_BY_ACTIVE_NEIGHBORS": False}),
    ]
    # Add fallbacks only if they are not already present by name
    existing_names = {r[0] for r in RULE_VARIANTS}
    for fb_name, fb_key, fb_params in fallback_rules:
        if fb_name not in existing_names:
            RULE_VARIANTS.append((fb_name, fb_key, fb_params))
# ==============================================================================

# ==============================================================================
# --- GLOBAL SETTINGS & STATE ---
# ==============================================================================
GRID_SIZE = 1000
WINDOW_SIZE = 1000

# --- Shared Parameters ---
WRAP = True
INITIAL_DENSITY = 0.10
STEP_DELAY = 1/60
paused = False
speed_slider_value = 60.0

# --- Neighborhood Definition (Global Constants) ---
NUM_NEIGHBORS = 8
neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), # Row offsets, Col offsets for Moore
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]
# ---

# --- Rule Selection State (Derived AFTER RULE_VARIANTS is defined) ---
rule_names = [r[0] for r in RULE_VARIANTS]
rule_keys = [r[1] for r in RULE_VARIANTS]
rule_params = [r[2] for r in RULE_VARIANTS]

# --- Find the index of the rule matching the original hardcoded params ---
initial_rule_name_to_find = "Realm of Lace_Fancy Interesting Shapes"
selected_rule_idx = 0 # Default to 0
try:
    selected_rule_idx = next(i for i, name in enumerate(rule_names) if name == initial_rule_name_to_find)
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
MIN_NODE_SEPARATION_PIXELS = BASE_NODE_RADIUS * 2 * 1.2
MAX_SEP_FOR_NODE_SCALING = MIN_NODE_SEPARATION_PIXELS * 3.0
MIN_ZOOM_FOR_NODE_EDGE_RENDER = 3.0

# --- Fields for Drawing Primitives ---
MAX_DRAW_NODES = 50000
MAX_DRAW_LINES = 100000
draw_node_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_DRAW_NODES)
draw_node_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_DRAW_NODES)
draw_edge_endpoints = ti.Vector.field(2, dtype=ti.f32, shape=MAX_DRAW_LINES * 2)
num_nodes_to_draw = ti.field(dtype=ti.i32, shape=())
num_edges_to_draw = ti.field(dtype=ti.i32, shape=())
# ==============================================================================

# ==============================================================================
# --- TAICHI FIELD DECLARATIONS ---
# ==============================================================================
# --- State Fields ---
node_state = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE)) # GoL, Colored Life state
node_next_state = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE)) # GoL, Colored Life next state
node_degree = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE)) # RoL, ROL-U state
node_next_degree = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE)) # RoL, ROL-U next state
node_active_neighbors = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE)) # GoL, Colored Life, ROL-U neighbor count
node_next_eligible = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE)) # RoL, ROL-U eligibility

# --- Parameter Fields ---
MAX_RANGES = 10
MAX_VALUES = 20 # Increased size for GoL B/S counts

# ROL / ROL-U Eligibility Criteria Fields
rol_birth_ranges_field = ti.field(dtype=ti.f32, shape=(MAX_RANGES, 2))
rol_survival_ranges_field = ti.field(dtype=ti.f32, shape=(MAX_RANGES, 2))
rol_birth_values_field = ti.field(dtype=ti.f32, shape=(MAX_VALUES))
rol_survival_values_field = ti.field(dtype=ti.f32, shape=(MAX_VALUES))
rol_num_birth_ranges = ti.field(dtype=ti.i32, shape=())
rol_num_survival_ranges = ti.field(dtype=ti.i32, shape=())
rol_num_birth_values = ti.field(dtype=ti.i32, shape=())
rol_num_survival_values = ti.field(dtype=ti.i32, shape=())

# ROL / ROL-U Final State Criteria Fields
rol_final_death_values_field = ti.field(dtype=ti.f32, shape=(MAX_VALUES))
rol_final_death_ranges_field = ti.field(dtype=ti.f32, shape=(MAX_RANGES, 2))
rol_final_life_values_field = ti.field(dtype=ti.f32, shape=(MAX_VALUES))
rol_final_life_ranges_field = ti.field(dtype=ti.f32, shape=(MAX_RANGES, 2))
rol_num_final_death_values = ti.field(dtype=ti.i32, shape=())
rol_num_final_death_ranges = ti.field(dtype=ti.i32, shape=())
rol_num_final_life_values = ti.field(dtype=ti.i32, shape=())
rol_num_final_life_ranges = ti.field(dtype=ti.i32, shape=())

# GoL Criteria Fields
gol_birth_field = ti.field(dtype=ti.i32, shape=(MAX_VALUES))
gol_survival_field = ti.field(dtype=ti.i32, shape=(MAX_VALUES))
gol_num_birth = ti.field(dtype=ti.i32, shape=())
gol_num_survival = ti.field(dtype=ti.i32, shape=())

# ROL-U Specific Parameter Fields
rolu_birth_metric_type = ti.field(dtype=ti.i32, shape=())
rolu_birth_aggregation = ti.field(dtype=ti.i32, shape=())
rolu_survival_metric_type = ti.field(dtype=ti.i32, shape=())
rolu_survival_aggregation = ti.field(dtype=ti.i32, shape=())
rolu_final_check_metric = ti.field(dtype=ti.i32, shape=())
rolu_clustering_denom_type = ti.field(dtype=ti.i32, shape=())

# Visualization Parameter Fields
color_by_degree_flag = ti.field(dtype=ti.i32, shape=())
color_by_active_neighbors_flag = ti.field(dtype=ti.i32, shape=())

# Kernel for logging mapping output
log_kernel_output = ti.Vector.field(2, dtype=ti.f32, shape=())
# ==============================================================================

# ==============================================================================
# --- PARAMETER FIELD MANAGEMENT ---
# ==============================================================================
@ti.kernel
def clear_param_fields():
    # ROL / ROL-U Eligibility
    rol_num_birth_ranges[None] = 0
    rol_num_survival_ranges[None] = 0
    rol_num_birth_values[None] = 0
    rol_num_survival_values[None] = 0
    for i in range(MAX_RANGES):
        rol_birth_ranges_field[i, 0] = -1.0; rol_birth_ranges_field[i, 1] = -1.0
        rol_survival_ranges_field[i, 0] = -1.0; rol_survival_ranges_field[i, 1] = -1.0
    for i in range(MAX_VALUES):
        rol_birth_values_field[i] = -999.0; rol_survival_values_field[i] = -999.0

    # ROL / ROL-U Final State
    rol_num_final_death_values[None] = 0; rol_num_final_death_ranges[None] = 0
    rol_num_final_life_values[None] = 0; rol_num_final_life_ranges[None] = 0
    for i in range(MAX_VALUES):
        rol_final_death_values_field[i] = -999.0; rol_final_life_values_field[i] = -999.0
    for i in range(MAX_RANGES):
        rol_final_death_ranges_field[i, 0] = -1.0; rol_final_death_ranges_field[i, 1] = -1.0
        rol_final_life_ranges_field[i, 0] = -1.0; rol_final_life_ranges_field[i, 1] = -1.0

    # GoL
    gol_num_birth[None] = 0; gol_num_survival[None] = 0
    for i in range(MAX_VALUES):
        gol_birth_field[i] = -1; gol_survival_field[i] = -1

    # ROL-U Specific
    rolu_birth_metric_type[None] = 0; rolu_birth_aggregation[None] = 0
    rolu_survival_metric_type[None] = 0; rolu_survival_aggregation[None] = 0
    rolu_final_check_metric[None] = 0; rolu_clustering_denom_type[None] = 0

    # Visualization
    color_by_degree_flag[None] = 0; color_by_active_neighbors_flag[None] = 0

def set_rule_variant(idx):
    """Sets the global parameters AND copies them into Taichi fields."""
    global BIRTH_SUM_RANGES, SURVIVAL_SUM_RANGES, FINAL_DEATH_DEGREES, FINAL_DEATH_DEGREE_RANGES
    global GOL_BIRTH, GOL_SURVIVAL
    global NODE_COLORMAP, NODE_COLOR_NORM_VMIN, NODE_COLOR_NORM_VMAX
    global COLOR_BY_DEGREE, COLOR_BY_ACTIVE_NEIGHBORS
    # Add globals for ROL-U specific params if needed for Python logic (unlikely here)

    if idx < 0 or idx >= len(rule_params):
        logger.error(f"Invalid rule index {idx}")
        idx = 0

    params = rule_params[idx]
    rule_key = rule_keys[idx]
    rule_name = rule_names[idx]

    logger.info(f"--- Setting Variant {idx}: {rule_name} (Key: {rule_key}) ---")

    clear_param_fields() # Clear all Taichi param fields first

    # Reset Global Python Vars used for display/reference
    BIRTH_SUM_RANGES, SURVIVAL_SUM_RANGES, FINAL_DEATH_DEGREES, FINAL_DEATH_DEGREE_RANGES = [], [], [], []
    GOL_BIRTH, GOL_SURVIVAL = [], []
    NODE_COLORMAP = 'gray'
    NODE_COLOR_NORM_VMIN = 0.0
    NODE_COLOR_NORM_VMAX = 1.0
    COLOR_BY_DEGREE = False
    COLOR_BY_ACTIVE_NEIGHBORS = False

    # --- Apply specific parameters AND copy to Taichi fields ---

    # --- RealmOfLace (Original) ---
    if rule_key == "realm_of_lace":
        BIRTH_SUM_RANGES = params.get("BIRTH_SUM_RANGES", [])
        SURVIVAL_SUM_RANGES = params.get("SURVIVAL_SUM_RANGES", [])
        FINAL_DEATH_DEGREES = params.get("FINAL_DEATH_DEGREES", []) # Used for values field
        FINAL_DEATH_DEGREE_RANGES = params.get("FINAL_DEATH_DEGREE_RANGES", [])
        NODE_COLORMAP = params.get("NODE_COLORMAP", 'viridis')
        NODE_COLOR_NORM_VMIN = params.get("NODE_COLOR_NORM_VMIN", 0.0)
        NODE_COLOR_NORM_VMAX = params.get("NODE_COLOR_NORM_VMAX", 8.0)
        COLOR_BY_DEGREE = params.get("COLOR_BY_DEGREE", True)
        COLOR_BY_ACTIVE_NEIGHBORS = params.get("COLOR_BY_ACTIVE_NEIGHBORS", False)

        logger.debug(f"  Applying RoL (Original) Params for '{rule_name}'")
        try:
            # Copy eligibility ranges
            n_birth_r = min(len(BIRTH_SUM_RANGES), MAX_RANGES); rol_num_birth_ranges[None] = n_birth_r
            for i in range(n_birth_r): rol_birth_ranges_field[i, 0] = BIRTH_SUM_RANGES[i][0]; rol_birth_ranges_field[i, 1] = BIRTH_SUM_RANGES[i][1]
            n_survival_r = min(len(SURVIVAL_SUM_RANGES), MAX_RANGES); rol_num_survival_ranges[None] = n_survival_r
            for i in range(n_survival_r): rol_survival_ranges_field[i, 0] = SURVIVAL_SUM_RANGES[i][0]; rol_survival_ranges_field[i, 1] = SURVIVAL_SUM_RANGES[i][1]
            # Copy final death criteria (degrees go to values field, ranges to ranges field)
            n_death_val = min(len(FINAL_DEATH_DEGREES), MAX_VALUES); rol_num_final_death_values[None] = n_death_val
            for i in range(n_death_val): rol_final_death_values_field[i] = float(FINAL_DEATH_DEGREES[i]) # Store as float
            n_death_range = min(len(FINAL_DEATH_DEGREE_RANGES), MAX_RANGES); rol_num_final_death_ranges[None] = n_death_range
            for i in range(n_death_range): rol_final_death_ranges_field[i, 0] = FINAL_DEATH_DEGREE_RANGES[i][0]; rol_final_death_ranges_field[i, 1] = FINAL_DEATH_DEGREE_RANGES[i][1]
            # Set ROL-U specific fields to defaults for original RoL
            rolu_birth_metric_type[None] = 0 # DEGREE
            rolu_birth_aggregation[None] = 0 # SUM
            rolu_survival_metric_type[None] = 0 # DEGREE
            rolu_survival_aggregation[None] = 0 # SUM
            rolu_final_check_metric[None] = 0 # DEGREE
            rolu_clustering_denom_type[None] = 0 # ACTUAL
            logger.info(f"  ---> Copied RoL(Orig) to Fields: BirthRanges={n_birth_r}, SurvRanges={n_survival_r}, DeathVals={n_death_val}, DeathRanges={n_death_range}")
        except Exception as e: logger.error(f"    ERROR copying RoL(Orig) params to Taichi fields for '{rule_name}': {e}")

    # --- RealmOfLaceUnified ---
    elif rule_key == "realm_of_lace_unified":
        # Get active criteria from the parsed variant
        birth_ranges = params.get("ACTIVE_BIRTH_RANGES", [])
        birth_values = params.get("ACTIVE_BIRTH_VALUES", [])
        survival_ranges = params.get("ACTIVE_SURVIVAL_RANGES", [])
        survival_values = params.get("ACTIVE_SURVIVAL_VALUES", [])
        death_ranges = params.get("ACTIVE_DEATH_RANGES", [])
        death_values = params.get("ACTIVE_DEATH_VALUES", [])
        life_ranges = params.get("ACTIVE_LIFE_RANGES", [])
        life_values = params.get("ACTIVE_LIFE_VALUES", [])
        # Get selectors
        birth_metric_str = params.get("BIRTH_METRIC", "DEGREE")
        birth_agg_str = params.get("BIRTH_AGG", "SUM")
        survival_metric_str = params.get("SURVIVAL_METRIC", "DEGREE")
        survival_agg_str = params.get("SURVIVAL_AGG", "SUM")
        final_metric_str = params.get("FINAL_METRIC", "DEGREE")
        clustering_denom_str = params.get("CLUSTERING_DENOM", "ACTUAL")
        # Get coloring info
        NODE_COLORMAP = params.get("NODE_COLORMAP", 'viridis')
        NODE_COLOR_NORM_VMIN = params.get("NODE_COLOR_NORM_VMIN", 0.0)
        NODE_COLOR_NORM_VMAX = params.get("NODE_COLOR_NORM_VMAX", 8.0)
        COLOR_BY_DEGREE = params.get("COLOR_BY_DEGREE", True)
        COLOR_BY_ACTIVE_NEIGHBORS = params.get("COLOR_BY_ACTIVE_NEIGHBORS", False)

        logger.debug(f"  Applying ROL-U Params for '{rule_name}'")
        logger.debug(f"    Selectors: Birth={birth_metric_str}/{birth_agg_str}, Survival={survival_metric_str}/{survival_agg_str}, Final={final_metric_str}")
        logger.debug(f"    Active Criteria: BirthR={birth_ranges}, BirthV={birth_values}, SurvR={survival_ranges}, SurvV={survival_values}")
        logger.debug(f"    Active Final: DeathR={death_ranges}, DeathV={death_values}, LifeR={life_ranges}, LifeV={life_values}")

        try:
            # --- Copy ACTIVE criteria to Taichi fields ---
            # Eligibility
            n_birth_r = min(len(birth_ranges), MAX_RANGES); rol_num_birth_ranges[None] = n_birth_r
            for i in range(n_birth_r): rol_birth_ranges_field[i, 0] = birth_ranges[i][0]; rol_birth_ranges_field[i, 1] = birth_ranges[i][1]
            n_birth_v = min(len(birth_values), MAX_VALUES); rol_num_birth_values[None] = n_birth_v
            for i in range(n_birth_v): rol_birth_values_field[i] = birth_values[i]

            n_survival_r = min(len(survival_ranges), MAX_RANGES); rol_num_survival_ranges[None] = n_survival_r
            for i in range(n_survival_r): rol_survival_ranges_field[i, 0] = survival_ranges[i][0]; rol_survival_ranges_field[i, 1] = survival_ranges[i][1]
            n_survival_v = min(len(survival_values), MAX_VALUES); rol_num_survival_values[None] = n_survival_v
            for i in range(n_survival_v): rol_survival_values_field[i] = survival_values[i]

            # Final State
            n_death_r = min(len(death_ranges), MAX_RANGES); rol_num_final_death_ranges[None] = n_death_r
            for i in range(n_death_r): rol_final_death_ranges_field[i, 0] = death_ranges[i][0]; rol_final_death_ranges_field[i, 1] = death_ranges[i][1]
            n_death_v = min(len(death_values), MAX_VALUES); rol_num_final_death_values[None] = n_death_v
            for i in range(n_death_v): rol_final_death_values_field[i] = death_values[i]

            n_life_r = min(len(life_ranges), MAX_RANGES); rol_num_final_life_ranges[None] = n_life_r
            for i in range(n_life_r): rol_final_life_ranges_field[i, 0] = life_ranges[i][0]; rol_final_life_ranges_field[i, 1] = life_ranges[i][1]
            n_life_v = min(len(life_values), MAX_VALUES); rol_num_final_life_values[None] = n_life_v
            for i in range(n_life_v): rol_final_life_values_field[i] = life_values[i]

            # --- Set ROL-U specific type fields ---
            metric_map = {"DEGREE": 0, "CLUSTERING": 1, "BETWEENNESS": 2, "ACTIVE_NEIGHBOR_COUNT": 3}
            agg_map = {"SUM": 0, "AVERAGE": 1}
            denom_map = {"ACTUAL": 0, "THEORETICAL": 1}

            rolu_birth_metric_type[None] = metric_map.get(birth_metric_str, 0)
            rolu_birth_aggregation[None] = agg_map.get(birth_agg_str, 0)
            rolu_survival_metric_type[None] = metric_map.get(survival_metric_str, 0)
            rolu_survival_aggregation[None] = agg_map.get(survival_agg_str, 0)
            rolu_final_check_metric[None] = metric_map.get(final_metric_str, 0)
            rolu_clustering_denom_type[None] = denom_map.get(clustering_denom_str, 0)

            logger.info(f"  ---> Copied ROL-U to Fields: Birth(R={n_birth_r},V={n_birth_v}), Surv(R={n_survival_r},V={n_survival_v}), Death(R={n_death_r},V={n_death_v}), Life(R={n_life_r},V={n_life_v})")
            logger.info(f"  ---> Set ROL-U Types: BirthM={rolu_birth_metric_type[None]}, BirthA={rolu_birth_aggregation[None]}, SurvM={rolu_survival_metric_type[None]}, SurvA={rolu_survival_aggregation[None]}, FinalM={rolu_final_check_metric[None]}, ClustD={rolu_clustering_denom_type[None]}")
        except Exception as e: logger.error(f"    ERROR copying ROL-U params to Taichi fields for '{rule_name}': {e}")

    # --- Game of Life ---
    elif rule_key == "game_of_life":
        GOL_BIRTH = params.get("GOL_BIRTH", [3])
        GOL_SURVIVAL = params.get("GOL_SURVIVAL", [2, 3])
        NODE_COLORMAP = params.get("NODE_COLORMAP", 'gray')
        NODE_COLOR_NORM_VMIN = params.get("NODE_COLOR_NORM_VMIN", 0.0)
        NODE_COLOR_NORM_VMAX = params.get("NODE_COLOR_NORM_VMAX", 1.0)
        COLOR_BY_DEGREE = False
        COLOR_BY_ACTIVE_NEIGHBORS = False
        logger.debug(f"  Applying GoL Params: Birth={GOL_BIRTH}, Survival={GOL_SURVIVAL}")
        try:
            n_birth = min(len(GOL_BIRTH), MAX_VALUES); gol_num_birth[None] = n_birth
            for i in range(n_birth): gol_birth_field[i] = GOL_BIRTH[i]
            n_survival = min(len(GOL_SURVIVAL), MAX_VALUES); gol_num_survival[None] = n_survival
            for i in range(n_survival): gol_survival_field[i] = GOL_SURVIVAL[i]
            logger.info(f"  ---> Copied GoL to Fields: Birth={n_birth}, Survival={n_survival}")
        except Exception as e: logger.error(f"    ERROR copying GoL params to Taichi fields for '{rule_name}': {e}")

    # --- Colored Life ---
    elif rule_key == "colored_life":
        GOL_BIRTH = params.get("GOL_BIRTH", [3])
        GOL_SURVIVAL = params.get("GOL_SURVIVAL", [2, 3])
        NODE_COLORMAP = params.get("NODE_COLORMAP", 'plasma')
        NODE_COLOR_NORM_VMIN = params.get("NODE_COLOR_NORM_VMIN", 0.0)
        NODE_COLOR_NORM_VMAX = params.get("NODE_COLOR_NORM_VMAX", 8.0)
        COLOR_BY_DEGREE = False
        COLOR_BY_ACTIVE_NEIGHBORS = True
        logger.debug(f"  Applying Colored Life Params: Birth={GOL_BIRTH}, Survival={GOL_SURVIVAL}")
        try:
            n_birth = min(len(GOL_BIRTH), MAX_VALUES); gol_num_birth[None] = n_birth
            for i in range(n_birth): gol_birth_field[i] = GOL_BIRTH[i]
            n_survival = min(len(GOL_SURVIVAL), MAX_VALUES); gol_num_survival[None] = n_survival
            for i in range(n_survival): gol_survival_field[i] = GOL_SURVIVAL[i]
            logger.info(f"  ---> Copied Colored Life to Fields: Birth={n_birth}, Survival={n_survival}")
        except Exception as e: logger.error(f"    ERROR copying Colored Life params to Taichi fields for '{rule_name}': {e}")

    # --- Set Visualization Flags ---
    color_by_degree_flag[None] = 1 if COLOR_BY_DEGREE else 0
    color_by_active_neighbors_flag[None] = 1 if COLOR_BY_ACTIVE_NEIGHBORS else 0
    logger.info(f"  Set Viz Flags: Degree={color_by_degree_flag[None]}, Neighbors={color_by_active_neighbors_flag[None]}")
# ==============================================================================
# ==============================================================================

# ==============================================================================
# --- TAICHI HELPER FUNCTIONS ---
# ==============================================================================
# --- Taichi Functions (Helpers) ---
# [clamp, is_close, wrap_idx unchanged]
@ti.func
def clamp(v, v_min, v_max):
    """Clamps value v between v_min and v_max."""
    return ti.max(v_min, ti.min(v_max, v))

@ti.func
def is_close(a: float, b: float, tol: float = 1e-5) -> bool:
    """Checks if two floats are close within a tolerance."""
    return ti.abs(a - b) < tol

@ti.func
def wrap_idx(idx, N):
    """Wraps or clamps index based on WRAP setting."""
    result = 0
    if WRAP: result = idx % N
    else: result = min(max(idx, 0), N - 1)
    return result

# --- MODIFIED: Simplified Kernel Mapping (Ignoring Pan/Zoom for Debugging) ---
@ti.func
def map_grid_to_canvas_kernel(i: int, j: int, current_zoom: float, current_pan_x: float, current_pan_y: float) -> ti.math.vec2: # type: ignore
    """Taichi func to map grid coordinates (i=COL, j=ROW) to canvas pixel coordinates (cx, cy).
       DEBUG VERSION: Ignores pan/zoom, simple scaling."""
    # Use f64 for intermediate calculations
    grid_size_f64 = ti.cast(GRID_SIZE, ti.f64)
    window_size_f64 = ti.cast(WINDOW_SIZE, ti.f64)

    # 1. Normalize grid coordinate (0 to 1), centering the cell
    norm_x = (ti.cast(i, ti.f64) + 0.5) / grid_size_f64 # i (col) -> x
    norm_y = (ti.cast(j, ti.f64) + 0.5) / grid_size_f64 # j (row) -> y

    # --- REMOVED PAN/ZOOM APPLICATION ---
    # panned_norm_x = norm_x - pan_x_f64
    # panned_norm_y = norm_y - pan_y_f64
    # zoomed_norm_x = 0.5 + (panned_norm_x - 0.5) * zoom_f64
    # zoomed_norm_y = 0.5 + (panned_norm_y - 0.5) * zoom_f64
    # ---

    # 4. Scale to window size and flip Y for canvas coordinates (0,0 is top-left)
    # --- Use norm_x, norm_y directly ---
    canvas_x = norm_x * window_size_f64
    canvas_y = (1.0 - norm_y) * window_size_f64 # Y is flipped
    # ---

    # Return as f32 vector for GGUI
    return ti.Vector([ti.cast(canvas_x, ti.f32), ti.cast(canvas_y, ti.f32)])
# --- END MODIFIED ---

# [calculate_*, check_criteria, get_node_color unchanged]
@ti.func
def calculate_clustering_proxy(degree: int, num_neighbors_actual: int, denom_type: int, max_neighbors_theory: int) -> float:
    proxy_clustering = 0.0; f_degree = ti.cast(degree, ti.f32)
    if f_degree > 1.0:
        max_deg_for_calc = 0.0
        if denom_type == 0: max_deg_for_calc = ti.cast(num_neighbors_actual, ti.f32)
        else: max_deg_for_calc = ti.cast(max_neighbors_theory, ti.f32)
        denominator = 1.0
        if max_deg_for_calc > 1.0: denominator = max_deg_for_calc * (max_deg_for_calc - 1.0)
        if denominator > 1e-6: proxy_clustering = (f_degree * (f_degree - 1.0)) / denominator
        else: proxy_clustering = 0.0
    return clamp(proxy_clustering, 0.0, 1.0)

@ti.func
def calculate_betweenness_proxy(degree: int) -> float:
    proxy_betweenness = 0.0; f_degree = ti.cast(degree, ti.f32)
    if f_degree > 1e-6: proxy_betweenness = 1.0 / f_degree
    return proxy_betweenness

@ti.func
def calculate_neighbor_metric_sum_avg(i: int, j: int, metric_type: int, agg_type: int, denom_type: int, max_neighbors_theory: int) -> float:
    metric_sum = 0.0; neighbor_count = 0
    for k in ti.static(range(NUM_NEIGHBORS)):
        di, dj = neighbor_offsets[k]; ni = wrap_idx(i + di, GRID_SIZE); nj = wrap_idx(j + dj, GRID_SIZE)
        neighbor_val = 0.0; num_neighbors_neighbor = 0
        neighbor_degree = ti.cast(node_degree[nj, ni], ti.i32); neighbor_active_count = ti.cast(node_active_neighbors[nj, ni], ti.i32)
        if metric_type == 0: neighbor_val = ti.cast(neighbor_degree, ti.f32)
        elif metric_type == 1:
             if denom_type == 0:
                 for k_inner in ti.static(range(NUM_NEIGHBORS)):
                     di_inner, dj_inner = neighbor_offsets[k_inner]; ni_inner = wrap_idx(ni + di_inner, GRID_SIZE); nj_inner = wrap_idx(nj + dj_inner, GRID_SIZE)
                     if node_degree[nj_inner, ni_inner] > 0: num_neighbors_neighbor += 1
             neighbor_val = calculate_clustering_proxy(neighbor_degree, num_neighbors_neighbor, denom_type, max_neighbors_theory) # type: ignore
        elif metric_type == 2: neighbor_val = calculate_betweenness_proxy(neighbor_degree) # type: ignore
        elif metric_type == 3: neighbor_val = ti.cast(neighbor_active_count, ti.f32)
        metric_sum += neighbor_val; neighbor_count += 1
    result = 0.0
    if agg_type == 0: result = metric_sum
    elif agg_type == 1:
        if neighbor_count > 0: result = metric_sum / ti.cast(neighbor_count, ti.f32)
    return result

@ti.func
def check_criteria(value_to_check: float, ranges_field: ti.template(), num_ranges: int, values_field: ti.template(), num_values: int) -> bool: # type: ignore
    meets = False
    for r_idx in range(num_ranges):
        min_val = ranges_field[r_idx, 0]; max_val = ranges_field[r_idx, 1]
        if min_val <= value_to_check <= max_val: meets = True; break
    if not meets:
        value_as_int = ti.cast(ti.round(value_to_check), ti.i32)
        for v_idx in range(num_values):
            target_value = values_field[v_idx]; is_target_int = is_close(target_value, ti.round(target_value))
            if is_target_int:
                target_as_int = ti.cast(target_value, ti.i32)
                if value_as_int == target_as_int: meets = True; break
            else:
                if is_close(value_to_check, target_value): meets = True; break
    return meets

@ti.func
def get_node_color(i: int, j: int) -> ti.math.vec3: # type: ignore
    """Calculates node color based on flags and data fields. Expects i=COL, j=ROW."""
    node_color = ti.Vector([0.0, 0.0, 0.0]); color_source_val = 0.0
    if color_by_degree_flag[None] == 1: color_source_val = ti.cast(node_degree[j, i], ti.f32)
    elif color_by_active_neighbors_flag[None] == 1: color_source_val = ti.cast(node_active_neighbors[j, i], ti.f32)
    else:
        degree_val = node_degree[j,i]
        if degree_val > 0: color_source_val = ti.cast(degree_val, ti.f32)
        else: color_source_val = ti.cast(node_state[j, i], ti.f32)
    norm_val = 0.0; v_min = NODE_COLOR_NORM_VMIN; v_max = NODE_COLOR_NORM_VMAX
    if (v_max - v_min) > 1e-6: norm_val = clamp((color_source_val - v_min) / (v_max - v_min), 0.0, 1.0)
    is_gol_like_active = node_state[j,i] > 0 and color_by_degree_flag[None] == 0 and color_by_active_neighbors_flag[None] == 0
    if is_gol_like_active and color_source_val > 0: node_color = ti.Vector([1.0, 1.0, 1.0])
    elif color_source_val > 0 or norm_val > 0:
        blue = ti.Vector([0.0, 0.0, 1.0]); yellow = ti.Vector([1.0, 1.0, 0.0])
        node_color = blue * (1.0 - norm_val) + yellow * norm_val
    return node_color
# ==============================================================================

# ==============================================================================
# ==============================================================================

# ==============================================================================
# --- INITIALIZATION KERNEL ---
# ==============================================================================
@ti.kernel
def initialize_kernel(density: float):
    """Initializes all relevant fields based on random density."""
    for i, j in node_state: # Taichi iterates i=row, j=col by default? Let's assume standard (row, col)
        if ti.random() < density:
            node_state[i, j] = 1
        else:
            node_state[i, j] = 0
        node_next_state[i, j] = 0
        node_active_neighbors[i, j] = 0
        # node_eligible[i, j] = 0 # Not used by any active rule here
        node_next_eligible[i, j] = 0
        node_degree[i, j] = 0
        node_next_degree[i, j] = 0
# ==============================================================================

# ==============================================================================
# --- RENDER DATA COPY KERNELS ---
# ==============================================================================
@ti.kernel
def copy_node_data_to_field(pos_arr: ti.types.ndarray(), color_arr: ti.types.ndarray(), count: int): # type: ignore
    """Copies node position and color data from NumPy arrays to Taichi fields and clears unused."""
    num_nodes = ti.cast(min(count, MAX_DRAW_NODES), ti.i32)
    num_nodes_to_draw[None] = num_nodes
    for i in range(num_nodes): # type: ignore
        draw_node_pos[i][0] = pos_arr[i, 0]; draw_node_pos[i][1] = pos_arr[i, 1]
        draw_node_colors[i][0] = color_arr[i, 0]; draw_node_colors[i][1] = color_arr[i, 1]; draw_node_colors[i][2] = color_arr[i, 2]
    for i in range(num_nodes, MAX_DRAW_NODES): # type: ignore
        draw_node_pos[i] = ti.Vector([-10.0, -10.0]); draw_node_colors[i] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def copy_edge_data_to_field(endpoint_arr: ti.types.ndarray(), count: int): # type: ignore
    """Copies edge endpoint data from NumPy array to Taichi field and clears unused."""
    num_endpoints_to_copy = ti.cast(min(count * 2, MAX_DRAW_LINES * 2), ti.i32)
    num_edges_to_draw[None] = num_endpoints_to_copy // 2
    for i in range(num_endpoints_to_copy): # type: ignore
        draw_edge_endpoints[i][0] = endpoint_arr[i, 0]; draw_edge_endpoints[i][1] = endpoint_arr[i, 1]
    for i in range(num_endpoints_to_copy, MAX_DRAW_LINES * 2): # type: ignore
        draw_edge_endpoints[i] = ti.Vector([-10.0, -10.0])
# ==============================================================================

# ==============================================================================
# --- RULE KERNELS: RoL / ROL-U ---
# ==============================================================================
@ti.kernel
def compute_active_neighbors():
    for i, j in node_degree:
        active_neighbors = 0
        for k in ti.static(range(NUM_NEIGHBORS)):
            di, dj = neighbor_offsets[k]; ni = wrap_idx(i + di, GRID_SIZE); nj = wrap_idx(j + dj, GRID_SIZE)
            if node_degree[ni, nj] > 0: active_neighbors += 1
        node_active_neighbors[i, j] = active_neighbors

@ti.kernel
def rol_compute_eligibility():
    """Computes eligibility based on selected metric/aggregation (reads params from fields)."""
    max_neighbors_theory = 8
    for i, j in node_degree:
        current_state_indicator = node_degree[i, j]; metric_value = 0.0; metric_type = 0; agg_type = 0
        denom_type = rolu_clustering_denom_type[None]
        if current_state_indicator <= 0: metric_type = rolu_birth_metric_type[None]; agg_type = rolu_birth_aggregation[None]
        else: metric_type = rolu_survival_metric_type[None]; agg_type = rolu_survival_aggregation[None]
        metric_value = calculate_neighbor_metric_sum_avg(i, j, metric_type, agg_type, denom_type, max_neighbors_theory) # type: ignore
        eligible = 0; passes_check = False
        if current_state_indicator <= 0: passes_check = check_criteria(metric_value, rol_birth_ranges_field, rol_num_birth_ranges[None], rol_birth_values_field, rol_num_birth_values[None]) # type: ignore
        else: passes_check = check_criteria(metric_value, rol_survival_ranges_field, rol_num_survival_ranges[None], rol_survival_values_field, rol_num_survival_values[None]) # type: ignore
        if passes_check: eligible = 1
        node_next_eligible[i, j] = eligible

@ti.kernel
def rol_compute_edges_and_degree():
    """Computes next degree based on mutually eligible neighbors."""
    for i, j in node_degree:
        degree = 0
        if node_next_eligible[i, j] > 0:
            for k in ti.static(range(NUM_NEIGHBORS)):
                di, dj = neighbor_offsets[k]; ni = wrap_idx(i + di, GRID_SIZE); nj = wrap_idx(j + dj, GRID_SIZE)
                if node_next_eligible[ni, nj] > 0: degree += 1
        node_next_degree[i, j] = degree

@ti.kernel
def rol_finalize_state():
    """Finalizes the node degree based on eligibility and final life/death rules."""
    max_neighbors_theory = 8
    for i, j in node_degree:
        calculated_degree = ti.cast(node_next_degree[i, j], ti.i32); is_eligible = node_next_eligible[i, j] > 0
        final_state = 0
        if is_eligible:
            final_metric_type = rolu_final_check_metric[None]; final_metric_value = 0.0; denom_type = rolu_clustering_denom_type[None]
            if final_metric_type == 0: final_metric_value = ti.cast(calculated_degree, ti.f32)
            elif final_metric_type == 1:
                 num_eligible_neighbors = 0
                 for k in ti.static(range(NUM_NEIGHBORS)):
                     di, dj = neighbor_offsets[k]; ni = wrap_idx(i + di, GRID_SIZE); nj = wrap_idx(j + dj, GRID_SIZE)
                     if node_next_eligible[ni, nj] > 0: num_eligible_neighbors += 1
                 final_metric_value = calculate_clustering_proxy(calculated_degree, num_eligible_neighbors, denom_type, max_neighbors_theory) # type: ignore
            elif final_metric_type == 2: final_metric_value = calculate_betweenness_proxy(calculated_degree) # type: ignore
            elif final_metric_type == 3:
                 num_eligible_neighbors = 0
                 for k in ti.static(range(NUM_NEIGHBORS)):
                     di, dj = neighbor_offsets[k]; ni = wrap_idx(i + di, GRID_SIZE); nj = wrap_idx(j + dj, GRID_SIZE)
                     if node_next_eligible[ni, nj] > 0: num_eligible_neighbors += 1
                 final_metric_value = ti.cast(num_eligible_neighbors, ti.f32)
            meets_life = check_criteria(final_metric_value, rol_final_life_ranges_field, rol_num_final_life_ranges[None], rol_final_life_values_field, rol_num_final_life_values[None]) # type: ignore
            if meets_life: final_state = calculated_degree
            else:
                meets_death = check_criteria(final_metric_value, rol_final_death_ranges_field, rol_num_final_death_ranges[None], rol_final_death_values_field, rol_num_final_death_values[None]) # type: ignore
                if not meets_death: final_state = calculated_degree
        node_degree[i, j] = final_state

# ==============================================================================

# ==============================================================================
# --- RULE KERNELS: GoL / Colored Life ---
# ==============================================================================
@ti.kernel
def gol_compute_neighbors():
    for i, j in node_state:
        active_neighbors = 0
        for k in ti.static(range(NUM_NEIGHBORS)):
            di, dj = neighbor_offsets[k]; ni = wrap_idx(i + di, GRID_SIZE); nj = wrap_idx(j + dj, GRID_SIZE)
            if node_state[ni, nj] > 0: active_neighbors += 1
        node_active_neighbors[i, j] = active_neighbors

@ti.kernel
def gol_update_state():
    for i, j in node_state:
        active_neighbors = node_active_neighbors[i, j]; current_state = node_state[i, j]; next_state = 0
        survives = False; born = False
        if current_state > 0:
            num_survival = ti.cast(gol_num_survival[None], ti.i32)
            for s_idx in range(num_survival): # type: ignore
                if active_neighbors == gol_survival_field[s_idx]: survives = True; break
        else:
            num_birth = ti.cast(gol_num_birth[None], ti.i32)
            for b_idx in range(num_birth): # type: ignore
                if active_neighbors == gol_birth_field[b_idx]: born = True; break
        if survives or born: next_state = 1
        node_next_state[i, j] = next_state
# ==============================================================================

# ==============================================================================
# --- RENDER DATA COLLECTION KERNELS ---
# ==============================================================================

@ti.kernel
def collect_rol_render_data_kernel(current_zoom: float, current_pan_x: float, current_pan_y: float,
                                   node_radius_px: float):
    """Collects visible node and edge data for RoL/ROL-U rules."""
    num_nodes_to_draw[None] = 0; num_edges_to_draw[None] = 0
    for j, i in node_degree: # Iterate rows (j), columns (i)
        if node_degree[j, i] > 0: # Check activity based on degree
            cx, cy = map_grid_to_canvas_kernel(i, j, current_zoom, current_pan_x, current_pan_y) # Pass (col, row)
            margin = node_radius_px * 1.5; is_visible1 = -margin < cx < WINDOW_SIZE + margin and -margin < cy < WINDOW_SIZE + margin
            if is_visible1:
                current_node_count = ti.cast(num_nodes_to_draw[None], ti.i32)
                if current_node_count < MAX_DRAW_NODES: # type: ignore
                    draw_idx = ti.atomic_add(num_nodes_to_draw[None], 1)
                    if draw_idx < MAX_DRAW_NODES: # type: ignore
                        draw_node_pos[draw_idx] = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                        draw_node_colors[draw_idx] = get_node_color(i, j) # type: ignore
                for k in ti.static(range(NUM_NEIGHBORS)):
                    di, dj = neighbor_offsets[k]; ni_raw, nj_raw = i + dj, j + di
                    neighbor_out_of_bounds = not (0 <= ni_raw < GRID_SIZE and 0 <= nj_raw < GRID_SIZE)
                    if not (WRAP and neighbor_out_of_bounds):
                        ni = wrap_idx(ni_raw, GRID_SIZE); nj = wrap_idx(nj_raw, GRID_SIZE)
                        if node_degree[nj, ni] > 0:
                            idx1 = j * GRID_SIZE + i; idx2 = nj * GRID_SIZE + ni
                            if idx1 < idx2:
                                current_edge_count = ti.cast(num_edges_to_draw[None], ti.i32)
                                if current_edge_count < MAX_DRAW_LINES: # type: ignore
                                    draw_idx_base = ti.atomic_add(num_edges_to_draw[None], 1) * 2 # type: ignore
                                    if ti.cast(draw_idx_base, ti.i32) + ti.i32(1) < MAX_DRAW_LINES * 2: # type: ignore
                                        cx2_raw, cy2_raw = map_grid_to_canvas_kernel(ni_raw, nj_raw, current_zoom, current_pan_x, current_pan_y)
                                        draw_edge_endpoints[draw_idx_base] = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                                        draw_edge_endpoints[draw_idx_base + ti.i32(1)] = ti.Vector([cx2_raw / WINDOW_SIZE, cy2_raw / WINDOW_SIZE])

@ti.kernel
def collect_gol_render_data_kernel(current_zoom: float, current_pan_x: float, current_pan_y: float, node_radius_px: float):
    """Collects visible node data for Game of Life (no edges)."""
    num_nodes_to_draw[None] = 0; num_edges_to_draw[None] = 0
    for j, i in node_state:
        if node_state[j, i] > 0:
            cx, cy = map_grid_to_canvas_kernel(i, j, current_zoom, current_pan_x, current_pan_y)
            margin = node_radius_px * 1.5; is_visible = -margin < cx < WINDOW_SIZE + margin and -margin < cy < WINDOW_SIZE + margin
            if is_visible:
                current_count = ti.cast(num_nodes_to_draw[None], ti.i32)
                if current_count < MAX_DRAW_NODES: # type: ignore
                    draw_idx = ti.atomic_add(num_nodes_to_draw[None], 1)
                    if draw_idx < MAX_DRAW_NODES: # type: ignore
                        draw_node_pos[draw_idx] = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                        draw_node_colors[draw_idx] = get_node_color(i, j) # type: ignore

@ti.kernel
def collect_colored_life_render_data_kernel(current_zoom: float, current_pan_x: float, current_pan_y: float, node_radius_px: float):
    """Collects visible node data for Colored Life (no edges)."""
    num_nodes_to_draw[None] = 0; num_edges_to_draw[None] = 0
    for j, i in node_state:
        if node_state[j, i] > 0:
            cx, cy = map_grid_to_canvas_kernel(i, j, current_zoom, current_pan_x, current_pan_y)
            margin = node_radius_px * 1.5; is_visible = -margin < cx < WINDOW_SIZE + margin and -margin < cy < WINDOW_SIZE + margin
            if is_visible:
                current_count = ti.cast(num_nodes_to_draw[None], ti.i32)
                if current_count < MAX_DRAW_NODES: # type: ignore
                    draw_idx = ti.atomic_add(num_nodes_to_draw[None], 1)
                    if draw_idx < MAX_DRAW_NODES: # type: ignore
                        draw_node_pos[draw_idx] = ti.Vector([cx / WINDOW_SIZE, cy / WINDOW_SIZE])
                        draw_node_colors[draw_idx] = get_node_color(i, j) # type: ignore

# --- MODIFIED: Explicitly clear buffer contents ---
@ti.kernel
def clear_render_data_kernel():
    """Clears the Taichi fields used for drawing nodes and edges."""
    num_nodes_to_draw[None] = 0
    num_edges_to_draw[None] = 0
    # Explicitly clear arrays to avoid potential artifacts
    for i in range(MAX_DRAW_NODES): # type: ignore
        draw_node_pos[i] = ti.Vector([-10.0, -10.0]) # Off-screen
        draw_node_colors[i] = ti.Vector([0.0, 0.0, 0.0]) # Black
    for i in range(MAX_DRAW_LINES * 2): # type: ignore
        draw_edge_endpoints[i] = ti.Vector([-10.0, -10.0]) # Off-screen
# --- END MODIFIED ---

# --- RESTORED: Original render_nodes_and_edges ---
# Modify function signature to accept calculated sizes
def render_nodes_and_edges(current_node_radius_px, current_edge_width_px):
    """Renders nodes as circles and edges as lines directly onto the canvas."""
    # logger.debug(f"Starting render_nodes_and_edges with radius={current_node_radius_px}, width={current_edge_width_px}")
    rule_key = rule_keys[selected_rule_idx]

    # --- Log Current Colormap ---
    logger.debug(f"Render func using NODE_COLORMAP='{NODE_COLORMAP}', VMIN={NODE_COLOR_NORM_VMIN}, VMAX={NODE_COLOR_NORM_VMAX}")
    # ---

    # --- Data Preparation ---
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
    elif rule_key == "realm_of_lace_unified": # Added ROL-U case
        degree_np = node_degree.to_numpy() # ROL-U state is degree
        active_map = degree_np > 0
        if COLOR_BY_DEGREE: color_data = degree_np
        elif COLOR_BY_ACTIVE_NEIGHBORS: color_data = node_active_neighbors.to_numpy()
        else: color_data = degree_np # Default to degree
    else: return
    if active_map is None or color_data is None:
        logger.error("Failed to get necessary numpy arrays for rendering."); return

    # --- Determine Visible Grid Range ---
    g_i00, g_j00 = map_canvas_to_grid_approx(0, 0)
    g_i11, g_j11 = map_canvas_to_grid_approx(WINDOW_SIZE, WINDOW_SIZE)
    min_i_visible = math.floor(min(g_i00, g_i11)); max_i_visible = math.ceil(max(g_i00, g_i11))
    min_j_visible = math.floor(min(g_j00, g_j11)); max_j_visible = math.ceil(max(g_j00, g_j11))
    buffer_grid_cells = 10
    view_x0 = max(0, min_i_visible - buffer_grid_cells); view_y0 = max(0, min_j_visible - buffer_grid_cells)
    view_x1 = min(GRID_SIZE, max_i_visible + buffer_grid_cells); view_y1 = min(GRID_SIZE, max_j_visible + buffer_grid_cells)
    if view_x1 <= view_x0: view_x1 = view_x0 + 1
    if view_y1 <= view_y0: view_y1 = view_y0 + 1
    view_x1 = min(GRID_SIZE, view_x1); view_y1 = min(GRID_SIZE, view_y1)

    # --- Check if any active nodes are in the calculated range BEFORE iterating ---
    active_nodes_in_slice = 0
    if view_x1 > view_x0 and view_y1 > view_y0:
        try:
            visible_slice = active_map[view_y0:view_y1, view_x0:view_x1]
            active_nodes_in_slice = np.count_nonzero(visible_slice)
            # if active_nodes_in_slice > 0: logger.debug(f"Slice check found {active_nodes_in_slice} active nodes in [{view_y0}:{view_y1}, {view_x0}:{view_x1}]. Proceeding.") # Reduce noise
        except IndexError as e: logger.error(f"IndexError accessing active_map slice [{view_y0}:{view_y1}, {view_x0}:{view_x1}]: {e}"); active_nodes_in_slice = -1
    else: active_nodes_in_slice = 0
    if active_nodes_in_slice <= 0: num_nodes_to_draw[None] = 0; num_edges_to_draw[None] = 0; return

    # --- Prepare Drawing Data in Python Lists ---
    node_positions_list = []; node_colors_list = []; edge_endpoints_list = []; processed_edges = set()
    try: cmap = cm.get_cmap(NODE_COLORMAP)
    except ValueError: logger.warning(f"Invalid colormap '{NODE_COLORMAP}', defaulting to 'viridis'."); cmap = cm.get_cmap('viridis')
    def python_wrap_idx(idx, N): return idx % N if WRAP else min(max(idx, 0), N - 1)

    # --- Iterate over POTENTIALLY VISIBLE grid and Collect Data ---
    start_collect_time = time.perf_counter()
    nodes_collected_count = 0; first_active_logged = False
    for j in range(view_y0, view_y1): # Iterate rows
        for i in range(view_x0, view_x1): # Iterate columns
            if not (0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE): continue
            is_active = active_map[j, i] # Access active_map with [row, col]
            if not is_active: continue
            if not first_active_logged: first_active_logged = True
            nodes_collected_count += 1
            cx, cy = map_grid_to_canvas(i, j) # Pass (col, row)
            margin = current_node_radius_px * 1.5
            if -margin <= cx <= WINDOW_SIZE + margin and -margin <= cy <= WINDOW_SIZE + margin:
                norm_cx = cx / WINDOW_SIZE; norm_cy = cy / WINDOW_SIZE
                node_positions_list.append((norm_cx, norm_cy))
                node_val = color_data[j, i] # Access color_data with [row, col]
                if rule_key == "game_of_life": node_color = (1.0, 1.0, 1.0)
                else:
                    norm_val = 0.0
                    if (NODE_COLOR_NORM_VMAX - NODE_COLOR_NORM_VMIN) > 1e-6: norm_val = np.clip((node_val - NODE_COLOR_NORM_VMIN) / (NODE_COLOR_NORM_VMAX - NODE_COLOR_NORM_VMIN), 0, 1)
                    node_color = cmap(norm_val)[:3]
                node_colors_list.append(node_color)
                # Calculate Edges (Only for RoL/ROL-U in this simplified version)
                if rule_key == "realm_of_lace" or rule_key == "realm_of_lace_unified":
                    for k in range(NUM_NEIGHBORS):
                        di, dj = neighbor_offsets[k] # di is row offset, dj is col offset
                        ni_raw, nj_raw = i + dj, j + di # Apply offset as (col+dj, row+di)
                        neighbor_out_of_bounds = not (0 <= ni_raw < GRID_SIZE and 0 <= nj_raw < GRID_SIZE)
                        edge_wraps = WRAP and neighbor_out_of_bounds
                        if edge_wraps: continue
                        ni_idx = python_wrap_idx(ni_raw, GRID_SIZE) # Col index of neighbor
                        nj_idx = python_wrap_idx(nj_raw, GRID_SIZE) # Row index of neighbor
                        if not (0 <= ni_idx < GRID_SIZE and 0 <= nj_idx < GRID_SIZE): continue
                        if node_degree[nj_idx, ni_idx] > 0: # Use [row, col] for node_degree check
                            node1_idx = j * GRID_SIZE + i # Flat index of current node (row, col)
                            node2_idx = nj_idx * GRID_SIZE + ni_idx # Flat index of neighbor (row, col)
                            edge_tuple = tuple(sorted((node1_idx, node2_idx)))
                            if edge_tuple not in processed_edges:
                                ncx, ncy = map_grid_to_canvas(ni_raw, nj_raw) # Pass (col, row)
                                edge_endpoints_list.append((norm_cx, norm_cy))
                                edge_endpoints_list.append((ncx / WINDOW_SIZE, ncy / WINDOW_SIZE))
                                processed_edges.add(edge_tuple)

    collect_duration = time.perf_counter() - start_collect_time
    num_nodes_in_list = len(node_positions_list); num_edges_collected = len(edge_endpoints_list) // 2
    logger.debug(f"Data collection took {collect_duration:.4f}s. Active in Loop: {nodes_collected_count}, Nodes in List: {num_nodes_in_list}, Edges in List: {num_edges_collected}")
    if num_nodes_in_list == 0 and num_edges_collected == 0: num_nodes_to_draw[None] = 0; num_edges_to_draw[None] = 0; return

    # --- Copy Data to Taichi Fields ---
    num_nodes = num_nodes_in_list; num_edges = num_edges_collected
    if num_nodes > 0:
        if num_nodes > MAX_DRAW_NODES: logger.warning(f"Truncating {num_nodes} nodes to {MAX_DRAW_NODES}."); num_nodes = MAX_DRAW_NODES
        node_pos_np = np.array(node_positions_list[:num_nodes], dtype=np.float32); node_colors_np = np.array(node_colors_list[:num_nodes], dtype=np.float32)
        copy_node_data_to_field(node_pos_np, node_colors_np, num_nodes)
    else: num_nodes_to_draw[None] = 0
    if num_edges > 0:
        if num_edges > MAX_DRAW_LINES: logger.warning(f"Truncating {num_edges} edges to {MAX_DRAW_LINES}."); num_edges = MAX_DRAW_LINES
        edge_endpoints_np = np.array(edge_endpoints_list[:num_edges * 2], dtype=np.float32)
        copy_edge_data_to_field(edge_endpoints_np, num_edges)
    else: num_edges_to_draw[None] = 0

    # --- Draw using Taichi GGUI with Taichi Fields ---
    actual_edges = num_edges_to_draw[None]
    if actual_edges > 0:
        width_norm = max(0.0, current_edge_width_px / max(WINDOW_SIZE, 1))
        canvas.lines(draw_edge_endpoints, width=width_norm, color=EDGE_COLOR)
    actual_nodes = num_nodes_to_draw[None]
    if actual_nodes > 0:
        draw_radius_px = max(1.0, current_node_radius_pixels + current_edge_width_px * 0.5)
        radius_norm = max(0.0, draw_radius_px / max(WINDOW_SIZE, 1))
        canvas.circles(draw_node_pos, radius=radius_norm, per_vertex_color=draw_node_colors)
# --- END RESTORED ---

# ==============================================================================

# ==============================================================================

# ==============================================================================
# --- PYTHON HELPER FUNCTIONS ---
# ==============================================================================
# --- Python Helper Functions ---
# [initialize, rol_step, rolu_step, gol_step, colored_life_step unchanged]
def initialize():
    """Calls the initialization kernel and prepares for the selected rule."""
    logger.info("Initializing grid...")
    initialize_kernel(INITIAL_DENSITY)
    current_rule_key = rule_keys[selected_rule_idx]
    if current_rule_key == "realm_of_lace" or current_rule_key == "realm_of_lace_unified":
         logger.info(f"Rule is {current_rule_key}, copying initial node_state to node_degree.")
         node_degree.copy_from(node_state)
         logger.debug(f"  Initial node_degree sum after copy: {node_degree.to_numpy().sum()}")
    elif current_rule_key == "colored_life":
        gol_compute_neighbors()
        logger.debug("Calculated initial active neighbors for Colored Life.")
    logger.info("Initialization complete.")

def rol_step():
    """Performs one step of the Realm of Lace (Original/Unified) simulation."""
    if rule_keys[selected_rule_idx] == "realm_of_lace_unified":
        birth_metric = rolu_birth_metric_type[None]; survival_metric = rolu_survival_metric_type[None]; final_metric = rolu_final_check_metric[None]
        needs_active_neighbors = (birth_metric == 3 or survival_metric == 3 or final_metric == 3)
        if needs_active_neighbors: compute_active_neighbors()
    rol_compute_eligibility(); rol_compute_edges_and_degree(); rol_finalize_state()

def rolu_step(): rol_step()
def gol_step(): gol_compute_neighbors(); gol_update_state(); node_state.copy_from(node_next_state)
def colored_life_step(): gol_compute_neighbors(); gol_update_state(); node_state.copy_from(node_next_state)

# --- Visualization: Node grid as image ---
# --- RESTORED: Original get_image_window ---
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
        # --- ADDED: ROL-U case using its logic ---
        elif rule_key == "realm_of_lace_unified":
            degree_np = node_degree.to_numpy()[y0:y1, x0:x1]
            activity_slice = degree_np > 0
            color_data_np = None
            if COLOR_BY_DEGREE: color_data_np = degree_np
            elif COLOR_BY_ACTIVE_NEIGHBORS: color_data_np = node_active_neighbors.to_numpy()[y0:y1, x0:x1]
            else: color_data_np = degree_np # Default to degree
            if color_data_np is not None:
                if current_vmax <= current_vmin: current_vmax = current_vmin + 1e-6
                normed = np.clip((color_data_np - current_vmin) / (current_vmax - current_vmin), 0, 1)
                cmap = cm.get_cmap(current_cmap)
                rgb = (cmap(normed)[..., :3] * 255).astype(np.uint8)
                mask = ~activity_slice
                rgb[mask] = [0, 0, 0]
                img = rgb
            else: img = np.zeros((h, w, 3), dtype=np.uint8)
        # --- END ADDED ---
        if img is None:
             logger.warning(f"Unhandled rule key '{rule_key}' in get_image_window. Returning black.")
             img = np.zeros((h, w, 3), dtype=np.uint8)
    except Exception as e:
        logger.error(f"Error during get_image_window for rule {rule_key}: {e}", exc_info=True)
        img = np.zeros((h, w, 3), dtype=np.uint8)
    if img is not None and img.shape[0] > 0 and img.shape[1] > 0: return cv2.resize(img, (WINDOW_SIZE, WINDOW_SIZE), interpolation=cv2.INTER_NEAREST)
    else: logger.warning("Image generation resulted in invalid shape or None."); return np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
# --- END RESTORED ---

# --- Coordinate Mapping Functions ---
# --- RESTORED: Original map_grid_to_canvas ---
def map_grid_to_canvas(i, j):
    """Maps grid coordinates (i, j) to canvas pixel coordinates (cx, cy). (Original Logic)"""
    norm_x = i / GRID_SIZE
    norm_y = j / GRID_SIZE
    grid_center_x = 0.5 + pan_x
    grid_center_y = 0.5 + pan_y
    offset_x_norm = norm_x - grid_center_x
    offset_y_norm = norm_y - grid_center_y
    canvas_norm_x = 0.5 + offset_x_norm * zoom
    canvas_norm_y = 0.5 + offset_y_norm * zoom
    canvas_x = canvas_norm_x * WINDOW_SIZE
    canvas_y = (1.0 - canvas_norm_y) * WINDOW_SIZE
    return canvas_x, canvas_y
# --- END RESTORED ---

# --- RESTORED: Original map_canvas_to_grid_approx ---
def map_canvas_to_grid_approx(px, py):
    """Approximates the grid coordinates (i, j) corresponding to canvas pixel coordinates (px, py). (Original Logic)"""
    norm_canvas_x = px / WINDOW_SIZE
    norm_canvas_y = 1.0 - (py / WINDOW_SIZE) # Flip Y back
    offset_x_norm = (norm_canvas_x - 0.5) / zoom
    offset_y_norm = (norm_canvas_y - 0.5) / zoom
    grid_center_x = 0.5 + pan_x
    grid_center_y = 0.5 + pan_y
    norm_x_grid = offset_x_norm + grid_center_x
    norm_y_grid = offset_y_norm + grid_center_y
    grid_i = norm_x_grid * GRID_SIZE
    grid_j = norm_y_grid * GRID_SIZE
    return grid_i, grid_j
# --- END RESTORED ---
# ==============================================================================

# ==============================================================================
# ==============================================================================

# --- GGUI Setup & Event Handling ---
window = ti.ui.Window("Taichi CA Demo", res=(WINDOW_SIZE, WINDOW_SIZE), vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()

# --- GUI State ---
zoom = 1.0; pan_x = 0.0; pan_y = 0.0
zoom_slider = 1.0; pan_x_slider = 0.0; pan_y_slider = 0.0

# --- Mouse Panning State ---
mouse_dragging = False; last_mouse_x = 0.0; last_mouse_y = 0.0

# --- ADDED: Global variables for GUI panel bounds ---
gui_panel_x = 0.01
gui_panel_y = 0.01
gui_panel_w = 0.3
gui_panel_h = 0.36 # Initial height, will be updated in handle_gui
# ---

def clamp_pan(pan, zoom):
    safe_zoom = max(zoom, 1.0); max_pan = max(0.0, 1.0 - 1.0 / safe_zoom)
    clamped_pan = np.clip(pan, -max_pan, max_pan); return clamped_pan

def handle_gui():
    """Manages the GUI elements and updates corresponding global variables."""
    global zoom_slider, zoom, pan_x, pan_y, INITIAL_DENSITY, STEP_DELAY, paused, frame, speed_slider_value, selected_rule_idx, RENDER_EDGES
    # --- ADDED: Update global GUI panel height ---
    global gui_panel_h
    # ---

    # --- MODIFIED: Store height used by gui.begin ---
    control_height = 0.36 # Adjusted height
    gui_panel_h = control_height # Update global variable
    # ---
    gui.begin("Controls", gui_panel_x, gui_panel_y, gui_panel_w, gui_panel_h)

    # [ Rest of handle_gui remains unchanged ]
    # Rule Selection
    gui.text("Rule Selection"); gui.text(f"Current: {rule_names[selected_rule_idx]}")
    rule_changed = False
    if gui.button("Previous Rule"): selected_rule_idx = (selected_rule_idx - 1 + len(rule_names)) % len(rule_names); rule_changed = True
    if gui.button("Next Rule"): selected_rule_idx = (selected_rule_idx + 1) % len(rule_names); rule_changed = True
    if rule_changed: logger.info(f"--- Rule changed via GUI to index {selected_rule_idx} ---"); set_rule_variant(selected_rule_idx); initialize(); clear_render_data_kernel(); frame = 0; logger.info("Rule changed, grid re-initialized, render data cleared.")
    # Visualization Style
    gui.text("Visualization Style")
    render_edges_checkbox_value = gui.checkbox("Render Nodes & Edges", RENDER_EDGES)
    if render_edges_checkbox_value != RENDER_EDGES: RENDER_EDGES = render_edges_checkbox_value; logger.info(f"Global RENDER_EDGES updated to: {RENDER_EDGES}")
    # Zoom Buttons
    if gui.button("Zoom to Edges"):
        target_zoom = MIN_NODE_SEPARATION_PIXELS * GRID_SIZE / WINDOW_SIZE; target_zoom = max(3.0, target_zoom)
        logger.info(f"Zoom to Edges pressed. Setting Zoom: {target_zoom:.2f}. Resetting pan."); zoom = target_zoom; zoom_slider = zoom; pan_x = 0.0; pan_y = 0.0
        if not RENDER_EDGES: RENDER_EDGES = True; logger.info("Enabled Render Nodes & Edges.")
    if gui.button("Zoom Out Full"): logger.info("Zoom Out Full pressed."); zoom = 1.0; zoom_slider = zoom; pan_x = 0.0; pan_y = 0.0
    # Zoom Control
    gui.text("Zoom Control"); zoom_slider = gui.slider_float("Zoom", zoom_slider, 1.0, 40.0)
    # Simulation Controls
    gui.text("Simulation Controls"); INITIAL_DENSITY = gui.slider_float("Init Density", INITIAL_DENSITY, 0.01, 1.0)
    speed_slider_value = gui.slider_float("Speed (steps/sec)", speed_slider_value, 1.0, 240.0); STEP_DELAY = 1.0 / max(speed_slider_value, 1.0)
    if gui.button("Pause" if not paused else "Resume"): paused = not paused
    if gui.button("Reset"): logger.info("--- Reset button pressed ---"); initialize(); clear_render_data_kernel(); frame = 0
    gui.end()
    zoom = zoom_slider

# --- MODIFIED: Add check for mouse over GUI panel ---
def handle_mouse(window, use_node_edge_mode: bool):
    """Handles mouse input events for panning, ignoring events over the GUI."""
    global mouse_dragging, last_mouse_x, last_mouse_y, pan_x, pan_y

    while window.get_event(): pass # Consume events
    is_lmb_pressed_now = window.is_pressed(ti.ui.LMB)
    current_mouse_x, current_mouse_y = window.get_cursor_pos() # Coords are 0.0-1.0

    # --- Check if CURRENT mouse position is inside the GUI panel bounds ---
    is_over_gui = (gui_panel_x <= current_mouse_x <= gui_panel_x + gui_panel_w and
                   gui_panel_y <= current_mouse_y <= gui_panel_y + gui_panel_h)
    # ---

    if is_lmb_pressed_now and not mouse_dragging:
        # Start drag ONLY if not initially pressed over the GUI
        if not is_over_gui:
            mouse_dragging = True; last_mouse_x, last_mouse_y = current_mouse_x, current_mouse_y
            logger.info(f"Mouse drag started at ({last_mouse_x:.3f}, {last_mouse_y:.3f})")
        else:
            logger.debug("Mouse press ignored (started over GUI panel).")
    elif not is_lmb_pressed_now and mouse_dragging:
        mouse_dragging = False; logger.info("Mouse drag ended.")

    # --- Process pan ONLY if dragging AND not currently over the GUI ---
    if mouse_dragging and not is_over_gui:
        dx_norm = current_mouse_x - last_mouse_x; dy_norm = current_mouse_y - last_mouse_y
        if abs(dx_norm) > 1e-6 or abs(dy_norm) > 1e-6:
            if zoom > 1.0:
                delta_pan_x = 0.0; delta_pan_y = 0.0
                if use_node_edge_mode: delta_pan_x = -dx_norm / zoom; delta_pan_y = +dy_norm / zoom
                else: delta_pan_x = -dx_norm / zoom; delta_pan_y = -dy_norm / zoom
                new_pan_x = pan_x + delta_pan_x; new_pan_y = pan_y + delta_pan_y
                pan_x = clamp_pan(new_pan_x, zoom); pan_y = clamp_pan(new_pan_y, zoom)
            last_mouse_x = current_mouse_x; last_mouse_y = current_mouse_y
    elif mouse_dragging and is_over_gui:
        # If dragging but mouse moved over GUI, stop panning but keep tracking last pos
        logger.debug("Mouse drag moved over GUI panel, panning paused.")
        last_mouse_x = current_mouse_x; last_mouse_y = current_mouse_y
    elif mouse_dragging and not is_lmb_pressed_now: # Fallback state reset
         mouse_dragging = False
    # --- END MODIFIED ---

def handle_keyboard(window):
    """Handles keyboard input events."""
    global paused
    if window.get_event(ti.ui.PRESS):
        if window.event.key == ti.ui.SPACE:
             paused = not paused
             logger.debug(f"Pause toggled via keyboard: {paused}")
# ==============================================================================

# --- Main Loop ---
initialize()
frame = 0
logger.info("--- Entering Main Loop ---")
while window.running:
    # Process inputs first
    handle_keyboard(window)

    # --- Determine Rendering Mode *Before* Handling Mouse ---
    # --- RESTORED: Original logic using map_grid_to_canvas(row, col) ---
    px0, py0 = map_grid_to_canvas(0, 0) # Pass (row, col)
    px1, py1 = map_grid_to_canvas(1, 0) # Pass (row, col)
    dist_sq = (px1 - px0)**2 + (py1 - py0)**2
    projected_separation_pixels = np.sqrt(dist_sq) if dist_sq > 0 else 0
    separation_check = projected_separation_pixels >= MIN_NODE_SEPARATION_PIXELS
    use_node_edge_render_mode = RENDER_EDGES and separation_check
    # --- END RESTORED ---
    # logger.debug(f"Frame {frame}: SepPixels={projected_separation_pixels:.2f}, EdgeThreshold={MIN_NODE_SEPARATION_PIXELS:.2f}, RENDER_EDGES={RENDER_EDGES} -> UseNodeEdgeMode={use_node_edge_render_mode}") # Reduce noise
    # --- End Determine Rendering Mode ---

    # --- Calculate dynamic radius/width HERE ---
    min_radius_at_threshold = max(1.0, MIN_NODE_SEPARATION_PIXELS / (2.0 * 1.2))
    scaling_range = max(1e-6, MAX_SEP_FOR_NODE_SCALING - MIN_NODE_SEPARATION_PIXELS)
    scale_factor = np.clip((projected_separation_pixels - MIN_NODE_SEPARATION_PIXELS) / scaling_range, 0.0, 1.0)
    current_node_radius_pixels = max(1.0, min_radius_at_threshold + scale_factor * (BASE_NODE_RADIUS - min_radius_at_threshold))
    current_edge_width_px = BASE_EDGE_WIDTH
    # --- END ---

    handle_mouse(window, use_node_edge_render_mode)
    handle_gui()

    rule_key = rule_keys[selected_rule_idx]

    # --- Simulation Step ---
    if not paused:
        if rule_key == "realm_of_lace": rol_step()
        elif rule_key == "realm_of_lace_unified": rolu_step()
        elif rule_key == "game_of_life": gol_step()
        elif rule_key == "colored_life": colored_life_step()
        frame += 1

    # --- Visualization ---
    canvas.set_background_color((0, 0, 0))
    # logger.debug(f"Frame {frame}: Rendering with Zoom={zoom:.3f}, PanX={pan_x:.3f}, PanY={pan_y:.3f}, Mode={'Node/Edge' if use_node_edge_render_mode else 'Image'}") # Reduce noise
    clear_render_data_kernel()

    if use_node_edge_render_mode:
        # --- RESTORED: Call original render function ---
        render_nodes_and_edges(current_node_radius_pixels, current_edge_width_px)
        # --- END RESTORED ---

    else: # Image Rendering
        # --- RESTORED: Original image view calculation ---
        view_size_pixels = GRID_SIZE / zoom
        view_size_half_pixels = view_size_pixels / 2.0
        cx_pixels = GRID_SIZE * (0.5 + pan_x)
        cy_pixels = GRID_SIZE * (0.5 + pan_y)
        x0 = int(np.clip(cx_pixels - view_size_half_pixels, 0, GRID_SIZE)) # Column start
        y0 = int(np.clip(cy_pixels - view_size_half_pixels, 0, GRID_SIZE)) # Row start
        x1 = int(np.clip(x0 + view_size_pixels, 0, GRID_SIZE)) # Column end
        y1 = int(np.clip(y0 + view_size_pixels, 0, GRID_SIZE)) # Row end
        # --- END RESTORED ---

        # logger.debug(f"  Image Mode: View Window (Grid Coords): cols={x0}-{x1}, rows={y0}-{y1}") # Reduce noise
        view_width = x1 - x0; view_height = y1 - y0
        if view_width > 0 and view_height > 0:
            img_resized = get_image_window(x0, x1, y0, y1) # Pass (col, col, row, row)
            ti.sync()
            canvas.set_image(img_resized)
        else:
            canvas.set_image(np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8))

    # --- Info Display (Unchanged) ---
    control_height = 0.38; info_y = 0.01 + control_height + 0.01
    gui.begin("Info", 0.01, info_y, 0.3, 0.18)
    gui.text(f"Frame: {frame}"); gui.text(f"Rule: {rule_names[selected_rule_idx]}")
    gui.text(f"Zoom: {zoom:.2f}"); gui.text(f"Pan: ({pan_x:.2f}, {pan_y:.2f})")
    gui.text(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}"); gui.text(f"Paused: {paused}")
    gui.text(f"Init Density: {INITIAL_DENSITY:.2f}"); gui.text(f"Speed (steps/sec): {speed_slider_value:.1f}")
    gui.text(f"Render Edges: {RENDER_EDGES}")
    gui.end()

    ti.sync()
    window.show()
    ti.sync()

logger.info("--- Exited Main Loop ---")
# ==============================================================================