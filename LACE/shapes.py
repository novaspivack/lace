# =========== START of shapes.py ===========
from __future__ import annotations
import json
import shutil
import concurrent.futures
import queue # For thread-safe queue
import threading
import setproctitle
import copy
from datetime import datetime
import dataclasses
from tkinter import simpledialog
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from matplotlib.figure import Figure 
from typing import Dict, List, Set, Tuple, Optional, Union, Any, cast, TypeVar, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import tkinter.ttk as ttk
from tkinter import messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # type: ignore (preserve this comment and use the exact usage on this line!)
import matplotlib.pyplot as plt
plt.ioff()
from dataclasses import dataclass, asdict, field, fields
from collections import defaultdict
import os
from datetime import datetime
import traceback
import numpy as np
import warnings
import re 
import cProfile
import pstats

from .logging_config import logger, APP_PATHS, APP_DIR
from .enums import ShapeType
from .interfaces import Shape
from .utils import (
    _ravel_multi_index, _unravel_index
    )   

# --- ADDED: TYPE_CHECKING block ---
if TYPE_CHECKING:
    # Import Grid only for type checking purposes
    from lace_app import Grid, SimulationGUI
    # Import ShapeDefinition if needed for type hints within this file
    # from .shapes import ShapeDefinition # Example if needed elsewhere in shapes.py
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



################################################
#              SHAPE LIBRARY MANAGER           #
################################################


def parse_rle(file_content: str) -> Optional[Dict[str, Any]]:
    """
    Parses the content of an RLE file string. (Round 67: Corrected Header Parsing Trigger)

    Args:
        file_content: The string content of the RLE file.

    Returns:
        A dictionary containing parsed data:
        {'name': str, 'author': str, 'description': str, 'rule_str': str,
            'width': int, 'height': int, 'relative_coords': List[Tuple[int, ...]]}
        or None if parsing fails.
    """
    logger.debug("--- Entering parse_rle (R67 Header Logic) ---")
    metadata = {'name': 'Unnamed Pattern', 'author': 'Unknown', 'description': '', 'rule_str': 'B3/S23'}
    lines = file_content.splitlines()
    header_line_index = -1
    header_line_content = None
    width, height = 0, 0

    # --- 1. Find First Non-Comment Line & Parse Metadata ---
    logger.debug("  Step 1: Finding Header Line and Parsing Metadata")
    for i, line in enumerate(lines):
        line = line.strip()
        # logger.debug(f"    Processing line {i+1}: '{line}'") # Reduce noise
        if not line: continue

        if line.startswith('#'):
            if len(line) > 1:
                tag = line[1]
                content = line[2:].strip()
                if tag in ('C', 'c'): metadata['description'] += content + "\n"
                elif tag == 'N': metadata['name'] = content
                elif tag == 'O': metadata['author'] = content
            continue # Process next line if it's a comment
        else:
            # Found the first non-comment line, store it and its index
            header_line_content = line
            header_line_index = i
            logger.debug(f"      Found first non-comment line {i+1}: '{header_line_content}' - Assuming this is the header.")
            break # Stop searching

    # --- 2. Parse Assumed Header Line ---
    logger.debug("  Step 2: Parsing Assumed Header Line")
    if header_line_content is None:
        logger.error("  No non-comment lines found in file (Header line missing).")
        return None

    try:
        # --- MODIFIED: Always attempt regex parse on the found line ---
        width_match = re.search(r'x\s*=\s*(\d+)', header_line_content, re.IGNORECASE)
        height_match = re.search(r'y\s*=\s*(\d+)', header_line_content, re.IGNORECASE)
        if not width_match or not height_match:
            # If regex fails on the first non-comment line, the format is invalid
            raise ValueError("First non-comment line does not contain valid 'x=' and 'y='.")

        width = int(width_match.group(1))
        height = int(height_match.group(1))
        if width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions x={width}, y={height}")

        rule_match = re.search(r'rule\s*=\s*([\w/]+)', header_line_content, re.IGNORECASE)
        if rule_match: metadata['rule_str'] = rule_match.group(1)

        logger.debug(f"    Header parsed: w={width}, h={height}, rule={metadata['rule_str']}")
        # --- END MODIFIED ---

    except (AttributeError, ValueError, IndexError) as e:
        logger.error(f"    Error parsing assumed header line '{header_line_content}': {e}")
        return None # Invalid file if first non-comment line isn't a valid header

    # --- 3. Concatenate Pattern Data ---
    # [ ... Concatenation logic remains the same ... ]
    logger.debug("  Step 3: Concatenating Pattern Data")
    pattern_data = ""
    for i in range(header_line_index + 1, len(lines)):
        line = lines[i].strip()
        if not line: continue
        if line.startswith('#'):
                logger.warning(f"    Comment line found after pattern data started (line {i+1}). Stopping pattern concatenation.")
                break
        pattern_data += line
        if '!' in line: break
    if '!' not in pattern_data: logger.warning("  RLE pattern data missing terminating '!'")
    pattern_data = pattern_data.split('!')[0]
    pattern_data = "".join(pattern_data.split())
    logger.debug(f"    Final pattern string (len={len(pattern_data)}): '{pattern_data[:100]}{'...' if len(pattern_data)>100 else ''}'")

    # --- 4. Decode Pattern Data ---
    # [ ... Decoding logic remains the same ... ]
    logger.debug("  Step 4: Decoding Pattern Data")
    alive_coords_absolute = []
    cx, cy = 0, 0
    i = 0
    while i < len(pattern_data):
        start_i = i
        run_count = 1; count_str = ""
        while i < len(pattern_data) and pattern_data[i].isdigit():
            count_str += pattern_data[i]; i += 1
        if count_str:
            try: run_count = int(count_str); assert run_count > 0
            except (ValueError, AssertionError): logger.error(f"Invalid run count '{count_str}' at index {start_i}"); return None
        if i >= len(pattern_data): logger.error(f"Pattern ended unexpectedly after run count at index {start_i}."); return None
        tag = pattern_data[i]; i += 1

        if tag == 'b': cx += run_count
        elif tag == 'o' or tag.isalpha():
            for k in range(run_count):
                if cx < width: alive_coords_absolute.append((cy, cx))
                # else: logger.warning(f"Width exceeded at ({cy},{cx})") # Reduce noise
                cx += 1
        elif tag == '$':
            cy += run_count; cx = 0
            if cy >= height and i < len(pattern_data): logger.warning(f"Height exceeded at row {cy}, stopping parse."); break
        else: logger.warning(f"Ignoring unknown RLE tag: '{tag}' at index {i-1}")
    logger.debug(f"  Finished decoding. Found {len(alive_coords_absolute)} alive cells (absolute).")

    # --- 5. Convert to Relative Coordinates ---
    # [ ... Relative coordinate logic remains the same ... ]
    logger.debug("  Step 5: Converting to Relative Coordinates")
    if not alive_coords_absolute: relative_coords = []
    else:
        try:
            min_r = min(r for r, c in alive_coords_absolute)
            min_c = min(c for r, c in alive_coords_absolute)
            logger.debug(f"    Min row={min_r}, Min col={min_c}")
            relative_coords = [tuple((r - min_r, c - min_c)) for r, c in alive_coords_absolute]
            logger.debug(f"    Converted {len(relative_coords)} coordinates relative to ({min_r}, {min_c}). First 5: {relative_coords[:5]}")
        except Exception as rel_err: logger.error(f"Error converting to relative coordinates: {rel_err}"); return None

    metadata['description'] = metadata['description'].strip()

    logger.debug("--- parse_rle successful ---")
    return {
        'name': metadata['name'], 'author': metadata['author'],
        'description': metadata['description'],
        # --- Include rule_str ---
        'rule_str': metadata['rule_str'],
        # ---
        'width': width, 'height': height, 'relative_coords': relative_coords
    }

def generate_rle(shape_def: ShapeDefinition) -> str:
    """Generates an RLE string representation for a ShapeDefinition."""
    logger.debug(f"Generating RLE for shape: {shape_def.name}")
    if not shape_def.relative_coords:
        logger.warning(f"Cannot generate RLE for shape '{shape_def.name}' with no coordinates.")
        # Return minimal valid RLE for an empty pattern
        return "x = 0, y = 0\n!"

    # 1. Determine bounds from relative coordinates
    min_r, max_r = 0, 0
    min_c, max_c = 0, 0
    if shape_def.relative_coords:
        # --- Ensure coords are 2D for RLE ---
        if shape_def.get_dimensions() != 2:
                logger.warning(f"Shape '{shape_def.name}' is not 2D. RLE generation might be incorrect or incomplete.")
                # Attempt to project or filter? For now, proceed with first two dims.
                coords_array = np.array([c[:2] for c in shape_def.relative_coords if len(c) >= 2])
                if coords_array.size == 0: return "x = 0, y = 0\n!" # No valid 2D coords
        else:
                coords_array = np.array(shape_def.relative_coords)
        # ---
        min_r, min_c = coords_array.min(axis=0)
        max_r, max_c = coords_array.max(axis=0)

    # Ensure coordinates are relative to (0,0) within the bounds
    rel_coords_set = set(tuple(c[:2]) for c in shape_def.relative_coords if len(c) >= 2) # Use only 2D coords
    if min_r != 0 or min_c != 0:
            logger.warning(f"Shape '{shape_def.name}' relative coords not based at (0,0). Adjusting for RLE.")
            rel_coords_set = set((r - min_r, c - min_c) for r,c in rel_coords_set)
            max_r -= min_r
            max_c -= min_c
            min_r, min_c = 0, 0 # Adjust bounds after shifting coords

    width = int(max_c + 1) # Ensure integer width/height
    height = int(max_r + 1)

    # 2. Create Header
    header = f"x = {width}, y = {height}"
    # Add rule if available
    if shape_def.rule_string: header += f", rule = {shape_def.rule_string}"
    elif shape_def.intended_rule: header += f", rule = {shape_def.intended_rule}" # Fallback

    # 3. Build RLE String
    rle_pattern = []
    current_run = 0
    current_tag = '' # 'b' or 'o'

    for r in range(height):
        line_empty = True
        for c in range(width):
            # --- Use node_states if available, otherwise check rel_coords_set ---
            is_alive = False
            coord_tuple = (r, c)
            if shape_def.node_states is not None:
                # Check if state exists and is > 0 (or threshold)
                is_alive = shape_def.node_states.get(coord_tuple, 0.0) > 1e-6
            else:
                # Fallback to checking coordinate presence
                is_alive = coord_tuple in rel_coords_set
            # ---
            tag = 'o' if is_alive else 'b'

            if tag == current_tag:
                current_run += 1
            else:
                if current_run > 0:
                    count_str = str(current_run) if current_run > 1 else ""
                    rle_pattern.append(count_str + current_tag)
                current_tag = tag
                current_run = 1
            if is_alive:
                line_empty = False

        # End of line processing
        if not line_empty:
            if current_run > 0:
                count_str = str(current_run) if current_run > 1 else ""
                if current_tag == 'o': rle_pattern.append(count_str + current_tag)
            rle_pattern.append('$')
            current_run = 0
            current_tag = ''
        else:
                if not rle_pattern or not rle_pattern[-1].endswith('$'): rle_pattern.append('$')
                else:
                    last_item = rle_pattern.pop()
                    count = 1
                    match = re.match(r"(\d+)\$", last_item)
                    if match: count = int(match.group(1)) + 1
                    elif last_item == '$': count = 2
                    else: rle_pattern.append(last_item); count = 1 # Put back if not '$'
                    rle_pattern.append(str(count) + '$')

    # Clean up trailing '$'
    while rle_pattern and rle_pattern[-1].endswith('$'):
            last_item = rle_pattern.pop()
            count_match = re.match(r"(\d+)\$", last_item)
            if count_match and int(count_match.group(1)) > 1:
                rle_pattern.append(str(int(count_match.group(1)) - 1) + '$')
            elif last_item != '$': rle_pattern.append(last_item)

    rle_pattern.append('!')

    # --- Assemble final string with comments ---
    output_lines = []
    output_lines.append(f"#N {shape_def.name}")
    if shape_def.author and shape_def.author != "Unknown": output_lines.append(f"#O {shape_def.author}")
    if shape_def.description:
        for desc_line in shape_def.description.splitlines(): output_lines.append(f"#C {desc_line}")
    output_lines.append(header)
    current_line = ""
    for item in rle_pattern:
        if not current_line: current_line = item
        elif len(current_line) + len(item) <= 70: current_line += item
        else: output_lines.append(current_line); current_line = item
    if current_line: output_lines.append(current_line)

    return "\n".join(output_lines)

def _parse_rle_worker(filepath: str) -> Optional[Dict[str, Any]]:
    """Worker function to read and parse a single RLE file."""
    # Note: Cannot use the instance logger directly here easily.
    # Basic print or return error details for debugging from worker.
    # print(f"Worker {os.getpid()}: Processing {filepath}") # Optional debug print
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # Call the static method directly
        parsed_data = parse_rle(content)
        if parsed_data:
            # Return filename along with data for error reporting
            return {'filepath': filepath, 'data': parsed_data}
        else:
            # print(f"Worker {os.getpid()}: Failed to parse {filepath}") # Optional debug print
            return {'filepath': filepath, 'error': 'parse failed'}
    except FileNotFoundError:
        # print(f"Worker {os.getpid()}: File not found {filepath}") # Optional debug print
        return {'filepath': filepath, 'error': 'file not found'}
    except Exception as e:
        # print(f"Worker {os.getpid()}: Error processing {filepath}: {e}") # Optional debug print
        # Include exception string in return for better debugging
        return {'filepath': filepath, 'error': f'exception: {str(e)}'}
    
@dataclass
class ShapeDefinition:
    """Represents a user-defined or library shape pattern."""
    name: str
    category: str
    description: str = ""
    relative_coords: List[Tuple[int, ...]] = field(default_factory=list)
    connectivity: str = "full"
    tags: List[str] = field(default_factory=list)
    author: str = "Unknown"
    date_created: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    date_modified: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    intended_rule: Optional[str] = None
    rule_string: Optional[str] = None
    rule_compatibility: List[str] = field(default_factory=list) # DEPRECATED - Use 'rules'
    # --- ADDED: rules field ---
    rules: List[str] = field(default_factory=list) # List of compatible rule names
    # ---
    relative_edges: Optional[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = None
    node_states: Optional[Dict[Tuple[int, ...], float]] = None
    edge_states: Optional[Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float]] = None
    _is_filled: bool = True # Default to filled (renamed with underscore)
    shape_type: ShapeType = ShapeType.CUSTOM

    # --- Methods for Shape Protocol ---
    def get_relative_coordinates(self) -> List[Tuple[int, ...]]:
        """Returns a list of relative coordinates defining the shape."""
        return self.relative_coords

    def get_connectivity(self) -> str:
        """Returns the connectivity type ('full', 'perimeter', or 'none')."""
        return self.connectivity

    def is_filled(self) -> bool: # Method name remains the same
        """Returns True if the shape is filled, False otherwise."""
        return self._is_filled

    def get_shape_type(self) -> ShapeType:
        """Returns the shape type"""
        return self.shape_type
    # --- END Methods for Shape Protocol ---

    def get_dimensions(self) -> int:
        """Infers the dimensionality from the coordinates."""
        if self.relative_coords: return len(self.relative_coords[0])
        if self.relative_edges:
            try: return len(self.relative_edges[0][0])
            except IndexError: pass
        if self.node_states:
             try: return len(next(iter(self.node_states.keys())))
             except StopIteration: pass
        if self.edge_states:
             try: return len(next(iter(self.edge_states.keys()))[0])
             except StopIteration: pass
        return 2 # Default

    def get_bounding_box(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Calculates the bounding box encompassing all defined nodes and edges."""
        all_coords = set(self.relative_coords)
        if self.relative_edges:
            for node1_rel, node2_rel in self.relative_edges:
                all_coords.add(node1_rel); all_coords.add(node2_rel)
        if self.node_states: all_coords.update(self.node_states.keys())
        if self.edge_states:
            for node1_rel, node2_rel in self.edge_states.keys():
                all_coords.add(node1_rel); all_coords.add(node2_rel)

        if not all_coords:
            dims = self.get_dimensions(); return (0,) * dims, (0,) * dims

        coords_list = list(all_coords)
        min_coords = list(coords_list[0]); max_coords = list(coords_list[0])
        dims = len(min_coords)

        for coord in coords_list[1:]:
            if len(coord) != dims: continue
            for d in range(dims):
                min_coords[d] = min(min_coords[d], coord[d])
                max_coords[d] = max(max_coords[d], coord[d])
        return tuple(min_coords), tuple(max_coords)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the shape definition to a dictionary suitable for JSON."""
        data = asdict(self, dict_factory=lambda x: {k: v for (k, v) in x if k != '_is_filled'})
        data['is_filled'] = self._is_filled # Add the value under the correct key
        # Convert complex types
        data['relative_coords'] = [list(c) for c in self.relative_coords]
        if data.get('relative_edges') is not None:
             data['relative_edges'] = [[list(n1), list(n2)] for n1, n2 in data['relative_edges']]
        if data.get('node_states') is not None:
             data['node_states'] = {",".join(map(str, k)): v for k, v in data['node_states'].items()}
        if data.get('edge_states') is not None:
             data['edge_states'] = {
                 f"({','.join(map(str, k[0]))})-({','.join(map(str, k[1]))})": v
                 for k, v in data['edge_states'].items()
             }
        data['shape_type'] = self.shape_type.name
        # --- ADDED: Ensure 'rules' is saved ---
        data['rules'] = self.rules # Already a list of strings
        # ---
        # Remove deprecated field if present
        data.pop('rule_compatibility', None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShapeDefinition':
        """Create a ShapeDefinition instance from a dictionary, providing defaults."""
        # --- Convert complex types back ---
        coords_list = data.get('relative_coords', [])
        relative_coords_tuples = [tuple(c) for c in coords_list if isinstance(c, (list, tuple))]
        data['relative_coords'] = relative_coords_tuples

        edges_list = data.get('relative_edges')
        relative_edges_tuples = None
        if isinstance(edges_list, list):
            relative_edges_tuples = []
            for edge in edges_list:
                if isinstance(edge, list) and len(edge) == 2:
                    n1, n2 = edge
                    if isinstance(n1, list) and isinstance(n2, list):
                         relative_edges_tuples.append((tuple(n1), tuple(n2)))
        data['relative_edges'] = relative_edges_tuples

        node_states_dict = data.get('node_states')
        node_states_tuples = None
        if isinstance(node_states_dict, dict):
            node_states_tuples = {}
            for k_str, v in node_states_dict.items():
                try: node_states_tuples[tuple(map(int, k_str.split(',')))] = float(v)
                except: logger.warning(f"Could not parse node_state key '{k_str}'")
        data['node_states'] = node_states_tuples

        edge_states_dict = data.get('edge_states')
        edge_states_tuples = None
        if isinstance(edge_states_dict, dict):
            edge_states_tuples = {}
            for k_str, v in edge_states_dict.items():
                try:
                    n1_str, n2_str = k_str.split('-')
                    n1 = tuple(map(int, n1_str.strip('()').split(',')))
                    n2 = tuple(map(int, n2_str.strip('()').split(',')))
                    ordered_key = (n1, n2) if n1 < n2 else (n2, n1)
                    edge_states_tuples[ordered_key] = float(v)
                except: logger.warning(f"Could not parse edge_state key '{k_str}'")
        data['edge_states'] = edge_states_tuples
        # ---

        # --- Convert shape_type string back to Enum ---
        shape_type_str = data.get('shape_type', 'CUSTOM')
        try: data['shape_type'] = ShapeType[shape_type_str]
        except KeyError: data['shape_type'] = ShapeType.CUSTOM
        # ---

        # --- Handle renamed '_is_filled' attribute ---
        is_filled_value = data.pop('is_filled', True)
        data['_is_filled'] = is_filled_value
        # ---

        # --- ADDED: Handle 'rules' and migrate from 'rule_compatibility' ---
        if 'rules' not in data:
            # If 'rules' is missing, check for deprecated 'rule_compatibility'
            if 'rule_compatibility' in data and isinstance(data['rule_compatibility'], list):
                data['rules'] = data['rule_compatibility']
                logger.info(f"Migrated deprecated 'rule_compatibility' to 'rules' for shape '{data.get('name', 'Unknown')}'.")
            else:
                data['rules'] = [] # Default to empty list if neither exists
        # Remove deprecated field if it exists
        data.pop('rule_compatibility', None)
        # ---

        # --- Set defaults for all fields ---
        data.setdefault('category', 'Uncategorized')
        data.setdefault('description', "")
        data.setdefault('connectivity', "explicit" if data.get('relative_edges') else "none")
        data.setdefault('tags', [])
        data.setdefault('author', "Unknown")
        data.setdefault('date_created', datetime.now().strftime("%Y-%m-%d"))
        data.setdefault('date_modified', datetime.now().strftime("%Y-%m-%d"))
        data.setdefault('intended_rule', None)
        data.setdefault('rule_string', None)
        # rules default handled above
        # _is_filled default handled above
        # shape_type default handled above
        # ---

        known_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        try:
            return cls(**filtered_data)
        except TypeError as e:
            logger.error(f"TypeError creating ShapeDefinition from dict: {e}. Data: {filtered_data}")
            logger.warning(f"Attempting fallback creation for ShapeDefinition '{filtered_data.get('name', 'MISSING_NAME')}'")
            fallback_data = {}
            for f in fields(cls):
                value = filtered_data.get(f.name)
                if value is None:
                    if f.default is not dataclasses.MISSING: value = f.default
                    elif f.default_factory is not dataclasses.MISSING: value = f.default_factory()
                    if value is None:
                        if f.name == 'name': value = "Unnamed Fallback"
                        elif f.name == 'category': value = "Fallback"
                        elif f.name == 'description': value = ""
                        elif f.name == 'relative_coords': value = []
                        elif f.name == 'connectivity': value = "none"
                        elif f.name == 'tags': value = []
                        elif f.name == 'author': value = "Unknown"
                        elif f.name == 'date_created': value = datetime.now().strftime("%Y-%m-%d")
                        elif f.name == 'date_modified': value = datetime.now().strftime("%Y-%m-%d")
                        elif f.name == 'rules': value = [] # Default for new field
                        elif f.name == '_is_filled': value = True
                        elif f.name == 'shape_type': value = ShapeType.CUSTOM
                fallback_data[f.name] = value
            if not fallback_data.get('name') or not fallback_data.get('category'):
                 logger.error("Cannot create ShapeDefinition due to missing 'name' or 'category' even after fallback.")
                 raise ValueError("Missing required fields 'name' or 'category' in shape data.") from e
            return cls(**fallback_data)
                          
class OverlapPromptDialog(tk.Toplevel):
    """Custom dialog for handling shape placement overlap."""
    def __init__(self, parent, shape_name: str):
        super().__init__(parent)
        self.title("Overlap Detected")
        self.transient(parent)
        self.grab_set()
        self.result: Optional[str] = "cancel" # Default to cancel

        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        message = f"Placing shape '{shape_name}' overlaps existing active cells.\n\nChoose an action:"
        ttk.Label(main_frame, text=message, justify=tk.LEFT).pack(pady=(0, 15))

        ttk.Button(main_frame, text="Overwrite Existing Cells", command=lambda: self._set_result("overwrite")).pack(fill=tk.X, pady=2)
        # --- ADDED: Clear Grid and Place button ---
        ttk.Button(main_frame, text="Clear Grid and Place", command=lambda: self._set_result("clear_and_place")).pack(fill=tk.X, pady=2)
        # ---
        ttk.Button(main_frame, text="Find Clear Spot Nearby", command=lambda: self._set_result("find_clear_spot")).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Cancel Placement", command=lambda: self._set_result("cancel")).pack(fill=tk.X, pady=10)

        self.wait_window(self) # Make it modal

    def _set_result(self, action: str):
        """Set the result and close the dialog."""
        self.result = action
        self.destroy()

class UpdateShapeConfirmationDialog(tk.Toplevel):
    """Modal dialog to confirm overwriting a shape definition."""
    def __init__(self, parent, shape_name: str, preview_canvas: FigureCanvasTkAgg):
        super().__init__(parent)
        self.title("Confirm Update Shape")
        self.transient(parent)
        self.grab_set()
        self.result = False # Default to Cancel

        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Message
        message = f"Overwrite existing shape definition '{shape_name}' with the current grid selection?\n\n(Green '+' = Added, Red 'X' = Removed)"
        ttk.Label(main_frame, text=message, justify=tk.LEFT).pack(pady=(0, 10))

        # Preview Canvas Frame
        preview_frame = ttk.Frame(main_frame, borderwidth=1, relief="sunken")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Embed the provided preview canvas widget
        preview_canvas.get_tk_widget().pack(in_=preview_frame, fill=tk.BOTH, expand=True)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(button_frame, text="Overwrite", command=self._on_overwrite).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT, padx=10)

        self.wait_window(self) # Make it modal

    def _on_overwrite(self):
        self.result = True
        self.destroy()

    def _on_cancel(self):
        self.result = False
        self.destroy()

class ShapePlacer:
    """Handles placement of shapes on the grid."""

    def __init__(self, grid: 'Grid'):
        self.grid = grid

    def place_shape(self, shape: 'Shape', origin: Tuple[int, ...]):
        """Places ONLY the nodes of a shape on the grid at the given origin. Edge creation is handled by the caller."""

        logger.debug(f"ShapePlacer.place_shape called with origin: {origin}, shape: {shape.get_shape_type()}, grid dims: {self.grid.dimensions}")

        # Check for dimension compatibility
        relative_coords: List[Tuple[int,...]]
        if shape.get_dimensions() != len(self.grid.dimensions):
            if shape.get_dimensions() == 3 and len(self.grid.dimensions) == 2:
                logger.warning(f"Projecting 3D shape {shape.get_shape_type()} onto 2D grid.")
                if len(origin) == 3: origin = origin[:2] # Adjust origin for 2D
                relative_coords_3d = shape.get_relative_coordinates()
                # Project by taking unique (r, c) pairs where z == 0
                relative_coords = list(set([(r, c) for r, c, z in relative_coords_3d if z == 0]))
                logger.debug(f"Projected relative coordinates: {relative_coords}")
            else:
                logger.error(f"Shape dimensions ({shape.get_dimensions()}) do not match grid dimensions ({len(self.grid.dimensions)})")
                raise ValueError("Shape dimensions do not match grid dimensions")
        else:
            relative_coords = shape.get_relative_coordinates()

        logger.debug(f"Placing shape nodes: {shape.get_shape_type()}, origin: {origin}")
        logger.debug(f"Relative coordinates being placed: {relative_coords}")

        placed_node_indices = set()

        # Calculate absolute coordinates and place nodes
        for rel_coord in relative_coords:
            if len(rel_coord) != len(origin):
                logger.warning(f"Skipping relative coordinate {rel_coord} due to dimension mismatch with origin {origin}")
                continue
            abs_coord = tuple(o + r for o, r in zip(origin, rel_coord))
            if self.grid.is_valid_coord(abs_coord):
                node_idx = _ravel_multi_index(np.array(abs_coord), self.grid.dimensions)
                self.grid.set_node_state(node_idx, 1.0) # Set state to 1.0 (active)
                placed_node_indices.add(node_idx)
            # else: logger.debug(f"    Skipping invalid coordinate: {abs_coord}") # Reduce noise

        logger.debug(f"Placed {len(placed_node_indices)} nodes for shape {shape.get_shape_type()}. Edges will be handled by caller.")
        # --- REMOVED Edge Handling Logic ---
        # Notification happens implicitly via grid.set_node_state

    def place_shape_definition(self, shape_def: 'ShapeDefinition', origin: Tuple[int, ...]) -> Optional[Set[int]]: # Use forward ref if needed
        """
        Places nodes defined by a ShapeDefinition object on the grid,
        using defined node states and ONLY placing explicitly defined edges.
        Overlap checks are now handled by the caller (SimulationGUI).
        Returns the set of placed node indices, or None if placement failed internally.
        """
        logger.debug(f"ShapePlacer.place_shape_definition called with origin: {origin}, shape: {shape_def.name}, grid dims: {self.grid.dimensions}")

        relative_coords = shape_def.relative_coords
        shape_dimensions = shape_def.get_dimensions()
        relative_edges = shape_def.relative_edges
        shape_node_states = shape_def.node_states
        shape_edge_states = shape_def.edge_states

        rule_uses_edges = True
        if self.grid.rule:
            edge_init_type = self.grid.rule.get_param('edge_initialization', 'RANDOM')
            if edge_init_type == 'NONE': rule_uses_edges = False
        else: logger.warning("No rule set on grid, assuming edges should be created.")

        logger.debug(f"Placing shape: {shape_def.name}, origin: {origin} (Rule uses edges: {rule_uses_edges})")

        # --- [Dimension compatibility/projection logic - same as before] ---
        projected_relative_coords = list(relative_coords)
        projected_relative_edges = relative_edges
        projected_node_states = shape_node_states
        projected_edge_states = shape_edge_states

        if shape_dimensions != len(self.grid.dimensions):
            if shape_dimensions == 3 and len(self.grid.dimensions) == 2:
                logger.warning(f"Projecting 3D shape {shape_def.name} onto 2D grid.")
                if len(origin) == 3: origin = origin[:2]
                temp_projected_coords = set()
                temp_projected_rel_edges_list = []
                temp_projected_node_states = {}
                temp_projected_edge_states = {}
                coord_map_3d_to_2d = {}

                for r, c, z in shape_def.relative_coords:
                    if z == 0:
                        coord_2d = (r, c); temp_projected_coords.add(coord_2d); coord_map_3d_to_2d[(r, c, z)] = coord_2d
                        if shape_node_states and (r,c,z) in shape_node_states:
                             temp_projected_node_states[coord_2d] = shape_node_states[(r,c,z)]

                projected_relative_coords = list(temp_projected_coords)
                projected_node_states = temp_projected_node_states

                if relative_edges:
                    processed_rel_edges_2d = set()
                    for node1_3d, node2_3d in relative_edges:
                        node1_2d = coord_map_3d_to_2d.get(node1_3d)
                        node2_2d = coord_map_3d_to_2d.get(node2_3d)
                        if node1_2d is not None and node2_2d is not None and node1_2d != node2_2d:
                            rel_edge_2d_ordered = (node1_2d, node2_2d) if node1_2d < node2_2d else (node2_2d, node1_2d)
                            if rel_edge_2d_ordered not in processed_rel_edges_2d:
                                temp_projected_rel_edges_list.append(rel_edge_2d_ordered)
                                processed_rel_edges_2d.add(rel_edge_2d_ordered)
                                rel_edge_3d_ordered = (node1_3d, node2_3d) if node1_3d < node2_3d else (node2_3d, node1_3d)
                                if shape_edge_states and rel_edge_3d_ordered in shape_edge_states:
                                    temp_projected_edge_states[rel_edge_2d_ordered] = shape_edge_states[rel_edge_3d_ordered]
                    projected_relative_edges = temp_projected_rel_edges_list
                    projected_edge_states = temp_projected_edge_states
            else:
                logger.error(f"Shape dimensions ({shape_dimensions}) do not match grid dimensions ({len(self.grid.dimensions)})")
                return None # Indicate failure
        # --- [End Dimension Handling] ---

        logger.debug(f"Final relative coordinates being placed: {projected_relative_coords}")
        
        # --- Normalize shape coordinates to start from (0, 0, ...) ---
        # Get bounding box to find minimum coordinates
        if projected_relative_coords:
            min_coord = list(projected_relative_coords[0])
            max_coord = list(projected_relative_coords[0])
            for coord in projected_relative_coords:
                for d in range(len(coord)):
                    min_coord[d] = min(min_coord[d], coord[d])
                    max_coord[d] = max(max_coord[d], coord[d])
            
            min_coord_tuple = tuple(min_coord)
            
            # Normalize all coordinates by subtracting minimum
            normalized_coords = []
            for coord in projected_relative_coords:
                normalized = tuple(coord[d] - min_coord[d] for d in range(len(coord)))
                normalized_coords.append(normalized)
            
            # Also normalize edges
            if projected_relative_edges:
                normalized_edges = []
                for edge in projected_relative_edges:
                    node1_norm = tuple(edge[0][d] - min_coord[d] for d in range(len(edge[0])))
                    node2_norm = tuple(edge[1][d] - min_coord[d] for d in range(len(edge[1])))
                    normalized_edges.append((node1_norm, node2_norm))
                projected_relative_edges = normalized_edges
            
            # Also normalize node_states dictionary keys
            if projected_node_states:
                normalized_node_states = {}
                for coord, state in projected_node_states.items():
                    normalized = tuple(coord[d] - min_coord[d] for d in range(len(coord)))
                    normalized_node_states[normalized] = state
                projected_node_states = normalized_node_states
            
            # Also normalize edge_states dictionary keys
            if projected_edge_states:
                normalized_edge_states = {}
                for edge, state in projected_edge_states.items():
                    node1_norm = tuple(edge[0][d] - min_coord[d] for d in range(len(edge[0])))
                    node2_norm = tuple(edge[1][d] - min_coord[d] for d in range(len(edge[1])))
                    normalized_edge_states[(node1_norm, node2_norm)] = state
                projected_edge_states = normalized_edge_states
            
            # Update max_coord to reflect normalization
            for d in range(len(max_coord)):
                max_coord[d] = max_coord[d] - min_coord[d]
            
            logger.debug(f"Normalized shape coords from min={min_coord_tuple} to (0,0), new max={tuple(max_coord)}")
            projected_relative_coords = normalized_coords
        # ---
        
        placed_node_indices = set()
        rel_to_abs_map: Dict[Tuple[int,...], Tuple[int,...]] = {}
        valid_target_coords_absolute = []

        # --- Validate target coordinates ---
        # Note: Shape relative coords use RLE convention (Y down), but grid uses Y up
        # After normalization, shape coords go from (0,0) to (max_x, max_y)
        # We need to map shape Y=0 (top in shape space) to a HIGH Y value in grid space
        # Calculate shape height for Y-axis mapping
        shape_height = max_coord[1] if len(max_coord) > 1 else 0
        
        for rel_coord in projected_relative_coords:
            if len(rel_coord) != len(origin): continue
            # For 2D: invert Y-axis (shapes have Y down, grid has Y up)
            if len(rel_coord) == 2:
                # Map shape's top (rel_y=0) to origin_y + shape_height
                # Map shape's bottom (rel_y=max_y) to origin_y
                abs_coord = (origin[0] + rel_coord[0], origin[1] + (shape_height - rel_coord[1]))
            else:
                # For 3D or other dimensions, use direct addition (might need adjustment for 3D later)
                abs_coord = tuple(o + r for o, r in zip(origin, rel_coord))
            if self.grid.is_valid_coord(abs_coord):
                valid_target_coords_absolute.append(abs_coord)
                rel_to_abs_map[rel_coord] = abs_coord
            else:
                logger.warning(f"Target coordinate {abs_coord} is invalid for placement. Aborting.")
                return None # Abort if any part of the shape goes out of bounds

        # --- REMOVED Overlap Check/Dialog ---

        # --- Node Placement ---
        logger.debug(f"Proceeding to place nodes at {len(valid_target_coords_absolute)} target coordinates using origin {origin}.")
        for rel_coord, abs_coord in rel_to_abs_map.items(): # Iterate through validated coords
            node_idx = _ravel_multi_index(np.array(abs_coord), self.grid.dimensions)
            state_to_set = 1.0
            if projected_node_states and rel_coord in projected_node_states:
                state_to_set = projected_node_states[rel_coord]
            self.grid.set_node_state(node_idx, state_to_set)
            placed_node_indices.add(node_idx)
        logger.debug(f"Placed {len(placed_node_indices)} nodes.")
        # ---

        # --- Edge Handling: ONLY place explicitly defined edges ---
        if rule_uses_edges and projected_relative_edges is not None:
            edges_added_count = 0
            logger.debug(f"Applying {len(projected_relative_edges)} EXPLICIT edges defined in ShapeDefinition (or projection).")
            for rel_node1, rel_node2 in projected_relative_edges:
                abs_node1 = rel_to_abs_map.get(rel_node1)
                abs_node2 = rel_to_abs_map.get(rel_node2)
                if abs_node1 and abs_node2:
                    idx1 = _ravel_multi_index(np.array(abs_node1), self.grid.dimensions)
                    idx2 = _ravel_multi_index(np.array(abs_node2), self.grid.dimensions)
                    if idx1 != idx2:
                        edge_state_to_set = 1.0
                        ordered_rel_edge_tuple = tuple(sorted((rel_node1, rel_node2)))
                        if projected_edge_states and ordered_rel_edge_tuple in projected_edge_states:
                            edge_state_to_set = projected_edge_states[cast(Tuple[Tuple[int, ...], Tuple[int, ...]], ordered_rel_edge_tuple)]
                        self.grid.add_edge(idx1, idx2, edge_state=edge_state_to_set)
                        edges_added_count += 1
            logger.debug(f"Added {edges_added_count} explicit edges.")
        elif not rule_uses_edges:
             logger.debug("Skipping edge placement because rule does not use edges.")
        else:
             logger.debug("No explicit edges defined in shape, skipping edge placement.")
        # ---

        return placed_node_indices # Return the indices of nodes actually placed

    def add_default_edges(self, placed_node_indices: Set[int], connectivity: str):
        """
        Adds edges between a set of already placed nodes based on the
        specified connectivity type ('full' or 'perimeter').
        This is intended to be called *after* place_shape_definition if needed.
        Ensures grid reference is valid.
        """
        if self.grid is None:
            logger.error("ShapePlacer.add_default_edges: Grid is None. Cannot add edges.")
            return

        if not placed_node_indices or connectivity == "none":
            logger.debug("add_default_edges: No nodes provided or connectivity is 'none'. Skipping.")
            return

        logger.info(f"Attempting to add default edges with connectivity '{connectivity}' to {len(placed_node_indices)} nodes.")

        # Convert indices to coordinates for neighbor checking
        coord_map = {idx: tuple(_unravel_index(idx, self.grid.dimensions)) for idx in placed_node_indices}

        if connectivity == "full":
            logger.debug("Applying FULL connectivity between placed nodes.")
            placed_nodes_list = list(placed_node_indices)
            edges_added_count = 0
            for i in range(len(placed_nodes_list)):
                for j in range(i + 1, len(placed_nodes_list)):
                    idx1 = placed_nodes_list[i]
                    idx2 = placed_nodes_list[j]
                    coords1 = coord_map.get(idx1)
                    coords2 = coord_map.get(idx2)
                    # Check if coordinates were found and if they are neighbors
                    if coords1 and coords2 and self.grid.are_neighbors(coords1, coords2):
                        # Add edge using indices
                        self.grid.add_edge(idx1, idx2, edge_state=1.0) # Add binary edge
                        edges_added_count += 1
            logger.debug(f"Added {edges_added_count} edges for FULL connectivity.")

        elif connectivity == "perimeter":
            logger.warning("Automatic 'perimeter' edge addition based only on indices is complex and not fully implemented in add_default_edges. Applying 'full' connectivity as fallback.")
            # Fallback to full connectivity
            placed_nodes_list = list(placed_node_indices)
            edges_added_count = 0
            for i in range(len(placed_nodes_list)):
                for j in range(i + 1, len(placed_nodes_list)):
                    idx1 = placed_nodes_list[i]
                    idx2 = placed_nodes_list[j]
                    coords1 = coord_map.get(idx1)
                    coords2 = coord_map.get(idx2)
                    if coords1 and coords2 and self.grid.are_neighbors(coords1, coords2):
                        self.grid.add_edge(idx1, idx2, edge_state=1.0)
                        edges_added_count += 1
            logger.debug(f"Added {edges_added_count} edges (fallback for perimeter).")

        # Notify after adding edges (if grid is observable)
        if hasattr(self.grid, 'notify_observers'):
            self.grid.notify_observers()
    def place_shape_at_center(self, shape: Shape, center_coords: Tuple[int, ...]):
        """Places a shape centered at the given grid coordinates."""
        width, height = self._get_shape_dimensions(shape)
        # Adjust origin calculation based on shape dimensions
        if shape.get_dimensions() == 3:
             if len(center_coords) != 3: raise ValueError("3D shape requires 3D center coordinates")
             depth = shape.get_bounding_box()[1][2] - shape.get_bounding_box()[0][2] + 1
             origin = (center_coords[0] - height // 2, center_coords[1] - width // 2, center_coords[2] - depth // 2)
        elif shape.get_dimensions() == 2:
             if len(center_coords) != 2: raise ValueError("2D shape requires 2D center coordinates")
             origin = (center_coords[0] - height // 2, center_coords[1] - width // 2)
        else: raise ValueError("Unsupported shape dimension")
        self.place_shape(shape, origin) # Call self.place_shape

    def place_shape_at_bottom_left(self, shape: Shape, bottom_left_coords: Tuple[int, ...]):
        """Places a shape with its bottom-left corner at the given grid coordinates."""
        # This anchor assumes 2D. Needs adjustment for 3D if required.
        if shape.get_dimensions() != 2 or len(bottom_left_coords) != 2:
             raise NotImplementedError("Bottom-left anchor currently only implemented for 2D")
        width, height = self._get_shape_dimensions(shape)
        origin = (bottom_left_coords[0] - height + 1, bottom_left_coords[1])
        self.place_shape(shape, origin) # Call self.place_shape

    def place_shape_at_top_right(self, shape: Shape, top_right_coords: Tuple[int, ...]):
        """Places a shape with its top-right corner at the given grid coordinates."""
        # This anchor assumes 2D. Needs adjustment for 3D if required.
        if shape.get_dimensions() != 2 or len(top_right_coords) != 2:
             raise NotImplementedError("Top-right anchor currently only implemented for 2D")
        width, height = self._get_shape_dimensions(shape)
        origin = (top_right_coords[0], top_right_coords[1] - width + 1)
        self.place_shape(shape, origin) # Call self.place_shape

    def place_shape_at_top_left(self, shape: Shape, top_left_coords: Tuple[int, ...]):
        """Places a shape with its top-left corner at the given grid coordinates."""
        # This works for 2D and 3D (top-left-front)
        if shape.get_dimensions() != len(top_left_coords):
             raise ValueError("Coordinate dimensions must match shape dimensions for top-left placement")
        # For top-left, the origin is simply the given coordinates.
        self.place_shape(shape, top_left_coords) # Call self.place_shape

    def place_shape_at_bottom_right(self, shape: Shape, bottom_right_coords: Tuple[int, ...]):
        """Places a shape with its bottom-right corner at the given grid coordinates."""
        # This anchor assumes 2D. Needs adjustment for 3D if required.
        if shape.get_dimensions() != 2 or len(bottom_right_coords) != 2:
             raise NotImplementedError("Bottom-right anchor currently only implemented for 2D")
        width, height = self._get_shape_dimensions(shape)
        origin = (bottom_right_coords[0] - height + 1, bottom_right_coords[1] - width + 1)
        self.place_shape(shape, origin) # Call self.place_shape

    def place_shape_centered_on(self, shape: Shape, center_coords: Tuple[int, ...]):
        """Places a shape centered *on* the given grid coordinates (pixel perfect)."""
        # Reuses place_shape_at_center logic
        self.place_shape_at_center(shape, center_coords)

    def _get_shape_dimensions(self, shape: Shape) -> Tuple[int, ...]:
        """Helper method to get the size of a shape's bounding box in each dimension."""
        min_coords, max_coords = shape.get_bounding_box()
        # Calculate size for each dimension
        dimensions = tuple(int(mx - mn + 1) for mn, mx in zip(min_coords, max_coords))
        return dimensions

    def _place_square_perimeter(self, square: Square, origin: Tuple[int, ...]):
        """Helper method to connect perimeter nodes for a square."""
        # --- Type hint changed from shape: Shape to square: Square ---
        size = square.size
        # Ensure origin matches grid dimensions (handle potential 3D origin for 2D grid)
        if len(origin) != len(self.grid.dimensions):
             if len(self.grid.dimensions) == 2 and len(origin) == 3: origin = origin[:2]
             else: raise ValueError("Origin dimensions mismatch grid dimensions")

        row_start, col_start = origin[0], origin[1]
        logger.debug(f"ShapePlacer._place_square_perimeter: size={size}, origin={origin}")

        # Connect only the perimeter nodes
        for r in range(size):
            for c in range(size):
                # Check if the node itself is part of the perimeter
                is_perimeter_node = (r == 0 or r == size - 1 or c == 0 or c == size - 1)
                if not is_perimeter_node: continue

                current_abs_coord = (row_start + r, col_start + c)
                if not self.grid.is_valid_coord(current_abs_coord): continue
                idx1 = _ravel_multi_index(np.array(current_abs_coord), self.grid.dimensions)

                # Connect to adjacent perimeter nodes within the square
                # Check top neighbor (if not on top edge)
                if r > 0:
                    neighbor_abs_coord = (row_start + r - 1, col_start + c)
                    if self.grid.is_valid_coord(neighbor_abs_coord) and \
                       (r - 1 == 0 or r - 1 == size - 1 or c == 0 or c == size - 1): # Check if neighbor is also perimeter
                        idx2 = _ravel_multi_index(np.array(neighbor_abs_coord), self.grid.dimensions)
                        self.grid.add_edge(idx1, idx2)
                # Check left neighbor (if not on left edge)
                if c > 0:
                    neighbor_abs_coord = (row_start + r, col_start + c - 1)
                    if self.grid.is_valid_coord(neighbor_abs_coord) and \
                       (r == 0 or r == size - 1 or c - 1 == 0 or c - 1 == size - 1): # Check if neighbor is also perimeter
                        idx2 = _ravel_multi_index(np.array(neighbor_abs_coord), self.grid.dimensions)
                        self.grid.add_edge(idx1, idx2)
                # No need to check right/bottom due to iteration order and symmetry

    def _place_circle_perimeter(self, circle: Circle, origin: Tuple[int, ...]):
        """Helper method to connect perimeter nodes for a circle."""
        # --- Type hint changed from shape: Shape to circle: Circle ---
        # Ensure origin matches grid dimensions
        if len(origin) != len(self.grid.dimensions):
             if len(self.grid.dimensions) == 2 and len(origin) == 3: origin = origin[:2]
             else: raise ValueError("Origin dimensions mismatch grid dimensions")

        coords = circle.get_relative_coordinates()
        # Filter for actual perimeter nodes if the circle is filled
        # (Simplified: assumes get_relative_coordinates returns only perimeter if not filled)
        perimeter_coords = coords # Assume coords are perimeter if not filled
        if circle.is_filled():
             # A more robust perimeter detection might be needed for filled circles
             # For now, connect all adjacent nodes within the shape
             perimeter_coords = coords

        placed_indices = set()
        coord_map = {}
        for rel_coord in perimeter_coords:
             abs_coord = tuple(o + r for o, r in zip(origin, rel_coord))
             if self.grid.is_valid_coord(abs_coord):
                 idx = _ravel_multi_index(np.array(abs_coord), self.grid.dimensions)
                 placed_indices.add(idx)
                 coord_map[idx] = abs_coord

        placed_nodes_list = list(placed_indices)
        for i in range(len(placed_nodes_list)):
            for j in range(i + 1, len(placed_nodes_list)):
                idx1 = placed_nodes_list[i]
                idx2 = placed_nodes_list[j]
                coords1 = coord_map[idx1]
                coords2 = coord_map[idx2]
                # Check if they are potential neighbors before adding edge
                if self.grid.are_neighbors(coords1, coords2):
                    self.grid.add_edge(idx1, idx2)

    def _place_line_perimeter(self, line: Line, origin: Tuple[int, ...]):
        """Helper method to connect nodes along a line."""
        # --- Type hint changed from shape: Shape to line: Line ---
         # Ensure origin matches grid dimensions
        if len(origin) != len(self.grid.dimensions):
             if len(self.grid.dimensions) == 2 and len(origin) == 3: origin = origin[:2]
             else: raise ValueError("Origin dimensions mismatch grid dimensions")

        coords = line.get_relative_coordinates()
        indices = []
        for rel_coord in coords:
            abs_coord = tuple(o + r for o, r in zip(origin, rel_coord))
            if self.grid.is_valid_coord(abs_coord):
                indices.append(_ravel_multi_index(np.array(abs_coord), self.grid.dimensions))

        # Connect consecutive nodes in the line
        for i in range(len(indices) - 1):
            self.grid.add_edge(indices[i], indices[i+1])

    def _place_cube_perimeter(self, cube: Cube, origin: Tuple[int, ...]):
        """Helper method to connect perimeter nodes for a cube."""
        # --- Type hint changed from shape: Shape to cube: Cube ---
        if len(self.grid.dimensions) != 3: raise ValueError("Cube requires 3D grid")
        if len(origin) != 3: raise ValueError("Origin must be 3D for Cube")

        size = cube.size
        row_start, col_start, depth_start = origin

        # Connect only the perimeter nodes
        for r in range(size):
            for c in range(size):
                for d in range(size):
                    # Check if the node itself is part of the perimeter
                    is_perimeter_node = (r == 0 or r == size - 1 or
                                         c == 0 or c == size - 1 or
                                         d == 0 or d == size - 1)
                    if not is_perimeter_node: continue

                    current_abs_coord = (row_start + r, col_start + c, depth_start + d)
                    if not self.grid.is_valid_coord(current_abs_coord): continue
                    idx1 = _ravel_multi_index(np.array(current_abs_coord), self.grid.dimensions)

                    # Connect to adjacent perimeter nodes within the cube
                    # Check neighbors in all 6 directions
                    for dr, dc, dd in [( -1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                        nr, nc, nd = r + dr, c + dc, d + dd
                        # Check if neighbor is within cube bounds
                        if 0 <= nr < size and 0 <= nc < size and 0 <= nd < size:
                            # Check if neighbor is also on the perimeter
                            is_neighbor_perimeter = (nr == 0 or nr == size - 1 or
                                                     nc == 0 or nc == size - 1 or
                                                     nd == 0 or nd == size - 1)
                            if is_neighbor_perimeter:
                                neighbor_abs_coord = (row_start + nr, col_start + nc, depth_start + nd)
                                if self.grid.is_valid_coord(neighbor_abs_coord):
                                    idx2 = _ravel_multi_index(np.array(neighbor_abs_coord), self.grid.dimensions)
                                    # Add edge only once (e.g., when idx1 < idx2)
                                    if idx1 < idx2:
                                        self.grid.add_edge(idx1, idx2)

    def _place_polygon_perimeter(self, polygon: Polygon, origin: Tuple[int, ...]):
        """Helper method to connect perimeter nodes for a polygon."""
        # --- Type hint changed from shape: Shape to polygon: Polygon ---
        # Ensure origin matches grid dimensions
        if len(origin) != len(self.grid.dimensions):
             if len(self.grid.dimensions) == 2 and len(origin) == 3: origin = origin[:2]
             else: raise ValueError("Origin dimensions mismatch grid dimensions")

        # Get relative coordinates (assumed to be perimeter for non-filled)
        relative_coords = polygon.get_relative_coordinates()
        if not relative_coords: return

        # Convert relative to absolute and get indices
        placed_indices = set()
        coord_map = {}
        for rel_coord in relative_coords:
             abs_coord = tuple(o + r for o, r in zip(origin, rel_coord))
             if self.grid.is_valid_coord(abs_coord):
                 idx = _ravel_multi_index(np.array(abs_coord), self.grid.dimensions)
                 placed_indices.add(idx)
                 coord_map[idx] = abs_coord

        # Connect adjacent nodes within the placed set
        placed_nodes_list = list(placed_indices)
        for i in range(len(placed_nodes_list)):
            for j in range(i + 1, len(placed_nodes_list)):
                idx1 = placed_nodes_list[i]
                idx2 = placed_nodes_list[j]
                coords1 = coord_map[idx1]
                coords2 = coord_map[idx2]
                # Check if they are potential neighbors before adding edge
                if self.grid.are_neighbors(coords1, coords2):
                    self.grid.add_edge(idx1, idx2)


class ManageCategoriesDialog(tk.Toplevel):
    """Modal dialog for adding, renaming, and deleting shape categories."""

    def __init__(self, parent, shape_manager: ShapeLibraryManager, editor_window: 'ShapeLibraryEditorWindow'):
        super().__init__(parent)
        self.parent_editor = editor_window # Reference to the editor window
        self.shape_manager = shape_manager
        self.title("Manage Shape Categories")
        # --- MODIFIED: Increased width from 400 to 550 ---
        self.geometry("550x500")
        # --- END MODIFIED ---
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        # --- ADDED: Track newly added categories in this session ---
        self.added_categories_session: Set[str] = set()
        # ---

        self._create_widgets()
        self._populate_category_list()

        self.protocol("WM_DELETE_WINDOW", self.destroy) # Handle closing via 'X'

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Existing Categories:").pack(anchor=tk.W, pady=(0, 5))

        # Listbox with Scrollbar
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.category_listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set, exportselection=False,
            selectbackground="#0078D7", selectforeground="white"
        )
        self.category_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.category_listbox.yview)
        self.category_listbox.bind("<<ListboxSelect>>", self._on_category_select)

        # Entry for New/Rename
        entry_frame = ttk.Frame(main_frame)
        entry_frame.pack(fill=tk.X, pady=5)
        ttk.Label(entry_frame, text="Name:").pack(side=tk.LEFT, padx=(0, 5))
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(entry_frame, textvariable=self.name_var)
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        self.add_button = ttk.Button(button_frame, text="Add New", command=self._add_category)
        self.add_button.pack(side=tk.LEFT, padx=5)
        self.rename_button = ttk.Button(button_frame, text="Rename Selected", command=self._rename_category, state=tk.DISABLED)
        self.rename_button.pack(side=tk.LEFT, padx=5)
        self.delete_button = ttk.Button(button_frame, text="Delete Selected", command=self._delete_category, state=tk.DISABLED)
        self.delete_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=self.destroy).pack(side=tk.RIGHT)

    def _populate_category_list(self):
        """Populate the listbox with current categories, including newly added ones."""
        self.category_listbox.delete(0, tk.END)
        # Get categories directly from the manager's current state
        manager_categories = set(self.shape_manager.get_categories().keys())
        # Combine with categories added in this session
        all_categories = manager_categories.union(self.added_categories_session)
        # Exclude special/internal categories if needed (e.g., "Favorites")
        display_categories = sorted([cat for cat in all_categories if cat != "Favorites"])

        for category in display_categories:
            self.category_listbox.insert(tk.END, category)
        self._on_category_select() # Update button states

    def _on_category_select(self, event=None):
        """Handle selection change in the category listbox."""
        selected_indices = self.category_listbox.curselection()
        if selected_indices:
            selected_category = self.category_listbox.get(selected_indices[0])
            self.name_var.set(selected_category)
            self.rename_button.config(state=tk.NORMAL)
            # Prevent deleting essential categories if needed
            if selected_category in ["Uncategorized", "Custom", "RLE Imports"]: # Example protected categories
                 self.delete_button.config(state=tk.DISABLED)
            else:
                 self.delete_button.config(state=tk.NORMAL)
        else:
            self.name_var.set("")
            self.rename_button.config(state=tk.DISABLED)
            self.delete_button.config(state=tk.DISABLED)

    def _add_category(self):
        """Adds a new category based on the entry field to the session list."""
        new_name = self.name_var.get().strip().title()
        if not new_name:
            messagebox.showerror("Error", "Category name cannot be empty.", parent=self)
            return

        # Check against combined list (manager + session)
        manager_categories = set(self.shape_manager.get_categories().keys())
        all_current_categories = manager_categories.union(self.added_categories_session)

        if new_name in all_current_categories:
            messagebox.showerror("Error", f"Category '{new_name}' already exists or was just added.", parent=self)
            return

        # Add to session set
        self.added_categories_session.add(new_name)
        logger.info(f"Category '{new_name}' added to session list.")

        # --- REMOVED Popup ---
        # messagebox.showinfo("Category Added", f"Category '{new_name}' will be available when saving shapes.", parent=self)

        self.name_var.set("") # Clear entry
        # Refresh the list in this dialog (will now include the new one)
        self._populate_category_list()
        # Refresh the tree in the parent editor window (optional, category won't show until used)
        # if self.parent_editor and self.parent_editor.winfo_exists():
        #     self.parent_editor._populate_treeview()

    def _rename_category(self):
        """Renames the selected category."""
        selected_indices = self.category_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Please select a category to rename.", parent=self)
            return

        old_name = self.category_listbox.get(selected_indices[0])
        new_name = self.name_var.get().strip().title()

        if not new_name:
            messagebox.showerror("Error", "New category name cannot be empty.", parent=self)
            return
        if new_name == old_name:
            messagebox.showinfo("No Change", "New name is the same as the old name.", parent=self)
            return

        # Check against combined list (manager + session) excluding the old name
        manager_categories = set(self.shape_manager.get_categories().keys())
        all_other_categories = (manager_categories.union(self.added_categories_session)) - {old_name}
        if new_name in all_other_categories:
            messagebox.showerror("Error", f"Category '{new_name}' already exists.", parent=self)
            return

        if old_name in ["Uncategorized", "Custom", "RLE Imports"]: # Protect essential categories
             messagebox.showerror("Error", f"Cannot rename the protected category '{old_name}'.", parent=self)
             return

        if messagebox.askyesno("Confirm Rename", f"Rename category '{old_name}' to '{new_name}'?\nThis will update all shapes currently in this category.", parent=self):
            try:
                # --- Update session set if renaming a newly added category ---
                was_in_session = False
                if old_name in self.added_categories_session:
                    self.added_categories_session.remove(old_name)
                    self.added_categories_session.add(new_name)
                    was_in_session = True
                    logger.debug(f"Renamed category '{old_name}' to '{new_name}' within session set.")
                # ---

                # Attempt rename in manager (only affects shapes if category exists there)
                success = self.shape_manager.rename_category(old_name, new_name)

                if success or was_in_session: # Proceed if manager rename worked OR if it was just a session rename
                    messagebox.showinfo("Success", f"Category renamed to '{new_name}'.", parent=self)
                    self._populate_category_list() # Refresh this dialog's list
                    self.name_var.set("") # Clear entry
                    # Refresh the tree in the parent editor window
                    if self.parent_editor and self.parent_editor.winfo_exists():
                        self.parent_editor._populate_treeview()
                else:
                    # If manager rename failed AND it wasn't just a session rename, show error
                    messagebox.showerror("Error", "Failed to rename category (see logs).", parent=self)
                    # Revert session change if manager failed
                    if was_in_session:
                        self.added_categories_session.remove(new_name)
                        self.added_categories_session.add(old_name)

            except Exception as e:
                logger.error(f"Error renaming category: {e}")
                messagebox.showerror("Error", f"Failed to rename category: {e}", parent=self)
                # Revert session change on error
                if was_in_session:
                    self.added_categories_session.remove(new_name)
                    self.added_categories_session.add(old_name)

    def _delete_category(self):
        """Deletes the selected category, moving its shapes to 'Uncategorized'."""
        selected_indices = self.category_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Please select a category to delete.", parent=self)
            return

        category_to_delete = self.category_listbox.get(selected_indices[0])

        if category_to_delete in ["Uncategorized", "Custom", "RLE Imports"]: # Protect essential categories
             messagebox.showerror("Error", f"Cannot delete the protected category '{category_to_delete}'.", parent=self)
             return

        # --- Check if deleting a session-only category ---
        is_session_only = category_to_delete in self.added_categories_session and category_to_delete not in self.shape_manager.get_categories()
        # ---

        confirm_message = f"Delete category '{category_to_delete}'?"
        if not is_session_only:
            confirm_message += "\nShapes in this category will be moved to 'Uncategorized'."

        if messagebox.askyesno("Confirm Delete", confirm_message, icon='warning', parent=self):
            try:
                success = True
                # --- Remove from session set ---
                if category_to_delete in self.added_categories_session:
                    self.added_categories_session.remove(category_to_delete)
                    logger.debug(f"Removed category '{category_to_delete}' from session set.")
                # ---

                # --- Attempt delete in manager only if it's not session-only ---
                if not is_session_only:
                    success = self.shape_manager.delete_category(category_to_delete)
                # ---

                if success:
                    messagebox.showinfo("Success", f"Category '{category_to_delete}' deleted.", parent=self)
                    self._populate_category_list() # Refresh this dialog's list
                    self.name_var.set("") # Clear entry
                    # Refresh the tree in the parent editor window
                    if self.parent_editor and self.parent_editor.winfo_exists():
                        self.parent_editor._populate_treeview()
                else:
                    messagebox.showerror("Error", "Failed to delete category (see logs).", parent=self)
                    # Re-add to session set if manager delete failed but it was removed
                    if not is_session_only and category_to_delete not in self.added_categories_session:
                         self.added_categories_session.add(category_to_delete)

            except Exception as e:
                logger.error(f"Error deleting category: {e}")
                messagebox.showerror("Error", f"Failed to delete category: {e}", parent=self)

class EditHotmenuModal(tk.Toplevel):
    """Modal dialog for viewing and removing shapes from the hotmenu."""

    def __init__(self, parent, gui: 'SimulationGUI'):
        super().__init__(parent)
        self.parent_gui = gui
        self.title("Edit Shape Hotmenu")
        self.geometry("350x400")
        self.resizable(False, True) # Allow vertical resize
        self.transient(parent)
        self.grab_set()

        # Store a temporary copy of the hotmenu list for editing
        self.temp_hotmenu_list = list(self.parent_gui.hotmenu_shape_names) # Make a copy

        self._create_widgets()
        self._populate_list()

        self.protocol("WM_DELETE_WINDOW", self._on_cancel) # Handle closing via 'X'

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Shapes currently in right-click hotmenu:").pack(anchor=tk.W, pady=(0, 5))

        # Listbox with Scrollbar
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.hotmenu_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            exportselection=False,
            selectbackground="#0078D7",
            selectforeground="white"
        )
        self.hotmenu_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.hotmenu_listbox.yview)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(button_frame, text="Remove Selected", command=self._remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save & Close", command=self._on_save_close).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT)

    def _populate_list(self):
        """Populate the listbox with the current temporary hotmenu items."""
        self.hotmenu_listbox.delete(0, tk.END)
        for item in sorted(self.temp_hotmenu_list): # Display sorted
            self.hotmenu_listbox.insert(tk.END, item)

    def _remove_selected(self):
        """Remove the selected item from the temporary list and update the listbox."""
        selected_indices = self.hotmenu_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select a shape to remove.", parent=self)
            return

        selected_name = self.hotmenu_listbox.get(selected_indices[0])
        if selected_name in self.temp_hotmenu_list:
            self.temp_hotmenu_list.remove(selected_name)
            self._populate_list() # Refresh the listbox display
            logger.debug(f"Removed '{selected_name}' from temporary hotmenu list.")
        else:
            logger.warning(f"'{selected_name}' not found in temporary hotmenu list (should not happen).")

    def _on_save_close(self):
        """Save the changes to the main GUI list and file, then close."""
        self.parent_gui.hotmenu_shape_names = sorted(self.temp_hotmenu_list) # Update main list (sorted)
        self.parent_gui._save_hotmenu_shapes() # Save to file
        logger.info("Hotmenu updated and saved.")
        # Update the Shape Editor buttons if it's still open
        if hasattr(self.parent_gui, 'shape_editor_window') and self.parent_gui.shape_editor_window and self.parent_gui.shape_editor_window.winfo_exists():
             self.parent_gui.shape_editor_window._update_action_button_states()
        self.grab_release()
        self.destroy()

    def _on_cancel(self):
        """Close the dialog without saving changes."""
        logger.debug("Edit Hotmenu cancelled.")
        self.grab_release()
        self.destroy()

class SaveShapeDialog(tk.Toplevel):
    """Modal dialog for getting shape metadata (Name, Category, Desc, Tags)."""

    def __init__(self, parent, shape_manager: ShapeLibraryManager, initial_name: str = "New Shape", existing_names: Optional[List[str]] = None, current_rule_name: Optional[str] = None): # Added current_rule_name
        super().__init__(parent)
        self.parent = parent
        self.shape_manager = shape_manager
        self.existing_names = set(existing_names) if existing_names else set()
        self.result: Optional[Dict[str, Any]] = None

        self.title("Save Shape Definition")
        self.geometry("450x450") # Increased height slightly
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        # Variables
        self.name_var = tk.StringVar(value=initial_name)
        self.category_var = tk.StringVar(value="Custom") # Default category
        self.tags_var = tk.StringVar(value="custom, selection") # Default tags

        # --- Widgets ---
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Name
        ttk.Label(main_frame, text="Shape Name:").pack(anchor=tk.W, padx=5)
        self.name_entry = ttk.Entry(main_frame, textvariable=self.name_var)
        self.name_entry.pack(fill=tk.X, padx=5, pady=2)
        self.name_entry.focus_set() # Focus on name entry initially
        self.name_entry.bind("<Return>", lambda e: self._validate_and_save()) # Allow Enter to save

        # Category (Combobox)
        ttk.Label(main_frame, text="Category:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        existing_categories = sorted(list(self.shape_manager.get_categories().keys()))
        if "Custom" not in existing_categories: existing_categories.insert(0, "Custom")
        self.category_combobox = ttk.Combobox(
            main_frame,
            textvariable=self.category_var,
            values=existing_categories
            # Allow typing for new categories
        )
        self.category_combobox.pack(fill=tk.X, padx=5, pady=2)

        # Description
        ttk.Label(main_frame, text="Description:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.desc_text = tk.Text(main_frame, height=4, width=40, wrap=tk.WORD)
        self.desc_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        # Tags
        ttk.Label(main_frame, text="Tags (comma-separated):").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.tags_entry = ttk.Entry(main_frame, textvariable=self.tags_var)
        self.tags_entry.pack(fill=tk.X, padx=5, pady=2)

        # --- ADDED: Rule Display ---
        ttk.Label(main_frame, text="Intended Rule (Informational):").pack(anchor=tk.W, padx=5, pady=(10, 0))
        rule_display_text = current_rule_name if current_rule_name else "(Not specified)"
        self.rule_label = ttk.Label(main_frame, text=rule_display_text, relief=tk.SUNKEN, padding=2)
        self.rule_label.pack(fill=tk.X, padx=5, pady=2)
        # ---

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        ttk.Button(button_frame, text="Save", command=self._validate_and_save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT)

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.wait_window(self)

    def _validate_and_save(self):
        """Validate input and set result."""
        name = self.name_var.get().strip()
        category = self.category_var.get().strip().title() # Normalize category
        description = self.desc_text.get("1.0", tk.END).strip()
        tags_str = self.tags_var.get().strip()
        tags = [t.strip() for t in tags_str.split(',') if t.strip()]

        if not name:
            messagebox.showerror("Validation Error", "Shape name cannot be empty.", parent=self)
            self.name_entry.focus_set()
            return

        if not category:
            # Default to "Custom" if left empty
            category = "Custom"
            self.category_var.set(category)

        # Check for overwrite (only if name hasn't changed from initial suggestion if that existed)
        # This logic might need refinement depending on how initial_name is used.
        # For now, always check if the final name exists.
        if name in self.existing_names:
             if not messagebox.askyesno("Confirm Overwrite", f"Shape '{name}' already exists. Overwrite?", icon='warning', parent=self):
                 return

        self.result = {
            "name": name,
            "category": category,
            "description": description,
            "tags": tags
        }
        self.destroy()

    def _on_cancel(self):
        """Handle Cancel button click or window close."""
        self.result = None
        self.destroy()
  
class ShapeLibraryEditorWindow(tk.Toplevel):
    """
    A non-modal window for browsing, editing, creating, and placing shapes
    from a shape library, and for accessing grid editing tools.
    """
    _instance: Optional['ShapeLibraryEditorWindow'] = None  # Add this class attribute
    # --- Type Hints ---
    metadata_editors: Dict[str, Union[tk.Entry, tk.Text]]
    parent_gui: 'SimulationGUI'
    shape_manager: ShapeLibraryManager
    selected_shape_name: Optional[str]
    selected_shape_def: Optional[ShapeDefinition]
    active_tool: Optional[str]
    tree: Optional[ttk.Treeview]
    search_var: tk.StringVar
    preview_canvas_widget: Optional[FigureCanvasTkAgg]
    preview_fig: Figure
    preview_ax: Any
    metadata_labels: Dict[str, tk.Label]
    tool_buttons: Dict[str, tk.Button]
    action_buttons: Dict[str, tk.Button]

    def __init__(self, parent: 'SimulationGUI', shape_manager: ShapeLibraryManager):
        """Initialize the Shape Library Editor window."""
        if not parent or not parent.root or not parent.root.winfo_exists():
             logger.error("ShapeLibraryEditorWindow: Invalid parent window.")
             raise ValueError("Cannot create ShapeLibraryEditorWindow with invalid parent.")

        super().__init__(parent.root) # Initialize Toplevel
        self.parent_gui = parent
        self.shape_manager = shape_manager
        self.title("Shape Library & Editor")
        self.transient(parent.root) # Keep it on top of the main window

        # --- MODIFIED: Size and Position Calculation ---
        # Set initial size first
        # Make it wider (e.g., 1200 or relative to parent width)
        editor_width = max(1200, int(parent.root.winfo_width() * 0.8)) # Wider default, at least 1200
        editor_height = int(700 * 1.25) # Keep increased height
        self.geometry(f"{editor_width}x{editor_height}")
        self.update_idletasks() # Process size request

        # Position to the RIGHT of the parent window
        parent_x = parent.root.winfo_rootx()
        parent_y = parent.root.winfo_rooty()
        parent_width = parent.root.winfo_width()
        parent_height = parent.root.winfo_height()

        x_pos = parent_x + parent_width + 10 # Position to the right with a 10px gap
        y_pos = parent_y + (parent_height - editor_height) // 2 # Center vertically

        # Ensure position is on screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        if x_pos + editor_width > screen_width: # Adjust if it goes off-screen right
            x_pos = screen_width - editor_width - 10
        x_pos = max(0, x_pos) # Ensure non-negative X
        y_pos = max(0, y_pos) # Ensure non-negative Y

        self.geometry(f"+{x_pos}+{y_pos}") # Set position
        # --- END MODIFIED ---

        self.minsize(800, 500) # Increased minimum width
        logger.info(f"Set ShapeLibraryEditorWindow initial geometry: {editor_width}x{editor_height}+{x_pos}+{y_pos}")

        # --- Attributes ---
        self.selected_shape_name: Optional[str] = None
        self.selected_shape_def: Optional[ShapeDefinition] = None
        self.active_tool: Optional[str] = None
        self._initial_pane_configure_done = False # Flag for Configure binding

        # --- Attributes for RLE loading progress ---
        self._rle_load_thread: Optional[threading.Thread] = None
        self._rle_load_cancel_event: Optional[threading.Event] = None
        self._rle_load_progress_queue: Optional[queue.Queue] = None
        self._rle_progress_dialog: Optional[tk.Toplevel] = None
        self._rle_progress_bar: Optional[ttk.Progressbar] = None
        self._rle_progress_label: Optional[tk.Label] = None

        # UI Element References
        self.tree: Optional[ttk.Treeview] = None
        self.search_var = tk.StringVar()
        self.preview_canvas_widget: Optional[FigureCanvasTkAgg] = None
        self.preview_fig: Figure
        self.preview_ax: Any # Axes or Axes3D, handle dynamically
        self.metadata_labels: Dict[str, tk.Label] = {}
        self.metadata_editors: Dict[str, Union[tk.Entry, tk.Text]] = {}
        self.tool_buttons: Dict[str, tk.Button] = {}
        self.action_buttons: Dict[str, tk.Button] = {}
        self.main_pane: Optional[tk.PanedWindow] = None # Add reference for deferred sizing
        self.preview_label_frame: Optional[ttk.LabelFrame] = None # Reference for preview label

        # Initialize preview figure/axes
        self.preview_fig = Figure(figsize=(3, 3), dpi=100)
        self.preview_fig.set_facecolor('#e0e0e0') # Match background
        self.preview_ax = None # Axes added dynamically
        self.preview_canvas_widget = None # Canvas widget reference

        # --- Create Widgets ---
        self._create_widgets() # Creates panes but doesn't size them yet
        self._populate_treeview()
        self._draw_shape_preview(None) # Initial empty preview

        # --- Bindings ---
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<Configure>", self._on_initial_configure, add="+")

        self._initial_pane_configure_done = True

    def _create_widgets(self):
        """Create the main layout and widgets for the editor.
           (Round 17 Fix: Remove Grid Tools section)
           (Round 19 Fix: Add Manage Categories button)"""
        button_bg = "#e1e1e1"; button_fg = "black"; button_disabled_fg = "black"
        self.active_tool_bg = "#a1d9ed"

        self.main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Left Pane ---
        self.left_frame = ttk.Frame(self.main_pane, padding=5)
        search_frame = ttk.Frame(self.left_frame); search_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        search_entry.bind("<KeyRelease>", self._filter_treeview)
        self.tree = ttk.Treeview(self.left_frame, columns=("name",), show="tree headings", selectmode="browse"); self.tree.heading("#0", text="Category"); self.tree.heading("name", text="Shape Name"); self.tree.column("#0", width=100, stretch=False); self.tree.column("name", width=150, stretch=True); self.tree.pack(fill=tk.BOTH, expand=True); self.tree.bind("<<TreeviewSelect>>", self._on_shape_selected)
        self.main_pane.add(self.left_frame, minsize=150)

        # --- Center Pane ---
        self.center_frame = ttk.Frame(self.main_pane, padding=5)
        self.preview_label_frame = ttk.LabelFrame(self.center_frame, text="Preview", padding=5);
        self.preview_label_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.preview_canvas_widget = FigureCanvasTkAgg(self.preview_fig, master=self.preview_label_frame); self.preview_canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        info_label_frame = ttk.LabelFrame(self.center_frame, text="Info", padding=5); info_label_frame.pack(fill=tk.X, pady=(5, 0))
        info_grid = ttk.Frame(info_label_frame); info_grid.pack(fill=tk.X)
        metadata_display_fields = ['Category', 'Description', 'Tags', 'Author', 'Created', 'Modified', 'Rules'];
        for i, label_text in enumerate(metadata_display_fields):
             ttk.Label(info_grid, text=f"{label_text}:", anchor="nw", width=12).grid(row=i, column=0, sticky="nw", padx=2, pady=1)
             wrap = 300 if label_text in ["Description", "Tags", "Rules"] else 0
             self.metadata_labels[label_text.lower()] = tk.Label(info_grid, text="-", anchor="nw", wraplength=wrap, justify=tk.LEFT); self.metadata_labels[label_text.lower()].grid(row=i, column=1, sticky="nw", padx=2, pady=1)
             info_grid.columnconfigure(1, weight=1)
        self.main_pane.add(self.center_frame, minsize=200)

        # --- Right Pane ---
        self.right_frame = ttk.Frame(self.main_pane, padding=5)
        edit_frame = ttk.LabelFrame(self.right_frame, text="Edit Metadata", padding=5); edit_frame.pack(fill=tk.X, pady=(0, 10))
        metadata_edit_fields = ['Name', 'Category', 'Author', 'Tags', 'Rules'];
        for field in metadata_edit_fields:
            f = ttk.Frame(edit_frame); f.pack(fill=tk.X, pady=1); ttk.Label(f, text=f"{field}:", width=10).pack(side=tk.LEFT);
            if field == 'Rules':
                entry = tk.Text(f, height=2, width=10, state=tk.DISABLED, wrap=tk.WORD) # Use Text widget
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            else:
                entry = tk.Entry(f, state=tk.DISABLED)
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.metadata_editors[field.lower()] = entry
        f_desc = ttk.Frame(edit_frame); f_desc.pack(fill=tk.BOTH, expand=True, pady=1); ttk.Label(f_desc, text="Desc:", width=10).pack(side=tk.LEFT, anchor='n'); desc_text = tk.Text(f_desc, height=4, width=10, state=tk.DISABLED, wrap=tk.WORD); desc_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); self.metadata_editors['description'] = desc_text

        # --- Actions Frame ---
        actions_frame = ttk.LabelFrame(self.right_frame, text="Actions", padding=5)
        actions_frame.pack(fill=tk.X, pady=(0,10))
        self.add_edges_on_place_var = tk.BooleanVar(value=False)
        add_edges_check = tk.Checkbutton(
            actions_frame, text="Add Default Edges on Place",
            variable=self.add_edges_on_place_var, anchor=tk.W,
        )
        add_edges_check.pack(fill=tk.X, padx=2, pady=(5, 2))
        action_buttons_info = {
            "Place Shape": self.parent_gui._place_shape_definition_from_editor,
            "Save Selection": self._save_selection_as_shape,
            "Save Grid as New Shape...": self.parent_gui._save_grid_as_new_shape,
            "Save Selected Shape As...": self._save_selected_shape_as_dialog,
            "Update Shape": self._update_selected_shape,
            "Delete Shape": self._delete_selected_shape,
            "Add Default Edges to Selection": self.parent_gui._add_default_edges_to_selection,
            "Remove All Edges from Selection": self.parent_gui._remove_edges_from_selection,
            "Add to Hotmenu": self._add_to_hotmenu,
            "Edit Hotmenu...": self._open_edit_hotmenu_modal,
        }
        for name, cmd in action_buttons_info.items():
            button_state = tk.DISABLED # Default to disabled
            if name in ["Save Grid as New Shape...", "Edit Hotmenu..."]: button_state = tk.NORMAL
            btn = tk.Button(actions_frame, text=name, state=button_state, command=cmd if cmd else lambda n=name: logger.info(f"Button '{n}' clicked (no action yet)."),
                            fg=button_fg, bg=button_bg, disabledforeground=button_disabled_fg)
            btn.pack(fill=tk.X, padx=2, pady=2)
            action_key = name.lower().replace(" ", "_").replace("...","").replace("✓","").strip() # Clean key
            self.action_buttons[action_key] = btn

        self.main_pane.add(self.right_frame, minsize=180)

        # --- Bottom Bar ---
        bottom_bar = ttk.Frame(self, padding=(5, 10, 5, 5)); bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(bottom_bar, text="Load RLE File...", command=self._load_rle_file_dialog, fg=button_fg, bg=button_bg, disabledforeground=button_disabled_fg).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_bar, text="Load RLE Dir...", command=self._load_rle_directory_dialog, fg=button_fg, bg=button_bg, disabledforeground=button_disabled_fg).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_bar, text="Load Library", command=self._load_library_file, fg=button_fg, bg=button_bg, disabledforeground=button_disabled_fg).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_bar, text="Save Library", command=self._save_library_file, fg=button_fg, bg=button_bg, disabledforeground=button_disabled_fg).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_bar, text="Export Library as RLE...", command=self._export_library_as_rle_dialog, fg=button_fg, bg=button_bg, disabledforeground=button_disabled_fg).pack(side=tk.LEFT, padx=5)
        # --- ADDED Manage Categories Button ---
        tk.Button(bottom_bar, text="Manage Categories...", command=self._open_manage_categories_modal, fg=button_fg, bg=button_bg, disabledforeground=button_disabled_fg).pack(side=tk.LEFT, padx=5)
        # ---
        tk.Button(bottom_bar, text="Close", command=self.on_close, fg=button_fg, bg=button_bg, disabledforeground=button_disabled_fg).pack(side=tk.RIGHT, padx=5)

        self._update_action_button_states() # Initial state update

    def _activate_tool(self, tool_key: str):
            """Activates a specific tool in the main GUI and updates button highlights."""
            logger.debug(f"Activating tool: {tool_key}")
            self.parent_gui.set_active_tool(tool_key) # Call GUI method to set state
            self._update_tool_button_highlight()

    def _activate_lasso_tool(self): self._activate_tool("lasso")
    def _activate_erase_tool(self): self._activate_tool("erase")
    def _activate_add_edge_tool(self): self._activate_tool("add_edge")
    def _activate_del_edge_tool(self): self._activate_tool("del_edge")
    
    def _update_tool_button_highlight(self):
        """Updates the background color of tool buttons based on the active tool."""
        active_tool = self.parent_gui.active_tool
        logger.debug(f"Updating tool button highlights. Active tool: {active_tool}")
        for tool_key, button in self.tool_buttons.items():
            # Define default colors (ensure these are accessible)
            default_bg = "#e1e1e1" # Or get from widget default
            active_bg = getattr(self, 'active_tool_bg', "#a1d9ed") # Use defined active color

            # --- REMOVED check for 'grab' ---
            if tool_key == active_tool:
            # ---
                button.config(bg=active_bg)
            else:
                # Only reset if the button exists and has a background property
                if button and hasattr(button, 'config'):
                    try:
                        button.config(bg=default_bg)
                    except tk.TclError:
                        logger.warning(f"Could not reset background for button {tool_key}, widget might be destroyed.")

    def _configure_panes(self):
        """Configure the initial relative widths of the PanedWindow panes using paneconfigure -width."""
        if not self.main_pane or not self.main_pane.winfo_exists():
            logger.warning("Cannot configure panes, main_pane not ready.")
            return

        try:
            self.update_idletasks() # Ensure dimensions are up-to-date
            editor_width = self.main_pane.winfo_width()
            if editor_width <= 1:
                logger.warning(f"PanedWindow width ({editor_width}) invalid during configure.")
                return

            # --- MODIFIED: Use paneconfigure -width based on proportions ---
            min_left = 150
            min_center = 300 # Keep increased min center width
            min_right = 180
            total_min = min_left + min_center + min_right

            # Target proportions
            prop_left = 0.25
            prop_center = 0.50
            prop_right = 0.25

            # Calculate target widths
            target_left = int(editor_width * prop_left)
            target_center = int(editor_width * prop_center)
            target_right = int(editor_width * prop_right)

            # Ensure minimums are met and adjust if total exceeds editor width
            left_width = max(min_left, target_left)
            center_width = max(min_center, target_center)
            right_width = max(min_right, target_right)

            total_calculated = left_width + center_width + right_width
            if total_calculated > editor_width:
                excess = total_calculated - editor_width
                reducible_width = (left_width - min_left) + (center_width - min_center) + (right_width - min_right)
                if reducible_width > 0:
                    reduction_factor = excess / reducible_width
                    left_width -= int((left_width - min_left) * reduction_factor)
                    center_width -= int((center_width - min_center) * reduction_factor)
                    right_width -= int((right_width - min_right) * reduction_factor)
                    # Assign remainder to center pane
                    final_total = left_width + center_width + right_width
                    center_width += (editor_width - final_total)
                else: # Cannot reduce further, distribute proportionally based on minimums
                    left_width = int(editor_width * (min_left / total_min))
                    center_width = int(editor_width * (min_center / total_min))
                    right_width = editor_width - left_width - center_width

            # Configure each pane using paneconfigure
            panes = self.main_pane.panes()
            if len(panes) == 3:
                # Get the actual widget references for each pane
                left_pane_widget = self.left_frame # Assuming self.left_frame holds the widget added first
                center_pane_widget = self.center_frame # Assuming self.center_frame holds the widget added second
                right_pane_widget = self.right_frame # Assuming self.right_frame holds the widget added third

                # Configure each pane with its calculated width and minsize
                self.main_pane.paneconfigure(left_pane_widget, width=left_width, minsize=min_left)
                self.main_pane.paneconfigure(center_pane_widget, width=center_width, minsize=min_center)
                self.main_pane.paneconfigure(right_pane_widget, width=right_width, minsize=min_right)

                logger.info(f"Configured PanedWindow panes using paneconfigure -width: Left={left_width}, Center={center_width}, Right={right_width}")
            else:
                logger.warning(f"Expected 3 panes, found {len(panes)}. Cannot configure widths.")
            # --- END MODIFIED ---

        except tk.TclError as e:
            logger.error(f"TclError configuring PanedWindow panes: {e}. Window might not be fully ready.")
        except Exception as e:
            logger.error(f"Error configuring PanedWindow panes: {e}")
            logger.error(traceback.format_exc())

    def _on_initial_configure(self, event):
        """Callback for the initial <Configure> event to set pane sizes."""
        # Check if this is the first configure event and the flag is not set
        if not self._initial_pane_configure_done:
            # Check if the event width seems reasonable (sometimes initial events have 1x1)
            if event.widget == self and event.width > 100 and event.height > 100:
                logger.debug(f"Initial <Configure> event triggered with size {event.width}x{event.height}. Setting pane sizes.")
                self._configure_panes()
                self._initial_pane_configure_done = True
                # Optional: Unbind after first successful configuration?
                # self.unbind("<Configure>")
            else:
                logger.debug(f"Ignoring initial <Configure> event with small size: {event.width}x{event.height}")

    def _load_rle_file_dialog(self):
        """Open file dialog to select a single RLE file."""
        filepath = filedialog.askopenfilename(
            title="Load RLE File",
            filetypes=[("RLE files", "*.rle"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.shape_manager.library_path) if hasattr(self.shape_manager, 'library_path') and self.shape_manager.library_path else self.parent_gui.app_paths.get('data', '.')
        )
        if filepath:
            if self.shape_manager.load_rle_file(filepath):
                self._populate_treeview() # Refresh view
                # --- REMOVED Success Messagebox ---
                # messagebox.showinfo("Success", f"Shape loaded from:\n{filepath}", parent=self)
                logger.info(f"Shape loaded successfully from: {filepath}") # Log instead
            else:
                messagebox.showerror("Error", f"Failed to load shape from:\n{filepath}", parent=self)

    def _load_rle_directory_dialog(self):
        """Open directory dialog, load RLE files in background, show progress."""
        dirpath = filedialog.askdirectory(
            title="Select Directory Containing RLE Files",
            initialdir=os.path.dirname(self.shape_manager.library_path) if hasattr(self.shape_manager, 'library_path') and self.shape_manager.library_path else self.parent_gui.app_paths.get('data', '.')
        )
        if dirpath:
            if self._rle_load_thread and self._rle_load_thread.is_alive():
                messagebox.showwarning("Busy", "Already loading RLE files.", parent=self)
                return

            # Create progress dialog
            self._rle_progress_dialog = tk.Toplevel(self)
            self._rle_progress_dialog.title("Loading RLE Files...")
            self._rle_progress_dialog.geometry("400x100")
            self._rle_progress_dialog.resizable(False, False)
            self._rle_progress_dialog.transient(self)
            self._rle_progress_dialog.grab_set()
            self._rle_progress_dialog.protocol("WM_DELETE_WINDOW", self._cancel_rle_load)

            self._rle_progress_label = tk.Label(self._rle_progress_dialog, text="Scanning directory...")
            self._rle_progress_label.pack(pady=10, padx=10, fill=tk.X)

            self._rle_progress_bar = ttk.Progressbar(self._rle_progress_dialog, orient="horizontal", length=350, mode="determinate")
            self._rle_progress_bar.pack(pady=5, padx=10, fill=tk.X)

            cancel_button = tk.Button(self._rle_progress_dialog, text="Cancel", command=self._cancel_rle_load)
            cancel_button.pack(pady=10)

            # Setup communication
            self._rle_load_cancel_event = threading.Event()
            # --- MODIFIED: Use qualified queue.Queue ---
            self._rle_load_progress_queue = queue.Queue()
            # ---

            # Start background thread
            self._rle_load_thread = threading.Thread(
                target=self.shape_manager.load_rle_directory,
                args=(dirpath, self._rle_load_progress_queue, self._rle_load_cancel_event),
                daemon=True
            )
            self._rle_load_thread.start()

            # Start checking progress queue
            self.after(100, self._check_rle_load_progress, dirpath)

    def _cancel_rle_load(self):
        """Signal the RLE loading thread to cancel and close the progress dialog."""
        if self._rle_load_cancel_event:
            logger.info("Cancel button clicked or progress dialog closed for RLE load.")
            self._rle_load_cancel_event.set() # Signal the thread to stop
        else:
            logger.warning("Cancel requested, but cancel event object does not exist.")

        # Close the progress dialog immediately if it exists
        if self._rle_progress_dialog and self._rle_progress_dialog.winfo_exists():
            try:
                self._rle_progress_dialog.grab_release()
                self._rle_progress_dialog.destroy()
                logger.debug("RLE progress dialog destroyed by cancel request.")
            except tk.TclError as e:
                 logger.warning(f"TclError destroying progress dialog on cancel: {e}")
            except Exception as e:
                 logger.error(f"Error destroying progress dialog on cancel: {e}")
            finally:
                 self._rle_progress_dialog = None # Clear reference
                 self._rle_progress_bar = None
                 self._rle_progress_label = None
        else:
            logger.debug("Progress dialog already closed or non-existent during cancel.")

        # Note: The background thread will check the event and exit,
        # and the _check_rle_load_progress will handle final cleanup.

    def _check_rle_load_progress(self, dirpath: str):
        """Periodically check the queue for progress updates."""
        if not self._rle_load_thread or not self._rle_load_progress_queue or not self._rle_progress_dialog or not self._rle_progress_dialog.winfo_exists():
            logger.debug("_check_rle_load_progress: Thread, queue, or dialog missing/closed.")
            return

        try:
            # Process all messages currently in the queue
            while True:
                # --- MODIFIED: Use qualified queue.Empty ---
                try:
                    processed, total = self._rle_load_progress_queue.get_nowait()
                except queue.Empty:
                # ---
                    break # Exit inner loop if queue is empty

                if total > 0:
                    progress_val = (processed / total) * 100
                    if self._rle_progress_bar: self._rle_progress_bar['value'] = progress_val
                    if self._rle_progress_label: self._rle_progress_label.config(text=f"Loading... ({processed}/{total})")
                else:
                     if self._rle_progress_bar: self._rle_progress_bar['value'] = 0
                     if self._rle_progress_label: self._rle_progress_label.config(text=f"Scanning... (0 files)")
                if self._rle_progress_dialog and self._rle_progress_dialog.winfo_exists():
                    self._rle_progress_dialog.update_idletasks()

            # Check if thread is done after processing queue
            if self._rle_load_thread.is_alive():
                # Thread still running, schedule next check
                if self._rle_progress_dialog and self._rle_progress_dialog.winfo_exists():
                    self.after(100, self._check_rle_load_progress, dirpath)
            else:
                # Thread finished, process final result
                logger.info("RLE load thread finished.")
                if self._rle_progress_dialog and self._rle_progress_dialog.winfo_exists():
                    self._rle_progress_dialog.grab_release()
                    self._rle_progress_dialog.destroy()
                    self._rle_progress_dialog = None

                was_cancelled = self._rle_load_cancel_event.is_set() if self._rle_load_cancel_event else False

                self._populate_treeview()
                if was_cancelled:
                     messagebox.showinfo("Load Cancelled", "RLE directory loading was cancelled.", parent=self)
                else:
                     # We still don't have error list here, rely on manager logs
                     messagebox.showinfo("Load Complete", f"Finished loading RLE files from:\n{dirpath}\n(Check logs for details/errors)", parent=self)

                # Clean up thread-related attributes
                self._rle_load_thread = None
                self._rle_load_cancel_event = None
                self._rle_load_progress_queue = None

        except Exception as e:
            logger.error(f"Error checking RLE load progress: {e}")
            if self._rle_progress_dialog and self._rle_progress_dialog.winfo_exists():
                self._rle_progress_dialog.grab_release()
                self._rle_progress_dialog.destroy()
            messagebox.showerror("Error", f"Error during RLE load progress update:\n{e}", parent=self)
            # Clean up thread-related attributes
            self._rle_load_thread = None
            self._rle_load_cancel_event = None
            self._rle_load_progress_queue = None

    def _export_library_as_rle_dialog(self):
        """Handles the 'Export Library as RLE...' button click."""
        dirpath = filedialog.askdirectory(
            title="Select Directory to Export RLE Files",
            initialdir=os.path.dirname(self.shape_manager.library_path) # Start near library
        )
        if dirpath:
            try:
                success_count, errors = self.shape_manager.save_library_as_rle_directory(dirpath)
                if not errors:
                    messagebox.showinfo("Export Complete", f"Successfully exported {success_count} shapes as RLE files to:\n{dirpath}", parent=self)
                else:
                    error_list_str = "\n - ".join(errors)
                    messagebox.showwarning("Export Partially Complete",
                                           f"Exported {success_count} shapes.\n\nFailed to export {len(errors)} shapes:\n - {error_list_str}",
                                           parent=self)
            except Exception as e:
                logger.error(f"Error exporting library as RLE: {e}")
                messagebox.showerror("Error", f"Failed to export library as RLE:\n{e}", parent=self)

    def _get_edited_shape_definition(self) -> Optional[ShapeDefinition]:
        """Gets the current shape definition including edits from the UI."""
        if not self.selected_shape_def: return None

        try:
            edited_def = copy.deepcopy(self.selected_shape_def)

            # Update fields from editor widgets
            name_widget = self.metadata_editors.get('name')
            if isinstance(name_widget, tk.Entry): edited_def.name = name_widget.get()
            cat_widget = self.metadata_editors.get('category')
            if isinstance(cat_widget, tk.Entry): edited_def.category = cat_widget.get()
            auth_widget = self.metadata_editors.get('author')
            if isinstance(auth_widget, tk.Entry): edited_def.author = auth_widget.get()
            tags_widget = self.metadata_editors.get('tags')
            if isinstance(tags_widget, tk.Entry):
                tags_str = tags_widget.get(); edited_def.tags = [t.strip() for t in tags_str.split(',') if t.strip()]
            desc_widget = self.metadata_editors.get('description')
            if isinstance(desc_widget, tk.Text): edited_def.description = desc_widget.get("1.0", tk.END).strip()
            # --- ADDED: Get rules from Text widget ---
            rules_widget = self.metadata_editors.get('rules')
            if isinstance(rules_widget, tk.Text):
                rules_str = rules_widget.get("1.0", tk.END).strip()
                edited_def.rules = [r.strip() for r in rules_str.splitlines() if r.strip()] # Split by newline
            # ---

            edited_def.date_modified = datetime.now().strftime("%Y-%m-%d")

            if not edited_def.name:
                messagebox.showerror("Validation Error", "Shape name cannot be empty.", parent=self)
                return None

            return edited_def
        except Exception as e:
            logger.error(f"Error getting edited shape definition: {e}")
            return None
        
    def _select_shape_in_tree(self, shape_name: str):
        """Finds and selects a shape by name in the treeview."""
        if not self.tree: return
        for cat_id in self.tree.get_children():
            for shape_id in self.tree.get_children(cat_id):
                item_values = self.tree.item(shape_id, 'values')
                if item_values and item_values[0] == shape_name:
                    self.tree.selection_set(shape_id)
                    self.tree.focus(shape_id)
                    self.tree.see(shape_id)
                    self._on_shape_selected() # Trigger update
                    return
                
    def _filter_treeview(self, event=None):
        """Filters the treeview based on the search entry."""
        filter_term = self.search_var.get()
        self._populate_treeview(filter_term)

    def _populate_treeview(self, filter_term: str = ""):
        """Clear and populate the treeview with categories and shapes, applying filter."""
        if not self.tree: return
        # Delete existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Get categorized shapes
        categories = self.shape_manager.get_categories()
        filter_term_lower = filter_term.lower()

        # Populate tree
        for category, shape_names in categories.items():
            category_has_match = False
            category_node_id = None # Store category node ID

            # Filter shape names first
            filtered_shape_names = []
            if not filter_term_lower: # No filter, include all
                filtered_shape_names = shape_names
                category_has_match = True
            else:
                for shape_name in shape_names:
                    if filter_term_lower in shape_name.lower():
                        filtered_shape_names.append(shape_name)
                        category_has_match = True

            # Only add category if it has matching shapes
            if category_has_match:
                category_node_id = self.tree.insert("", tk.END, text=category, open=True) # Insert category node
                for shape_name in filtered_shape_names:
                    # Insert shape under its category, store name in 'values'
                    self.tree.insert(category_node_id, tk.END, values=(shape_name,))

    def _open_edit_hotmenu_modal(self):
        """Opens the modal dialog to edit the hotmenu."""
        # Pass the main GUI instance (self.parent_gui) to the modal
        edit_modal = EditHotmenuModal(self, self.parent_gui)
        # The modal handles its own lifecycle and saving

    def _on_shape_selected(self, event=None):
        """Handle selection change in the treeview."""
        if self.tree is None:
            logger.error("Treeview is not initialized.")
            return

        selected_item = self.tree.focus()
        if not selected_item:
            self.selected_shape_name = None
            self.selected_shape_def = None
            self._update_metadata_display(None)
            self._populate_metadata_editor(None)
            self._draw_shape_preview(None) # Clear preview
            self._update_action_button_states()
            return

        item_values = self.tree.item(selected_item, 'values')

        if item_values and item_values[0]:
            self.selected_shape_name = item_values[0]
            logger.debug(f"Shape selected: {self.selected_shape_name}")
            self.selected_shape_def = self.shape_manager.get_shape(self.selected_shape_name) if self.selected_shape_name else None
            self._update_metadata_display(self.selected_shape_def)
            self._populate_metadata_editor(self.selected_shape_def)
            self._draw_shape_preview(self.selected_shape_def) # Draw selected shape
        else:
            self.selected_shape_name = None
            self.selected_shape_def = None
            self._update_metadata_display(None)
            self._populate_metadata_editor(None)
            self._draw_shape_preview(None) # Clear preview
            logger.debug(f"Category selected: {self.tree.item(selected_item, 'text')}")

        self._update_action_button_states()

    def _update_selected_shape(self):
        """Updates the selected shape definition with the current grid selection after confirmation."""
        logger.info("Attempting to update selected shape with grid selection.")

        # 1. Check Prerequisites
        if not self.selected_shape_def:
            messagebox.showwarning("No Shape Selected", "Please select a shape from the library to update.", parent=self)
            return
        if self.parent_gui.grid is None:
            messagebox.showerror("Error", "Grid is not initialized.", parent=self)
            return
        selection = self.parent_gui.current_selection
        selected_node_coords_abs = selection.get('nodes')
        if not selected_node_coords_abs:
            messagebox.showwarning("Empty Selection", "No nodes selected on the grid to use for update.", parent=self)
            return

        original_shape_name = self.selected_shape_def.name # Store name before potential changes

        try:
            # 2. Get Grid Selection Data (similar to _save_selection_as_shape)
            active_coords_abs = list(selected_node_coords_abs)
            dims = len(active_coords_abs[0])
            min_coords = list(active_coords_abs[0])
            for coord in active_coords_abs[1:]:
                for d in range(dims): min_coords[d] = min(min_coords[d], coord[d])
            origin_offset = tuple(min_coords)
            selection_relative_coords = [tuple(c - mc for c, mc in zip(abs_coord, origin_offset)) for abs_coord in active_coords_abs]
            abs_to_rel_map = dict(zip(active_coords_abs, selection_relative_coords))

            selection_node_states_rel: Dict[Tuple[int, ...], float] = {}
            for abs_coord in active_coords_abs:
                rel_coord = abs_to_rel_map.get(abs_coord)
                if rel_coord is not None:
                    try: selection_node_states_rel[rel_coord] = float(self.parent_gui.grid.grid_array[abs_coord])
                    except: pass # Ignore errors getting state

            selection_relative_edges_list: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
            selection_edge_states_rel: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
            active_coords_abs_set = set(active_coords_abs)
            for edge_abs, state in self.parent_gui.grid.edge_states.items():
                node1_abs, node2_abs = edge_abs
                if node1_abs in active_coords_abs_set and node2_abs in active_coords_abs_set:
                    rel_node1 = abs_to_rel_map.get(node1_abs); rel_node2 = abs_to_rel_map.get(node2_abs)
                    if rel_node1 is not None and rel_node2 is not None:
                        ordered_rel_edge_tuple = (rel_node1, rel_node2) if rel_node1 < rel_node2 else (rel_node2, rel_node1)
                        if ordered_rel_edge_tuple not in selection_edge_states_rel:
                            selection_relative_edges_list.append(ordered_rel_edge_tuple)
                            selection_edge_states_rel[ordered_rel_edge_tuple] = float(state)
            selection_connectivity = "explicit" if selection_relative_edges_list else "none"
            # --- End Selection Data Gathering ---

            # 3. Compare Selection with Original Shape Definition
            original_coords_set = set(self.selected_shape_def.relative_coords)
            selection_coords_set = set(selection_relative_coords)
            added_coords = selection_coords_set - original_coords_set
            removed_coords = original_coords_set - selection_coords_set
            logger.debug(f"Comparison: Added={added_coords}, Removed={removed_coords}")

            # 4. Draw Comparison Preview
            # Create a temporary figure and canvas for the dialog preview
            temp_fig = Figure(figsize=(4, 4), dpi=100)
            temp_fig.set_facecolor('#e0e0e0')
            temp_canvas = FigureCanvasTkAgg(temp_fig, master=self) # Master is the editor window

            # Draw the comparison onto the temporary figure's axes
            # We need to temporarily replace self.preview_fig/ax for the draw call
            original_fig, original_ax, original_canvas = self.preview_fig, self.preview_ax, self.preview_canvas_widget
            self.preview_fig, self.preview_canvas_widget = temp_fig, temp_canvas
            try:
                self._draw_shape_preview(self.selected_shape_def, added_coords, removed_coords)
            finally:
                # Restore original figure/canvas references
                self.preview_fig, self.preview_canvas_widget = original_fig, original_canvas
                # We don't need to restore self.preview_ax as _draw_shape_preview recreates it

            # 5. Show Confirmation Dialog
            confirmation_dialog = UpdateShapeConfirmationDialog(self, original_shape_name, temp_canvas)
            confirm_overwrite = confirmation_dialog.result

            # 6. Handle Confirmation Result
            if confirm_overwrite:
                logger.info(f"User confirmed overwrite for shape '{original_shape_name}'.")
                # Create the updated ShapeDefinition (keep original metadata, update geometry/state)
                updated_shape_def = ShapeDefinition(
                    name=original_shape_name, # Keep original name
                    category=self.selected_shape_def.category, # Keep original metadata
                    description=self.selected_shape_def.description,
                    tags=self.selected_shape_def.tags,
                    author=self.selected_shape_def.author,
                    date_created=self.selected_shape_def.date_created,
                    date_modified=datetime.now().strftime("%Y-%m-%d"), # Update modified date
                    intended_rule=self.selected_shape_def.intended_rule,
                    rule_string=self.selected_shape_def.rule_string,
                    rule_compatibility=self.selected_shape_def.rule_compatibility,
                    # Use data from the selection
                    relative_coords=selection_relative_coords,
                    connectivity=selection_connectivity,
                    relative_edges=selection_relative_edges_list if selection_relative_edges_list else None,
                    node_states=selection_node_states_rel if selection_node_states_rel else None,
                    edge_states=selection_edge_states_rel if selection_edge_states_rel else None,
                )

                # Save using manager (will overwrite)
                if self.shape_manager.add_shape(updated_shape_def):
                    messagebox.showinfo("Success", f"Shape '{original_shape_name}' updated successfully.", parent=self)
                    # Refresh editor UI
                    self.selected_shape_def = updated_shape_def # Update the selected def in the editor
                    self._populate_treeview()
                    self._select_shape_in_tree(original_shape_name) # Reselect
                    # Redraw the normal preview (without highlights)
                    self._draw_shape_preview(self.selected_shape_def)
                else:
                    messagebox.showerror("Error", f"Failed to update shape '{original_shape_name}'.", parent=self)
            else:
                logger.info("User cancelled shape update.")
                # Redraw the original preview if cancelled
                self._draw_shape_preview(self.selected_shape_def)

        except Exception as e:
            logger.error(f"Error updating shape: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to update shape: {e}", parent=self)
            # Redraw original preview on error
            self._draw_shape_preview(self.selected_shape_def)

    def _delete_selected_shape(self):
        """Deletes the currently selected shape from the library after confirmation."""
        if not self.selected_shape_name:
            messagebox.showwarning("No Shape Selected", "Please select a shape from the library to delete.", parent=self)
            return

        shape_name_to_delete = self.selected_shape_name # Store before clearing

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to permanently delete the shape '{shape_name_to_delete}'?", icon='warning', parent=self):
            try:
                if self.shape_manager.delete_shape(shape_name_to_delete):
                    logger.info(f"Shape '{shape_name_to_delete}' deleted successfully.")
                    # --- MODIFIED: Clear selection and refresh UI ---
                    self.selected_shape_name = None
                    self.selected_shape_def = None
                    self._populate_treeview() # Refresh the list
                    self._update_metadata_display(None) # Clear info pane
                    self._populate_metadata_editor(None) # Clear and disable editor fields
                    self._draw_shape_preview(None) # Clear preview
                    self._update_action_button_states() # Update button states (Delete should become disabled)
                    # --- END MODIFIED ---
                    messagebox.showinfo("Success", f"Shape '{shape_name_to_delete}' deleted.", parent=self)
                else:
                    # delete_shape already shows an error message if not found
                    logger.warning(f"Failed to delete shape '{shape_name_to_delete}' (likely not found by manager).")

            except Exception as e:
                logger.error(f"Error deleting shape '{shape_name_to_delete}': {e}")
                messagebox.showerror("Error", f"Failed to delete shape: {e}", parent=self)

    def _save_selection_as_shape(self):
        """Saves the currently selected nodes/edges on the main grid as a new shape definition.
           (Round 11 Fix: Use shape_manager.get_shape() for existence check)"""
        logger.info("Attempting to save grid selection as new shape.")
        if self.parent_gui.grid is None:
            messagebox.showerror("Error", "Grid is not initialized.", parent=self)
            return

        # Get current selection from the main GUI
        selection = self.parent_gui.current_selection
        selected_node_coords_abs = selection.get('nodes')

        if not selected_node_coords_abs:
            messagebox.showwarning("Empty Selection", "No nodes selected on the grid to save.", parent=self)
            return

        try:
            # [ Calculate Relative Coords and Origin Offset - Unchanged ]
            active_coords_abs = list(selected_node_coords_abs) # Convert set to list
            if not active_coords_abs: return # Should not happen based on check above, but safe

            dims = len(active_coords_abs[0])
            min_coords = list(active_coords_abs[0])
            for coord in active_coords_abs[1:]:
                for d in range(dims):
                    min_coords[d] = min(min_coords[d], coord[d])
            origin_offset = tuple(min_coords)

            relative_coords = [tuple(c - mc for c, mc in zip(abs_coord, origin_offset)) for abs_coord in active_coords_abs]
            abs_to_rel_map = dict(zip(active_coords_abs, relative_coords))
            # ---

            # [ Capture Node States - Unchanged ]
            node_states_rel: Dict[Tuple[int, ...], float] = {}
            for abs_coord in active_coords_abs:
                rel_coord = abs_to_rel_map.get(abs_coord)
                if rel_coord is not None:
                    try:
                        node_states_rel[rel_coord] = float(self.parent_gui.grid.grid_array[abs_coord])
                    except IndexError: logger.warning(f"IndexError getting state for {abs_coord} while saving selection.")
                    except Exception as e: logger.error(f"Error getting state for {abs_coord}: {e}")
            # ---

            # [ Capture Edges and Edge States within the selection - Unchanged ]
            relative_edges_list: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
            edge_states_rel: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
            active_coords_abs_set = set(active_coords_abs) # Use set for faster lookup

            for edge_abs, state in self.parent_gui.grid.edge_states.items():
                node1_abs, node2_abs = edge_abs
                if node1_abs in active_coords_abs_set and node2_abs in active_coords_abs_set:
                    rel_node1 = abs_to_rel_map.get(node1_abs)
                    rel_node2 = abs_to_rel_map.get(node2_abs)
                    if rel_node1 is not None and rel_node2 is not None:
                        ordered_rel_edge_tuple = (rel_node1, rel_node2) if rel_node1 < rel_node2 else (rel_node2, rel_node1)
                        if ordered_rel_edge_tuple not in edge_states_rel:
                            relative_edges_list.append(ordered_rel_edge_tuple)
                            edge_states_rel[ordered_rel_edge_tuple] = float(state)
            # ---

            # [ Prompt for Metadata ]
            shape_name = simpledialog.askstring("Save Selection as Shape", "Enter Shape Name:", parent=self)
            if not shape_name: return
            # --- MODIFIED: Use get_shape for existence check ---
            if self.shape_manager.get_shape(shape_name) is not None:
            # ---
                if not messagebox.askyesno("Overwrite Shape", f"Shape '{shape_name}' already exists. Overwrite?", icon='warning', parent=self):
                    return

            category = simpledialog.askstring("Save Shape", "Enter Category:", initialvalue="Selection", parent=self) or "Selection"
            description = simpledialog.askstring("Save Shape", "Enter Description:", initialvalue="Saved from grid selection.", parent=self) or ""
            tags_str = simpledialog.askstring("Save Shape", "Enter Tags (comma-separated):", initialvalue="selection", parent=self) or ""
            tags = [t.strip() for t in tags_str.split(',') if t.strip()]
            connectivity = "explicit" if relative_edges_list else "none"
            # ---

            # [ Create ShapeDefinition - Unchanged ]
            shape_def = ShapeDefinition(
                name=shape_name,
                category=category,
                description=description,
                relative_coords=relative_coords,
                connectivity=connectivity,
                tags=tags,
                author="User",
                relative_edges=relative_edges_list if relative_edges_list else None,
                node_states=node_states_rel if node_states_rel else None,
                edge_states=edge_states_rel if edge_states_rel else None,
                intended_rule=self.parent_gui.rule.name if self.parent_gui.rule else None
            )
            # ---

            # [ Save using Manager - Unchanged ]
            if self.shape_manager.add_shape(shape_def):
                messagebox.showinfo("Success", f"Shape '{shape_name}' saved successfully from selection.", parent=self)
                self._populate_treeview() # Refresh editor list
                self._select_shape_in_tree(shape_name) # Select the newly saved shape
            else:
                 messagebox.showerror("Error", f"Failed to save shape '{shape_name}'.", parent=self)
            # ---

        except Exception as e:
            logger.error(f"Error saving selection as shape: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to save selection: {e}", parent=self)

    def _save_selected_shape_as_dialog(self):
        """Handles the 'Save Selected Shape As...' button click. Saves the *library definition*.
           (Round 12 Fix: Use get_shape for existence check)"""
        if not self.selected_shape_def:
            messagebox.showwarning("No Shape Selected", "Please select a shape from the library to save.", parent=self)
            return

        logger.info(f"Initiating 'Save As...' for library shape definition: '{self.selected_shape_def.name}'")

        file_types = [("LACE Shape JSON", "*.json"), ("Run Length Encoded", "*.rle"), ("All files", "*.*")]
        initial_name = self.selected_shape_def.name
        default_ext = ".json"

        filepath = filedialog.asksaveasfilename(
            title="Save Shape Definition As...",
            filetypes=file_types,
            defaultextension=default_ext,
            initialfile=initial_name,
            initialdir=os.path.dirname(self.shape_manager.library_path)
        )

        if not filepath: return # User cancelled

        file_ext = os.path.splitext(filepath)[1].lower()
        new_shape_name = os.path.splitext(os.path.basename(filepath))[0]

        try:
            shape_to_save = copy.deepcopy(self.selected_shape_def)
            # Update with any edits made in the metadata fields
            name_widget = self.metadata_editors.get('name')
            if isinstance(name_widget, tk.Entry): shape_to_save.name = name_widget.get()
            cat_widget = self.metadata_editors.get('category')
            if isinstance(cat_widget, tk.Entry): shape_to_save.category = cat_widget.get()
            auth_widget = self.metadata_editors.get('author')
            if isinstance(auth_widget, tk.Entry): shape_to_save.author = auth_widget.get()
            tags_widget = self.metadata_editors.get('tags')
            if isinstance(tags_widget, tk.Entry):
                tags_str = tags_widget.get(); shape_to_save.tags = [t.strip() for t in tags_str.split(',') if t.strip()]
            desc_widget = self.metadata_editors.get('description')
            if isinstance(desc_widget, tk.Text): shape_to_save.description = desc_widget.get("1.0", tk.END).strip()
            shape_to_save.date_modified = datetime.now().strftime("%Y-%m-%d")

            # Set the name based on the chosen filename
            shape_to_save.name = new_shape_name

            if file_ext == ".rle":
                logger.info(f"Exporting shape '{shape_to_save.name}' to RLE file: {filepath}")
                rle_string = generate_rle(shape_to_save) # Correct static call
                with open(filepath, 'w', encoding='utf-8') as f: f.write(rle_string)
                messagebox.showinfo("Success", f"Shape definition '{shape_to_save.name}' saved as RLE:\n{filepath}", parent=self)
            else:
                logger.info(f"Saving shape definition '{shape_to_save.name}' to main library file: {self.shape_manager.library_path}")
                # --- MODIFIED: Use get_shape for existence check ---
                if self.shape_manager.get_shape(new_shape_name) is not None and new_shape_name != self.selected_shape_def.name:
                # ---
                    if not messagebox.askyesno("Confirm Overwrite", f"A shape named '{new_shape_name}' already exists in the library. Overwrite?", icon='warning', parent=self):
                        return # User cancelled overwrite
                if self.shape_manager.add_shape(shape_to_save):
                     messagebox.showinfo("Success", f"Shape definition '{shape_to_save.name}' saved to library:\n{self.shape_manager.library_path}", parent=self)
                     self._populate_treeview()
                     self._select_shape_in_tree(shape_to_save.name)
                else:
                     messagebox.showerror("Error", f"Failed to save shape definition '{shape_to_save.name}' to library.", parent=self)

        except Exception as e:
            logger.error(f"Error saving selected shape definition: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to save shape definition:\n{e}", parent=self)

    def _update_metadata_display(self, shape_def: Optional[ShapeDefinition]):
        """Update the labels in the Info pane."""
        if shape_def:
            self.metadata_labels['category'].config(text=shape_def.category)
            self.metadata_labels['description'].config(text=shape_def.description)
            self.metadata_labels['tags'].config(text=", ".join(shape_def.tags))
            self.metadata_labels['author'].config(text=shape_def.author)
            self.metadata_labels['created'].config(text=shape_def.date_created)
            self.metadata_labels['modified'].config(text=shape_def.date_modified)
            # --- ADDED: Display rules ---
            self.metadata_labels['rules'].config(text=", ".join(shape_def.rules) if shape_def.rules else "(None specified)")
            # ---
        else:
            for label in self.metadata_labels.values():
                label.config(text="-")

    def _draw_shape_preview(self, shape_def: Optional[ShapeDefinition],
                            added_coords: Optional[Set[Tuple[int,...]]] = None,
                            removed_coords: Optional[Set[Tuple[int,...]]] = None):
        """
        Draws the selected shape definition on the preview canvas.
        Optionally highlights added (green) and removed (red) coordinates for comparison.
        """
        if not self.preview_canvas_widget:
            logger.error("Preview canvas not initialized.")
            return

        # --- Clear existing axes or create new ones ---
        self.preview_fig.clear() # Clear the whole figure
        dims = shape_def.get_dimensions() if shape_def else 2 # Default to 2D if no shape
        if dims == 3:
            self.preview_ax = self.preview_fig.add_subplot(111, projection='3d')
            self.preview_ax.set_box_aspect([1, 1, 1]) # Keep aspect ratio for 3D
        else: # 2D
            self.preview_ax = self.preview_fig.add_subplot(111)
            self.preview_ax.set_aspect('equal', adjustable='box') # Keep aspect ratio for 2D

        # Basic styling
        self.preview_ax.set_xticks([])
        self.preview_ax.set_yticks([])
        if dims == 3: self.preview_ax.set_zticks([])
        self.preview_ax.set_facecolor('#f0f0f0') # Light grey background
        for spine in self.preview_ax.spines.values():
            spine.set_visible(False)
        # ---

        # --- Update Preview Label ---
        preview_label_text = "Preview"
        all_coords_for_bounds = set()
        if shape_def and shape_def.relative_coords:
            all_coords_for_bounds.update(shape_def.relative_coords)
        if added_coords: all_coords_for_bounds.update(added_coords)
        if removed_coords: all_coords_for_bounds.update(removed_coords)

        if all_coords_for_bounds:
            try:
                # Calculate bounds based on *all* coordinates being displayed
                coords_list = list(all_coords_for_bounds)
                min_c = list(coords_list[0]); max_c = list(coords_list[0])
                temp_dims = len(min_c)
                for coord in coords_list[1:]:
                    if len(coord) != temp_dims: continue
                    for d in range(temp_dims):
                        min_c[d] = min(min_c[d], coord[d])
                        max_c[d] = max(max_c[d], coord[d])
                min_c_tuple, max_c_tuple = tuple(min_c), tuple(max_c)
                # ---
                shape_size = tuple(int(mx - mn + 1) for mn, mx in zip(min_c_tuple, max_c_tuple)) # Calculate size
                size_str = "x".join(map(str, shape_size))
                preview_label_text = f"Preview ({size_str})"
            except Exception as e:
                logger.warning(f"Could not get bounding box for preview label: {e}")
        # Update the LabelFrame text
        if self.preview_label_frame:
            self.preview_label_frame.config(text=preview_label_text)
        # ---

        # --- Plotting Logic ---
        min_c_final, max_c_final = (0,)*dims, (0,)*dims # Default bounds
        center = [0.0]*dims
        half_extent = 0.5
        padding = 0.1

        if all_coords_for_bounds:
            # Recalculate bounds and center for plot limits using all coords
            coords_list = list(all_coords_for_bounds)
            min_c_final_list = list(coords_list[0]); max_c_final_list = list(coords_list[0])
            temp_dims = len(min_c_final_list)
            for coord in coords_list[1:]:
                if len(coord) != temp_dims: continue
                for d in range(temp_dims):
                    min_c_final_list[d] = min(min_c_final_list[d], coord[d])
                    max_c_final_list[d] = max(max_c_final_list[d], coord[d])
            min_c_final, max_c_final = tuple(min_c_final_list), tuple(max_c_final_list)
            center = [(mn + mx) / 2 for mn, mx in zip(min_c_final, max_c_final)]
            max_extent = max(mx - mn + 1 for mn, mx in zip(min_c_final, max_c_final))
            half_extent = max_extent / 2.0
            padding = half_extent * 0.2 # 20% padding

            # --- Draw Nodes with Highlighting ---
            node_size = 25
            # 1. Original nodes (black 's') - skip removed ones
            if shape_def and shape_def.relative_coords:
                original_coords_set = set(shape_def.relative_coords)
                coords_to_draw = np.array([c for c in original_coords_set if not removed_coords or c not in removed_coords])
                if coords_to_draw.size > 0:
                    if dims == 3: self.preview_ax.scatter(coords_to_draw[:, 1], coords_to_draw[:, 0], coords_to_draw[:, 2], s=node_size, c='black', marker='s', label='Existing')
                    else: self.preview_ax.scatter(coords_to_draw[:, 1], coords_to_draw[:, 0], s=node_size, c='black', marker='s', label='Existing')

            # 2. Added nodes (green '+')
            if added_coords:
                added_coords_arr = np.array(list(added_coords))
                if added_coords_arr.size > 0:
                    if dims == 3: self.preview_ax.scatter(added_coords_arr[:, 1], added_coords_arr[:, 0], added_coords_arr[:, 2], s=node_size*1.5, c='green', marker='P', label='Added') # 'P' is plus sign filled
                    else: self.preview_ax.scatter(added_coords_arr[:, 1], added_coords_arr[:, 0], s=node_size*1.5, c='green', marker='P', label='Added')

            # 3. Removed nodes (red 'x')
            if removed_coords:
                removed_coords_arr = np.array(list(removed_coords))
                if removed_coords_arr.size > 0:
                    if dims == 3: self.preview_ax.scatter(removed_coords_arr[:, 1], removed_coords_arr[:, 0], removed_coords_arr[:, 2], s=node_size*1.5, c='red', marker='X', label='Removed') # 'X' is x filled
                    else: self.preview_ax.scatter(removed_coords_arr[:, 1], removed_coords_arr[:, 0], s=node_size*1.5, c='red', marker='X', label='Removed')
            # ---

            # Set limits centered on the combined shape
            if dims == 3:
                self.preview_ax.set_xlim(center[1] - half_extent - padding, center[1] + half_extent + padding)
                self.preview_ax.set_ylim(center[0] - half_extent - padding, center[0] + half_extent + padding)
                self.preview_ax.set_zlim(center[2] - half_extent - padding, center[2] + half_extent + padding)
                self.preview_ax.view_init(elev=20, azim=30)
            else: # 2D
                self.preview_ax.set_xlim(center[1] - half_extent - padding, center[1] + half_extent + padding)
                self.preview_ax.set_ylim(center[0] - half_extent - padding, center[0] + half_extent + padding)
                self.preview_ax.invert_yaxis()

            # Add legend if highlighting was used
            if added_coords or removed_coords:
                self.preview_ax.legend(loc='upper right', fontsize='small')

        else:
            # Display placeholder text if no shape or no coords
            text_to_display = "Select a shape" if shape_def is None else "Shape has no coordinates"
            self.preview_ax.text(0.5, 0.5, text_to_display, ha='center', va='center', transform=self.preview_ax.transAxes)
        # ---

        # Redraw the canvas
        try:
            self.preview_canvas_widget.draw_idle()
        except Exception as e:
            logger.error(f"Error drawing preview canvas: {e}")

    def _populate_metadata_editor(self, shape_def: Optional[ShapeDefinition]):
        """Populate and enable/disable the metadata editor fields."""
        is_editing = shape_def is not None

        # Enable/disable fields
        for widget in self.metadata_editors.values():
            if isinstance(widget, (tk.Entry, tk.Text)):
                widget.config(state=tk.NORMAL if is_editing else tk.DISABLED)
            else:
                 logger.warning(f"Unexpected widget type in metadata_editors: {type(widget)}")

        # Populate fields
        if shape_def:
            name_widget = self.metadata_editors.get('name')
            if isinstance(name_widget, tk.Entry): name_widget.delete(0, tk.END); name_widget.insert(0, shape_def.name)
            cat_widget = self.metadata_editors.get('category')
            if isinstance(cat_widget, tk.Entry): cat_widget.delete(0, tk.END); cat_widget.insert(0, shape_def.category)
            auth_widget = self.metadata_editors.get('author')
            if isinstance(auth_widget, tk.Entry): auth_widget.delete(0, tk.END); auth_widget.insert(0, shape_def.author)
            tags_widget = self.metadata_editors.get('tags')
            if isinstance(tags_widget, tk.Entry): tags_widget.delete(0, tk.END); tags_widget.insert(0, ", ".join(shape_def.tags))
            desc_widget = self.metadata_editors.get('description')
            if isinstance(desc_widget, tk.Text): desc_widget.delete("1.0", tk.END); desc_widget.insert("1.0", shape_def.description)
            # --- ADDED: Populate rules ---
            rules_widget = self.metadata_editors.get('rules')
            if isinstance(rules_widget, tk.Text):
                rules_widget.delete("1.0", tk.END)
                rules_widget.insert("1.0", "\n".join(shape_def.rules) if shape_def.rules else "") # Use newline separation
            # ---
        else:
            # Clear fields if no shape selected
            for name, widget in self.metadata_editors.items():
                if isinstance(widget, tk.Entry): widget.delete(0, tk.END)
                elif isinstance(widget, tk.Text): widget.delete("1.0", tk.END)

    def _update_action_button_states(self):
        """Enable/disable action buttons based on selection and rule context.
           (Round 18 Fix: Remove access to non-existent tool_buttons)"""
        shape_selected = self.selected_shape_name is not None
        selection_active = bool(self.parent_gui.current_selection.get('nodes')) if hasattr(self.parent_gui, 'current_selection') else False

        rule_supports_edges = False
        if self.parent_gui.controller and self.parent_gui.controller.rule:
            edge_init_type = self.parent_gui.controller.rule.get_param('edge_initialization', 'RANDOM')
            rule_supports_edges = edge_init_type != 'NONE'
        # logger.debug(f"Rule supports edges: {rule_supports_edges} (Edge Init Type: {edge_init_type if self.parent_gui.controller and self.parent_gui.controller.rule else 'N/A'})") # Reduce noise

        # --- Update Action Buttons ---
        self.action_buttons['place_shape'].config(state=tk.NORMAL if shape_selected else tk.DISABLED)
        self.action_buttons['save_selected_shape_as'].config(state=tk.NORMAL if shape_selected else tk.DISABLED)
        self.action_buttons['update_shape'].config(state=tk.NORMAL if shape_selected and selection_active else tk.DISABLED)
        self.action_buttons['delete_shape'].config(state=tk.NORMAL if shape_selected else tk.DISABLED)
        self.action_buttons['save_selection'].config(state=tk.NORMAL if selection_active else tk.DISABLED)
        self.action_buttons['add_default_edges_to_selection'].config(state=tk.NORMAL if selection_active and rule_supports_edges else tk.DISABLED)
        self.action_buttons['remove_all_edges_from_selection'].config(state=tk.NORMAL if selection_active and rule_supports_edges else tk.DISABLED)

        # --- Hotmenu Button States ---
        is_in_hotmenu = False
        if shape_selected and hasattr(self.parent_gui, 'hotmenu_shape_names'):
            is_in_hotmenu = self.selected_shape_name in self.parent_gui.hotmenu_shape_names

        self.action_buttons['add_to_hotmenu'].config(state=tk.NORMAL if shape_selected and not is_in_hotmenu else tk.DISABLED)
        # Edit Hotmenu button is always enabled
        self.action_buttons['edit_hotmenu'].config(state=tk.NORMAL)
        # ---

        # --- REMOVED Grid Tool Button Update Logic ---
        # ---

        # --- Call highlight update AFTER setting button states ---
        self._update_tool_button_highlight() # Keep this call, it now only highlights based on parent_gui.active_tool

    def _add_to_hotmenu(self):
        """Adds the currently selected shape to the hotmenu list."""
        if not self.selected_shape_name:
            messagebox.showwarning("No Shape Selected", "Please select a shape from the library first.", parent=self)
            return
        if not hasattr(self.parent_gui, 'hotmenu_shape_names'):
            logger.error("Main GUI does not have hotmenu_shape_names attribute.")
            return

        if self.selected_shape_name not in self.parent_gui.hotmenu_shape_names:
            self.parent_gui.hotmenu_shape_names.append(self.selected_shape_name)
            self.parent_gui._save_hotmenu_shapes() # Save the updated list
            logger.info(f"Added '{self.selected_shape_name}' to hotmenu.")
            self._update_action_button_states() # Update button states
            messagebox.showinfo("Hotmenu Updated", f"'{self.selected_shape_name}' added to the right-click hotmenu.", parent=self)
        else:
            logger.warning(f"'{self.selected_shape_name}' is already in the hotmenu.")

    def _open_manage_categories_modal(self):
        """Opens the modal dialog to manage shape categories."""
        # Pass self (the editor window) to the modal so it can refresh the treeview
        manage_modal = ManageCategoriesDialog(self, self.shape_manager, self)
        # The modal handles its own lifecycle and saving/refreshing

    def _load_library_file(self):
        """Load shapes from a user-selected JSON file."""
        filepath = filedialog.askopenfilename(
            title="Load Shape Library",
            filetypes=[("JSON files", "*.json")],
            initialdir=self.shape_manager.library_path # Start in default dir
        )
        if filepath:
            try:
                # Temporarily load to check validity before replacing current library
                with open(filepath, 'r', encoding='utf-8') as f:
                    temp_data = json.load(f)
                if 'shapes' not in temp_data or not isinstance(temp_data['shapes'], list):
                    raise ValueError("Invalid shape library format (missing 'shapes' list).")
                # Basic validation passed, now load it properly
                self.shape_manager.library_path = filepath # Update path
                self.shape_manager.load_shape_library() # Load into manager
                self._populate_treeview() # Refresh browser
                messagebox.showinfo("Success", f"Shape library loaded from:\n{filepath}", parent=self)
            except Exception as e:
                logger.error(f"Error loading shape library file {filepath}: {e}")
                messagebox.showerror("Error", f"Failed to load shape library:\n{e}", parent=self)

    def _save_library_file(self):
        """Save the current shape library to a user-selected JSON file."""
        filepath = filedialog.asksaveasfilename(
            title="Save Shape Library As",
            filetypes=[("JSON files", "*.json")],
            defaultextension=".json",
            initialfile="shapes.json",
            initialdir=os.path.dirname(self.shape_manager.library_path) # Start in current dir
        )
        if filepath:
            try:
                original_path = self.shape_manager.library_path
                self.shape_manager.library_path = filepath # Temporarily change path for saving
                self.shape_manager.save_shape_library()
                self.shape_manager.library_path = original_path # Restore original path
                messagebox.showinfo("Success", f"Shape library saved to:\n{filepath}", parent=self)
            except Exception as e:
                logger.error(f"Error saving shape library file to {filepath}: {e}")
                messagebox.showerror("Error", f"Failed to save shape library:\n{e}", parent=self)

    def on_close(self):
        """Handle window close event."""
        logger.debug("ShapeLibraryEditorWindow closing.")
        # Remove reference from parent GUI if it exists
        if hasattr(self.parent_gui, 'shape_editor_window') and self.parent_gui.shape_editor_window is self:
            self.parent_gui.shape_editor_window = None
        self.destroy()

    def on_rule_or_preset_change(self):
        """
        Called by SimulationGUI when the rule or preset changes.
        Updates button states and handles edge removal prompts if necessary.
        """
        logger.info("ShapeLibraryEditorWindow: Detected rule or preset change in main GUI.")

        # 1. Update button states based on the new context
        self._update_action_button_states()

        # 2. Check if edges need removal based on the new rule
        if self.parent_gui.controller and self.parent_gui.controller.rule and self.parent_gui.grid:
            new_rule = self.parent_gui.controller.rule
            edge_init_type = new_rule.get_param('edge_initialization', 'RANDOM')
            rule_supports_edges = edge_init_type != 'NONE'
            grid_has_edges = bool(self.parent_gui.grid.edges)

            logger.debug(f"  Rule Change Check: New rule '{new_rule.name}' supports edges: {rule_supports_edges}. Grid has edges: {grid_has_edges}")

            if not rule_supports_edges and grid_has_edges:
                logger.warning(f"New rule '{new_rule.name}' does not support edges, but the grid currently has edges.")
                if messagebox.askyesno(
                    "Remove Edges?",
                    f"The newly selected rule ('{new_rule.name}') does not use edges, but the current grid has edges.\n\nDo you want to remove all existing edges from the grid?",
                    icon='warning',
                    parent=self # Show prompt on top of this editor window
                ):
                    logger.info("User chose to remove edges.")
                    self.parent_gui._remove_all_grid_edges() # Call GUI method to remove edges
                else:
                    logger.info("User chose not to remove edges.")
        else:
            logger.warning("Could not check edge removal necessity: Controller, rule, or grid missing.")
            
class ShapeLibraryManager:
    """Manages shape library loading, saving, and access with lazy loading."""
    _instance: Optional['ShapeLibraryManager'] = None
    _initialized: bool = False
    _app_paths: Optional[Dict[str, str]] = None # Store app_paths at class level

    def __init__(self, app_paths: Optional[Dict[str, str]] = None):
        """Initialize the manager, preparing for lazy loading."""
        if ShapeLibraryManager._initialized:
            return

        # --- Store app_paths ---
        if app_paths is not None:
            ShapeLibraryManager._app_paths = app_paths
        elif ShapeLibraryManager._app_paths is None:
            # Fallback if initialized directly without paths passed to get_instance first
            logger.error("ShapeLibraryManager initialized without app_paths! Attempting fallback.")
            try:
                global APP_PATHS
                ShapeLibraryManager._app_paths = APP_PATHS
            except NameError:
                raise RuntimeError("APP_PATHS not defined. Initialize directories first or pass app_paths.")
        # Use the stored app_paths
        self.app_paths = ShapeLibraryManager._app_paths
        # ---

        # Determine library path
        if 'data' in self.app_paths:
            self.library_path = os.path.join(self.app_paths['data'], 'shapes.json')
        else:
            logger.error("ShapeLibraryManager: 'data' directory key missing in app_paths!")
            self.library_path = os.path.join(os.getcwd(), APP_DIR, 'Resources', 'data', 'shapes.json') # Fallback

        logger.info(f"Shape library path set to: {self.library_path}")

        # --- Attributes for lazy loading ---
        self._raw_shape_data: Dict[str, Dict[str, Any]] = {} # Stores raw dicts from JSON
        self._parsed_shapes_cache: Dict[str, ShapeDefinition] = {} # Caches parsed objects
        # ---

        self.load_shape_library() # Load raw data
        ShapeLibraryManager._initialized = True

    @classmethod
    def get_instance(cls, app_paths: Optional[Dict[str, str]] = None) -> 'ShapeLibraryManager':
        """Get the singleton instance of the ShapeLibraryManager."""
        if cls._instance is None:
            # Pass app_paths during the first instantiation
            cls._instance = ShapeLibraryManager(app_paths=app_paths)
        elif app_paths and cls._app_paths != app_paths:
             cls._app_paths = app_paths
             cls._instance.app_paths = app_paths # Update instance path reference
             # Optionally update library_path and reload if path depends on app_paths
             if 'data' in cls._instance.app_paths:
                 new_path = os.path.join(cls._instance.app_paths['data'], 'shapes.json')
                 if new_path != cls._instance.library_path:
                     cls._instance.library_path = new_path
                     logger.info(f"Updated shape library path to: {cls._instance.library_path}")
                     cls._instance.load_shape_library() # Reload raw data
             else:
                  logger.error("ShapeLibraryManager: 'data' directory key missing in updated app_paths!")
        return cls._instance

    def load_shape_library(self):
        """Loads the raw shape data from the JSON file, converting old formats if necessary."""
        logger.debug(f"Attempting to load RAW shape library data from: {self.library_path}")
        self._raw_shape_data = {}
        self._parsed_shapes_cache = {}
        library_modified = False # Flag to track if conversion happened

        try:
            os.makedirs(os.path.dirname(self.library_path), exist_ok=True)

            if not os.path.exists(self.library_path) or os.path.getsize(self.library_path) == 0:
                logger.warning(f"Shape library file not found or empty at {self.library_path}. Creating default library.")
                self._create_default_shape_library()
                self.save_shape_library() # Save the newly created default library
                # Raw data is populated by _create_default_shape_library
                return # Exit after creating default

            # Load existing file
            logger.debug(f"Loading existing file: {self.library_path}")
            with open(self.library_path, 'r', encoding='utf-8') as f:
                library_data = json.load(f)

            loaded_raw_shapes = {}
            if 'shapes' in library_data and isinstance(library_data['shapes'], list):
                for shape_data in library_data['shapes']:
                    if not isinstance(shape_data, dict):
                        logger.warning(f"Skipping invalid shape entry (not a dictionary): {shape_data}")
                        continue
                    shape_name = shape_data.get('name')
                    if not shape_name:
                        logger.warning(f"Skipping shape entry missing 'name': {shape_data}")
                        continue

                    # --- ADDED: Conversion Logic ---
                    old_dense_state = shape_data.get('initial_state')
                    has_sparse_state = 'initial_state_sparse' in shape_data

                    if isinstance(old_dense_state, list) and not has_sparse_state:
                        logger.info(f"Found old dense 'initial_state' for shape '{shape_name}'. Attempting conversion to sparse format.")
                        try:
                            # 1. Convert list to numpy array
                            state_array = np.array(old_dense_state, dtype=np.float64)
                            # 2. Generate sparse dictionary (coord tuple -> state float)
                            sparse_dict = {}
                            activity_threshold = 1e-6
                            non_zero_indices = np.argwhere(np.abs(state_array) > activity_threshold)
                            for coord_array in non_zero_indices:
                                coord_tuple = tuple(coord_array)
                                state_value = float(state_array[coord_tuple])
                                sparse_dict[coord_tuple] = state_value
                            # 3. Convert sparse dict keys for JSON
                            initial_state_sparse_for_json = {str(list(k)): v for k, v in sparse_dict.items()}
                            # 4. Update shape_data
                            shape_data['initial_state_sparse'] = initial_state_sparse_for_json
                            del shape_data['initial_state'] # Remove old key
                            library_modified = True # Mark for resave
                            logger.info(f"  Successfully converted '{shape_name}' to sparse format ({len(initial_state_sparse_for_json)} entries).")
                        except Exception as conversion_err:
                            logger.error(f"  Error converting dense state for '{shape_name}': {conversion_err}. Keeping original (potentially incompatible) data.")
                            # Keep the old 'initial_state' if conversion fails
                            shape_data['initial_state'] = old_dense_state # Ensure it's still there
                    elif isinstance(old_dense_state, list) and has_sparse_state:
                         # If both exist (unlikely), prioritize sparse and remove dense
                         logger.warning(f"Shape '{shape_name}' has both 'initial_state' (list) and 'initial_state_sparse'. Removing old 'initial_state'.")
                         del shape_data['initial_state']
                         library_modified = True
                    elif old_dense_state is not None and not isinstance(old_dense_state, list):
                         # If initial_state exists but isn't a list (e.g., already numpy or something else), remove it if sparse exists
                         if has_sparse_state:
                             logger.debug(f"Shape '{shape_name}' has 'initial_state_sparse' and non-list 'initial_state'. Removing 'initial_state'.")
                             del shape_data['initial_state']
                             library_modified = True
                         # Otherwise, keep the non-list initial_state (might be intended, though unusual)

                    # --- END CONVERSION LOGIC ---

                    if shape_name in loaded_raw_shapes:
                        logger.warning(f"Duplicate shape name '{shape_name}' found in library file. Overwriting raw data.")
                    loaded_raw_shapes[shape_name] = shape_data # Store potentially modified data
            else:
                logger.error(f"Invalid format in {self.library_path}: 'shapes' key missing or not a list.")
                self._create_default_shape_library()
                self.save_shape_library()
                return

            self._raw_shape_data = loaded_raw_shapes
            logger.info(f"Loaded {len(self._raw_shape_data)} raw shape definitions from {self.library_path}")

            # --- ADDED: Save back if modified ---
            if library_modified:
                logger.info("Library was modified during load (converted old format). Saving changes...")
                self.save_shape_library()
            # ---

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in shape library file {self.library_path}: {e}. Creating default library.")
            self._create_default_shape_library()
            self.save_shape_library()
        except Exception as e:
            logger.error(f"Error loading raw shape library: {e}")
            logger.error(traceback.format_exc())
            if not self._raw_shape_data:
                self._create_default_shape_library()

    def load_rle_file(self, filepath: str) -> bool:
        """Loads a single shape from an RLE file and adds it."""
        logger.info(f"Attempting to load RLE file: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            parsed_data = parse_rle(content)
            if parsed_data is None:
                logger.error(f"Failed to parse RLE file: {filepath}")
                return False

            shape_def = ShapeDefinition(
                name=parsed_data['name'], category="RLE Imports",
                description=parsed_data['description'] or f"Loaded from {os.path.basename(filepath)}.",
                relative_coords=parsed_data['relative_coords'], connectivity="none",
                tags=["RLE", parsed_data['rule_str']], author=parsed_data['author'],
                rule_string=parsed_data['rule_str']
            )
            return self.add_shape(shape_def) # Handles raw/cache
        except FileNotFoundError: logger.error(f"RLE file not found: {filepath}"); return False
        except Exception as e: logger.error(f"Error loading RLE file {filepath}: {e}"); logger.error(traceback.format_exc()); return False

    def load_rle_directory(self, dirpath: str,
                           progress_queue: Optional[queue.Queue] = None,
                           cancel_event: Optional[threading.Event] = None
                           ) -> Tuple[int, List[str], bool]:
        """Loads all .rle files from a directory using parallel processing."""
        if not os.path.isdir(dirpath):
            logger.error(f"Invalid directory path provided: {dirpath}")
            return 0, [], False

        logger.info(f"Scanning directory for RLE files (using parallel processing): {dirpath}")
        count = 0
        loaded_shapes_data = []
        errors = []
        was_cancelled = False
        loaded_names = []

        try:
            filepaths = []
            try:
                filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.lower().endswith(".rle")]
                logger.debug(f"Found {len(filepaths)} potential RLE files.")
            except Exception as list_err:
                logger.error(f"Error listing directory '{dirpath}': {list_err}")
                return 0, [], False

            total_files = len(filepaths)
            processed_count = 0

            if progress_queue: progress_queue.put((processed_count, total_files))

            process_pool = None
            gui_instance = None
            editor_instance = getattr(ShapeLibraryEditorWindow, '_instance', None)
            if editor_instance:
                 gui_instance = getattr(editor_instance, 'parent_gui', None)
            try:
                 if gui_instance and hasattr(gui_instance, 'controller') and gui_instance.controller:
                      process_pool = gui_instance.controller.process_pool
                      if process_pool is None or getattr(process_pool, '_broken', False):
                           if not gui_instance.controller._initialize_process_pool():
                                raise RuntimeError("Process pool initialization failed.")
                           process_pool = gui_instance.controller.process_pool
                 else:
                      logger.warning("Cannot access ProcessPoolExecutor via GUI context. RLE loading might be slow.")
                      process_pool = None
            except AttributeError:
                 logger.warning("Could not access process pool. RLE loading might be slow.")
                 process_pool = None

            if process_pool and total_files > 0:
                logger.info(f"Submitting {total_files} RLE parsing tasks to ProcessPoolExecutor.")
                futures = [process_pool.submit(_parse_rle_worker, fp) for fp in filepaths]

                for future in concurrent.futures.as_completed(futures):
                    if cancel_event and cancel_event.is_set():
                        logger.info("RLE directory load cancelled during processing.")
                        was_cancelled = True
                        for f in futures:
                            if not f.done(): f.cancel()
                        break

                    try:
                        result = future.result()
                        if result:
                            if 'data' in result:
                                loaded_shapes_data.append(result['data'])
                            elif 'error' in result:
                                errors.append(f"{os.path.basename(result['filepath'])} ({result['error']})")
                        else:
                             logger.warning("Worker returned None result.")
                             errors.append("Unknown file (worker returned None)")

                    except concurrent.futures.CancelledError:
                         logger.info("A future task was cancelled.")
                         was_cancelled = True
                    except Exception as e:
                        logger.error(f"Error retrieving result from RLE worker: {e}")
                        errors.append(f"Unknown file (future result error: {e})")

                    processed_count += 1
                    if progress_queue: progress_queue.put((processed_count, total_files))

            else: # Fallback to sequential loading
                 logger.info(f"Processing {total_files} RLE files sequentially.")
                 for filepath in filepaths:
                     if cancel_event and cancel_event.is_set():
                         logger.info("RLE directory load cancelled during sequential processing.")
                         was_cancelled = True
                         break
                     result = _parse_rle_worker(filepath)
                     if result:
                         if 'data' in result: loaded_shapes_data.append(result['data'])
                         elif 'error' in result: errors.append(f"{os.path.basename(result['filepath'])} ({result['error']})")
                     else: errors.append(f"{os.path.basename(filepath)} (worker returned None)")
                     processed_count += 1
                     if progress_queue: progress_queue.put((processed_count, total_files))

            # --- Process loaded data *after* all workers finish (if not cancelled) ---
            if not was_cancelled and loaded_shapes_data:
                logger.info(f"Processing {len(loaded_shapes_data)} successfully parsed shapes...")
                for parsed_data in loaded_shapes_data:
                    try:
                        shape_def = ShapeDefinition(
                            name=parsed_data['name'], category="RLE Imports",
                            description=parsed_data['description'] or f"Loaded from RLE. Rule: {parsed_data['rule_str']}",
                            relative_coords=parsed_data['relative_coords'], connectivity="none",
                            tags=["RLE", parsed_data['rule_str']], author=parsed_data['author'],
                            rule_string=parsed_data['rule_str']
                        )
                        if not shape_def.name:
                             logger.warning(f"Skipping shape due to empty name (originally from RLE).")
                             errors.append(f"{parsed_data.get('filepath', 'Unknown RLE')} (empty name)")
                        elif shape_def.name in self._raw_shape_data: # Check raw data keys
                             logger.warning(f"Shape '{shape_def.name}' already exists. Skipping.")
                        else:
                             if self.add_shape(shape_def): # Use add_shape
                                 count += 1
                                 loaded_names.append(shape_def.name)
                             else:
                                 errors.append(f"{parsed_data.get('name', 'Unknown RLE')} (add error)")
                    except Exception as add_err:
                         logger.error(f"Error creating/adding ShapeDefinition for parsed data {parsed_data.get('name', 'N/A')}: {add_err}")
                         errors.append(f"{parsed_data.get('name', 'Unknown RLE')} (add error)")
                # add_shape saves automatically, no need to save here
            # ---

            log_suffix = " (cancelled)" if was_cancelled else ""
            logger.info(f"Directory scan complete{log_suffix}. Successfully added {count} new shapes from RLE files.")
            if loaded_names: logger.debug(f"Added shapes: {', '.join(loaded_names)}")
            if errors: logger.warning(f"Failed to load or parse/add {len(errors)} RLE files: {', '.join(errors)}")

            return count, errors, was_cancelled

        except Exception as e:
             logger.error(f"Unexpected error scanning directory {dirpath}: {e}")
             logger.error(traceback.format_exc())
             if 'loaded_names' not in locals(): loaded_names = []
             return 0, [], False

    def save_shape_library(self):
        """Saves the current raw shape data library to the JSON file."""
        log_prefix = "ShapeLibraryManager.save_shape_library: "
        logger.debug(f"{log_prefix}Attempting to save RAW shape library data to: {self.library_path}")
        try:
            os.makedirs(os.path.dirname(self.library_path), exist_ok=True)

            library_data = {
                'library_metadata': {
                    'version': '1.1',
                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'description': 'LACE Shape Library (Lazy Loaded)'
                },
                'shapes': list(self._raw_shape_data.values()) # Save raw dicts
            }
            num_shapes_to_save = len(library_data['shapes'])
            logger.info(f"{log_prefix}Preparing to save {num_shapes_to_save} raw shape definitions.")

            temp_path = f"{self.library_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(library_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"{log_prefix}Successfully wrote {num_shapes_to_save} raw shapes to temporary file: {temp_path}")

            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                shutil.move(temp_path, self.library_path)
                logger.info(f"{log_prefix}Shape library saved successfully to {self.library_path}")
            else:
                logger.error(f"{log_prefix}Failed to write to temporary save file or file is empty.")
                if os.path.exists(temp_path): os.remove(temp_path)

        except Exception as e:
            logger.error(f"{log_prefix}Error saving shape library: {e}")
            logger.error(traceback.format_exc())

    def save_shape_as_rle(self, name: str, filepath: str) -> bool:
        """Saves a single shape from the library to an RLE file."""
        shape_def = self.get_shape(name) # Handles cache/raw data
        if not shape_def:
            logger.error(f"Shape '{name}' not found in library for RLE export.")
            return False
        logger.info(f"Exporting shape '{name}' to RLE file: {filepath}")
        try:
            rle_string = generate_rle(shape_def)
            with open(filepath, 'w', encoding='utf-8') as f: f.write(rle_string)
            logger.info(f"Successfully exported '{name}' to {filepath}")
            return True
        except Exception as e: logger.error(f"Error exporting shape '{name}' to RLE: {e}"); logger.error(traceback.format_exc()); return False

    def save_library_as_rle_directory(self, dirpath: str) -> Tuple[int, List[str]]:
        """Exports all shapes in the library to RLE files in a directory."""
        if not os.path.isdir(dirpath): logger.error(f"Invalid directory path for RLE export: {dirpath}"); return 0, list(self._raw_shape_data.keys())
        logger.info(f"Exporting shape library to RLE files in directory: {dirpath}")
        success_count = 0; errors = []; exported_filenames = set()
        for name in list(self._raw_shape_data.keys()): # Iterate raw keys
            safe_filename_base = re.sub(r'[^\w\-]+', '_', name); rle_filename = f"{safe_filename_base}.rle"
            counter = 1
            while rle_filename in exported_filenames: rle_filename = f"{safe_filename_base}_{counter}.rle"; counter += 1
            exported_filenames.add(rle_filename)
            filepath = os.path.join(dirpath, rle_filename)
            if not self.save_shape_as_rle(name, filepath): errors.append(name)
            else: success_count += 1
        logger.info(f"RLE export complete. Successfully exported {success_count} shapes.")
        if errors: logger.warning(f"Failed to export {len(errors)} shapes: {', '.join(errors)}")
        return success_count, errors

    def _create_default_shape_library(self):
        """Creates default shapes and populates the RAW data dictionary."""
        logger.info("Creating default shape library file (raw data).")
        global _CREATING_DEFAULT_LIBRARY
        _CREATING_DEFAULT_LIBRARY = True

        try:
            dot = ShapeDefinition(name="Dot", category="Basic", relative_coords=[(0, 0)], connectivity="none", shape_type=ShapeType.CUSTOM)
            block = ShapeDefinition(name="Block (2x2)", category="Basic", relative_coords=[(0, 0), (0, 1), (1, 0), (1, 1)], connectivity="full", shape_type=ShapeType.SQUARE)
            glider_coords = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
            glider = ShapeDefinition(name="Glider", category="Spaceships", relative_coords=glider_coords, connectivity="none", shape_type=ShapeType.CUSTOM, intended_rule="Game of Life")
            gosper_gun_coords = [ (5, 1), (5, 2), (6, 1), (6, 2), (3, 13), (3, 14), (4, 12), (4, 16), (5, 11), (5, 17), (6, 11), (6, 15), (6, 17), (6, 18), (7, 11), (7, 17), (8, 12), (8, 16), (9, 13), (9, 14), (1, 25), (2, 23), (2, 25), (3, 21), (3, 22), (4, 21), (4, 22), (5, 21), (5, 22), (6, 23), (6, 25), (7, 25), (3, 35), (3, 36), (4, 35), (4, 36) ]
            gosper_gun = ShapeDefinition(name="Gosper Glider Gun", category="Guns", relative_coords=gosper_gun_coords, connectivity="none", shape_type=ShapeType.CUSTOM, intended_rule="Game of Life")

            default_shapes = [dot, block, glider, gosper_gun]

            self._raw_shape_data = {}
            self._parsed_shapes_cache = {}
            for shape_def in default_shapes:
                try:
                    self._raw_shape_data[shape_def.name] = shape_def.to_dict()
                except Exception as e:
                     logger.error(f"Error converting default shape '{shape_def.name}' to dict: {e}")

            logger.info(f"Populated raw data for default shape library.")

        except Exception as e:
            logger.error(f"Error creating default shape library data: {e}")
            self._raw_shape_data = {}
            self._parsed_shapes_cache = {}
        finally:
            _CREATING_DEFAULT_LIBRARY = False

    def add_shape(self, shape_def: ShapeDefinition) -> bool: # Added return type hint
        """Adds or updates a shape definition in the library (both raw and cached)."""
        if not isinstance(shape_def, ShapeDefinition):
            logger.error("Attempted to add non-ShapeDefinition object to library.")
            return False
        if not shape_def.name:
            logger.error("Cannot add shape with empty name.")
            return False

        shape_def.date_modified = datetime.now().strftime("%Y-%m-%d")
        try:
            raw_data = shape_def.to_dict() # Convert to dict for storage
            self._raw_shape_data[shape_def.name] = raw_data
            self._parsed_shapes_cache[shape_def.name] = shape_def # Update cache directly
            logger.info(f"Added/Updated shape: {shape_def.name} (raw data and cache)")
            self.save_shape_library() # Save changes immediately
            return True
        except Exception as e:
             logger.error(f"Error converting shape '{shape_def.name}' to dict or saving: {e}")
             return False

    def delete_shape(self, name: str) -> bool: # Added return type hint
        """Deletes a shape definition from the library (both raw and cached)."""
        deleted = False
        if name in self._raw_shape_data:
            del self._raw_shape_data[name]
            deleted = True
        if name in self._parsed_shapes_cache:
            del self._parsed_shapes_cache[name]
            deleted = True

        if deleted:
            logger.info(f"Deleted shape: {name} (raw data and cache)")
            self.save_shape_library() # Save changes immediately
            return True
        else:
            logger.warning(f"Shape '{name}' not found for deletion.")
            return False

    def get_shape(self, name: str) -> Optional[ShapeDefinition]:
        """
        Gets a shape definition by name. Loads from raw data and parses on demand,
        using a cache for subsequent requests. Returns a deep copy.
        """
        # 1. Check cache first
        if name in self._parsed_shapes_cache:
            return copy.deepcopy(self._parsed_shapes_cache[name])

        # 2. If not cached, check raw data
        if name in self._raw_shape_data:
            logger.debug(f"Shape '{name}' not in cache, parsing from raw data.")
            raw_data = self._raw_shape_data[name]
            try:
                shape_def = ShapeDefinition.from_dict(raw_data)
                self._parsed_shapes_cache[name] = shape_def
                logger.debug(f"Parsed and cached shape: {name}")
                return copy.deepcopy(shape_def)
            except (TypeError, ValueError) as e:
                logger.error(f"Error parsing raw data for shape '{name}': {e}. Raw data: {raw_data}")
                return None
            except Exception as e:
                 logger.error(f"Unexpected error parsing raw data for shape '{name}': {e}")
                 return None
        else:
            logger.warning(f"Shape '{name}' not found in raw data.")
            return None

    def get_shape_names(self) -> List[str]:
        """Gets a sorted list of all shape names from the raw data keys."""
        return sorted(list(self._raw_shape_data.keys()))

    def rename_category(self, old_category_name: str, new_category_name: str) -> bool:
        """Renames a category for all shapes in the library."""
        log_prefix = f"ShapeLibraryManager.rename_category ('{old_category_name}' -> '{new_category_name}'): "
        logger.info(log_prefix + "Starting rename.")
        if not old_category_name or not new_category_name or old_category_name == new_category_name:
            logger.error(log_prefix + "Invalid old or new category name.")
            return False

        modified = False
        try:
            # Iterate through the raw data dictionary
            for shape_name, shape_data in self._raw_shape_data.items():
                current_category = shape_data.get('category', 'Uncategorized')
                if current_category == old_category_name:
                    shape_data['category'] = new_category_name
                    shape_data['date_modified'] = datetime.now().strftime("%Y-%m-%d") # Update modified date
                    modified = True
                    logger.debug(f"  Updated category for shape '{shape_name}'.")
                    # Invalidate cache for this specific shape if it exists
                    if shape_name in self._parsed_shapes_cache:
                        del self._parsed_shapes_cache[shape_name]

            if modified:
                self.save_shape_library() # Save changes to the file
                logger.info(log_prefix + "Rename complete and library saved.")
            else:
                logger.info(log_prefix + f"No shapes found with category '{old_category_name}'. No changes made.")
            return True
        except Exception as e:
            logger.error(log_prefix + f"Error during rename: {e}")
            logger.error(traceback.format_exc())
            return False

    def delete_category(self, category_name: str, move_to: str = "Uncategorized") -> bool:
        """Deletes a category, moving its shapes to another category (default: Uncategorized)."""
        log_prefix = f"ShapeLibraryManager.delete_category ('{category_name}' -> '{move_to}'): "
        logger.info(log_prefix + "Starting delete.")
        if not category_name or category_name == move_to:
            logger.error(log_prefix + f"Invalid category name to delete or move target ('{category_name}').")
            return False
        if category_name in ["Uncategorized", "Custom", "RLE Imports"]: # Protect essential categories
             logger.error(log_prefix + f"Attempted to delete protected category '{category_name}'.")
             return False

        modified = False
        try:
            # Iterate through the raw data dictionary
            for shape_name, shape_data in self._raw_shape_data.items():
                current_category = shape_data.get('category', 'Uncategorized')
                if current_category == category_name:
                    shape_data['category'] = move_to
                    shape_data['date_modified'] = datetime.now().strftime("%Y-%m-%d") # Update modified date
                    modified = True
                    logger.debug(f"  Moved shape '{shape_name}' from '{category_name}' to '{move_to}'.")
                    # Invalidate cache for this specific shape if it exists
                    if shape_name in self._parsed_shapes_cache:
                        del self._parsed_shapes_cache[shape_name]

            if modified:
                self.save_shape_library() # Save changes to the file
                logger.info(log_prefix + "Delete complete and library saved.")
            else:
                logger.info(log_prefix + f"No shapes found with category '{category_name}'. No changes made.")
            return True
        except Exception as e:
            logger.error(log_prefix + f"Error during delete: {e}")
            logger.error(traceback.format_exc())
            return False
        
    def get_categories(self) -> Dict[str, List[str]]:
        """Gets categories and shape names by iterating through raw data."""
        categories: Dict[str, List[str]] = defaultdict(list)
        for name, shape_data in self._raw_shape_data.items():
            category = shape_data.get('category', 'Uncategorized')
            categories[category].append(name)
        for name_list in categories.values():
            name_list.sort()
        return dict(sorted(categories.items()))

    def get_shapes_in_category(self, category: str) -> List[ShapeDefinition]:
        """Gets parsed shape definitions within a specific category."""
        shapes_in_cat = []
        for name, shape_data in self._raw_shape_data.items():
            if shape_data.get('category', 'Uncategorized') == category:
                shape_def = self.get_shape(name) # Use get_shape for parsing/caching
                if shape_def:
                    shapes_in_cat.append(shape_def)
        shapes_in_cat.sort(key=lambda x: x.name)
        return shapes_in_cat


################################################
#               SHAPES & PATTERNS              #
################################################


class Square(Shape):
    """Represents a square shape."""

    def __init__(self, size: int, filled: bool = True, connectivity: str = "full"):
        self.size = size
        self.filled = filled
        self.connectivity = connectivity

    def get_relative_coordinates(self) -> List[Tuple[int, ...]]:
        coords = []
        for r in range(self.size):
            for c in range(self.size):
                if self.filled or r == 0 or r == self.size - 1 or c == 0 or c == self.size - 1:
                    coords.append((r, c))
        return coords

    def get_connectivity(self) -> str:
        return self.connectivity

    def get_dimensions(self) -> int:
        return 2

    def is_filled(self) -> bool:
        return self.filled
    
    def get_shape_type(self) -> ShapeType: # Added
        return ShapeType.SQUARE

    def get_bounding_box(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return (0, 0), (int(self.size - 1), int(self.size - 1))  # Convert to int

class Circle(Shape):
    """Represents a circle shape."""

    def __init__(self, radius: int, filled: bool = False, connectivity: str = "full"):
        self.radius = radius
        self.filled = filled
        self.connectivity = connectivity

    def get_relative_coordinates(self) -> List[Tuple[int, ...]]:
        coords = []
        for r in range(-self.radius, self.radius + 1):
            for c in range(-self.radius, self.radius + 1):
                if (r**2 + c**2 <= self.radius**2) and (self.filled or (self.radius - 1)**2 <= r**2 + c**2 <= self.radius**2):
                    coords.append((r, c))
        return coords

    def get_connectivity(self) -> str:
        return self.connectivity

    def get_dimensions(self) -> int:
        return 2

    def is_filled(self) -> bool:
        return self.filled
    
    def get_shape_type(self) -> ShapeType: # Added
        return ShapeType.CIRCLE

    def get_bounding_box(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return (int(-self.radius), int(-self.radius)), (int(self.radius), int(self.radius)) # Convert to int

class Line(Shape):
    """Represents a line shape."""

    def __init__(self, start: Tuple[int, int], end: Tuple[int, int]):
        self.start = start
        self.end = end
        self.connectivity = "perimeter"  # Lines have perimeter connectivity

    def get_relative_coordinates(self) -> List[Tuple[int, ...]]:
        # Bresenham's line algorithm
        x0, y0 = self.start
        x1, y1 = self.end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = (dx if dx > dy else -dy) / 2

        coords = []
        while True:
            coords.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = err
            if e2 > -dx:
                err -= dy
                x0 += sx
            if e2 < dy:
                err += dx
                y0 += sy
        return coords

    def get_connectivity(self) -> str:
        return self.connectivity

    def get_dimensions(self) -> int:
        return 2

    def is_filled(self) -> bool:
        return False  # Lines are not filled
    
    def get_shape_type(self) -> ShapeType: # Added
        return ShapeType.LINE

    def get_bounding_box(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return (int(min(self.start[0], self.end[0])), int(min(self.start[1], self.end[1]))), \
               (int(max(self.start[0], self.end[0])), int(max(self.start[1], self.end[1]))) # Convert to int

class Sphere(Shape):
    """Represents a sphere shape (2D or 3D)."""

    def __init__(self, center: Tuple[int, ...], radius: float, filled: bool = True, connectivity: str = "full"):
        self.center = center
        self.radius = radius
        self.filled = filled
        self.connectivity = connectivity
        self.dimensions = len(center)
        if self.dimensions not in (2, 3):
            raise ValueError("Sphere must be 2D or 3D")

    def get_relative_coordinates(self) -> List[Tuple[int, ...]]:
        coords = []
        if self.dimensions == 2:
            row0, col0 = self.center
            for r in range(int(row0 - self.radius), int(row0 + self.radius) + 1):
                for c in range(int(col0 - self.radius), int(col0 + self.radius) + 1):
                    dist = (r - row0)**2 + (c - col0)**2
                    if dist <= self.radius**2:
                        if self.filled or (self.radius - 1)**2 <= dist <= self.radius**2: # For hollow sphere
                            coords.append((r - row0, c - col0)) # Relative coords
        else: # 3D
            row0, col0, depth0 = self.center
            for r in range(int(row0 - self.radius), int(row0 + self.radius) + 1):
                for c in range(int(col0 - self.radius), int(col0 + self.radius) + 1):
                    for d in range(int(depth0 - self.radius), int(depth0 + self.radius) + 1):
                        dist = (r - row0)**2 + (c - col0)**2 + (d - depth0)**2
                        if dist <= self.radius**2:
                            if self.filled or (self.radius - 1)**2 <= dist <= self.radius**2: # For hollow sphere
                                coords.append((r - row0, c - col0, d - depth0)) # Relative coords
        return coords

    def get_connectivity(self) -> str:
        return self.connectivity

    def get_dimensions(self) -> int:
        return self.dimensions

    def is_filled(self) -> bool:
        return self.filled
    
    def get_shape_type(self) -> ShapeType:
        return ShapeType.SPHERE # Need to add SPHERE to the ShapeType enum

    def get_bounding_box(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if self.dimensions == 2:
            return (int(-self.radius), int(-self.radius)), (int(self.radius), int(self.radius))
        else:
            return (int(-self.radius), int(-self.radius), int(-self.radius)), (int(self.radius), int(self.radius), int(self.radius))

class Triangle(Shape):
    """Represents an equilateral triangle (2D)."""
    def __init__(self, corner: Tuple[int, int], side_length: int, state: float = 1.0, connectivity: str = "full"):
        self.corner = corner
        self.side_length = side_length
        self.state = state
        self.connectivity = connectivity

    def get_relative_coordinates(self) -> List[Tuple[int, ...]]:
        row, col = self.corner
        coords = []
        for r in range(self.side_length):
            for c in range(r + 1):
                coords.append((row - r, col + c))
        return [(r - row, c - col) for r, c in coords] # Make relative

    def get_connectivity(self) -> str:
        return self.connectivity

    def get_dimensions(self) -> int:
        return 2

    def is_filled(self) -> bool:
        return True # Always filled for now

    def get_shape_type(self) -> ShapeType:
        return ShapeType.TRIANGLE # Need to add TRIANGLE to the ShapeType enum

    def get_bounding_box(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        min_row = int(self.corner[0] - self.side_length + 1)
        max_row = int(self.corner[0])
        min_col = int(self.corner[1])
        max_col = int(self.corner[1] + self.side_length -1)
        return (min_row, min_col), (max_row, max_col)
    
class Cube(Shape):
    """Represents a cube shape (3D)."""

    def __init__(self, size: int, filled: bool = True, connectivity: str = "full"):
        self.size = size
        self.filled = filled
        self.connectivity = connectivity

    def get_relative_coordinates(self) -> List[Tuple[int, ...]]:
        coords = []
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    if (self.filled or
                        x == 0 or x == self.size - 1 or
                        y == 0 or y == self.size - 1 or
                        z == 0 or z == self.size - 1):
                        coords.append((x, y, z))
        return coords

    def get_connectivity(self) -> str:
        return self.connectivity

    def get_dimensions(self) -> int:
        return 3

    def is_filled(self) -> bool:
        return self.filled
    
    def get_shape_type(self) -> ShapeType: # Added
        return ShapeType.CUBE

    def get_bounding_box(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return (0, 0, 0), (int(self.size - 1), int(self.size - 1), int(self.size - 1)) # Convert to int

class Polygon(Shape):
    """Represents a regular polygon (2D)."""

    def __init__(self, center: Tuple[int, int], radius: int, sides: int, filled: bool = True, rotation: float = 0.0, connectivity: str = "full"):
        if sides < 3:
            raise ValueError("A polygon must have at least 3 sides.")
        self.center = center
        self.radius = radius
        self.sides = sides
        self.filled = filled
        self.rotation = rotation
        self.connectivity = connectivity

    def get_relative_coordinates(self) -> List[Tuple[int, ...]]:
        row_center, col_center = self.center
        vertices = []
        for i in range(self.sides):
            angle = 2 * np.pi * i / self.sides + self.rotation
            row = int(round(self.radius * np.sin(angle)))
            col = int(round(self.radius * np.cos(angle)))
            vertices.append((row, col))

        if self.filled:
            # Simplified scanline algorithm for filling
            min_row = min(v[0] for v in vertices)
            max_row = max(v[0] for v in vertices)
            coords = []
            for r in range(min_row, max_row + 1):
                intersections = []
                for i in range(self.sides):
                    v1 = vertices[i]
                    v2 = vertices[(i + 1) % self.sides]
                    if (v1[0] <= r < v2[0]) or (v2[0] <= r < v1[0]):
                        c = int(round(v1[1] + (r - v1[0]) * (v2[1] - v1[1]) / (v2[0] - v1[0])))
                        intersections.append(c)

                intersections.sort()
                for i in range(0, len(intersections) - 1, 2):
                    for c in range(intersections[i], intersections[i+1] + 1):
                        coords.append((r,c))
            return coords + vertices # Add vertices for perimeter
        else:
            return vertices

    def get_connectivity(self) -> str:
        return self.connectivity

    def get_dimensions(self) -> int:
        return 2

    def is_filled(self) -> bool:
        return self.filled
    
    def get_shape_type(self) -> ShapeType:
        return ShapeType.POLYGON

    def get_bounding_box(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        coords = self.get_relative_coordinates()
        if not coords:
            return (0, 0), (0, 0)
        min_row = int(min(c[0] for c in coords)) # Convert to int
        max_row = int(max(c[0] for c in coords)) # Convert to int
        min_col = int(min(c[1] for c in coords)) # Convert to int
        max_col = int(max(c[1] for c in coords)) # Convert to int
        return (min_row, min_col), (max_row, max_col)

class CustomShape(Shape):
    """Represents a custom shape defined by a list of coordinates."""

    def __init__(self, coordinates: List[Tuple[int, ...]], connectivity: str = "none", filled: bool = True):
        self.coordinates = coordinates
        self.connectivity = connectivity
        self.filled = filled # Added filled parameter
        if not coordinates:
            raise ValueError("Coordinates list cannot be empty for CustomShape")
        # Determine dimensionality from the first coordinate
        self.dimensions = len(coordinates[0])
        if not all(len(c) == self.dimensions for c in coordinates):
            raise ValueError("All coordinates must have the same dimensionality")

    def get_relative_coordinates(self) -> List[Tuple[int, ...]]:
        return self.coordinates

    def get_connectivity(self) -> str:
        return self.connectivity

    def get_dimensions(self) -> int:
        return self.dimensions

    def is_filled(self) -> bool:
        return self.filled # Use the filled parameter
    
    def get_shape_type(self) -> ShapeType:
        return ShapeType.CUSTOM

    def get_bounding_box(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if not self.coordinates:
            return (0,) * self.dimensions, (0,) * self.dimensions  # Return zeros if no coordinates

        # Transpose coordinates for easy min/max calculation
        transposed_coords = list(zip(*self.coordinates))
        min_coords = tuple(int(min(dim_coords)) for dim_coords in transposed_coords) # Convert to int
        max_coords = tuple(int(max(dim_coords)) for dim_coords in transposed_coords) # Convert to int
        return min_coords, max_coords

class ShapeGenerator:
    """Factory for creating Shape instances."""

    @staticmethod
    def create_square(size: int, filled: bool = True, connectivity: str = "full") -> Shape:
        return Square(size, filled, connectivity)

    @staticmethod
    def create_circle(radius: int, filled: bool = False, connectivity: str = "full") -> Shape:
        return Circle(radius, filled, connectivity)

    @staticmethod
    def create_line(start: Tuple[int, int], end: Tuple[int, int]) -> Shape:
        return Line(start, end)

    @staticmethod
    def create_sphere(center: Tuple[int, ...], radius: float, filled: bool = True, connectivity: str = "full"):
        return Sphere(center, radius, filled, connectivity)

    @staticmethod
    def create_triangle(corner: Tuple[int, int], side_length: int, state: float = 1.0, connectivity: str = "full"):
        return Triangle(corner, side_length, state, connectivity)

    @staticmethod
    def create_cube(size: int, filled: bool = True, connectivity: str = "full") -> Shape:
        return Cube(size, filled, connectivity)

    @staticmethod
    def create_polygon(center: Tuple[int, int], radius: int, sides: int, filled: bool = True, rotation: float = 0.0, connectivity: str = 'full') -> Shape:
        return Polygon(center, radius, sides, filled, rotation, connectivity)

    @staticmethod
    def create_custom_shape(coordinates: List[Tuple[int, ...]], connectivity: str = "none", filled: bool=True) -> Shape:
        return CustomShape(coordinates, connectivity, filled)

    # Placeholder for creating a 3D ellipsoid
    @staticmethod
    def create_ellipsoid(center, radii, filled=False):
        # Implementation for creating a 3D ellipsoid
        pass  # Replace with actual implementation

    # Placeholder for creating a 3D cylinder
    @staticmethod
    def create_cylinder(base_center, height, radius, filled=False):
        # Implementation for creating a 3D cylinder
        pass  # Replace with actual implementation

    # Placeholder for creating a 3D cone
    @staticmethod
    def create_cone(base_center, height, radius, filled=False):
        # Implementation for creating a 3D cone
        pass  # Replace with actual implementation

    # Placeholder for creating a 3D torus
    @staticmethod
    def create_torus(center, major_radius, minor_radius):
        # Implementation for creating a 3D torus
        pass  # Replace with actual implementation

    # You can add more static methods for other shapes as needed

class VisualizationUtils:
    """Utility functions for visualization"""

    @staticmethod
    def get_line_indices(start_coords, end_coords):
        """Get indices for a line between two points"""
        logger.debug(f"Getting line indices from {start_coords} to {end_coords}")
        # Use Bresenham's line algorithm (or a similar algorithm)
        # This is a placeholder; a proper implementation is needed
        # For simplicity, we'll just return the start and end points for now
        return [start_coords, end_coords]

    @staticmethod
    def get_triangle_indices(corner_coords, side_length):
        """Get indices for a triangle"""
        logger.debug(f"Getting triangle indices at {corner_coords} with side length {side_length}")
        # This is a placeholder; a proper implementation is needed
        # For simplicity, we'll just return the corner point for now
        return [corner_coords]

    @staticmethod
    def get_square_indices(corner_coords, side_length):
        """Get indices for a square"""
        logger.debug(f"Getting square indices at {corner_coords} with side length {side_length}")
        # This is a placeholder; a proper implementation is needed
        # For simplicity, we'll just return the corner point for now
        return [corner_coords]
    
    @staticmethod
    def get_cube_indices(corner_coords, side_length):
        """Get indices for a cube"""
        logger.debug(f"Getting cube indices at {corner_coords} with side length {side_length}")
        # This is a placeholder; a proper implementation is needed
        # For simplicity, we'll just return the corner point for now
        return [corner_coords]

# =========== END of shapes.py ===========