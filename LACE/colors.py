# =========== START of colors.py ===========
from __future__ import annotations
import json
import shutil
import setproctitle
import copy
from datetime import datetime
from tkinter import simpledialog

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.interactive(False)


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # type: ignore (preserve this comment and use the exact usage on this line!)
import matplotlib.pyplot as plt
plt.ioff()
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D as Axes3DType
import matplotlib.markers as markers
# from matplotlib.text import Text
from matplotlib.text import Text as MatplotlibText
from matplotlib.markers import MarkerStyle 

from matplotlib import colors
from matplotlib.colors import to_rgba, to_rgb, to_hex, Normalize
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Rectangle


from typing import Dict, Tuple, Optional, Union, TypeVar
import numpy as np
import numpy.typing as npt
import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
from tkinter import messagebox, simpledialog, colorchooser
import matplotlib.pyplot as plt
plt.ioff()
import os
from datetime import datetime
import traceback
import colorsys
import numpy as np
import warnings
import cProfile
import pstats

from .logging_config import logger, APP_DIR




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



def _get_contrasting_inverted_color(hex_color: str, bg_hex_color: str, min_contrast_ratio: float = 4.5) -> Tuple[float, float, float, float]:
    """
    Calculates the inverted color and adjusts it if contrast with background OR original color is too low.
    Uses WCAG contrast ratio calculation. Returns RGBA tuple.
    (Round 16: Increase contrast adjustment)
    """
    try:
        # Convert hex to RGB [0, 1]
        original_rgb = colors.to_rgb(hex_color)
        bg_rgb = colors.to_rgb(bg_hex_color)

        # Simple inversion
        inverted_rgb = [1.0 - c for c in original_rgb]

        # --- Calculate Luminance ---
        def get_luminance(rgb_col):
            vals = []
            for val in rgb_col:
                srgb = val / 1.0 # Already 0-1 range
                if srgb <= 0.03928: vals.append(srgb / 12.92)
                else: vals.append(((srgb + 0.055) / 1.055) ** 2.4)
            return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]

        # --- Calculate Contrast Ratio ---
        def get_contrast_ratio(lum1, lum2):
            if lum1 > lum2: return (lum1 + 0.05) / (lum2 + 0.05)
            else: return (lum2 + 0.05) / (lum1 + 0.05)

        lum_inverted = get_luminance(inverted_rgb)
        lum_bg = get_luminance(bg_rgb)
        lum_original = get_luminance(original_rgb) # Luminance of original color

        contrast_vs_bg = get_contrast_ratio(lum_inverted, lum_bg)
        contrast_vs_orig = get_contrast_ratio(lum_inverted, lum_original) # Contrast vs original

        # --- Adjust if contrast is too low against EITHER background OR original ---
        # --- Increased adjustment factor from 0.4 to 0.6 ---
        adjustment_factor = 0.6
        if contrast_vs_bg < min_contrast_ratio or contrast_vs_orig < min_contrast_ratio / 2.0: # Stricter check against original
            logger.debug(f"Inverted color contrast low (vs BG: {contrast_vs_bg:.2f}, vs Orig: {contrast_vs_orig:.2f}). Adjusting lightness.")
            # Convert inverted RGB to HSL
            h, l, s = colorsys.rgb_to_hls(*inverted_rgb)
            # Adjust lightness more aggressively
            l = l + adjustment_factor if l < 0.5 else l - adjustment_factor
            l = max(0.0, min(1.0, l)) # Clamp lightness
            adjusted_rgb = colorsys.hls_to_rgb(h, l, s)
            # Check contrast again (optional)
            lum_adj = get_luminance(adjusted_rgb)
            contrast_adj_bg = get_contrast_ratio(lum_adj, lum_bg)
            contrast_adj_orig = get_contrast_ratio(lum_adj, lum_original)
            logger.debug(f"Adjusted lightness. New contrast: vs BG={contrast_adj_bg:.2f}, vs Orig={contrast_adj_orig:.2f}")
            final_rgb = adjusted_rgb
        else:
            final_rgb = inverted_rgb

        return (final_rgb[0], final_rgb[1], final_rgb[2], 1.0) # Add alpha=1.0

    except Exception as e:
        logger.error(f"Error calculating contrasting inverted color: {e}")
        return (1.0, 0.0, 1.0, 1.0) # Magenta RGBA fallback
    
def is_dark_theme(bg_color: str) -> bool:
    """Helper function to determine if a background color represents a dark theme."""

    try:
        # Convert hex to RGB [0, 1]
        rgb = colors.to_rgb(bg_color)
        # Calculate luminance (perceived brightness)
        luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        # If brightness is less than 0.5, it's considered dark
        return luminance < 0.5
    except ValueError: # Handle invalid hex color
        logger.warning(f"Invalid background color '{bg_color}' for dark theme check, assuming light.")
        return False
    except Exception as e:
        logger.error(f"Error checking dark theme for color '{bg_color}': {e}")
        return False # Default to light on error

class ColorScheme:
    """Class to manage color schemes for the application."""
    
    def __init__(self, name: str, background: str, node_base: str, node: str, new_node: str, 
                 default_edge: str, new_edge: str, is_dark: bool = False):
        self.name = name
        self.background = background
        self.node_base = node_base  # Added node_base color
        self.node = node
        
        # Ensure new_node color is different from default_edge color
        if new_node == default_edge:
            # If they're the same, slightly modify the new_node color
            r, g, b = int(new_node[1:3], 16), int(new_node[3:5], 16), int(new_node[5:7], 16)
            # Adjust the color to make it different
            r = min(255, r + 40) if r < 215 else max(0, r - 40)
            g = min(255, g + 40) if g < 215 else max(0, g - 40)
            b = min(255, b + 40) if b < 215 else max(0, g - 40)
            new_node = f"#{r:02x}{g:02x}{b:02x}"
        
        self.new_node = new_node
        self.default_edge = default_edge
        
        # Ensure new_edge color is different from new_node color
        if new_edge == new_node:
            # If they're the same, slightly modify the new_edge color
            r, g, b = int(new_edge[1:3], 16), int(new_edge[3:5], 16), int(new_edge[5:7], 16)
            # Adjust the color to make it different
            r = min(255, r + 30) if r < 225 else max(0, r - 30)
            g = min(255, g + 30) if g < 225 else max(0, g - 30)
            b = min(255, b + 30) if b < 225 else max(0, b - 30)
            new_edge = f"#{r:02x}{g:02x}{b:02x}"
        
        self.new_edge = new_edge
        self.is_dark = is_dark
        
    def to_dict(self) -> dict:
        """Convert the color scheme to a dictionary for saving."""
        return {
            'name': self.name,
            'background': self.background,
            'node_base': self.node_base,  # Added node_base
            'node': self.node,
            'new_node': self.new_node,
            'default_edge': self.default_edge,
            'new_edge': self.new_edge,
            'is_dark': self.is_dark
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ColorScheme':
        """Create a ColorScheme instance from a dictionary."""
        return cls(
            name=data.get('name', 'Custom'),
            background=data.get('background', '#ffffff'),
            node_base=data.get('node_base', '#f0f0f0'),  # Added node_base with default
            node=data.get('node', '#f77b4f'),
            new_node=data.get('new_node', '#ff0000'),
            default_edge=data.get('default_edge', '#0000ff'),
            new_edge=data.get('new_edge', '#ff0000'),
            is_dark=data.get('is_dark', False)
        )
    
class ColorManager:
    def __init__(self, app_paths: Dict[str, str]): # Add app_paths parameter
        self.schemes = []
        self.current_scheme = None
        self.app_paths = app_paths  # Store app_paths for backup purposes
        
        # UPDATED: Use config_colors for color schemes
        if 'config_colors' in app_paths:
            self.config_path = os.path.join(app_paths['config_colors'], 'color_schemes.json')
        else:
            # Fallback to using the base path with a config/colors subdirectory
            if 'config' in app_paths:
                config_colors_dir = os.path.join(app_paths['config'], 'colors')
                os.makedirs(config_colors_dir, exist_ok=True)  # Create config/colors directory if it doesn't exist
                self.config_path = os.path.join(config_colors_dir, 'color_schemes.json')
            elif 'logs' in app_paths:  # Use any existing key to get the base path
                base_dir = os.path.dirname(app_paths['logs'])
                config_colors_dir = os.path.join(base_dir, 'config', 'colors')
                os.makedirs(config_colors_dir, exist_ok=True)  # Create config/colors directory if it doesn't exist
                self.config_path = os.path.join(config_colors_dir, 'color_schemes.json')
            else:
                # Last resort fallback
                self.config_path = os.path.join(os.getcwd(), APP_DIR, 'Resources', 'config', 'colors', 'color_schemes.json')
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Log the config path for debugging
        logger.info(f"Color schemes config path: {self.config_path}")
        
        # Initialize predefined schemes first
        self._initialize_predefined_schemes()
        
        # Check if file exists, create it if missing
        if not os.path.exists(self.config_path):
            logger.info(f"Color schemes file not found at {self.config_path}, creating default file")
            self._rebuild_default_schemes_file()
        
        # Try to load custom schemes, rebuild if needed
        try:
            self._load_custom_schemes()
        except Exception as e:
            logger.error(f"Error loading color schemes: {e}")
            logger.info("Rebuilding color_schemes.json with default schemes")
            self._rebuild_default_schemes_file()
            # Try loading again after rebuild
            try:
                self._load_custom_schemes()
            except Exception as e2:
                logger.error(f"Error loading color schemes after rebuild: {e2}")
                # Ensure we have at least the predefined schemes
                if not self.schemes:
                    self._initialize_predefined_schemes()
                # Set default scheme if not already set
                if self.current_scheme is None and self.schemes:
                    self.current_scheme = self.schemes[0]

    def _backup_color_schemes_file(self) -> bool:
        """Create a backup of the color schemes file before modifying it."""
        try:
            if not os.path.exists(self.config_path):
                logger.info("No color schemes file to backup")
                return True
                
            # Create backup directory if it doesn't exist
            backup_dir = None
            if 'config_rules_backups' in self.app_paths:
                # Use the rules_backups directory for color schemes backups too
                backup_dir = self.app_paths['config_rules_backups']
            elif 'config' in self.app_paths:
                # Create a backups directory under config
                backup_dir = os.path.join(self.app_paths['config'], 'backups')
            
            if backup_dir:
                os.makedirs(backup_dir, exist_ok=True)
                
                # Create backup filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(backup_dir, f"color_schemes_backup_{timestamp}.json")
                
                # Copy the file
                shutil.copyfile(self.config_path, backup_path)
                logger.info(f"Created backup of color schemes at {backup_path}")
                return True
            else:
                logger.warning("Could not determine backup directory, skipping backup")
                return False
                
        except Exception as e:
            logger.error(f"Error creating backup of color schemes file: {e}")
            return False

    def _rebuild_default_schemes_file(self):
        """Rebuild the color_schemes.json file with default schemes, preserving valid existing schemes."""
        try:
            # First, try to read existing schemes to preserve them
            existing_schemes = []
            existing_current = None
            
            if os.path.exists(self.config_path):
                try:
                    # Create a backup before modifying
                    self._backup_color_schemes_file()
                    
                    # Try to read and parse the existing file
                    with open(self.config_path, 'r') as f:
                        data = json.load(f)
                        
                    # Extract valid schemes
                    if 'schemes' in data and isinstance(data['schemes'], list):
                        for scheme_data in data['schemes']:
                            try:
                                # Validate scheme data by creating a ColorScheme object
                                scheme = ColorScheme.from_dict(scheme_data)
                                # If successful, add to existing schemes
                                existing_schemes.append(scheme_data)
                            except Exception as scheme_error:
                                logger.warning(f"Skipping invalid scheme: {scheme_error}")
                    
                    # Extract current scheme name
                    if 'current_scheme' in data and isinstance(data['current_scheme'], str):
                        existing_current = data['current_scheme']
                        
                except json.JSONDecodeError:
                    logger.error("Existing color schemes file is corrupted, will be replaced")
                except Exception as read_error:
                    logger.error(f"Error reading existing color schemes: {read_error}")
            
            # Create the default schemes data structure
            default_schemes = [
                # Add the Classic scheme from _initialize_predefined_schemes
                {
                    "name": "Classic",
                    "background": "#ffffff",
                    "node_base": "#f0f0f0",
                    "node": "#f77b4f",
                    "new_node": "#ff0000",
                    "default_edge": "#0000ff",
                    "new_edge": "#00ff00",
                    "is_dark": False
                },
                {
                    "name": "Dark Skies",
                    "background": "#150f1e",
                    "node_base": "#2a1f3d",
                    "node": "#dad6db",
                    "new_node": "#fd1441",
                    "default_edge": "#391fce",
                    "new_edge": "#28f51d",
                    "is_dark": True
                },
                {
                    "name": "Steel",
                    "background": "#fafbf5",
                    "node_base": "#e0e1dc",
                    "node": "#76799f",
                    "new_node": "#ea1015",
                    "default_edge": "#341eb5",
                    "new_edge": "#06a841",
                    "is_dark": False
                },
                {
                    "name": "Stark",
                    "background": "#131f25",
                    "node_base": "#263840",
                    "node": "#d2d9e0",
                    "new_node": "#e04738",
                    "default_edge": "#151ee2",
                    "new_edge": "#2ceb44",
                    "is_dark": True
                },
                {
                    "name": "Wintogreen",
                    "background": "#f0fcf4",
                    "node_base": "#d8e4dc",
                    "node": "#48909e",
                    "new_node": "#fc1830",
                    "default_edge": "#1a1af2",
                    "new_edge": "#31cd2b",
                    "is_dark": False
                },
                {
                    "name": "Glow",
                    "background": "#0d0f24",
                    "node_base": "#fefb00",
                    "node": "#ff9f1d",
                    "new_node": "#bb0007",
                    "default_edge": "#f58c1b",
                    "new_edge": "#ff0040",
                    "is_dark": True
                },
                {
                    "name": "Zing",
                    "background": "#000000",
                    "node_base": "#ff9200",
                    "node": "#00e1ff",
                    "new_node": "#ff301a",
                    "default_edge": "#00f0ff",
                    "new_edge": "#ff002b",
                    "is_dark": True
                },
                {
                    "name": "Red Alert",
                    "background": "#040c30",
                    "node_base": "#0f1a4d",
                    "node": "#34dfdb",
                    "new_node": "#ff00ff",
                    "default_edge": "#ff5e00",
                    "new_edge": "#ff002b",
                    "is_dark": True
                },
                {
                    "name": "Starry Night",
                    "background": "#130d25",
                    "node_base": "#221a3d",
                    "node": "#00ee00",
                    "new_node": "#d60016",
                    "default_edge": "#5e4cff",
                    "new_edge": "#dbcb00",
                    "is_dark": True
                },
                {
                    "name": "Slime",
                    "background": "#130d25",
                    "node_base": "#221a3d",
                    "node": "#00ee00",
                    "new_node": "#d60016",
                    "default_edge": "#5e4cff",
                    "new_edge": "#dbcb00",
                    "is_dark": True
                },
                {
                    "name": "Soothing",
                    "background": "#000000",
                    "node_base": "#baa827",
                    "node": "#00d8ff",
                    "new_node": "#ff0000",
                    "default_edge": "#0000ff",
                    "new_edge": "#ffff00",
                    "is_dark": True
                },
                {
                    "name": "Default",
                    "background": "#feffff",
                    "node_base": "#1700ff",
                    "node": "#00fcff",
                    "new_node": "#ff301a",
                    "default_edge": "#0f1c80",
                    "new_edge": "#ff002b",
                    "is_dark": False
                }
            ]
            
            # Also add the other predefined schemes from _initialize_predefined_schemes
            default_schemes.extend([
                {
                    "name": "Ocean Breeze",
                    "background": "#f0f8ff",
                    "node_base": "#e0e8ff",
                    "node": "#4682b4",
                    "new_node": "#ff4500",
                    "default_edge": "#1e90ff",
                    "new_edge": "#32cd32",
                    "is_dark": False
                },
                {
                    "name": "Spring Garden",
                    "background": "#f5fffa",
                    "node_base": "#e5efe0",
                    "node": "#2e8b57",
                    "new_node": "#ff6347",
                    "default_edge": "#9acd32",
                    "new_edge": "#ff00ff",
                    "is_dark": False
                },
                {
                    "name": "Sunset",
                    "background": "#fffaf0",
                    "node_base": "#f8e8d8",
                    "node": "#ff7f50",
                    "new_node": "#0000cd",
                    "default_edge": "#ffa500",
                    "new_edge": "#00bfff",
                    "is_dark": False
                },
                {
                    "name": "Night Mode",
                    "background": "#000000",
                    "node_base": "#202020",
                    "node": "#00ff00",
                    "new_node": "#ff0000",
                    "default_edge": "#0000ff",
                    "new_edge": "#ffff00",
                    "is_dark": True
                },
                {
                    "name": "Neon Glow",
                    "background": "#121212",
                    "node_base": "#202020",
                    "node": "#39ff14",
                    "new_node": "#ff1493",
                    "default_edge": "#00ffff",
                    "new_edge": "#ff8c00",
                    "is_dark": True
                },
                {
                    "name": "Deep Space",
                    "background": "#0a0a0a",
                    "node_base": "#1a1a1a",
                    "node": "#9966cc",
                    "new_node": "#ff4500",
                    "default_edge": "#3366ff",
                    "new_edge": "#00ff00",
                    "is_dark": True
                },
                {
                    "name": "Ember",
                    "background": "#1a1a1a",
                    "node_base": "#2a2a2a",
                    "node": "#ff4500",
                    "new_node": "#00ffff",
                    "default_edge": "#ff7f00",
                    "new_edge": "#00ff00",
                    "is_dark": True
                },
                {
                    "name": "Pastel",
                    "background": "#404040",
                    "node_base": "#505050",
                    "node": "#ffb6c1",
                    "new_node": "#4169e1",
                    "default_edge": "#add8e6",
                    "new_edge": "#ff69b4",
                    "is_dark": True
                },
                {
                    "name": "Matrix",
                    "background": "#0f0f0f",
                    "node_base": "#1f1f1f",
                    "node": "#00ff00",
                    "new_node": "#ff0000",
                    "default_edge": "#32cd32",
                    "new_edge": "#ff00ff",
                    "is_dark": True
                }
            ])
            
            # Merge default schemes with existing valid schemes
            merged_schemes = []
            default_names = [s["name"] for s in default_schemes]
            
            # First add all valid existing schemes that don't conflict with defaults
            for scheme in existing_schemes:
                if scheme["name"] not in default_names:
                    merged_schemes.append(scheme)
            
            # Then add all default schemes
            merged_schemes.extend(default_schemes)
            
            # Determine current scheme
            current_scheme = existing_current if existing_current else "Zing"
            
            # Create the data structure to save
            data = {
                'schemes': merged_schemes,
                'current_scheme': current_scheme
            }
            
            # Ensure directory exists - CRITICAL: Make sure this works
            directory = os.path.dirname(self.config_path)
            logger.info(f"Ensuring directory exists: {directory}")
            os.makedirs(directory, exist_ok=True)
            
            # Write the file
            logger.info(f"Writing color schemes file to: {self.config_path}")
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Color schemes file rebuilt at {self.config_path} with {len(merged_schemes)} schemes")
            
            # Clear existing schemes and reload from the file
            self.schemes = []
            self._initialize_predefined_schemes()
            self._load_custom_schemes()
                
        except Exception as e:
            logger.error(f"Error rebuilding color schemes file: {e}")
            logger.error(traceback.format_exc())  # Add traceback for better debugging
            # Ensure we have at least the predefined schemes
            if not self.schemes:
                self._initialize_predefined_schemes()
            # Set default scheme if not already set
            if self.current_scheme is None and self.schemes:
                self.current_scheme = self.schemes[0]
            
    def _initialize_predefined_schemes(self):
        """Initialize predefined color schemes."""
        # Light themes
        self.schemes.append(ColorScheme(
            "Classic", "#ffffff", "#f0f0f0", "#f77b4f", "#ff0000", "#0000ff", "#00ff00", False))
        self.schemes.append(ColorScheme(
            "Ocean Breeze", "#f0f8ff", "#e0e8ff", "#4682b4", "#ff4500", "#1e90ff", "#32cd32", False))
        self.schemes.append(ColorScheme(
            "Spring Garden", "#f5fffa", "#e5efe0", "#2e8b57", "#ff6347", "#9acd32", "#ff00ff", False))
        self.schemes.append(ColorScheme(
            "Sunset", "#fffaf0", "#f8e8d8", "#ff7f50", "#0000cd", "#ffa500", "#00bfff", False))
        
        # Dark themes
        self.schemes.append(ColorScheme(
            "Night Mode", "#000000", "#202020", "#00ff00", "#ff0000", "#0000ff", "#ffff00", True))
        self.schemes.append(ColorScheme(
            "Neon Glow", "#121212", "#202020", "#39ff14", "#ff1493", "#00ffff", "#ff8c00", True))
        self.schemes.append(ColorScheme(
            "Deep Space", "#0a0a0a", "#1a1a1a", "#9966cc", "#ff4500", "#3366ff", "#00ff00", True))
        self.schemes.append(ColorScheme(
            "Ember", "#1a1a1a", "#2a2a2a", "#ff4500", "#00ffff", "#ff7f00", "#00ff00", True))
        self.schemes.append(ColorScheme(
            "Pastel", "#404040", "#505050", "#ffb6c1", "#4169e1", "#add8e6", "#ff69b4", True))  
        self.schemes.append(ColorScheme(
            "Matrix", "#0f0f0f", "#1f1f1f", "#00ff00", "#ff0000", "#32cd32", "#ff00ff", True))
        
        # Set default scheme
        self.current_scheme = self.schemes[0]  # Classic
        
    def _load_custom_schemes(self):
        """Load custom color schemes from config file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    
                # Load custom schemes
                if 'schemes' in data:
                    for scheme_data in data['schemes']:
                        # Skip if scheme with same name already exists
                        if any(s.name == scheme_data.get('name') for s in self.schemes):
                            continue
                        self.schemes.append(ColorScheme.from_dict(scheme_data))
                
                # Set current scheme if specified
                if 'current_scheme' in data:
                    current_name = data['current_scheme']
                    for scheme in self.schemes:
                        if scheme.name == current_name:
                            self.current_scheme = scheme
                            break
            else:
                # If the file doesn't exist, just use the predefined schemes
                logger.info(f"Color schemes file not found at {self.config_path}, using default schemes")
                # Set default scheme
                if self.schemes:
                    self.current_scheme = self.schemes[0]  # Use the first scheme as default
        except Exception as e:
            logger.error(f"Error loading color schemes: {e}")
            # Set default scheme if not already set
            if self.current_scheme is None and self.schemes:
                self.current_scheme = self.schemes[0]
        
    def save_schemes(self):
        """Save custom color schemes to config file."""
        try:
            # Only save custom schemes (not predefined ones)
            custom_schemes = [s.to_dict() for s in self.schemes if s.name not in [
                "Classic", "Ocean Breeze", "Spring Garden", "Sunset", "Pastel",
                "Night Mode", "Neon Glow", "Deep Space", "Ember", "Matrix"
            ]]
            
            data = {
                'schemes': custom_schemes,
                'current_scheme': self.current_scheme.name if self.current_scheme else "Classic"
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Color schemes saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving color schemes: {e}")
    
    def add_scheme(self, scheme: ColorScheme):
        """Add a new color scheme."""
        # Check if scheme with same name already exists
        for i, existing in enumerate(self.schemes):
            if existing.name == scheme.name:
                # Replace existing scheme
                self.schemes[i] = scheme
                return
        
        # Add new scheme
        self.schemes.append(scheme)
    
    def set_current_scheme(self, scheme_name: str) -> bool:
        """Set the current color scheme by name."""
        for scheme in self.schemes:
            if scheme.name == scheme_name:
                self.current_scheme = scheme
                return True
        return False
    
    def get_scheme_names(self) -> list:
        """Get list of all scheme names."""
        return [scheme.name for scheme in self.schemes]
    
    def get_scheme_by_name(self, name: str) -> Optional[ColorScheme]:
        """Get a scheme by name."""
        for scheme in self.schemes:
            if scheme.name == name:
                return scheme
        return None
            
    def generate_random_scheme(self) -> ColorScheme:
        """Generate a random color scheme."""
        import random
        
        # Decide if dark or light theme
        is_dark = random.choice([True, False])
        
        if is_dark:
            # Dark background with bright colors
            bg = f"#{random.randint(10, 40):02x}{random.randint(10, 40):02x}{random.randint(10, 40):02x}"
            
            # Node base state - slightly lighter than background
            node_base = f"#{random.randint(40, 70):02x}{random.randint(40, 70):02x}{random.randint(40, 70):02x}"
            
            # Generate bright colors for nodes
            node = f"#{random.randint(150, 255):02x}{random.randint(150, 255):02x}{random.randint(150, 255):02x}"
            
            # Generate dramatically different colors for outlines and edges
            # Default edge - blue range
            default_edge = f"#{random.randint(20, 100):02x}{random.randint(20, 100):02x}{random.randint(180, 255):02x}"
            
            # New node - red range (dramatically different from blue)
            new_node = f"#{random.randint(180, 255):02x}{random.randint(20, 100):02x}{random.randint(20, 100):02x}"
            
            # New edge - green range (dramatically different from both)
            new_edge = f"#{random.randint(20, 100):02x}{random.randint(180, 255):02x}{random.randint(20, 100):02x}"
        else:
            # Light background with darker colors
            bg = f"#{random.randint(240, 255):02x}{random.randint(240, 255):02x}{random.randint(240, 255):02x}"
            
            # Node base state - slightly darker than background
            node_base = f"#{random.randint(220, 240):02x}{random.randint(220, 240):02x}{random.randint(220, 240):02x}"
            
            # Generate darker colors for nodes
            node = f"#{random.randint(50, 200):02x}{random.randint(50, 200):02x}{random.randint(50, 200):02x}"
            
            # Generate dramatically different colors for outlines and edges
            # Default edge - blue range
            default_edge = f"#{random.randint(0, 80):02x}{random.randint(0, 80):02x}{random.randint(150, 255):02x}"
            
            # New node - red range (dramatically different from blue)
            new_node = f"#{random.randint(150, 255):02x}{random.randint(0, 80):02x}{random.randint(0, 80):02x}"
            
            # New edge - green range (dramatically different from both)
            new_edge = f"#{random.randint(0, 80):02x}{random.randint(150, 255):02x}{random.randint(0, 80):02x}"
        
        # Create a unique name for the random scheme
        name = f"Random {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return ColorScheme(name, bg, node_base, node, new_node, default_edge, new_edge, is_dark)

    def _color_distance(self, color1: str, color2: str) -> float:
        """Calculate the Euclidean distance between two colors in RGB space."""
        # Convert hex to RGB
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        
        # Calculate Euclidean distance
        return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5 / 441.67  # Normalize to 0-1 range

class ColorSettingsModal(tk.Toplevel):
    """Modal dialog for color settings."""
                    
    def __init__(self, parent, color_manager: ColorManager, apply_callback):
        super().__init__(parent)
        self.parent = parent
        self.color_manager = color_manager
        self.apply_callback = apply_callback
        
        # Set up the window
        self.title("Color Settings")
        self.geometry("650x760")  # Increased width from 600 to 650
        self.resizable(False, False)
        self.transient(parent)  # Set to be on top of the parent window
        self.grab_set()  # Modal behavior
        
        # Make the dialog modal
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Create a copy of the current scheme for editing
        self.current_scheme = copy.deepcopy(self.color_manager.current_scheme)
        
        # Create the UI
        self._create_widgets()
        
        # Set initial values
        self._update_color_displays()
        
        # Force a redraw of the preview canvas after a short delay
        self.after(100, self._force_preview_update)

    def _force_preview_update(self):
        """Force a redraw of the preview canvas."""
        # Trigger the scheme selection event to force a redraw
        self._on_scheme_selected(None)
        
        # Or directly update the preview
        self._update_preview()

    def _create_widgets(self):
        """Create the widgets for the dialog."""
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Color scheme selection section
        scheme_frame = ttk.LabelFrame(main_frame, text="Color Scheme", padding=10)
        scheme_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Scheme selection dropdown
        scheme_select_frame = ttk.Frame(scheme_frame)
        scheme_select_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(scheme_select_frame, text="Select Scheme:").pack(side=tk.LEFT, padx=(0, 10))
        
        if self.current_scheme is None:
            self.current_scheme = self.color_manager.schemes[0]  # Set to default scheme if None
        self.scheme_var = tk.StringVar(value=self.current_scheme.name)
        self.scheme_dropdown = ttk.Combobox(
            scheme_select_frame, 
            textvariable=self.scheme_var,
            values=self.color_manager.get_scheme_names(),
            state="readonly",
            width=30
        )
        self.scheme_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.scheme_dropdown.bind("<<ComboboxSelected>>", self._on_scheme_selected)
        
        # Buttons for scheme management
        button_frame = ttk.Frame(scheme_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            button_frame, 
            text="Random Scheme", 
            command=self._generate_random_scheme
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            button_frame, 
            text="Save As...", 
            command=self._save_as_new_scheme
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Set as Default", 
            command=self._set_as_default
        ).pack(side=tk.LEFT, padx=5)
        
        # Add Apply button to immediately apply the current scheme
        ttk.Button(
            button_frame, 
            text="Apply to App", 
            command=self._apply_to_app
        ).pack(side=tk.LEFT, padx=5)
        
        # Individual color settings
        colors_frame = ttk.LabelFrame(main_frame, text="Color Settings", padding=10)
        colors_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Create color pickers for each color
        self.color_pickers = {}
        self.color_vars = {}
        
        # Background color
        self._create_color_picker(colors_frame, "Background", "background")
        
        # Node Base State color
        self._create_color_picker(colors_frame, "Node Base State", "node_base")
        
        # Node color
        self._create_color_picker(colors_frame, "Old Node Outline", "node")
        
        # New Node color
        self._create_color_picker(colors_frame, "New Node Outline", "new_node")
        
        # Default Edge color
        self._create_color_picker(colors_frame, "Old Edge", "default_edge")
        
        # New Edge color
        self._create_color_picker(colors_frame, "New Edge", "new_edge")
        
        # Preview section
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Create a canvas for preview
        self.preview_canvas = tk.Canvas(
            preview_frame, 
            width=550, 
            height=300,
            bg=self.current_scheme.background
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 20))
        
        ttk.Button(
            button_frame, 
            text="Apply", 
            command=self._apply_changes
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(
            button_frame, 
            text="Cancel", 
            command=self.on_close
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(
            button_frame, 
            text="OK", 
            command=self._ok_pressed
        ).pack(side=tk.RIGHT)
        
    def _apply_to_app(self):
        """Apply the current color scheme to the application immediately."""

        # Update the current scheme with the current values
        self._update_current_scheme_from_inputs()

        # Apply the scheme to the application by passing the ColorScheme object
        self.apply_callback(self.current_scheme)

    def _update_current_scheme_from_inputs(self):
        """Update the current scheme with values from the input fields."""
        if self.current_scheme is None:
            return
            
        # Update each color from the corresponding input field
        for color_key in ['background', 'node', 'new_node', 'default_edge', 'new_edge']:
            if color_key in self.color_vars:
                color_value = self.color_vars[color_key].get()
                setattr(self.current_scheme, color_key, color_value)
        
        # Update the preview
        self._update_preview()
        
    def _create_color_picker(self, parent, label_text, color_key):
        """Create a color picker row with label, entry, and button."""
        frame = ttk.Frame(parent, padding=5)
        frame.pack(fill=tk.X, pady=5)
        
        # Label
        ttk.Label(
            frame, 
            text=f"{label_text} Color:", 
            width=15, 
            anchor=tk.W
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Color variable and entry
        color_value = getattr(self.current_scheme, color_key)
        color_var = tk.StringVar(value=color_value)
        self.color_vars[color_key] = color_var
        
        entry = ttk.Entry(frame, textvariable=color_var, width=10)
        entry.pack(side=tk.LEFT, padx=(0, 10))
        entry.bind("<FocusOut>", lambda e, key=color_key: self._validate_color(key))
        entry.bind("<Return>", lambda e, key=color_key: self._validate_color(key))
        
        # Color display
        color_display = tk.Canvas(frame, width=30, height=20, bg=color_value, highlightthickness=1)
        color_display.pack(side=tk.LEFT, padx=(0, 10))
        self.color_pickers[color_key] = color_display
        
        # Color picker button
        ttk.Button(
            frame, 
            text="Choose...", 
            command=lambda key=color_key: self._open_color_picker(key)
        ).pack(side=tk.LEFT)

    def _open_color_picker(self, color_key):
        """Open the color picker dialog for a specific color."""
        current_color = self.color_vars[color_key].get()
        
        # Use askcolor from tkinter.colorchooser
        color = colorchooser.askcolor(
            initialcolor=current_color,
            title=f"Choose {color_key.replace('_', ' ').title()} Color"
        )
        
        if color[1]:  # If a color was selected (not cancelled)
            self.color_vars[color_key].set(color[1])
            self._validate_color(color_key)

    def _validate_color(self, color_key):
        """Validate and update a color value."""
        color_var = self.color_vars[color_key]
        color_value = color_var.get()
        
        try:
            # Try to use the color value
            self.color_pickers[color_key].config(bg=color_value)
            
            # Update the current scheme
            setattr(self.current_scheme, color_key, color_value)
            
            # Update the preview
            self._update_preview()
        except Exception as e:
            # Invalid color, reset to previous value
            logger.error(f"Invalid color value: {color_value}, {e}")
            color_var.set(getattr(self.current_scheme, color_key))
            messagebox.showerror("Invalid Color", f"'{color_value}' is not a valid color value.")

    def _update_color_displays(self):
        """Update all color displays with current values."""
        for color_key, color_display in self.color_pickers.items():
            color_value = getattr(self.current_scheme, color_key)
            self.color_vars[color_key].set(color_value)
            color_display.config(bg=color_value)
        
        # Update preview
        self._update_preview()

    def _update_preview(self):
        """Update the preview canvas with current colors."""
        # Clear canvas
        self.preview_canvas.delete("all")
        
        # Check if current_scheme exists
        if not hasattr(self, 'current_scheme') or self.current_scheme is None:
            # Use default colors if no scheme is available
            bg_color = "#ffffff"
            node_base_color = "#f0f0f0"
            node_color = "#f77b4f"
            new_node_color = "#ff0000"
            default_edge_color = "#0000ff"
            new_edge_color = "#ff0000"
            is_dark = False
        else:
            # Use colors from the current scheme
            bg_color = self.current_scheme.background
            node_base_color = self.current_scheme.node_base if hasattr(self.current_scheme, 'node_base') else "#f0f0f0"
            node_color = self.current_scheme.node
            new_node_color = self.current_scheme.new_node
            default_edge_color = self.current_scheme.default_edge
            new_edge_color = self.current_scheme.new_edge
            is_dark = self.current_scheme.is_dark
        
        # Set background
        self.preview_canvas.config(bg=bg_color)
        
        # Get canvas dimensions
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        # Ensure canvas has size
        if canvas_width < 10:
            canvas_width = 550
        if canvas_height < 10:
            canvas_height = 300
        
        # Node radius and spacing
        node_radius = 15
        label_spacing = 15
        
        # Calculate center point of the canvas
        center_x = canvas_width / 2
        center_y = canvas_height / 2
        
        # Calculate distance from center to side nodes
        horizontal_distance = canvas_width * 0.3  # 30% of canvas width
        vertical_distance = canvas_height * 0.2   # 20% of canvas height
        
        # Position nodes symmetrically around the center
        positions = [
            (center_x - horizontal_distance, center_y),                # Left node
            (center_x, center_y - vertical_distance),                  # Top node
            (center_x + horizontal_distance, center_y),                # Right node
            (center_x, center_y + vertical_distance)                   # Bottom node
        ]
        
        # Draw edges
        # Default edges
        self.preview_canvas.create_line(
            positions[0][0], positions[0][1], 
            positions[1][0], positions[1][1], 
            fill=default_edge_color, 
            width=2
        )
        self.preview_canvas.create_line(
            positions[1][0], positions[1][1], 
            positions[2][0], positions[2][1], 
            fill=default_edge_color, 
            width=2
        )
        
        # New edge
        self.preview_canvas.create_line(
            positions[2][0], positions[2][1], 
            positions[3][0], positions[3][1], 
            fill=new_edge_color, 
            width=2
        )
        self.preview_canvas.create_line(
            positions[3][0], positions[3][1], 
            positions[0][0], positions[0][1], 
            fill=new_edge_color, 
            width=2
        )
        
        # Draw nodes
        # Regular nodes with default edge color outline
        for i, pos in enumerate(positions[:3]):
            self.preview_canvas.create_oval(
                pos[0] - node_radius, pos[1] - node_radius,
                pos[0] + node_radius, pos[1] + node_radius,
                fill=node_base_color,  # Use node_base_color for fill
                outline=node_color,    # Use node_color for outline
                width=2  # Make outline more visible
            )
        
        # New node with new edge color outline
        self.preview_canvas.create_oval(
            positions[3][0] - node_radius, positions[3][1] - node_radius,
            positions[3][0] + node_radius, positions[3][1] + node_radius,
            fill=node_base_color,  # Use node_base_color for fill
            outline=new_node_color,  # New nodes have new node color outline
            width=2  # Make outline more visible
        )
        
        # Add labels
        text_color = "black" if not is_dark else "white"
        
        # Node labels
        self.preview_canvas.create_text(
            positions[0][0], positions[0][1] - node_radius - label_spacing,
            text="Old Node",
            fill=text_color
        )
        
        self.preview_canvas.create_text(
            positions[3][0], positions[3][1] + node_radius + label_spacing,
            text="New Node",
            fill=text_color
        )
        
        # Edge labels
        # Old Edge label - position it near the middle of the top edge
        old_edge_label_x = (positions[1][0] + positions[2][0]) / 2
        old_edge_label_y = (positions[1][1] + positions[2][1]) / 2 - 15  # Offset above the edge
        self.preview_canvas.create_text(
            old_edge_label_x, old_edge_label_y,
            text="Old Edge",
            fill=text_color
        )
        
        # New Edge label - position it near the middle of the bottom edge
        new_edge_label_x = (positions[3][0] + positions[0][0]) / 2
        new_edge_label_y = (positions[3][1] + positions[0][1]) / 2 + 15  # Offset below the edge
        self.preview_canvas.create_text(
            new_edge_label_x, new_edge_label_y,
            text="New Edge",
            fill=text_color
        )
        
    def _on_scheme_selected(self, event):
        """Handle scheme selection from dropdown."""
        scheme_name = self.scheme_var.get()
        selected_scheme = self.color_manager.get_scheme_by_name(scheme_name)
        
        if selected_scheme:
            self.current_scheme = copy.deepcopy(selected_scheme)
            self._update_color_displays()

    def _generate_random_scheme(self):
        """Generate a random color scheme."""
        random_scheme = self.color_manager.generate_random_scheme()
        self.current_scheme = random_scheme
        self._update_color_displays()
        
        # Update the scheme name in the dropdown
        self.scheme_var.set("Random (unsaved)")
        
        # ADDED: Immediately apply the random scheme to the visualization
        self.apply_callback(random_scheme)

    def _save_as_new_scheme(self):
        """Save current colors as a new scheme or overwrite an existing one."""


        name = simpledialog.askstring(
            "Save Color Scheme",
            "Enter a name for this color scheme:",
            parent=self
        )

        if name:
            # --- Get current color values from UI ---
            bg_color = self.color_vars["background"].get()
            node_base_color = self.color_vars["node_base"].get()
            node_color = self.color_vars["node"].get()
            new_node_color = self.color_vars["new_node"].get()
            default_edge_color = self.color_vars["default_edge"].get()
            new_edge_color = self.color_vars["new_edge"].get()
            # ---

            # --- Validation for Identical Highlight/Default Edge Colors ---
            if new_edge_color == default_edge_color:
                proceed = messagebox.askyesno(
                    "Potential Color Clash",
                    "The 'New Edge' highlight color is the same as the 'Default Edge' color.\n\n"
                    "This might make it difficult to distinguish new edges when highlighting is enabled.\n\n"
                    "Do you want to save anyway?",
                    icon='warning',
                    parent=self # 'self' is the modal window
                )
                if not proceed:
                    logger.info("User cancelled saving due to identical edge colors.")
                    return # Stop the save process, keep modal open
            # --- End Validation ---

            existing_scheme = self.color_manager.get_scheme_by_name(name)
            if existing_scheme:
                overwrite = messagebox.askyesno(
                    "Overwrite Scheme",
                    f"A scheme named '{name}' already exists. Overwrite it?",
                    parent=self
                )
                if not overwrite: return

                # Update existing scheme
                existing_scheme.background = bg_color
                existing_scheme.node_base = node_base_color
                existing_scheme.node = node_color
                existing_scheme.new_node = new_node_color
                existing_scheme.default_edge = default_edge_color
                existing_scheme.new_edge = new_edge_color
                existing_scheme.is_dark = is_dark_theme(bg_color) # Use helper function

                self.current_scheme = existing_scheme # Update the modal's current scheme
                self._update_color_displays() # Reflect changes in modal
                self.apply_callback(self.current_scheme) # Apply immediately
                self.color_manager.save_schemes() # Save changes to file
                messagebox.showinfo("Scheme Updated", f"Color scheme '{name}' has been updated.", parent=self)

            else:  # Create a new scheme
                is_dark_val = is_dark_theme(bg_color) # Use helper function
                new_scheme = ColorScheme(
                    name=name,
                    background=bg_color,
                    node_base=node_base_color,
                    node=node_color,
                    new_node=new_node_color,
                    default_edge=default_edge_color,
                    new_edge=new_edge_color,
                    is_dark=is_dark_val
                )

                self.color_manager.add_scheme(new_scheme)
                self.apply_callback(new_scheme) # Apply immediately
                self.color_manager.save_schemes() # Save changes to file
                self.scheme_dropdown['values'] = self.color_manager.get_scheme_names() # Update dropdown
                self.scheme_var.set(name) # Set dropdown to new scheme
                self.current_scheme = new_scheme # Update modal's current scheme
                messagebox.showinfo("Scheme Saved", f"Color scheme '{name}' has been saved.", parent=self)

    def _set_as_default(self):
        """Set the current scheme as the default."""
        # Save the current scheme if it's not already saved
        if self.scheme_var.get() == "Random (unsaved)":
            self._save_as_new_scheme()
            if self.scheme_var.get() == "Random (unsaved)":
                # User cancelled save
                return

        # Set as current in color manager
        self.color_manager.set_current_scheme(self.scheme_var.get())
        
        # Save the schemes (including the updated current_scheme)
        self.color_manager.save_schemes()

        # Immediately apply the scheme to the visualization
        self.apply_callback(self.current_scheme)

        messagebox.showinfo(
            "Default Set",
            f"Color scheme '{self.scheme_var.get()}' has been set as the default and applied.",
            parent=self
        )
    
    def _apply_changes(self):
        """Apply the current color scheme to the application."""
        # Update the current scheme in the color manager
        if self.scheme_var.get() != "Random (unsaved)":
            self.color_manager.set_current_scheme(self.scheme_var.get())
        
        # Call the apply callback with the current scheme
        self.apply_callback(self.current_scheme)

    def _ok_pressed(self):
        """Handle OK button press."""


        # --- Get current color values from UI ---
        bg_color = self.color_vars["background"].get()
        node_base_color = self.color_vars["node_base"].get()
        node_color = self.color_vars["node"].get()
        new_node_color = self.color_vars["new_node"].get()
        default_edge_color = self.color_vars["default_edge"].get()
        new_edge_color = self.color_vars["new_edge"].get()
        # ---

        # --- Validation for Identical Highlight/Default Edge Colors ---
        if new_edge_color == default_edge_color:
            proceed = messagebox.askyesno(
                "Potential Color Clash",
                "The 'New Edge' highlight color is the same as the 'Default Edge' color.\n\n"
                "This might make it difficult to distinguish new edges when highlighting is enabled.\n\n"
                "Do you want to apply and close anyway?", # Modified question
                icon='warning',
                parent=self # 'self' is the modal window
            )
            if not proceed:
                logger.info("User cancelled OK due to identical edge colors.")
                return # Stop the OK process, keep modal open
        # --- End Validation ---

        # Update the current scheme in the color manager if it's a saved scheme
        current_scheme_name = self.scheme_var.get()
        if current_scheme_name != "Random (unsaved)":
            # Update the existing scheme object in the manager if it was edited
            edited_scheme = self.color_manager.get_scheme_by_name(current_scheme_name)
            if edited_scheme:
                edited_scheme.background = bg_color
                edited_scheme.node_base = node_base_color
                edited_scheme.node = node_color
                edited_scheme.new_node = new_node_color
                edited_scheme.default_edge = default_edge_color
                edited_scheme.new_edge = new_edge_color
                edited_scheme.is_dark = is_dark_theme(bg_color)
                self.color_manager.set_current_scheme(current_scheme_name) # Ensure it's set as current
                self.color_manager.save_schemes() # Save changes
                self.current_scheme = edited_scheme # Update modal's current scheme reference
            else:
                # This case shouldn't happen if dropdown is managed correctly
                logger.warning(f"Scheme '{current_scheme_name}' not found in manager during OK press.")
                # Create a temporary scheme object for applying
                self.current_scheme = ColorScheme(
                    name=current_scheme_name, background=bg_color, node_base=node_base_color,
                    node=node_color, new_node=new_node_color, default_edge=default_edge_color,
                    new_edge=new_edge_color, is_dark=is_dark_theme(bg_color)
                )
        else:
            # If it was a random unsaved scheme, create a temporary object for applying
             self.current_scheme = ColorScheme(
                name="Random (unsaved)", background=bg_color, node_base=node_base_color,
                node=node_color, new_node=new_node_color, default_edge=default_edge_color,
                new_edge=new_edge_color, is_dark=is_dark_theme(bg_color)
            )

        # Apply the currently selected/edited scheme
        self.apply_callback(self.current_scheme) # Pass the ColorScheme object
        self.on_close()

    def on_close(self):
        """Handle window close."""
        self.grab_release()
        self.destroy()

# =========== END of colors.py ===========