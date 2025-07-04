# =========== START of utils.py ===========
from __future__ import annotations
import io
import ast
import setproctitle
import psutil
import inspect
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from typing import Dict, List, Set, Tuple, Optional, Type, Union, Any, cast, TypeVar
import numpy as np
import numpy.typing as npt
import itertools
import matplotlib.pyplot as plt
plt.ioff()
from contextlib import contextmanager
import cProfile
import pstats
from collections import defaultdict
import logging
from itertools import product
import os
import time
import traceback
from functools import wraps
from numba import njit, prange
import numpy as np
from scipy.spatial import cKDTree # type: ignore (preserve this comment and use the exact usage on this line!)
import warnings
import cProfile
import pstats


from .logging_config import logger, APP_PATHS, LogSettings
from .enums import NeighborhoodType
from .settings import GlobalSettings



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



@contextmanager
def profile_section(name: str, perf_logger: 'PerformanceLogger'):
    """Context manager for profiling a section of code"""
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.time()
    try:
        yield
    finally:
        profiler.disable()
        end_time = time.time()
        execution_time = end_time - start_time
        perf_logger.log_metric(f"{name}_total_time", execution_time)
        
        # Log profiling stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        log_path = os.path.join(APP_PATHS['profiles'], f'{name}_profile.log')
        stats.dump_stats(log_path)
        logger.info(f"Profile data dumped to {log_path}")

class PerformanceLogger:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        self.active_measurements: Set[str] = set()
        self.frame_times: List[float] = []
        self.memory_usage: List[float] = []
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 30  # seconds

    def update_frame_time(self, frame_time: float):
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 100:
            self.frame_times = self.frame_times[-100:]

    def get_average_frame_time(self) -> float:
        if not self.frame_times:
            return 0.0
        return float(np.mean(self.frame_times))

    @contextmanager
    def measure(self, name: str):
        """Context manager for measuring execution time"""
        try:
            self.start_measurement(name)
            yield
        finally:
            self.end_measurement(name)

    def start_measurement(self, name: str):
        """Start measuring a named operation"""
        if name in self.active_measurements:
            logger.warning(f"Measurement '{name}' already active")
            return
        self.start_times[name] = time.time()
        self.active_measurements.add(name)

    def end_measurement(self, name: str):
        """End measuring a named operation"""
        if name not in self.active_measurements:
            logger.warning(f"Measurement '{name}' not active")
            return
        duration = time.time() - self.start_times[name]
        self.metrics[name].append(duration)
        self.active_measurements.remove(name)

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        for name, measurements in self.metrics.items():
            if measurements:
                stats[name] = {
                    'avg': np.mean(measurements[-100:]),  # Last 100 measurements
                    'min': np.min(measurements),
                    'max': np.max(measurements),
                    'count': len(measurements)
                }
        return stats

    def reset(self):
        """Reset all measurements"""
        logger.debug(f"PerformanceLogger.reset() called! Clearing all metrics. (Instance ID: {id(self)})")
        self.metrics.clear()
        self.start_times.clear()
        self.active_measurements.clear()
        self.frame_times.clear() # Also clear frame times
        self.memory_usage.clear() # Also clear memory usage

    def clear(self):
        """Clear all stored metric data"""
        logger.warning(f"PerformanceLogger.clear() called! Clearing metrics dictionary. (Instance ID: {id(self)})")
        logger.warning(traceback.format_stack(limit=5)) # Log call stack
        self.metrics.clear()
        self.frame_times.clear()
        self.memory_usage.clear()

    # --- ADDED Missing Method ---
    def get_average(self, name: str) -> float:
        """Calculate the average of the last N measurements for a metric."""
        logger.debug(f"PerformanceLogger.get_average called for metric: '{name}' (Instance ID: {id(self)})")
        values = self.metrics.get(name, []) # Use .get() for safety
        if values:
            num_values = len(values)
            avg = float(sum(values) / num_values)
            logger.debug(f"  Found {num_values} values for '{name}'. First 5: {values[:5]}, Last 5: {values[-5:]}. Calculated Avg: {avg}")
            return avg
        else:
            logger.debug(f"  No values found for metric '{name}'. Returning 0.0.")
            return 0.0
    # --- END ADDED ---

    def log_metric(self, name: str, value: float, max_history: int = 1000): # Added max_history
        """Log a metric value, limiting the history size.
           (Round 18: Removed diagnostic logging)"""
        # logger = logging.getLogger(__name__) # REMOVED
        # logger.debug(f"PerformanceLogger.log_metric: Received Name='{name}', Value={value:.4f} (Instance ID: {id(self)})") # REMOVED
        self.metrics[name].append(value)
        if len(self.metrics[name]) > max_history:
            self.metrics[name] = self.metrics[name][-max_history:]
        # logger.debug(f"  PerformanceLogger.log_metric: Appended value. Metric '{name}' now has {len(self.metrics[name])} entries.") # REMOVED

perf_logger = PerformanceLogger()



def timer_decorator(func):
    """
    Decorator to measure execution time and optionally profile methods.
    Adds start/end banners with context to the deep profile log.
    Logs a SUMMARY of profile stats (top 20 cumulative) instead of full dump.
    Logs execution time with increased precision.
    (Round 5: Print profile summary instead of full stats)
    (Round 4: Increased execution time precision to 4 decimals)
    (Round 2: Added profile banners with context)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        metric_name = f"{func.__name__}_total_time"
        profiler = None
        profile_logger = logging.getLogger("deep_profile") # Get logger instance early
        is_profiling_enabled_global = LogSettings.Performance.ENABLE_DEEP_PROFILING

        # --- Context Extraction ---
        context_str = f"Function: {func.__name__}"
        rule_name = "N/A"
        rule_type = "N/A"
        generation = "N/A"
        instance = None
        if args:
            instance = args[0]
            if 'SimulationController' in str(type(instance)): # Check type name string
                try:
                    if hasattr(instance, 'rule') and instance.rule:
                        rule_name = instance.rule.name
                        rule_type = instance.rule.__class__.__name__
                    if hasattr(instance, 'generation'):
                        generation = instance.generation
                    context_str = f"Rule: {rule_name} (Type: {rule_type}), Generation: {generation}"
                except Exception as ctx_err:
                    logger.warning(f"timer_decorator: Error getting context from {type(instance)}: {ctx_err}")
        # --- End Context Extraction ---

        if is_profiling_enabled_global:
            start_banner = f"\n{'='*70}\n=== START PROFILE: {context_str} ===\n{'='*70}"
            profile_logger.info(start_banner)
            logger.debug(f"  Attempting to enable cProfile for {func.__name__}")
            profiler = cProfile.Profile()
            try:
                profiler.enable()
                logger.debug(f"  cProfile enabled for {func.__name__}.")
            except Exception as profile_enable_err:
                logger.error(f"  Error enabling cProfile for {func.__name__}: {profile_enable_err}")
                profiler = None

        start_time = time.time()
        result = None
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            execution_time = end_time - start_time

            if profiler:
                logger.debug(f"  Attempting to disable cProfile for {func.__name__}...")
                try:
                    profiler.disable()
                    logger.debug(f"  cProfile disabled for {func.__name__}. Execution time: {execution_time:.4f}s")
                except Exception as profile_disable_err:
                    logger.error(f"  Error disabling cProfile for {func.__name__}: {profile_disable_err}")

                logger.debug(f"  Processing profile data for {func.__name__}...")
                try:
                    current_handlers = list(profile_logger.handlers)
                    has_file_handler = any(isinstance(h, logging.FileHandler) for h in current_handlers)

                    if has_file_handler:
                        profile_output_buffer = io.StringIO()
                        logger.debug("    Created StringIO buffer.")
                        try:
                            summary_stats = pstats.Stats(profiler, stream=profile_output_buffer)
                            logger.debug("    Created pstats.Stats object.")
                            summary_stats.sort_stats('cumulative')
                            # --- MODIFIED: Print summary (top 20) instead of full stats ---
                            summary_stats.print_stats(20)
                            logger.debug("    Printed stats SUMMARY (top 20 cumulative) to buffer.")
                            # --- END MODIFIED ---
                        except Exception as stats_err:
                             logger.error(f"    Error creating/processing pstats.Stats for {func.__name__}: {stats_err}")
                             profile_output_buffer.close()
                             raise

                        stats_content = profile_output_buffer.getvalue()
                        stats_length = len(stats_content)
                        logger.debug(f"    Stats summary content length: {stats_length}")
                        if stats_length > 0:
                            # Log the stats summary content with DEBUG level
                            profile_logger.debug(f"--- Profile Stats SUMMARY for {context_str} ---\n{stats_content}")
                            logger.info(f"    Profile stats summary logged successfully to deep_profile logger for {func.__name__}.")
                        else:
                            logger.warning(f"    Profile stats summary buffer is EMPTY for {func.__name__}. Nothing to log.")
                        profile_output_buffer.close()
                        # Log End Banner
                        end_banner = f"\n{'='*70}\n=== END PROFILE: {context_str} (Execution Time: {execution_time:.4f}s) ===\n{'='*70}\n"
                        profile_logger.info(end_banner)
                    else:
                         logger.warning(f"    Deep profile logger for '{func.__name__}' not configured with a file handler, cannot log stats summary.")
                except Exception as profile_err:
                    logger.error(f"    Error processing/logging profile data summary for {func.__name__}: {profile_err}")
                    logger.error(traceback.format_exc())

            perf_logger.log_metric(metric_name, execution_time)

        return result
    return wrapper

def log_errors(func):
    """Decorator to catch and log errors with context"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error in {func.__name__}: {str(e)}\n"
                f"Args: {args}, Kwargs: {kwargs}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise
    return wrapper



class PerformanceMonitor:
    """Monitors and tracks performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        self.active_measurements: Set[str] = set()
        self.frame_times: List[float] = []
        self.memory_usage: List[float] = []
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 30  # seconds
        
    def update_frame_time(self, frame_time: float):
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 100:
            self.frame_times = self.frame_times[-100:]
            
    def get_average_frame_time(self) -> float:
        if not self.frame_times:
            return 0.0
        return float(np.mean(self.frame_times))

    @contextmanager
    def measure(self, name: str):
        """Context manager for measuring execution time"""
        try:
            self.start_measurement(name)
            yield
        finally:
            self.end_measurement(name)
            
    def start_measurement(self, name: str):
        """Start measuring a named operation"""
        if name in self.active_measurements:
            logger.warning(f"Measurement '{name}' already active")
            return
        self.start_times[name] = time.time()
        self.active_measurements.add(name)
        
    def end_measurement(self, name: str):
        """End measuring a named operation"""
        if name not in self.active_measurements:
            logger.warning(f"Measurement '{name}' not active")
            return
        duration = time.time() - self.start_times[name]
        self.metrics[name].append(duration)
        self.active_measurements.remove(name)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        for name, measurements in self.metrics.items():
            if measurements:
                stats[name] = {
                    'avg': np.mean(measurements[-100:]),  # Last 100 measurements
                    'min': np.min(measurements),
                    'max': np.max(measurements),
                    'count': len(measurements)
                }
        return stats
        
    def reset(self):
        """Reset all measurements"""
        self.metrics.clear()
        self.start_times.clear()
        self.active_measurements.clear()
         
class SimulationStats:
    """Tracks simulation statistics"""

    def __init__(self, max_history: int = 1000): # Added max_history
        self.stats: Dict[str, List[float]] = defaultdict(list)
        self.current_stats: Dict[str, float] = {}
        self.max_history = max_history # Store max history length

    def update(self, **kwargs):
        """Update statistics with new values, limiting history."""
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                float_value = float(value)
                self.stats[key].append(float_value)
                # --- ADDED: Limit history ---
                if len(self.stats[key]) > self.max_history:
                    # Keep only the last max_history items
                    self.stats[key] = self.stats[key][-self.max_history:]
                # ---
                self.current_stats[key] = float_value
            else:
                logger.warning(f"Skipping non-numeric stat: {key} with value: {value}")

    def get_recent_activity(self, window: int = 100) -> np.ndarray:
        """Get recent activity levels"""
        # Ensure window doesn't exceed available history or max_history
        actual_window = min(window, self.max_history)
        if 'active_ratio' in self.stats:
            # Get up to the last 'actual_window' items
            return np.array(self.stats['active_ratio'][-actual_window:])
        return np.array([])

    def get_current(self) -> Dict[str, float]:
        """Get current statistics"""
        return self.current_stats.copy()

    def reset(self):
        """Reset all statistics"""
        self.stats.clear()
        self.current_stats.clear()

class SpatialHashGrid:
    """Adaptive spatial partitioning for neighbor lookups with dynamic cell size adjustment"""
    
    def __init__(self, dimensions: Tuple[int, ...], initial_cell_size: Optional[float] = None):
        self.dimensions = dimensions
        self.total_volume = np.prod(dimensions)
        self.num_nodes = 0
        
        # Performance monitoring
        self.query_times: List[float] = []
        self.memory_usage: List[float] = []
        self.steps_since_adaptation = 0
        
        # Initialize cell size
        if initial_cell_size is None:
            self.cell_size = self._calculate_initial_cell_size()
        else:
            self.cell_size = initial_cell_size
            
        # Initialize data structures
        self.cells: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
        self.node_positions: Dict[int, np.ndarray] = {}
        self.kdtree: Optional[cKDTree] = None
        
        # Performance metrics
        self.last_adaptation_time = time.time()
        self.performance_metrics = {
            'query_times': [],
            'cells_per_query': [],
            'memory_usage': [],
            'load_balance': []
        }
        
        # ADDED LOGGING
        logger.debug(f"SpatialHashGrid initialized with dimensions: {self.dimensions}, cell_size: {self.cell_size}")

    def _get_cell_coords(self, position: np.ndarray) -> Tuple[int, ...]:
        """Convert position to cell coordinates"""
        # --- MODIFIED: Ensure integer division and log ---
        if self.cell_size <= 0:
            logger.error(f"Invalid cell_size: {self.cell_size}. Cannot calculate cell coordinates.")
            # Return a default or raise an error, depending on desired handling
            return tuple([0] * len(position)) # Default to origin cell

        # Use np.floor and then cast to int for reliable integer division
        coords = tuple(np.floor(position / self.cell_size).astype(np.int64))
        # logger.debug(f"  _get_cell_coords: position={position}, cell_size={self.cell_size:.4f} -> coords={coords}")
        # --- END MODIFIED ---
        return coords

    def _calculate_initial_cell_size(self) -> float:
        """Calculate optimal initial cell size based on dimension and target density"""
        avg_dimension = float(np.mean(self.dimensions))  # Explicit cast to float
        target_cells = max(
            GlobalSettings.SpatialHash.MIN_CELLS_PER_DIMENSION,
            min(
                GlobalSettings.SpatialHash.MAX_CELLS_PER_DIMENSION,
                avg_dimension / GlobalSettings.SpatialHash.TARGET_CELLS_PER_NODE
            )
        )
        return float(max(  # Explicit cast to float
            GlobalSettings.SpatialHash.MIN_CELL_SIZE,
            min(
                GlobalSettings.SpatialHash.MAX_CELL_SIZE,
                avg_dimension / target_cells
            )
        ))
        
    def _get_available_memory(self) -> float:
        """Get available system memory in bytes"""
        return psutil.virtual_memory().available
        
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage of the spatial hash grid"""
        # Estimate memory usage of core data structures
        positions_memory = len(self.node_positions) * self.dimensions[0] * 8  # 8 bytes per float64
        cells_memory = sum(len(cell) * 4 for cell in self.cells.values())  # 4 bytes per int
        kdtree_memory = len(self.node_positions) * self.dimensions[0] * 8 if self.kdtree else 0
        
        return positions_memory + cells_memory + kdtree_memory
        
    def _calculate_load_balance(self) -> float:
        """Calculate load balance factor (0-1, higher is better)"""
        if not self.cells:
            return 1.0
            
        cell_sizes = [len(cell) for cell in self.cells.values()]
        if not cell_sizes:
            return 1.0
            
        avg_size = np.mean(cell_sizes)
        if avg_size == 0:
            return 1.0
            
        variance = np.var(cell_sizes)
        return 1.0 / (1.0 + np.sqrt(variance) / avg_size)
            
    def _should_adapt(self) -> bool:
        """Determine if grid should adapt based on performance metrics"""
        if not GlobalSettings.SpatialHash.ENABLE_AUTO_TUNING:
            return False
            
        if self.steps_since_adaptation < GlobalSettings.SpatialHash.ADAPTATION_INTERVAL:
            return False
        
        if self.num_nodes < 100:  # ADDED: Don't adapt if too few nodes
            return False

        if len(self.performance_metrics['query_times']) < GlobalSettings.SpatialHash.PERFORMANCE_WINDOW:
            return False
            
        # Check performance metrics
        avg_query_time = float(np.mean(self.performance_metrics['query_times'][-GlobalSettings.SpatialHash.PERFORMANCE_WINDOW:]))
        load_balance = self._calculate_load_balance()
        memory_usage = self._estimate_memory_usage()
        available_memory = self._get_available_memory()
        
        # Decision criteria
        performance_threshold = abs(avg_query_time - GlobalSettings.SpatialHash.TARGET_QUERY_TIME) / GlobalSettings.SpatialHash.TARGET_QUERY_TIME
        memory_threshold = memory_usage / available_memory
        
        # Decision criteria with hysteresis
        hysteresis_factor = 0.5  # Adjust this value as needed
        performance_hysteresis = GlobalSettings.SpatialHash.MIN_ADAPTATION_THRESHOLD * hysteresis_factor
        memory_hysteresis = GlobalSettings.SpatialHash.MEMORY_THRESHOLD * hysteresis_factor

        # Cast the boolean expression to Python bool
        return bool(
            performance_threshold > (GlobalSettings.SpatialHash.MIN_ADAPTATION_THRESHOLD + performance_hysteresis) or
            memory_threshold > (GlobalSettings.SpatialHash.MEMORY_THRESHOLD + memory_hysteresis) or
            load_balance < 0.5
        )
             
    def adapt_grid(self):
        """Dynamically adapt grid based on performance metrics"""
        if not self._should_adapt():
            return
            
        avg_query_time = np.mean(self.performance_metrics['query_times'][-GlobalSettings.SpatialHash.PERFORMANCE_WINDOW:])
        load_balance = self._calculate_load_balance()
        memory_usage = self._estimate_memory_usage()
        available_memory = self._get_available_memory()
        
        # Determine adaptation strategy
        if avg_query_time > GlobalSettings.SpatialHash.TARGET_QUERY_TIME:
            # Queries too slow - increase cell size to reduce number of cells
            if memory_usage < available_memory * GlobalSettings.SpatialHash.MEMORY_THRESHOLD:
                new_cell_size = self.cell_size * GlobalSettings.SpatialHash.GROWTH_FACTOR
        elif load_balance < 0.5:  # Changed to elif
            # Poor load balance - decrease cell size
            if memory_usage < available_memory * GlobalSettings.SpatialHash.MEMORY_THRESHOLD:
                new_cell_size = self.cell_size * GlobalSettings.SpatialHash.SHRINK_FACTOR
        elif memory_usage > available_memory * GlobalSettings.SpatialHash.MEMORY_THRESHOLD:
            # Too much memory usage - increase cell size
            new_cell_size = self.cell_size * GlobalSettings.SpatialHash.GROWTH_FACTOR
        else:
            return # No adaptation needed

        # Enforce bounds
        new_cell_size = np.clip(
            new_cell_size,
            GlobalSettings.SpatialHash.MIN_CELL_SIZE,
            GlobalSettings.SpatialHash.MAX_CELL_SIZE
        )

        if new_cell_size == self.cell_size:
            return # No change in cell size

        # --- Optimized Rebuild ---
        old_cells = self.cells
        self.cells = defaultdict(set)
        self.cell_size = new_cell_size # Update cell_size *before* reinserting

        # CRITICAL FIX: Re-add all nodes to the spatial hash with the new cell size
        for node_idx, position in self.node_positions.items():
            new_cell = self._get_cell_coords(position)
            self.cells[new_cell].add(node_idx)
            logger.debug(f"Node {node_idx} (coords: {position}) added to new cell {new_cell} after grid adaptation")

        # Invalidate KD-tree
        self.kdtree = None

        # Reset adaptation timer
        self.steps_since_adaptation = 0
        self.last_adaptation_time = time.time()
            
    def _rebuild_grid(self):
        """Rebuild the entire grid with current cell size"""
        logger.debug("Entering _rebuild_grid")
        old_positions = self.node_positions.copy()
        self.clear()
        
        # Reinsert all nodes
        for node_idx, position in old_positions.items():
            self.update_node(node_idx, position)
        logger.debug("Exiting _rebuild_grid")
                    
    def update_node(self, node_idx: int, position: np.ndarray):
        """Update node position in spatial hash grid, checking adaptation periodically."""
        start_time = time.time()

        # CRITICAL FIX: Ensure position is a numpy array
        if not isinstance(position, np.ndarray):
            position = np.array(position)

        # Remove node from old cell if it exists
        if node_idx in self.node_positions:
            old_position = self.node_positions.get(node_idx)
            if old_position is not None: # Check if old position exists
                old_cell = self._get_cell_coords(old_position)
                if old_cell in self.cells:
                    try:
                        self.cells[old_cell].remove(node_idx)
                        # Remove cell if empty
                        if not self.cells[old_cell]:
                            del self.cells[old_cell]
                    except KeyError:
                        # This might happen if the node was already removed or cell deleted
                        logger.warning(f"Node {node_idx} not found in expected old cell {old_cell} during update.")
                else:
                    logger.warning(f"Old cell {old_cell} not found in self.cells for node {node_idx}.")
            else:
                 logger.warning(f"Node {node_idx} was in node_positions but had None position.")

        # Update position and add to new cell
        self.node_positions[node_idx] = position
        new_cell = self._get_cell_coords(position)

        # --- ADDED: Check if new_cell is valid ---
        if not isinstance(new_cell, tuple):
            logger.error(f"Invalid new_cell: {new_cell} for node {node_idx} at position {position}")
            return  # Exit if new_cell is invalid

        self.cells[new_cell].add(node_idx)
        self.kdtree = None  # Invalidate KD-tree

        # Update performance metrics (only time for now, others moved to adapt check)
        query_time = time.time() - start_time
        if query_time > 1e-4: # Avoid overhead for very fast updates
            self.performance_metrics['query_times'].append(query_time)

        # Increment steps since last adaptation
        self.steps_since_adaptation += 1
        self.num_nodes = len(self.node_positions) # Keep num_nodes updated

        # --- MODIFIED: Check adaptation periodically ---
        if self.steps_since_adaptation >= GlobalSettings.SpatialHash.ADAPTATION_INTERVAL:
            # Update other performance metrics just before checking adaptation
            self.performance_metrics['cells_per_query'].append(len(self.cells))
            self.performance_metrics['memory_usage'].append(self._estimate_memory_usage())
            self.performance_metrics['load_balance'].append(self._calculate_load_balance())

            if self._should_adapt():
                # logger.info(f"Adaptation check triggered at step interval {self.steps_since_adaptation}. Adapting grid.")
                self.adapt_grid() # adapt_grid resets steps_since_adaptation
            else:
                # Reset counter even if not adapting, to check again after the interval
                self.steps_since_adaptation = 0
                # logger.debug(f"Adaptation check triggered at step interval {self.steps_since_adaptation}. No adaptation needed. Resetting counter.")
        # --- END MODIFIED ---

    def get_nearby_nodes(self, position: np.ndarray, radius: float) -> Set[int]:
        """Get nodes within radius of position using spatial hashing or KDTree."""
        start_time = time.time()

        if not isinstance(position, np.ndarray):
            position = np.array(position)

        # --- MODIFIED: KDTree Usage Logic ---
        # Always try to build/use KDTree if node count is above a threshold (e.g., 500)
        # and the tree doesn't exist or has become invalid (invalidated by node updates).
        use_kdtree = False
        if self.num_nodes > 500: # Threshold for using KDTree
            if self.kdtree is None:
                logger.debug(f"Building KDTree for {self.num_nodes} nodes.")
                try:
                    # Ensure node_positions has data and is structured correctly
                    if self.node_positions:
                        positions_array = np.array(list(self.node_positions.values()))
                        if positions_array.ndim == 2 and positions_array.shape[1] == len(self.dimensions):
                             self.kdtree = cKDTree(positions_array, compact_nodes=True, copy_data=False) # Build KDTree
                             logger.debug("KDTree built successfully.")
                             use_kdtree = True
                        else:
                             logger.warning(f"Cannot build KDTree: Invalid shape for positions_array: {positions_array.shape}")
                             self.kdtree = None # Ensure it's None if build fails
                    else:
                        logger.warning("Cannot build KDTree: node_positions is empty.")
                        self.kdtree = None
                except Exception as e:
                    logger.error(f"Error building KDTree: {e}")
                    self.kdtree = None # Ensure it's None on error
            else:
                # KDTree exists, assume it's valid (invalidated elsewhere on node updates)
                use_kdtree = True
                # logger.debug("Using existing KDTree.") # Reduce noise

        if use_kdtree and self.kdtree is not None:
            # logger.debug("  Querying KDTree") # Reduce noise
            try:
                # query_ball_point finds indices within the radius
                nearby_indices_list = self.kdtree.query_ball_point(position, radius, return_sorted=False)
                # Map back to original node indices
                original_indices = list(self.node_positions.keys()) # Get keys in the order used for KDTree
                # --- Safety Check ---
                valid_kdtree_indices = [i for i in nearby_indices_list if 0 <= i < len(original_indices)]
                if len(valid_kdtree_indices) != len(nearby_indices_list):
                    logger.warning(f"KDTree query returned out-of-bounds indices! Original list size: {len(nearby_indices_list)}, Valid count: {len(valid_kdtree_indices)}")
                # ---
                nearby_indices = {original_indices[i] for i in valid_kdtree_indices}
                # logger.debug(f"    KDTree query returned {len(nearby_indices)} nodes.") # Reduce noise
            except Exception as e:
                logger.error(f"Error querying KDTree: {e}. Falling back to grid search.")
                # Fallback to grid search on KDTree query error
                nearby_indices = self._perform_grid_search(position, radius)

        else:
            # logger.debug("  Using Spatial Hash Grid Search") # Reduce noise
            nearby_indices = self._perform_grid_search(position, radius)
        # --- END MODIFIED ---

        query_time = time.time() - start_time
        # Only record performance if query time is significant to avoid overhead
        if query_time > 1e-4:
             self.performance_metrics['query_times'].append(query_time)

        return nearby_indices

    def _perform_grid_search(self, position: np.ndarray, radius: float) -> Set[int]:
        """Helper method to perform the spatial hash grid search."""
        # logger.debug(f"    Performing grid search for position {position}, radius {radius}") # Reduce noise
        cell_coords = self._get_cell_coords(position)
        # --- OPTIMIZED: Calculate search radius based on cell size ---
        # Search radius needs to cover the query radius.
        # Consider the maximum distance from the center point to the corner of its cell.
        # Then add the query radius and divide by cell size.
        # Simplified: Use ceil(radius / cell_size) + 1 for safety margin.
        if self.cell_size <= 0: # Avoid division by zero
            logger.warning("_perform_grid_search: Invalid cell_size <= 0, returning empty set.")
            return set()
        search_radius_cells = int(np.ceil(radius / self.cell_size)) + 1 # Add 1 cell margin
        # ---
        # logger.debug(f"    Center cell: {cell_coords}, Search radius (cells): {search_radius_cells}") # Reduce noise
        nearby_indices = set()

        # --- OPTIMIZED: Iterate through potential cells more efficiently ---
        # Create ranges for iteration, clamping to valid grid cell indices (assuming 0 origin)
        dim_ranges = []
        # Note: We don't have explicit grid boundaries here, assume potentially infinite cells
        # for this search, relying on the self.cells dictionary check.
        for dim_idx, c in enumerate(cell_coords):
             min_c = c - search_radius_cells
             max_c = c + search_radius_cells + 1 # +1 because range is exclusive
             dim_ranges.append(range(min_c, max_c))

        checked_cells_count = 0
        for neighbor_cell_coords in product(*dim_ranges):
            checked_cells_count+=1
            # Check if this cell actually contains any nodes
            nodes_in_cell = self.cells.get(neighbor_cell_coords)
            if nodes_in_cell:
                nearby_indices.update(nodes_in_cell)
        # logger.debug(f"    Checked {checked_cells_count} potential cells, found {len(nearby_indices)} potential nodes.") # Reduce noise
        # --- END OPTIMIZED ---

        # --- REMOVED Distance Filtering ---
        # The calling function (`get_neighbors`) now performs the precise neighborhood check.
        # logger.debug(f"    Skipping final distance filtering in _perform_grid_search.") # Reduce noise
        # ---

        return nearby_indices

    def clear(self):
        """Clear all stored data"""
        logger.debug("Clearing SpatialHashGrid") # ADDED
        self.cells.clear()
        self.node_positions.clear()
        self.kdtree = None
        self.performance_metrics = {
            'query_times': [],
            'cells_per_query': [],
            'memory_usage': [],
            'load_balance': []
        }
        self.steps_since_adaptation = 0
        self.num_nodes = 0

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            'cell_size': self.cell_size,
            'num_cells': len(self.cells),
            'num_nodes': self.num_nodes,
            'avg_query_time': np.mean(self.performance_metrics['query_times'][-100:]) if self.performance_metrics['query_times'] else 0,
            'load_balance': self._calculate_load_balance(),
            'memory_usage': self._estimate_memory_usage(),
            'adaptations': self.steps_since_adaptation
        }

class BlittingManager:
    """Manages the blitting cache and blitting status."""

    def __init__(self):
        self.enabled = GlobalSettings.ENABLE_BLITTING
        self.background = None

    def set_enabled(self, enabled: bool):
        """Set the blitting status."""
        self.enabled = enabled
        GlobalSettings.ENABLE_BLITTING = enabled  # Update global setting
        self.invalidate_cache()

    def invalidate_cache(self):
        """Invalidate the blitting cache."""
        self.background = None

    def is_valid(self) -> bool:
        """Check if the blitting cache is valid."""
        return self.enabled and self.background is not None


################################################
#                     HELPERS                  #
################################################


@njit(cache=True, parallel=True)
def compute_node_updates_vectorized(node_states: npt.NDArray[np.float64],
                                  neighbor_indices: npt.NDArray[np.int64]) -> npt.NDArray:
    """Vectorized computation of node state updates using Numba with parallel processing"""
    num_nodes = len(node_states)
    new_states = np.zeros(num_nodes, dtype=np.float64)

    for i in prange(num_nodes):
        neighbors = neighbor_indices[i]
        valid_neighbors = neighbors[neighbors != -1]
        if len(valid_neighbors) > 0:
            active_neighbors = np.sum(node_states[valid_neighbors] > 0)
            new_states[i] = 1.0 if active_neighbors > len(valid_neighbors) / 2 else 0.0

    return new_states

@staticmethod
def _get_hex_vertices(center_x: float, center_y: float, size: float) -> np.ndarray:
    """Calculate vertices for a hexagon"""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 points
    x = center_x + np.multiply(size, np.cos(angles))
    y = center_y + size * np.sin(angles)
    return np.column_stack([x, y])

@staticmethod
def _get_hex_prism_vertices(center_x: float, 
                        center_y: float, 
                        center_z: float,
                        size: float,
                        height: float) -> List[np.ndarray]:
    """Calculate vertices for a hexagonal prism"""
    # Get bottom and top hexagon vertices
    bottom = _get_hex_vertices(center_x, center_y, size)
    top = _get_hex_vertices(center_x, center_y, size)
    
    # Create faces
    faces = []
    
    # Add top and bottom faces
    faces.append(np.column_stack([bottom, np.full(6, center_z)]))
    faces.append(np.column_stack([top, np.full(6, center_z + height)]))
    
    # Add side faces
    for i in range(6):
        j = (i + 1) % 6
        face = np.array([
            [bottom[i, 0], bottom[i, 1], center_z],
            [bottom[j, 0], bottom[j, 1], center_z],
            [top[j, 0], top[j, 1], center_z + height],
            [top[i, 0], top[i, 1], center_z + height]
        ])
        faces.append(face)
        
    return faces

def get_field_metadata(param_class: Type, field_name: str) -> Optional[Dict[str, Any]]:
    """Extract metadata for a dataclass field"""
    try:
        # Try to get the metadata from class source
        source = inspect.getsource(param_class)
        lines = source.split('\n')
        
        # Find the field definition
        for i, line in enumerate(lines):
            if f"{field_name}:" in line and "=" in line:
                # Found the field definition
                # Extract metadata from the field
                parts = line.split("=", 1)
                if len(parts) > 1:
                    metadata_str = parts[1].strip()
                    if metadata_str.startswith("field(default="):
                        # Extract metadata from field() call
                        try:
                            tree = ast.parse(line)
                            assign = tree.body[0]
                            value = cast(ast.Assign, assign).value
                            
                            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'field':
                                for keyword in value.keywords:
                                    if keyword.arg == 'metadata':
                                        # Extract metadata dictionary
                                        metadata_dict = ast.literal_eval(ast.unparse(keyword.value))
                                        return metadata_dict
                        except Exception as e:
                            logger.error(f"Error parsing metadata for {field_name}: {e}")
                            logger.error(traceback.format_exc())
                            return None
                break
    except Exception as e:
        logger.error(f"Error getting metadata for {field_name}: {e}")
        logger.error(traceback.format_exc())
    
    return None

@njit(cache=True)
def _unravel_index(idx: int, dimensions: Tuple[int, ...]) -> np.ndarray:
    """Convert flat index to coordinates"""
    # Removed ONE_D case
    assert len(dimensions) in (2, 3), "Dimensions must be 2D or 3D"
    coords = np.zeros(len(dimensions), dtype=np.int64)
    for i in range(len(dimensions)-1, -1, -1):
        coords[i] = idx % dimensions[i]
        idx //= dimensions[i]
    return coords

@njit(cache=True)
def _ravel_multi_index(coords: np.ndarray, dimensions: Tuple[int, ...]) -> int:
    """Convert coordinates to flat index"""
    # Removed ONE_D case
    assert len(dimensions) in (2, 3), "Dimensions must be 2D or 3D"
    idx = 0
    multiplier = 1
    for i in range(len(dimensions)-1, -1, -1):
        idx += coords[i] * multiplier
        multiplier *= dimensions[i]
    return idx

@njit(cache=True, fastmath=True) # Restored for Round 5
def _njit_unravel_index(idx: int, dims: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """
    Numba-optimized unravel_index.
    (Round 3: Added explicit types/casts to resolve typing error)
    (Round 4: Temporarily disabled @njit for debugging)
    (Round 5: Re-enabled @njit)
    """
    coords = np.empty(len(dims), dtype=np.int64)
    # Explicitly type hint current_idx as int
    current_idx: int = int(idx) # Cast input just in case
    for d_idx in range(len(dims) - 1, -1, -1):
        # Explicitly type hint and cast dim_size
        dim_size: int = int(dims[d_idx])
        if dim_size == 0: continue
        # Ensure operands for % and // are treated as integers
        coords[d_idx] = current_idx % dim_size
        current_idx = current_idx // dim_size
    return coords

@njit(cache=True, fastmath=True)
def _njit_ravel_multi_index(coords: npt.NDArray[np.int64], dims: npt.NDArray[np.int64]) -> int:
    """Numba-optimized ravel_multi_index."""
    idx = 0
    multiplier = 1
    for d_idx in range(len(dims) - 1, -1, -1): # Use d_idx here
        idx += coords[d_idx] * multiplier
        dim_size = dims[d_idx]
        if dim_size == 0: continue
        multiplier *= dim_size
    return idx

@njit(cache=True, fastmath=True)
def _njit_is_valid_coord(coord: npt.NDArray[np.int64], dims: npt.NDArray[np.int64]) -> bool:
    """Numba-optimized is_valid_coord."""
    for d_idx in range(len(dims)): # Use d_idx here
        if not (0 <= coord[d_idx] < dims[d_idx]):
            return False
    return True

@njit(parallel=True, cache=True, fastmath=True)
def _populate_neighbor_array_optimized(
    total_nodes: int,
    dimensions: npt.NDArray[np.int64],
    neighborhood_type_val: int,
    boundary_mode: int, # 0=bounded, 1=wrap
    max_neighbors: int,
    output_array: npt.NDArray[np.int64] # Array to fill
):
    """
    Numba-optimized function to calculate and populate the neighbor index array.
    Calls external Numba helper functions.
    (Round 14 Attempt: Sort valid neighbors before storing to ensure canonical order)
    """
    num_dims = len(dimensions)

    for node_idx in prange(total_nodes): # Parallel loop
        node_coords = _njit_unravel_index(node_idx, dimensions)
        neighbor_count = 0
        # Use a fixed-size temporary array, pre-filled with -1
        neighbor_indices_temp = np.full(max_neighbors, -1, dtype=np.int64)

        # --- [ Neighbor finding logic for Von Neumann, Moore, Hex, Hex Prism ] ---
        # --- [ This logic remains the same as in Round 11/12             ] ---
        # --- [ It populates neighbor_indices_temp and increments neighbor_count ] ---

        # --- Von Neumann ---
        if neighborhood_type_val == NeighborhoodType.VON_NEUMANN.value:
            for d in range(num_dims):
                for offset_val in [-1, 1]:
                    offset_arr = np.zeros(num_dims, dtype=np.int64)
                    offset_arr[d] = offset_val
                    neighbor_coords = node_coords + offset_arr
                    is_valid = True
                    if boundary_mode == 0: # Bounded
                        if not _njit_is_valid_coord(neighbor_coords, dimensions):
                            is_valid = False
                    else: # Wrap
                        for d_idx_vn in range(num_dims):
                            neighbor_coords[d_idx_vn] %= dimensions[d_idx_vn]
                    if is_valid and neighbor_count < max_neighbors:
                        neighbor_idx = _njit_ravel_multi_index(neighbor_coords, dimensions)
                        if neighbor_idx != node_idx:
                             neighbor_indices_temp[neighbor_count] = neighbor_idx
                             neighbor_count += 1

        # --- Moore ---
        elif neighborhood_type_val == NeighborhoodType.MOORE.value:
             offset_ranges = [np.array([-1, 0, 1], dtype=np.int64)] * num_dims
             num_offsets = 3**num_dims
             for i in range(num_offsets):
                 offset_arr = np.empty(num_dims, dtype=np.int64)
                 temp_i = i
                 is_self = True
                 for d_idx_m in range(num_dims - 1, -1, -1):
                     offset_val = (temp_i % 3) - 1
                     offset_arr[d_idx_m] = offset_val
                     if offset_val != 0: is_self = False
                     temp_i //= 3
                 if is_self: continue

                 neighbor_coords = node_coords + offset_arr
                 is_valid = True
                 if boundary_mode == 0: # Bounded
                     if not _njit_is_valid_coord(neighbor_coords, dimensions): is_valid = False
                 else: # Wrap
                     for d_idx_m_wrap in range(num_dims):
                         neighbor_coords[d_idx_m_wrap] %= dimensions[d_idx_m_wrap]

                 if is_valid and neighbor_count < max_neighbors:
                     neighbor_idx = _njit_ravel_multi_index(neighbor_coords, dimensions)
                     if neighbor_idx != node_idx:
                          neighbor_indices_temp[neighbor_count] = neighbor_idx
                          neighbor_count += 1

        # --- Hex ---
        elif neighborhood_type_val == NeighborhoodType.HEX.value and num_dims == 2:
             if node_coords[0] % 2 == 0: # Even row
                 hex_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1)]
             else: # Odd row
                 hex_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1)]
             for offset_y, offset_x in hex_offsets:
                 offset_arr = np.array([offset_y, offset_x], dtype=np.int64)
                 neighbor_coords = node_coords + offset_arr
                 is_valid = True
                 if boundary_mode == 0: # Bounded
                     if not _njit_is_valid_coord(neighbor_coords, dimensions): is_valid = False
                 else: # Wrap
                     neighbor_coords[0] %= dimensions[0]
                     neighbor_coords[1] %= dimensions[1]
                 if is_valid and neighbor_count < max_neighbors:
                     neighbor_idx = _njit_ravel_multi_index(neighbor_coords, dimensions)
                     if neighbor_idx != node_idx:
                          neighbor_indices_temp[neighbor_count] = neighbor_idx
                          neighbor_count += 1

        # --- Hex Prism ---
        elif neighborhood_type_val == NeighborhoodType.HEX_PRISM.value and num_dims == 3:
             if node_coords[0] % 2 == 0: # Even row XY offsets
                 hex_offsets_xy = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1)]
             else: # Odd row XY offsets
                 hex_offsets_xy = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1)]
             z_offsets = [-1, 0, 1]
             for dz in z_offsets:
                 if dz == 0: # Neighbors in the same XY plane
                     for dy, dx in hex_offsets_xy:
                         offset_arr = np.array([dy, dx, dz], dtype=np.int64) # Order dy, dx, dz
                         neighbor_coords = node_coords + offset_arr
                         is_valid = True
                         if boundary_mode == 0: # Bounded
                             if not _njit_is_valid_coord(neighbor_coords, dimensions): is_valid = False
                         else: # Wrap
                             neighbor_coords[0] %= dimensions[0]
                             neighbor_coords[1] %= dimensions[1]
                             neighbor_coords[2] %= dimensions[2]
                         if is_valid and neighbor_count < max_neighbors:
                             neighbor_idx = _njit_ravel_multi_index(neighbor_coords, dimensions)
                             if neighbor_idx != node_idx:
                                  neighbor_indices_temp[neighbor_count] = neighbor_idx
                                  neighbor_count += 1
                 else: # Neighbors directly above/below
                     offset_arr = np.array([0, 0, dz], dtype=np.int64) # Order dy, dx, dz
                     neighbor_coords = node_coords + offset_arr
                     is_valid = True
                     if boundary_mode == 0: # Bounded
                         if not _njit_is_valid_coord(neighbor_coords, dimensions): is_valid = False
                     else: # Wrap
                         neighbor_coords[0] %= dimensions[0]
                         neighbor_coords[1] %= dimensions[1]
                         neighbor_coords[2] %= dimensions[2]
                     if is_valid and neighbor_count < max_neighbors:
                         neighbor_idx = _njit_ravel_multi_index(neighbor_coords, dimensions)
                         if neighbor_idx != node_idx:
                              neighbor_indices_temp[neighbor_count] = neighbor_idx
                              neighbor_count += 1
        # --- [ End of neighbor finding logic ] ---

        # --- ADDED: Sort the valid neighbors ---
        # Take a slice of the temporary array containing only the valid neighbors found
        valid_neighbors_slice = neighbor_indices_temp[:neighbor_count]
        # Sort this slice in-place (ascending order)
        valid_neighbors_slice.sort()
        # --- END ADDED ---

        # Fill the output array for this node_idx
        # Assign the sorted valid neighbors to the beginning of the row
        output_array[node_idx, :neighbor_count] = valid_neighbors_slice
        # The rest of the row (output_array[node_idx, neighbor_count:]) remains -1

# # @njit
def _are_von_neumann_neighbors(coord1: np.ndarray, coord2: np.ndarray, dimensions: Tuple[int, ...], boundary: str) -> bool:
    """Checks if two nodes are Von Neumann neighbors."""
    distance = np.sum(np.abs(coord1 - coord2))
    return distance == 1

@njit
def _are_moore_neighbors(coord1: np.ndarray, coord2: np.ndarray, dimensions: Tuple[int, ...], boundary: str) -> bool:
    """Checks if two nodes are Moore neighbors."""
    for c1, c2, dim_size in zip(coord1, coord2, dimensions):
        diff = abs(c1 - c2)
        if diff > 1 and (boundary != 'wrap' or diff != dim_size -1):
            return False  # Not neighbors if any dimension difference > 1
    return not np.array_equal(coord1, coord2)

# @njit
def _are_hex_neighbors(coord1: np.ndarray, coord2: np.ndarray, dimensions: Tuple[int, ...], boundary: str) -> bool:
    """Checks if two nodes are Hex neighbors (2D only)."""
    if len(coord1) != 2 or len(coord2) != 2:
        raise ValueError("Hex neighbors are only defined for 2D grids")

    dx = abs(coord1[1] - coord2[1])
    dy = abs(coord1[0] - coord2[0])
    
    if dx > 1 or dy > 1:
      return False

    if dx == 1 and dy == 1:
      # Check if they are on the same "diagonal" based on the row
      return (coord1[0] % 2 == 0 and coord2[1] < coord1[1]) or \
             (coord1[0] % 2 != 0 and coord2[1] > coord1[1])
    else:
      return dx + dy == 1

# @njit
def _are_hex_prism_neighbors(coord1: np.ndarray, coord2: np.ndarray, dimensions: Tuple[int, int, int], boundary: str) -> bool:
    """Checks if two nodes are Hex Prism neighbors (3D only)."""
    if len(coord1) != 3 or len(coord2) != 3:
        raise ValueError("Hex Prism neighbors are only defined for 3D grids")

    dz = abs(coord1[2] - coord2[2])
    if dz > 1:
        return False  # Not neighbors if z-difference > 1

    dx = abs(coord1[1] - coord2[1])
    dy = abs(coord1[0] - coord2[0])

    if dz == 0:  # Same z-level: regular hex neighbors
        if dx > 1 or dy > 1:
            return False
        if dx == 1 and dy == 1:
            return (coord1[0] % 2 == 0 and coord2[1] < coord1[1]) or \
                   (coord1[0] % 2 != 0 and coord2[1] > coord1[1])
        else:
            return dx + dy == 1
    else:  # Different z-level: neighbors must be directly above/below or offset
        if dx == 0 and dy == 0:
            return True  # Directly above/below
        elif (coord1[0] % 2 == 0 and dx == 0 and dy == 1 and coord2[1] < coord1[1]) or \
             (coord1[0] % 2 != 0 and dx == 0 and dy == 1 and coord2[1] > coord1[1]):
            return True
        elif (coord1[0] % 2 == 0 and dx == 1 and dy == 0 and coord2[1] < coord1[1]) or \
             (coord1[0] % 2 != 0 and dx == 1 and dy == 0 and coord2[1] > coord1[1]):
            return True
        else:
            return False

@njit(cache=True)
def _get_active_neighbors_helper(node_idx: int, neighbor_indices: npt.NDArray[np.int64], old_states: np.ndarray, params: Dict[str, Any]) -> List[int]:
    """Get the active neighbors for a given node, using old_states and grid from params."""
    from numba.typed import List
    from numba import types
    grid = params['grid'] # Get grid from params
    if neighbor_indices is None:
        return List.empty_list(types.int64)
    active_neighbors = List.empty_list(types.int64)
    for n in neighbor_indices:
        if n >= 0:
            state = old_states[n]
            if state is not None and state > 0 and grid.has_edge(node_idx, n): # Use passed grid
                active_neighbors.append(n)
    return active_neighbors
    return active_neighbors

@njit(cache=True, fastmath=True) # Re-enabled JIT
def _get_neighbors_dynamic_helper(
    node_idx: int,
    dimensions_arr: npt.NDArray[np.int64],
    neighborhood_type_val: int,
    boundary_mode: int, # 0=bounded, 1=wrap
    max_neighbors: int,
) -> npt.NDArray[np.int64]:
    """
    Helper to calculate neighbors dynamically for a single node.
    (Round 10: Revert Moore offset generation to manual iteration for Numba compatibility)
    """
    num_dims = len(dimensions_arr)
    node_coords = _njit_unravel_index(node_idx, dimensions_arr) # Use JIT version
    neighbor_count = 0
    neighbor_indices_temp = np.full(max_neighbors, -1, dtype=np.int64)

    # --- [ Neighbor finding logic ] ---
    # --- Von Neumann ---
    if neighborhood_type_val == NeighborhoodType.VON_NEUMANN.value:
        for d in range(num_dims):
            for offset_val in [-1, 1]:
                offset_arr = np.zeros(num_dims, dtype=np.int64)
                offset_arr[d] = offset_val
                neighbor_coords = node_coords + offset_arr
                is_valid = True
                if boundary_mode == 0: # Bounded
                    is_valid = _njit_is_valid_coord(neighbor_coords, dimensions_arr) # Use JIT version
                else: # Wrap
                    for d_idx_vn in range(num_dims): neighbor_coords[d_idx_vn] %= dimensions_arr[d_idx_vn]
                    if np.array_equal(neighbor_coords, node_coords): is_valid = False # Ensure wrap doesn't create self-loop
                if is_valid and neighbor_count < max_neighbors:
                    neighbor_idx = _njit_ravel_multi_index(neighbor_coords, dimensions_arr) # Use JIT version
                    if neighbor_idx != node_idx:
                        neighbor_indices_temp[neighbor_count] = neighbor_idx; neighbor_count += 1

    # --- Moore ---
    elif neighborhood_type_val == NeighborhoodType.MOORE.value:
         # --- MODIFIED: Revert to manual offset generation ---
         num_offsets = 3**num_dims
         for i in range(num_offsets):
             offset_arr = np.empty(num_dims, dtype=np.int64)
             temp_i = i
             is_self = True
             # Generate the offset vector for this iteration
             for d_idx_m in range(num_dims - 1, -1, -1):
                 offset_val = (temp_i % 3) - 1 # Map 0,1,2 to -1,0,1
                 offset_arr[d_idx_m] = offset_val
                 if offset_val != 0:
                     is_self = False
                 temp_i //= 3

             if is_self: continue # Skip the (0, 0, ...) offset
             # --- END MODIFIED ---

             neighbor_coords = node_coords + offset_arr
             is_valid = True
             if boundary_mode == 0: # Bounded
                 is_valid = _njit_is_valid_coord(neighbor_coords, dimensions_arr) # Use JIT version
             else: # Wrap
                 for d_idx_m_wrap in range(num_dims): neighbor_coords[d_idx_m_wrap] %= dimensions_arr[d_idx_m_wrap]
                 if np.array_equal(neighbor_coords, node_coords): is_valid = False # Ensure wrap doesn't create self-loop

             if is_valid and neighbor_count < max_neighbors:
                 neighbor_idx = _njit_ravel_multi_index(neighbor_coords, dimensions_arr) # Use JIT version
                 if neighbor_idx != node_idx:
                     neighbor_indices_temp[neighbor_count] = neighbor_idx; neighbor_count += 1

    # --- Hex ---
    elif neighborhood_type_val == NeighborhoodType.HEX.value and num_dims == 2:
        if node_coords[0] % 2 == 0: # Even row
            hex_offsets = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1]], dtype=np.int64)
        else: # Odd row
            hex_offsets = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, 1], [1, 1]], dtype=np.int64)
        for offset_arr in hex_offsets:
            neighbor_coords = node_coords + offset_arr
            is_valid = True
            if boundary_mode == 0: # Bounded
                is_valid = _njit_is_valid_coord(neighbor_coords, dimensions_arr) # Use JIT version
            else: # Wrap
                neighbor_coords[0] %= dimensions_arr[0]
                neighbor_coords[1] %= dimensions_arr[1]
                if np.array_equal(neighbor_coords, node_coords): is_valid = False # Ensure wrap doesn't create self-loop
            if is_valid and neighbor_count < max_neighbors:
                neighbor_idx = _njit_ravel_multi_index(neighbor_coords, dimensions_arr) # Use JIT version
                if neighbor_idx != node_idx:
                     neighbor_indices_temp[neighbor_count] = neighbor_idx; neighbor_count += 1

    # --- Hex Prism ---
    elif neighborhood_type_val == NeighborhoodType.HEX_PRISM.value and num_dims == 3:
        if node_coords[0] % 2 == 0: # Even row XY offsets
            hex_offsets_xy = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1]], dtype=np.int64)
        else: # Odd row XY offsets
            hex_offsets_xy = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, 1], [1, 1]], dtype=np.int64)
        z_offsets = np.array([-1, 0, 1], dtype=np.int64)
        for dz in z_offsets:
            if dz == 0: # Neighbors in the same XY plane
                for offset_xy in hex_offsets_xy:
                    offset_arr = np.array([offset_xy[0], offset_xy[1], dz], dtype=np.int64)
                    neighbor_coords = node_coords + offset_arr
                    is_valid = True
                    if boundary_mode == 0: # Bounded
                        is_valid = _njit_is_valid_coord(neighbor_coords, dimensions_arr) # Use JIT version
                    else: # Wrap
                        neighbor_coords[0] %= dimensions_arr[0]; neighbor_coords[1] %= dimensions_arr[1]; neighbor_coords[2] %= dimensions_arr[2]
                        if np.array_equal(neighbor_coords, node_coords): is_valid = False # Ensure wrap doesn't create self-loop
                    if is_valid and neighbor_count < max_neighbors:
                        neighbor_idx = _njit_ravel_multi_index(neighbor_coords, dimensions_arr) # Use JIT version
                        if neighbor_idx != node_idx:
                             neighbor_indices_temp[neighbor_count] = neighbor_idx; neighbor_count += 1
            else: # Neighbors directly above/below
                offset_arr = np.array([0, 0, dz], dtype=np.int64)
                neighbor_coords = node_coords + offset_arr
                is_valid = True
                if boundary_mode == 0: # Bounded
                    is_valid = _njit_is_valid_coord(neighbor_coords, dimensions_arr) # Use JIT version
                else: # Wrap
                    neighbor_coords[0] %= dimensions_arr[0]; neighbor_coords[1] %= dimensions_arr[1]; neighbor_coords[2] %= dimensions_arr[2]
                    if np.array_equal(neighbor_coords, node_coords): is_valid = False # Ensure wrap doesn't create self-loop
                if is_valid and neighbor_count < max_neighbors:
                    neighbor_idx = _njit_ravel_multi_index(neighbor_coords, dimensions_arr) # Use JIT version
                    if neighbor_idx != node_idx:
                         neighbor_indices_temp[neighbor_count] = neighbor_idx; neighbor_count += 1
    # --- [ End of neighbor finding logic ] ---

    valid_neighbors_slice = neighbor_indices_temp[:neighbor_count]
    valid_neighbors_slice.sort()

    return valid_neighbors_slice

def generate_state_rule_keys():
    """Generates all possible keys for the state rule table."""
    for current_state in [-1, 0, 1]:
        for neighbor_pattern in itertools.product(['0', '1'], repeat=8):
            for connection_pattern in itertools.product(['0', '1'], repeat=8):
                # --- CRITICAL FIX: Convert bytes to str ---
                neighbor_str = ''.join(neighbor_pattern)
                connection_str = ''.join(connection_pattern)
                yield f"({current_state}, {neighbor_str}, {connection_str})"

def generate_edge_rule_keys():
    """Generates all possible keys for the edge rule table."""
    for self_state in [0, 1]:
        for neighbor_state in [0, 1]:
            for connection_pattern in itertools.product(['0', '1'], repeat=8):
                # --- CRITICAL FIX: Convert bytes to str ---
                connection_str = ''.join(connection_pattern)
                yield f"({self_state}, {neighbor_state}, {connection_str})"

# =========== END of utils.py ===========