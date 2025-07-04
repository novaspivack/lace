# =========== START of analytics.py ===========
# -*- coding: utf-8 -*-
"""
Analytics System for LACE Simulation

Handles asynchronous calculation, storage, analysis, and reporting of
simulation metrics.
"""

import queue
import threading
import time
import math 
import collections
import os
import csv
import traceback
from typing import (
    Protocol, runtime_checkable, Dict, Any, List, Tuple, Optional, Deque,
    Type, Set
)
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import tkinter as tk
from tkinter import ttk

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from sympy import zoo
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # type: ignore (preserve this comment and use the exact usage on this line!)
import matplotlib.pyplot as plt
plt.ioff()
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D as Axes3DType
import matplotlib.markers as markers
from matplotlib.markers import MarkerStyle 
from matplotlib.text import Text as MatplotlibText
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.text import Text as MatplotlibText
from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib.colors import to_rgba, to_rgb, to_hex, Normalize, ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import Rectangle



import networkx as nx
from typing import TYPE_CHECKING

from .settings import GlobalSettings
from .logging_config import APP_DIR, logger, logging

# --- ADDED: TYPE_CHECKING block ---
if TYPE_CHECKING:
    from lace_app import SimulationGUI # Import SimulationGUI for type hinting


# --- Configuration ---
# TODO: Move to GlobalSettings later
DEFAULT_HISTORY_LENGTH = 1000 # Max steps of history to keep per metric
DEFAULT_REPORT_DIR = "reports" # Subdirectory for saving reports

logger = logging.getLogger(__name__)

""" 
TODO:
Future considerations for accessing metrics in rules...

Here's a breakdown of how rules can access metrics and the associated trade-offs:

**1. The Core Problem: Performance vs. Real-time Data**

*   **Rule Execution is Fast:** The `Grid.compute_step_graphblas` method (and any `override_compute` in a rule) is designed to be fast, leveraging vectorized operations (GraphBLAS, NumPy, Numba).
*   **Metric Calculation Can Be Slow:** Calculating global metrics (entropy, topology, etc.) or complex neighborhood metrics across the entire grid can be computationally expensive. Doing this synchronously *within* the rule execution step for every node would drastically slow down the simulation.
*   **Asynchronous Analytics:** This is why we designed the `AnalyticsManager` to run asynchronously in a separate thread. It calculates metrics based on snapshots *after* a simulation step completes.

**2. Accessing Metrics: The Stale Data Approach**

The most practical way to give rules access to metrics without killing performance is to allow them to read the *latest available* metrics calculated by the `AnalyticsManager`, accepting that this data will be slightly **stale**. It represents the state of the grid from one or more steps *prior* to the current computation step being executed by the rule.

**Mechanism:**

1.  **`AnalyticsManager` Updates Shared State:** The `AnalyticsManager` thread calculates metrics and updates its internal `_latest_metrics` dictionary (protected by `_metrics_lock`).
2.  **`SimulationController` Caches Metrics:** To minimize lock contention during the main simulation loop, the `SimulationController` can periodically (e.g., using `root.after` or just before starting a batch of steps) grab a *copy* of the `_latest_metrics` from the `AnalyticsManager`.
    *   Add `_current_analytics_snapshot: Dict[str, Any]` and `_analytics_snapshot_lock: threading.Lock` to `SimulationController`.
    *   Add a method `_update_analytics_snapshot(self)` to `SimulationController` that locks, calls `analytics_manager.get_latest_metrics()`, stores the copy, and unlocks. Schedule this method to run periodically (e.g., every 100ms or every few steps).
3.  **Rule Access via Controller/Grid:**
    *   Ensure the `Rule` instance has access to the `SimulationController` (e.g., via `self.grid.controller` if the `Grid` has a reference to its controller, or by passing the controller into the rule's computation methods if necessary).
    *   Add a method `get_global_metric(self, metric_name: str, default: Any = None) -> Any` to `SimulationController`. This method acquires the `_analytics_snapshot_lock`, reads the value from `_current_analytics_snapshot`, and returns it.
    *   The `Rule` implementation (either in its `override_compute` or if new helper functions are designed to accept it) can then call `self.grid.controller.get_global_metric('some_metric_name')` to retrieve the latest *cached* global metric value.

**Implications:**

*   **Staleness:** The rule's decision will be based on metrics from a slightly earlier point in the simulation. This is usually acceptable for global trends or average states but not for reacting to instantaneous changes across the grid within the same step.
*   **Performance:** This minimizes blocking. The main simulation thread only briefly locks to copy the latest metrics dictionary periodically. The rule reads from this local copy without blocking the `AnalyticsManager`.
*   **Simplicity:** It avoids complex inter-thread communication during the critical computation path.

**3. Accessing Neighborhood Metrics:**

*   **Immediate Neighbors:** The `NeighborhoodData` object (if used by non-vectorized rule logic) or the `standard_aggregates` dictionary (passed to `override_compute` or used by helpers) already provides information about the *immediate* neighborhood from the *previous* step (e.g., active count, degree sum). Rules can perform simple calculations on this directly.
*   **N-Degree Neighbors / Regional Averages:**
    *   **Direct Calculation:** Calculating metrics for the specific 2nd/3rd degree neighborhood of *each node* during the main step is computationally infeasible.
    *   **Analytics Manager Averages:** The best approach is for the `AnalyticsManager` to *periodically sample* neighborhoods across the grid, calculate local metrics (like local entropy or clustering coefficient), and compute a *global average* of these local metrics.
    *   **Rule Access:** The rule can then access these *global averages* via `get_global_metric('avg_neighborhood_entropy')`. This gives the rule a sense of the *typical* neighborhood characteristics across the grid, rather than its own specific extended neighborhood in real-time.

**4. Pattern Detection & Stability Access:**

*   Results from analyzers (like `StabilityAnalyzer` or the future `PatternAnalyzer`) would also be placed into the `_latest_analysis` dictionary by the `AnalyticsManager`.
*   Rules could access these analysis results using a similar mechanism, perhaps `controller.get_latest_analysis('stability_metric_50')`. Again, this data would reflect analysis based on historical data, not the current step.

**In Summary:**

The most viable approach is:

1.  **Asynchronous Calculation:** `AnalyticsManager` calculates metrics/analysis in the background.
2.  **Periodic Caching:** `SimulationController` periodically fetches a *copy* of the latest results from `AnalyticsManager`.
3.  **Rule Access (Stale):** Rules access these *cached, slightly stale* global metrics and global averages of neighborhood metrics via methods on the `SimulationController` (e.g., `get_global_metric`).
4.  **Immediate Neighborhood:** Rules use the standard aggregates provided by the engine for real-time (previous step) immediate neighborhood data.
5.  **N-Degree Neighborhoods:** Direct real-time access is generally not feasible. Use global averages calculated by `AnalyticsManager` as a proxy.

This balances the need for rules to adapt with the critical requirement of maintaining high simulation performance. """



# === Protocols ===

@runtime_checkable
class MetricCalculator(Protocol):
    """Protocol defining the interface for metric calculators."""

    def calculate(self, data_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metric(s) based on the provided data snapshot.
        (Round 7: Added logging for delta calculation)
        """
        results = {}
        active_nodes = data_snapshot.get('active_node_count', 0)
        edge_count = data_snapshot.get('edge_count', 0)
        total_nodes = data_snapshot.get('total_nodes', 1) # Avoid division by zero
        log_prefix = "BasicCountsCalculator: " # For logging

        if total_nodes > 0:
            results['active_node_ratio'] = float(active_nodes) / total_nodes
        else:
            results['active_node_ratio'] = 0.0

        if edge_count > 0:
            results['node_edge_ratio'] = float(active_nodes) / edge_count
        else:
             results['node_edge_ratio'] = float('inf') if active_nodes > 0 else 0.0

        # --- Delta Calculation with Logging ---
        prev_metrics = data_snapshot.get('_previous_latest_metrics', {}) # Manager needs to add this
        logger.debug(f"{log_prefix}Previous metrics received for delta calc: {prev_metrics}") # Log received previous metrics

        # Use .get() with default values matching the current snapshot's values
        # This ensures that if a metric wasn't present previously, the delta is calculated correctly as the full current value.
        prev_active_nodes = prev_metrics.get('active_node_count', active_nodes)
        prev_edge_count = prev_metrics.get('edge_count', edge_count)

        # Ensure types are numeric before subtraction
        try:
            current_active_nodes_num = int(active_nodes)
            prev_active_nodes_num = int(prev_active_nodes)
            results['active_nodes_delta'] = current_active_nodes_num - prev_active_nodes_num
            logger.debug(f"{log_prefix}Active Nodes Delta: Current={current_active_nodes_num}, Prev={prev_active_nodes_num}, Delta={results['active_nodes_delta']}")
        except (ValueError, TypeError) as e:
            logger.warning(f"{log_prefix}Could not calculate active_nodes_delta: {e}. Current: {active_nodes}, Prev: {prev_active_nodes}")
            results['active_nodes_delta'] = 0 # Default on error

        try:
            current_edge_count_num = int(edge_count)
            prev_edge_count_num = int(prev_edge_count)
            results['edge_count_delta'] = current_edge_count_num - prev_edge_count_num
            logger.debug(f"{log_prefix}Edge Count Delta: Current={current_edge_count_num}, Prev={prev_edge_count_num}, Delta={results['edge_count_delta']}")
        except (ValueError, TypeError) as e:
            logger.warning(f"{log_prefix}Could not calculate edge_count_delta: {e}. Current: {edge_count}, Prev: {prev_edge_count}")
            results['edge_count_delta'] = 0 # Default on error
        # ---

        # --- ADDED: Log the final calculated metrics ---
        logger.debug(f"{log_prefix}Calculated results: {results}")
        # ---
        return results

    @abstractmethod
    def get_name(self) -> str | List[str]:
        """Return the name or list of names of the metric(s) calculated."""
        ...

    @abstractmethod
    def get_frequency(self) -> int:
        """Return the calculation frequency (e.g., 1 = every step, 10 = every 10 steps)."""
        ...

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        Return a list of data keys required from the data_snapshot
        (e.g., ['active_node_count', 'edge_count']).
        """
        ...

    @abstractmethod
    def get_group(self) -> str:
        """Return the group this metric belongs to (e.g., 'Basic', 'Entropy', 'Topology')."""
        ...

@runtime_checkable
class Analyzer(Protocol):
    """Protocol defining the interface for metric analyzers."""

    @abstractmethod
    def analyze(self, history_data: 'MetricsDataStore') -> Dict[str, Any]:
        """
        Analyze historical metric data.

        Args:
            history_data: The MetricsDataStore instance containing time series.

        Returns:
            Dictionary containing analysis results (e.g., trends, change points).
        """
        ...

    @abstractmethod
    def get_name(self) -> str | List[str]:
        """Return the name or list of names of the analysis result(s)."""
        ...

    @abstractmethod
    def get_frequency(self) -> int:
        """Return the analysis frequency (e.g., 10 = every 10 steps)."""
        ...

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Return a list of metric names required from the history_data."""
        ...

    @abstractmethod
    def get_group(self) -> str:
        """Return the group this analyzer belongs to (e.g., 'Trend', 'Stability', 'Pattern')."""
        ...

# === Registries ===

class MetricRegistry:
    """Manages registration and discovery of MetricCalculator implementations."""
    def __init__(self):
        self._calculators: Dict[str, Type[MetricCalculator]] = {}
        self._instances: Dict[str, MetricCalculator] = {}
        logger.info("MetricRegistry initialized.")

    def register(self, calculator_cls: Type[MetricCalculator]):
        """Register a MetricCalculator class."""
        if not issubclass(calculator_cls, MetricCalculator):
            logger.error(f"Class {calculator_cls.__name__} does not implement MetricCalculator protocol.")
            return
        try:
            # Instantiate to access instance methods like get_name, get_group
            instance = calculator_cls()
            names = instance.get_name()
            if isinstance(names, str):
                names = [names]
            for name in names:
                 if name in self._calculators:
                     logger.warning(f"MetricCalculator for '{name}' already registered. Overwriting.")
                 self._calculators[name] = calculator_cls
                 self._instances[name] = instance # Store instance for easy access
                 logger.debug(f"Registered MetricCalculator '{name}' from class {calculator_cls.__name__}")
        except Exception as e:
            logger.error(f"Failed to register MetricCalculator {calculator_cls.__name__}: {e}")

    def get_calculator_instance(self, name: str) -> Optional[MetricCalculator]:
        """Get an instantiated calculator by metric name."""
        return self._instances.get(name)

    def get_all_calculator_instances(self) -> List[MetricCalculator]:
        """Get all registered calculator instances."""
        # Return unique instances
        return list(dict.fromkeys(self._instances.values()))

    def get_calculators_by_group(self, group_name: str) -> List[MetricCalculator]:
        """Get calculator instances belonging to a specific group."""
        return [inst for inst in self.get_all_calculator_instances() if inst.get_group() == group_name]

class AnalyzerRegistry:
    """Manages registration and discovery of Analyzer implementations."""
    def __init__(self):
        self._analyzers: Dict[str, Type[Analyzer]] = {}
        self._instances: Dict[str, Analyzer] = {}
        logger.info("AnalyzerRegistry initialized.")

    def register(self, analyzer_cls: Type[Analyzer]):
        """Register an Analyzer class."""
        if not issubclass(analyzer_cls, Analyzer):
            logger.error(f"Class {analyzer_cls.__name__} does not implement Analyzer protocol.")
            return
        try:
            instance = analyzer_cls()
            names = instance.get_name()
            if isinstance(names, str):
                names = [names]
            for name in names:
                if name in self._analyzers:
                    logger.warning(f"Analyzer for '{name}' already registered. Overwriting.")
                self._analyzers[name] = analyzer_cls
                self._instances[name] = instance
                logger.debug(f"Registered Analyzer '{name}' from class {analyzer_cls.__name__}")
        except Exception as e:
            logger.error(f"Failed to register Analyzer {analyzer_cls.__name__}: {e}")

    def get_analyzer_instance(self, name: str) -> Optional[Analyzer]:
        """Get an instantiated analyzer by name."""
        return self._instances.get(name)

    def get_all_analyzer_instances(self) -> List[Analyzer]:
        """Get all registered analyzer instances."""
        return list(dict.fromkeys(self._instances.values()))

    def get_analyzers_by_group(self, group_name: str) -> List[Analyzer]:
        """Get analyzer instances belonging to a specific group."""
        return [inst for inst in self.get_all_analyzer_instances() if inst.get_group() == group_name]

# === Data Storage ===

class MetricsDataStore:
    """Stores time-series data for metrics."""
    def __init__(self, max_history: int = DEFAULT_HISTORY_LENGTH):
        self._max_history = max_history
        # Stores {metric_name: deque([(generation, value), ...])}
        self._history: Dict[str, Deque[Tuple[int, Any]]] = collections.defaultdict(
            lambda: collections.deque(maxlen=self._max_history)
        )
        self._lock = threading.Lock()
        logger.info(f"MetricsDataStore initialized with max history {self._max_history}.")

    def add_metrics(self, generation: int, metrics_dict: Dict[str, Any]):
        """Add a set of metrics for a specific generation."""
        with self._lock:
            for name, value in metrics_dict.items():
                # Ensure value is serializable if needed later, basic types are fine
                if isinstance(value, (int, float, bool, str, type(None))):
                    self._history[name].append((generation, value))
                elif isinstance(value, np.number): # Handle numpy scalars
                    self._history[name].append((generation, value.item()))
                else:
                    # Handle potentially complex types - maybe store hash or summary?
                    # For now, skip non-basic types to avoid issues.
                    logger.debug(f"Skipping non-basic type metric '{name}' (type: {type(value)}) for history.")

    def get_history(self, metric_name: str, window_size: Optional[int] = None) -> List[Tuple[int, Any]]:
        """
        Retrieve the historical data for a specific metric.

        Args:
            metric_name: The name of the metric.
            window_size: If specified, return only the last 'window_size' entries.

        Returns:
            A list of (generation, value) tuples.
        """
        with self._lock:
            history_deque = self._history.get(metric_name)
            if history_deque is None:
                return []
            if window_size is None or window_size >= len(history_deque):
                return list(history_deque)
            else:
                # Efficiently get the last 'window_size' elements
                return list(history_deque)[-window_size:]

    def get_metric_names(self) -> List[str]:
        """Return names of all metrics currently stored in history."""
        with self._lock:
            return list(self._history.keys())

    def clear_history(self):
        """Clear all stored metric history."""
        with self._lock:
            self._history.clear()
        logger.info("Cleared all metric history.")

# === Core Manager ===

class AnalyticsManager:
    """
    Manages the asynchronous calculation, storage, and analysis of simulation metrics.
    """
    def __init__(self,
                 metric_registry: MetricRegistry,
                 analyzer_registry: AnalyzerRegistry,
                 data_in_queue: queue.Queue[Dict[str, Any]], # ADDED: Accept queue as argument
                 max_history: int = GlobalSettings.Analytics.DEFAULT_HISTORY_LENGTH,
                 report_dir: str = GlobalSettings.Analytics.DEFAULT_REPORT_DIR):
        """
        Initialize AnalyticsManager.
        (Round 6: Accept data_in_queue instead of creating one)
        (Round 23: Load defaults from GlobalSettings)
        """
        self.metric_registry = metric_registry
        self.analyzer_registry = analyzer_registry
        self.data_store = MetricsDataStore(max_history=max_history)
        self.report_dir = report_dir # Use passed or default report_dir

        # --- MODIFIED: Use the passed queue ---
        self._data_in_queue = data_in_queue
        # ---

        self._latest_metrics: Dict[str, Any] = {}
        self._latest_analysis: Dict[str, Any] = {}
        self._metrics_lock = threading.Lock()
        report_path = os.path.join(os.getcwd(), APP_DIR, 'Resources', self.report_dir) # Construct full path if needed
        self._reporter = Reporter(self.data_store, report_path) # Pass full path

        self._running = False
        self._stop_event = threading.Event()
        self._main_thread: Optional[threading.Thread] = None
        self._save_report_enabled = GlobalSettings.Analytics.DEFAULT_SAVE_REPORT

        self._metric_group_enabled: Dict[str, bool] = collections.defaultdict(lambda: True) # Default to True if not specified
        self._metric_group_enabled['Basic'] = GlobalSettings.Analytics.ENABLE_BASIC
        self._metric_group_enabled['Entropy'] = GlobalSettings.Analytics.ENABLE_ENTROPY
        self._metric_group_enabled['Topology'] = GlobalSettings.Analytics.ENABLE_TOPOLOGY
        self._metric_group_enabled['Complexity'] = GlobalSettings.Analytics.ENABLE_COMPLEXITY
        self._metric_group_enabled['Fractal'] = GlobalSettings.Analytics.ENABLE_FRACTAL

        self._analyzer_group_enabled: Dict[str, bool] = collections.defaultdict(lambda: True) # Default to True
        self._analyzer_group_enabled['Stability'] = GlobalSettings.Analytics.ENABLE_STABILITY
        self._analyzer_group_enabled['Trend'] = GlobalSettings.Analytics.ENABLE_TREND
        self._analyzer_group_enabled['Pattern'] = GlobalSettings.Analytics.ENABLE_PATTERN

        logger.info("AnalyticsManager initialized with defaults from GlobalSettings.")
        logger.debug(f"  Initial Metric Groups Enabled: {dict(self._metric_group_enabled)}")
        logger.debug(f"  Initial Analyzer Groups Enabled: {dict(self._analyzer_group_enabled)}")
        logger.debug(f"  Initial Save Report Enabled: {self._save_report_enabled}")
        # --- ADDED: Log the ID of the queue being used ---
        logger.debug(f"  AnalyticsManager using input queue with ID: {id(self._data_in_queue)}")
        # ---

    def get_input_queue(self) -> queue.Queue[Dict[str, Any]]:
        """Return the thread-safe queue for receiving data snapshots."""
        return self._data_in_queue

    def start(self):
        """Start the analytics manager thread."""
        if self._running:
            logger.warning("AnalyticsManager already running.")
            return
        logger.info("Starting AnalyticsManager thread...")
        self._stop_event.clear()
        self._main_thread = threading.Thread(target=self._run, daemon=True)
        self._running = True
        self._main_thread.start()
        logger.info("AnalyticsManager thread started.")

    def stop(self):
        """Stop the analytics manager thread gracefully."""
        if not self._running or self._main_thread is None:
            logger.warning("AnalyticsManager not running.")
            return
        logger.info("Stopping AnalyticsManager thread...")
        self._stop_event.set()
        # Put a dummy item in queue to unblock the get() if it's waiting
        try:
            self._data_in_queue.put_nowait({'_signal': 'stop'})
        except queue.Full:
            logger.warning("Analytics input queue full while trying to signal stop.")

        self._main_thread.join(timeout=5.0) # Wait for thread to finish
        if self._main_thread.is_alive():
            logger.error("AnalyticsManager thread did not stop gracefully.")
        else:
            logger.info("AnalyticsManager thread stopped.")
        self._running = False
        self._main_thread = None

    def _run(self):
        """Main loop for the analytics manager thread.
        (Round 20: Re-apply fix for storing raw counts in _latest_metrics)
        (Round 15: Correct timing for adding raw counts to _latest_metrics)
        (Round 13: Add raw counts to _latest_metrics)
        (Round 11: Added detailed logging for _latest_metrics handling)"""
        logger.debug("AnalyticsManager _run loop started.")
        log_prefix = "AnalyticsManager._run (R20 Delta Fix): " # Updated round

        while not self._stop_event.is_set():
                try:
                    # --- Log state BEFORE getting item ---
                    with self._metrics_lock:
                        latest_metrics_before_get = self._latest_metrics.copy()
                    logger.debug(f"{log_prefix}Waiting for item. _latest_metrics BEFORE get: {latest_metrics_before_get}")
                    # ---

                    data_snapshot = self._data_in_queue.get(timeout=1.0)
                    logger.debug(f"{log_prefix}Dequeued item. Type: {type(data_snapshot)}")

                    if data_snapshot.get('_signal') == 'stop':
                        logger.info("Stop signal received in analytics queue.")
                        break

                    generation = data_snapshot.get('generation', -1)
                    if generation == -1:
                        logger.warning("Received data snapshot without generation number.")
                        continue
                    logger.debug(f"{log_prefix}Processing data for Generation {generation}")

                    # --- Inject previous metrics for delta calculations ---
                    with self._metrics_lock:
                        previous_metrics_copy = self._latest_metrics.copy()
                        # Add the PREVIOUS metrics to the CURRENT snapshot for calculators
                        data_snapshot['_previous_latest_metrics'] = previous_metrics_copy
                        logger.debug(f"{log_prefix}Injected _previous_latest_metrics for Gen {generation}: {previous_metrics_copy}")
                    # ---

                    # --- 1. Calculate Metrics ---
                    calculated_metrics: Dict[str, Any] = {}
                    all_calculators = self.metric_registry.get_all_calculator_instances()
                    logger.debug(f"{log_prefix}Found {len(all_calculators)} metric calculators.")

                    for calculator in all_calculators:
                        group = calculator.get_group()
                        freq = calculator.get_frequency()
                        calc_name = calculator.get_name()

                        if self._metric_group_enabled.get(group, True) and generation % freq == 0:
                            logger.debug(f"{log_prefix}Attempting to run calculator: {calc_name} (Group: {group}) for Gen {generation}")
                            dependencies = calculator.get_dependencies()
                            deps_met = True
                            missing_deps = []
                            for dep in dependencies:
                                if dep not in data_snapshot:
                                    deps_met = False
                                    missing_deps.append(dep)
                            if deps_met:
                                try:
                                    result = calculator.calculate(data_snapshot) # Pass the snapshot containing previous metrics
                                    calculated_metrics.update(result)
                                    logger.debug(f"{log_prefix}Successfully ran calculator {calc_name}. Result keys: {list(result.keys())}")
                                except Exception as e:
                                    logger.error(f"Error calculating metric '{calculator.get_name()}' in group '{group}': {e}")
                            else:
                                if dependencies != ['_previous_latest_metrics']:
                                    logger.debug(f"Skipping metric '{calculator.get_name()}' due to missing dependencies: {missing_deps}")
                        elif not self._metric_group_enabled.get(group, True):
                            logger.debug(f"{log_prefix}Skipping calculator {calc_name} because group '{group}' is disabled.")
                        else: # Frequency mismatch
                            logger.debug(f"{log_prefix}Skipping calculator {calc_name} for Gen {generation} due to frequency ({freq}).")

                    # --- 2. Update Latest Metrics & History ---
                    # --- RE-APPLYING Round 15 Logic ---
                    if calculated_metrics:
                        with self._metrics_lock:
                            latest_metrics_before_update = self._latest_metrics.copy()
                            # Update with the metrics calculated in THIS step (ratios, deltas, etc.)
                            self._latest_metrics.update(calculated_metrics)
                            logger.debug(f"{log_prefix}Updated _latest_metrics with calculated metrics for Gen {generation}: {calculated_metrics}")

                            # NOW, add/overwrite the raw counts from the CURRENT snapshot
                            # These will be used as the 'previous' counts in the NEXT iteration
                            raw_active_count = data_snapshot.get('active_node_count')
                            raw_edge_count = data_snapshot.get('edge_count')
                            if raw_active_count is not None:
                                self._latest_metrics['active_node_count'] = raw_active_count
                                logger.debug(f"{log_prefix}Stored raw 'active_node_count': {raw_active_count} in _latest_metrics for next step.")
                            if raw_edge_count is not None:
                                self._latest_metrics['edge_count'] = raw_edge_count
                                logger.debug(f"{log_prefix}Stored raw 'edge_count': {raw_edge_count} in _latest_metrics for next step.")

                            latest_metrics_after_update = self._latest_metrics.copy() # Get final state for logging

                        # Store the calculated metrics (ratios, deltas) in history
                        self.data_store.add_metrics(generation, calculated_metrics)
                        logger.debug(f"{log_prefix}Added calculated metrics to history for Gen {generation}.")
                        # Log the state transitions
                        logger.debug(f"  _latest_metrics BEFORE update: {latest_metrics_before_update}")
                        logger.debug(f"  Calculated metrics this step (incl. raw): {calculated_metrics}")
                        logger.debug(f"  _latest_metrics AFTER update (incl. raw counts): {latest_metrics_after_update}")
                    else:
                        logger.debug(f"{log_prefix}No metrics calculated for Gen {generation}.")
                        # Still store raw counts even if no other metrics calculated
                        with self._metrics_lock:
                            raw_active_count = data_snapshot.get('active_node_count')
                            raw_edge_count = data_snapshot.get('edge_count')
                            if raw_active_count is not None: self._latest_metrics['active_node_count'] = raw_active_count
                            if raw_edge_count is not None: self._latest_metrics['edge_count'] = raw_edge_count
                            if raw_active_count is not None or raw_edge_count is not None:
                                logger.debug(f"{log_prefix}Stored raw counts in _latest_metrics even though no other metrics were calculated.")
                    # --- END RE-APPLYING Round 15 Logic ---

                    # --- 3. Run Analyzers ---
                    analysis_results: Dict[str, Any] = {}
                    all_analyzers = self.analyzer_registry.get_all_analyzer_instances()
                    logger.debug(f"{log_prefix}Found {len(all_analyzers)} analyzers.")

                    for analyzer in all_analyzers:
                        group = analyzer.get_group()
                        freq = analyzer.get_frequency()
                        analyzer_name = analyzer.get_name()
                        if self._analyzer_group_enabled.get(group, True) and generation % freq == 0:
                            logger.debug(f"{log_prefix}Attempting to run analyzer: {analyzer_name} (Group: {group}) for Gen {generation}")
                            try:
                                result = analyzer.analyze(self.data_store)
                                analysis_results.update(result)
                                logger.debug(f"{log_prefix}Successfully ran analyzer {analyzer_name}. Result keys: {list(result.keys())}")
                            except Exception as e:
                                logger.error(f"Error running analyzer '{analyzer.get_name()}' in group '{group}': {e}")
                        elif not self._analyzer_group_enabled.get(group, True):
                            logger.debug(f"{log_prefix}Skipping analyzer {analyzer_name} because group '{group}' is disabled.")
                        else: # Frequency mismatch
                            logger.debug(f"{log_prefix}Skipping analyzer {analyzer_name} for Gen {generation} due to frequency ({freq}).")

                    # --- 4. Update Latest Analysis ---
                    if analysis_results:
                        with self._metrics_lock:
                            self._latest_analysis.update(analysis_results)
                        logger.debug(f"{log_prefix}Updated latest analysis for Gen {generation}. Keys: {list(analysis_results.keys())}")
                    else:
                        logger.debug(f"{log_prefix}No analysis results calculated for Gen {generation}.")

                    # --- 5. Trigger Reporter ---
                    if self._save_report_enabled:
                        try:
                            logger.debug(f"{log_prefix}Triggering report generation for Gen {generation}.")
                            self._reporter.generate_report(generation, self._latest_metrics, self._latest_analysis)
                        except Exception as e:
                            logger.error(f"Error generating report: {e}")

                    self._data_in_queue.task_done() # Mark task as complete

                except queue.Empty:
                    logger.debug("AnalyticsManager: _data_in_queue empty, looping.")
                    continue
                except Exception as e:
                    logger.error(f"Error in AnalyticsManager loop: {e}", exc_info=True)
                    time.sleep(0.1)

                logger.debug("AnalyticsManager _run loop finished.")

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Return a copy of the latest calculated metrics."""
        with self._metrics_lock:
            return self._latest_metrics.copy()

    def get_latest_analysis(self) -> Dict[str, Any]:
        """Return a copy of the latest analysis results."""
        with self._metrics_lock:
            return self._latest_analysis.copy()

    def get_metric_history(self, metric_name: str, window_size: Optional[int] = None) -> List[Tuple[int, Any]]:
         """Retrieve historical data for a specific metric."""
         return self.data_store.get_history(metric_name, window_size)

    def get_all_metric_names_from_history(self) -> List[str]:
        """Get names of all metrics currently in the data store history."""
        return self.data_store.get_metric_names()

    def toggle_metric_group(self, group_name: str, enabled: bool):
        """Enable or disable calculation for a group of metrics."""
        logger.info(f"Setting metric group '{group_name}' enabled state to: {enabled}")
        self._metric_group_enabled[group_name] = enabled

    def toggle_analyzer_group(self, group_name: str, enabled: bool):
        """Enable or disable execution for a group of analyzers."""
        logger.info(f"Setting analyzer group '{group_name}' enabled state to: {enabled}")
        self._analyzer_group_enabled[group_name] = enabled

    def set_save_report_enabled(self, enabled: bool):
        """Enable or disable automatic report saving."""
        logger.info(f"Setting report saving enabled state to: {enabled}")
        self._save_report_enabled = enabled

    def save_report_now(self, filename_suffix: str = "manual_save") -> Optional[str]:
        """
        Forces the generation and saving of a report with the current data,
        using a specific filename suffix. Closes the file afterwards.
        Returns the path of the saved file or None on error.
        (Round 9: Fix lock name and missing value placeholder)
        (Round 7: New method)
        """
        # --- MODIFIED: Use correct lock name ---
        with self._metrics_lock: # Protect access to reporter and data
        # ---
            if self._reporter is None:
                logger.error("Reporter not initialized, cannot save report now.")
                return None

            # Ensure headers are up-to-date before potentially creating a new file
            current_headers = set(['generation'])
            current_headers.update(self._latest_metrics.keys())
            current_headers.update(self._latest_analysis.keys())
            sorted_headers = ['generation'] + sorted(list(current_headers - {'generation'}))

            # Use a specific filename for manual save
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # --- MODIFIED: Get generation safely ---
            generation = self._latest_metrics.get('generation')
            if generation is None: # Try getting max gen from history if latest metrics empty
                 max_gen_hist = 0
                 all_metric_names_hist = self.data_store.get_metric_names()
                 if all_metric_names_hist:
                     for name in all_metric_names_hist:
                         history = self.data_store.get_history(name)
                         if history: max_gen_hist = max(max_gen_hist, history[-1][0])
                 generation = max_gen_hist
            generation_str = str(generation) if generation is not None else 'unknown'
            # ---
            filename = f"lace_report_{timestamp}_gen{generation_str}_{filename_suffix}.csv"
            report_path = os.path.join(self.report_dir, filename)

            try:
                # Ensure directory exists
                if not os.path.exists(self.report_dir):
                    os.makedirs(self.report_dir)

                logger.info(f"Manually saving analytics report to: {report_path}")
                with open(report_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted_headers)
                    writer.writeheader()
                    # Write all historical data
                    all_metric_names = self.data_store.get_metric_names()
                    all_analysis_names = list(self._latest_analysis.keys()) # Use latest analysis keys
                    max_gen = 0
                    if all_metric_names:
                        # Find the max generation across all metrics
                        for name in all_metric_names:
                            history = self.data_store.get_history(name)
                            if history:
                                max_gen = max(max_gen, history[-1][0])

                    # Write rows generation by generation
                    for gen in range(max_gen + 1):
                        row_data = {'generation': gen}
                        # Get metric values for this generation
                        for name in all_metric_names:
                            history = self.data_store.get_history(name)
                            # Find the value for the current generation
                            gen_value = next((val for g, val in reversed(history) if g == gen), None)
                            # --- MODIFIED: Use None instead of '' for missing values ---
                            row_data[name] = int(gen_value) if gen_value is not None else 0 # Assign 0 if missing
                            # ---
                        # Get analysis values (assuming they are stored per generation, might need adjustment)
                        # For simplicity, let's just write the latest analysis for the last row
                        if gen == max_gen:
                             row_data.update(self._latest_analysis)

                        # Ensure all headers are present, using None for missing values
                        final_row = {header: row_data.get(header) for header in sorted_headers}
                        writer.writerow(final_row)

                logger.info(f"Manual report saved successfully: {report_path}")
                return report_path

            except Exception as e:
                logger.error(f"Error during manual report save to '{report_path}': {e}")
                return None
            finally:
                # Close the main reporter file if it was open, as we just saved manually
                self._reporter.close_report_file()

    def clear_history(self):
        """Clear the metric history in the data store."""
        self.data_store.clear_history()
        with self._metrics_lock:
            self._latest_metrics.clear()
            self._latest_analysis.clear()
        logger.info("Cleared analytics history and latest values.")

# === Basic Calculators ===

class BasicCountsCalculator(MetricCalculator):
    """Calculates basic ratios from counts provided by the engine."""

    def calculate(self, data_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metric(s) based on the provided data snapshot.
        (Round 12: Log input counts explicitly)
        (Round 7: Added logging for delta calculation)
        """
        results = {}
        # --- ADDED: Log received counts ---
        active_nodes = data_snapshot.get('active_node_count', 'MISSING')
        edge_count = data_snapshot.get('edge_count', 'MISSING')
        total_nodes = data_snapshot.get('total_nodes', 1) # Avoid division by zero
        log_prefix = f"BasicCountsCalculator (Gen {data_snapshot.get('generation', 'N/A')}): " # For logging
        logger.debug(f"{log_prefix}Received counts - ActiveNodes: {active_nodes}, EdgeCount: {edge_count}, TotalNodes: {total_nodes}")
        # ---

        # Convert to numeric, default to 0 if missing or invalid
        try: active_nodes_num = int(active_nodes)
        except (ValueError, TypeError): active_nodes_num = 0
        try: edge_count_num = int(edge_count)
        except (ValueError, TypeError): edge_count_num = 0
        try: total_nodes_num = int(total_nodes)
        except (ValueError, TypeError): total_nodes_num = 1 # Avoid division by zero

        if total_nodes_num > 0:
            results['active_node_ratio'] = float(active_nodes_num) / total_nodes_num
        else:
            results['active_node_ratio'] = 0.0

        if edge_count_num > 0:
            results['node_edge_ratio'] = float(active_nodes_num) / edge_count_num
        else:
             results['node_edge_ratio'] = float('inf') if active_nodes_num > 0 else 0.0

        # --- Delta Calculation with Logging ---
        prev_metrics = data_snapshot.get('_previous_latest_metrics', {}) # Manager needs to add this
        logger.debug(f"{log_prefix}Previous metrics received for delta calc: {prev_metrics}") # Log received previous metrics

        # Use .get() with default values matching the CURRENT snapshot's values
        prev_active_nodes = prev_metrics.get('active_node_count', active_nodes_num) # Default to current if missing
        prev_edge_count = prev_metrics.get('edge_count', edge_count_num)       # Default to current if missing

        # Ensure types are numeric before subtraction
        try:
            # Use the already validated current counts
            prev_active_nodes_num = int(prev_active_nodes)
            results['active_nodes_delta'] = active_nodes_num - prev_active_nodes_num
            logger.debug(f"{log_prefix}Active Nodes Delta: Current={active_nodes_num}, Prev={prev_active_nodes_num}, Delta={results['active_nodes_delta']}")
        except (ValueError, TypeError) as e:
            logger.warning(f"{log_prefix}Could not calculate active_nodes_delta: {e}. Current: {active_nodes}, Prev: {prev_active_nodes}")
            results['active_nodes_delta'] = 0 # Default on error

        try:
            # Use the already validated current counts
            prev_edge_count_num = int(prev_edge_count)
            results['edge_count_delta'] = edge_count_num - prev_edge_count_num
            logger.debug(f"{log_prefix}Edge Count Delta: Current={edge_count_num}, Prev={prev_edge_count_num}, Delta={results['edge_count_delta']}")
        except (ValueError, TypeError) as e:
            logger.warning(f"{log_prefix}Could not calculate edge_count_delta: {e}. Current: {edge_count}, Prev: {prev_edge_count}")
            results['edge_count_delta'] = 0 # Default on error
        # ---

        logger.debug(f"{log_prefix}Calculated results: {results}")
        return results

    def get_name(self) -> List[str]:
        # --- ADDED: New delta metrics ---
        return ['active_node_ratio', 'node_edge_ratio', 'active_nodes_delta', 'edge_count_delta']
        # ---

    def get_frequency(self) -> int:
        return 1 # Calculate every step

    def get_dependencies(self) -> List[str]:
        # --- ADDED: Dependency on previous metrics ---
        return ['active_node_count', 'edge_count', 'total_nodes', '_previous_latest_metrics']
        # ---

    def get_group(self) -> str:
        return "Basic"

# === ADDED: Advanced Calculators ===


class ShannonEntropyCalculator(MetricCalculator):
    """Calculates Shannon entropy based on node state distribution."""

    def calculate(self, data_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        grid_array = data_snapshot.get('grid_array_snapshot')
        if grid_array is None or not isinstance(grid_array, np.ndarray) or grid_array.size == 0:
            return {'node_state_entropy': 0.0}

        # Flatten and get unique states and their counts
        states = grid_array.ravel()
        unique_states, counts = np.unique(states, return_counts=True)

        # Calculate probabilities
        probabilities = counts / states.size

        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12)) # Add epsilon for log(0)

        return {'node_state_entropy': float(entropy)} # Ensure float return

    def get_name(self) -> str:
        return 'node_state_entropy'

    def get_frequency(self) -> int:
        return 10 # Calculate every 10 steps (configurable later)

    def get_dependencies(self) -> List[str]:
        return ['grid_array_snapshot'] # Requires the grid state

    def get_group(self) -> str:
        return "Entropy"

class GraphTopologyCalculator(MetricCalculator):
    """Calculates basic graph topology metrics using NetworkX."""

    def calculate(self, data_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        # --- Get edge data (assuming COO format is pushed if this metric is enabled) ---
        edge_data_coo = data_snapshot.get('edge_snapshot_coo')
        total_nodes = data_snapshot.get('total_nodes', 0)
        active_nodes_indices = data_snapshot.get('active_node_indices') # Need active indices

        if edge_data_coo is None or total_nodes == 0 or active_nodes_indices is None:
            logger.debug("GraphTopologyCalculator: Missing edge data, total_nodes, or active_node_indices. Skipping.")
            results['num_connected_components'] = 0
            results['largest_component_size'] = 0
            results['largest_component_ratio'] = 0.0
            return results

        rows, cols, vals = edge_data_coo
        # Filter edges to include only those connecting two *active* nodes
        active_set = set(active_nodes_indices)
        active_edges = []
        min_state_threshold = 1e-6 # Assuming numeric state, adjust if bool
        for r, c, v in zip(rows, cols, vals):
             # Consider upper triangle, check if both endpoints are active, and edge exists
             if r < c and r in active_set and c in active_set and abs(float(v)) > min_state_threshold:
                 active_edges.append((r, c)) # Add edge between active nodes

        if not active_edges and not active_set: # No edges and no active nodes
             results['num_connected_components'] = 0
             results['largest_component_size'] = 0
             results['largest_component_ratio'] = 0.0
             return results
        elif not active_edges and active_set: # Active nodes but no edges between them
             results['num_connected_components'] = len(active_set) # Each node is a component
             results['largest_component_size'] = 1 if active_set else 0
             results['largest_component_ratio'] = 1.0 / len(active_set) if active_set else 0.0
             return results

        try:
            # Create NetworkX graph from the active edges
            # Only include nodes that are part of at least one active edge OR are active but isolated
            graph = nx.Graph()
            graph.add_nodes_from(active_set) # Add all active nodes first
            graph.add_edges_from(active_edges) # Add edges between active nodes

            # Calculate connected components
            components = list(nx.connected_components(graph))
            num_components = len(components)
            largest_component_size = 0
            if components:
                largest_component_size = max(len(c) for c in components)

            num_active_nodes = len(active_set)
            largest_component_ratio = float(largest_component_size) / num_active_nodes if num_active_nodes > 0 else 0.0

            results['num_connected_components'] = num_components
            results['largest_component_size'] = largest_component_size
            results['largest_component_ratio'] = largest_component_ratio

        except Exception as e:
            logger.error(f"Error calculating graph topology: {e}")
            results['num_connected_components'] = -1 # Indicate error
            results['largest_component_size'] = -1
            results['largest_component_ratio'] = -1.0

        return results

    def get_name(self) -> List[str]:
        return ['num_connected_components', 'largest_component_size', 'largest_component_ratio']

    def get_frequency(self) -> int:
        # Use frequency from GlobalSettings
        return GlobalSettings.Analytics.FREQ_TOPOLOGY

    def get_dependencies(self) -> List[str]:
        # Requires edge data (COO) and the set of active node indices
        return ['edge_snapshot_coo', 'total_nodes', 'active_node_indices']

    def get_group(self) -> str:
        return "Topology"


# === Analyzers ===

class StabilityAnalyzer(Analyzer):
    """Calculates stability based on standard deviation of active node ratio."""
    def analyze(self, history_data: 'MetricsDataStore') -> Dict[str, Any]:
        history = history_data.get_history('active_node_ratio', window_size=50)
        if len(history) > 1:
            values = [val for gen, val in history]
            std_dev = np.std(values)
            # Lower std dev means more stable
            stability_metric = 1.0 / (1.0 + std_dev) # Example metric (0 to 1)
            return {'stability_metric_50': stability_metric}
        return {'stability_metric_50': 0.0}

    def get_name(self) -> str:
        return 'stability_metric_50'

    def get_frequency(self) -> int:
        return 10

    def get_dependencies(self) -> List[str]:
        return ['active_node_ratio']

    def get_group(self) -> str:
        return "Stability"

# --- ADDED: Trend Analyzer ---
class TrendAnalyzer(Analyzer):
    """Calculates rolling average and basic trend direction for metrics."""
    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def analyze(self, history_data: 'MetricsDataStore') -> Dict[str, Any]:
        results = {}
        metrics_to_analyze = ['active_node_ratio', 'node_edge_ratio', 'node_state_entropy'] # Example metrics

        for metric_name in metrics_to_analyze:
            history = history_data.get_history(metric_name, window_size=self.window_size * 2) # Get more history for trend
            if len(history) >= self.window_size:
                generations, values = zip(*history)
                numeric_values = []
                valid_generations = []
                for gen, val in zip(generations, values):
                    try: numeric_values.append(float(val)); valid_generations.append(gen)
                    except (ValueError, TypeError): continue

                if len(numeric_values) >= self.window_size:
                    # Rolling Average
                    rolling_avg = np.mean(numeric_values[-self.window_size:])
                    results[f'{metric_name}_avg_{self.window_size}'] = rolling_avg

                    # Basic Trend (Slope of linear regression on last window_size points)
                    if len(numeric_values) >= 2: # Need at least 2 points for slope
                        try:
                            x = np.array(valid_generations[-self.window_size:])
                            y = np.array(numeric_values[-self.window_size:])
                            # Handle potential constant x values (vertical line)
                            if np.all(x == x[0]):
                                slope = 0.0 # Or handle as undefined/error
                            else:
                                slope, intercept = np.polyfit(x, y, 1)
                            results[f'{metric_name}_trend_{self.window_size}'] = slope
                        except (np.linalg.LinAlgError, ValueError) as lin_err:
                             logger.warning(f"Could not calculate trend for {metric_name}: {lin_err}")
                             results[f'{metric_name}_trend_{self.window_size}'] = 0.0 # Default on error
                else:
                    results[f'{metric_name}_avg_{self.window_size}'] = None
                    results[f'{metric_name}_trend_{self.window_size}'] = None
            else:
                results[f'{metric_name}_avg_{self.window_size}'] = None
                results[f'{metric_name}_trend_{self.window_size}'] = None

        return results

    def get_name(self) -> List[str]:
        # Dynamically generate names based on analyzed metrics
        base_metrics = ['active_node_ratio', 'node_edge_ratio', 'node_state_entropy']
        names = []
        for m in base_metrics:
            names.append(f'{m}_avg_{self.window_size}')
            names.append(f'{m}_trend_{self.window_size}')
        return names

    def get_frequency(self) -> int:
        return 5 # Analyze every 5 steps

    def get_dependencies(self) -> List[str]:
        return ['active_node_ratio', 'node_edge_ratio', 'node_state_entropy'] # Depends on these metrics being in history

    def get_group(self) -> str:
        return "Trend"
# ---

# === Placeholder Reporter ===

class Reporter:
    """Handles saving metrics and analysis results to reports."""
    def __init__(self, data_store: MetricsDataStore, report_dir: str):
        self.data_store = data_store
        self.report_dir = report_dir
        self._report_file_path: Optional[str] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._file_handle: Optional[Any] = None
        self._headers_written = False
        self._lock = threading.Lock() # Protect file access

        if not os.path.exists(self.report_dir):
            try:
                os.makedirs(self.report_dir)
                logger.info(f"Created report directory: {self.report_dir}")
            except OSError as e:
                logger.error(f"Failed to create report directory '{self.report_dir}': {e}")
                # Proceed without a valid directory, saving will fail later

    def _initialize_report_file(self, generation: int, headers: List[str]):
        """Creates a new report file and writes headers."""
        self.close_report_file() # Close existing if any
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"lace_report_{timestamp}_gen{generation}_start.csv" # Indicate start gen
        self._report_file_path = os.path.join(self.report_dir, filename)
        try:
            # Ensure directory exists before opening file
            if not os.path.exists(self.report_dir):
                 os.makedirs(self.report_dir)

            self._file_handle = open(self._report_file_path, 'w', newline='', encoding='utf-8')
            # --- Ensure 'generation' is first header ---
            if 'generation' in headers: headers.remove('generation')
            final_headers = ['generation'] + sorted(headers)
            # ---
            self._csv_writer = csv.DictWriter(self._file_handle, fieldnames=final_headers)
            self._csv_writer.writeheader()
            self._headers_written = True
            logger.info(f"Initialized report file: {self._report_file_path}")
        except IOError as e:
            logger.error(f"Failed to initialize report file '{self._report_file_path}': {e}")
            self._report_file_path = None
            self._file_handle = None
            self._csv_writer = None
            self._headers_written = False
        except Exception as e: # Catch other potential errors
             logger.error(f"Unexpected error initializing report file '{self._report_file_path}': {e}")
             self._report_file_path = None; self._file_handle = None; self._csv_writer = None; self._headers_written = False

    def generate_report(self, generation: int, latest_metrics: Dict[str, Any], latest_analysis: Dict[str, Any]):
        """Writes the current metrics and analysis to the report file."""
        with self._lock:
            # Combine all potential headers
            current_headers = set(['generation'])
            current_headers.update(latest_metrics.keys())
            current_headers.update(latest_analysis.keys())
            sorted_headers = ['generation'] + sorted(list(current_headers - {'generation'}))

            # Initialize file/writer if it's the first write or headers changed
            if self._csv_writer is None or not self._headers_written or set(self._csv_writer.fieldnames) != set(sorted_headers):
                logger.info(f"Initializing/Reinitializing report file. Headers changed or first write. New headers: {sorted_headers}")
                self._initialize_report_file(generation, sorted_headers)
                if self._csv_writer is None: # Check if initialization failed
                    logger.error("Cannot generate report: CSV writer not initialized.")
                    return

            try:
                row_data = {'generation': generation}
                row_data.update(latest_metrics)
                row_data.update(latest_analysis)

                # Ensure all headers defined in the writer are present in the row
                final_row = {header: row_data.get(header, '') for header in self._csv_writer.fieldnames} # Use '' for missing

                self._csv_writer.writerow(final_row)
                if self._file_handle: self._file_handle.flush() # Ensure data is written periodically
            except Exception as e:
                logger.error(f"Error writing row for generation {generation} to report file: {e}")

    def close_report_file(self):
        """Closes the current report file if open."""
        with self._lock:
            if self._file_handle:
                try:
                    self._file_handle.close()
                    logger.info(f"Closed report file: {self._report_file_path}")
                except Exception as e:
                    logger.error(f"Error closing report file: {e}")
            self._report_file_path = None
            self._csv_writer = None
            self._file_handle = None
            self._headers_written = False

    def __del__(self):
        """Ensure file is closed when Reporter object is destroyed."""
        self.close_report_file()
# === Initialization Function (Optional) ===

def initialize_analytics_system() -> Tuple[AnalyticsManager, MetricRegistry, AnalyzerRegistry]:
    """Helper function to initialize and register basic components."""
    metric_registry = MetricRegistry()
    analyzer_registry = AnalyzerRegistry()

    # Register basic calculators/analyzers
    metric_registry.register(BasicCountsCalculator)
    metric_registry.register(ShannonEntropyCalculator)
    metric_registry.register(GraphTopologyCalculator) # Register new calculator
    analyzer_registry.register(StabilityAnalyzer)
    analyzer_registry.register(TrendAnalyzer)
    # TODO: Register more calculators/analyzers here as they are implemented

    data_in_queue = queue.Queue()  # Initialize the required queue
    manager = AnalyticsManager(metric_registry, analyzer_registry, data_in_queue=data_in_queue)

    return manager, metric_registry, analyzer_registry

class AnalyticsWindow(tk.Toplevel):
    """
    A non-modal window for displaying simulation metrics and analysis results.
    """
    def __init__(self, parent: 'SimulationGUI', analytics_manager: AnalyticsManager):
        """
        Initialize the Analytics Window.
        (Round 17: Added Matplotlib elements for Time Series tab)
        (Round 18: Added Config tab elements and variables)
        (Round 19: Added Analysis tab elements)
        """
        if not parent or not parent.root or not parent.root.winfo_exists():
            logger.error("AnalyticsWindow: Invalid parent window.")
            return

        super().__init__(parent.root)
        # self.parent = parent
        self.parent = parent
        # --- ADDED: Make window transient for parent ---
        self.transient(parent.root)
        logger.debug("Set RuleEditorWindow as transient for the main application window.")
        # ---
        self.analytics_manager = analytics_manager
        self.title("LACE Analytics")
        self.geometry("800x600")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._update_job_id: Optional[str] = None
        self._update_interval_ms: int = 1000
        self._is_closing = False

        # --- Attributes for Time Series Plot ---
        self._ts_fig: Optional[Figure] = None
        self._ts_ax: Optional[Axes] = None
        self._ts_canvas: Optional[FigureCanvasTkAgg] = None
        self._ts_selected_metric = tk.StringVar()
        self._ts_line: Optional[Line2D] = None
        # ---

        # --- Attributes for Config Tab ---
        self._metric_group_vars: Dict[str, tk.BooleanVar] = {}
        self._analyzer_group_vars: Dict[str, tk.BooleanVar] = {}
        self.save_report_var = tk.BooleanVar(value=self.analytics_manager._save_report_enabled)
        # ---

        # --- ADDED: Attributes for Analysis Tab ---
        self._analysis_text_area: Optional[tk.Text] = None
        # ---

        # --- Main Frame ---
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Top Control Frame ---
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 5))

        if not hasattr(self.parent, 'analytics_enabled_var'):
            logger.error("Parent GUI missing 'analytics_enabled_var'. Cannot link checkbox.")
            self.local_analytics_enabled_var = tk.BooleanVar(value=False)
            analytics_enable_checkbox = tk.Checkbutton(
                control_frame, text="Enable Analytics Collection",
                variable=self.local_analytics_enabled_var
            )
        else:
            analytics_enable_checkbox = tk.Checkbutton(
                control_frame, text="Enable Analytics Collection",
                variable=self.parent.analytics_enabled_var,
                command=self.parent._on_toggle_analytics_enabled
            )
        analytics_enable_checkbox.pack(side=tk.LEFT, padx=5)

        # --- Notebook for Tabs ---
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # --- Create Tabs ---
        self.summary_tab = self._create_summary_tab()
        self.time_series_tab = self._create_time_series_tab()
        self.analysis_tab = self._create_analysis_tab() # Creates analysis elements
        self.config_tab = self._create_config_tab()

        self.notebook.add(self.summary_tab, text="Summary")
        self.notebook.add(self.time_series_tab, text="Time Series")
        self.notebook.add(self.analysis_tab, text="Analysis")
        self.notebook.add(self.config_tab, text="Configuration")

        # --- Start the UI update loop ---
        self._schedule_update()

        logger.info("AnalyticsWindow initialized.")

    def _create_summary_tab(self) -> tk.Frame:
        """Creates the Summary tab content with a Treeview for latest metrics."""
        from .lace_app import ScrollableFrame 
        frame = tk.Frame(self.notebook)

        # --- Treeview for Latest Metrics ---
        tree_frame = tk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Define columns
        columns = ('metric', 'value')
        self.summary_tree = ttk.Treeview(tree_frame, columns=columns, show='headings')

        # Define headings
        self.summary_tree.heading('metric', text='Metric Name')
        self.summary_tree.heading('value', text='Latest Value')

        # Configure column widths (adjust as needed)
        self.summary_tree.column('metric', anchor=tk.W, width=200)
        self.summary_tree.column('value', anchor=tk.E, width=150)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.summary_tree.yview)
        self.summary_tree.configure(yscrollcommand=scrollbar.set)

        # Pack Treeview and Scrollbar
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # ---

        return frame

    def _create_time_series_tab(self) -> tk.Frame:
        """Creates the Time Series Plots tab content."""
        from .lace_app import ScrollableFrame 
        frame = tk.Frame(self.notebook)

        # --- Top frame for controls ---
        top_frame = tk.Frame(frame)
        top_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(top_frame, text="Metric:").pack(side=tk.LEFT, padx=(0, 5))

        # Combobox for metric selection
        self.ts_metric_combobox = ttk.Combobox(
            top_frame,
            textvariable=self._ts_selected_metric,
            state="readonly",
            width=30
        )
        self.ts_metric_combobox.pack(side=tk.LEFT, padx=5)
        self.ts_metric_combobox.bind("<<ComboboxSelected>>", self._on_ts_metric_selected)

        # --- Matplotlib Plot Area ---
        plot_frame = tk.Frame(frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        try:
            self._ts_fig = Figure(figsize=(6, 4), dpi=100) # Create Figure
            self._ts_ax = self._ts_fig.add_subplot(111) # Add Axes
            self._ts_ax.set_xlabel("Generation")
            self._ts_ax.set_ylabel("Value")
            self._ts_ax.grid(True, linestyle='--', alpha=0.6)
            if self._ts_fig is not None:
                self._ts_fig.tight_layout() # Adjust layout

            self._ts_canvas = FigureCanvasTkAgg(self._ts_fig, master=plot_frame) # Embed in Tk frame
            self._ts_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._ts_canvas.draw() # Initial draw
            logger.debug("Time series plot canvas created.")
        except Exception as e:
            logger.error(f"Error creating time series plot canvas: {e}")
            tk.Label(plot_frame, text="Error creating plot.").pack()

        return frame

    def _create_analysis_tab(self) -> tk.Frame:
        """Creates the Analysis Results tab content."""
        # --- MODIFIED: Use ScrollableFrame ---
        from .lace_app import ScrollableFrame # Local import
        frame = tk.Frame(self.notebook)
        scrollable = ScrollableFrame(frame, self.parent)
        scrollable.pack(fill=tk.BOTH, expand=True)
        content_frame = scrollable.scrolled_frame
        # ---

        # --- Text Area for Analysis Results ---
        text_frame = tk.Frame(content_frame) # Add to scrollable content
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._analysis_text_area = tk.Text(
            text_frame,
            wrap=tk.WORD,
            state=tk.DISABLED, # Read-only initially
            height=10 # Initial height
        )

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self._analysis_text_area.yview)
        self._analysis_text_area.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._analysis_text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # ---

        # Placeholder message
        self._analysis_text_area.config(state=tk.NORMAL)
        self._analysis_text_area.insert(tk.END, "Analysis results will appear here...\n(Trends, Stability, Phase Changes, etc.)")
        self._analysis_text_area.config(state=tk.DISABLED)

        return frame

    def _create_config_tab(self) -> tk.Frame:
        """Creates the Configuration tab content."""
        # --- ADDED: Local import ---
        from .lace_app import ScrollableFrame
        # ---
        frame = tk.Frame(self.notebook)
        scrollable = ScrollableFrame(frame, self.parent)
        scrollable.pack(fill=tk.BOTH, expand=True)
        content_frame = scrollable.scrolled_frame

        # --- Metric Group Toggles ---
        metric_group_frame = tk.LabelFrame(content_frame, text="Metric Calculation Groups")
        metric_group_frame.pack(fill=tk.X, padx=5, pady=5)

        metric_groups = set(calc.get_group() for calc in self.analytics_manager.metric_registry.get_all_calculator_instances())
        logger.debug(f"Found metric groups: {metric_groups}")
        for group_name in sorted(list(metric_groups)):
            if group_name not in self._metric_group_vars:
                initial_state = self.analytics_manager._metric_group_enabled.get(group_name, True)
                self._metric_group_vars[group_name] = tk.BooleanVar(value=initial_state)

            cb = tk.Checkbutton(
                metric_group_frame,
                text=f"Enable {group_name} Metrics",
                variable=self._metric_group_vars[group_name],
                command=lambda name=group_name: self._on_toggle_metric_group(name)
            )
            cb.pack(anchor=tk.W, padx=10)
            # TODO: Add frequency control (e.g., Entry or Scale) for each group later

        # --- Analyzer Group Toggles ---
        analyzer_group_frame = tk.LabelFrame(content_frame, text="Analysis Groups")
        analyzer_group_frame.pack(fill=tk.X, padx=5, pady=5)

        analyzer_groups = set(analyzer.get_group() for analyzer in self.analytics_manager.analyzer_registry.get_all_analyzer_instances())
        logger.debug(f"Found analyzer groups: {analyzer_groups}")
        for group_name in sorted(list(analyzer_groups)):
            if group_name not in self._analyzer_group_vars:
                initial_state = self.analytics_manager._analyzer_group_enabled.get(group_name, True)
                self._analyzer_group_vars[group_name] = tk.BooleanVar(value=initial_state)

            cb = tk.Checkbutton(
                analyzer_group_frame,
                text=f"Enable {group_name} Analysis",
                variable=self._analyzer_group_vars[group_name],
                command=lambda name=group_name: self._on_toggle_analyzer_group(name)
            )
            cb.pack(anchor=tk.W, padx=10)
            # TODO: Add frequency control for each group later

        # --- Reporting Section ---
        reporting_frame = tk.LabelFrame(content_frame, text="Reporting")
        reporting_frame.pack(fill=tk.X, padx=5, pady=5)

        # Save Report Toggle
        # --- MODIFIED: Use self.save_report_var initialized in __init__ ---
        save_report_cb = tk.Checkbutton(
            reporting_frame,
            text="Save Metrics Report (CSV)",
            variable=self.save_report_var, # Use instance variable
            command=self._on_toggle_save_report # Use instance method
        )
        # ---
        save_report_cb.pack(anchor=tk.W, padx=10)

        # TODO: Add controls for report frequency, content selection?
        # TODO: Add button to manually trigger report save?
        # TODO: Add button to clear history?

        return frame

    def _schedule_update(self):
        """Schedules the next UI update."""
        if self._is_closing or not self.winfo_exists():
            return # Don't reschedule if closing or destroyed
        # Cancel previous job if any
        if self._update_job_id:
            self.after_cancel(self._update_job_id)
        # Schedule the next update
        self._update_job_id = self.after(self._update_interval_ms, self._update_ui)

    def _update_ui(self):
        """Periodically fetches latest data and updates the UI elements."""
        if self._is_closing or not self.winfo_exists():
            # logger.debug("AnalyticsWindow: Skipping UI update, window closing or destroyed.") # Reduce noise
            return

        # logger.debug("AnalyticsWindow: Updating UI...") # Reduce log noise
        try:
            # Fetch latest data (use thread lock)
            latest_metrics = self.analytics_manager.get_latest_metrics()
            latest_analysis = self.analytics_manager.get_latest_analysis()

            # --- Update Summary Tab Treeview ---
            if hasattr(self, 'summary_tree'):
                existing_items = self.summary_tree.get_children()
                if existing_items: self.summary_tree.delete(*existing_items)
                for metric_name in sorted(latest_metrics.keys()):
                    value = latest_metrics[metric_name]
                    value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    self.summary_tree.insert('', tk.END, values=(metric_name, value_str))

            # --- Update Time Series Tab ---
            if hasattr(self, 'ts_metric_combobox'):
                available_metrics = sorted(self.analytics_manager.get_all_metric_names_from_history())
                current_values = tuple(self.ts_metric_combobox['values'])
                new_values = tuple(available_metrics) if available_metrics else ()
                if current_values != new_values:
                    self.ts_metric_combobox['values'] = new_values
                    # logger.debug(f"Updated time series combobox values: {new_values}") # Reduce noise
                    current_selection = self._ts_selected_metric.get()
                    if current_selection not in available_metrics and available_metrics:
                        self._ts_selected_metric.set(available_metrics[0])
                        self._plot_time_series()
                    elif not available_metrics:
                         self._ts_selected_metric.set("")
                         self._clear_time_series_plot()
                if self._ts_selected_metric.get(): self._plot_time_series()

            # --- Update Analysis Tab Text Area ---
            if hasattr(self, '_analysis_text_area') and self._analysis_text_area:
                # --- MODIFIED: Include both metrics and analysis ---
                display_text = "Latest Metrics:\n"
                if latest_metrics:
                    display_text += "\n".join(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}" for k, v in sorted(latest_metrics.items()))
                else:
                    display_text += "  (No metrics calculated yet)"

                display_text += "\n\nLatest Analysis Results:\n"
                if latest_analysis:
                    display_text += "\n".join(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}" for k, v in sorted(latest_analysis.items()))
                else:
                    display_text += "  (No analysis results calculated yet)"
                # ---

                # Update text area content
                current_scroll_pos = self._analysis_text_area.yview() # Get current scroll position
                self._analysis_text_area.config(state=tk.NORMAL)
                self._analysis_text_area.delete("1.0", tk.END)
                self._analysis_text_area.insert(tk.END, display_text)
                self._analysis_text_area.config(state=tk.DISABLED)
                self._analysis_text_area.yview_moveto(current_scroll_pos[0]) # Restore scroll position

            # --- Update Config Tab Checkboxes ---
            if hasattr(self, '_metric_group_vars'):
                for group_name, var in self._metric_group_vars.items():
                    current_state = self.analytics_manager._metric_group_enabled.get(group_name, True)
                    if var.get() != current_state: var.set(current_state)
            if hasattr(self, '_analyzer_group_vars'):
                 for group_name, var in self._analyzer_group_vars.items():
                    current_state = self.analytics_manager._analyzer_group_enabled.get(group_name, True)
                    if var.get() != current_state: var.set(current_state)
            if hasattr(self, 'save_report_var'):
                 current_state = self.analytics_manager._save_report_enabled
                 if self.save_report_var.get() != current_state: self.save_report_var.set(current_state)

        except Exception as e:
            logger.error(f"Error updating AnalyticsWindow UI: {e}")
            logger.error(traceback.format_exc())
        finally:
            self._schedule_update()

    def _on_ts_metric_selected(self, event=None):
        """Callback when a metric is selected in the time series combobox."""
        selected_metric = self._ts_selected_metric.get()
        logger.debug(f"Time series metric selected: {selected_metric}")
        self._plot_time_series() # Update plot with the new selection

    def _on_toggle_metric_group(self, group_name: str):
        """Callback when a metric group checkbox is toggled."""
        if group_name in self._metric_group_vars:
            is_enabled = self._metric_group_vars[group_name].get()
            logger.info(f"Metric group '{group_name}' toggled {'ON' if is_enabled else 'OFF'}.")
            self.analytics_manager.toggle_metric_group(group_name, is_enabled)
        else:
            logger.warning(f"Toggled metric group '{group_name}' but no corresponding variable found.")

    def _on_toggle_analyzer_group(self, group_name: str):
        """Callback when an analyzer group checkbox is toggled."""
        if group_name in self._analyzer_group_vars:
            is_enabled = self._analyzer_group_vars[group_name].get()
            logger.info(f"Analyzer group '{group_name}' toggled {'ON' if is_enabled else 'OFF'}.")
            self.analytics_manager.toggle_analyzer_group(group_name, is_enabled)
        else:
            logger.warning(f"Toggled analyzer group '{group_name}' but no corresponding variable found.")

    def _on_toggle_save_report(self):
        """Callback when the 'Save Metrics Report' checkbox is toggled."""
        is_enabled = self.save_report_var.get()
        logger.info(f"Report saving toggled {'ON' if is_enabled else 'OFF'}.")
        self.analytics_manager.set_save_report_enabled(is_enabled)
        if not is_enabled:
            # Close the report file if saving is disabled
            self.analytics_manager._reporter.close_report_file()

    def _plot_time_series(self):
        """Fetches history for the selected metric and updates the plot.
           (Round 7: Added logging for plotted data)"""
        if self._is_closing or not self.winfo_exists(): return
        if not hasattr(self, '_ts_ax') or self._ts_ax is None or not hasattr(self, '_ts_canvas') or self._ts_canvas is None:
            logger.warning("Time series plot axes or canvas not initialized.")
            return

        metric_name = self._ts_selected_metric.get()
        if not metric_name:
            self._clear_time_series_plot()
            return

        log_prefix = f"AnalyticsWindow._plot_time_series({metric_name}): " # Add metric name to log

        try:
            # Fetch history data
            history = self.analytics_manager.get_metric_history(metric_name)

            if not history:
                self._clear_time_series_plot(title=f"{metric_name} (No Data)")
                return

            generations, values = zip(*history)

            # Convert potential non-numeric values safely
            numeric_values = []
            valid_generations = []
            for gen, val in zip(generations, values):
                try:
                    numeric_values.append(float(val)) # Attempt conversion
                    valid_generations.append(gen)
                except (ValueError, TypeError):
                    logger.debug(f"Skipping non-numeric value '{val}' for metric '{metric_name}' at gen {gen}")
                    continue # Skip non-numeric points

            if not numeric_values: # Check if any valid points remain
                self._clear_time_series_plot(title=f"{metric_name} (No Numeric Data)")
                return

            # --- ADDED: Log the data being plotted ---
            log_limit = min(20, len(valid_generations)) # Log up to 20 points
            logger.debug(f"{log_prefix}Plotting {len(valid_generations)} points. First {log_limit}:")
            for i in range(log_limit):
                logger.debug(f"  Gen: {valid_generations[i]}, Value: {numeric_values[i]:.4f}")
            if len(valid_generations) > log_limit:
                logger.debug("  ...")
                last_few_start = max(log_limit, len(valid_generations) - 5)
                for i in range(last_few_start, len(valid_generations)):
                     logger.debug(f"  Gen: {valid_generations[i]}, Value: {numeric_values[i]:.4f}")
            # ---

            # Update plot data
            if self._ts_line is None: # Create line if it doesn't exist
                self._ts_line, = self._ts_ax.plot(valid_generations, numeric_values, marker='.', linestyle='-')
            else: # Update existing line data
                self._ts_line.set_data(valid_generations, numeric_values)

            # Adjust plot limits and redraw
            self._ts_ax.relim()
            self._ts_ax.autoscale_view()
            self._ts_ax.set_title(f"{metric_name} vs. Generation")
            self._ts_ax.set_ylabel(metric_name) # Update y-axis label
            if self._ts_fig is not None:
                self._ts_fig.tight_layout() # Adjust layout
            self._ts_canvas.draw_idle() # Schedule redraw

        except Exception as e:
            logger.error(f"Error plotting time series for '{metric_name}': {e}")
            self._clear_time_series_plot(title=f"{metric_name} (Error)")

    def reset_display(self):
        """Clears all data displays in the Analytics Window."""
        logger.info("AnalyticsWindow: Resetting display.")
        # Clear Summary Tree
        if hasattr(self, 'summary_tree'):
            existing_items = self.summary_tree.get_children()
            if existing_items: self.summary_tree.delete(*existing_items)

        # Clear Time Series Plot
        self._clear_time_series_plot(title="Time Series Plot (Reset)")
        # Clear and reset metric selector
        if hasattr(self, 'ts_metric_combobox'):
            self.ts_metric_combobox['values'] = ()
            self._ts_selected_metric.set("")

        # Clear Analysis Text Area
        if hasattr(self, '_analysis_text_area') and self._analysis_text_area:
            self._analysis_text_area.config(state=tk.NORMAL)
            self._analysis_text_area.delete("1.0", tk.END)
            self._analysis_text_area.insert(tk.END, "Analytics reset. Run simulation to collect new data.")
            self._analysis_text_area.config(state=tk.DISABLED)

        # Optionally update config tab if needed (e.g., reset toggles, though maybe not desired)
        logger.info("AnalyticsWindow: Display reset complete.")

    def _clear_time_series_plot(self, title="Time Series Plot"):
        """Clears the time series plot."""
        if self._is_closing or not self.winfo_exists(): return
        if not hasattr(self, '_ts_ax') or self._ts_ax is None or not hasattr(self, '_ts_canvas') or self._ts_canvas is None:
            return

        if self._ts_line:
            try:
                self._ts_line.remove()
            except ValueError: pass # Already removed
            self._ts_line = None
        self._ts_ax.cla() # Clear axes content
        self._ts_ax.set_xlabel("Generation")
        self._ts_ax.set_ylabel("Value")
        self._ts_ax.set_title(title)
        self._ts_ax.grid(True, linestyle='--', alpha=0.6)
        if self._ts_fig is not None:
            self._ts_fig.tight_layout()
        self._ts_canvas.draw_idle()

    def _update_control_states(self):
        """Updates the state of controls within the AnalyticsWindow based on external changes (e.g., main GUI toggle)."""
        # This method can be called by the main GUI's _on_toggle_analytics_enabled
        # to ensure the checkbox inside this window reflects the current state.
        logger.debug("AnalyticsWindow: Updating internal control states.")
        if hasattr(self.parent, 'analytics_enabled_var'):
            is_enabled = self.parent.analytics_enabled_var.get()
            # Find the checkbox within this window and update its visual state if needed
            # (This assumes the variable binding handles the actual state)
            pass # Placeholder - visual update might not be needed if var binding works
        else:
            logger.warning("AnalyticsWindow: Cannot update control states, parent GUI missing analytics_enabled_var.")

    def _on_close(self):
        """Handles the closing of the Analytics window."""
        logger.info("AnalyticsWindow closing.")
        self._is_closing = True
        if self._update_job_id:
            try:
                self.after_cancel(self._update_job_id)
                logger.debug("Cancelled scheduled UI update.")
            except Exception as e:
                logger.warning(f"Error cancelling UI update job: {e}")

        # --- ADDED: Cleanup Matplotlib figure/canvas ---
        if hasattr(self, '_ts_canvas') and self._ts_canvas:
            try:
                self._ts_canvas.get_tk_widget().destroy()
                logger.debug("Destroyed time series canvas widget.")
            except Exception as e:
                logger.warning(f"Error destroying time series canvas widget: {e}")
        if hasattr(self, '_ts_fig') and self._ts_fig:
            try:
                plt.close(self._ts_fig) # Close the figure
                logger.debug("Closed time series figure.")
            except Exception as e:
                logger.warning(f"Error closing time series figure: {e}")
        self._ts_canvas = None
        self._ts_fig = None
        self._ts_ax = None
        self._ts_line = None
        # ---

        # Clear the reference in the main GUI
        if hasattr(self.parent, 'analytics_window'):
            self.parent.analytics_window = None
            logger.debug("Cleared analytics_window reference in parent GUI.")
        self.destroy()


# =========== END of analytics.py ===========