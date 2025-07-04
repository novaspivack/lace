import hashlib
import json
import ast
import platform
import shutil
import sys
import types
import uuid
import psutil
import copy
from datetime import datetime
from math import log
import inspect
import dataclasses
from pickle import TRUE
import dill
from tkinter import simpledialog
from turtle import Turtle
import matplotlib
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, MultipleLocator
from typing import Dict, List, Set, Tuple, Optional, Type, Union, Any, Callable, cast, TypeVar, NamedTuple
import numpy as np
import numpy.typing as npt
import itertools
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import tkinter.ttk as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # type: ignore
import matplotlib.pyplot as plt
plt.ioff()
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D as Axes3DType
import matplotlib.markers as markers
from matplotlib.colors import Normalize
from matplotlib.markers import MarkerStyle 
from matplotlib.colors import to_rgba
import networkx as nx # type: ignore
from contextlib import contextmanager
import asyncio
import cProfile
import pstats
from enum import Enum, auto
from dataclasses import dataclass, asdict, field, fields
from collections import defaultdict
from _collections_abc import Mapping
from abc import ABC, abstractmethod
import logging
from itertools import combinations, product
import random
import os
from datetime import datetime
import time
import traceback
from functools import wraps, lru_cache
from numba import jit, njit, prange, cuda
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, shared_memory
import setproctitle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from queue import Full, Empty
from typing_extensions import Protocol
from scipy.spatial import cKDTree, distance # type: ignore
import warnings
import re 
import cProfile
import pstats

from NetworkCA import (
    RuleMetrics, RuleParameters, BaseMetrics, GlobalSettings, NeighborhoodData
)

warnings.filterwarnings('ignore', category=UserWarning)
_current_log_file: Optional[str] = None
# Type aliases for improved type hints
NodeIndex = int
GridArray = npt.NDArray[np.float64]
NeighborIndices = npt.NDArray[np.int64]
Coordinates = Tuple[float, ...]
StateVarType = TypeVar('StateVarType', bound=Union[bool, int, float])
_global_grid_boundary: str = 'bounded'

class StateEntropy(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float:  # Use NeighborhoodData
        return BaseMetrics.state_entropy(states, neighbor_indices, neighborhood_data) # Pass neighborhood_data
    
class ClusteringCoefficient(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float:  # Use NeighborhoodData
        return BaseMetrics.clustering_coefficient(states, neighbor_indices, neighborhood_data) # Pass neighborhood_data

class ActiveNeighborRatio(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float:
        """Ratio of active neighbors to total neighbors"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        return float(np.sum(states[valid_neighbors] > 0) / len(valid_neighbors))
    
class EdgeDensity(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float:
        """Local edge density in neighborhood"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) < 2:
            return 0.0
        possible_edges = len(valid_neighbors) * (len(valid_neighbors) - 1) / 2
        if possible_edges == 0:
            return 0.0
            
        actual_edges = 0.0
        for i in range(len(valid_neighbors)):
            for j in range(i + 1, len(valid_neighbors)):
                if neighborhood_data.has_edge(valid_neighbors[i], valid_neighbors[j]):
                    actual_edges += 1.0
                    
        return float(actual_edges / possible_edges)
    
class AverageNeighborDegree(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float:
        """Average degree (number of connections) of a node's neighbors"""
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        total_neighbor_degrees = 0
        for neighbor in valid_neighbors:
            # Use get_active_edges to count only edges to active neighbors
            total_neighbor_degrees += len(neighborhood_data.get_active_edges(neighbor, states))
        
        return float(total_neighbor_degrees / len(valid_neighbors))
    
class VarianceNeighborState(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float:
        return BaseMetrics.variance_neighbor_state(states, neighbor_indices, neighborhood_data)

class ModeNeighborState(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float:
        return BaseMetrics.mode_neighbor_state(states, neighbor_indices, neighborhood_data)

class NodeEdgeRatio(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float:
        """Compute Node Edge Ratio, ignoring ghost edges"""
        
        valid_neighbors = neighbor_indices
        
        # Calculate active edges using has_edge
        active_edges = 0
        for n in valid_neighbors:
            if n != -1 and neighborhood_data.has_edge(0, n) and states[n] > GlobalSettings.Visualization.NODE_VISIBILITY_THRESHOLD:
                active_edges += 1
        
        if len(valid_neighbors) == 0:
            return 0.0
        
        return float(len(valid_neighbors) / active_edges) if active_edges > 0 else float('inf')
    
class LocalEdgeDensity(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float:
        """Compute Local Edge Density, ignoring ghost edges"""
        
        valid_neighbors = neighbor_indices
        if len(valid_neighbors) < 2:
            return 0.0
        
        active_neighbors = valid_neighbors[states[valid_neighbors] > GlobalSettings.Visualization.NODE_VISIBILITY_THRESHOLD]
        
        local_edges = 0
        for i in range(len(active_neighbors)):
            for j in range(i + 1, len(active_neighbors)):
                if neighborhood_data.has_edge(active_neighbors[i], active_neighbors[j]):
                    local_edges += 1
        
        possible_edges = len(active_neighbors) * (len(active_neighbors) - 1) / 2
        return float(local_edges / possible_edges) if possible_edges > 0 else 0.0
    
class EdgeDivisibility(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float:
        """Compute Edge Divisibility, ignoring ghost edges"""
        
        valid_neighbors = neighbor_indices
        
        edge_count = 0
        for n in valid_neighbors:
            if n != -1 and neighborhood_data.has_edge(0, n):
                edge_count += 1
        
        # Return 1.0 if edge count is divisible by target (e.g., 3 or 4), 0.0 otherwise
        # Default to checking divisibility by 3 (for triangular patterns)
        return 1.0 if edge_count % 3 == 0 else 0.0
    
class MedianNeighborState(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float:
        """Compute Edge Divisibility, ignoring ghost edges"""
        
        valid_neighbors = neighbor_indices
        
        edge_count = 0
        for n in valid_neighbors:
            if n != -1 and neighborhood_data.has_edge(0, n):
                edge_count += 1
        
        # Return 1.0 if edge count is divisible by target (e.g., 3 or 4), 0.0 otherwise
        # Default to checking divisibility by 3 (for triangular patterns)
        return 1.0 if edge_count % 3 == 0 else 0.0
    
class Assortativity(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float: # Pass NeighborhoodData
        """Measures the assortativity coefficient of the local network"""
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) < 2:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the edges array
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if neighborhood_data.has_edge(subgraph_nodes[i], subgraph_nodes[j]): # Use has_edge
                    G.add_edge(i, j)
        
        # Calculate assortativity
        try:
            return float(nx.degree_assortativity_coefficient(G))
        except Exception:
            return 0.0  # Handle disconnected graphs
        
class BetweennessCentrality(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float: # Pass NeighborhoodData
        """Measures the betweenness centrality of a node in the local network"""
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the edges array
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if neighborhood_data.has_edge(subgraph_nodes[i], subgraph_nodes[j]): # Use has_edge
                    G.add_edge(i, j)
        
        # Calculate betweenness centrality
        try:
            centrality = nx.betweenness_centrality(G)
            return float(centrality[0])
        except Exception:
            return 0.0  # Handle disconnected graphs
        
class ClosenessCentrality(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float: # Pass NeighborhoodData
        """Measures the closeness centrality of a node in the local network"""
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the edges array
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if neighborhood_data.has_edge(subgraph_nodes[i], subgraph_nodes[j]): # Use has_edge
                    G.add_edge(i, j)
        
        # Calculate closeness centrality
        try:
            centrality = nx.closeness_centrality(G, u=0)
            return float(centrality[0])
        except Exception:
            return 0.0  # Handle disconnected graphs
        
class EigenvectorCentrality(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float: # Pass NeighborhoodData
        """Measures the eigenvector centrality of a node in the local network"""
        
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the edges array
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if neighborhood_data.has_edge(subgraph_nodes[i], subgraph_nodes[j]): # Use has_edge
                    G.add_edge(i, j)
        
        # Calculate eigenvector centrality
        try:
            centrality = nx.eigenvector_centrality(G)[0]
            return float(centrality)
        except Exception:
            return 0.0  # Handle disconnected graphs
        
class GraphLaplacianEnergy(RuleMetrics):
    @staticmethod
    def compute(states: npt.NDArray[np.float64],
                neighbor_indices: npt.NDArray[np.int64],
                neighborhood_data: 'NeighborhoodData') -> float: # Pass NeighborhoodData
        """Measures the graph laplacian energy of the local network"""
        
        
        valid_neighbors = neighbor_indices[neighbor_indices != -1]
        if len(valid_neighbors) == 0:
            return 0.0
        
        # Create a subgraph with the node and its neighbors
        subgraph_nodes = np.concatenate([[states], valid_neighbors])
        G = nx.Graph()
        for i, node in enumerate(subgraph_nodes):
            G.add_node(i)
        
        # Add edges from the edges array
        for i in range(1, len(subgraph_nodes)):
            for j in range(i + 1, len(subgraph_nodes)):
                if neighborhood_data.has_edge(subgraph_nodes[i], subgraph_nodes[j]): # Use has_edge
                    G.add_edge(i, j)
        
        # Calculate graph laplacian energy
        try:
            laplacian = nx.laplacian_matrix(G).asfptype()
            eigenvalues = np.linalg.eigvalsh(laplacian.toarray())
            energy = np.sum(np.abs(eigenvalues))
            return float(energy)
        except Exception:
            return 0.0  # Handle disconnected graphs


################################################
#            RULE LIBRARY HELPERS              #
################################################

@dataclass
class SymmetryData:
    """Container for symmetry calculation results"""
    score: float
    axes: np.ndarray
    rotation_score: float
    axis_alignments: Dict[int, float]
    symmetric_pairs: List[Tuple[int, int]]

@njit(cache=True)
def calculate_vectors_and_angles(center: np.ndarray, 
                               points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate vectors, angles, and distances from center to points"""
    vectors = points - center
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    distances = np.linalg.norm(vectors, axis=1)
    return vectors, angles, distances

@njit(cache=True)
def calculate_rotation_score(center: np.ndarray,
                           neighbor_positions: np.ndarray,
                           neighbor_states: np.ndarray,
                           rotational_order: int,
                           rotation_tolerance: float,
                           symmetry_tolerance: float) -> float:
    """Calculate rotational symmetry score"""
    if len(neighbor_positions) == 0:
        return 0.0
        
    vectors, angles, distances = calculate_vectors_and_angles(center, neighbor_positions)
    
    angle_step = 2*np.pi / rotational_order
    rotational_matches = 0
    total_checks = 0
    
    for i, angle in enumerate(angles):
        for rot in range(1, rotational_order):
            target_angle = angle + rot * angle_step
            if target_angle >= 2*np.pi:
                target_angle -= 2*np.pi
                
            angle_diffs = np.abs(angles - target_angle)
            angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
            closest_idx = np.argmin(angle_diffs)
            
            if (angle_diffs[closest_idx] <= rotation_tolerance and
                abs(distances[i] - distances[closest_idx]) <= symmetry_tolerance and
                abs(neighbor_states[i] - neighbor_states[closest_idx]) <= symmetry_tolerance):
                rotational_matches += 1
            total_checks += 1
    
    return rotational_matches / total_checks if total_checks > 0 else 0.0

@njit(cache=True)
def _calculate_symmetry_score(center: np.ndarray,
                           neighbor_positions: np.ndarray,
                           neighbor_states: np.ndarray,
                           symmetry_tolerance: float,
                           axis_angle_tolerance: float) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Calculate symmetry score and identify symmetric pairs
    
    Returns:
        Tuple of (symmetry_score, list_of_symmetric_pairs)
    """
    if len(neighbor_positions) == 0:
        return 0.0, []
        
    vectors, angles, distances = calculate_vectors_and_angles(center, neighbor_positions)
    
    symmetric_count = 0
    total_pairs = 0
    symmetric_pairs = []
    
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            angle_diff = abs(angles[i] - angles[j])
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            
            distance_match = abs(distances[i] - distances[j]) <= symmetry_tolerance
            state_match = abs(neighbor_states[i] - neighbor_states[j]) <= symmetry_tolerance
            
            if (abs(angle_diff - np.pi) <= axis_angle_tolerance and 
                distance_match and state_match):
                symmetric_count += 1
                symmetric_pairs.append((i, j))
            total_pairs += 1
    
    symmetry_score = symmetric_count / total_pairs if total_pairs > 0 else 0.0
    return symmetry_score, symmetric_pairs

def calculate_symmetry_score(center: np.ndarray,
                           neighbor_positions: np.ndarray,
                           neighbor_states: np.ndarray,
                           symmetry_tolerance: float,
                           axis_angle_tolerance: float) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Calculate symmetry score and identify symmetric pairs
    
    Returns:
        Tuple of (symmetry_score, list_of_symmetric_pairs)
    """
    return _calculate_symmetry_score(center, neighbor_positions, neighbor_states, symmetry_tolerance, axis_angle_tolerance)

@njit(cache=True)
def _calculate_axis_alignment(center: np.ndarray,
                           point: np.ndarray,
                           axes: np.ndarray) -> float:
    """Calculate alignment of point with symmetry axes"""
    if len(axes) == 0:
        return 0.0
        
    vector = point - center
    angle = np.arctan2(vector[1], vector[0])
    
    angle_diffs = np.abs(angle - axes)
    angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
    min_diff = np.min(angle_diffs)
    
    # Convert to alignment score (1 = perfect alignment, 0 = maximum misalignment)
    return 1.0 - (min_diff / (np.pi/2))

def calculate_axis_alignment(center: np.ndarray,
                           point: np.ndarray,
                           axes: np.ndarray) -> float:
    """Calculate alignment of point with symmetry axes"""
    return _calculate_axis_alignment(center, point, axes)

@dataclass
class HierarchyData:
    """Container for hierarchy calculation results"""
    level: int                      # Current level in hierarchy
    level_neighbors: Dict[int, List[int]]  # Neighbors by level {level: [node_indices]}
    influence_up: float             # Influence from lower levels
    influence_down: float           # Influence from higher levels
    level_density: float            # Density of connections within level
    cross_level_density: float      # Density of connections between levels

@njit(cache=True)
def _calculate_hierarchy_level(node_idx: int,
                            neighbor_indices: np.ndarray,
                            adj_list: Dict[int, Set[int]],
                            neighbor_levels: np.ndarray,
                            max_level: int) -> int:
    """
    Calculate hierarchical level for a node based on neighbor levels
    
    Args:
        node_idx: Index of node to calculate level for
        neighbor_indices: Array of neighbor indices
        adj_list: Adjacency list
        neighbor_levels: Array of known neighbor levels
        max_level: Maximum allowed hierarchy level
    
    Returns:
        Calculated level for the node
    """
    if len(neighbor_indices) == 0:
        return 0
        
    # Get levels of connected neighbors
    connected_neighbors = []
    for i in range(len(neighbor_indices)):
        n = neighbor_indices[i]
        if n in adj_list.get(node_idx, set()):
            connected_neighbors.append(n)
    
    if not len(connected_neighbors):
        return 0
        
    # Calculate level based on neighbors
    avg_neighbor_level = np.mean(neighbor_levels)
    max_neighbor_level = np.max(neighbor_levels)
    min_neighbor_level = np.min(neighbor_levels)
    
    # Determine level based on connection patterns
    if len(neighbor_levels) > 2:
        # Node connects multiple levels - likely a bridge node
        return min(int((max_neighbor_level + min_neighbor_level) / 2), max_level)
    else:
        # Node follows standard hierarchy
        return min(int(avg_neighbor_level + 0.5), max_level)

@staticmethod
@njit(cache=True)
def _calculate_level_influence(node_idx: int,
                            node_level: int,
                            neighbor_indices: np.ndarray,
                            adj_list: np.ndarray, # CHANGED
                            neighbor_levels: np.ndarray,
                            neighbor_states: np.ndarray) -> Tuple[float, float]:
    """
    Calculate influence from higher and lower levels
    
    Returns:
        Tuple of (influence_up, influence_down)
    """
    if len(neighbor_indices) == 0:
        return 0.0, 0.0
    
    influence_up = 0.0
    influence_down = 0.0
    total_neighbors = 0
    
    for i in range(len(neighbor_indices)):
        n = neighbor_indices[i]
        if n != -1: # ADDED CHECK
            if adj_list[node_idx, n]: # CHANGED
                neighbor_level = neighbor_levels[i]
                if neighbor_level > node_level:
                    influence_down += neighbor_states[i]
                elif neighbor_level < node_level:
                    influence_up += neighbor_states[i]
                total_neighbors += 1
            
    if total_neighbors == 0:
        return 0.0, 0.0
        
    return (float(influence_up / total_neighbors), float(influence_down / total_neighbors))

def calculate_level_influence(node_idx: int,
                            node_level: int,
                            neighborhood_data: NeighborhoodData,
                            current_levels: Dict[int, int]) -> Tuple[float, float]:
    """
    Calculate influence from higher and lower levels
    
    Returns:
        Tuple of (influence_up, influence_down)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0
        
    # Get levels of connected neighbors
    neighbor_levels = np.array([current_levels.get(n, 0) for n in neighbors], dtype=np.int64)
    neighbor_states = neighborhood_data.states[neighbors]
    
    # Create a local adjacency matrix for the node's neighbors
    local_adj_matrix = np.zeros((len(neighborhood_data.states), len(neighborhood_data.states)), dtype=np.bool_)
    for n in neighbors:
        if neighborhood_data.has_edge(node_idx, n):
            local_adj_matrix[node_idx, n] = True
            local_adj_matrix[n, node_idx] = True
    
    influence_up, influence_down = _calculate_level_influence(
        node_idx,
        node_level,
        neighbors,
        local_adj_matrix,
        neighbor_levels,
        neighbor_states
    )
    
    return influence_up, influence_down

def analyze_hierarchy(node_idx: int,
                     neighborhood_data: NeighborhoodData,
                     current_levels: Dict[int, int],
                     max_level: int) -> HierarchyData:
    """
    Comprehensive hierarchy analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        current_levels: Dictionary of current node levels
        max_level: Maximum allowed hierarchy level
    
    Returns:
        HierarchyData containing hierarchy metrics
    """
    # Calculate node's level
    neighbor_levels = np.array([current_levels.get(n, 0) for n in range(len(neighborhood_data.states))], dtype=np.int64)
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    
    # Create a local adjacency list for the node's neighbors
    local_adj_list = {}
    for n in neighbors:
        if neighborhood_data.has_edge(node_idx, n):
            local_adj_list[n] = 1  # Value doesn't matter, just presence in dict
    
    level = _calculate_hierarchy_level(node_idx, neighbors, local_adj_list, neighbor_levels, max_level)
    
    # Get neighbors by level
    level_neighbors = defaultdict(list)
    for n in neighbors:
        if neighborhood_data.has_edge(node_idx, n):
            n_level = current_levels.get(n, 0)
            level_neighbors[n_level].append(n)
            
    # Calculate influences
    influence_up, influence_down = calculate_level_influence(
        node_idx, level, neighborhood_data, current_levels
    )
    
    # Calculate density metrics
    same_level_edges = sum(1 for n in level_neighbors[level]
                          if neighborhood_data.has_edge(node_idx, n))
    level_density = (same_level_edges / len(level_neighbors[level]) 
                    if level_neighbors[level] else 0.0)
    
    cross_edges = sum(1 for l in level_neighbors if l != level
                     for n in level_neighbors[l]
                     if neighborhood_data.has_edge(node_idx, n))
    cross_level_density = (cross_edges / (len(neighbors) - len(level_neighbors[level]))
                          if len(neighbors) > len(level_neighbors[level]) else 0.0)
    
    return HierarchyData(
        level=level,
        level_neighbors=dict(level_neighbors),
        influence_up=influence_up,
        influence_down=influence_down,
        level_density=level_density,
        cross_level_density=cross_level_density
    )
     
@njit(cache=True)
def calculate_symmetry_axes(center: np.ndarray, 
                          neighbor_positions: np.ndarray,
                          num_axes: int) -> np.ndarray:
    """Calculate symmetry axes based on neighbor positions"""
    if len(neighbor_positions) == 0:
        return np.array([])
        
    # Calculate angles to all neighbors
    vectors = neighbor_positions - center
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    # Sort angles
    angles = np.sort(angles)
    
    # Calculate angle differences
    angle_diffs = np.diff(angles)
    angle_diffs = np.append(angle_diffs, 2*np.pi - (angles[-1] - angles[0]))
    
    # Find most evenly spaced angles
    best_axes = []
    min_variance = float('inf')
    
    # Try different starting angles
    for start_angle in angles:
        # Generate candidate axes
        axis_angles = np.array([
            start_angle + i * (2*np.pi / num_axes)
            for i in range(num_axes)
        ])
        
        # Calculate variance in neighbor distances to axes
        variances = []
        for axis in axis_angles:
            # Calculate distances from neighbors to this axis
            angle_diffs = np.abs(angles - axis)
            angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
            variances.append(np.var(angle_diffs))
            
        total_variance = np.mean(variances)
        
        if total_variance < min_variance:
            min_variance = total_variance
            best_axes = axis_angles
            
    return np.array(best_axes)

def analyze_symmetry(center: np.ndarray,
                    neighbor_positions: np.ndarray,
                    neighbor_states: np.ndarray,
                    params: RuleParameters) -> SymmetryData:
    """
    Comprehensive symmetry analysis returning all relevant metrics
    
    Args:
        center: Position of central node
        neighbor_positions: Positions of neighboring nodes
        neighbor_states: States of neighboring nodes
        params: Rule parameters containing symmetry-related settings
    
    Returns:
        SymmetryData containing all symmetry metrics
    """
    # Calculate basic symmetry score and pairs
    symmetry_score, symmetric_pairs = calculate_symmetry_score(
        center, neighbor_positions, neighbor_states,
        params.symmetry_tolerance, params.axis_angle_tolerance
    )
    
    # Calculate rotational symmetry
    rotation_score = calculate_rotation_score(
        center, neighbor_positions, neighbor_states,
        params.rotational_order, params.rotation_tolerance,
        params.symmetry_tolerance
    )
    
    # Calculate symmetry axes
    axes = calculate_symmetry_axes(
        center, neighbor_positions, params.num_symmetry_axes
    )
    
    # Calculate axis alignments for each neighbor
    axis_alignments = {}
    for i in range(len(neighbor_positions)):
        alignment = calculate_axis_alignment(center, neighbor_positions[i], axes)
        axis_alignments[i] = alignment
    
    return SymmetryData(
        score=symmetry_score,
        axes=axes,
        rotation_score=rotation_score,
        axis_alignments=axis_alignments,
        symmetric_pairs=symmetric_pairs
    )
    
@dataclass
class ResonanceData:
    """Container for resonance/wave pattern calculation results"""
    phase: float                    # Current phase (0-2π)
    frequency: float                # Current frequency
    amplitude: float                # Current amplitude
    phase_coherence: float         # Measure of phase alignment with neighbors
    frequency_match: float         # Measure of frequency match with neighbors
    standing_wave_strength: float  # Measure of standing wave pattern
    wave_nodes: List[int]          # Nodes participating in wave pattern
    resonance_score: float         # Overall resonance score

@njit(cache=True)
def calculate_phase_metrics(node_idx: int,
                          node_phase: float,
                          neighborhood_data: NeighborhoodData,
                          node_phases: Dict[int, float]) -> Tuple[float, float]:
    """
    Calculate phase coherence and frequency match with neighbors
    
    Returns:
        Tuple of (phase_coherence, frequency_match)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0
        
    # Calculate phase differences
    phase_diffs = []
    for n in neighbors:
        if n in node_phases:
            diff = abs(node_phase - node_phases[n])
            # Normalize to [-π, π]
            if diff > np.pi:
                diff = 2*np.pi - diff
            phase_diffs.append(diff)
            
    if not phase_diffs:
        return 0.0, 0.0
        
    # Phase coherence (1 = perfect alignment, 0 = random)
    phase_coherence = 1.0 - (np.mean(phase_diffs) / np.pi)
    
    # Frequency match (based on phase difference stability)
    freq_diffs = np.diff(sorted(phase_diffs))
    frequency_match = 1.0 - (np.std(freq_diffs) if len(freq_diffs) > 0 else 0.0)
    
    return float(phase_coherence), float(frequency_match)

@njit(cache=True)
def calculate_standing_wave_pattern(node_idx: int,
                                  node_positions: np.ndarray,
                                  neighborhood_data: NeighborhoodData,
                                  node_phases: Dict[int, float],
                                  wavelength: float) -> Tuple[float, List[int]]:
    """
    Calculate standing wave pattern strength and participating nodes
    
    Returns:
        Tuple of (standing_wave_strength, wave_node_indices)
    """
    if node_positions is None:
        return 0.0, []
        
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, []
        
    center_pos = node_positions[node_idx]
    wave_nodes = []
    phase_positions = []
    
    # Collect phases and positions
    for n in neighbors:
        if n in node_phases:
            pos = node_positions[n]
            dist = np.linalg.norm(pos - center_pos)
            if dist > 0:  # Avoid division by zero
                expected_phase = 2 * np.pi * (dist / wavelength)
                actual_phase = node_phases[n]
                # Compare expected vs actual phase
                phase_diff = abs(expected_phase - actual_phase) % (2 * np.pi)
                if phase_diff < np.pi/4:  # Node follows standing wave pattern
                    wave_nodes.append(n)
                    phase_positions.append(phase_diff)
                    
    if not phase_positions:
        return 0.0, []
        
    # Calculate standing wave strength
    strength = 1.0 - (np.mean(phase_positions) / (np.pi/2))
    return float(strength), wave_nodes

def analyze_resonance(node_idx: int,
                     node_positions: Optional[np.ndarray],
                     neighborhood_data: NeighborhoodData,
                     node_phases: Dict[int, float],
                     node_frequencies: Dict[int, float],
                     wavelength: float) -> ResonanceData:
    """
    Comprehensive resonance analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        node_positions: Array of node positions
        neighborhood_data: Network neighborhood data
        node_phases: Dictionary of current node phases
        node_frequencies: Dictionary of current node frequencies
        wavelength: Current wavelength for standing wave calculation
    
    Returns:
        ResonanceData containing all resonance metrics
    """
    # Get current node's phase and frequency
    phase = node_phases.get(node_idx, 0.0)
    frequency = node_frequencies.get(node_idx, 0.0)
    
    # Calculate phase metrics
    phase_coherence, frequency_match = calculate_phase_metrics(
        node_idx, phase, neighborhood_data, node_phases
    )
    
    # Calculate standing wave pattern if positions available
    if node_positions is not None:
        standing_wave_strength, wave_nodes = calculate_standing_wave_pattern(
            node_idx, node_positions, neighborhood_data, node_phases, wavelength
        )
    else:
        standing_wave_strength, wave_nodes = 0.0, []
    
    # Calculate amplitude based on neighbor states
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    amplitude = np.mean(neighborhood_data.states[neighbors]) if len(neighbors) > 0 else 0.0
    
    # Calculate overall resonance score
    resonance_score = (phase_coherence + frequency_match + standing_wave_strength) / 3.0
    
    return ResonanceData(
        phase=phase,
        frequency=frequency,
        amplitude=float(amplitude),
        phase_coherence=phase_coherence,
        frequency_match=frequency_match,
        standing_wave_strength=standing_wave_strength,
        wave_nodes=wave_nodes,
        resonance_score=resonance_score
    )

@dataclass
class ModuleData:
    """Container for module analysis results"""
    module_id: int                      # Current module identifier
    module_nodes: Set[int]              # Nodes in same module
    module_density: float               # Internal connection density
    external_density: float             # External connection density
    module_role: str                    # Role within module ('core', 'boundary', 'bridge')
    module_stability: float             # Measure of module stability
    inter_module_connections: Dict[int, Set[int]]  # Connections to other modules {module_id: {node_ids}}
    specialization_score: float         # Measure of module specialization

@njit(cache=True)
def calculate_module_metrics(node_idx: int,
                           neighborhood_data: NeighborhoodData,
                           module_assignments: Dict[int, int]) -> Tuple[float, float]:
    """
    Calculate internal and external connection densities for a node's module
    
    Returns:
        Tuple of (internal_density, external_density)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0
        
    current_module = module_assignments.get(node_idx, -1)
    if current_module == -1:
        return 0.0, 0.0
        
    internal_connections = 0
    external_connections = 0
    internal_possible = 0
    external_possible = 0
    
    for n in neighbors:
        if n == -1: # Skip invalid neighbors
            continue
        n_module = module_assignments.get(n, -1)
        if n_module == current_module:
            internal_possible += 1
            if neighborhood_data.has_edge(node_idx, n):  # Use has_edge
                internal_connections += 1
        else:
            external_possible += 1
            if neighborhood_data.has_edge(node_idx, n):  # Use has_edge
                external_connections += 1
                
    internal_density = (internal_connections / internal_possible 
                       if internal_possible > 0 else 0.0)
    external_density = (external_connections / external_possible 
                       if external_possible > 0 else 0.0)
    
    return float(internal_density), float(external_density)

@njit(cache=True)
def determine_module_role(node_idx: int,
                         neighborhood_data: NeighborhoodData,
                         module_assignments: Dict[int, int],
                         internal_density: float,
                         external_density: float) -> str:
    """
    Determine node's role within its module
    
    Returns:
        Role classification ('core', 'boundary', 'bridge')
    """
    if internal_density > 0.7 and external_density < 0.3:
        return 'core'
    elif external_density > 0.5:
        return 'bridge'
    else:
        return 'boundary'

@njit(cache=True)
def _calculate_module_stability(node_idx: int,
                               neighbor_indices: np.ndarray,
                               adj_list: Dict[int, Set[int]],
                               module_assignments: np.ndarray,
                               states: np.ndarray,
                               current_module: int) -> float:
    """
    Calculate stability of node's module based on state patterns
    
    Returns:
        Stability score (0-1)
    """
    if len(neighbor_indices) == 0:
        return 0.0
        
    # Get module neighbors
    module_neighbors = neighbor_indices[module_assignments[neighbor_indices] == current_module]
    
    if not len(module_neighbors):
        return 0.0
        
    # Calculate state pattern stability
    state_differences = np.abs(states[node_idx] - states[module_neighbors])
    
    return float(1.0 - (np.mean(state_differences) if len(state_differences) > 0 else 1.0))

def calculate_module_stability(node_idx: int,
                             neighborhood_data: NeighborhoodData,
                             module_assignments: Dict[int, int],
                             states: np.ndarray) -> float:
    """
    Calculate stability of node's module based on state patterns
    
    Returns:
        Stability score (0-1)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0
        
    current_module = module_assignments.get(node_idx, -1)
    if current_module == -1:
        return 0.0
    
    # Convert module assignments to numpy array
    module_assignments_array = np.array([module_assignments.get(n, -1) for n in range(len(states))], dtype=np.int64)
    
    # Create a local adjacency list for the node's neighbors
    local_adj_list = {}
    for n in neighbors:
        if neighborhood_data.has_edge(node_idx, n):
            local_adj_list[n] = 1  # Value doesn't matter, just presence in dict
    
    return _calculate_module_stability(
        node_idx,
        neighbors,
        local_adj_list,
        module_assignments_array,
        states,
        current_module
    )
     
def analyze_module(node_idx: int,
                     neighborhood_data: NeighborhoodData,
                     module_assignments: Dict[int, int],
                     specialization_scores: Dict[int, float]) -> ModuleData:
    """
    Comprehensive module analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        module_assignments: Dictionary of current module assignments
        specialization_scores: Dictionary of module specialization scores
    
    Returns:
        ModuleData containing all module metrics
    """
    current_module = module_assignments.get(node_idx, -1)
    
    # Get basic metrics
    internal_density, external_density = calculate_module_metrics(
        node_idx, neighborhood_data, module_assignments
    )
    
    # Determine role
    role = determine_module_role(
        node_idx, neighborhood_data, module_assignments,
        internal_density, external_density
    )
    
    # Calculate stability
    stability = calculate_module_stability(
        node_idx, neighborhood_data, module_assignments,
        neighborhood_data.states
    )
    
    # Get neighbors by level
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    level_neighbors = defaultdict(list)
    current_levels = {n: 0 for n in neighbors}  # Define current_levels
    for n in neighbors:
        if neighborhood_data.has_edge(node_idx, n):
            n_level = current_levels.get(n, 0)
            level_neighbors[n_level].append(n)
    
    # Define level based on some logic (example: max level of neighbors + 1)
    level = max(current_levels.values()) + 1 if current_levels else 0
    
    # Calculate influences
    influence_up, influence_down = calculate_level_influence(
        node_idx, level, neighborhood_data, current_levels
    )
    
    # Calculate density metrics
    same_level_edges = sum(1 for n in level_neighbors[level]
                          if neighborhood_data.has_edge(node_idx, n))
    level_density = (same_level_edges / len(level_neighbors[level]) 
                    if level_neighbors[level] else 0.0)
    
    cross_edges = sum(1 for l in level_neighbors if l != level
                     for n in level_neighbors[l]
                     if neighborhood_data.has_edge(node_idx, n))
    cross_level_density = (cross_edges / (len(neighbors) - len(level_neighbors[level]))
                          if len(neighbors) > len(level_neighbors[level]) else 0.0)
    
    # Calculate inter-module connections
    inter_module_connections: Dict[int, Set[int]] = defaultdict(set)
    for n in neighbors:
        neighbor_module = module_assignments.get(n, -1)
        if neighbor_module != current_module and neighbor_module != -1:
            inter_module_connections[neighbor_module].add(n)

    # Get specialization score
    specialization = specialization_scores.get(current_module, 0.0)

    module_nodes = {n for n in neighbors if module_assignments.get(n, -1) == current_module}

    return ModuleData(
        module_id=current_module,
        module_nodes=module_nodes,
        module_density=internal_density,
        external_density=external_density,
        module_role=role,
        module_stability=stability,
        inter_module_connections=dict(inter_module_connections),
        specialization_score=specialization
    )

@dataclass
class MemoryPattern:
    """Represents a stored pattern in the network"""
    pattern_id: int
    node_states: Dict[int, float]
    node_connections: Set[Tuple[int, int]]
    last_seen: int
    frequency: int
    stability: float
    influence: float

@dataclass
class MemoryData:
    """Container for memory analysis results"""
    matching_patterns: List[Tuple[MemoryPattern, float]]  # (pattern, match_score)
    best_match_score: float
    pattern_influence: float
    reconstruction_score: float
    memory_stability: float
    pattern_completion: float

@njit(cache=True)
def _calculate_pattern_match(current_states: np.ndarray,
                          adj_matrix: np.ndarray,
                          pattern_states: Dict[int, float],
                          pattern_connections: List[Tuple[int, int]],
                          neighborhood: np.ndarray) -> float:
    """
    Calculate how well current state matches a stored pattern
    
    Returns:
        Match score (0-1)
    """
    if len(neighborhood) == 0:
        return 0.0
        
    # Calculate state match
    state_matches = 0
    total_states = 0
    for idx in neighborhood:
        if idx in pattern_states:
            state_matches += 1 - abs(current_states[idx] - pattern_states[idx])
            total_states += 1
            
    state_score = state_matches / total_states if total_states > 0 else 0.0
    
    # Calculate connection match
    connection_matches = 0
    total_connections = 0
    for i in range(len(neighborhood)):
        for j in range(i + 1, len(neighborhood)):
            idx_i = neighborhood[i]
            idx_j = neighborhood[j]
            if idx_i < adj_matrix.shape[0] and idx_j < adj_matrix.shape[1]: # ADDED BOUNDS CHECK
                has_connection = adj_matrix[idx_i, idx_j]
            else:
                has_connection = False
            should_have_connection = (idx_i, idx_j) in pattern_connections
            if has_connection == should_have_connection:
                connection_matches += 1
            total_connections += 1
                
    connection_score = connection_matches / total_connections if total_connections > 0 else 0.0
    
    # Combine scores
    return float(0.6 * state_score + 0.4 * connection_score)

def calculate_pattern_match(current_states: np.ndarray,
                          adj_list: Dict[int, Set[int]],
                          pattern_states: Dict[int, float],
                          pattern_connections: Set[Tuple[int, int]],
                          neighborhood: np.ndarray) -> float:
    """
    Calculate how well current state matches a stored pattern
    
    Returns:
        Match score (0-1)
    """
    # Convert adj_list to adjacency matrix
    num_nodes = len(current_states)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.bool_)
    for i in range(num_nodes):
        for j in range(num_nodes):
            adj_matrix[i, j] = j in adj_list.get(i, set())
    
    return _calculate_pattern_match(current_states, adj_matrix, pattern_states, list(pattern_connections), neighborhood)

@njit(cache=True)
def calculate_pattern_completion(current_states: np.ndarray,
                               pattern_states: Dict[int, float],
                               neighborhood: np.ndarray) -> float:
    """
    Calculate pattern completion percentage
    
    Returns:
        Completion score (0-1)
    """
    if len(neighborhood) == 0:
        return 0.0
        
    matching_nodes = 0
    pattern_nodes = 0
    
    for idx in neighborhood:
        if idx in pattern_states:
            pattern_nodes += 1
            if abs(current_states[idx] - pattern_states[idx]) < 0.1:
                matching_nodes += 1
                
    return float(matching_nodes / pattern_nodes if pattern_nodes > 0 else 0.0)

def calculate_memory_stability(patterns: List[MemoryPattern],
                             current_step: int,
                             decay_rate: float) -> Dict[int, float]:
    """
    Calculate stability scores for stored patterns
    
    Returns:
        Dictionary of pattern_id to stability score
    """
    stability_scores = {}
    
    for pattern in patterns:
        # Calculate time-based decay
        time_factor = np.exp(-decay_rate * (current_step - pattern.last_seen))
        # Combine with frequency and stored stability
        stability = (0.4 * time_factor + 
                    0.3 * min(pattern.frequency / 10.0, 1.0) +
                    0.3 * pattern.stability)
        stability_scores[pattern.pattern_id] = stability
        
    return stability_scores

def analyze_memory(node_idx: int,
                  neighborhood_data: NeighborhoodData,
                  patterns: List[MemoryPattern],
                  current_step: int,
                  decay_rate: float) -> MemoryData:
    """
    Comprehensive memory analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        patterns: List of stored memory patterns
        current_step: Current simulation step
        decay_rate: Rate of memory decay
    
    Returns:
        MemoryData containing all memory metrics
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    
    # Calculate pattern matches
    matching_patterns = []
    best_match_score = 0.0
    
    for pattern in patterns:
        # Create a boolean array indicating which neighbors are connected
        current_connections = np.zeros((neighborhood_data.total_nodes, neighborhood_data.total_nodes), dtype=np.bool_)
        for i in range(neighborhood_data.total_nodes):
            for j in range(neighborhood_data.total_nodes):
                current_connections[i, j] = neighborhood_data.has_edge(i, j)
        
        match_score = calculate_pattern_match(
            neighborhood_data.states,
            {i: {j for j in range(neighborhood_data.total_nodes) if current_connections[i, j]} for i in range(neighborhood_data.total_nodes)},
            pattern.node_states,
            pattern.node_connections,
            neighbors
        )
        if match_score > 0:
            matching_patterns.append((pattern, match_score))
            best_match_score = max(best_match_score, match_score)
    
    # Sort by match score
    matching_patterns.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate stability scores
    stability_scores = calculate_memory_stability(patterns, current_step, decay_rate)
    
    # Calculate pattern influence
    pattern_influence = 0.0
    if matching_patterns:
        # Weight influence by match score and pattern stability
        for pattern, match_score in matching_patterns:
            stability = stability_scores[pattern.pattern_id]
            pattern_influence += match_score * stability * pattern.influence
        pattern_influence /= len(matching_patterns)
    
    # Calculate reconstruction score
    reconstruction_score = 0.0
    if matching_patterns:
        best_pattern = matching_patterns[0][0]
        reconstruction_score = calculate_pattern_completion(
            neighborhood_data.states,
            best_pattern.node_states,
            neighbors
        )
    
    # Calculate overall memory stability
    memory_stability = float(np.mean(list(stability_scores.values()))) if stability_scores else 0.0
    
    # Calculate pattern completion tendency
    pattern_completion = reconstruction_score if matching_patterns else 0.0
    
    return MemoryData(
        matching_patterns=matching_patterns,
        best_match_score=best_match_score,
        pattern_influence=pattern_influence,
        reconstruction_score=reconstruction_score,
        memory_stability=memory_stability,
        pattern_completion=pattern_completion
    )
    
@dataclass
class FlowData:
    """Container for flow analysis results"""
    flow_direction: np.ndarray          # Primary direction of flow
    flow_strength: float                # Strength of flow
    bottleneck_score: float            # Measure of flow bottleneck (0-1, 1=severe)
    path_redundancy: float             # Measure of alternative paths
    capacity_utilization: float        # Current flow capacity usage
    upstream_pressure: float           # Pressure from upstream nodes
    downstream_pressure: float         # Pressure from downstream nodes
    flow_stability: float             # Measure of flow stability
    critical_paths: Set[Tuple[int, int]]  # Critical path connections

@njit(cache=True)
def _calculate_flow_direction(node_idx: int,
                           node_positions: npt.NDArray[np.float64],
                           neighbor_indices: npt.NDArray[np.int64],
                           neighborhood_data: 'NeighborhoodData',  # Pass NeighborhoodData
                           neighbor_states: npt.NDArray[np.float64],
                           flow_bias: npt.NDArray[np.float64]) -> Tuple[np.ndarray, float]:
    """
    Calculate primary flow direction and strength
    
    Returns:
        Tuple of (direction_vector, flow_strength)
    """
    if len(neighbor_indices) == 0:
        return np.zeros(3), 0.0
        
    # Calculate weighted direction based on connections and states
    flow_vector = np.zeros(3)
    total_weight = 0.0
    
    for n in neighbor_indices:
        if n != -1 and neighborhood_data.has_edge(node_idx, n):  # Use has_edge
            direction = node_positions[n] - node_positions[node_idx]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
                # Weight by state and alignment with global bias
                weight = (neighbor_states[n] * 
                         (1.0 + np.dot(direction, flow_bias)))
                flow_vector += direction * weight
                total_weight += weight
    
    if total_weight > 0:
        flow_vector = flow_vector / total_weight
        
    flow_strength = np.linalg.norm(flow_vector)
    if flow_strength > 0:
        flow_vector = flow_vector / flow_strength
        
    return flow_vector, float(flow_strength)

def calculate_flow_direction(node_idx: int,
                           node_positions: npt.NDArray[np.float64],
                           neighborhood_data: NeighborhoodData,
                           flow_bias: npt.NDArray[np.float64]) -> Tuple[np.ndarray, float]:
    """
    Calculate primary flow direction and strength
    
    Returns:
        Tuple of (direction_vector, flow_strength)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    return _calculate_flow_direction(
        node_idx,
        node_positions,
        neighbors,
        neighborhood_data,  # Pass NeighborhoodData
        neighborhood_data.states,  # Pass states
        flow_bias
    )
      
@njit(cache=True)
def _calculate_bottleneck(node_idx: int,
                        neighbor_indices: npt.NDArray[np.int64],
                        neighborhood_data: 'NeighborhoodData',  # Pass NeighborhoodData
                        flow_direction: npt.NDArray[np.float64],
                        node_positions: npt.NDArray[np.float64]) -> Tuple[float, float]:
    """
    Calculate bottleneck severity and path redundancy
    
    Returns:
        Tuple of (bottleneck_score, redundancy_score)
    """
    if len(neighbor_indices) == 0:
        return 1.0, 0.0
        
    # Count upstream and downstream connections
    upstream_count = 0
    downstream_count = 0
    
    for n in neighbor_indices:
        if n != -1 and neighborhood_data.has_edge(node_idx, n):  # Use has_edge
            direction = node_positions[n] - node_positions[node_idx]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
                alignment = np.dot(direction, flow_direction)
                if alignment > 0.1:  # Downstream
                    downstream_count += 1
                elif alignment < -0.1:  # Upstream
                    upstream_count += 1
    
    # Calculate bottleneck score
    min_connections = min(upstream_count, downstream_count)
    max_connections = max(upstream_count, downstream_count)
    bottleneck = 1.0 - (min_connections / max_connections if max_connections > 0 else 0.0)
    
    # Calculate path redundancy
    redundancy = min_connections / len(neighbor_indices) if len(neighbor_indices) > 0 else 0.0
    
    return float(bottleneck), float(redundancy)

def calculate_bottleneck(node_idx: int,
                        neighborhood_data: NeighborhoodData,
                        flow_direction: np.ndarray,
                        node_positions: np.ndarray) -> Tuple[float, float]:
    """
    Calculate bottleneck severity and path redundancy
    
    Returns:
        Tuple of (bottleneck_score, redundancy_score)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    return _calculate_bottleneck(
        node_idx,
        neighbors,
        neighborhood_data,  # Pass NeighborhoodData
        flow_direction,
        node_positions
    )
           
@njit(cache=True)
def calculate_flow_pressure(node_idx: int,
                          neighborhood_data: 'NeighborhoodData',  # Pass NeighborhoodData
                          flow_direction: npt.NDArray[np.float64],
                          node_positions: npt.NDArray[np.float64],
                          neighbor_states: npt.NDArray[np.float64]) -> Tuple[float, float]:
    """
    Calculate upstream and downstream pressure
    
    Returns:
        Tuple of (upstream_pressure, downstream_pressure)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0
        
    upstream_pressure = 0.0
    downstream_pressure = 0.0
    upstream_count = 0
    downstream_count = 0
    
    for n in neighbors:
        if n != -1 and neighborhood_data.has_edge(node_idx, n):  # Use has_edge and check for valid index
            direction = node_positions[n] - node_positions[node_idx]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
                alignment = np.dot(direction, flow_direction)
                if alignment > 0.1:  # Downstream
                    downstream_pressure += neighbor_states[n]
                    downstream_count += 1
                elif alignment < -0.1:  # Upstream
                    upstream_pressure += neighbor_states[n]
                    upstream_count += 1
    
    upstream_pressure = (upstream_pressure / upstream_count 
                        if upstream_count > 0 else 0.0)
    downstream_pressure = (downstream_pressure / downstream_count 
                         if downstream_count > 0 else 0.0)
    
    return float(upstream_pressure), float(downstream_pressure)
      
def analyze_flow(node_idx: int,
                neighborhood_data: NeighborhoodData,
                node_positions: np.ndarray,
                flow_bias: np.ndarray,
                capacity_threshold: float) -> 'FlowData':
    """
    Comprehensive flow analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        node_positions: Array of node positions
        flow_bias: Global flow direction bias
        capacity_threshold: Threshold for capacity utilization
    
    Returns:
        FlowData containing all flow metrics
    """
    # Calculate primary flow direction and strength
    flow_direction, flow_strength = calculate_flow_direction(
        node_idx, node_positions, neighborhood_data, flow_bias
    )
    
    # Calculate bottleneck and redundancy
    bottleneck_score, path_redundancy = calculate_bottleneck(
        node_idx, neighborhood_data, flow_direction, node_positions
    )
    
    # Calculate pressures
    upstream_pressure, downstream_pressure = calculate_flow_pressure(
        node_idx, neighborhood_data, flow_direction, node_positions, neighborhood_data.states # Pass neighborhood_data
    )
    
@dataclass
class Territory:
    """Represents a controlled territory in the network"""
    territory_id: int
    controller_id: int
    member_nodes: Set[int]
    boundary_nodes: Set[int]
    resource_level: float
    strength: float
    alliances: Set[int]  # Set of allied territory IDs

@dataclass
class CompetitionData:
    """Container for competition analysis results"""
    territory: Optional[Territory]     # Current territory
    territory_position: str            # Position in territory ('core', 'boundary', 'contested')
    resource_access: float            # Access to resources (0-1)
    competitive_pressure: float       # Pressure from competitors (0-1)
    alliance_strength: float          # Strength from alliances (0-1)
    territorial_stability: float      # Stability of territory (0-1)
    expansion_potential: float        # Potential for expansion (0-1)
    defense_strength: float          # Defensive capability (0-1)
    contested_neighbors: Set[int]     # Neighboring nodes under competition

@njit(cache=True)
def _calculate_territory_metrics(node_idx: int,
                              neighbor_indices: np.ndarray,
                              territory_assignments: np.ndarray,
                              resource_distribution: np.ndarray,
                              current_territory: int) -> Tuple[float, float, int]:
    """
    Calculate territory-related metrics for a node
    
    Returns:
        Tuple of (resource_access, competitive_pressure, position)
    """
    if len(neighbor_indices) == 0:
        return 0.0, 0.0, 0
        
    # Calculate resource access
    resource_access = resource_distribution[node_idx]
    different_territory_count = 0
    
    for n in neighbor_indices:
        if territory_assignments[n] == current_territory:
            resource_access += resource_distribution[n] * 0.5
        else:
            different_territory_count += 1
            
    # Calculate competitive pressure
    competitive_pressure = different_territory_count / len(neighbor_indices)
    
    # Determine position
    if competitive_pressure == 0:
        position = 1  # core
    elif competitive_pressure > 0.5:
        position = 2  # contested
    else:
        position = 3  # boundary
        
    return float(resource_access), float(competitive_pressure), int(position)

def calculate_territory_metrics(node_idx: int,
                              neighborhood_data: NeighborhoodData,
                              territory_assignments: Dict[int, int],
                              resource_distribution: np.ndarray) -> Tuple[float, float, str]:
    """
    Calculate territory-related metrics for a node
    
    Returns:
        Tuple of (resource_access, competitive_pressure, position)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0, 'isolated'
        
    current_territory = territory_assignments.get(node_idx, -1)
    if current_territory == -1:
        return 0.0, 1.0, 'contested'
        
    # Convert territory assignments to numpy array
    territory_assignments_array = np.array([territory_assignments.get(n, -1) for n in range(len(neighborhood_data.states))], dtype=np.int64)
    
    resource_access, competitive_pressure, position_code = _calculate_territory_metrics(
        node_idx,
        neighbors,
        territory_assignments_array,
        resource_distribution,
        current_territory
    )
    
    # Convert position code to string
    if position_code == 1:
        position = 'core'
    elif position_code == 2:
        position = 'contested'
    else:
        position = 'boundary'
        
    return resource_access, competitive_pressure, position

@njit(cache=True)
def _calculate_alliance_metrics(node_idx: int,
                             neighbor_indices: np.ndarray,
                             territory_assignments: np.ndarray,
                             alliances: np.ndarray,
                             states: np.ndarray,
                             current_territory: int) -> Tuple[float, float]:
    """
    Calculate alliance-related metrics
    
    Returns:
        Tuple of (alliance_strength, defense_strength)
    """
    if len(neighbor_indices) == 0:
        return 0.0, 0.0
        
    # Calculate alliance strength
    allied_neighbors = 0
    friendly_strength = 0.0
    total_strength = 0.0
    
    for n in neighbor_indices:
        n_territory = territory_assignments[n]
        if n_territory == current_territory or n_territory in alliances:
            allied_neighbors += 1
            friendly_strength += states[n]
        total_strength += states[n]
            
    alliance_strength = allied_neighbors / len(neighbor_indices)
    defense_strength = friendly_strength / total_strength if total_strength > 0 else 0.0
    
    return float(alliance_strength), float(defense_strength)

def calculate_alliance_metrics(node_idx: int,
                             neighborhood_data: NeighborhoodData,
                             territory_assignments: Dict[int, int],
                             alliances: Dict[int, Set[int]]) -> Tuple[float, float]:
    """
    Calculate alliance-related metrics
    
    Returns:
        Tuple of (alliance_strength, defense_strength)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0
        
    current_territory = territory_assignments.get(node_idx, -1)
    if current_territory == -1:
        return 0.0, 0.0
        
    # Convert territory assignments and alliances to numpy arrays
    territory_assignments_array = np.array([territory_assignments.get(n, -1) for n in range(len(neighborhood_data.states))], dtype=np.int64)
    alliances_set = alliances.get(current_territory, set())
    alliances_array = np.array(list(alliances_set), dtype=np.int64)
    
    alliance_strength, defense_strength = _calculate_alliance_metrics(
        node_idx,
        neighbors,
        territory_assignments_array,
        alliances_array,
        neighborhood_data.states,
        current_territory
    )
    
    return alliance_strength, defense_strength

def calculate_expansion_potential(node_idx: int,
                                neighborhood_data: NeighborhoodData,
                                territory_assignments: Dict[int, int],
                                resource_distribution: np.ndarray,
                                territory_strengths: Dict[int, float]) -> float:
    """
    Calculate potential for territory expansion
    
    Returns:
        Expansion potential score (0-1)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0
        
    current_territory = territory_assignments.get(node_idx, -1)
    if current_territory == -1:
        return 0.0
        
    current_strength = territory_strengths.get(current_territory, 0.0)
    
    # Evaluate expansion opportunities
    expansion_score = 0.0
    opportunities = 0
    
    for n in neighbors:
        n_territory = territory_assignments.get(n, -1)
        if n_territory != current_territory:
            # Consider resource value and relative strength
            n_strength = territory_strengths.get(n_territory, 0.0)
            strength_ratio = current_strength / (n_strength + 0.1)  # Avoid division by zero
            resource_value = resource_distribution[n]
            
            opportunity_score = resource_value * strength_ratio
            expansion_score += opportunity_score
            opportunities += 1
            
    return float(expansion_score / opportunities if opportunities > 0 else 0.0)

def analyze_competition(node_idx: int,
                       neighborhood_data: NeighborhoodData,
                       territories: Dict[int, Territory],
                       territory_assignments: Dict[int, int],
                       resource_distribution: np.ndarray) -> CompetitionData:
    """
    Comprehensive competition analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        territories: Dictionary of territory objects
        territory_assignments: Dictionary of node territory assignments
        resource_distribution: Array of resource values for nodes
    
    Returns:
        CompetitionData containing all competition metrics
    """
    current_territory_id = territory_assignments.get(node_idx, -1)
    current_territory = territories.get(current_territory_id, None)
    
    # Calculate basic metrics
    resource_access, competitive_pressure, position = calculate_territory_metrics(
        node_idx, neighborhood_data, territory_assignments, resource_distribution
    )
    
    # Calculate alliance metrics
    alliance_strength, defense_strength = calculate_alliance_metrics(
        node_idx, neighborhood_data, territory_assignments,
        {t.territory_id: t.alliances for t in territories.values()}
    )
    
    # Calculate territory strengths
    territory_strengths = {
        t.territory_id: t.strength for t in territories.values()
    }
    
    # Calculate expansion potential
    expansion_potential = calculate_expansion_potential(
        node_idx, neighborhood_data, territory_assignments,
        resource_distribution, territory_strengths
    )
    
    # Calculate territorial stability
    if current_territory:
        stability = (defense_strength * (1.0 - competitive_pressure) * 
                    (0.5 + 0.5 * alliance_strength))
    else:
        stability = 0.0
    
    # Identify contested neighbors
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    contested_neighbors = {
        n for n in neighbors
        if territory_assignments.get(n, -1) != current_territory_id
    }
    
    return CompetitionData(
        territory=current_territory,
        territory_position=position,
        resource_access=resource_access,
        competitive_pressure=competitive_pressure,
        alliance_strength=alliance_strength,
        territorial_stability=stability,
        expansion_potential=expansion_potential,
        defense_strength=defense_strength,
        contested_neighbors=contested_neighbors
    )
    
@dataclass
class FractalPattern:
    """Represents a fractal pattern in the network"""
    pattern_id: int
    scale: float                    # Current scale of pattern
    center_node: int               # Central node of pattern
    member_nodes: Set[int]         # Nodes in pattern
    symmetry_points: List[np.ndarray]  # Key symmetry points
    recursion_depth: int           # Current depth of recursion
    similarity_score: float        # Self-similarity measure

@dataclass
class FractalData:
    """Container for fractal analysis results"""
    current_pattern: Optional[FractalPattern]  # Current fractal pattern
    pattern_role: str              # Role in pattern ('center', 'branch', 'terminal')
    scale_position: float          # Position in current scale (0-1)
    self_similarity: float         # Local self-similarity measure
    fractal_dimension: float       # Local fractal dimension
    branching_score: float         # Measure of branching completeness
    scale_transition_potential: float  # Potential for scale transition
    pattern_stability: float       # Stability of fractal pattern
    recursive_patterns: List[FractalPattern]  # Nested patterns

@njit(cache=True)
def _calculate_fractal_metrics(node_idx: int,
                            neighbor_indices: np.ndarray,
                            node_positions: np.ndarray,
                            reference_scale: float) -> Tuple[float, float, int]:
    """
    Calculate basic fractal metrics for a node
    
    Returns:
        Tuple of (self_similarity, fractal_dimension, pattern_role)
    """
    if len(neighbor_indices) == 0:
        return 0.0, 1.0, 2  # terminal
        
    # Calculate distances to neighbors
    distances = np.zeros(len(neighbor_indices))
    for i, n in enumerate(neighbor_indices):
        diff = node_positions[n] - node_positions[node_idx]
        distances[i] = np.sqrt(np.sum(diff * diff))
    
    # Calculate scale ratios
    scale_ratios = distances / reference_scale
    
    # Calculate self-similarity through distance distribution
    unique_ratios = np.unique(np.round(scale_ratios, decimals=2))
    self_similarity = len(unique_ratios) / len(distances) if len(distances) > 0 else 0.0
    
    # Estimate local fractal dimension
    if len(distances) > 1:
        log_counts = np.log(np.arange(1, len(distances) + 1))
        log_distances = np.log(np.sort(distances))
        fractal_dim = np.abs(np.polyfit(log_distances, log_counts, 1)[0])
    else:
        fractal_dim = 1.0
    
    # Determine pattern role
    if len(neighbor_indices) >= 3 and self_similarity > 0.7:
        role = 0  # center
    elif len(neighbor_indices) == 1:
        role = 2  # terminal
    else:
        role = 1  # branch
        
    return float(self_similarity), float(fractal_dim), int(role)

def calculate_fractal_metrics(node_idx: int,
                            neighborhood_data: NeighborhoodData,
                            node_positions: np.ndarray,
                            reference_scale: float) -> Tuple[float, float, str]:
    """
    Calculate basic fractal metrics for a node
    
    Returns:
        Tuple of (self_similarity, fractal_dimension, pattern_role)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    self_similarity, fractal_dim, role_code = _calculate_fractal_metrics(
        node_idx,
        neighbors,
        node_positions,
        reference_scale
    )
    
    # Convert role code to string
    if role_code == 0:
        role = 'center'
    elif role_code == 1:
        role = 'branch'
    else:
        role = 'terminal'
        
    return self_similarity, fractal_dim, role

@njit(cache=True)
def _calculate_branching_metrics(node_idx: int,
                              neighbor_indices: np.ndarray,
                              node_positions: np.ndarray,
                              target_branching: int) -> Tuple[float, float]:
    """
    Calculate branching-related metrics
    
    Returns:
        Tuple of (branching_score, scale_transition_potential)
    """
    if len(neighbor_indices) == 0:
        return 0.0, 0.0
        
    # Calculate angular distribution of neighbors
    angles = np.zeros(len(neighbor_indices))
    for i, n in enumerate(neighbor_indices):
        diff = node_positions[n] - node_positions[node_idx]
        angles[i] = np.arctan2(diff[1], diff[0])
    
    # Sort angles and calculate angular gaps
    angles = np.sort(angles)
    gaps = np.diff(angles)
    gaps = np.append(gaps, 2*np.pi - (angles[-1] - angles[0]))
    
    # Calculate branching score based on angular distribution
    ideal_gap = 2*np.pi / target_branching
    gap_scores = 1.0 - np.abs(gaps - ideal_gap) / np.pi
    branching_score = float(np.mean(gap_scores))
    
    # Calculate scale transition potential
    if len(neighbor_indices) == target_branching:
        scale_potential = branching_score
    else:
        scale_potential = branching_score * (len(neighbor_indices) / target_branching)
        
    return float(branching_score), float(scale_potential)

def calculate_branching_metrics(node_idx: int,
                              neighborhood_data: NeighborhoodData,
                              node_positions: np.ndarray,
                              target_branching: int) -> Tuple[float, float]:
    """
    Calculate branching-related metrics
    
    Returns:
        Tuple of (branching_score, scale_transition_potential)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    return _calculate_branching_metrics(
        node_idx,
        neighbors,
        node_positions,
        target_branching
    )
    
def calculate_pattern_stability(node_idx: int,
                              neighborhood_data: NeighborhoodData,
                              pattern: Optional[FractalPattern],
                              self_similarity: float,
                              branching_score: float) -> float:
    """
    Calculate stability of fractal pattern
    
    Returns:
        Stability score (0-1)
    """
    if pattern is None:
        return 0.0
        
    # Consider multiple stability factors
    stability_factors = [
        self_similarity,                    # Local self-similarity
        branching_score,                    # Branching completeness
        pattern.similarity_score,           # Pattern-wide similarity
        1.0 - (pattern.recursion_depth / 5.0)  # Depth penalty
    ]
    
    # Calculate member state consistency
    if pattern.member_nodes:
        states = [neighborhood_data.states[n] for n in pattern.member_nodes]
        state_consistency = 1.0 - np.std(states) if states else 0.0
        stability_factors.append(float(state_consistency))
    
    return float(np.mean(stability_factors))

def analyze_fractal(node_idx: int,
                   neighborhood_data: NeighborhoodData,
                   node_positions: np.ndarray,
                   patterns: Dict[int, FractalPattern],
                   reference_scale: float,
                   target_branching: int) -> FractalData:
    """
    Comprehensive fractal analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        node_positions: Array of node positions
        patterns: Dictionary of existing fractal patterns
        reference_scale: Base scale for pattern measurement
        target_branching: Target number of branches
    
    Returns:
        FractalData containing all fractal metrics
    """
    # Calculate basic metrics
    self_similarity, fractal_dimension, pattern_role = calculate_fractal_metrics(
        node_idx, neighborhood_data, node_positions, reference_scale
    )
    
    # Calculate branching metrics
    branching_score, scale_transition_potential = calculate_branching_metrics(
        node_idx, neighborhood_data, node_positions, target_branching
    )
    
    # Find current pattern
    current_pattern = None
    for pattern in patterns.values():
        if node_idx in pattern.member_nodes:
            current_pattern = pattern
            break
    
    # Calculate scale position
    if current_pattern:
        center_pos = node_positions[current_pattern.center_node]
        node_pos = node_positions[node_idx]
        distance = np.linalg.norm(node_pos - center_pos)
        scale_position = distance / (current_pattern.scale * reference_scale)
    else:
        scale_position = 0.0
    
    # Calculate pattern stability
    pattern_stability = calculate_pattern_stability(
        node_idx, neighborhood_data, current_pattern,
        self_similarity, branching_score
    )
    
    # Find recursive patterns
    recursive_patterns = []
    if current_pattern:
        for pattern in patterns.values():
            if (pattern.recursion_depth > current_pattern.recursion_depth and
                pattern.center_node in current_pattern.member_nodes):
                recursive_patterns.append(pattern)
    
    return FractalData(
        current_pattern=current_pattern,
        pattern_role=pattern_role,
        scale_position=float(scale_position),
        self_similarity=self_similarity,
        fractal_dimension=fractal_dimension,
        branching_score=branching_score,
        scale_transition_potential=scale_transition_potential,
        pattern_stability=pattern_stability,
        recursive_patterns=recursive_patterns
    )

@dataclass
class Organism:
    """Represents a living organism in the network"""
    organism_id: int
    cells: Set[int]                # Node indices that make up the organism
    age: int                       # Age in simulation steps
    energy: float                  # Current energy level
    genome: Dict[str, float]       # Genetic parameters
    generation: int                # Generational distance from origin
    parent_id: Optional[int]       # ID of parent organism
    offspring: Set[int]            # IDs of child organisms
    phenotype: str                 # Current expressed behavior type
    fitness: float                 # Current fitness score

@dataclass
class LifeData:
    """Container for artificial life analysis results"""
    organism: Optional[Organism]    # Current organism
    cell_role: str                 # Role in organism ('core', 'membrane', 'specialized')
    energy_balance: float          # Current energy input/output balance
    metabolic_rate: float          # Rate of energy consumption
    reproduction_potential: float   # Potential for reproduction
    mutation_rate: float           # Current mutation rate
    adaptation_score: float        # Measure of environmental adaptation
    survival_score: float          # Overall survival probability
    interaction_partners: Set[int]  # Organisms interacting with this cell

@njit(cache=True)
def _calculate_metabolic_metrics(node_idx: int,
                              neighbor_indices: npt.NDArray[np.int64],
                              neighborhood_data: 'NeighborhoodData',  # Pass NeighborhoodData
                              energy_distribution: npt.NDArray[np.float64],
                              base_metabolism: float) -> Tuple[float, float]:
    """
    Calculate metabolism-related metrics
    
    Returns:
        Tuple of (energy_balance, metabolic_rate)
    """
    if len(neighbor_indices) == 0:
        return 0.0, base_metabolism
        
    # Calculate energy intake from environment
    energy_intake = energy_distribution[node_idx]
    
    # Calculate energy exchange with neighbors
    energy_exchange = 0.0
    total_neighbors = 0
    
    for n in neighbor_indices:
        if n != -1 and neighborhood_data.has_edge(node_idx, n):  # Use has_edge
            energy_diff = energy_distribution[n] - energy_distribution[node_idx]
            energy_exchange += energy_diff * 0.1  # Exchange rate
            total_neighbors += 1
            
    # Calculate metabolic rate based on activity
    metabolic_rate = base_metabolism * (1.0 + 0.1 * total_neighbors)
    
    # Calculate energy balance
    energy_balance = energy_intake + energy_exchange - metabolic_rate
    
    return float(energy_balance), float(metabolic_rate)

def calculate_metabolic_metrics(node_idx: int,
                              neighborhood_data: NeighborhoodData,
                              energy_distribution: np.ndarray,
                              base_metabolism: float) -> Tuple[float, float]:
    """
    Calculate metabolism-related metrics
    
    Returns:
        Tuple of (energy_balance, metabolic_rate)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    return _calculate_metabolic_metrics( # Pass neighborhood_data
        node_idx,
        neighbors,
        neighborhood_data, # Pass neighborhood_data
        energy_distribution,
        base_metabolism
    )
      
@njit(cache=True)
def _calculate_reproduction_metrics(node_idx: int,
                                 neighbor_indices: np.ndarray,
                                 neighborhood_data: 'NeighborhoodData', # Pass NeighborhoodData
                                 organism_assignments: np.ndarray,
                                 energy_levels: np.ndarray,
                                 reproduction_threshold: float,
                                 current_organism: int) -> Tuple[float, float]:
    """
    Calculate reproduction-related metrics
    
    Returns:
        Tuple of (reproduction_potential, mutation_rate)
    """
    if len(neighbor_indices) == 0:
        return 0.0, 0.0
        
    # Calculate available space for reproduction
    empty_neighbors = 0
    for n in neighbor_indices:
        if n != -1 and organism_assignments[n] == -1 and neighborhood_data.has_edge(node_idx, n): # Use has_edge
            empty_neighbors += 1
    space_factor = empty_neighbors / len(neighbor_indices) if len(neighbor_indices) > 0 else 0.0 # Avoid division by zero
    
    # Calculate energy-based reproduction potential
    energy_level = energy_levels[current_organism]
    energy_factor = max(0.0, min(1.0, energy_level / reproduction_threshold))
    
    # Calculate base mutation rate inversely proportional to fitness
    base_mutation_rate = 0.1
    mutation_rate = base_mutation_rate * (1.0 - energy_factor)
    
    # Calculate final reproduction potential
    reproduction_potential = space_factor * energy_factor
    
    return float(reproduction_potential), float(mutation_rate)

def calculate_reproduction_metrics(node_idx: int,
                                 neighborhood_data: NeighborhoodData,
                                 organism_assignments: Dict[int, int],
                                 energy_levels: Dict[int, float],
                                 reproduction_threshold: float) -> Tuple[float, float]:
    """
    Calculate reproduction-related metrics
    
    Returns:
        Tuple of (reproduction_potential, mutation_rate)
    """
    neighbors = neighborhood_data.get_neighbor_indices(node_idx)
    if len(neighbors) == 0:
        return 0.0, 0.0
        
    current_organism = organism_assignments.get(node_idx, -1)
    if current_organism == -1:
        return 0.0, 0.0
    
    # Convert organism assignments to numpy array
    organism_assignments_array = np.array([organism_assignments.get(n, -1) for n in range(len(neighborhood_data.states))], dtype=np.int64)
    
    # Convert energy levels to a NumPy array, handling missing values
    energy_levels_array = np.array([energy_levels.get(organism_id, 0.0) for organism_id in range(len(energy_levels))], dtype=np.float64)

    return _calculate_reproduction_metrics(
        node_idx,
        neighbors,
        neighborhood_data, # Pass neighborhood_data
        organism_assignments_array,
        energy_levels_array,
        reproduction_threshold,
        current_organism
    )

def calculate_adaptation_score(node_idx: int,
                             neighborhood_data: NeighborhoodData,
                             organism: Optional[Organism],
                             environment_factors: Dict[str, float]) -> float:
    """
    Calculate environmental adaptation score
    
    Returns:
        Adaptation score (0-1)
    """
    if organism is None:
        return 0.0
        
    # Consider multiple adaptation factors
    adaptation_factors = []
    
    # Genome-environment match
    for factor, value in environment_factors.items():
        if factor in organism.genome:
            match_score = 1.0 - abs(organism.genome[factor] - value)
            adaptation_factors.append(match_score)
    
    # Age-based adaptation
    age_factor = min(1.0, organism.age / 100.0)  # Normalize age
    adaptation_factors.append(age_factor)
    
    # Size-based adaptation
    size_factor = min(1.0, len(organism.cells) / 10.0)  # Normalize size
    adaptation_factors.append(size_factor)
    
    # Generation-based adaptation
    generation_factor = min(1.0, organism.generation / 10.0)  # Normalize generation
    adaptation_factors.append(generation_factor)
    
    return float(np.mean(adaptation_factors) if adaptation_factors else 0.0)

def analyze_life(node_idx: int,
                neighborhood_data: NeighborhoodData,
                organisms: Dict[int, Organism],
                organism_assignments: Dict[int, int],
                energy_distribution: np.ndarray,
                environment_factors: Dict[str, float],
                base_metabolism: float,
                reproduction_threshold: float) -> LifeData:
    """
    Comprehensive artificial life analysis for a node
    
    Args:
        node_idx: Index of node to analyze
        neighborhood_data: Network neighborhood data
        organisms: Dictionary of living organisms
        organism_assignments: Dictionary of node organism assignments
        energy_distribution: Array of environmental energy values
        environment_factors: Dictionary of environmental factors
        base_metabolism: Base metabolic rate
        reproduction_threshold: Energy threshold for reproduction
    
    Returns:
        LifeData containing all life metrics
    """
    # Get current organism
    organism_id = organism_assignments.get(node_idx, -1)
    current_organism = organisms.get(organism_id, None)
    
    # Calculate metabolic metrics
    energy_balance, metabolic_rate = calculate_metabolic_metrics(
        node_idx, neighborhood_data, energy_distribution, base_metabolism
    )
    
    # Calculate reproduction metrics
    reproduction_potential, mutation_rate = calculate_reproduction_metrics(
        node_idx, neighborhood_data, organism_assignments,
        {org.organism_id: org.energy for org in organisms.values()},
        reproduction_threshold
    )
    
    # Calculate adaptation score
    adaptation_score = calculate_adaptation_score(
        node_idx, neighborhood_data, current_organism, environment_factors
    )
    
    # Determine cell role
    if current_organism:
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        foreign_neighbors = sum(1 for n in neighbors 
                              if organism_assignments.get(n, -1) != organism_id)
        if foreign_neighbors == 0:
            cell_role = 'core'
        elif foreign_neighbors > len(neighbors) / 2:
            cell_role = 'membrane'
        else:
            cell_role = 'specialized'
    else:
        cell_role = 'none'
    
    # Calculate survival score
    survival_factors = [
        max(0.0, energy_balance),
        1.0 - metabolic_rate / (base_metabolism * 2),
        adaptation_score
    ]
    survival_score = float(np.mean(survival_factors))
    
    # Identify interaction partners
    interaction_partners = set()
    if current_organism:
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        for n in neighbors:
            n_organism = organism_assignments.get(n, -1)
            if n_organism != -1 and n_organism != organism_id:
                interaction_partners.add(n_organism)
    
    return LifeData(
        organism=current_organism,
        cell_role=cell_role,
        energy_balance=energy_balance,
        metabolic_rate=metabolic_rate,
        reproduction_potential=reproduction_potential,
        mutation_rate=mutation_rate,
        adaptation_score=adaptation_score,
        survival_score=survival_score,
        interaction_partners=interaction_partners
    )

