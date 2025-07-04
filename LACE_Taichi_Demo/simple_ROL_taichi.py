# taichi_ca_demo.py
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import cv2  # Requires: pip install opencv-python

# --- Call ti.init FIRST ---
# Initialize Taichi, targeting GPU (Metal on Mac, CUDA/Vulkan elsewhere)
# default_fp=ti.f32 uses 32-bit floats for performance
# device_memory_fraction reserves GPU memory to avoid crashes on large grids
ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_fraction=0.9)

# --- Global Settings ---
GRID_SIZE = 1000      # Grid dimensions (GRID_SIZE x GRID_SIZE) - Try 2000, 4000 etc.
WINDOW_SIZE = 900     # Display window width and height in pixels
RULE = "realm_of_lace"  # Rule to run: "realm_of_lace" or "game_of_life"

# --- Shared Parameters ---
WRAP = True             # Use wrap-around boundary conditions?
INITIAL_DENSITY = 0.3   # Initial density of active cells for random start
NODE_COLORMAP = 'viridis' # Colormap for node degree visualization (RealmOfLace)
NODE_COLOR_NORM_VMIN = 0.0 # Minimum value for colormap normalization
NODE_COLOR_NORM_VMAX = 8.0 # Maximum value for colormap normalization (adjust based on typical degrees)

# --- RealmOfLace Specific Parameters ---
BIRTH_SUM_RANGES = [(5.0, 6.0), (8.0, 9.0), (15.0, 16.0)] # Neighbor degree sum ranges for birth
SURVIVAL_SUM_RANGES = [(3.0, 6.0), (8.0, 11.0), (15.0, 16.0)] # Neighbor degree sum ranges for survival
FINAL_DEATH_DEGREES = [0, 11, 12, 13, 14] # Node dies if its final degree is in this list

# --- Game of Life Specific Parameters ---
GOL_BIRTH = [3]         # Active neighbor counts for birth
GOL_SURVIVAL = [2, 3]   # Active neighbor counts for survival

# --- Taichi Fields ---
# These fields store the simulation state on the GPU/CPU memory managed by Taichi

# node_state: Stores the primary state (0 or 1) for Game of Life
node_state = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
# node_next_state: Temporary buffer for the next state in Game of Life
node_next_state = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))

# node_degree: Stores the node degree (connection count) for RealmOfLace state
node_degree = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
# node_eligible: Stores eligibility proxy (0 or 1) for RealmOfLace (Phase 1)
node_eligible = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
# node_next_eligible: Temporary buffer for next eligibility state (RealmOfLace)
node_next_eligible = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))
# node_next_degree: Temporary buffer for next degree state (RealmOfLace)
node_next_degree = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE))

# --- Neighborhood Offsets (Moore 2D) ---
# Precompute offsets for checking 8 neighbors
NUM_NEIGHBORS = 8
neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]

# --- Taichi Helper Functions (@ti.func runs inside kernels) ---

@ti.func
def wrap_idx(idx, N):
    """Applies boundary conditions (wrap or clamp) to an index."""
    result = 0
    if WRAP:
        # Modulo operator for wrap-around
        result = idx % N
    else:
        # Clamp index to grid boundaries [0, N-1]
        result = min(max(idx, 0), N - 1)
    return result

@ti.func
def in_range(val, ranges):
    """Checks if a value falls within any of the specified (min, max) ranges."""
    found = False
    # ti.static unrolls the loop at compile time for efficiency
    for i in ti.static(range(len(ranges))):
        if ranges[i][0] <= val <= ranges[i][1]:
            found = True
            # No early return allowed in Taichi loops
    return found

# --- Initialization Kernel ---
@ti.kernel
def initialize():
    """Initializes all Taichi fields to zero or based on random density."""
    for i, j in node_state: # Iterate over all grid cells
        # Initialize Game of Life state
        if ti.random() < INITIAL_DENSITY:
            node_state[i, j] = 1
        else:
            node_state[i, j] = 0
        node_next_state[i, j] = 0

        # Initialize RealmOfLace fields (start with 0 degree/eligibility)
        node_degree[i, j] = 0
        node_eligible[i, j] = 0
        node_next_eligible[i, j] = 0
        node_next_degree[i, j] = 0

# --- RealmOfLace Rule Kernels ---

@ti.kernel
def rol_compute_eligibility():
    """Phase 1: Compute node eligibility based on neighbor degree sum."""
    for i, j in node_degree: # Iterate using node_degree field
        sum_neighbor_degree = 0.0
        # Sum degrees of neighbors from the *previous* step
        for k in ti.static(range(NUM_NEIGHBORS)):
            di, dj = neighbor_offsets[k]
            ni = wrap_idx(i + di, GRID_SIZE)
            nj = wrap_idx(j + dj, GRID_SIZE)
            sum_neighbor_degree += node_degree[ni, nj] # Read previous degree

        eligible = 0
        if node_degree[i, j] <= 0: # Check birth condition if node was inactive (degree 0)
            if in_range(sum_neighbor_degree, ti.static(BIRTH_SUM_RANGES)):
                eligible = 1
        else: # Check survival condition if node was active (degree > 0)
            if in_range(sum_neighbor_degree, ti.static(SURVIVAL_SUM_RANGES)):
                eligible = 1
        node_next_eligible[i, j] = eligible # Store result in buffer

@ti.kernel
def rol_compute_edges_and_degree():
    """Phase 2: Compute next degree based on mutual eligibility."""
    for i, j in node_degree:
        degree = 0
        # Check eligibility of current node (calculated in Phase 1)
        if node_next_eligible[i, j] > 0:
            # Count eligible neighbors
            for k in ti.static(range(NUM_NEIGHBORS)):
                di, dj = neighbor_offsets[k]
                ni = wrap_idx(i + di, GRID_SIZE)
                nj = wrap_idx(j + dj, GRID_SIZE)
                # Check eligibility of neighbor (calculated in Phase 1)
                if node_next_eligible[ni, nj] > 0:
                    degree += 1 # Increment degree if neighbor is also eligible
        node_next_degree[i, j] = degree # Store result in buffer

@ti.kernel
def rol_finalize_state():
    """Phase 3: Apply death list and update final node degree state."""
    for i, j in node_degree:
        deg = node_next_degree[i, j] # Get degree calculated in Phase 2
        eligible = node_next_eligible[i, j] # Get eligibility from Phase 1

        # Check death list
        is_dead = False
        for d in ti.static(FINAL_DEATH_DEGREES):
            if deg == d:
                is_dead = True

        # Update final state: set degree if eligible and not dead, else set to 0
        if eligible > 0 and not is_dead:
            node_degree[i, j] = deg
        else:
            node_degree[i, j] = 0

def rol_step():
    """Perform one step of the RealmOfLace rule."""
    # On the very first frame, copy the initial binary state to degree field
    if frame == 0:
        node_degree.copy_from(node_state)
    rol_compute_eligibility()
    rol_compute_edges_and_degree()
    rol_finalize_state()

# --- Game of Life Rule Kernel ---
@ti.kernel
def gol_step_kernel():
    """Compute the next state for Game of Life."""
    for i, j in node_state: # Iterate using node_state field
        active_neighbors = 0
        # Count active neighbors from the *previous* step
        for k in ti.static(range(NUM_NEIGHBORS)):
            di, dj = neighbor_offsets[k]
            ni = wrap_idx(i + di, GRID_SIZE)
            nj = wrap_idx(j + dj, GRID_SIZE)
            if node_state[ni, nj] > 0: # Read previous state
                active_neighbors += 1

        # Apply GoL rules (B3/S23)
        if node_state[i, j] > 0: # If currently alive
            is_survival = False
            for s_count in ti.static(GOL_SURVIVAL):
                if active_neighbors == s_count:
                    is_survival = True
            if is_survival:
                node_next_state[i, j] = 1 # Survive
            else:
                node_next_state[i, j] = 0 # Die (under/overpopulation)
        else: # If currently dead
            is_birth = False
            for b_count in ti.static(GOL_BIRTH):
                if active_neighbors == b_count:
                    is_birth = True
            if is_birth:
                node_next_state[i, j] = 1 # Birth
            else:
                node_next_state[i, j] = 0 # Remain dead

def gol_step():
    """Perform one step of the Game of Life rule."""
    gol_step_kernel()
    # Copy the computed next state back to the main state field
    node_state.copy_from(node_next_state)

# --- Visualization: Node grid as image ---
def get_image_window(x0, x1, y0, y1):
    """Extracts the visible portion of the grid and applies colormap."""
    if RULE == "realm_of_lace":
        # Use node_degree field for RealmOfLace
        state_np = node_degree.to_numpy()
        # Normalize degree values using specified min/max
        normed = np.clip((state_np - NODE_COLOR_NORM_VMIN) / (NODE_COLOR_NORM_VMAX - NODE_COLOR_NORM_VMIN), 0, 1)
        cmap = cm.get_cmap(NODE_COLORMAP)
        # Apply colormap and convert to uint8 RGB
        rgb = (cmap(normed)[..., :3] * 255).astype(np.uint8)
        # Extract the visible window
        img = rgb[y0:y1, x0:x1]
        return img
    elif RULE == "game_of_life":
        # Use node_state field for Game of Life
        state_np = node_state.to_numpy()
        # Create a black image and set active cells to white
        img = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        img[state_np[y0:y1, x0:x1] > 0] = [255, 255, 255] # White for active
        return img
    else:
        # Fallback for unknown rule: return black image
        return np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)

# --- GGUI Setup ---
window = ti.ui.Window("Taichi CA Demo", res=(WINDOW_SIZE, WINDOW_SIZE), vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()

# GGUI state variables
zoom = 1.0
pan_x = 0.0
pan_y = 0.0
zoom_slider = 1.0
pan_x_slider = 0.0
pan_y_slider = 0.0

def clamp_pan(pan, zoom):
    """Helper to clamp pan values based on zoom to prevent panning off-grid."""
    max_pan = 1.0 - 1.0 / max(zoom, 1.0) # Ensure zoom is at least 1 for calculation
    return np.clip(pan, -max_pan, max_pan)

def handle_gui():
    """Update GGUI widgets and simulation parameters."""
    global zoom_slider, pan_x_slider, pan_y_slider, zoom, pan_x, pan_y
    gui.begin("Controls", 0.01, 0.01, 0.25, 0.18)
    gui.text("Zoom and Pan Controls")
    gui.text(f"Rule: {RULE}")
    zoom_slider = gui.slider_float("Zoom", zoom_slider, 1.0, 10.0)
    pan_x_slider = gui.slider_float("Pan X", pan_x_slider, -1.0, 1.0)
    pan_y_slider = gui.slider_float("Pan Y", pan_y_slider, -1.0, 1.0)
    gui.end()
    # Update simulation zoom/pan based on sliders
    zoom = zoom_slider
    pan_x = clamp_pan(pan_x_slider, zoom)
    pan_y = clamp_pan(pan_y_slider, zoom)

# --- Main Loop ---
initialize()
frame = 0
while window.running:
    handle_gui() # Update GUI state first

    # Execute the selected rule's step function
    if RULE == "realm_of_lace":
        rol_step()
    elif RULE == "game_of_life":
        gol_step()
    frame += 1

    # Set black background every frame BEFORE drawing
    canvas.set_background_color((0, 0, 0))

    # Compute view window based on zoom and pan
    cx = GRID_SIZE // 2 + int(pan_x * GRID_SIZE // 2)
    cy = GRID_SIZE // 2 + int(pan_y * GRID_SIZE // 2)
    view_size = int(GRID_SIZE / zoom)
    x0 = np.clip(cx - view_size // 2, 0, GRID_SIZE - view_size)
    y0 = np.clip(cy - view_size // 2, 0, GRID_SIZE - view_size)
    x1 = x0 + view_size
    y1 = y0 + view_size

    # Get the current window size for resizing
    w, h = window.get_window_shape()

    # Get the colored image for the current view
    img = get_image_window(x0, x1, y0, y1)

    # Resize the image to fill the window exactly
    if img is not None and img.size > 0:
        img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        canvas.set_image(img_resized)
    else:
        # Fill with black if img is invalid (shouldn't happen with current logic)
        canvas.set_image(np.zeros((h, w, 3), dtype=np.uint8))

    # --- Edge drawing is DISABLED ---

    # Show frame/zoom info in GUI overlay
    gui.begin("Info", 0.01, 0.20, 0.25, 0.10)
    gui.text(f"Frame: {frame}")
    gui.text(f"Zoom: {zoom:.2f}")
    gui.text(f"Pan: ({pan_x:.2f}, {pan_y:.2f})")
    gui.text(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    gui.text(f"Rule: {RULE}")
    gui.end()

    # Display the frame
    window.show()
    ti.sync() # Ensure GPU commands are flushed
    time.sleep(1/60) # Throttle frame rate to ~60 FPS