# LACE - Link Automata Computation Engine

A sophisticated cellular automata framework for exploring complex emergent behaviors through network-based rules.

## Overview

LACE extends traditional cellular automata (like Conway's Game of Life) by treating cells as **nodes in a dynamic network** with explicit **edges** between neighbors. This allows for rules based on:

- **Node degrees** (number of connections)
- **Neighbor degree sums** (connectivity patterns)
- **Edge dynamics** (connections can form and break)
- **Network topology metrics** (betweenness, clustering, etc.)

## Features

- 🎨 **182 Built-in Rules** including classic Game of Life and novel "Realm of Lace" variants
- 🎯 **Multiple Rule Types**: Standard Life-like, Network CA, Metric-based, and more
- 🌐 **2D Support** with multiple coordinate systems (3D support partially implemented; needs further work before available)
- 📊 **Real-time Analytics** with customizable metrics
- 🎨 **Flexible Visualization** with 21 color schemes
- 💾 **4,795 Shape Library** (via Git LFS)
- ⚡ **High Performance** with multiprocessing and Numba JIT compilation
- 🚀 **Taichi GPU Demos** for ultra-fast large-scale simulations

## Installation

### Prerequisites

- Python 3.10 or higher
- Git with Git LFS installed (`brew install git-lfs` on macOS)

### Setup

```bash
# Clone the repository
git clone https://github.com/novaspivack/lace.git
cd lace

# Install Git LFS (if not already installed)
git lfs install

# Install dependencies
pip install -r requirements.txt

# Optional: For Taichi GPU demos
pip install taichi opencv-python
```

## Usage

### Running the Main Application

```bash
# From the project root
python LACE/lace_app.py
```

Or using module syntax:
```bash
python -m LACE.lace_app
```

### Running Taichi Demos

```bash
# High-performance GPU rendering demo
python -m LACE_Taichi_Demo.taichi_advanced_render

# Or directly
cd LACE_Taichi_Demo
python taichi_advanced_render.py
```

## Project Structure

```
lace/
├── LACE/                          # Main application package
│   ├── lace_app.py               # Main GUI application
│   ├── rules.py                  # Rule definitions and logic
│   ├── shapes.py                 # Shape library management
│   ├── analytics.py              # Metrics and analysis
│   ├── presets.py                # Grid presets
│   ├── colors.py                 # Color scheme management
│   └── Resources/                # Runtime resources
│       ├── config/               # Configuration files
│       │   ├── rules/           # Rule definitions
│       │   ├── presets/         # Grid presets
│       │   └── colors/          # Color schemes
│       ├── data/                # Shape library (206MB via LFS)
│       ├── saves/               # Saved simulations
│       └── logs/                # Application logs
│
├── LACE_Taichi_Demo/            # GPU-accelerated demos
│   ├── taichi_advanced_render.py
│   └── simple_ROL_taichi.py
│
├── Tests/                        # Test scripts
├── LICENSE                       # CC BY-NC 4.0
└── README.md                     # This file
```

## Key Concepts

### Network Cellular Automata

Unlike traditional CA where cells are simply "alive" or "dead", LACE cells:
- Have **degrees** (number of connections to neighbors)
- Form **edges** with eligible neighbors
- Follow rules based on **network topology**
- Can use **graph metrics** for state transitions

### Realm of Lace Rules

The signature "Realm of Lace" family of rules uses:
- **Birth conditions**: Based on sum of neighbor degrees
- **Survival conditions**: Based on sum of neighbor degrees  
- **Death conditions**: Specific degree counts that cause node death
- **Eligibility**: Nodes must meet conditions to form connections


## Grid Editing

You can edit the grid - it's a little buggy and still needs some work but here are the basic controls...

To activate edit mode right click on the canvas.

Mouse & Keyboard Controls:

        --- Canvas Interaction ---

        Left Click: Toggle node On/Off (or activate/deactivate)

        Left Drag: Scribble Draw (activate nodes, connect new binary edges if rule supports)

        Shift + Left Drag: Scribble Erase (deactivate nodes, remove connected edges)

        Ctrl/Cmd + Left Drag: Scribble Delete Edges (remove edges along path)

        Alt/Opt + Left Click: Increment Node State (+10% Real) [Only for REAL state rules]

        Alt/Opt + Left Drag: Scribble Increment Node/Edge State (+10% Real) [Only for REAL state rules]

        Alt/Opt+Shift + L Click: Decrement Node State (-10% Real) [Only for REAL state rules]

        Alt/Opt+Shift + L Drag: Scribble Decrement Node/Edge State (-10% Real, removes edge if state reaches minimum) [Only for REAL state rules]

        Right Click: Open Context Menu

        Mouse Wheel: Pan View (Vertical)

        Shift + Mouse Wheel: Pan View (Horizontal - OS/driver dependent)

        Ctrl/Cmd + Mouse Wheel: Zoom View (not implemented yet)

        --- Keyboard Shortcuts ---

        Spacebar: Start/Pause Simulation (not implemented yet)

        S: Step Simulation Once (not implemented yet)

        R: Reset Simulation (not implemented yet)

        C: Toggle Control Panel Visibility (not implemented yet)

        L: Open Shape Library & Editor (not implemented yet)

        Ctrl/Cmd + Z: Undo Grid Action (not implemented yet)

        Ctrl/Cmd + Y / Ctrl+Shift+Z: Redo Grid Action (not implemented yet)

        Ctrl/Cmd + S: Save Simulation State (not implemented yet)

        Ctrl/Cmd + O: Load Simulation State (not implemented yet)

        Ctrl/Cmd + N: Create New Preset (not implemented yet)

        Ctrl/Cmd + M: Manage Presets (not implemented yet)

        Ctrl/Cmd + E: Edit Current Rule (not implemented yet)

        Ctrl/Cmd + L: Open Shape Library & Editor (Duplicate - not implemented yet)

        Ctrl/Cmd + C: Copy Selection (not implemented yet)

        Ctrl/Cmd + X: Cut Selection (not implemented yet)

        Ctrl/Cmd + V: Paste Selection Here (not implemented yet)

        Delete/Backspace: Erase Selected Nodes (not implemented yet)

        Ctrl/Cmd + Delete/Backspace: Delete Edges Within Selection (not implemented yet)

        Ctrl/Cmd + A: Select All Active Nodes (not implemented yet)

        Escape: Deselect All / Cancel Tool Action (not implemented yet)

        (Note: Alt = Alt on Win/Lin, Option on macOS)


## Gallery

**🎬 [View Full Video Gallery →](https://www.novaspivack.com/science/introducing-lace-a-new-kind-of-cellular-automata)**

Watch 70+ simulation captures with inline playback on the dedicated gallery page.

### Featured Samples

A few highlights from the collection (download links):

<table>
<tr>
<td width="33%" align="center">
<b>Realm of Lace Unified</b><br>
<a href="https://github.com/novaspivack/lace/raw/master/LACE/Captures/Realm%20of%20Lace%20Unified_20251015_155821.mp4">
<img src="https://img.shields.io/badge/▶-Download-blue?style=for-the-badge" alt="Download"/>
</a>
</td>
<td width="33%" align="center">
<b>Lace Life with Edges</b><br>
<a href="https://github.com/novaspivack/lace/raw/master/LACE/Captures/Lace%20Life%20with%20Edges_20251015_160125.mp4">
<img src="https://img.shields.io/badge/▶-Download-blue?style=for-the-badge" alt="Download"/>
</a>
</td>
<td width="33%" align="center">
<b>Realm of Lace</b><br>
<a href="https://github.com/novaspivack/lace/raw/master/LACE/Captures/Realm%20of%20Lace_20251015_163431.mp4">
<img src="https://img.shields.io/badge/▶-Download-blue?style=for-the-badge" alt="Download"/>
</a>
</td>
</tr>
<tr>
<td width="33%" align="center">
<b>Edge Feedback Life</b><br>
<a href="https://github.com/novaspivack/lace/raw/master/LACE/Captures/Edge%20Feedback%20Life_20251015_162645.mp4">
<img src="https://img.shields.io/badge/▶-Download-blue?style=for-the-badge" alt="Download"/>
</a>
</td>
<td width="33%" align="center">
<b>Multi-State Life</b><br>
<a href="https://github.com/novaspivack/lace/raw/master/LACE/Captures/Multi-State%20Life%20with%20Edges_20251015_165135.mp4">
<img src="https://img.shields.io/badge/▶-Download-blue?style=for-the-badge" alt="Download"/>
</a>
</td>
<td width="33%" align="center">
<b>Configurable Continuous</b><br>
<a href="https://github.com/novaspivack/lace/raw/master/LACE/Captures/Configurable%20Continuous%20Life_20251015_164910.mp4">
<img src="https://img.shields.io/badge/▶-Download-blue?style=for-the-badge" alt="Download"/>
</a>
</td>
</tr>
</table>

**Note:** All 70+ videos are available in the [`LACE/Captures`](LACE/Captures) folder, each with metadata JSON files documenting capture settings.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License** (CC BY-NC 4.0).

**You are free to:**
- Share and adapt the code for non-commercial purposes
- Create derivative works

**Under these terms:**
- **Attribution required** - Credit the original author
- **Non-commercial only** - Commercial use requires permission
- **No forced contribution** - Derivatives don't need to be merged back

See [LICENSE](LICENSE) for full details.

## Contact

For commercial licensing inquiries or questions, please contact Nova Spivack.

## Acknowledgments

Built with Python, NumPy, Matplotlib, NetworkX, Numba, and Taichi.

