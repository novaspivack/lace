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
- 🌐 **2D and 3D** support with multiple coordinate systems
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

