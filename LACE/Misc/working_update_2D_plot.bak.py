    @timer_decorator                                                         
    def _update_2d_plot(self, new_grid=None):
        """Update the 2D plot by only drawing what has changed."""
        logger.debug("Updating 2D plot")
        if not self._initialization_complete:
            logger.debug("Skipping plot update during initialization")
            return

        # Add a recursion guard with a counter
        if not hasattr(self, '_update_2d_recursion_count'):
            self._update_2d_recursion_count = 0

        self._update_2d_recursion_count += 1
        if self._update_2d_recursion_count > 2:
            logger.warning(f"Excessive recursion in _update_2d_plot detected ({self._update_2d_recursion_count}), breaking recursion")
            self._update_2d_recursion_count = 0
            return

        try:
            # Use the passed grid parameter, which should be self.grid
            grid = new_grid if new_grid is not None else self.grid

            if grid is None:
                logger.error("Grid is None, cannot update plot")
                return

            # Get grid data
            grid_array = grid.grid_array

            # Identify active nodes (state > NODE_VISIBILITY_THRESHOLD)
            visible_mask = grid_array > GlobalSettings.Visualization.NODE_VISIBILITY_THRESHOLD

            # Collect active node positions and states
            x_coords, y_coords, states = [], [], []
            for i in range(grid.dimensions[0]):
                for j in range(grid.dimensions[1]):
                    if visible_mask[i, j]:
                        x_coords.append(j)  # Swap i and j for correct visualization
                        y_coords.append(i)
                        states.append(grid_array[i, j])

            # Get current edges
            current_edges = set()

            # Check if neighborhood_data is None and initialize it if needed
            if grid.neighborhood_data is None:
                logger.warning("neighborhood_data is None, initializing it")
                grid.neighborhood_data = NeighborhoodData(
                    grid.dimensions,
                    grid._calculate_max_neighbors(),
                    neighborhood_type=grid.neighborhood_type
                )
                grid.neighborhood_data.states = grid.grid_array.ravel()
                neighbor_indices = grid._initialize_neighbors()
                grid.neighborhood_data.neighbor_indices = neighbor_indices
                logger.debug("Initialized neighborhood_data and updated states")

            # Get edges from the grid
            for edge in grid.get_edges():
                # Convert edge to tuple of tuples for consistent comparison
                edge_tuple = (tuple(edge[0]), tuple(edge[1]))
                current_edges.add(edge_tuple)

            # Initialize last_updated_nodes and last_updated_edges if they don't exist
            if not hasattr(self, 'last_updated_nodes'):
                self.last_updated_nodes = set()
            if not hasattr(self, 'last_updated_edges'):
                self.last_updated_edges = set()
            if not hasattr(self, 'previous_active_nodes'):
                self.previous_active_nodes = set()
            if not hasattr(self, 'previous_edges'):
                self.previous_edges = set()

            # Track current active nodes
            current_active_nodes = set()
            
            # Determine what has changed
            new_nodes = current_active_nodes - self.previous_active_nodes
            removed_nodes = self.previous_active_nodes - current_active_nodes
            new_edges = current_edges - self.previous_edges
            removed_edges = self.previous_edges - current_edges

            # --- Edge Handling ---
            # Collect edge data for drawing
            edge_data = []
            for edge in current_edges:
                (i, j), (ni, nj) = edge

                # Determine edge color based on whether highlights are enabled
                if self.highlight_var.get():
                    # Check if this edge is in last_updated_edges
                    is_new_edge = edge in self.last_updated_edges
                    edge_color = GlobalSettings.Colors.NODE_EDGE_NEW if is_new_edge else GlobalSettings.Colors.NODE_EDGE_OLD
                else:
                    # With highlights off, all edges are blue
                    edge_color = GlobalSettings.Colors.NODE_EDGE_OLD

                edge_data.append((((j, i), (nj, ni)), edge_color))  # Swap coordinates

            # --- Node Handling ---
            # Determine node outline colors
            edge_colors = []
            for x, y in zip(x_coords, y_coords):
                if self.highlight_var.get():
                    # Check if this node is in last_updated_nodes
                    is_new_node = (y, x) in self.last_updated_nodes  # Note: we need to swap back to (i,j) format
                    edge_colors.append(GlobalSettings.Colors.NODE_EDGE_NEW if is_new_node else GlobalSettings.Colors.NODE_EDGE_OLD)
                else:
                    # With highlights off, all node outlines are blue
                    edge_colors.append(GlobalSettings.Colors.NODE_EDGE_OLD)

            # --- Drawing Operations ---
            # Clear the axes
            self.ax.clear()

            # Set axes properties
            self.ax.set_facecolor(GlobalSettings.Colors.BACKGROUND)
            self.ax.grid(False)
            self.ax.set_axisbelow(True)

            # Remove ticks and labels
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            # Remove spines
            for spine in self.ax.spines.values():
                spine.set_visible(False)

            # Update plot limits
            self.ax.set_xlim(-0.5, grid.dimensions[1] - 0.5)  # Swap dimensions
            self.ax.set_ylim(-0.5, grid.dimensions[0] - 0.5)

            # Draw edges using LineCollection
            if edge_data:
                segments = [data[0] for data in edge_data]
                colors = [data[1] for data in edge_data]

                line_collection = LineCollection(segments,
                                                colors=colors,
                                                alpha=GlobalSettings.Visualization.EDGE_OPACITY,
                                                linewidths=GlobalSettings.Visualization.EDGE_WIDTH,
                                                zorder=1)
                self.ax.add_collection(line_collection) # type: ignore
                self._edge_lines = [line_collection]  # Store the LineCollection

            # Draw nodes if there are any
            if x_coords:
                # Convert to numpy arrays
                x_coords = np.array(x_coords)
                y_coords = np.array(y_coords)
                states = np.array(states)

                # Use kwargs to avoid parameter name conflicts
                scatter_kwargs = {
                    'c': states,  # Use node states for coloring
                    'cmap': 'viridis',  # Use a colormap
                    'norm': Normalize(0, 1),  # Normalize values between 0 and 1
                    'alpha': GlobalSettings.Visualization.NODE_OPACITY,
                    'edgecolors': edge_colors,  # Use the determined edge colors
                    'linewidths': 1.0,  # Increased linewidth for visibility
                    'zorder': 2,
                    's': GlobalSettings.Visualization.NODE_SIZE * 100
                }

                # Create new scatter plot
                self._node_scatter = self.ax.scatter(x_coords, y_coords, **scatter_kwargs) # type: ignore

            # Store current state for next comparison
            self.previous_active_nodes = current_active_nodes
            self.previous_edges = current_edges

            # Store current counts for next comparison
            self._prev_node_count = len(current_active_nodes)
            self._prev_edge_count = len(current_edges)

            # Use regular drawing if blitting is disabled or background not saved
            self.canvas.draw_idle()

            # Update the GUI
            self.canvas.flush_events()

        except Exception as e:
            logger.error(f"Error updating 2D plot: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Always decrement the recursion counter
            self._update_2d_recursion_count -= 1
