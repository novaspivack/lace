class TestRule(Rule):
    """
    Test Rule: Creates balanced patterns through state transitions and edge formation.
    Demonstrates core cellular automata principles with configurable parameters.
    """

    # Parameter metadata and validation
    PARAMETER_METADATA = {
        "min_active_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required for survival",
            "min": 0,
            "max": 8
        },
        "max_active_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed for survival",
            "min": 0,
            "max": 8
        },
        "min_connected_active": {
            "type": int,
            "description": "Minimum number of connected active neighbors required for survival",
            "min": 0,
            "max": 8
        },
        "max_connected_active": {
            "type": int,
            "description": "Maximum number of connected active neighbors allowed for survival",
            "min": 0,
            "max": 8
        },
        "birth_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required for birth",
            "min": 0,
            "max": 8
        },
        "birth_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed for birth",
            "min": 0,
            "max": 8
        },
        "birth_min_connected": {
            "type": int,
            "description": "Minimum number of connected neighbors required for birth",
            "min": 0,
            "max": 8
        },
        "birth_max_connected": {
            "type": int,
            "description": "Maximum number of connected neighbors allowed for birth",
            "min": 0,
            "max": 8
        },
        "use_edge_density": {
            "type": bool,
            "description": "Enable edge density-based state changes"
        },
        "death_edge_density": {
            "type": float,
            "description": "Edge density threshold that triggers death",
            "min": 0.0,
            "max": 1.0
        },
        "birth_edge_density": {
            "type": float,
            "description": "Maximum edge density allowed for birth",
            "min": 0.0,
            "max": 1.0
        },
        "use_entropy": {
            "type": bool,
            "description": "Enable entropy-based state changes"
        },
        "death_entropy_threshold": {
            "type": float,
            "description": "Minimum entropy required to prevent death",
            "min": 0.0,
            "max": 1.0
        },
        "birth_entropy_threshold": {
            "type": float,
            "description": "Minimum entropy required for birth",
            "min": 0.0,
            "max": 1.0
        },
        "use_random_death": {
            "type": bool,
            "description": "Enable random death chance"
        },
        "death_probability": {
            "type": float,
            "description": "Probability of random death when enabled",
            "min": 0.0,
            "max": 1.0
        },
        "edge_formation_rate": {
            "type": float,
            "description": "Probability of forming a new edge",
            "min": 0.0,
            "max": 1.0
        },
        "edge_removal_rate": {
            "type": float,
            "description": "Probability of removing an existing edge",
            "min": 0.0,
            "max": 1.0
        },
        "min_shared_neighbors": {
            "type": int,
            "description": "Minimum shared neighbors required to maintain an edge",
            "min": 0,
            "max": 8
        },
        "max_edges_per_node": {
            "type": int,
            "description": "Maximum number of edges a node can have",
            "min": 0,
            "max": 8
        },
        "min_edges_per_node": {
            "type": int,
            "description": "Minimum number of edges a node must maintain",
            "min": 0,
            "max": 8
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": "RANDOM",
        "allowed_values": [
            "HIGHER_STATE",
            "LOWER_STATE",
            "MORE_CONNECTIONS",
            "FEWER_CONNECTIONS",
            "HIGHER_STATE_MORE_NEIGHBORS",
            "LOWER_STATE_FEWER_NEIGHBORS",
            "HIGHER_STATE_FEWER_NEIGHBORS",
            "LOWER_STATE_MORE_NEIGHBORS",
            "RANDOM",
            "AGREEMENT"
        ]
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData, 
                        node_positions: Optional[np.ndarray] = None, 
                        dimension_type: Optional[Dimension] = None) -> float:
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0:
            return -1.0

        active_neighbors = np.sum(neighborhood_data.states[neighbors] > 0)
        connected_active = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors] & 
                                (neighborhood_data.states[neighbors] > 0))

        if self.get_param('use_edge_density', False):
            edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)
        else:
            edge_density = 0.0

        if self.get_param('use_entropy', False):
            entropy = self.get_metric(StateEntropy, node_idx, neighborhood_data)
        else:
            entropy = 1.0

        if current_state > 0:
            min_active = self.get_param('min_active_neighbors')
            max_active = self.get_param('max_active_neighbors')
            min_connected = self.get_param('min_connected_active')
            max_connected = self.get_param('max_connected_active')
            
            if None in (min_active, max_active, min_connected, max_connected):
                return 0.0

            if edge_density > self.get_param('death_edge_density', 1.0):
                return 0.0

            if entropy < self.get_param('death_entropy_threshold', 0.0):
                return 0.0

            survives = (min_active <= active_neighbors <= max_active and
                    min_connected <= connected_active <= max_connected)

            if active_neighbors == min_active or active_neighbors == max_active:
                survives = TieBreaker.resolve(
                    current_state,
                    neighborhood_data.states[neighbors[0]],
                    connected_active,
                    np.sum(neighborhood_data.edge_matrix[neighbors[0]]),
                    TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
                )

            if (survives and 
                self.get_param('use_random_death', False) and
                random.random() < self.get_param('death_probability', 0.0)):
                return 0.0

            return 1.0 if survives else 0.0

        else:
            birth_min = self.get_param('birth_min_neighbors')
            birth_max = self.get_param('birth_max_neighbors')
            birth_min_connected = self.get_param('birth_min_connected')
            birth_max_connected = self.get_param('birth_max_connected')

            if None in (birth_min, birth_max, birth_min_connected, birth_max_connected):
                return 0.0

            if edge_density > self.get_param('birth_edge_density', 1.0):
                return 0.0

            if entropy < self.get_param('birth_entropy_threshold', 0.0):
                return 0.0

            born = (birth_min <= active_neighbors <= birth_max and
                birth_min_connected <= connected_active <= birth_max_connected)

            if active_neighbors == birth_min or active_neighbors == birth_max:
                born = TieBreaker.resolve(
                    current_state,
                    neighborhood_data.states[neighbors[0]],
                    connected_active,
                    np.sum(neighborhood_data.edge_matrix[neighbors[0]]),
                    TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
                )

            return 1.0 if born else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                        node_positions: Optional[np.ndarray] = None,
                        dimension_type: Optional[Dimension] = None) -> Set[int]:
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        edge_formation_rate = self.get_param('edge_formation_rate')
        edge_removal_rate = self.get_param('edge_removal_rate')
        min_shared = self.get_param('min_shared_neighbors')
        max_edges = self.get_param('max_edges_per_node')
        min_edges = self.get_param('min_edges_per_node')

        if None in (edge_formation_rate, edge_removal_rate, min_shared, max_edges, min_edges):
            return new_edges

        current_edges = {n for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}

        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            shared = np.sum(neighborhood_data.edge_matrix[node_idx] & 
                        neighborhood_data.edge_matrix[n])

            if n in current_edges:
                if shared >= min_shared:
                    new_edges.add(n)
                elif random.random() >= edge_removal_rate:
                    new_edges.add(n)
            else:
                if (len(new_edges) < max_edges and
                    shared >= min_shared and
                    random.random() < edge_formation_rate):
                    new_edges.add(n)

        if len(new_edges) < min_edges:
            available = [n for n in neighbors 
                        if neighborhood_data.states[n] > 0 and n not in new_edges]
            while len(new_edges) < min_edges and available:
                n = available.pop(random.randrange(len(available)))
                new_edges.add(n)

        return new_edges

class PureRuleTableRule(Rule):
    """
    PureRuleTable: A rule driven entirely by comprehensive rule tables for both state
    transitions and edge updates. Uses only local neighborhood state patterns and
    connection patterns without any additional metrics. Includes tiebreaker logic
    for equivalent patterns.
    """

    PARAMETER_METADATA = {
        "state_rule_table": {
            "type": dict,
            "description": "Rule table for state transitions. Keys are (current_state, neighbor_pattern, connection_pattern)",
            "default": {}
        },
        "edge_rule_table": {
            "type": dict,
            "description": "Rule table for edge updates. Keys are (self_state, neighbor_state, connection_pattern)",
            "default": {}
        },
        "state_memory": {
            "type": int,
            "description": "Number of steps to remember previous states",
            "min": 0,
            "max": 10
        },
        "edge_memory": {
            "type": int,
            "description": "Number of steps to remember previous edges",
            "min": 0,
            "max": 10
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": "RANDOM",
        "allowed_values": [
            "HIGHER_STATE",
            "LOWER_STATE",
            "MORE_CONNECTIONS",
            "FEWER_CONNECTIONS",
            "HIGHER_STATE_MORE_NEIGHBORS",
            "LOWER_STATE_FEWER_NEIGHBORS",
            "HIGHER_STATE_FEWER_NEIGHBORS",
            "LOWER_STATE_MORE_NEIGHBORS",
            "RANDOM",
            "AGREEMENT"
        ]
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        self.state_history: List[Dict[int, float]] = []
        self.edge_history: List[Set[Tuple[int, int]]] = []

    def _validate_parameter(self, name: str, value: Any) -> bool:
        """Override parameter validation to handle rule tables"""
        if name == 'state_rule_table':
            return self._validate_state_rule_table(value)
        elif name == 'edge_rule_table':
            return self._validate_edge_rule_table(value)
        
        return super()._validate_parameter(name, value)

    def _validate_state_rule_table(self, table: Dict[str, int]) -> bool:
        """Validate state rule table format and values"""
        try:
            if "default" not in table:
                return False
                
            for key, value in table.items():
                if key == "default":
                    if value not in [-1, 0, 1]:
                        return False
                    continue
                    
                try:
                    # Parse key format: (current_state, neighbor_pattern, connection_pattern)
                    parts = key.strip("()").split(",")
                    if len(parts) != 3:
                        return False
                        
                    current_state = int(parts[0])
                    neighbor_pattern = parts[1].strip()
                    connection_pattern = parts[2].strip()
                    
                    # Validate current state
                    if current_state not in [-1, 0, 1]:
                        return False
                        
                    # Validate neighbor pattern (exactly 8 bits)
                    if len(neighbor_pattern) != 8 or not all(c in '01' for c in neighbor_pattern):
                        return False
                        
                    # Validate connection pattern (exactly 8 bits)
                    if len(connection_pattern) != 8 or not all(c in '01' for c in connection_pattern):
                        return False
                        
                    # Validate new state
                    if value not in [-1, 0, 1]:
                        return False
                        
                except ValueError:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"State rule table validation error: {e}")
            return False

    def _validate_edge_rule_table(self, table: Dict[str, str]) -> bool:
        """Validate edge rule table format and values"""
        try:
            if "default" not in table:
                return False
                
            valid_actions = {"add", "remove", "maintain"}
            
            for key, value in table.items():
                if key == "default":
                    if value not in valid_actions:
                        return False
                    continue
                    
                try:
                    # Parse key format: (self_state, neighbor_state, connection_pattern)
                    parts = key.strip("()").split(",")
                    if len(parts) != 3:
                        return False
                        
                    self_state = int(parts[0])
                    neighbor_state = int(parts[1])
                    connection_pattern = parts[2].strip()
                    
                    # Validate states
                    if self_state not in [0, 1] or neighbor_state not in [0, 1]:
                        return False
                        
                    # Validate connection pattern (exactly 8 bits)
                    if len(connection_pattern) != 8 or not all(c in '01' for c in connection_pattern):
                        return False
                        
                    # Validate action
                    if value not in valid_actions:
                        return False
                        
                except ValueError:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Edge rule table validation error: {e}")
            return False
        
    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based purely on rule table patterns"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0:
            return -1.0

        # Get rule table
        state_rule_table = self.get_param('state_rule_table')
        if state_rule_table is None:
            return 0.0

        # Create neighbor pattern string
        neighbor_pattern = ''.join(['1' if neighborhood_data.states[n] > 0 else '0' 
                                  for n in neighbors])
        
        # Create connection pattern string
        connection_pattern = ''.join(['1' if neighborhood_data.edge_matrix[node_idx, n] else '0'
                                    for n in neighbors])

        # Create lookup key
        key = f"({int(current_state)}, {neighbor_pattern}, {connection_pattern})"
        
        # Get new state from rule table
        new_state = state_rule_table.get(key, state_rule_table.get('default', 0))

        # Handle ties using tiebreaker
        if new_state == current_state:
            new_state = int(TieBreaker.resolve(
                current_state,
                neighborhood_data.states[neighbors[0]],
                np.sum(neighborhood_data.edge_matrix[node_idx]),
                np.sum(neighborhood_data.edge_matrix[neighbors[0]]),
                TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
            ))

        # Update state history
        state_memory = self.get_param('state_memory', 0)
        if state_memory > 0:
            self.state_history.append({node_idx: new_state})
            if len(self.state_history) > state_memory:
                self.state_history.pop(0)

        return float(new_state)

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based purely on rule table patterns"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get rule table
        edge_rule_table = self.get_param('edge_rule_table')
        if edge_rule_table is None:
            return new_edges

        # Get current connection pattern
        connection_pattern = ''.join(['1' if neighborhood_data.edge_matrix[node_idx, n] else '0'
                                    for n in neighbors])

        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            # Create lookup key
            key = f"({int(neighborhood_data.states[node_idx] > 0)}, {int(neighborhood_data.states[n] > 0)}, {connection_pattern})"
            
            # Get action from rule table
            action = edge_rule_table.get(key, edge_rule_table.get('default', 'maintain'))

            # Apply action
            if action == 'add':
                new_edges.add(n)
            elif action == 'maintain' and neighborhood_data.edge_matrix[node_idx, n]:
                new_edges.add(n)

        # Update edge history
        edge_memory = self.get_param('edge_memory', 0)
        if edge_memory > 0:
            self.edge_history.append(new_edges.copy())
            if len(self.edge_history) > edge_memory:
                self.edge_history.pop(0)

        return new_edges
    
class MajorityRule(Rule):
    """
    Highly Parameterized Majority Rule with extensive control over all aspects of the simulation.
    """

    PARAMETER_METADATA = {
        "activation_threshold": {
            "type": float,
            "description": "Ratio of active neighbors required for activation",
            "min": 0.0,
            "max": 1.0
        },
        "deactivation_threshold": {
            "type": float,
            "description": "Ratio of active neighbors below which a node deactivates",
            "min": 0.0,
            "max": 1.0
        },
        "rebellion_chance": {
            "type": float,
            "description": "Base probability of a node rebelling against the majority",
            "min": 0.0,
            "max": 1.0
        },
        "rebellion_activation_bonus": {
            "type": float,
            "description": "Bonus to activation probability when rebelling",
            "min": 0.0,
            "max": 1.0
        },
        "rebellion_deactivation_penalty": {
            "type": float,
            "description": "Penalty to deactivation probability when rebelling",
            "min": 0.0,
            "max": 1.0
        },
        "random_birth_rate": {
            "type": float,
            "description": "Probability of a new node being born randomly",
            "min": 0.0,
            "max": 1.0
        },
        "random_death_rate": {
            "type": float,
            "description": "Probability of a node dying randomly",
            "min": 0.0,
            "max": 1.0
        },
        "edge_formation_rate": {
            "type": float,
            "description": "Probability of forming new edges",
            "min": 0.0,
            "max": 1.0
        },
        "edge_removal_rate": {
            "type": float,
            "description": "Probability of removing existing edges",
            "min": 0.0,
            "max": 1.0
        },
        "min_shared_neighbors": {
            "type": int,
            "description": "Minimum shared neighbors required to maintain an edge",
            "min": 0,
            "max": 8
        },
        "max_connections": {
            "type": int,
            "description": "Maximum number of connections a node can have",
            "min": 0,
            "max": 8
        },
        "min_connections": {
            "type": int,
            "description": "Minimum number of connections a node must have",
            "min": 0,
            "max": 8
        },
        "birth_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required for birth",
            "min": 0,
            "max": 8
        },
        "birth_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed for birth",
            "min": 0,
            "max": 8
        },
        "death_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required for survival",
            "min": 0,
            "max": 8
        },
        "death_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed for survival",
            "min": 0,
            "max": 8
        },
        "connect_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required to form a connection",
            "min": 0,
            "max": 8
        },
        "connect_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed to form a connection",
            "min": 0,
            "max": 8
        },
        "disconnect_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required to maintain a connection",
            "min": 0,
            "max": 8
        },
        "disconnect_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed to maintain a connection",
            "min": 0,
            "max": 8
        },
        "rebellion_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required to trigger a rebellion",
            "min": 0,
            "max": 8
        },
        "rebellion_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed to trigger a rebellion",
            "min": 0,
            "max": 8
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT"
            ]
        },
        "edge_formation_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required to form an edge",
            "min": 0,
            "max": 8
        },
        "edge_formation_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed to form an edge",
            "min": 0,
            "max": 8
        },
        "edge_removal_min_neighbors": {
            "type": int,
            "description": "Minimum number of active neighbors required to remove an edge",
            "min": 0,
            "max": 8
        },
        "edge_removal_max_neighbors": {
            "type": int,
            "description": "Maximum number of active neighbors allowed to remove an edge",
            "min": 0,
            "max": 8
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}  # Initialize the params dictionary

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on majority/minority relationships"""
        try:
            # Get current state
            current_state = neighborhood_data.states[node_idx]
            neighbors = neighborhood_data.get_neighbor_indices(node_idx)
            
            # Handle empty cell case
            if current_state == -1.0:
                if random.random() < self.get_param('random_birth_rate', 0.01):
                    return 1.0
                else:
                    return -1.0

            # Calculate metrics
            if len(neighbors) == 0:
                return 0.0
                
            active_neighbors = np.sum(neighborhood_data.states[neighbors] > 0)
            active_ratio = active_neighbors / len(neighbors) if len(neighbors) > 0 else 0.0

            # Get parameters with defaults
            activation_threshold = self.get_param('activation_threshold', 0.6)
            deactivation_threshold = self.get_param('deactivation_threshold', 0.4)
            rebellion_chance = self.get_param('rebellion_chance', 0.0)
            rebellion_activation_bonus = self.get_param('rebellion_activation_bonus', 0.1)
            rebellion_deactivation_penalty = self.get_param('rebellion_deactivation_penalty', 0.1)
            random_death_rate = self.get_param('random_death_rate', 0.01)
            birth_min_neighbors = self.get_param('birth_min_neighbors', 0)
            birth_max_neighbors = self.get_param('birth_max_neighbors', 8)
            death_min_neighbors = self.get_param('death_min_neighbors', 0)
            death_max_neighbors = self.get_param('death_max_neighbors', 8)
            rebellion_min_neighbors = self.get_param('rebellion_min_neighbors', 0)
            rebellion_max_neighbors = self.get_param('rebellion_max_neighbors', 8)

            # State transition logic
            if current_state > 0:  # Active cell
                # Simplified survival check
                survives = active_ratio >= deactivation_threshold and death_min_neighbors <= active_neighbors <= death_max_neighbors
                
                # Apply rebellion chance
                if rebellion_min_neighbors <= active_neighbors <= rebellion_max_neighbors and random.random() < rebellion_chance:
                    survives = not survives
                    if survives:
                        survives = survives and random.random() > rebellion_deactivation_penalty
                    else:
                        survives = survives or random.random() < rebellion_activation_bonus
                
                if random.random() < random_death_rate:
                    survives = False
                
                return 1.0 if survives else 0.0
            else:  # Inactive cell
                # Simplified birth check
                born = active_ratio >= activation_threshold and birth_min_neighbors <= active_neighbors <= birth_max_neighbors
                
                # Apply rebellion chance
                if rebellion_min_neighbors <= active_neighbors <= rebellion_max_neighbors and random.random() < rebellion_chance:
                    born = not born
                    if born:
                        born = born and random.random() > rebellion_deactivation_penalty
                    else:
                        born = born or random.random() < rebellion_activation_bonus
                    
                return 1.0 if born else 0.0

        except Exception as e:
            logger.error(f"Error in MajorityRule compute_state_update for node {node_idx}: {e}")
            logger.error(traceback.format_exc())
            return 0.0  # Default to inactive on error
                
    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on majority relationships and state matching"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get edge parameters
        edge_formation_rate = self.get_param('edge_formation_rate', 0.5)
        edge_removal_rate = self.get_param('edge_removal_rate', 0.1)
        min_shared_neighbors = self.get_param('min_shared_neighbors', 1)
        max_connections = self.get_param('max_connections', 8)
        min_connections = self.get_param('min_connections', 2)
        connect_min_neighbors = self.get_param('connect_min_neighbors', 0)
        connect_max_neighbors = self.get_param('connect_max_neighbors', 8)
        disconnect_min_neighbors = self.get_param('disconnect_min_neighbors', 0)
        disconnect_max_neighbors = self.get_param('disconnect_max_neighbors', 8)
        edge_formation_min_neighbors = self.get_param('edge_formation_min_neighbors', 0)
        edge_formation_max_neighbors = self.get_param('edge_formation_max_neighbors', 8)
        edge_removal_min_neighbors = self.get_param('edge_removal_min_neighbors', 0)
        edge_removal_max_neighbors = self.get_param('edge_removal_max_neighbors', 8)

        current_edges = {n for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}

        for n in neighbors:
            if neighborhood_data.states[n] > 0:
                # Check connection conditions
                if connect_min_neighbors <= len(neighbors) <= connect_max_neighbors:
                    if len(new_edges) < max_connections:
                        if random.random() < edge_formation_rate:
                            new_edges.add(n)
                # Check disconnection conditions
                if n in current_edges and disconnect_min_neighbors <= len(neighbors) <= disconnect_max_neighbors:
                    if random.random() < edge_removal_rate:
                        new_edges.discard(n)

        # Enforce maximum connections
        if len(new_edges) > max_connections:
            new_edges = set(random.sample(new_edges, max_connections))

        # Enforce minimum connections
        while len(new_edges) < min_connections and len(new_edges) < len(neighbors):
            remaining_neighbors = set(neighbors) - new_edges
            if remaining_neighbors:
                new_neighbor = random.choice(list(remaining_neighbors))
                new_edges.add(new_neighbor)

        return new_edges

class ConnectLife(Rule):
    """A network-based variant of Conway's Game of Life that uses rule tables for both state transitions and edge updates."""

    PARAMETER_METADATA = {
            "state_rule_table": {
                "type": dict,
                "description": "Rule table for state transitions. Keys are (current_state, neighbor_pattern, connection_pattern)",
                "default": {}
            },
            "edge_rule_table": {
                "type": dict,
                "description": "Rule table for edge updates. Keys are (self_state, neighbor_state, connection_pattern)",
                "default": {}
            },
            "min_connections": {
                "type": int,
                "description": "Minimum number of connections required for a cell to be considered active.",
                "min": 0,
                "max": 8
            },
            "max_connections": {
                "type": int,
                "description": "Maximum number of connections allowed for a cell to prevent overcrowding.",
                "min": 0,
                "max": 8
            },
            "min_shared_neighbors": {
                "type": int,
                "description": "Minimum number of shared neighbors required to maintain an edge.",
                "min": 0,
                "max": 8
            },
            "use_clustering": {
                "type": bool,
                "description": "Enable clustering coefficient-based state changes"
            },
            "min_clustering": {
                "type": float,
                "description": "Minimum clustering coefficient required for survival",
                "min": 0.0,
                "max": 1.0
            },
            "max_clustering": {
                "type": float,
                "description": "Maximum clustering coefficient allowed before death",
                "min": 0.0,
                "max": 1.0
            },
            "use_entropy": {
                "type": bool,
                "description": "Enable entropy-based state modifications"
            },
            "min_entropy": {
                "type": float,
                "description": "Minimum entropy required for state changes",
                "min": 0.0,
                "max": 1.0
            },
            "max_entropy": {
                "type": float,
                "description": "Maximum entropy allowed before state changes",
                "min": 0.0,
                "max": 1.0
            },
            "edge_memory": {
                "type": int,
                "description": "Number of steps to remember previous edges",
                "min": 0,
                "max": 10
            },
            "state_memory": {
                "type": int,
                "description": "Number of steps to remember previous states",
                "min": 0,
                "max": 10
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT"
                ]
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        
    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on rule table and metrics"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0:
            return -1.0

        # Get rule table and parameters
        state_rule_table = self.get_param('state_rule_table')
        min_connections = self.get_param('min_connections')
        max_connections = self.get_param('max_connections')
        use_clustering = self.get_param('use_clustering')
        use_entropy = self.get_param('use_entropy')

        # Get connection count
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])
        
        # Check connection bounds
        if not (min_connections <= connected_neighbors <= max_connections):
            return 0.0

        # Create neighbor pattern string
        neighbor_pattern = ''.join(['1' if neighborhood_data.states[n] > 0 else '0' 
                                  for n in neighbors])
        
        # Create connection pattern string
        connection_pattern = ''.join(['1' if neighborhood_data.edge_matrix[node_idx, n] else '0'
                                    for n in neighbors])

        # Create lookup key
        key = f"({int(current_state)}, {neighbor_pattern}, {connection_pattern})"
        
        # Get base state from rule table
        new_state = state_rule_table.get(key, state_rule_table.get('default', 0))

        # Apply metric modifiers if enabled
        survives = True
        
        if use_clustering:
            clustering = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)
            min_clustering = self.get_param('min_clustering')
            max_clustering = self.get_param('max_clustering')
            if not (min_clustering <= clustering <= max_clustering):
                survives = False

        if use_entropy and survives:
            entropy = self.get_metric(StateEntropy, node_idx, neighborhood_data)
            min_entropy = self.get_param('min_entropy')
            max_entropy = self.get_param('max_entropy')
            if not (min_entropy <= entropy <= max_entropy):
                survives = False

        # Handle ties using tiebreaker
        if new_state == current_state:
            new_state = int(TieBreaker.resolve(
                current_state,
                neighborhood_data.states[neighbors[0]] if len(neighbors) > 0 else 0,
                connected_neighbors,
                np.sum(neighborhood_data.edge_matrix[neighbors[0]]) if len(neighbors) > 0 else 0,
                TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
            ))

        # Update state history
        state_memory = self.get_param('state_memory')
        if state_memory > 0:
            if not hasattr(self, 'state_history'):
                self.state_history = []
            self.state_history.append({node_idx: new_state})
            if len(self.state_history) > state_memory:
                self.state_history.pop(0)

        return float(new_state) if survives else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on rule table and metrics"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get parameters
        edge_rule_table = self.get_param('edge_rule_table')
        min_connections = self.get_param('min_connections')
        max_connections = self.get_param('max_connections')
        min_shared_neighbors = self.get_param('min_shared_neighbors')

        # Get current connection pattern
        connection_pattern = ''.join(['1' if neighborhood_data.edge_matrix[node_idx, n] else '0'
                                    for n in neighbors])

        current_connections = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            # Calculate shared neighbors
            shared = np.sum(neighborhood_data.edge_matrix[node_idx] & 
                          neighborhood_data.edge_matrix[n])

            # Create lookup key
            key = f"({int(neighborhood_data.states[node_idx] > 0)}, {int(neighborhood_data.states[n] > 0)}, {connection_pattern})"
            
            # Get action from rule table
            action = edge_rule_table.get(key, edge_rule_table.get('default', 'maintain'))

            # Apply action with connection bounds
            should_connect = False
            
            if action == 'add':
                should_connect = True
            elif action == 'maintain' and neighborhood_data.edge_matrix[node_idx, n]:
                should_connect = True
            elif action == 'remove':
                should_connect = False

            # Check connection bounds
            if should_connect:
                if current_connections >= max_connections:
                    should_connect = False
                elif shared < min_shared_neighbors:
                    should_connect = False

            if should_connect:
                new_edges.add(n)

        # Ensure minimum connections if possible
        if len(new_edges) < min_connections:
            available = [n for n in neighbors 
                        if neighborhood_data.states[n] > 0 and n not in new_edges]
            while len(new_edges) < min_connections and available:
                n = available.pop(random.randrange(len(available)))
                new_edges.add(n)

        return new_edges
       
class LifeAndDeath(Rule):
    """
    Life and Death: Creates complex patterns through detailed birth/death dynamics.
    Features balanced thresholds for survival and reproduction, with multiple
    factors influencing state transitions and edge formation.
    """

    PARAMETER_METADATA = {
            "survival_min_active": {
                "type": int,
                "description": "Minimum active neighbors required for a cell to survive.",
                "min": 0,
                "max": 8
            },
            "survival_max_active": {
                "type": int,
                "description": "Maximum active neighbors allowed for a cell to survive.",
                "min": 0,
                "max": 8
            },
            "survival_min_connected": {
                "type": int,
                "description": "Minimum number of connected neighbors required for a cell to survive.",
                "min": 0,
                "max": 8
            },
            "survival_max_connected": {
                "type": int,
                "description": "Maximum number of connected neighbors allowed for a cell to survive.",
                "min": 0,
                "max": 8
            },
            "birth_min_active": {
                "type": int,
                "description": "Minimum number of active neighbors required for a new cell to be born.",
                "min": 0,
                "max": 8
            },
            "birth_max_active": {
                "type": int,
                "description": "Maximum number of active neighbors allowed for a new cell to be born.",
                "min": 0,
                "max": 8
            },
            "birth_min_connected": {
                "type": int,
                "description": "Minimum number of connected neighbors required for a new cell to be born.",
                "min": 0,
                "max": 8
            },
            "birth_max_connected": {
                "type": int,
                "description": "Maximum number of connected neighbors allowed for a new cell to be born.",
                "min": 0,
                "max": 8
            },
            "use_edge_density": {
                "type": bool,
                "description": "Enable edge density-based state changes"
            },
            "survival_min_density": {
                "type": float,
                "description": "Minimum edge density required for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "survival_max_density": {
                "type": float,
                "description": "Maximum edge density allowed for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "birth_min_density": {
                "type": float,
                "description": "Minimum edge density required for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "birth_max_density": {
                "type": float,
                "description": "Maximum edge density allowed for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "use_clustering": {
                "type": bool,
                "description": "Enable clustering coefficient-based state changes"
            },
            "survival_min_clustering": {
                "type": float,
                "description": "Minimum clustering coefficient required for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "survival_max_clustering": {
                "type": float,
                "description": "Maximum clustering coefficient allowed for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "birth_min_clustering": {
                "type": float,
                "description": "Minimum clustering coefficient required for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "birth_max_clustering": {
                "type": float,
                "description": "Maximum clustering coefficient allowed for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "use_entropy": {
                "type": bool,
                "description": "Enable entropy-based state changes"
            },
            "survival_min_entropy": {
                "type": float,
                "description": "Minimum entropy required for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "survival_max_entropy": {
                "type": float,
                "description": "Maximum entropy allowed for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "birth_min_entropy": {
                "type": float,
                "description": "Minimum entropy required for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "birth_max_entropy": {
                "type": float,
                "description": "Maximum entropy allowed for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "edge_formation_rate": {
                "type": float,
                "description": "Base probability of forming new edges",
                "min": 0.0,
                "max": 1.0
            },
            "edge_removal_rate": {
                "type": float,
                "description": "Base probability of removing existing edges",
                "min": 0.0,
                "max": 1.0
            },
            "min_edges": {
                "type": int,
                "description": "Minimum edges a node must maintain",
                "min": 0,
                "max": 8
            },
            "max_edges": {
                "type": int,
                "description": "Maximum edges a node can have",
                "min": 0,
                "max": 8
            },
            "min_shared_neighbors": {
                "type": int,
                "description": "Minimum shared neighbors required to maintain an edge",
                "min": 0,
                "max": 8
            },
            "use_random_death": {
                "type": bool,
                "description": "Enable random death chance"
            },
            "death_probability": {
                "type": float,
                "description": "Probability of random death when enabled",
                "min": 0.0,
                "max": 1.0
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        
    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on survival and birth rules"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        # Handle empty cell case
        if current_state == -1.0:
            return -1.0

        # Calculate basic metrics
        active_neighbors = np.sum(neighborhood_data.states[neighbors] > 0)
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        # Get optional metrics
        if self.get_param('use_edge_density', False):
            edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)
        else:
            edge_density = 0.5

        if self.get_param('use_clustering', False):
            clustering = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)
        else:
            clustering = 0.5

        if self.get_param('use_entropy', False):
            entropy = self.get_metric(StateEntropy, node_idx, neighborhood_data)
        else:
            entropy = 0.5

        if current_state > 0:  # Active node
            # Get survival parameters
            survival_min_active = self.get_param('survival_min_active')
            survival_max_active = self.get_param('survival_max_active')
            survival_min_connected = self.get_param('survival_min_connected')
            survival_max_connected = self.get_param('survival_max_connected')
            
            if None in (survival_min_active, survival_max_active,
                       survival_min_connected, survival_max_connected):
                return 0.0

            # Check basic survival conditions
            survives = (
                survival_min_active <= active_neighbors <= survival_max_active and
                survival_min_connected <= connected_neighbors <= survival_max_connected
            )

            # Check edge density if enabled
            if self.get_param('use_edge_density', False):
                survival_min_density = self.get_param('survival_min_density')
                survival_max_density = self.get_param('survival_max_density')
                if None not in (survival_min_density, survival_max_density):
                    survives = survives and (
                        survival_min_density <= edge_density <= survival_max_density
                    )

            # Check clustering if enabled
            if self.get_param('use_clustering', False):
                survival_min_clustering = self.get_param('survival_min_clustering')
                survival_max_clustering = self.get_param('survival_max_clustering')
                if None not in (survival_min_clustering, survival_max_clustering):
                    survives = survives and (
                        survival_min_clustering <= clustering <= survival_max_clustering
                    )

            # Check entropy if enabled
            if self.get_param('use_entropy', False):
                survival_min_entropy = self.get_param('survival_min_entropy')
                survival_max_entropy = self.get_param('survival_max_entropy')
                if None not in (survival_min_entropy, survival_max_entropy):
                    survives = survives and (
                        survival_min_entropy <= entropy <= survival_max_entropy
                    )

            # Handle random death
            if (survives and 
                self.get_param('use_random_death', False) and
                random.random() < self.get_param('death_probability', 0.0)):
                return 0.0

            # Handle ties
            if active_neighbors in (survival_min_active, survival_max_active):
                survives = TieBreaker.resolve(
                    current_state,
                    neighborhood_data.states[neighbors[0]],
                    connected_neighbors,
                    np.sum(neighborhood_data.edge_matrix[neighbors[0]]),
                    TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
                )

            return 1.0 if survives else 0.0

        else:  # Inactive node
            # Get birth parameters
            birth_min_active = self.get_param('birth_min_active')
            birth_max_active = self.get_param('birth_max_active')
            birth_min_connected = self.get_param('birth_min_connected')
            birth_max_connected = self.get_param('birth_max_connected')
            
            if None in (birth_min_active, birth_max_active,
                       birth_min_connected, birth_max_connected):
                return 0.0

            # Check basic birth conditions
            born = (
                birth_min_active <= active_neighbors <= birth_max_active and
                birth_min_connected <= connected_neighbors <= birth_max_connected
            )

            # Check edge density if enabled
            if self.get_param('use_edge_density', False):
                birth_min_density = self.get_param('birth_min_density')
                birth_max_density = self.get_param('birth_max_density')
                if None not in (birth_min_density, birth_max_density):
                    born = born and (
                        birth_min_density <= edge_density <= birth_max_density
                    )

            # Check clustering if enabled
            if self.get_param('use_clustering', False):
                birth_min_clustering = self.get_param('birth_min_clustering')
                birth_max_clustering = self.get_param('birth_max_clustering')
                if None not in (birth_min_clustering, birth_max_clustering):
                    born = born and (
                        birth_min_clustering <= clustering <= birth_max_clustering
                    )

            # Check entropy if enabled
            if self.get_param('use_entropy', False):
                birth_min_entropy = self.get_param('birth_min_entropy')
                birth_max_entropy = self.get_param('birth_max_entropy')
                if None not in (birth_min_entropy, birth_max_entropy):
                    born = born and (
                        birth_min_entropy <= entropy <= birth_max_entropy
                    )

            # Handle ties
            if active_neighbors in (birth_min_active, birth_max_active):
                born = TieBreaker.resolve(
                    current_state,
                    neighborhood_data.states[neighbors[0]],
                    connected_neighbors,
                    np.sum(neighborhood_data.edge_matrix[neighbors[0]]),
                    TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
                )

            return 1.0 if born else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on node states and metrics"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get edge parameters
        edge_formation_rate = self.get_param('edge_formation_rate')
        edge_removal_rate = self.get_param('edge_removal_rate')
        min_edges = self.get_param('min_edges')
        max_edges = self.get_param('max_edges')
        min_shared_neighbors = self.get_param('min_shared_neighbors')

        if None in (edge_formation_rate, edge_removal_rate,
                   min_edges, max_edges, min_shared_neighbors):
            return new_edges

        current_edges = {n for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}

        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            # Calculate shared neighbors
            shared = np.sum(neighborhood_data.edge_matrix[node_idx] &
                          neighborhood_data.edge_matrix[n])

            if n in current_edges:
                if shared >= min_shared_neighbors:
                    new_edges.add(n)
                elif random.random() >= edge_removal_rate:
                    new_edges.add(n)
            else:
                if random.random() < edge_formation_rate:
                    new_edges.add(n)

        # Ensure minimum edges if possible
        if len(new_edges) < min_edges:
            available = [n for n in neighbors 
                        if neighborhood_data.states[n] > 0 and n not in new_edges]
            while len(new_edges) < min_edges and available:
                n = available.pop(random.randrange(len(available)))
                new_edges.add(n)

        # Enforce maximum edges if needed
        if len(new_edges) > max_edges:
            new_edges = set(random.sample(list(new_edges), max_edges))

        return new_edges
                                            
class NetworkLife(Rule):
    """A rule that evolves the network based on node states, connection counts, neighbor states, and local network metrics."""

    PARAMETER_METADATA = {
            "activation_threshold": {
                "type": float,
                "description": "Threshold for active neighbor ratio to activate a node.",
                "min": 0.0,
                "max": 1.0
            },
            "deactivation_threshold": {
                "type": float,
                "description": "Threshold for active neighbor ratio below which an active node becomes inactive.",
                "min": 0.0,
                "max": 1.0
            },
            "min_connections": {
                "type": int,
                "description": "Minimum connections for a node to be considered active.",
                "min": 0,
                "max": 8
            },
            "max_connections": {
                "type": int,
                "description": "Max connections for a node to be considered active.",
                "min": 0,
                "max": 8
            },
            "connection_preference": {
                "type": str,
                "description": "Preference for connecting to neighbors ('similar', 'lower', 'higher').",
                "allowed_values": ['similar', 'lower', 'higher']
            },
            "similarity_tolerance": {
                "type": int,
                "description": "Tolerance for degree difference when connecting to 'similar' neighbors.",
                "min": 0,
                "max": 8
            },
            "connect_probability": {
                "type": float,
                "description": "Base probability of connecting to a valid neighbor.",
                "min": 0.0,
                "max": 1.0
            },
            "disconnect_probability": {
                "type": float,
                "description": "Probability of disconnecting from a neighbor.",
                "min": 0.0,
                "max": 1.0
            },
            "min_shared_neighbors": {
                "type": int,
                "description": "Minimum number of shared neighbors to maintain an edge.",
                "min": 0,
                "max": 8
            },
            "use_assortativity": {
                "type": bool,
                "description": "Enable assortativity-based edge formation"
            },
            "assortativity_weight": {
                "type": float,
                "description": "Weight given to degree assortativity in edge formation",
                "min": 0.0,
                "max": 1.0
            },
            "use_clustering": {
                "type": bool,
                "description": "Enable clustering coefficient-based state changes"
            },
            "min_clustering": {
                "type": float,
                "description": "Minimum clustering coefficient required for survival",
                "min": 0.0,
                "max": 1.0
            },
            "max_clustering": {
                "type": float,
                "description": "Maximum clustering coefficient allowed before death",
                "min": 0.0,
                "max": 1.0
            },
            "use_betweenness": {
                "type": bool,
                "description": "Enable betweenness centrality-based state changes"
            },
            "min_betweenness": {
                "type": float,
                "description": "Minimum betweenness centrality required for survival",
                "min": 0.0,
                "max": 1.0
            },
            "max_betweenness": {
                "type": float,
                "description": "Maximum betweenness centrality allowed before death",
                "min": 0.0,
                "max": 1.0
            },
            "use_eigenvector": {
                "type": bool,
                "description": "Enable eigenvector centrality-based state changes"
            },
            "min_eigenvector": {
                "type": float,
                "description": "Minimum eigenvector centrality required for survival",
                "min": 0.0,
                "max": 1.0
            },
            "max_eigenvector": {
                "type": float,
                "description": "Maximum eigenvector centrality allowed before death",
                "min": 0.0,
                "max": 1.0
            },
            "use_random_death": {
                "type": bool,
                "description": "Enable random death chance"
            },
            "death_probability": {
                "type": float,
                "description": "Probability of random death when enabled",
                "min": 0.0,
                "max": 1.0
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            },
            "connect_if_neighbor_connections_lt": {
                "type": int,
                "description": "Connect to a neighbor if the neighbor's connections are less than this value.",
                "min": 0,
                "max": 8
            },
            "connect_if_neighbor_connections_gt": {
                "type": int,
                "description": "Connect to a neighbor if the neighbor's connections are greater than this value.",
                "min": 0,
                "max": 8
            },
            "connect_if_neighbor_connections_eq": {
                "type": int,
                "description": "Connect to a neighbor if the neighbor's connections are equal to this value.",
                "min": 0,
                "max": 8
            },
            "disconnect_if_neighbor_connections_lt": {
                "type": int,
                "description": "Disconnect from a neighbor if the neighbor's connections are less than this value.",
                "min": 0,
                "max": 8
            },
            "disconnect_if_neighbor_connections_gt": {
                "type": int,
                "description": "Disconnect from a neighbor if the neighbor's connections are greater than this value.",
                "min": 0,
                "max": 8
            },
            "disconnect_if_neighbor_connections_eq": {
                "type": int,
                "description": "Disconnect from a neighbor if the neighbor's connections are equal to this value.",
                "min": 0,
                "max": 8
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        
    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on connection counts and neighbor states"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0:
            return -1.0

        # Get metrics
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        # Get parameters
        activation_threshold = self.get_param('activation_threshold')
        deactivation_threshold = self.get_param('deactivation_threshold')
        min_connections = self.get_param('min_connections')
        max_connections = self.get_param('max_connections')
        use_clustering = self.get_param('use_clustering')
        use_betweenness = self.get_param('use_betweenness')
        use_eigenvector = self.get_param('use_eigenvector')
        use_random_death = self.get_param('use_random_death')

        # Check survival conditions
        survives = True

        # Basic connection checks
        if not (min_connections <= connected_neighbors <= max_connections):
            survives = False

        # Active neighbor ratio check
        if current_state > 0 and active_ratio < deactivation_threshold:
            survives = False
        elif current_state <= 0 and active_ratio < activation_threshold:
            survives = False

        # Optional metric checks
        if survives and use_clustering:
            clustering = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)
            if not (self.get_param('min_clustering') <= clustering <= self.get_param('max_clustering')):
                survives = False

        if survives and use_betweenness:
            betweenness = self.get_metric(BetweennessCentrality, node_idx, neighborhood_data)
            if not (self.get_param('min_betweenness') <= betweenness <= self.get_param('max_betweenness')):
                survives = False

        if survives and use_eigenvector:
            eigenvector = self.get_metric(EigenvectorCentrality, node_idx, neighborhood_data)
            if not (self.get_param('min_eigenvector') <= eigenvector <= self.get_param('max_eigenvector')):
                survives = False

        # Random death check
        if survives and use_random_death and random.random() < self.get_param('death_probability'):
            survives = False

        return 1.0 if survives else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on connection rules"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get parameters
        connection_preference = self.get_param('connection_preference')
        similarity_tolerance = self.get_param('similarity_tolerance')
        min_shared_neighbors = self.get_param('min_shared_neighbors')
        
        # Connection thresholds
        connect_lt = self.get_param('connect_if_neighbor_connections_lt')
        connect_gt = self.get_param('connect_if_neighbor_connections_gt')
        connect_eq = self.get_param('connect_if_neighbor_connections_eq')
        disconnect_lt = self.get_param('disconnect_if_neighbor_connections_lt')
        disconnect_gt = self.get_param('disconnect_if_neighbor_connections_gt')
        disconnect_eq = self.get_param('disconnect_if_neighbor_connections_eq')

        self_connections = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            neighbor_connections = np.sum(neighborhood_data.edge_matrix[n, neighborhood_data.get_neighbor_indices(n)])
            shared = np.sum(neighborhood_data.edge_matrix[node_idx] & neighborhood_data.edge_matrix[n])

            # Check connection conditions
            should_connect = False

            # Connection preference check
            if connection_preference == 'similar':
                should_connect = abs(self_connections - neighbor_connections) <= similarity_tolerance
            elif connection_preference == 'lower':
                should_connect = neighbor_connections < self_connections
            elif connection_preference == 'higher':
                should_connect = neighbor_connections > self_connections

            # Connection threshold checks
            if connect_lt is not None and neighbor_connections < connect_lt:
                should_connect = True
            if connect_gt is not None and neighbor_connections > connect_gt:
                should_connect = True
            if connect_eq is not None and neighbor_connections == connect_eq:
                should_connect = True

            # Disconnection threshold checks
            if disconnect_lt is not None and neighbor_connections < disconnect_lt:
                should_connect = False
            if disconnect_gt is not None and neighbor_connections > disconnect_gt:
                should_connect = False
            if disconnect_eq is not None and neighbor_connections == disconnect_eq:
                should_connect = False

            # Shared neighbor check
            if shared >= min_shared_neighbors:
                should_connect = True

            if should_connect:
                new_edges.add(n)

        return new_edges
    
class NeighborConnections(Rule):
    """
    Neighbor Connections and States Rule: Determines connections based on
    the states and connection counts of both the self node and its neighbors.
    """

    PARAMETER_METADATA = {
        "connections_for_high_state": {
            "type": int,
            "description": "Number of connections for a cell to be considered in the 'high' state (1.0).",
            "min": 0,
            "max": 8
        },
        "connections_for_medium_state": {
            "type": int,
            "description": "Number of connections for a cell to be considered in the 'medium' state (0.5).",
            "min": 0,
            "max": 8
        },
        "connections_for_low_state": {
            "type": int,
            "description": "Number of connections for a cell to be considered in the 'low' state (0.2).",
            "min": 0,
            "max": 8
        },
        "connect_if_self_state_gt": {
            "type": float,
            "description": "Connect to a neighbor if the cell's own state is greater than this threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_self_state_lt": {
            "type": float,
            "description": "Connect to a neighbor if the cell's own state is less than this threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_self_state_eq": {
            "type": float,
            "description": "Connect to a neighbor if the cell's own state is equal to this value.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_neighbor_state_gt": {
            "type": float,
            "description": "Connect to a neighbor if the neighbor's state is greater than this threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_neighbor_state_lt": {
            "type": float,
            "description": "Connect to a neighbor if the neighbor's state is less than this threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_neighbor_state_eq": {
            "type": float,
            "description": "Connect to a neighbor if the neighbor's state is equal to this value.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_state_difference_gt": {
            "type": float,
            "description": "Connect to a neighbor if the absolute difference between the cell's state and the neighbor's state is greater than this threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_state_difference_lt": {
            "type": float,
            "description": "Connect to a neighbor if the absolute difference between the cell's state and the neighbor's state is less than this threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_if_neighbor_connections_gt_self": {
            "type": bool,
            "description": "If True, connect to neighbors that have more connections than the cell itself."
        },
        "connect_if_neighbor_connections_lt_self": {
            "type": bool,
            "description": "If True, connect to neighbors that have fewer connections than the cell itself."
        },
        "connect_if_neighbor_connections_eq_self": {
            "type": bool,
            "description": "If True, connect to neighbors that have the same number of connections as the cell itself."
        },
        "min_connections_allowed": {
            "type": int,
            "description": "Minimum number of connections a cell is allowed to have.",
            "min": 0,
            "max": 8
        },
        "max_connections_allowed": {
            "type": int,
            "description": "Maximum number of connections a cell is allowed to have.",
            "min": 0,
            "max": 8
        },
        "preferred_connection_range": {
            "type": list,
            "description": "Preferred range for the number of connections a cell should have.",
            "element_type": int,
            "min": 0,
            "max": 8,
            "length": 2
        },
        "base_connection_probability": {
            "type": float,
            "description": "Base probability of forming a new connection with a neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "state_multiplier_effect": {
            "type": float,
            "description": "Multiplier effect of the cell's state on the connection probability.",
            "min": 0.0,
            "max": 1.0
        },
        "maintain_existing_if_stable": {
            "type": bool,
            "description": "If True, maintain existing connections if the cell's state is stable."
        },
        "prevent_isolation": {
            "type": bool,
            "description": "If True, prevent cells from becoming completely isolated by removing their last connection."
        },
        "min_shared_neighbors": {
            "type": int,
            "description": "Minimum number of shared neighbors required to maintain a connection between two cells.",
            "min": 0,
            "max": 8
        },
        "max_shared_neighbors": {
            "type": int,
            "description": "Maximum number of shared neighbors allowed before considering disconnection.",
            "min": 0,
            "max": 8
        },
        "connection_distance_threshold": {
            "type": float,
            "description": "Maximum distance between two cells for a connection to be formed.",
            "min": 0.0
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
        "type": str,
        "description": "Method to resolve ties in state transitions",
        "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        self.state_history: List[Dict[int, float]] = []
        self.edge_history: List[Set[Tuple[int, int]]] = []

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData, node_positions: Optional[np.ndarray] = None, dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on connection count"""
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell case (-1.0)
        if current_state == -1.0:
            # You can define birth conditions here if needed
            return -1.0  # Remain empty for now

        # Determine state based on connection count
        high_state_threshold = self.get_param('connections_for_high_state')
        medium_state_threshold = self.get_param('connections_for_medium_state')
        low_state_threshold = self.get_param('connections_for_low_state')

        if high_state_threshold is not None and connected_neighbors >= high_state_threshold:
            return 1.0  # High state
        elif medium_state_threshold is not None and connected_neighbors >= medium_state_threshold:
            return 0.5  # Medium state
        elif low_state_threshold is not None and connected_neighbors >= low_state_threshold:
            return 0.2  # Low state
        else:
            return 0.0  # Inactive
                                    
    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData, node_positions: Optional[np.ndarray] = None, dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on connection rules and neighbor states"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        current_state = neighborhood_data.states[node_idx]
        self_connections = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        for n in neighbors:
            # Skip if neighbor is empty
            if neighborhood_data.states[n] == -1:
                continue

            neighbor_state = neighborhood_data.states[n]
            neighbor_indices = neighborhood_data.get_neighbor_indices(n)
            neighbor_connections = np.sum(neighborhood_data.edge_matrix[n, neighbor_indices]) if len(neighbor_indices) > 0 else 0 # ADDED check for empty neighbor indices

            connect = False

            # Apply connection rules based on parameters
            connect_if_self_state_gt = self.get_param('connect_if_self_state_gt')
            if connect_if_self_state_gt is not None and current_state > connect_if_self_state_gt:
                connect = True
            connect_if_self_state_lt = self.get_param('connect_if_self_state_lt')
            if connect_if_self_state_lt is not None and current_state < connect_if_self_state_lt:
                connect = True
            connect_if_self_state_eq = self.get_param('connect_if_self_state_eq')
            if connect_if_self_state_eq is not None and current_state == connect_if_self_state_eq:
                connect = True
            connect_if_neighbor_state_gt = self.get_param('connect_if_neighbor_state_gt')
            if connect_if_neighbor_state_gt is not None and isinstance(neighbor_state, (int, float)) and neighbor_state > connect_if_neighbor_state_gt:
                connect = True
            connect_if_neighbor_state_lt = self.get_param('connect_if_neighbor_state_lt')
            if connect_if_neighbor_state_lt is not None and isinstance(neighbor_state, (int, float)) and neighbor_state < connect_if_neighbor_state_lt:
                connect = True
            connect_if_neighbor_state_eq = self.get_param('connect_if_neighbor_state_eq')
            if self.get_param('connect_if_neighbor_state_eq') is not None and neighbor_state == self.params.get('connect_if_neighbor_state_eq'):
                connect = True
            if self.params.get('connect_if_neighbor_connections_gt_self') and (self_connections is not None and neighbor_connections is not None) and neighbor_connections > self_connections:
                connect = True
            if self.params.get('connect_if_neighbor_connections_lt_self') and (self_connections is not None and neighbor_connections is not None) and neighbor_connections < self_connections:
                connect = True
            if self.params.get('connect_if_neighbor_connections_eq_self') and (self_connections is not None and neighbor_connections is not None) and neighbor_connections == self_connections:
                connect = True

            # Apply disconnection rules (only if not already connected)
            if not neighborhood_data.edge_matrix[node_idx, n]:
                disconnect_if_neighbor_connections_lt = self.params.get('disconnect_if_neighbor_connections_lt')
                if disconnect_if_neighbor_connections_lt is not None and neighbor_connections < disconnect_if_neighbor_connections_lt:
                    connect = False  # Override connection
                disconnect_if_neighbor_connections_gt = self.params.get('disconnect_if_neighbor_connections_gt')
                if disconnect_if_neighbor_connections_gt is not None and neighbor_connections > disconnect_if_neighbor_connections_gt:
                    connect = False  # Override connection
                if self.params.get('disconnect_if_neighbor_connections_eq') is not None and neighbor_connections == self.params.get('disconnect_if_neighbor_connections_eq'):
                    connect = False  # Override connection
                disconnect_if_neighbor_states_sum_lt = self.params.get('disconnect_if_neighbor_states_sum_lt')
                
                if disconnect_if_neighbor_states_sum_lt is not None:
                    neighbor_state_sum = np.sum(neighborhood_data.states[neighbor_indices]) if len(neighbor_indices) > 0 else None
                    if neighbor_state_sum is not None and neighbor_state_sum < disconnect_if_neighbor_states_sum_lt:
                        connect = False  # Override connection
                
                disconnect_if_neighbor_states_sum_gt = self.params.get('disconnect_if_neighbor_states_sum_gt')
                if disconnect_if_neighbor_states_sum_gt is not None:
                    neighbor_state_sum = np.sum(neighborhood_data.states[neighbor_indices]) if len(neighbor_indices) > 0 else None
                    if neighbor_state_sum is not None and neighbor_state_sum > disconnect_if_neighbor_states_sum_gt:
                        connect = False  # Override connection
                
                disconnect_if_neighbor_states_sum_eq = self.params.get('disconnect_if_neighbor_states_sum_eq')
                if self.params.get('disconnect_if_neighbor_states_sum_eq') is not None:
                    neighbor_state_sum = np.sum(neighborhood_data.states[neighbor_indices]) if len(neighbor_indices) > 0 else None
                    if neighbor_state_sum is not None and neighbor_state_sum == self.params.get('disconnect_if_neighbor_states_sum_eq'):
                        connect = False  # Override connection

                if connect:
                    new_edges.add(n)

        return new_edges
    
class AdaptiveNetworkRule(Rule):
    """Creates adaptive networks based on local and global conditions."""

    PARAMETER_METADATA = {
            "state_rule_table": {
                "type": dict,
                "description": "Rule table for state transitions. Keys are (current_state, neighbor_pattern, connection_pattern)",
                "default": {}
            },
            "edge_rule_table": {
                "type": dict,
                "description": "Rule table for edge updates. Keys are (self_state, neighbor_state, connection_pattern)",
                "default": {}
            },
            "birth_threshold": {
                "type": float,
                "description": "Minimum active neighbor ratio for an empty cell to become active.",
                "min": 0.0,
                "max": 1.0
            },
            "death_threshold": {
                "type": float,
                "description": "Maximum active neighbor ratio for an active cell to become inactive.",
                "min": 0.0,
                "max": 1.0
            },
            "min_connections": {
                "type": int,
                "description": "Minimum connections for a node to be considered active.",
                "min": 0,
                "max": 8
            },
            "max_connections": {
                "type": int,
                "description": "Max connections for a node to be considered active.",
                "min": 0,
                "max": 8
            },
            "adaptation_rate": {
                "type": float,
                "description": "Rate at which the network adapts to changes.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_threshold": {
                "type": float,
                "description": "Threshold for stability to trigger adaptation.",
                "min": 0.0,
                "max": 1.0
            },
            "use_clustering": {
                "type": bool,
                "description": "Enable clustering coefficient-based state changes"
            },
            "min_clustering": {
                "type": float,
                "description": "Minimum clustering coefficient required for survival",
                "min": 0.0,
                "max": 1.0
            },
            "max_clustering": {
                "type": float,
                "description": "Maximum clustering coefficient allowed before death",
                "min": 0.0,
                "max": 1.0
            },
            "use_entropy": {
                "type": bool,
                "description": "Enable entropy-based state modifications"
            },
            "min_entropy": {
                "type": float,
                "description": "Minimum entropy required for state changes",
                "min": 0.0,
                "max": 1.0
            },
            "max_entropy": {
                "type": float,
                "description": "Maximum entropy allowed before state changes",
                "min": 0.0,
                "max": 1.0
            },
            "edge_memory": {
                "type": int,
                "description": "Number of steps to remember previous edges",
                "min": 0,
                "max": 10
            },
            "state_memory": {
                "type": int,
                "description": "Number of steps to remember previous states",
                "min": 0,
                "max": 10
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        
    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on rule table and adaptive metrics"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0:
            return -1.0

        # Get parameters
        state_rule_table = self.get_param('state_rule_table')
        birth_threshold = self.get_param('birth_threshold')
        death_threshold = self.get_param('death_threshold')
        min_connections = self.get_param('min_connections')
        max_connections = self.get_param('max_connections')
        adaptation_rate = self.get_param('adaptation_rate')
        stability_threshold = self.get_param('stability_threshold')

        # Get metrics
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])
        
        # Check connection bounds
        if not (min_connections <= connected_neighbors <= max_connections):
            return 0.0

        # Create neighbor pattern string
        neighbor_pattern = ''.join(['1' if neighborhood_data.states[n] > 0 else '0' 
                                  for n in neighbors])
        
        # Create connection pattern string
        connection_pattern = ''.join(['1' if neighborhood_data.edge_matrix[node_idx, n] else '0'
                                    for n in neighbors])

        # Create lookup key
        key = f"({int(current_state)}, {neighbor_pattern}, {connection_pattern})"
        
        # Get base state from rule table
        new_state = state_rule_table.get(key, state_rule_table.get('default', 0))

        # Apply adaptive modifiers
        survives = True

        # Birth/death thresholds
        if current_state > 0 and active_ratio < death_threshold:
            survives = False
        elif current_state <= 0 and active_ratio < birth_threshold:
            survives = False

        # Apply metric modifiers if enabled
        if self.get_param('use_clustering'):
            clustering = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)
            if not (self.get_param('min_clustering') <= clustering <= self.get_param('max_clustering')):
                survives = False

        if self.get_param('use_entropy') and survives:
            entropy = self.get_metric(StateEntropy, node_idx, neighborhood_data)
            if not (self.get_param('min_entropy') <= entropy <= self.get_param('max_entropy')):
                survives = False

        # Calculate stability score
        edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)
        stability_score = (active_ratio + edge_density) / 2

        # Apply adaptation based on stability
        if stability_score < stability_threshold:
            adaptation_chance = random.random() * adaptation_rate
            if adaptation_chance > stability_score:
                new_state = 1 - new_state  # Flip state

        # Handle ties using tiebreaker
        if new_state == current_state:
            new_state = int(TieBreaker.resolve(
                current_state,
                neighborhood_data.states[neighbors[0]] if len(neighbors) > 0 else 0,
                connected_neighbors,
                np.sum(neighborhood_data.edge_matrix[neighbors[0]]) if len(neighbors) > 0 else 0,
                TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
            ))

        # Update state memory
        state_memory = self.get_param('state_memory')
        if state_memory > 0:
            if not hasattr(self, 'state_history'):
                self.state_history = []
            self.state_history.append({node_idx: new_state})
            if len(self.state_history) > state_memory:
                self.state_history.pop(0)

        return float(new_state) if survives else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on rule table and adaptive metrics"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get parameters
        edge_rule_table = self.get_param('edge_rule_table')
        min_connections = self.get_param('min_connections')
        max_connections = self.get_param('max_connections')
        min_shared_neighbors = self.get_param('min_shared_neighbors')
        adaptation_rate = self.get_param('adaptation_rate')

        # Get current connection pattern
        connection_pattern = ''.join(['1' if neighborhood_data.edge_matrix[node_idx, n] else '0'
                                    for n in neighbors])

        current_connections = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            # Calculate shared neighbors
            shared = np.sum(neighborhood_data.edge_matrix[node_idx] & 
                          neighborhood_data.edge_matrix[n])

            # Create lookup key
            key = f"({int(neighborhood_data.states[node_idx] > 0)}, {int(neighborhood_data.states[n] > 0)}, {connection_pattern})"
            
            # Get action from rule table
            action = edge_rule_table.get(key, edge_rule_table.get('default', 'maintain'))

            # Apply action with adaptive modification
            should_connect = False
            
            if action == 'add':
                should_connect = True
            elif action == 'maintain' and neighborhood_data.edge_matrix[node_idx, n]:
                should_connect = True
            elif action == 'remove':
                should_connect = False

            # Apply adaptation
            if should_connect:
                # Calculate edge stability
                edge_stability = shared / max(current_connections, 1)
                
                # Chance to adapt based on stability
                if random.random() < adaptation_rate * (1 - edge_stability):
                    should_connect = not should_connect

            # Check connection bounds
            if should_connect:
                if current_connections >= max_connections:
                    should_connect = False
                elif shared < min_shared_neighbors:
                    should_connect = False

            if should_connect:
                new_edges.add(n)

        # Ensure minimum connections if possible
        if len(new_edges) < min_connections:
            available = [n for n in neighbors 
                        if neighborhood_data.states[n] > 0 and n not in new_edges]
            while len(new_edges) < min_connections and available:
                n = available.pop(random.randrange(len(available)))
                new_edges.add(n)

        return new_edges
        
class StablePolygons(Rule):
    """Creates and maintains stable polygonal structures."""

    PARAMETER_METADATA = {
            "min_active_neighbors": {
                "type": int,
                "description": "Minimum active neighbors required for a cell to survive.",
                "min": 0,
                "max": 8
            },
            "max_active_neighbors": {
                "type": int,
                "description": "Maximum active neighbors allowed for a cell to survive.",
                "min": 0,
                "max": 8
            },
            "target_neighbors": {
                "type": int,
                "description": "Target number of neighbors for stable polygon formation.",
                "min": 3,
                "max": 8
            },
            "angle_tolerance": {
                "type": float,
                "description": "Tolerance for the angular spacing between neighbors in a stable polygon.",
                "min": 0.0,
                "max": 1.0
            },
            "edge_length_variance": {
                "type": float,
                "description": "Allowed variance in the lengths of edges connecting a cell to its neighbors.",
                "min": 0.0,
                "max": 1.0
            },
            "use_symmetry": {
                "type": bool,
                "description": "Enable symmetry-based state changes"
            },
            "min_symmetry": {
                "type": float,
                "description": "Minimum symmetry score required for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "max_symmetry": {
                "type": float,
                "description": "Maximum symmetry score allowed for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "use_regularity": {
                "type": bool,
                "description": "Enable polygon regularity checks"
            },
            "min_regularity": {
                "type": float,
                "description": "Minimum regularity score required for survival",
                "min": 0.0,
                "max": 1.0
            },
            "max_regularity": {
                "type": float,
                "description": "Maximum regularity score allowed for survival",
                "min": 0.0,
                "max": 1.0
            },
            "growth_threshold": {
                "type": float,
                "description": "Minimum ratio of active neighbors required for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "growth_rate": {
                "type": float,
                "description": "Probability of a new cell forming in an empty space next to an active cell.",
                "min": 0.0,
                "max": 1.0
            },
            "min_density": {
                "type": float,
                "description": "Minimum density of active cells in the neighborhood required to maintain the structure.",
                "min": 0.0,
                "max": 1.0
            },
            "max_density": {
                "type": float,
                "description": "Maximum density of active cells allowed in the neighborhood to prevent overcrowding.",
                "min": 0.0,
                "max": 1.0
            },
            "edge_formation_rate": {
                "type": float,
                "description": "Base probability of forming new edges",
                "min": 0.0,
                "max": 1.0
            },
            "edge_removal_rate": {
                "type": float,
                "description": "Base probability of removing existing edges",
                "min": 0.0,
                "max": 1.0
            },
            "min_edges": {
                "type": int,
                "description": "Minimum edges a node must maintain",
                "min": 0,
                "max": 8
            },
            "max_edges": {
                "type": int,
                "description": "Maximum edges a node can have",
                "min": 0,
                "max": 8
            },
            "use_angle_preference": {
                "type": bool,
                "description": "Enable preferred angle-based edge formation"
            },
            "preferred_angle": {
                "type": float,
                "description": "Preferred angle between edges (in radians)",
                "min": 0.0,
                "max": 6.28318530718  # 2*pi
            },
            "angle_weight": {
                "type": float,
                "description": "Weight given to angle preference in edge formation",
                "min": 0.0,
                "max": 1.0
            },
            "use_random_death": {
                "type": bool,
                "description": "Enable random death chance"
            },
            "death_probability": {
                "type": float,
                "description": "Probability of random death when enabled",
                "min": 0.0,
                "max": 1.0
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        
    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on polygon formation rules"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0 or node_positions is None:
            return -1.0

        # Get parameters
        min_active = self.get_param('min_active_neighbors')
        max_active = self.get_param('max_active_neighbors')
        target_neighbors = self.get_param('target_neighbors')
        angle_tolerance = self.get_param('angle_tolerance')
        edge_length_variance = self.get_param('edge_length_variance')
        min_density = self.get_param('min_density')
        max_density = self.get_param('max_density')
        growth_threshold = self.get_param('growth_threshold')
        growth_rate = self.get_param('growth_rate')

        # Calculate basic metrics
        active_neighbors = np.sum(neighborhood_data.states[neighbors] > 0)
        edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        # Check basic survival conditions
        survives = (min_active <= active_neighbors <= max_active and
                   min_density <= edge_density <= max_density)

        if survives and len(neighbors) > 0:
            # Calculate polygon metrics
            center = node_positions[node_idx]
            active_positions = node_positions[neighbors][neighborhood_data.states[neighbors] > 0]
            
            if len(active_positions) > 0:
                # Calculate edge lengths and angles
                vectors = active_positions - center
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])
                lengths = np.linalg.norm(vectors, axis=1)
                
                # Sort angles and calculate differences
                angles = np.sort(angles)
                angle_diffs = np.diff(angles)
                angle_diffs = np.append(angle_diffs, 2*np.pi - (angles[-1] - angles[0]))
                
                # Calculate regularity metrics
                target_angle = 2*np.pi / target_neighbors
                angle_variance = np.std(angle_diffs) / target_angle
                length_variance = np.std(lengths) / np.mean(lengths)
                regularity_score = 1.0 - (angle_variance + length_variance) / 2

                # Check symmetry if enabled
                if self.get_param('use_symmetry'):
                    symmetry_score = 1.0 - np.std(angle_diffs) / np.pi
                    if not (self.get_param('min_symmetry') <= symmetry_score <= self.get_param('max_symmetry')):
                        survives = False

                # Check regularity if enabled
                if self.get_param('use_regularity'):
                    if not (self.get_param('min_regularity') <= regularity_score <= self.get_param('max_regularity')):
                        survives = False

                # Check edge length variance
                if length_variance > edge_length_variance:
                    survives = False

        # Birth logic for inactive nodes
        elif current_state <= 0:
            if (active_neighbors / len(neighbors) >= growth_threshold and
                random.random() < growth_rate):
                return 1.0

        # Handle random death
        if survives and self.get_param('use_random_death'):
            if random.random() < self.get_param('death_probability'):
                survives = False

        # Handle ties using tiebreaker
        if survives and active_neighbors in (min_active, max_active):
            survives = TieBreaker.resolve(
                current_state,
                neighborhood_data.states[neighbors[0]] if len(neighbors) > 0 else 0,
                connected_neighbors,
                np.sum(neighborhood_data.edge_matrix[neighbors[0]]) if len(neighbors) > 0 else 0,
                TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
            )

        return 1.0 if survives else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates to maintain polygon structure"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0 or node_positions is None:
            return new_edges

        # Get parameters
        target_neighbors = self.get_param('target_neighbors')
        angle_tolerance = self.get_param('angle_tolerance')
        edge_formation_rate = self.get_param('edge_formation_rate')
        edge_removal_rate = self.get_param('edge_removal_rate')
        min_edges = self.get_param('min_edges')
        max_edges = self.get_param('max_edges')
        use_angle_preference = self.get_param('use_angle_preference')
        preferred_angle = self.get_param('preferred_angle')
        angle_weight = self.get_param('angle_weight')

        center = node_positions[node_idx]
        current_edges = {n for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}

        # Calculate angles to all neighbors
        neighbor_angles = {}
        for n in neighbors:
            if neighborhood_data.states[n] > 0:
                direction = node_positions[n] - center
                angle = np.arctan2(direction[1], direction[0])
                neighbor_angles[n] = angle

        # Process each neighbor
        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            angle = neighbor_angles[n]
            should_connect = False

            if use_angle_preference:
                # Check alignment with preferred angle
                angle_diff = abs(angle - preferred_angle)
                if angle_diff > np.pi:
                    angle_diff = 2*np.pi - angle_diff
                
                # Calculate connection probability based on angle
                angle_factor = 1.0 - (angle_diff / np.pi)
                connection_prob = edge_formation_rate * (1.0 - angle_weight + angle_weight * angle_factor)
                
                should_connect = random.random() < connection_prob
            else:
                # Calculate ideal angle based on target neighbors
                target_angle = 2*np.pi / target_neighbors
                angle_diff = abs(angle % target_angle)
                
                if angle_diff <= angle_tolerance:
                    should_connect = random.random() < edge_formation_rate

            # Maintain existing edges with some probability
            if n in current_edges and not should_connect:
                should_connect = random.random() > edge_removal_rate

            if should_connect:
                new_edges.add(n)

        # Ensure minimum edges if possible
        if len(new_edges) < min_edges:
            available = [n for n in neighbors 
                        if neighborhood_data.states[n] > 0 and n not in new_edges]
            while len(new_edges) < min_edges and available:
                n = available.pop(random.randrange(len(available)))
                new_edges.add(n)

        # Limit maximum edges
        if len(new_edges) > max_edges:
            new_edges = set(sorted(list(new_edges), 
                                 key=lambda n: neighbor_angles.get(n, 0))[:max_edges])

        return new_edges
        
class GeometricAngle(Rule):
    """Creates geometric patterns based on neighbor angles and edge properties."""

    PARAMETER_METADATA = {
            "target_neighbors": {
                "type": int,
                "description": "Target number of neighbors each cell should have to form a stable geometric structure.",
                "min": 0,
                "max": 8
            },
            "preferred_angle": {
                "type": float,
                "description": "Preferred angle between neighbors in a stable geometric structure (in radians).",
                "min": 0.0,
                "max": 6.28318530718  # 2*pi
            },
            "angle_tolerance": {
                "type": float,
                "description": "Tolerance for the angular deviation between neighbors in a stable geometric structure.",
                "min": 0.0,
                "max": 1.0
            },
            "edge_length_variance": {
                "type": float,
                "description": "Allowed variance in the lengths of edges connecting a cell to its neighbors.",
                "min": 0.0,
                "max": 1.0
            },
            "growth_threshold": {
                "type": float,
                "description": "Minimum ratio of active neighbors required for a new cell to be born.",
                "min": 0.0,
                "max": 1.0
            },
            "growth_rate": {
                "type": float,
                "description": "Probability of a new cell forming in an empty space next to an active cell.",
                "min": 0.0,
                "max": 1.0
            },
            "min_density": {
                "type": float,
                "description": "Minimum density of active cells in the neighborhood required to maintain the structure.",
                "min": 0.0,
                "max": 1.0
            },
            "max_density": {
                "type": float,
                "description": "Maximum density of active cells allowed in the neighborhood to prevent overcrowding.",
                "min": 0.0,
                "max": 1.0
            },
            "min_clustering": {
                "type": float,
                "description": "Minimum clustering coefficient required to maintain the stability of the structure.",
                "min": 0.0,
                "max": 1.0
            },
            "edge_formation_rate": {
                "type": float,
                "description": "Probability of forming a new edge between two active cells.",
                "min": 0.0,
                "max": 1.0
            },
            "edge_removal_rate": {
                "type": float,
                "description": "Probability of removing an unstable edge between two cells.",
                "min": 0.0,
                "max": 1.0
            },
            "min_shared_neighbors": {
                "type": int,
                "description": "Minimum number of shared neighbors required to keep an edge between two cells.",
                "min": 0,
                "max": 8
            },
            "angular_stability": {
                "type": float,
                "description": "Angular stability factor that influences the survival of cells based on their angular alignment with neighbors.",
                "min": 0.0,
                "max": 1.0
            },
            "angular_influence": {
                "type": float,
                "description": "Influence of angular alignment on the overall state update of a cell.",
                "min": 0.0,
                "max": 1.0
            },
            "prevent_isolation": {
                "type": bool,
                "description": "If True, prevents cells from becoming completely isolated by removing their last connection."
            },
            "maintain_existing": {
                "type": bool,
                "description": "If True, maintains existing connections between cells, even if they don't perfectly fit the geometric criteria."
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        
    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on geometric relationships"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0 or node_positions is None:
            return -1.0

        # Get parameters
        target_neighbors = self.get_param('target_neighbors')
        min_connections = self.get_param('min_connections')
        max_connections = self.get_param('max_connections')
        preferred_angle = self.get_param('preferred_angle')
        angle_tolerance = self.get_param('angle_tolerance')
        min_density = self.get_param('min_density')
        max_density = self.get_param('max_density')
        angular_stability = self.get_param('angular_stability')
        angular_influence = self.get_param('angular_influence')

        # Calculate basic metrics
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)

        # Check basic survival conditions
        survives = (min_connections <= connected_neighbors <= max_connections and
                   min_density <= edge_density <= max_density)

        if survives and len(neighbors) > 0:
            # Calculate angles between neighbors
            center = node_positions[node_idx]
            neighbor_positions = node_positions[neighbors]
            vectors = neighbor_positions - center
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            
            # Sort angles and calculate differences
            angles = np.sort(angles)
            angle_diffs = np.diff(angles)
            angle_diffs = np.append(angle_diffs, 2*np.pi - (angles[-1] - angles[0]))
            
            # Calculate angular metrics
            target_angle = 2*np.pi / target_neighbors
            angle_deviations = np.abs(angle_diffs - target_angle)
            mean_deviation = float(np.mean(angle_deviations))
            
            # Calculate angular stability score
            stability_score = 1.0 - (mean_deviation / np.pi)
            
            # Apply angular influence
            if stability_score < angular_stability:
                survives = False
            elif random.random() > stability_score * angular_influence:
                survives = False

            # Check clustering if enabled
            if survives and self.get_param('use_clustering'):
                clustering = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)
                if clustering < self.get_param('min_clustering'):
                    survives = False

        # Handle ties using tiebreaker
        if survives and active_ratio == self.get_param('activation_threshold'):
            survives = TieBreaker.resolve(
                current_state,
                neighborhood_data.states[neighbors[0]] if len(neighbors) > 0 else 0,
                connected_neighbors,
                np.sum(neighborhood_data.edge_matrix[neighbors[0]]) if len(neighbors) > 0 else 0,
                TieBreaker[self.get_param('tiebreaker_type', 'RANDOM')]
            )

        return 1.0 if survives else 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on geometric relationships"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0 or node_positions is None:
            return new_edges

        # Get parameters
        preferred_angle = self.get_param('preferred_angle')
        angle_tolerance = self.get_param('angle_tolerance')
        edge_formation_rate = self.get_param('edge_formation_rate')
        edge_removal_rate = self.get_param('edge_removal_rate')
        min_shared_neighbors = self.get_param('min_shared_neighbors')
        maintain_existing = self.get_param('maintain_existing')
        prevent_isolation = self.get_param('prevent_isolation')

        center = node_positions[node_idx]
        current_edges = {n for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}

        # Calculate angles to all neighbors
        neighbor_angles = {}
        for n in neighbors:
            if neighborhood_data.states[n] > 0:
                direction = node_positions[n] - center
                angle = np.arctan2(direction[1], direction[0])
                neighbor_angles[n] = angle

        # Process each neighbor
        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            # Calculate shared neighbors
            shared = np.sum(neighborhood_data.edge_matrix[node_idx] & 
                          neighborhood_data.edge_matrix[n])

            # Check angle alignment
            angle = neighbor_angles[n]
            angle_diff = abs(angle - preferred_angle)
            if angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff

            should_connect = False

            # Check angle alignment
            if angle_diff <= angle_tolerance:
                should_connect = random.random() < edge_formation_rate
            elif n in current_edges:
                # Maintain existing edges with some probability
                if maintain_existing:
                    should_connect = random.random() > edge_removal_rate
                elif shared >= min_shared_neighbors:
                    should_connect = True

            if should_connect:
                new_edges.add(n)

        # Prevent isolation if enabled
        if prevent_isolation and len(new_edges) == 0 and len(current_edges) > 0:
            # Keep at least one existing edge
            new_edges.add(random.choice(list(current_edges)))

        return new_edges
    
class SymmetryRule(Rule):
    """
    Symmetry Rule: Creates and maintains symmetric patterns across multiple axes.
    """

    PARAMETER_METADATA = {
        "num_symmetry_axes": {
            "type": int,
            "description": "Number of symmetry axes to maintain in the pattern.",
            "min": 1,
            "max": 8
        },
        "axis_angle_tolerance": {
            "type": float,
            "description": "Tolerance for the angular alignment of cells with the symmetry axes.",
            "min": 0.0,
            "max": 1.0
        },
        "symmetry_radius": {
            "type": int,
            "description": "Radius within which to check for symmetry around a cell.",
            "min": 1,
            "max": 100
        },
        "symmetry_tolerance": {
            "type": float,
            "description": "Tolerance for matching the states of cells that are symmetrically positioned.",
            "min": 0.0,
            "max": 1.0
        },
        "min_symmetric_nodes": {
            "type": int,
            "description": "Minimum number of cells required to form a symmetric pattern.",
            "min": 1,
            "max": 100
        },
        "max_asymmetric_nodes": {
            "type": int,
            "description": "Maximum number of asymmetric cells allowed in a symmetric pattern.",
            "min": 0,
            "max": 100
        },
        "rotational_order": {
            "type": int,
            "description": "Order of rotational symmetry to enforce (e.g., 4 for 90-degree rotational symmetry).",
            "min": 1,
            "max": 8
        },
        "rotation_tolerance": {
            "type": float,
            "description": "Tolerance for the rotational alignment of cells in a rotationally symmetric pattern.",
            "min": 0.0,
            "max": 1.0
        },
        "pattern_density": {
            "type": float,
            "description": "Target density of active cells within a symmetric pattern.",
            "min": 0.0,
            "max": 1.0
        },
        "pattern_scale": {
            "type": float,
            "description": "Scale factor for the size of the symmetric pattern.",
            "min": 0.1,
            "max": 10.0
        },
        "min_pattern_size": {
            "type": int,
            "description": "Minimum number of cells required to form a symmetric pattern.",
            "min": 1,
            "max": 100
        },
        "max_pattern_size": {
            "type": int,
            "description": "Maximum number of cells allowed in a symmetric pattern.",
            "min": 1,
            "max": 100
        },
        "symmetry_break_threshold": {
            "type": float,
            "description": "Threshold for the symmetry score above which intentional symmetry breaking can occur.",
            "min": 0.0,
            "max": 1.0
        },
        "break_probability": {
            "type": float,
            "description": "Probability of a cell breaking the symmetry if the symmetry score is above the threshold.",
            "min": 0.0,
            "max": 1.0
        },
        "max_broken_duration": {
            "type": int,
            "description": "Maximum number of simulation steps to maintain a broken symmetry.",
            "min": 0,
            "max": 100
        },
        "min_symmetric_connections": {
            "type": int,
            "description": "Minimum number of connections a cell must have to other cells in the symmetric pattern.",
            "min": 0,
            "max": 8
        },
        "max_connections_per_axis": {
            "type": int,
            "description": "Maximum number of connections allowed along each symmetry axis.",
            "min": 0,
            "max": 8
        },
        "connection_angle_weight": {
            "type": float,
            "description": "Weight of the angular alignment in the connection decision.",
            "min": 0.0,
            "max": 1.0
        },
        "maintain_existing_symmetry": {
            "type": bool,
            "description": "If True, preserves existing symmetric patterns by preventing cells from disrupting the symmetry."
        },
        "prevent_isolation": {
            "type": bool,
            "description": "If True, prevents cells from becoming completely isolated by removing their last connection."
        },
        "min_connections": {
            "type": int,
            "description": "Minimum number of connections a cell must have to prevent isolation.",
            "min": 0,
            "max": 8
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on symmetry analysis"""
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell
        if current_state == -1.0:
            # For now, no birth logic specific to symmetry.  Could be added.
            return -1.0

        if node_positions is None:
            return current_state

        center = node_positions[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        neighbor_positions = node_positions[neighbors]
        neighbor_states = neighborhood_data.states[neighbors]

        # Get parameters
        symmetry_tolerance = self.get_param('symmetry_tolerance', 0.2)

        # Analyze symmetry
        rule_params = RuleParameters(**self.params)
        symmetry_data = analyze_symmetry(center, neighbor_positions, neighbor_states, rule_params)

        # Deactivate if symmetry is too low
        if symmetry_data.score < symmetry_tolerance:
            return 0.0
        else:
            return 1.0
            
    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData, node_positions: Optional[np.ndarray] = None, dimension_type: Optional[Dimension] = None) -> Set[int]:
        """
        Compute edge updates to maintain or enhance symmetry.
        If the node is active and positions are provided, the function analyzes symmetry and updates edges accordingly.
        It can also maintain existing symmetry if the option is enabled.
        """
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)

        if neighborhood_data.states[node_idx] > 0 and node_positions is not None:
            center = node_positions[node_idx]
            
            # Get parameters
            maintain_existing_symmetry = self.get_param('maintain_existing_symmetry', False)
            
            # Analyze symmetry
            rule_params = RuleParameters(**self.params)
            symmetry_data = analyze_symmetry(center, node_positions[neighbors], neighborhood_data.states[neighbors], rule_params)

            for n_idx in neighbors:
                if neighborhood_data.states[n_idx] > 0:
                    # Add edges to symmetric pairs
                    for pair_idx in symmetry_data.symmetric_pairs:
                        if n_idx == neighbors[pair_idx[0]] or n_idx == neighbors[pair_idx[1]]:
                            new_edges.add(n_idx)
                            break

                    # Maintain existing edges if option is enabled
                    if maintain_existing_symmetry and neighborhood_data.edge_matrix[node_idx, n_idx]:
                        new_edges.add(n_idx)

        return new_edges
        
class FractalRule(Rule):
    """Creates self-similar patterns at different scales."""

    PARAMETER_METADATA = {
            "min_pattern_size": {
                "type": int,
                "description": "Minimum number of cells required to form a fractal pattern.",
                "min": 1,
                "max": 100
            },
            "max_pattern_size": {
                "type": int,
                "description": "Maximum number of cells allowed in a fractal pattern.",
                "min": 1,
                "max": 100
            },
            "target_branching": {
                "type": int,
                "description": "Target number of branches each cell should have in a fractal pattern.",
                "min": 1,
                "max": 8
            },
            "max_recursion_depth": {
                "type": int,
                "description": "Maximum depth of recursion allowed for fractal patterns.",
                "min": 0,
                "max": 10
            },
            "min_scale": {
                "type": float,
                "description": "Minimum scale factor for fractal patterns.",
                "min": 0.0,
                "max": 10.0
            },
            "max_scale": {
                "type": float,
                "description": "Maximum scale factor for fractal patterns.",
                "min": 0.0,
                "max": 10.0
            },
            "scale_ratio": {
                "type": float,
                "description": "Ratio between successive scales in the fractal pattern.",
                "min": 0.0,
                "max": 10.0
            },
            "scale_tolerance": {
                "type": float,
                "description": "Tolerance for deviations from the ideal scale ratio.",
                "min": 0.0,
                "max": 1.0
            },
            "min_self_similarity": {
                "type": float,
                "description": "Minimum self-similarity score required for a cell to be considered part of a fractal pattern.",
                "min": 0.0,
                "max": 1.0
            },
            "similarity_threshold": {
                "type": float,
                "description": "Threshold for the similarity score above which a new pattern is recognized as a fractal.",
                "min": 0.0,
                "max": 1.0
            },
            "similarity_tolerance": {
                "type": float,
                "description": "Tolerance for deviations in the similarity score.",
                "min": 0.0,
                "max": 1.0
            },
            "target_dimension": {
                "type": float,
                "description": "Target fractal dimension for the patterns.",
                "min": 0.0,
                "max": 3.0
            },
            "dimension_tolerance": {
                "type": float,
                "description": "Tolerance for deviations from the target fractal dimension.",
                "min": 0.0,
                "max": 1.0
            },
            "dimension_weight": {
                "type": float,
                "description": "Weight of the fractal dimension in the overall state update of a cell.",
                "min": 0.0,
                "max": 1.0
            },
            "min_branching_score": {
                "type": float,
                "description": "Minimum branching score required for a cell to be considered part of a branching structure.",
                "min": 0.0,
                "max": 1.0
            },
            "branching_tolerance": {
                "type": float,
                "description": "Tolerance for deviations from the ideal branching structure.",
                "min": 0.0,
                "max": 1.0
            },
            "branch_angle_variance": {
                "type": float,
                "description": "Allowed variance in the angles between branches.",
                "min": 0.0,
                "max": 1.0
            },
            "formation_threshold": {
                "type": float,
                "description": "Threshold for the formation score above which a new fractal pattern is created.",
                "min": 0.0,
                "max": 1.0
            },
            "recursion_threshold": {
                "type": float,
                "description": "Threshold for the recursion score above which a pattern will recursively generate smaller patterns.",
                "min": 0.0,
                "max": 1.0
            },
            "pattern_merge_threshold": {
                "type": float,
                "description": "Threshold for the similarity score above which two patterns will merge.",
                "min": 0.0,
                "max": 1.0
            },
            "min_pattern_stability": {
                "type": float,
                "description": "Minimum stability score required for a pattern to be considered stable.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_threshold": {
                "type": float,
                "description": "Threshold for the overall stability of a cell, based on its participation in stable fractal patterns.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_tolerance": {
                "type": float,
                "description": "Tolerance for deviations in the stability score.",
                "min": 0.0,
                "max": 1.0
            },
            "maintain_symmetry": {
                "type": bool,
                "description": "If True, maintains the symmetry of fractal patterns by preventing cells from disrupting the symmetry."
            },
            "prevent_pattern_collapse": {
                "type": bool,
                "description": "If True, prevents fractal patterns from collapsing by ensuring cells always have enough connections to maintain the pattern."
            },
            "min_pattern_connections": {
                "type": int,
                "description": "Minimum number of connections a cell must have to other cells in a stored pattern.",
                "min": 0,
                "max": 8
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        
    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on fractal properties"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0 or node_positions is None:
            return -1.0

        # Get parameters
        min_pattern_size = self.get_param('min_pattern_size')
        max_pattern_size = self.get_param('max_pattern_size')
        target_branching = self.get_param('target_branching')
        min_scale = self.get_param('min_scale')
        max_scale = self.get_param('max_scale')
        scale_ratio = self.get_param('scale_ratio')
        min_self_similarity = self.get_param('min_self_similarity')
        target_dimension = self.get_param('target_dimension')
        dimension_tolerance = self.get_param('dimension_tolerance')
        min_branching_score = self.get_param('min_branching_score')
        formation_threshold = self.get_param('formation_threshold')

        # Calculate fractal metrics
        if len(neighbors) > 0:
            # Calculate self-similarity score
            self_similarity, fractal_dim, role = calculate_fractal_metrics(
                node_idx, neighborhood_data, node_positions, reference_scale=1.0
            )

            # Calculate branching metrics
            branching_score, scale_potential = calculate_branching_metrics(
                node_idx, neighborhood_data, node_positions, target_branching
            )

            # Check survival conditions
            survives = True

            # Check self-similarity
            if self_similarity < min_self_similarity:
                survives = False

            # Check fractal dimension
            if abs(fractal_dim - target_dimension) > dimension_tolerance:
                survives = False

            # Check branching
            if branching_score < min_branching_score:
                survives = False

            # Check scale
            if node_positions is not None:
                center = node_positions[node_idx]
                neighbor_positions = node_positions[neighbors]
                distances = np.linalg.norm(neighbor_positions - center, axis=1)
                avg_scale = np.mean(distances)
                
                if not (min_scale <= avg_scale <= max_scale):
                    survives = False

            # Check pattern size
            pattern_size = np.sum(neighborhood_data.edge_matrix[node_idx])
            if not (min_pattern_size <= pattern_size <= max_pattern_size):
                survives = False

            # Birth logic for inactive nodes
            if current_state <= 0:
                active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
                if active_ratio >= formation_threshold and branching_score >= min_branching_score:
                    return 1.0

            # Apply stability checks
            if survives:
                stability_score = (self_similarity + branching_score) / 2
                if stability_score < self.get_param('stability_threshold'):
                    survives = False

            # Handle pattern maintenance
            if survives and self.get_param('maintain_symmetry'):
                symmetry_data = analyze_symmetry(
                    center,
                    neighbor_positions,
                    neighborhood_data.states[neighbors],
                    RuleParameters(**self.params)
                )
                if symmetry_data.score < self.get_param('min_symmetry'):
                    survives = False

            return 1.0 if survives else 0.0
        
        return 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates to maintain fractal structure"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0 or node_positions is None:
            return new_edges

        # Get parameters
        target_branching = self.get_param('target_branching')
        scale_ratio = self.get_param('scale_ratio')
        scale_tolerance = self.get_param('scale_tolerance')
        min_pattern_connections = self.get_param('min_pattern_connections')

        center = node_positions[node_idx]
        current_edges = {n for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}

        # Calculate distances to neighbors
        neighbor_distances = {}
        for n in neighbors:
            if neighborhood_data.states[n] > 0:
                distance = np.linalg.norm(node_positions[n] - center)
                neighbor_distances[n] = distance

        # Sort neighbors by distance
        sorted_neighbors = sorted(
            [n for n in neighbors if n in neighbor_distances],
            key=lambda n: neighbor_distances[n]
        )

        # Create fractal branching pattern
        if sorted_neighbors:
            base_distance = neighbor_distances[sorted_neighbors[0]]
            current_scale = base_distance

            for n in sorted_neighbors:
                distance = neighbor_distances[n]
                
                # Check if distance matches current scale
                scale_match = abs(distance / current_scale - 1.0) <= scale_tolerance

                # Check if distance matches next scale
                next_scale_match = abs(distance / (current_scale * scale_ratio) - 1.0) <= scale_tolerance

                if scale_match or next_scale_match:
                    # Calculate branching angle
                    direction = node_positions[n] - center
                    angle = np.arctan2(direction[1], direction[0])
                    
                    # Check if angle fits branching pattern
                    angle_step = 2 * np.pi / target_branching
                    angle_match = any(
                        abs((angle - i * angle_step) % (2 * np.pi)) <= self.get_param('branching_tolerance')
                        for i in range(target_branching)
                    )

                    if angle_match:
                        new_edges.add(n)

                    if next_scale_match:
                        current_scale *= scale_ratio

        # Ensure minimum connections
        if len(new_edges) < min_pattern_connections:
            available = [n for n in sorted_neighbors if n not in new_edges]
            while len(new_edges) < min_pattern_connections and available:
                new_edges.add(available.pop(0))

        # Prevent pattern collapse if enabled
        if self.get_param('prevent_pattern_collapse'):
            if len(current_edges) > 0 and len(new_edges) == 0:
                new_edges.add(next(iter(current_edges)))

        return new_edges
    
class ModularRule(Rule):
    """
    Modular Rule: Creates distinct functional modules with specific internal/external connection patterns.
    """

    PARAMETER_METADATA = {
        "min_module_size": {
            "type": int,
            "description": "Minimum number of cells required to form a module.",
            "min": 1,
            "max": 100
        },
        "max_module_size": {
            "type": int,
            "description": "Maximum number of cells allowed in a module.",
            "min": 1,
            "max": 100
        },
        "optimal_module_size": {
            "type": int,
            "description": "Target number of cells for a module.",
            "min": 1,
            "max": 100
        },
        "min_internal_density": {
            "type": float,
            "description": "Minimum density of connections within a module.",
            "min": 0.0,
            "max": 1.0
        },
        "max_external_density": {
            "type": float,
            "description": "Maximum density of connections from a module to other modules.",
            "min": 0.0,
            "max": 1.0
        },
        "core_node_ratio": {
            "type": float,
            "description": "Target ratio of core nodes within a module.",
            "min": 0.0,
            "max": 1.0
        },
        "boundary_node_ratio": {
            "type": float,
            "description": "Target ratio of boundary nodes within a module.",
            "min": 0.0,
            "max": 1.0
        },
        "bridge_node_ratio": {
            "type": float,
            "description": "Target ratio of bridge nodes within a module.",
            "min": 0.0,
            "max": 1.0
        },
        "specialization_rate": {
            "type": float,
            "description": "Rate at which a module becomes specialized.",
            "min": 0.0,
            "max": 1.0
        },
        "max_specialization": {
            "type": float,
            "description": "Maximum level of specialization a module can reach.",
            "min": 0.0,
            "max": 1.0
        },
        "specialization_threshold": {
            "type": float,
            "description": "Threshold for the specialization score above which a module is considered specialized.",
            "min": 0.0,
            "max": 1.0
        },
        "inter_module_connection_rate": {
            "type": float,
            "description": "Rate at which connections form between different modules.",
            "min": 0.0,
            "max": 1.0
        },
        "max_connected_modules": {
            "type": int,
            "description": "Maximum number of other modules a module can connect to.",
            "min": 0,
            "max": 8
        },
        "min_module_connections": {
            "type": int,
            "description": "Minimum number of connections a module must have to other modules.",
            "min": 0,
            "max": 8
        },
        "module_formation_threshold": {
            "type": float,
            "description": "Threshold for the formation score above which a new module is created.",
            "min": 0.0,
            "max": 1.0
        },
        "module_merge_threshold": {
            "type": float,
            "description": "Threshold for the similarity score above which two modules will merge.",
            "min": 0.0,
            "max": 1.0
        },
        "module_split_threshold": {
            "type": float,
            "description": "Threshold for the dissimilarity score below which a module will split.",
            "min": 0.0,
            "max": 1.0
        },
        "stability_threshold": {
            "type": float,
            "description": "Threshold for the stability score above which a module is considered stable.",
            "min": 0.0,
            "max": 1.0
        },
        "adaptation_rate": {
            "type": float,
            "description": "Rate at which a module adapts to changes in the environment.",
            "min": 0.0,
            "max": 1.0
        },
        "maintain_module_integrity": {
            "type": bool,
            "description": "If True, maintains the integrity of modules by preventing cells from leaving or joining without proper cause."
        },
        "prevent_module_isolation": {
            "type": bool,
            "description": "If True, prevents modules from becoming completely isolated by ensuring they always have at least one connection to other modules."
        },
        "min_module_connections_internal": {
            "type": int,
            "description": "Minimum number of internal connections a module must have to prevent collapse.",
            "min": 0,
            "max": 8
        },
        "same_level_density": {
            "type": float,
            "description": "Target density of connections within the same hierarchical level.",
            "min": 0.0,
            "max": 1.0
        },
        "up_connection_density": {
            "type": float,
            "description": "Target density of connections from a cell to cells in the level above it.",
            "min": 0.0,
            "max": 1.0
        },
        "down_connection_density": {
            "type": float,
            "description": "Target density of connections from a cell to cells in the level below it.",
            "min": 0.0,
            "max": 1.0
        },
        "min_connections_per_level": {
            "type": int,
            "description": "Minimum number of connections a cell must have to each adjacent level.",
            "min": 0,
            "max": 8
        },
        "max_connections_per_level": {
            "type": int,
            "description": "Maximum number of connections a cell is allowed to have to each adjacent level.",
            "min": 0,
            "max": 8
        },
        "up_influence_weight": {
            "type": float,
            "description": "Weight of the influence from cells in higher levels on a cell's state.",
            "min": 0.0,
            "max": 1.0
        },
        "down_influence_weight": {
            "type": float,
            "description": "Weight of the influence from cells in lower levels on a cell's state.",
            "min": 0.0,
            "max": 1.0
        },
        "same_level_weight": {
            "type": float,
            "description": "Weight of the influence from cells within the same level on a cell's state.",
            "min": 0.0,
            "max": 1.0
        },
        "activation_threshold": {
            "type": float,
            "description": "Threshold for the combined influence required for an inactive cell to become active.",
            "min": 0.0,
            "max": 1.0
        },
        "deactivation_threshold": {
            "type": float,
            "description": "Threshold for the combined influence below which an active cell becomes inactive.",
            "min": 0.0,
            "max": 1.0
        },
        "level_activation_bonus": {
            "type": float,
            "description": "Additional activation chance given to cells based on their level in the hierarchy.",
            "min": 0.0,
            "max": 1.0
        },
        "level_up_threshold": {
            "type": float,
            "description": "Threshold for the combined influence required for a cell to move up a level in the hierarchy.",
            "min": 0.0,
            "max": 1.0
        },
        "level_down_threshold": {
            "type": float,
            "description": "Threshold for the combined influence below which a cell moves down a level in the hierarchy.",
            "min": 0.0,
            "max": 1.0
        },
        "level_change_probability": {
            "type": float,
            "description": "Probability of a cell changing levels in the hierarchy, even if it doesn't meet the thresholds.",
            "min": 0.0,
            "max": 1.0
        },
        "maintain_level_diversity": {
            "type": bool,
            "description": "If True, maintains a mix of different levels in the hierarchy."
        },
        "prevent_level_isolation": {
            "type": bool,
            "description": "If True, prevents levels from becoming completely disconnected from each other."
        },
        "min_level_connections": {
            "type": int,
            "description": "Minimum number of connections a level must have to other levels.",
            "min": 0,
            "max": 8
        },
        "connection_stability_bonus": {
            "type": float,
            "description": "Bonus added to the stability of a cell if it has stable connections to other cells.",
            "min": 0.0,
            "max": 1.0
        },
        "level_stability_threshold": {
            "type": float,
            "description": "Threshold for the stability of a level, based on the consistency of cell states within the level.",
            "min": 0.0,
            "max": 1.0
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        self.module_assignments: Dict[int, int] = {}
        self.specialization_scores: Dict[int, float] = {}

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on module analysis"""
        current_state = neighborhood_data.states[node_idx]
        
        # Get parameters safely using .get() with defaults
        activation_threshold = self.get_param('activation_threshold', 0.5)
        deactivation_threshold = self.get_param('deactivation_threshold', 0.3)
        level_activation_bonus = self.get_param('level_activation_bonus', 0.1)
        up_influence_weight = self.get_param('up_influence_weight', 0.3)
        down_influence_weight = self.get_param('down_influence_weight', 0.3)
        same_level_weight = self.get_param('same_level_weight', 0.4)
        
        # Get module assignments
        module_assignments = self.module_assignments  # Replace with actual module assignments
        
        # Analyze hierarchy
        hierarchy_data = analyze_hierarchy(
            node_idx, neighborhood_data, module_assignments, max_level=3
        )
        module_data = analyze_module(
            node_idx, neighborhood_data, module_assignments, self.specialization_scores
        )
        
        # Calculate combined influence
        influence = (
            hierarchy_data.influence_up * up_influence_weight +
            hierarchy_data.influence_down * down_influence_weight +
            hierarchy_data.level_density * same_level_weight
        )
        
        # Adjust influence based on level
        if hierarchy_data.level > 1:
            influence += level_activation_bonus
        
        # State update logic
        if current_state > 0:
            if influence < deactivation_threshold:
                return 0.0  # Deactivate
            else:
                return 1.0  # Maintain
        else:
            if influence > activation_threshold:
                return 1.0  # Activate
            else:
                return 0.0  # Remain inactive

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates to maintain module structure"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        # Get parameters safely using .get() with defaults
        inter_module_connection_rate = self.get_param('inter_module_connection_rate', 0.2)
        same_level_density = self.get_param('same_level_density', 0.5)
        up_connection_density = self.get_param('up_connection_density', 0.3)
        down_connection_density = self.get_param('down_connection_density', 0.2)
        min_connections_per_level = self.get_param('min_connections_per_level', 1)

        # Get module assignments
        module_assignments = self.module_assignments  # Replace with actual module assignments

        # Analyze hierarchy
        hierarchy_data = analyze_hierarchy(
            node_idx, neighborhood_data, module_assignments, max_level=3
        )
        
        # Get neighbors
        same_level_edges = hierarchy_data.level_neighbors.get(hierarchy_data.level, [])

        # Calculate state
        for n in same_level_edges:
            if random.random() < same_level_density:
                new_edges.add(n)
            
        # Add connections to higher and lower levels
        for level, neighbors in hierarchy_data.level_neighbors.items():
            if level < hierarchy_data.level and random.random() < down_connection_density:
                for n in neighbors:
                    new_edges.add(n)
            elif level > hierarchy_data.level and random.random() < up_connection_density:
                for n in neighbors:
                    new_edges.add(n)
                
        # Add inter-module connections
        module_data = analyze_module(
            node_idx, neighborhood_data, module_assignments, self.specialization_scores
        )
        
        for other_module, connected_nodes in module_data.inter_module_connections.items():
            if random.random() < inter_module_connection_rate:
                for n in connected_nodes:
                    new_edges.add(n)
                    
        return new_edges
       
class FlowRule(Rule):
    """Creates structures optimized for directional flow or information transfer."""

    PARAMETER_METADATA = {
            "flow_alignment_threshold": {
                "type": float,
                "description": "Minimum alignment of a cell's direction with the global flow direction to be considered part of the flow.",
                "min": 0.0,
                "max": 1.0
            },
            "flow_direction_weight": {
                "type": float,
                "description": "Weight of the flow direction in the overall state update of a cell.",
                "min": 0.0,
                "max": 1.0
            },
            "flow_variation_tolerance": {
                "type": float,
                "description": "Allowed variation from the main flow direction for a cell to be considered part of the flow.",
                "min": 0.0,
                "max": 1.0
            },
            "min_path_capacity": {
                "type": float,
                "description": "Minimum capacity required for a flow path to be considered viable.",
                "min": 0.0,
                "max": 1.0
            },
            "max_path_capacity": {
                "type": float,
                "description": "Maximum capacity allowed for a flow path to prevent congestion.",
                "min": 0.0,
                "max": 1.0
            },
            "optimal_capacity": {
                "type": float,
                "description": "Target capacity utilization for flow paths.",
                "min": 0.0,
                "max": 1.0
            },
            "max_bottleneck_severity": {
                "type": float,
                "description": "Maximum allowed severity of bottlenecks in the flow network.",
                "min": 0.0,
                "max": 1.0
            },
            "min_path_redundancy": {
                "type": float,
                "description": "Minimum path redundancy required to ensure flow can be rerouted around bottlenecks.",
                "min": 0.0,
                "max": 1.0
            },
            "bottleneck_threshold": {
                "type": float,
                "description": "Threshold for bottleneck detection.",
                "min": 0.0,
                "max": 1.0
            },
            "max_pressure_difference": {
                "type": float,
                "description": "Maximum allowed difference between upstream and downstream pressure.",
                "min": 0.0,
                "max": 1.0
            },
            "pressure_equalization_rate": {
                "type": float,
                "description": "Rate at which cells equalize pressure with their neighbors.",
                "min": 0.0,
                "max": 1.0
            },
            "pressure_threshold": {
                "type": float,
                "description": "Threshold for pressure response.",
                "min": 0.0,
                "max": 1.0
            },
            "path_optimization_rate": {
                "type": float,
                "description": "Rate at which cells optimize their connections to improve flow efficiency.",
                "min": 0.0,
                "max": 1.0
            },
            "path_efficiency_threshold": {
                "type": float,
                "description": "Required path efficiency for a cell to maintain its connections.",
                "min": 0.0,
                "max": 1.0
            },
            "alternative_path_weight": {
                "type": float,
                "description": "Weight given to alternative paths when optimizing flow.",
                "min": 0.0,
                "max": 1.0
            },
            "min_flow_stability": {
                "type": float,
                "description": "Minimum required flow stability for a cell to survive.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_threshold": {
                "type": float,
                "description": "Threshold for stable flow patterns.",
                "min": 0.0,
                "max": 1.0
            },
            "flow_adaptation_rate": {
                "type": float,
                "description": "Rate at which cells adapt their behavior to maintain flow stability.",
                "min": 0.0,
                "max": 1.0
            },
            "maintain_critical_paths": {
                "type": bool,
                "description": "If True, preserves critical flow paths by preventing cells from disrupting the flow."
            },
            "prevent_flow_disruption": {
                "type": bool,
                "description": "If True, prevents cells from completely disrupting the flow by ensuring they always have enough connections to maintain the flow."
            },
            "min_flow_connections": {
                "type": int,
                "description": "Minimum number of flow-aligned connections a cell must have to prevent flow disruption.",
                "min": 0,
                "max": 8
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            },
            "activation_threshold": {
                "type": float,
                "description": "Threshold for active neighbor ratio to activate a node.",
                "min": 0.0,
                "max": 1.0
            },
            "deactivation_threshold": {
                "type": float,
                "description": "Threshold for active neighbor ratio below which an active node becomes inactive.",
                "min": 0.0,
                "max": 1.0
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        
    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on flow dynamics"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if current_state == -1.0 or node_positions is None:
            return -1.0

        # Get parameters
        flow_alignment_threshold = self.get_param('flow_alignment_threshold')
        flow_direction_weight = self.get_param('flow_direction_weight')
        min_path_capacity = self.get_param('min_path_capacity')
        max_path_capacity = self.get_param('max_path_capacity')
        optimal_capacity = self.get_param('optimal_capacity')
        max_bottleneck_severity = self.get_param('max_bottleneck_severity')
        min_path_redundancy = self.get_param('min_path_redundancy')
        min_flow_stability = self.get_param('min_flow_stability')
        activation_threshold = self.get_param('activation_threshold')
        deactivation_threshold = self.get_param('deactivation_threshold')

        # Calculate flow bias direction (e.g., left to right)
        flow_bias = np.array([1.0, 0.0, 0.0]) if dimension_type == Dimension.THREE_D else np.array([1.0, 0.0])

        # Get flow metrics
        flow_data = analyze_flow(
            node_idx,
            neighborhood_data,
            node_positions,
            flow_bias,
            optimal_capacity
        )

        # Check survival conditions
        survives = True

        # Check flow alignment
        alignment = np.dot(flow_data.flow_direction[:len(flow_bias)], flow_bias)
        alignment = alignment / (np.linalg.norm(flow_data.flow_direction[:len(flow_bias)]) * np.linalg.norm(flow_bias) + 1e-6)
        if abs(alignment) < flow_alignment_threshold:
            survives = False

        # Check path capacity
        if not (min_path_capacity <= flow_data.capacity_utilization <= max_path_capacity):
            survives = False

        # Check bottleneck severity
        if flow_data.bottleneck_score > max_bottleneck_severity:
            survives = False

        # Check path redundancy
        if flow_data.path_redundancy < min_path_redundancy:
            survives = False

        # Check flow stability
        if flow_data.flow_stability < min_flow_stability:
            survives = False

        # Check pressure balance
        max_pressure_diff = self.get_param('max_pressure_difference')
        if abs(flow_data.upstream_pressure - flow_data.downstream_pressure) > self.get_param('max_pressure_difference'):
            survives = False

        # Apply pressure equalization
        if survives:
            pressure_rate = self.get_param('pressure_equalization_rate')
            pressure_threshold = self.get_param('pressure_threshold')
            
            if abs(flow_data.upstream_pressure - flow_data.downstream_pressure) > pressure_threshold:
                if random.random() < pressure_rate:
                    survives = False

        # Apply path optimization
        if survives:
            optimization_rate = self.get_param('path_optimization_rate')
            efficiency_threshold = self.get_param('path_efficiency_threshold')
            
            path_efficiency = 1.0 - flow_data.bottleneck_score
            if path_efficiency < efficiency_threshold:
                if random.random() < optimization_rate:
                    survives = False

        # Maintain critical paths
        if survives and self.get_param('maintain_critical_paths'):
            if len(flow_data.critical_paths) > 0 and random.random() < 0.8:
                survives = True

        # Prevent flow disruption
        if survives and self.get_param('prevent_flow_disruption'):
            min_flow_conn = self.get_param('min_flow_connections')
            flow_aligned_connections = sum(1 for path in flow_data.critical_paths 
                                        if alignment > flow_alignment_threshold)
            if flow_aligned_connections < min_flow_conn:
                survives = False

        # Update state based on survival
        if current_state > 0:
            return 1.0 if survives else 0.0
        else:
            active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
            if active_ratio > activation_threshold:
                return 1.0
            else:
                return 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates to optimize flow"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0 or node_positions is None:
            return new_edges

        # Get parameters
        flow_alignment_threshold = self.get_param('flow_alignment_threshold')
        min_flow_connections = self.get_param('min_flow_connections')
        path_optimization_rate = self.get_param('path_optimization_rate')
        edge_formation_rate = self.get_param('edge_formation_rate')
        edge_removal_rate = self.get_param('edge_removal_rate')

        # Calculate flow bias direction
        flow_bias = np.array([1.0, 0.0, 0.0]) if dimension_type == Dimension.THREE_D else np.array([1.0, 0.0])

        # Get flow data
        flow_data = analyze_flow(
            node_idx,
            neighborhood_data,
            node_positions,
            flow_bias,
            self.get_param('optimal_capacity')
        )

        # Calculate angles to neighbors
        neighbor_angles = {}
        for n in neighbors:
            if neighborhood_data.states[n] > 0:
                direction = node_positions[n] - node_positions[node_idx]
                angle = np.arctan2(direction[1], direction[0])
                neighbor_angles[n] = angle

        # Process each neighbor
        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            # Calculate alignment with flow direction
            direction = node_positions[n] - node_positions[node_idx]
            direction = direction / np.linalg.norm(direction)
            alignment = np.dot(direction[:len(flow_bias)], flow_bias)

            # Check if part of critical path
            is_critical = (node_idx, n) in flow_data.critical_paths

            # Check if angle is aligned with flow
            angle = neighbor_angles.get(n, 0.0)
            angle_diff = abs(angle - np.arctan2(flow_bias[1], flow_bias[0]))
            if angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff

            # Calculate connection probability
            connect_prob = edge_formation_rate
            if abs(alignment) >= flow_alignment_threshold:
                connect_prob += 0.2  # Bonus for alignment

            # Connect or disconnect based on probability
            if is_critical or random.random() < connect_prob:
                new_edges.add(n)
            elif neighborhood_data.edge_matrix[node_idx, n] and random.random() < edge_removal_rate:
                continue  # Skip adding the edge, effectively removing it

        # Ensure minimum flow connections
        flow_aligned_edges = sum(1 for n in new_edges if (node_idx, n) in flow_data.critical_paths)
        if flow_aligned_edges < min_flow_connections:
            available = [n for n in neighbors 
                        if (neighborhood_data.states[n] > 0 and 
                            n not in new_edges and 
                            (node_idx, n) in flow_data.critical_paths)]
            while flow_aligned_edges < min_flow_connections and available:
                n = available.pop(random.randrange(len(available)))
                new_edges.add(n)
                flow_aligned_edges += 1

        return new_edges
    
class CompetitiveRule(Rule):
    """Creates structures based on competition for resources or connections."""

    PARAMETER_METADATA = {
            "min_territory_size": {
                "type": int,
                "description": "Minimum number of cells required to form a territory.",
                "min": 1,
                "max": 100
            },
            "max_territory_size": {
                "type": int,
                "description": "Maximum number of cells allowed in a territory.",
                "min": 1,
                "max": 100
            },
            "optimal_territory_size": {
                "type": int,
                "description": "Target number of cells for a territory.",
                "min": 1,
                "max": 100
            },
            "territory_spacing": {
                "type": float,
                "description": "Minimum spatial spacing between territories.",
                "min": 0.0,
                "max": 10.0
            },
            "min_resource_level": {
                "type": float,
                "description": "Minimum resource level required for a territory to maintain its control over a cell.",
                "min": 0.0,
                "max": 1.0
            },
            "optimal_resource_level": {
                "type": float,
                "description": "Target resource level for a territory.",
                "min": 0.0,
                "max": 1.0
            },
            "resource_decay_rate": {
                "type": float,
                "description": "Rate at which resources decay within a territory.",
                "min": 0.0,
                "max": 1.0
            },
            "competition_threshold": {
                "type": float,
                "description": "Threshold for active competition between territories.",
                "min": 0.0,
                "max": 1.0
            },
            "competitive_advantage_threshold": {
                "type": float,
                "description": "Minimum competitive advantage required for a territory to expand into a neighboring cell.",
                "min": 0.0,
                "max": 1.0
            },
            "territory_defense_threshold": {
                "type": float,
                "description": "Minimum defense strength required for a territory to maintain control over its cells.",
                "min": 0.0,
                "max": 1.0
            },
            "alliance_formation_threshold": {
                "type": float,
                "description": "Threshold for the similarity score above which two territories will form an alliance.",
                "min": 0.0,
                "max": 1.0
            },
            "max_alliances": {
                "type": int,
                "description": "Maximum number of alliances a territory can form.",
                "min": 0,
                "max": 8
            },
            "alliance_strength_weight": {
                "type": float,
                "description": "Weight of the alliance strength in the overall state update of a cell.",
                "min": 0.0,
                "max": 1.0
            },
            "expansion_threshold": {
                "type": float,
                "description": "Threshold for the expansion potential above which a territory will attempt to expand.",
                "min": 0.0,
                "max": 1.0
            },
            "max_expansion_rate": {
                "type": float,
                "description": "Maximum rate at which a territory can expand into neighboring cells.",
                "min": 0.0,
                "max": 1.0
            },
            "min_expansion_stability": {
                "type": float,
                "description": "Minimum stability required for a territory to expand.",
                "min": 0.0,
                "max": 1.0
            },
            "min_stability_threshold": {
                "type": float,
                "description": "Minimum stability required for a cell to remain part of a territory.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_bonus_threshold": {
                "type": float,
                "description": "Threshold for the stability bonus above which a cell gains an advantage in the competition for resources.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_decay_rate": {
                "type": float,
                "description": "Rate at which the stability of a territory decays over time.",
                "min": 0.0,
                "max": 1.0
            },
            "maintain_core_territory": {
                "type": bool,
                "description": "If True, maintains the core territory of each territory by preventing cells from leaving without proper cause."
            },
            "prevent_territory_collapse": {
                "type": bool,
                "description": "If True, prevents territories from completely collapsing by ensuring they always have enough cells to maintain their structure."
            },
            "min_territory_connections": {
                "type": int,
                "description": "Minimum number of connections a cell must have to other cells in its territory.",
                "min": 0,
                "max": 8
            },
            "edge_formation_rate": {
                "type": float,
                "description": "Probability of forming a new edge between two active cells.",
                "min": 0.0,
                "max": 1.0
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        self.territories: Dict[int, Territory] = {}
        self.territory_assignments: Dict[int, int] = {}
        self.resource_distribution: np.ndarray = np.random.rand(GlobalSettings.Simulation.get_current_grid_size()**2)
        self.last_territory_id = 0  # Initialize territory ID counter

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on competition and territory dynamics"""
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell
        if current_state == -1.0:
            # Add birth logic here, possibly based on proximity to existing territories
            return -1.0  # Remain empty for now

        # Get parameters
        competition_threshold = self.get_param('competition_threshold')
        min_resource_level = self.get_param('min_resource_level')
        territory_defense_threshold = self.get_param('territory_defense_threshold')

        # Get territory assignment
        territory_id = self.territory_assignments.get(node_idx, -1)
        territory = self.territories.get(territory_id)

        # Calculate competition data
        competition_data = analyze_competition(
            node_idx, neighborhood_data, self.territories, self.territory_assignments, self.resource_distribution
        )

        # Check survival conditions
        survives = True

        # Check resource level
        if territory and territory.resource_level < min_resource_level:
            survives = False

        # Check defense strength
        if territory and competition_data.defense_strength < territory_defense_threshold:
            survives = False

        # Check competitive pressure
        if competition_data.competitive_pressure > competition_threshold:
            survives = False

        # Update state based on survival
        if current_state > 0:
            return 1.0 if survives else 0.0
        else:
            # Attempt to claim cell if conditions are favorable
            if competition_data.resource_access > 0.5 and competition_data.competitive_pressure < 0.3:
                return 1.0
            else:
                return 0.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates to maintain territory and compete with neighbors"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get parameters
        edge_formation_rate = self.get_param('edge_formation_rate')
        alliance_formation_threshold = self.get_param('alliance_formation_threshold')
        max_alliances = self.get_param('max_alliances')

        # Get territory assignment
        territory_id = self.territory_assignments.get(node_idx, -1)
        territory = self.territories.get(territory_id)

        # Calculate competition data
        competition_data = analyze_competition(
            node_idx, neighborhood_data, self.territories, self.territory_assignments, self.resource_distribution
        )

        # Process each neighbor
        for n in neighbors:
            if neighborhood_data.states[n] <= 0:
                continue

            neighbor_territory = self.territory_assignments.get(n, -1)

            # Connect to cells within the same territory
            if neighbor_territory == territory_id:
                new_edges.add(n)
            else:
                # Attempt to form alliances
                if (territory and neighbor_territory != -1 and
                    len(territory.alliances) < max_alliances and
                    competition_data.alliance_strength > alliance_formation_threshold and
                    random.random() < edge_formation_rate):
                    # Form alliance
                    territory.alliances.add(neighbor_territory)
                    new_edges.add(n)

        return new_edges
    
class AdaptiveMemoryRule(Rule):
    """Creates structures that can remember and recreate previous patterns."""

    PARAMETER_METADATA = {
            "max_patterns": {
                "type": int,
                "description": "Maximum number of patterns that can be stored in the memory.",
                "min": 1,
                "max": 100
            },
            "min_pattern_size": {
                "type": int,
                "description": "Minimum number of cells required to form a valid pattern in the memory.",
                "min": 1,
                "max": 100
            },
            "max_pattern_size": {
                "type": int,
                "description": "Maximum number of cells allowed in a pattern.",
                "min": 1,
                "max": 100
            },
            "pattern_match_threshold": {
                "type": float,
                "description": "Threshold for the pattern match score above which a stored pattern is considered a good match to the current state.",
                "min": 0.0,
                "max": 1.0
            },
            "pattern_completion_threshold": {
                "type": float,
                "description": "Threshold for the pattern completion score above which a stored pattern is considered to be actively reconstructed.",
                "min": 0.0,
                "max": 1.0
            },
            "min_pattern_stability": {
                "type": float,
                "description": "Minimum stability score required for a pattern to be stored in the memory.",
                "min": 0.0,
                "max": 1.0
            },
            "memory_decay_rate": {
                "type": float,
                "description": "Rate at which the stability of stored patterns decays over time.",
                "min": 0.0,
                "max": 1.0
            },
            "reinforcement_rate": {
                "type": float,
                "description": "Rate at which the stability of a matching pattern is reinforced.",
                "min": 0.0,
                "max": 1.0
            },
            "pattern_influence_rate": {
                "type": float,
                "description": "Rate at which stored patterns influence the state of cells in the simulation.",
                "min": 0.0,
                "max": 1.0
            },
            "new_pattern_threshold": {
                "type": float,
                "description": "Threshold for the novelty score above which a new pattern is formed.",
                "min": 0.0,
                "max": 1.0
            },
            "pattern_merge_threshold": {
                "type": float,
                "description": "Threshold for the similarity score above which two patterns are merged into a single pattern.",
                "min": 0.0,
                "max": 1.0
            },
            "pattern_split_threshold": {
                "type": float,
                "description": "Threshold for the dissimilarity score below which a pattern is split into smaller patterns.",
                "min": 0.0,
                "max": 1.0
            },
            "reconstruction_strength": {
                "type": float,
                "description": "Strength of the pattern reconstruction effect on cell states.",
                "min": 0.0,
                "max": 1.0
            },
            "completion_influence": {
                "type": float,
                "description": "Influence of pattern completion on the state of cells.",
                "min": 0.0,
                "max": 1.0
            },
            "partial_match_threshold": {
                "type": float,
                "description": "Threshold for the pattern match score above which a partial match is considered valid.",
                "min": 0.0,
                "max": 1.0
            },
            "adaptation_rate": {
                "type": float,
                "description": "Rate at which the rule adapts to changes in the environment by modifying stored patterns.",
                "min": 0.0,
                "max": 1.0
            },
            "novelty_threshold": {
                "type": float,
                "description": "Threshold for the novelty score above which a new pattern is considered novel and worth storing.",
                "min": 0.0,
                "max": 1.0
            },
            "stability_threshold": {
                "type": float,
                "description": "Threshold for the stability score above which a pattern is considered stable and reliable.",
                "min": 0.0,
                "max": 1.0
            },
            "maintain_stable_patterns": {
                "type": bool,
                "description": "If True, maintains stable patterns by preventing them from being easily overwritten or modified."
            },
            "prevent_pattern_decay": {
                "type": bool,
                "description": "If True, prevents patterns from completely decaying and being lost from memory."
            },
            "min_pattern_connections": {
                "type": int,
                "description": "Minimum number of connections a cell must have to other cells in a stored pattern.",
                "min": 0,
                "max": 8
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        self.memory_patterns: List[MemoryPattern] = []
        self.last_pattern_id = 0  # Initialize pattern ID counter

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on adaptive memory"""
        current_state = neighborhood_data.states[node_idx]
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)

        if current_state == -1.0:
            return -1.0

        # Get parameters
        pattern_influence_rate = self.get_param('pattern_influence_rate')
        best_match_threshold = self.get_param('pattern_match_threshold')
        memory_decay_rate = self.get_param('memory_decay_rate')
        novelty_threshold = self.get_param('novelty_threshold')
        maintain_stable_patterns = self.get_param('maintain_stable_patterns')
        stability_threshold = self.get_param('stability_threshold')

        # Analyze memory
        memory_data = analyze_memory(
            node_idx, neighborhood_data, self.memory_patterns, self.cache_generation, memory_decay_rate
        )

        # Influence from matching patterns
        if memory_data.best_match_score > best_match_threshold:
            # Reconstruct based on best matching pattern
            best_pattern = memory_data.matching_patterns[0][0]
            pattern_state = best_pattern.node_states.get(node_idx, current_state)
            new_state = (current_state * (1 - pattern_influence_rate) +
                        pattern_state * pattern_influence_rate)

            # Reinforce the pattern if it's stable
            if best_pattern.stability > stability_threshold:
                best_pattern.frequency += 1

            return float(new_state)
        else:
            # Decay towards inactive state if no strong pattern match
            return float(current_state * 0.9)

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates to maintain patterns and adapt to changes"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:
            return new_edges

        # Get parameters
        pattern_completion_threshold = self.get_param('pattern_completion_threshold')
        edge_formation_rate = self.get_param('edge_formation_rate')
        memory_decay_rate = self.get_param('memory_decay_rate')
        maintain_stable_patterns = self.get_param('maintain_stable_patterns')
        min_pattern_connections = self.get_param('min_pattern_connections')
        novelty_threshold = self.get_param('novelty_threshold')

        # Analyze memory
        memory_data = analyze_memory(
            node_idx, neighborhood_data, self.memory_patterns, self.cache_generation, memory_decay_rate
        )

        # Reconstruct connections from best matching pattern
        if memory_data.best_match_score > 0.5:
            best_pattern = memory_data.matching_patterns[0][0]
            if memory_data.reconstruction_score > pattern_completion_threshold:
                for n in neighbors:
                    if (n in best_pattern.node_states and
                        random.random() < edge_formation_rate):
                        new_edges.add(n)

        # Store new patterns if novel enough
        if memory_data.best_match_score < novelty_threshold:
            self.store_pattern(node_idx, neighborhood_data)

        return new_edges

    def store_pattern(self, node_idx: int, neighborhood_data: NeighborhoodData):
        """Store a new pattern in memory"""
        # Get parameters
        max_patterns = self.get_param('max_patterns')
        min_pattern_size = self.get_param('min_pattern_size')
        max_pattern_size = self.get_param('max_pattern_size')
        stability_threshold = self.get_param('stability_threshold')

        # Get neighbors and states
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        # Create pattern
        pattern_states = {n: float(neighborhood_data.states[n]) for n in neighbors}
        pattern_connections = {(node_idx, n) for n in neighbors if neighborhood_data.edge_matrix[node_idx, n]}
        
        # Check pattern size
        if not (min_pattern_size <= len(pattern_states) <= max_pattern_size):
            return

        # Calculate pattern stability
        stability = self.calculate_pattern_stability(pattern_states, pattern_connections)
        if stability < stability_threshold:
            return

        # Create new pattern
        self.last_pattern_id += 1
        new_pattern = MemoryPattern(
            pattern_id=self.last_pattern_id,
            node_states=pattern_states,
            node_connections=pattern_connections,
            last_seen=self.cache_generation,
            frequency=1,
            stability=stability,
            influence=1.0
        )

        # Add to memory
        self.memory_patterns.append(new_pattern)

        # Enforce max patterns
        if len(self.memory_patterns) > max_patterns:
            self.memory_patterns.pop(0)

    def calculate_pattern_stability(self, pattern_states: Dict[int, float], pattern_connections: Set[Tuple[int, int]]) -> float:
        """Calculate stability of a pattern"""
        # Placeholder: Implement stability calculation based on state and connection consistency
        return 0.7  # Return a default value for now
             
class ArtificialLifeRule(Rule):
    """
    Artificial Life Rule: Creates evolving organisms with metabolism, reproduction, and adaptation.
    """

    PARAMETER_METADATA = {
        "base_metabolism": {
            "type": float,
            "description": "Base energy consumption rate for all cells.",
            "min": 0.0,
            "max": 1.0
        },
        "max_metabolism": {
            "type": float,
            "description": "Maximum metabolic rate a cell can reach.",
            "min": 0.0,
            "max": 1.0
        },
        "energy_transfer_rate": {
            "type": float,
            "description": "Rate at which energy is transferred between neighboring cells.",
            "min": 0.0,
            "max": 1.0
        },
        "min_survival_energy": {
            "type": float,
            "description": "Minimum energy level required for a cell to survive.",
            "min": 0.0,
            "max": 1.0
        },
        "reproduction_threshold": {
            "type": float,
            "description": "Energy level required for a cell to reproduce.",
            "min": 0.0,
            "max": 1.0
        },
        "min_reproduction_age": {
            "type": int,
            "description": "Minimum age (in simulation steps) a cell must reach before it can reproduce.",
            "min": 0,
            "max": 1000
        },
        "max_reproduction_size": {
            "type": int,
            "description": "Maximum size an organism can reach before it stops reproducing.",
            "min": 1,
            "max": 1000
        },
        "offspring_energy_ratio": {
            "type": float,
            "description": "Ratio of energy transferred from a parent cell to its offspring during reproduction.",
            "min": 0.0,
            "max": 1.0
        },
        "base_mutation_rate": {
            "type": float,
            "description": "Base rate of mutation for offspring cells.",
            "min": 0.0,
            "max": 1.0
        },
        "max_mutation_rate": {
            "type": float,
            "description": "Maximum mutation rate for offspring cells.",
            "min": 0.0,
            "max": 1.0
        },
        "mutation_effect_size": {
            "type": float,
            "description": "Size of the effect of a mutation on a cell's genome.",
            "min": 0.0,
            "max": 1.0
        },
        "beneficial_mutation_bias": {
            "type": float,
            "description": "Bias towards beneficial mutations.",
            "min": 0.0,
            "max": 1.0
        },
        "min_adaptation_score": {
            "type": float,
            "description": "Minimum adaptation score required for a cell to survive.",
            "min": 0.0,
            "max": 1.0
        },
        "adaptation_rate": {
            "type": float,
            "description": "Rate at which cells adapt to changes in the environment.",
            "min": 0.0,
            "max": 1.0
        },
        "phenotype_plasticity": {
            "type": float,
            "description": "Ability of a cell to change its phenotype (expressed behavior) in response to environmental changes.",
            "min": 0.0,
            "max": 1.0
        },
        "environmental_sensitivity": {
            "type": float,
            "description": "Sensitivity of cells to changes in the environment.",
            "min": 0.0,
            "max": 1.0
        },
        "min_organism_size": {
            "type": int,
            "description": "Minimum number of cells required to form a viable organism.",
            "min": 1,
            "max": 1000
        },
        "max_organism_size": {
            "type": int,
            "description": "Maximum number of cells allowed in an organism.",
            "min": 1,
            "max": 1000
        },
        "optimal_organism_size": {
            "type": int,
            "description": "Target number of cells for an organism.",
            "min": 1,
            "max": 1000
        },
        "max_organism_age": {
            "type": int,
            "description": "Maximum age (in simulation steps) an organism can reach before it dies.",
            "min": 0,
            "max": 10000
        },
        "specialization_threshold": {
            "type": float,
            "description": "Threshold for the specialization score above which a cell is considered specialized.",
            "min": 0.0,
            "max": 1.0
        },
        "membrane_ratio": {
            "type": float,
            "description": "Target ratio of membrane cells within an organism.",
            "min": 0.0,
            "max": 1.0
        },
        "core_cell_ratio": {
            "type": float,
            "description": "Target ratio of core cells within an organism.",
            "min": 0.0,
            "max": 1.0
        },
        "interaction_range": {
            "type": int,
            "description": "Range within which organisms can interact with each other.",
            "min": 0,
            "max": 100
        },
        "competition_strength": {
            "type": float,
            "description": "Strength of competition between organisms for resources.",
            "min": 0.0,
            "max": 1.0
        },
        "cooperation_bonus": {
            "type": float,
            "description": "Bonus given to cells that cooperate with other cells.",
            "min": 0.0,
            "max": 1.0
        },
        "maintain_integrity": {
            "type": bool,
            "description": "If True, maintains the integrity of organisms by preventing cells from leaving or joining without proper cause."
        },
        "prevent_fragmentation": {
            "type": bool,
            "description": "If True, prevents organisms from fragmenting into smaller, non-viable pieces."
        },
        "min_connected_cells": {
            "type": int,
            "description": "Minimum number of connected cells an organism must have to prevent fragmentation.",
            "min": 1,
            "max": 8
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}
        self.organism_assignments: Dict[int, int] = {}
        self.energy_distribution: np.ndarray = np.zeros(GlobalSettings.Simulation.get_current_grid_size()**2)
        self.energy_levels: Dict[int, float] = {}

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on artificial life rules (placeholder)"""
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell
        if current_state == -1.0:
            # Add birth logic here, possibly based on proximity to existing organisms
            return -1.0  # Remain empty for now

        # Get parameters safely using .get() with defaults
        base_metabolism = self.get_param('base_metabolism', 0.1)
        energy_transfer_rate = self.get_param('energy_transfer_rate', 0.2)
        min_survival_energy = self.get_param('min_survival_energy', 0.3)
        
        # Get organism ID
        organism_id = self.organism_assignments.get(node_idx, -1)
        
        # If not part of an organism, decay and return
        if organism_id == -1:
            return float(max(current_state - base_metabolism, 0.0))
        
        # Get energy balance
        energy_balance, metabolic_rate = calculate_metabolic_metrics(
            node_idx, neighborhood_data, self.energy_distribution, base_metabolism
        )
        
        # Update energy level
        self.energy_levels[organism_id] = self.energy_levels.get(organism_id, 0.0) + energy_balance
        
        # Check for survival
        if self.energy_levels[organism_id] < min_survival_energy:
            return 0.0  # Cell dies due to lack of energy
        
        # Otherwise, maintain state
        return 1.0

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on artificial life rules (placeholder)"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        # Get parameters safely using .get() with defaults
        interaction_range = self.get_param('interaction_range', 3)
        edge_formation_rate = self.get_param('edge_formation_rate', 0.2)
        
        # Get organism ID
        organism_id = self.organism_assignments.get(node_idx, -1)
        
        # If not part of an organism, no edges
        if organism_id == -1:
            return new_edges
        
        # Connect to nearby cells within interaction range
        for n in neighbors:
            if n != node_idx:
                neighbor_organism = self.organism_assignments.get(n, -1)
                
                # Connect to cells in same organism
                if neighbor_organism == organism_id:
                    new_edges.add(n)
                # Connect to cells in other organisms within interaction range
                elif (neighbor_organism != -1 and 
                      node_positions is not None and np.linalg.norm(node_positions[node_idx] - node_positions[n]) < interaction_range and
                      random.random() < edge_formation_rate):
                    new_edges.add(n)
                
        return new_edges
    
class EdgeCountMatchingRule(Rule):
    """
    Edge Count Matching Rule: Nodes try to match the average number of edges
    of their neighbors.
    """

    PARAMETER_METADATA = {
        "activation_threshold": {
            "type": float,
            "description": "Threshold for active neighbor ratio to activate a node.",
            "min": 0.0,
            "max": 1.0
        },
        "edge_match_tolerance": {
            "type": float,
            "description": "How close the edge counts need to be for a connection (fraction of the average neighbor edge count).",
            "min": 0.0,
            "max": 1.0
        },
        "connect_probability": {
            "type": float,
            "description": "Probability of connecting to a valid neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "disconnect_probability": {
            "type": float,
            "description": "Probability of disconnecting from a neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on active neighbor ratio."""
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell case (-1.0)
        if current_state == -1.0:
            # Birth logic: activate if enough active neighbors
            if active_ratio > self.get_param('activation_threshold', 0.4):
                return 1.0  # Cell is born
            else:
                return -1.0 # Remain empty

        # For active cells, stay active based on the threshold
        if active_ratio >= self.get_param('activation_threshold', 0.4):
            return 1.0  # Survive
        else:
            return 0.0  # Become inactive

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on edge count matching."""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        self_connections = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        if neighborhood_data.states[node_idx] > 0:  # Active node
            edge_match_tolerance = self.get_param('edge_match_tolerance', 0.2)
            connect_probability = self.get_param('connect_probability', 0.7)
            disconnect_probability = self.get_param('disconnect_probability', 0.1)

            for n in neighbors:
                if neighborhood_data.states[n] > 0:  # Only consider active neighbors
                    neighbor_connections = np.sum(neighborhood_data.edge_matrix[n, neighborhood_data.get_neighbor_indices(n)])

                    # Calculate the difference in edge counts
                    edge_count_difference = abs(self_connections - neighbor_connections)

                    # Determine connection/disconnection based on tolerance
                    if edge_count_difference <= edge_match_tolerance * self_connections:
                        if random.random() < connect_probability:
                            new_edges.add(n)
                    elif neighborhood_data.edge_matrix[node_idx, n]:  # Existing edge
                        if random.random() < disconnect_probability:
                            continue  # Skip adding the edge, effectively removing it if it existed
                    
                    if neighborhood_data.edge_matrix[node_idx, n]:
                        new_edges.add(n)

        return new_edges
          
class PreferentialAttachmentRule(Rule):
    """
    Preferential Attachment Rule (Modified): Favors connections to nodes with
    degrees similar to the self node, promoting a more balanced network.
    Includes explicit birth/death and edge addition/removal logic.
    """

    PARAMETER_METADATA = {
            "activation_threshold": {
                "type": float,
                "description": "Threshold for active neighbor ratio to activate a node.",
                "min": 0.0,
                "max": 1.0
            },
            "deactivation_threshold": {
                "type": float,
                "description": "Threshold for active neighbor ratio below which an active node becomes inactive.",
                "min": 0.0,
                "max": 1.0
            },
            "min_connections": {
                "type": int,
                "description": "Minimum connections for a node to be considered active.",
                "min": 0,
                "max": 8
            },
            "max_connections": {
                "type": int,
                "description": "Max connections for a node to be considered active.",
                "min": 0,
                "max": 8
            },
            "connection_preference": {
                "type": str,
                "description": "Preference for connecting to neighbors ('similar', 'lower', 'higher').",
                "allowed_values": ['similar', 'lower', 'higher']
            },
            "similarity_tolerance": {
                "type": int,
                "description": "Tolerance for degree difference when connecting to 'similar' neighbors.",
                "min": 0,
                "max": 8
            },
            "connect_probability": {
                "type": float,
                "description": "Base probability of connecting to a valid neighbor.",
                "min": 0.0,
                "max": 1.0
            },
            "disconnect_probability": {
                "type": float,
                "description": "Probability of disconnecting from a neighbor.",
                "min": 0.0,
                "max": 1.0
            },
            "min_shared_neighbors": {
                "type": int,
                "description": "Minimum number of shared neighbors to maintain an edge.",
                "min": 0,
                "max": 8
            },
            "initial_conditions": {
                "type": list,
                "description": "List of allowed initial conditions for this rule",
                "element_type": str,
                "default": ["Random"]
            },
            "tiebreaker_type": {
                "type": str,
                "description": "Method to resolve ties in state transitions",
                "allowed_values": [
                    "HIGHER_STATE",
                    "LOWER_STATE",
                    "MORE_CONNECTIONS",
                    "FEWER_CONNECTIONS",
                    "HIGHER_STATE_MORE_NEIGHBORS",
                    "LOWER_STATE_FEWER_NEIGHBORS",
                    "HIGHER_STATE_FEWER_NEIGHBORS",
                    "LOWER_STATE_MORE_NEIGHBORS",
                    "RANDOM",
                    "AGREEMENT" 
                ]
            }
        }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData, node_positions: Optional[np.ndarray] = None, dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on active neighbor ratio and connectivity"""
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        connected_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])
        current_state = neighborhood_data.states[node_idx]

        # Get parameter values
        activation_threshold = self.params.get('activation_threshold', 0.3) # ADDED default
        min_connections = self.params.get('min_connections', 1) # ADDED default
        max_connections = self.params.get('max_connections', 8) # ADDED default

        if current_state == -1.0: # Empty cell
            # Birth conditions
            if (active_ratio >= activation_threshold and
                connected_neighbors >= min_connections):
                return 1.0  # Born
            else:
                return -1.0 # Remain empty

        if current_state > 0:  # Active node
            # Survival conditions
            if (active_ratio >= activation_threshold and
                connected_neighbors >= min_connections and
                connected_neighbors <= max_connections):
                return 1.0  # Survive
            else:
                return 0.0  # Die
        else:  # Inactive node
            # Birth conditions
            if (active_ratio >= activation_threshold and
                connected_neighbors >= min_connections):
                return 1.0  # Born
            else:
                return 0.0  # Remain inactive
                
    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData, node_positions: Optional[np.ndarray] = None, dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on preferential attachment (modified)"""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        self_connections = np.sum(neighborhood_data.edge_matrix[node_idx, neighbors])

        if neighborhood_data.states[node_idx] > 0:  # Active node
            # Get parameter values
            connection_preference = self.params.get('connection_preference', 'similar') # ADDED default
            similarity_tolerance = self.params.get('similarity_tolerance', 2) # ADDED default
            connect_probability = self.params.get('connect_probability', 0.6) # ADDED default
            disconnect_probability = self.params.get('disconnect_probability', 0.1) # ADDED default
            min_shared_neighbors = self.params.get('min_shared_neighbors', 1) # ADDED default

            for n in neighbors:
                neighbor_connections = np.sum(neighborhood_data.edge_matrix[n, neighborhood_data.get_neighbor_indices(n)])
                    
                # Skip if neighbor is empty
                if neighborhood_data.states[n] == -1:
                    continue

                # Connection logic based on preference
                connect = False
                if connection_preference == 'similar':
                    if abs(self_connections - neighbor_connections) <= similarity_tolerance:
                        connect = True
                elif connection_preference == 'lower':
                    if neighbor_connections < self_connections:
                        connect = True
                elif connection_preference == 'higher':
                    if neighbor_connections > self_connections:
                        connect = True

                if connect and random.random() < connect_probability:
                    new_edges.add(n)

                # Disconnection logic (remove edges if too many connections or not enough shared neighbors)
                if neighborhood_data.edge_matrix[node_idx, n]:  # If edge exists
                    shared_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx] & neighborhood_data.edge_matrix[n])
                    if (self_connections > self.params.get('max_connections', 8) or # ADDED default
                        neighbor_connections > self.params.get('max_connections', 8) or # ADDED default
                        shared_neighbors < min_shared_neighbors):
                        if random.random() < disconnect_probability:
                            continue # Skip adding, effectively removing the edge
                        
                    new_edges.add(n) # keep the edge

        return new_edges

class StateDependentEdgeAngleRule(Rule):
    """
    State-Dependent Edge Angle Rule: Creates connections based on node state and
    relative angles, allowing for multiple angle preferences.
    """

    PARAMETER_METADATA = {
        "activation_threshold": {
            "type": float,
            "description": "Threshold for active neighbor ratio to activate a node.",
            "min": 0.0,
            "max": 1.0
        },
        "active_angle_1": {
            "type": float,
            "description": "Preferred angle 1 for connections when the node is active (in degrees).",
            "min": 0.0,
            "max": 360.0
        },
        "active_tolerance_1": {
            "type": float,
            "description": "Tolerance for angle 1 when the node is active (in degrees).",
            "min": 0.0,
            "max": 180.0
        },
        "active_angle_2": {
            "type": float,
            "description": "Preferred angle 2 for connections when the node is active (in degrees).",
            "min": 0.0,
            "max": 360.0
        },
        "active_tolerance_2": {
            "type": float,
            "description": "Tolerance for angle 2 when the node is active (in degrees).",
            "min": 0.0,
            "max": 180.0
        },
        "inactive_angle_1": {
            "type": float,
            "description": "Preferred angle 1 for connections when the node is inactive (in degrees).",
            "min": 0.0,
            "max": 360.0
        },
        "inactive_tolerance_1": {
            "type": float,
            "description": "Tolerance for angle 1 when the node is inactive (in degrees).",
            "min": 0.0,
            "max": 180.0
        },
        "inactive_angle_2": {
            "type": float,
            "description": "Preferred angle 2 for connections when the node is inactive (in degrees).",
            "min": 0.0,
            "max": 360.0
        },
        "inactive_tolerance_2": {
            "type": float,
            "description": "Tolerance for angle 2 when the node is inactive (in degrees).",
            "min": 0.0,
            "max": 180.0
        },
        "connect_probability": {
            "type": float,
            "description": "Base probability of connecting to a valid neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "disconnect_probability": {
            "type": float,
            "description": "Base probability of disconnecting from a neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on active neighbor ratio."""
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell case (-1.0)
        if current_state == -1.0:
            # Birth logic: activate if enough active neighbors
            if active_ratio > self.get_param('activation_threshold', 0.4):
                return 1.0  # Cell is born
            else:
                return -1.0 # Remain empty

        # Active/inactive cell logic
        activation_threshold = self.get_param('activation_threshold', 0.4)
        if active_ratio >= activation_threshold:
            return 1.0  # Become/remain active
        else:
            return 0.0  # Become/remain inactive

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on neighborhood state matching."""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0 or node_positions is None:  # Only active nodes update edges
            return new_edges

        # Get parameters
        connect_probability = self.get_param('connect_probability', 0.7)
        disconnect_probability = self.get_param('disconnect_probability', 0.1)

        # Get preferred angles and tolerances based on node state
        if neighborhood_data.states[node_idx] > 0:  # Active
            angle1 = self.get_param('active_angle_1', 45.0)
            tolerance1 = self.get_param('active_tolerance_1', 15.0)
            angle2 = self.get_param('active_angle_2', 135.0)
            tolerance2 = self.get_param('active_tolerance_2', 15.0)
        else:  # Inactive
            angle1 = self.get_param('inactive_angle_1', 225.0)
            tolerance1 = self.get_param('inactive_tolerance_1', 15.0)
            angle2 = self.get_param('inactive_angle_2', 315.0)
            tolerance2 = self.get_param('inactive_tolerance_2', 15.0)

        # Convert angles to radians
        angle1_rad = np.radians(angle1)
        angle2_rad = np.radians(angle2)
        tolerance1_rad = np.radians(tolerance1)
        tolerance2_rad = np.radians(tolerance2)

        # Calculate neighbor positions
        center = node_positions[node_idx]
        for n in neighbors:
            if neighborhood_data.states[n] > 0 and node_positions is not None:
                # Calculate angle to neighbor
                direction = node_positions[n] - center
                angle = np.arctan2(direction[1], direction[0])

                # Check if angle is within tolerance of preferred angles
                angle_diff1 = abs(angle - angle1_rad)
                angle_diff2 = abs(angle - angle2_rad)

                # Normalize angle differences to be between 0 and pi
                angle_diff1 = min(angle_diff1, 2*np.pi - angle_diff1)
                angle_diff2 = min(angle_diff2, 2*np.pi - angle_diff2)

                # Connect or disconnect based on angle and probability
                if (angle_diff1 <= tolerance1_rad or angle_diff2 <= tolerance2_rad) and random.random() < connect_probability:
                    new_edges.add(n)
                elif neighborhood_data.edge_matrix[node_idx, n] and random.random() < disconnect_probability:
                    continue  # Skip adding the edge, effectively removing it if it existed
                
                if neighborhood_data.edge_matrix[node_idx, n]:
                    new_edges.add(n)

        return new_edges
        
class NeighborhoodStateMatchingRule(Rule):
    """
    Neighborhood State Matching Rule: Connects nodes based on the similarity
    of the average states of their neighborhoods.
    """

    PARAMETER_METADATA = {
        "activation_threshold": {
            "type": float,
            "description": "Threshold for active neighbor ratio to activate a node.",
            "min": 0.0,
            "max": 1.0
        },
        "state_match_threshold": {
            "type": float,
            "description": "Threshold for the difference in average neighbor states.  If the difference is BELOW this, connection is favored.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_probability": {
            "type": float,
            "description": "Base probability of connecting to a valid neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "disconnect_probability": {
            "type": float,
            "description": "Base probability of disconnecting from a neighbor.",
            "min": 0.0,
            "max": 1.0
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on a weighted combination of metrics."""
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell case (-1.0)
        if current_state == -1.0:
            # Use a simple birth condition for now
            if self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data) > 0.5:
                return 1.0  # Cell is born
            return -1.0

        # Get metrics
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        avg_neighbor_degree = self.get_metric(AverageNeighborDegree, node_idx, neighborhood_data)
        clustering_coefficient = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)

        # Get parameter values
        activation_threshold = self.get_param('activation_threshold', 0.6)
        active_ratio_weight = self.get_param('active_neighbor_ratio_weight', 0.4)
        avg_degree_weight = self.get_param('average_neighbor_degree_weight', 0.3)
        clustering_weight = self.get_param('clustering_coefficient_weight', 0.3)

        # Calculate weighted sum
        weighted_sum = (active_ratio * active_ratio_weight +
                        avg_neighbor_degree * avg_degree_weight +
                        clustering_coefficient * clustering_weight)

        # Normalize by the sum of weights (to keep it between 0 and 1)
        total_weight = active_ratio_weight + avg_degree_weight + clustering_weight
        if total_weight > 0:
            normalized_sum = weighted_sum / total_weight
        else:
            normalized_sum = 0.0

        # Determine new state
        if current_state > 0:  # Active node
            if normalized_sum >= activation_threshold:
                return 1.0  # Survive
            else:
                return 0.0  # Die
        else:  # Inactive node
            if normalized_sum >= activation_threshold:
                return 1.0  # Born
            else:
                return 0.0  # Remain inactive

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on a weighted combination of metrics."""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:  # Only active nodes update edges
            return new_edges

        # Get parameter values
        connect_threshold = self.get_param('connect_threshold', 0.7)
        disconnect_threshold = self.get_param('disconnect_threshold', 0.3)
        self_edge_density_weight = self.get_param('self_edge_density_weight', 0.4)
        neighbor_edge_density_weight = self.get_param('neighbor_edge_density_weight', 0.4)
        shared_neighbors_weight = self.get_param('shared_neighbors_weight', 0.2)

        self_edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)

        for n in neighbors:
            if neighborhood_data.states[n] == -1:
                continue #skip if empty
            if neighborhood_data.states[n] > 0:  # Only consider active neighbors for connection
                neighbor_edge_density = self.get_metric(EdgeDensity, n, neighborhood_data)
                shared_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx] & neighborhood_data.edge_matrix[n])

                # Calculate weighted sum for edge connection
                edge_score = (self_edge_density * self_edge_density_weight +
                            neighbor_edge_density * neighbor_edge_density_weight +
                            shared_neighbors * shared_neighbors_weight)
                
                # Normalize
                total_weight = self_edge_density_weight + neighbor_edge_density_weight + shared_neighbors_weight
                if total_weight > 0:
                    edge_score /= total_weight

                if edge_score >= connect_threshold:
                    new_edges.add(n)
                
                # Disconnect logic (only if the edge exists)
                if neighborhood_data.edge_matrix[node_idx, n]:
                    if edge_score < disconnect_threshold:
                        # Don't add to new_edges, effectively removing it
                        continue

                # Maintain existing edges
                if neighborhood_data.edge_matrix[node_idx, n]:
                    new_edges.add(n)

        return new_edges
    
class CombinedMetricRule(Rule):
    """
    Combined Metric Rule: Uses a weighted combination of local and neighbor metrics
    to determine node states and edge connections.
    """

    PARAMETER_METADATA = {
        "active_neighbor_ratio_weight": {
            "type": float,
            "description": "Weight of active neighbor ratio in state update.",
            "min": 0.0,
            "max": 1.0
        },
        "average_neighbor_degree_weight": {
            "type": float,
            "description": "Weight of average neighbor degree in state update.",
            "min": 0.0,
            "max": 1.0
        },
        "clustering_coefficient_weight": {
            "type": float,
            "description": "Weight of local clustering coefficient in state update.",
            "min": 0.0,
            "max": 1.0
        },
        "activation_threshold": {
            "type": float,
            "description": "Threshold for combined metrics to activate a node.",
            "min": 0.0,
            "max": 1.0
        },
        "connect_threshold": {
            "type": float,
            "description": "Threshold for combined metrics to add an edge.",
            "min": 0.0,
            "max": 1.0
        },
        "disconnect_threshold": {
            "type": float,
            "description": "Threshold for combined metrics to remove an edge.",
            "min": 0.0,
            "max": 1.0
        },
        "self_edge_density_weight": {
            "type": float,
            "description": "Weight of self node's edge density in edge update.",
            "min": 0.0,
            "max": 1.0
        },
        "neighbor_edge_density_weight": {
            "type": float,
            "description": "Weight of neighbor node's edge density in edge update.",
            "min": 0.0,
            "max": 1.0
        },
        "shared_neighbors_weight": {
            "type": float,
            "description": "Weight of shared neighbors count in edge update.",
            "min": 0.0,
            "max": 1.0
        },
        "initial_conditions": {
            "type": list,
            "description": "List of allowed initial conditions for this rule",
            "element_type": str,
            "default": ["Random"]
        },
        "tiebreaker_type": {
            "type": str,
            "description": "Method to resolve ties in state transitions",
            "allowed_values": [
                "HIGHER_STATE",
                "LOWER_STATE",
                "MORE_CONNECTIONS",
                "FEWER_CONNECTIONS",
                "HIGHER_STATE_MORE_NEIGHBORS",
                "LOWER_STATE_FEWER_NEIGHBORS",
                "HIGHER_STATE_FEWER_NEIGHBORS",
                "LOWER_STATE_MORE_NEIGHBORS",
                "RANDOM",
                "AGREEMENT" 
            ]
        }
    }

    def __init__(self, metadata: RuleMetadata):
        super().__init__(metadata)
        self._params = {}

    def compute_state_update(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> float:
        """Compute new state based on a weighted combination of metrics."""
        current_state = neighborhood_data.states[node_idx]

        # Handle empty cell
        if current_state == -1.0:
            # Add birth logic here, possibly based on proximity to existing patterns
            return -1.0  # Remain empty for now

        # Get metrics
        active_ratio = self.get_metric(ActiveNeighborRatio, node_idx, neighborhood_data)
        avg_neighbor_degree = self.get_metric(AverageNeighborDegree, node_idx, neighborhood_data)
        clustering_coefficient = self.get_metric(ClusteringCoefficient, node_idx, neighborhood_data)

        # Get parameter values
        active_ratio_weight = self.get_param('active_neighbor_ratio_weight', 0.4)
        avg_degree_weight = self.get_param('average_neighbor_degree_weight', 0.3)
        clustering_weight = self.get_param('clustering_coefficient_weight', 0.3)
        activation_threshold = self.get_param('activation_threshold', 0.6)

        # Calculate weighted sum
        weighted_sum = (active_ratio * active_ratio_weight +
                        avg_neighbor_degree * avg_degree_weight +
                        clustering_coefficient * clustering_weight)

        # Normalize by the sum of weights (to keep it between 0 and 1)
        total_weight = active_ratio_weight + avg_degree_weight + clustering_weight
        if total_weight > 0:
            normalized_sum = weighted_sum / total_weight
        else:
            normalized_sum = 0.0

        # Determine new state
        if current_state > 0:  # Active node
            if normalized_sum >= activation_threshold:
                return 1.0  # Survive
            else:
                return 0.0  # Die
        else:  # Inactive node
            if normalized_sum >= activation_threshold:
                return 1.0  # Born
            else:
                return 0.0  # Remain inactive

    def compute_edge_updates(self, node_idx: int, neighborhood_data: NeighborhoodData,
                           node_positions: Optional[np.ndarray] = None,
                           dimension_type: Optional[Dimension] = None) -> Set[int]:
        """Compute edge updates based on a weighted combination of metrics."""
        new_edges = set()
        neighbors = neighborhood_data.get_neighbor_indices(node_idx)
        
        if neighborhood_data.states[node_idx] <= 0:  # Only active nodes update edges
            return new_edges

        # Get parameter values
        connect_threshold = self.get_param('connect_threshold', 0.7)
        disconnect_threshold = self.get_param('disconnect_threshold', 0.3)
        self_edge_density_weight = self.get_param('self_edge_density_weight', 0.4)
        neighbor_edge_density_weight = self.get_param('neighbor_edge_density_weight', 0.4)
        shared_neighbors_weight = self.get_param('shared_neighbors_weight', 0.2)

        self_edge_density = self.get_metric(EdgeDensity, node_idx, neighborhood_data)

        for n in neighbors:
            if neighborhood_data.states[n] == -1:
                continue #skip if empty
            if neighborhood_data.states[n] > 0:  # Only consider active neighbors for connection
                neighbor_edge_density = self.get_metric(EdgeDensity, n, neighborhood_data)
                shared_neighbors = np.sum(neighborhood_data.edge_matrix[node_idx] & neighborhood_data.edge_matrix[n])

                # Calculate weighted sum for edge connection
                edge_score = (self_edge_density * self_edge_density_weight +
                            neighbor_edge_density * neighbor_edge_density_weight +
                            shared_neighbors * shared_neighbors_weight)
                
                # Normalize
                total_weight = self_edge_density_weight + neighbor_edge_density_weight + shared_neighbors_weight
                if total_weight > 0:
                    edge_score /= total_weight

                if edge_score >= connect_threshold:
                    new_edges.add(n)
                
                # Disconnect logic (only if the edge exists)
                if neighborhood_data.edge_matrix[node_idx, n]:
                    if edge_score < disconnect_threshold:
                        # Don't add to new_edges, effectively removing it
                        continue

                # Maintain existing edges
                if neighborhood_data.edge_matrix[node_idx, n]:
                    new_edges.add(n)

        return new_edges