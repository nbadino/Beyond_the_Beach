import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from matplotlib.colors import ListedColormap
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class HotellingTwoDimensional:
    def __init__(self, 
                 n_firms=2, 
                 market_shape=(1, 1), 
                 alpha_demand=10.0, # New: Base demand parameter
                 gamma_demand=1.0,  # New: Price sensitivity of demand
                 t_transport_cost=1.0, # New: Transportation cost rate
                 beta_logit=1.0,    # New: Logit choice sensitivity (formerly beta, but with different role)
                 c=1.0, 
                 max_price=10.0,    # Should be reviewed based on alpha/gamma: p_bar < alpha/gamma
                 d_type='euclidean',
                 rho_type='uniform',
                 density_params=None):
        """
        Initialize the 2D Hotelling model with linear demand and logit choice.
        
        Parameters:
        n_firms (int): Number of firms
        market_shape (tuple): Size of the market in x and y dimensions
        alpha_demand (float): Intercept of the linear demand function q = alpha - gamma*P.
        gamma_demand (float): Slope of the linear demand function (price coefficient).
        t_transport_cost (float): Transportation cost rate.
        beta_logit (float): Sensitivity parameter in the logit choice model.
        c (float): Marginal cost.
        max_price (float): Upper bound for prices.
        d_type (str): Type of distance function ('euclidean', 'manhattan', 'quadratic').
        rho_type (str): Type of density function ('uniform', 'linear', 'gaussian', 'sine', 'multi_gaussian').
        density_params (list): Optional. For 'multi_gaussian', a list of dicts for foci.
        """
        self.n_firms = n_firms
        self.market_shape = market_shape
        self.alpha_demand = alpha_demand
        self.gamma_demand = gamma_demand
        self.t_transport_cost = t_transport_cost
        self.beta_logit = beta_logit
        self.c = c
        self.max_price = max_price # Ensure this is consistent with alpha/gamma to keep demand positive
        self.d_type = d_type
        self.rho_type = rho_type
        self.density_params = density_params
        
        # Initialize firm locations and prices
        self.locations = np.random.uniform(0, 1, (n_firms, 2)) * market_shape
        self.prices = np.ones(n_firms) * c * 1.5  # Starting at 50% markup
        
        # History for tracking convergence
        self.price_history = []
        self.location_history = []
        self.profit_history = []
        self.last_run_iterations = 0
        self.last_run_time = 0

        # For dynamic entry simulation
        self.dynamic_simulation_history = [] # Stores snapshots of the market state over periods
    
    def _initialize_firms(self, n_firms, locations=None, prices=None):
        """Helper to initialize or re-initialize firm data."""
        self.n_firms = n_firms
        if locations is not None:
            self.locations = np.array(locations)
        else:
            self.locations = np.random.uniform(0, 1, (n_firms, 2)) * self.market_shape
        
        if prices is not None:
            self.prices = np.array(prices)
        else:
            self.prices = np.ones(n_firms) * self.c * 1.5
        
        # Reset histories if firms change
        self.price_history = []
        self.location_history = []
        self.profit_history = []

    def add_firm(self, location, price_guess=None):
        """Adds a new firm to the market."""
        new_location = np.array(location).reshape(1, 2)
        self.locations = np.vstack([self.locations, new_location])
        
        new_price = price_guess if price_guess is not None else self.c * 1.5 # Default initial price
        self.prices = np.append(self.prices, new_price)
        
        self.n_firms += 1
        # Histories should be managed by the dynamic simulation loop

    def rho(self, x, y):
        """Population density function based on specified type."""
        if self.rho_type == 'uniform':
            return 1.0
        elif self.rho_type == 'linear':
            # Linear gradient from (0,0) to (1,1)
            return 0.5 + x * y
        elif self.rho_type == 'gaussian':
            # Gaussian centered at market center
            center_x, center_y = self.market_shape[0]/2, self.market_shape[1]/2
            return np.exp(-((x-center_x)**2 + (y-center_y)**2) / 0.2)
        elif self.rho_type == 'sine':
            # Sinusoidal pattern (as used in the paper)
            return 1 + np.sin(np.pi * x) * np.sin(np.pi * y)
        elif self.rho_type == 'multi_gaussian':
            if not (self.density_params and isinstance(self.density_params, list) and len(self.density_params) > 0):
                return 1.0 # Fallback to uniform if no valid params for multi_gaussian

            total_density = 0.0
            for params in self.density_params:
                center_x, center_y = params.get('center', (self.market_shape[0]/2, self.market_shape[1]/2))
                strength = params.get('strength', 1.0)
                sigma = params.get('sigma', np.sqrt(0.1)) # Default sigma such that 2*sigma^2 = 0.2 (consistent with 'gaussian')
                
                if strength > 0 and sigma > 0: # Ensure valid parameters
                    total_density += strength * np.exp(-((x-center_x)**2 + (y-center_y)**2) / (2 * sigma**2))
            
            return total_density
        else:
            return 1.0  # Default uniform
    
    def d(self, consumer_loc, firm_loc):
        """Transportation cost function based on specified type.
        Can handle consumer_loc as a single point (2,) or an array of points (N,2).
        """
        diff = consumer_loc - firm_loc
        # Determine axis for sum based on dimensionality of diff
        # If consumer_loc is (N,2), diff is (N,2), sum along axis 1.
        # If consumer_loc is (2,), diff is (2,), sum along axis 0.
        axis_sum = diff.ndim - 1

        if self.d_type == 'euclidean':
            return np.sqrt(np.sum(diff**2, axis=axis_sum))
        elif self.d_type == 'manhattan':
            return np.sum(np.abs(diff), axis=axis_sum)
        elif self.d_type == 'quadratic':
            # Quadratic distance (strongly convex with μ=2)
            return np.sum(diff**2, axis=axis_sum)
        else:
            return np.sqrt(np.sum(diff**2, axis=axis_sum))  # Default euclidean
    
    def effective_price(self, consumer_loc, firm_idx):
        """Calculate effective price P_i^*(s) = p_i + t * d(s,x_i)."""
        # consumer_loc can be (2,) or (N,2)
        # self.d should handle this and return scalar or (N,)
        return self.prices[firm_idx] + self.t_transport_cost * self.d(consumer_loc, self.locations[firm_idx])
    
    def choice_prob(self, consumer_loc):
        """Calculate the logit choice probabilities Prob(i|s,p) = exp(-beta*P_i^*(s)) / sum_j exp(-beta*P_j^*(s))."""
        # consumer_loc can be (2,) for a single point, or (N,2) for N points.

        if consumer_loc.ndim == 1:  # Single consumer location (2,)
            effective_prices_all_firms = np.array([self.effective_price(consumer_loc, i)
                                                   for i in range(self.n_firms)])
            # effective_prices_all_firms is (n_firms,)
            # Using self.beta_logit as per new model
            logits = -self.beta_logit * effective_prices_all_firms
            max_logit = np.max(logits)  # Scalar for numerical stability
            exp_logits = np.exp(logits - max_logit)  # (n_firms,)
            sum_exp_logits = np.sum(exp_logits)
            if sum_exp_logits == 0: # Avoid division by zero if all exp_logits are zero (e.g. very large P_i^*)
                return np.zeros_like(exp_logits)
            return exp_logits / sum_exp_logits  # (n_firms,)

        elif consumer_loc.ndim == 2:  # Array of consumer locations (N,2)
            N = consumer_loc.shape[0]
            effective_prices_all_firms = np.zeros((N, self.n_firms))
            for i in range(self.n_firms):
                # self.effective_price called with (N,2) consumer_loc returns (N,)
                effective_prices_all_firms[:, i] = self.effective_price(consumer_loc, i)
            
            # effective_prices_all_firms is (N, n_firms)
            # Using self.beta_logit as per new model
            logits = -self.beta_logit * effective_prices_all_firms  # (N, n_firms)
            max_logit = np.max(logits, axis=1, keepdims=True)  # (N, 1) for numerical stability
            exp_logits = np.exp(logits - max_logit)  # (N, n_firms)
            sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True) # (N, 1)
            # Avoid division by zero for rows where sum_exp_logits is zero
            probabilities = np.divide(exp_logits, sum_exp_logits, 
                                      out=np.zeros_like(exp_logits), 
                                      where=sum_exp_logits!=0)
            return probabilities # (N, n_firms)
        else:
            raise ValueError(f"consumer_loc must be of shape (2,) or (N,2), got {consumer_loc.shape}")
    
    def demand_at_location(self, consumer_loc, firm_idx):
        """Calculate demand from a single location or array of locations,
           using linear demand q_i(s) = alpha - gamma * P_i^*(s) and logit choice.
           Demand is rho(s) * q_i(s) * Prob(i|s,p).
        """
        # eff_price will be scalar if consumer_loc is (2,), or (N,) if consumer_loc is (N,2)
        eff_price = self.effective_price(consumer_loc, firm_idx)
        
        # Linear demand: q_i(s) = alpha - gamma * P_i^*(s)
        # Ensure demand is non-negative, as per economic sense (consumers don't supply).
        # The constraint p_bar < alpha/gamma - t*d should ensure P_i^*(s) < alpha/gamma,
        # so q_i(s) > 0. If not, clipping at 0 is a practical approach here.
        quantity_demanded_by_consumer = np.maximum(0, self.alpha_demand - self.gamma_demand * eff_price) # scalar or (N,)
        
        # all_choice_probs will be (n_firms,) or (N, n_firms)
        all_choice_probs = self.choice_prob(consumer_loc)
        
        if consumer_loc.ndim == 1:  # Single consumer_loc (2,)
            choice_probability_for_firm_i = all_choice_probs[firm_idx]  # scalar
            rho_val = self.rho(consumer_loc[0], consumer_loc[1])  # scalar
        elif consumer_loc.ndim == 2:  # Array of consumer_locs (N,2)
            choice_probability_for_firm_i = all_choice_probs[:, firm_idx]  # (N,)
            rho_val = self.rho(consumer_loc[:, 0], consumer_loc[:, 1])  # (N,)
        else:
            raise ValueError(f"consumer_loc must be of shape (2,) or (N,2), got {consumer_loc.shape}")
            
        return rho_val * quantity_demanded_by_consumer * choice_probability_for_firm_i  # scalar or (N,)
    
    def total_demand_grid(self, firm_idx, grid_size=30):
        """Calculate total demand using grid-based numerical integration (vectorized)."""
        x_coords = np.linspace(0, self.market_shape[0], grid_size)
        y_coords = np.linspace(0, self.market_shape[1], grid_size)
        dx = self.market_shape[0] / grid_size
        dy = self.market_shape[1] / grid_size
        
        # Create a meshgrid. indexing='ij' ensures X[i,j] = x_coords[i] and Y[i,j] = y_coords[j]
        # This matches the loop order of the original implementation: x_coords outer, y_coords inner.
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # consumer_locs will be an array of shape (grid_size*grid_size, 2)
        # Each row is a consumer location [x, y]
        consumer_locs = np.vstack([X.ravel(), Y.ravel()]).T
        
        # demands_at_all_points will be an array of shape (grid_size*grid_size,)
        # containing the demand from each consumer grid point for the given firm_idx.
        demands_at_all_points = self.demand_at_location(consumer_locs, firm_idx)
        
        # Sum all demands and multiply by the area element (dx * dy)
        total = np.sum(demands_at_all_points) * dx * dy
        
        return total
    
    def profit(self, firm_idx, grid_size=30):
        """Calculate profit for a firm."""
        demand = self.total_demand_grid(firm_idx, grid_size)
        return (self.prices[firm_idx] - self.c) * demand
    
    def total_profit(self, grid_size=30):
        """Calculate total profit for all firms."""
        return np.array([self.profit(i, grid_size) for i in range(self.n_firms)])
    
    def update_price(self, firm_idx, grid_size=30):
        """Find best-response price for a firm."""
        def negative_profit(price):
            old_price = self.prices[firm_idx]
            self.prices[firm_idx] = price[0]
            profit = self.profit(firm_idx, grid_size)
            self.prices[firm_idx] = old_price
            return -profit
        
        result = minimize(negative_profit, np.array([self.prices[firm_idx]]), 
                         bounds=[(self.c, self.max_price)],
                         method='L-BFGS-B')
        
        if result.success:
            self.prices[firm_idx] = result.x[0]
        else:
            print(f"Price optimization failed for firm {firm_idx}: {result.message}")
        
        return result.success
    
    def update_location(self, firm_idx, grid_size=30):
        """Find best-response location for a firm."""
        def negative_profit(location):
            old_location = self.locations[firm_idx].copy()
            self.locations[firm_idx] = location
            profit = self.profit(firm_idx, grid_size)
            self.locations[firm_idx] = old_location
            return -profit
        
        bounds = [(0, self.market_shape[0]), (0, self.market_shape[1])]
        result = minimize(negative_profit, self.locations[firm_idx], 
                         bounds=bounds,
                         method='L-BFGS-B')
        
        if result.success:
            self.locations[firm_idx] = result.x
        else:
            print(f"Location optimization failed for firm {firm_idx}: {result.message}")
        
        return result.success
    
    def find_equilibrium(self, max_iterations=100, tolerance=1e-4, grid_size=30, 
                        update_method='sequential', verbose=True, optimize_locations=True):
        """
        Find Nash equilibrium by iterative best response.
        
        Parameters:
        max_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        grid_size (int): Grid size for numerical integration
        update_method (str): 'sequential' or 'simultaneous' updates
        verbose (bool): Whether to print progress
        optimize_locations (bool): Whether to optimize firm locations. If False, only prices are optimized.
        
        Returns:
        dict: {'converged': bool, 'iterations': int, 'time_elapsed': float}
        """
        start_time = time.time()
        self.price_history = [self.prices.copy()]
        self.location_history = [self.locations.copy()]
        self.profit_history = [self.total_profit(grid_size)]
        
        converged_status = False
        final_iteration = 0

        for iteration in range(max_iterations):
            final_iteration = iteration + 1
            old_prices = self.prices.copy()
            old_locations = self.locations.copy()
            
            if update_method == 'sequential':
                # Sequential updates
                for i in range(self.n_firms):
                    self.update_price(i, grid_size) # Always update price
                    if optimize_locations:
                        self.update_location(i, grid_size)
            elif update_method == 'simultaneous':
                # Simultaneous updates
                new_prices = np.zeros_like(self.prices)
                new_locations = self.locations.copy() # Start with current locations

                # Calculate new prices
                for i in range(self.n_firms):
                    old_price_sim = self.prices[i] # Store current price for this firm
                    self.update_price(i, grid_size) # Find best response price
                    new_prices[i] = self.prices[i] # Store the new best response price
                    self.prices[i] = old_price_sim # Restore original price for next firm's calculation
                
                if optimize_locations:
                    temp_new_locations = np.zeros_like(self.locations)
                    for i in range(self.n_firms):
                        # For location optimization, use the new_prices just calculated,
                        # as firms would anticipate these if prices and locations were set simultaneously.
                        # This requires temporarily setting self.prices to new_prices for the scope of location update.
                        original_prices_for_loc_opt = self.prices.copy()
                        self.prices = new_prices # Temporarily set new prices for location optimization

                        old_location_sim = self.locations[i].copy()
                        self.update_location(i, grid_size) # Find best response location given new prices
                        temp_new_locations[i] = self.locations[i].copy()
                        self.locations[i] = old_location_sim # Restore original location

                        self.prices = original_prices_for_loc_opt # Restore original prices array structure
                    new_locations = temp_new_locations
                
                # Update all at once
                self.prices = new_prices
                if optimize_locations:
                    self.locations = new_locations
            
            # Store history
            self.price_history.append(self.prices.copy())
            self.location_history.append(self.locations.copy()) # Store locations even if not optimized, to keep history consistent
            self.profit_history.append(self.total_profit(grid_size))
            
            # Check convergence
            price_change = np.max(np.abs(self.prices - old_prices))
            if optimize_locations:
                location_change = np.max(np.abs(self.locations - old_locations))
            else:
                location_change = 0.0 # No change if locations are not optimized
            
            if verbose:
                elapsed = time.time() - start_time
                log_msg = f"Iteration {final_iteration}/{max_iterations}: Price Δ = {price_change:.6f}"
                if optimize_locations:
                    log_msg += f", Location Δ = {location_change:.6f}"
                log_msg += f", Time = {elapsed:.2f}s"
                print(log_msg)
            
            converged_prices = price_change < tolerance
            converged_locations = True # Assume true if not optimizing locations
            if optimize_locations:
                converged_locations = location_change < tolerance
            
            if converged_prices and converged_locations:
                elapsed_time_at_convergence = time.time() - start_time # Capture time at convergence
                if verbose:
                    print(f"Converged after {final_iteration} iterations in {elapsed_time_at_convergence:.2f}s.")
                converged_status = True
                # Store final iteration and time before breaking
                self.last_run_iterations = final_iteration
                self.last_run_time = elapsed_time_at_convergence
                break # Exit the loop once converged
        
        # After the loop (either converged and broke, or max_iterations reached)
        if not converged_status: # If loop finished due to max_iterations
            elapsed_time_at_end = time.time() - start_time
            self.last_run_iterations = final_iteration # final_iteration will be max_iterations
            self.last_run_time = elapsed_time_at_end
            if verbose:
                print(f"Did not converge after {final_iteration} iterations in {elapsed_time_at_end:.2f}s.")
        
        return {'converged': converged_status, 'iterations': self.last_run_iterations, 'time_elapsed': self.last_run_time}

    def calculate_firm_profit_for_deviation(self, firm_idx, deviation_price=None, deviation_location=None, grid_size=30):
        """
        Calculates the profit for a specific firm if it deviates to a new price or location,
        without altering the model's current state.
        Other firms' prices and locations are taken from the model's current state.
        """
        # Store original state of the deviating firm
        _original_price_firm_idx = self.prices[firm_idx]
        _original_location_firm_idx = self.locations[firm_idx].copy()

        # Apply deviations temporarily to the specific firm
        if deviation_price is not None:
            self.prices[firm_idx] = deviation_price
        if deviation_location is not None:
            self.locations[firm_idx] = np.array(deviation_location)
        
        # Calculate profit. The self.profit method will use the current state of 
        # self.prices and self.locations, where only firm_idx has deviated.
        profit_with_deviation = self.profit(firm_idx, grid_size)

        # Restore the original state of the deviating firm to self.prices and self.locations
        self.prices[firm_idx] = _original_price_firm_idx
        self.locations[firm_idx] = _original_location_firm_idx
        
        return profit_with_deviation

    def calculate_profit_landscape(self, firm_idx, landscape_grid_size=15, profit_calc_grid_size=30):
        """
        Calculates the profit for a firm across a grid of possible locations,
        keeping its price and other firms' strategies fixed.
        """
        original_firm_location = self.locations[firm_idx].copy()
        
        x_coords = np.linspace(0, self.market_shape[0], landscape_grid_size)
        y_coords = np.linspace(0, self.market_shape[1], landscape_grid_size)
        
        profit_matrix = np.zeros((landscape_grid_size, landscape_grid_size))
        
        for r, ly in enumerate(y_coords): # Iterate rows (y-coordinates for imshow)
            for c, lx in enumerate(x_coords): # Iterate columns (x-coordinates for imshow)
                self.locations[firm_idx] = np.array([lx, ly])
                profit_matrix[r, c] = self.profit(firm_idx, profit_calc_grid_size)
        
        # Restore the original location of the firm
        self.locations[firm_idx] = original_firm_location
        
        return profit_matrix, x_coords, y_coords

    def max_transport_cost(self):
        """Calculate maximum possible transportation cost in the market."""
        if self.d_type == 'euclidean':
            return np.sqrt(self.market_shape[0]**2 + self.market_shape[1]**2)
        elif self.d_type == 'manhattan':
            return self.market_shape[0] + self.market_shape[1]
        elif self.d_type == 'quadratic':
            # Max possible squared distance in market
            return self.market_shape[0]**2 + self.market_shape[1]**2
        else:
            return np.sqrt(self.market_shape[0]**2 + self.market_shape[1]**2)
    
    def check_equilibrium_uniqueness(self, n_attempts=5, max_iterations=50, 
                                    tolerance=1e-4, grid_size=30, verbose=True):
        """
        Empirically check if the equilibrium is unique by starting from different random initial conditions.
        
        Returns:
        dict: Results of uniqueness check
        """
        equilibria = []
        
        original_locations = self.locations.copy()
        original_prices = self.prices.copy()
        
        for attempt in range(n_attempts):
            if verbose:
                print(f"\nAttempt {attempt+1}/{n_attempts} to find equilibrium:")
            
            # Randomize initial positions and prices
            self.locations = np.random.uniform(0, 1, (self.n_firms, 2)) * self.market_shape
            self.prices = np.random.uniform(self.c * 1.1, self.max_price * 0.9, self.n_firms)
            
            # Find equilibrium
            # The find_equilibrium method returns a dict, we need the 'converged' boolean
            sim_results = self.find_equilibrium(max_iterations, tolerance, grid_size, verbose=verbose)
            converged = sim_results['converged']
            
            if converged:
                equilibria.append({
                    'locations': self.locations.copy(),
                    'prices': self.prices.copy(),
                    'profits': self.total_profit(grid_size)
                })
        
        # Restore original state
        self.locations = original_locations
        self.prices = original_prices
        
        # Analyze uniqueness
        raw_converged_equilibria_count = len(equilibria) # Number of successfully converged attempts
        if raw_converged_equilibria_count == 0:
            return {"unique": False, "reason": "No equilibria found", "n_equilibria_found": 0, "max_price_diff": 0, "max_location_diff": 0, "equilibria": []}
        
        # Need to sort firms consistently for comparison
        sorted_equilibria = []
        for eq in equilibria:
            # Sort by x-coordinate as a simple heuristic
            indices = np.argsort(eq['locations'][:, 0])
            sorted_equilibria.append({
                'locations': eq['locations'][indices],
                'prices': eq['prices'][indices],
                'profits': eq['profits'][indices]
            })
        
        # Calculate overall max differences among all converged attempts
        max_price_diff_overall = 0
        max_location_diff_overall = 0
        if raw_converged_equilibria_count > 1:
            for i in range(raw_converged_equilibria_count):
                for j in range(i + 1, raw_converged_equilibria_count):
                    price_d = np.max(np.abs(sorted_equilibria[i]['prices'] - sorted_equilibria[j]['prices']))
                    location_d = np.max(np.abs(sorted_equilibria[i]['locations'] - sorted_equilibria[j]['locations']))
                    max_price_diff_overall = max(max_price_diff_overall, price_d)
                    max_location_diff_overall = max(max_location_diff_overall, location_d)

        # Identify distinct equilibrium clusters
        distinct_equilibria_clusters = []
        if raw_converged_equilibria_count > 0:
            distinct_equilibria_clusters.append(sorted_equilibria[0]) # First one is always a new cluster
            
            for eq_candidate in sorted_equilibria[1:]:
                is_new_cluster = True
                for cluster_rep in distinct_equilibria_clusters:
                    price_diff_to_rep = np.max(np.abs(eq_candidate['prices'] - cluster_rep['prices']))
                    location_diff_to_rep = np.max(np.abs(eq_candidate['locations'] - cluster_rep['locations']))
                    
                    if price_diff_to_rep <= tolerance * 10 and location_diff_to_rep <= tolerance * 10:
                        is_new_cluster = False
                        break # Belongs to this existing cluster
                
                if is_new_cluster:
                    distinct_equilibria_clusters.append(eq_candidate)
        
        num_distinct_clusters = len(distinct_equilibria_clusters)
        is_empirically_unique = (num_distinct_clusters <= 1) # Unique if 0 or 1 distinct cluster

        # If no equilibria converged, it's not unique in the sense of having AN equilibrium.
        # If multiple distinct clusters found, it's not unique.
        # If one distinct cluster found (even if from multiple attempts), it's unique.
        if raw_converged_equilibria_count == 0: # Should have been caught by the first check, but for safety
            is_empirically_unique = False 
            
        return {
            "unique": is_empirically_unique,
            "n_equilibria_found": num_distinct_clusters, # This now means number of *distinct* equilibria
            "max_price_diff": max_price_diff_overall,
            "max_location_diff": max_location_diff_overall,
            "equilibria": distinct_equilibria_clusters # List of distinct equilibrium representatives
        }
    
    def visualize(self, show_segmentation=True, show_density=False, grid_size=100):
        """
        Visualize the market solution.
        
        Parameters:
        show_segmentation (bool): Whether to show market segmentation
        show_density (bool): Whether to show population density
        grid_size (int): Grid size for visualization
        """
        if show_density and show_segmentation:
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        else:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # First subplot: Firm locations
        ax1 = axs[0]
        ax1.set_xlim(0, self.market_shape[0])
        ax1.set_ylim(0, self.market_shape[1])
        
        # Plot firm locations
        for i, (x, y) in enumerate(self.locations):
            ax1.scatter(x, y, s=100, label=f'Firm {i+1}: p={self.prices[i]:.2f}')
        
        ax1.set_title('Firm Locations and Prices')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend()
        
        # Current plot index
        plot_idx = 1
        
        # Second subplot: Market segmentation or density
        if show_segmentation:
            ax2 = axs[plot_idx]
            plot_idx += 1
            
            # Create grid
            x = np.linspace(0, self.market_shape[0], grid_size)
            y = np.linspace(0, self.market_shape[1], grid_size)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            # Calculate which firm has highest choice probability at each point
            for i in range(grid_size):
                for j in range(grid_size):
                    consumer_loc = np.array([X[i, j], Y[i, j]])
                    probs = self.choice_prob(consumer_loc)
                    Z[i, j] = np.argmax(probs)
            
            # Create custom colormap for better visualization
            colors = plt.cm.tab10(np.linspace(0, 1, self.n_firms))
            cmap = ListedColormap(colors[:self.n_firms])
            
            # Plot segmentation
            im = ax2.imshow(Z, extent=[0, self.market_shape[0], 0, self.market_shape[1]], 
                          origin='lower', aspect='auto', cmap=cmap)
            
            # Plot firm locations on top
            for i, (x, y) in enumerate(self.locations):
                ax2.scatter(x, y, s=100, c='white', edgecolor='black')
                ax2.text(x, y, str(i+1), ha='center', va='center')
            
            ax2.set_title('Market Segmentation (Pseudo-Voronoi Regions)')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
        
        # Population density
        if show_density:
            ax3 = axs[plot_idx]
            
            # Create grid
            x = np.linspace(0, self.market_shape[0], grid_size)
            y = np.linspace(0, self.market_shape[1], grid_size)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            # Calculate density at each point
            for i in range(grid_size):
                for j in range(grid_size):
                    Z[i, j] = self.rho(X[i, j], Y[i, j])
            
            # Plot density
            im = ax3.imshow(Z, extent=[0, self.market_shape[0], 0, self.market_shape[1]], 
                          origin='lower', aspect='auto', cmap='viridis')
            fig.colorbar(im, ax=ax3, label='Population Density')
            
            # Plot firm locations on top
            for i, (x, y) in enumerate(self.locations):
                ax3.scatter(x, y, s=100, c='red', edgecolor='black')
                ax3.text(x, y, str(i+1), ha='center', va='center', color='white')
            
            ax3.set_title('Population Density')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
        
        plt.tight_layout()
        return fig
    
    def plot_convergence(self):
        """Plot the convergence history of prices, locations, and profits."""
        if not self.price_history or not self.location_history or not self.profit_history:
            print("No history available. Run find_equilibrium() first.")
            # Return an empty figure or None if no history
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No history available.", ha='center', va='center')
            return fig
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10)) # Changed to 2x2 layout
        
        # Plot price convergence
        ax1 = axs[0,0]
        price_history = np.array(self.price_history)
        for i in range(self.n_firms):
            ax1.plot(price_history[:, i], label=f'Firm {i+1}')
        ax1.set_title('Price Convergence')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Price')
        ax1.legend()
        
        # Plot location convergence (Euclidean distance from final position)
        ax2 = axs[0,1]
        location_history_np = np.array(self.location_history) # Ensure it's a NumPy array
        if location_history_np.ndim == 3 and location_history_np.shape[0] > 0: # Check if history is not empty and has correct dimensions
            final_locations = location_history_np[-1]
            
            for i in range(self.n_firms):
                # Ensure firm_history and final_location_firm are 2D arrays for subtraction
                firm_history = location_history_np[:, i, :]
                final_location_firm = final_locations[i, :]
                distances = np.sqrt(np.sum((firm_history - final_location_firm)**2, axis=1))
                ax2.plot(distances, label=f'Firm {i+1}')
            
            ax2.set_title('Location Convergence (Distance from Final)')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Distance')
            ax2.set_yscale('log')  # Log scale to see small changes
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "Location history not available or in unexpected format.", ha='center', va='center')


        # Plot profit convergence
        ax3 = axs[1,0]
        profit_history_np = np.array(self.profit_history) # Ensure NumPy array
        if profit_history_np.ndim == 2 and profit_history_np.shape[0] > 0: # Check if history is not empty
            for i in range(self.n_firms):
                ax3.plot(profit_history_np[:, i], label=f'Firm {i+1}')
            ax3.set_title('Profit Convergence')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Profit')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, "Profit history not available or in unexpected format.", ha='center', va='center')

        # Plot location trajectories
        ax4 = axs[1,1]
        if location_history_np.ndim == 3 and location_history_np.shape[0] > 0:
            colors = plt.cm.viridis(np.linspace(0, 1, self.n_firms))
            for i in range(self.n_firms):
                ax4.plot(location_history_np[:, i, 0], location_history_np[:, i, 1], marker='o', markersize=2, linestyle='-', color=colors[i], label=f'Firm {i+1}')
                ax4.scatter(location_history_np[-1, i, 0], location_history_np[-1, i, 1], marker='X', s=100, color=colors[i], edgecolor='black', zorder=5) # Final position
            ax4.set_xlim(0, self.market_shape[0])
            ax4.set_ylim(0, self.market_shape[1])
            ax4.set_title('Location Trajectories')
            ax4.set_xlabel('X-coordinate')
            ax4.set_ylabel('Y-coordinate')
            ax4.legend(fontsize='small')
            ax4.set_aspect('equal', adjustable='box')
        else:
            ax4.text(0.5, 0.5, "Location history not available for trajectories.", ha='center', va='center')
        
        plt.tight_layout()
        return fig

    def generate_convergence_gif(self, filename="convergence.gif", fps=5, viz_grid_size_for_density_plot=20):
        """
        Generates a GIF of the convergence process showing firm locations and prices.

        Parameters:
        filename (str): The name of the file to save the GIF to.
        fps (int): Frames per second for the GIF.
        viz_grid_size_for_density_plot (int): Grid size for rendering density background.
        
        Returns:
        str: The filename of the generated GIF, or None if history is insufficient.
        """
        if not self.price_history or not self.location_history or len(self.location_history) < 2:
            print("Not enough history to generate GIF. Price or location history is too short.")
            return None

        frames = []
        # Determine overall market bounds for consistent plotting, adding a small margin
        margin_x = 0.1 * self.market_shape[0]
        margin_y = 0.1 * self.market_shape[1]
        market_min_x, market_max_x = 0 - margin_x, self.market_shape[0] + margin_x
        market_min_y, market_max_y = 0 - margin_y, self.market_shape[1] + margin_y

        density_Z = None
        if self.rho_type != 'uniform': # Pre-calculate density for background
            x_density = np.linspace(0, self.market_shape[0], viz_grid_size_for_density_plot)
            y_density = np.linspace(0, self.market_shape[1], viz_grid_size_for_density_plot)
            X_density, Y_density = np.meshgrid(x_density, y_density)
            density_Z = np.zeros_like(X_density)
            for i_row in range(viz_grid_size_for_density_plot):
                for j_col in range(viz_grid_size_for_density_plot):
                    density_Z[i_row, j_col] = self.rho(X_density[i_row, j_col], Y_density[i_row, j_col])
        
        num_frames = len(self.location_history)
        for iter_idx in range(num_frames):
            fig, ax = plt.subplots(figsize=(7, 6)) # Consistent figure size
            canvas = FigureCanvas(fig)

            current_locations = np.array(self.location_history[iter_idx])
            current_prices = np.array(self.price_history[iter_idx])

            if density_Z is not None:
                ax.imshow(density_Z, extent=[0, self.market_shape[0], 0, self.market_shape[1]],
                          origin='lower', aspect='auto', cmap='viridis', alpha=0.3, interpolation='bilinear')

            firm_colors = plt.cm.get_cmap('tab10', self.n_firms) # Get a distinct color for each firm
            for i in range(self.n_firms):
                ax.scatter(current_locations[i, 0], current_locations[i, 1], s=120, 
                           color=firm_colors(i), edgecolor='black', label=f'Firm {i+1}' if iter_idx == 0 else None,
                           zorder=5) # Ensure firms are on top
                ax.text(current_locations[i, 0], current_locations[i, 1] + 0.03 * self.market_shape[1], # Slightly offset text
                        f'P:{current_prices[i]:.2f}', ha='center', va='bottom', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none'), zorder=6)
            
            ax.set_xlim(market_min_x, market_max_x)
            ax.set_ylim(market_min_y, market_max_y)
            ax.set_title(f'Iteration {iter_idx+1}/{num_frames}', fontsize=10)
            ax.set_xlabel('x-coordinate', fontsize=9)
            ax.set_ylabel('y-coordinate', fontsize=9)
            if iter_idx == 0 and self.n_firms <= 10: # Add legend only on first frame for clarity
                ax.legend(fontsize='x-small', loc='best')
            ax.set_aspect('equal', adjustable='box') # Maintain aspect ratio
            plt.tight_layout()

            canvas.draw()
            # Get ARGB buffer and convert to NumPy array
            buf = canvas.tostring_argb()
            image_argb = np.frombuffer(buf, dtype='uint8')
            image_argb = image_argb.reshape(canvas.get_width_height()[::-1] + (4,))
            
            # Convert ARGB to RGB (discarding Alpha channel, taking R, G, B)
            image_rgb = image_argb[:, :, 1:] # ARGB -> RGB
            
            frames.append(image_rgb)
            plt.close(fig) # Close figure to free memory

        if frames:
            # Use duration in milliseconds per frame. loop=0 means infinite loop.
            imageio.mimsave(filename, frames, duration=(1000/fps), loop=0) 
            print(f"Convergence GIF saved as {filename}")
            return filename
        else:
            print("No frames generated for GIF.")
            return None

    # ========= MONOPOLY CASE METHODS =========

    def calculate_total_consumer_mass(self, grid_size=30):
        """Calculates the total consumer mass M = integral rho(s) ds."""
        x_coords = np.linspace(0, self.market_shape[0], grid_size)
        y_coords = np.linspace(0, self.market_shape[1], grid_size)
        dx = self.market_shape[0] / grid_size
        dy = self.market_shape[1] / grid_size
        
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        consumer_locs = np.vstack([X.ravel(), Y.ravel()]).T # (grid_size*grid_size, 2)
        
        rho_values = self.rho(consumer_locs[:, 0], consumer_locs[:, 1]) # (grid_size*grid_size,)
        total_mass = np.sum(rho_values) * dx * dy
        return total_mass

    def calculate_weighted_average_distance(self, firm_location_candidate, grid_size=30, total_mass=None):
        """
        Calculates d_bar_rho(x1) = (1/M) * integral rho(s) * d(s, x1) ds.
        firm_location_candidate: np.array, potential location of the monopolist (x,y).
        total_mass: Pre-calculated M. If None, it will be calculated.
        """
        if total_mass is None:
            total_mass = self.calculate_total_consumer_mass(grid_size)
        
        if total_mass == 0:
            return np.inf # Avoid division by zero if no consumers

        x_coords = np.linspace(0, self.market_shape[0], grid_size)
        y_coords = np.linspace(0, self.market_shape[1], grid_size)
        dx = self.market_shape[0] / grid_size
        dy = self.market_shape[1] / grid_size
        
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        consumer_locs = np.vstack([X.ravel(), Y.ravel()]).T # (grid_size*grid_size, 2)
        
        rho_values = self.rho(consumer_locs[:, 0], consumer_locs[:, 1]) # (grid_size*grid_size,)
        distances = self.d(consumer_locs, firm_location_candidate) # (grid_size*grid_size,)
        
        integral_rho_d = np.sum(rho_values * distances) * dx * dy
        
        return integral_rho_d / total_mass

    def find_optimal_monopoly_location(self, grid_size=30, initial_guess=None):
        """
        Finds the location x1* that minimizes the weighted average distance to consumers.
        Returns: optimal_location (np.array), min_weighted_avg_dist (float)
        """
        total_mass = self.calculate_total_consumer_mass(grid_size)
        if total_mass == 0:
            # If no consumers, any location is "optimal" with zero profit.
            # Return center as a sensible default, distance is undefined or infinite.
            return np.array([self.market_shape[0]/2, self.market_shape[1]/2]), np.inf

        def objective_function(location_candidate):
            return self.calculate_weighted_average_distance(location_candidate, grid_size, total_mass)

        bounds = [(0, self.market_shape[0]), (0, self.market_shape[1])]
        if initial_guess is None:
            # Start search from the center of the market
            initial_guess = np.array([self.market_shape[0] / 2, self.market_shape[1] / 2])
        
        result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimal_location = result.x
            min_weighted_avg_dist = result.fun
            return optimal_location, min_weighted_avg_dist
        else:
            # Fallback or error handling
            print(f"Monopoly location optimization failed: {result.message}. Using initial guess.")
            return initial_guess, objective_function(initial_guess)

    def calculate_monopoly_price(self, optimal_weighted_avg_dist):
        """
        Calculates the monopoly price p1* = alpha/(2*gamma) + c/2 - t*d_bar_rho(x1*)/2.
        Assumes optimal_weighted_avg_dist (d_bar_rho(x1*)) is provided.
        """
        if self.gamma_demand == 0:
            # Avoid division by zero; implies perfectly inelastic demand, price is theoretically infinite or max_price.
            # This case might need special handling based on model interpretation.
            return self.max_price 
            
        price = (self.alpha_demand / (2 * self.gamma_demand)) + \
                (self.c / 2) - \
                (self.t_transport_cost * optimal_weighted_avg_dist / 2)
        
        # Ensure price is at least marginal cost and not exceeding max_price (if defined)
        # The paper's condition alpha - gamma*c - gamma*t*d_bar >= 0 ensures p1* >= c.
        # We should also respect self.max_price as an upper bound.
        price = np.clip(price, self.c, self.max_price)
        return price

    def calculate_monopoly_profit_formula(self, optimal_weighted_avg_dist, total_mass):
        """
        Calculates monopoly profit using the formula:
        Pi(x1*) = (M / (4*gamma)) * (alpha - gamma*c - gamma*t*d_bar_rho(x1*))^2
        """
        if self.gamma_demand == 0:
            return 0 # Or handle as infinite if price is infinite and demand is fixed positive

        term_in_parentheses = self.alpha_demand - self.gamma_demand * self.c - \
                              self.gamma_demand * self.t_transport_cost * optimal_weighted_avg_dist
        
        # Profit is non-negative if term_in_parentheses is real.
        # If term_in_parentheses < 0, it implies demand is negative even at cost c,
        # which means profit should be 0 (monopolist wouldn't operate).
        if term_in_parentheses < 0:
            return 0.0
            
        profit = (total_mass / (4 * self.gamma_demand)) * (term_in_parentheses**2)
        return profit

    def solve_monopoly_case(self, grid_size=30, initial_location_guess=None):
        """
        Solves for the optimal monopoly location, price, and profit.
        Returns a dictionary with results.
        """
        if self.n_firms != 1:
            # This method is intended for the n_firms=1 case.
            # For simulation purposes, we can still calculate what a monopolist *would* do.
            print("Warning: solve_monopoly_case called with n_firms != 1. Calculations proceed as if for a monopolist.")

        total_mass_M = self.calculate_total_consumer_mass(grid_size)
        
        optimal_loc_x1_star, d_bar_rho_at_x1_star = self.find_optimal_monopoly_location(
            grid_size, initial_guess=initial_location_guess
        )
        
        monopoly_price_p1_star = self.calculate_monopoly_price(d_bar_rho_at_x1_star)
        
        # Verify the condition for p1* >= c and positive demand from paper
        # alpha - gamma*c - gamma*t*d_bar_rho(x1*) >= 0
        profit_condition_term = self.alpha_demand - self.gamma_demand * self.c - \
                                self.gamma_demand * self.t_transport_cost * d_bar_rho_at_x1_star
        
        valid_solution = True
        if profit_condition_term < 0:
            print("Warning: Monopoly price condition (alpha - gc - gtd_bar >= 0) not met. "
                  "Optimal price might be 'c' and profit zero if demand is non-positive.")
            # If condition not met, price is 'c' (if demand still positive) or monopolist doesn't operate (profit 0).
            # The calculate_monopoly_price clips at 'c'.
            # The calculate_monopoly_profit_formula handles this by returning 0 profit.
            valid_solution = False # Indicates parameters might lead to non-operation or corner solution.

        monopoly_profit_pi1_star = self.calculate_monopoly_profit_formula(
            d_bar_rho_at_x1_star, total_mass_M
        )
        
        # For direct simulation with this monopolist (e.g. visualization)
        # we can update the model's state if n_firms is indeed 1.
        if self.n_firms == 1:
            self.locations[0] = optimal_loc_x1_star
            self.prices[0] = monopoly_price_p1_star

        return {
            "optimal_location": optimal_loc_x1_star,
            "optimal_price": monopoly_price_p1_star,
            "maximized_profit": monopoly_profit_pi1_star,
            "weighted_avg_distance_at_opt_loc": d_bar_rho_at_x1_star,
            "total_consumer_mass": total_mass_M,
            "profit_condition_term": profit_condition_term, # For diagnostics
            "solution_validity_check": valid_solution # True if profit_condition_term >=0
        }

    def get_weighted_average_distance_landscape(self, landscape_grid_size=15, calculation_grid_size=30):
        """
        Calculates the weighted average distance d_bar_rho(x) for a firm
        across a grid of possible locations x.
        """
        total_mass = self.calculate_total_consumer_mass(calculation_grid_size)
        if total_mass == 0:
            # Return a grid of NaNs or Infs if no consumers
            return np.full((landscape_grid_size, landscape_grid_size), np.nan), \
                   np.linspace(0, self.market_shape[0], landscape_grid_size), \
                   np.linspace(0, self.market_shape[1], landscape_grid_size)

        x_coords = np.linspace(0, self.market_shape[0], landscape_grid_size)
        y_coords = np.linspace(0, self.market_shape[1], landscape_grid_size)
        
        d_bar_matrix = np.zeros((landscape_grid_size, landscape_grid_size))
        
        for r, ly in enumerate(y_coords): # Iterate rows (y-coordinates for imshow)
            for c, lx in enumerate(x_coords): # Iterate columns (x-coordinates for imshow)
                candidate_loc = np.array([lx, ly])
                d_bar_matrix[r, c] = self.calculate_weighted_average_distance(
                    candidate_loc, calculation_grid_size, total_mass
                )
        
        return d_bar_matrix, x_coords, y_coords

    # ========= DYNAMIC ENTRY SIMULATION METHODS =========

    def _get_expected_profit_for_entrant(self, 
                                         candidate_entry_location, 
                                         existing_locations, 
                                         existing_prices, 
                                         grid_size_profit_calc):
        """
        Calculates the profit an entrant would expect at candidate_entry_location.
        The entrant optimizes its own price, given fixed prices/locations of existing firms.
        Returns: (expected_profit, optimal_entrant_price)
        """
        num_existing_firms = existing_locations.shape[0]
        entrant_idx = num_existing_firms # Index of the new entrant

        # Create a temporary model state for N+1 firms
        temp_locations = np.vstack([existing_locations, np.array(candidate_entry_location).reshape(1,2)])
        
        # Initial price guess for the entrant
        entrant_price_guess = self.c * 1.5 
        temp_prices = np.append(existing_prices, entrant_price_guess)

        # Store current model state to restore later
        original_n_firms, original_locations, original_prices_state = self.n_firms, self.locations.copy(), self.prices.copy()
        
        # Temporarily set the model to the N+1 firm scenario
        self._initialize_firms(num_existing_firms + 1, temp_locations, temp_prices)
        
        # Entrant optimizes its own price, given other firms' prices are fixed
        # The update_price method modifies self.prices[entrant_idx]
        self.update_price(entrant_idx, grid_size_profit_calc) 
        
        optimal_entrant_price = self.prices[entrant_idx]
        expected_profit = self.profit(entrant_idx, grid_size_profit_calc)
        
        # Restore original model state
        self._initialize_firms(original_n_firms, original_locations, original_prices_state)
        
        return expected_profit, optimal_entrant_price

    def _find_optimal_entry_location_and_profit(self, 
                                                existing_locations_np, 
                                                existing_prices_np, 
                                                grid_size_loc_opt, # Grid for optimizing entry location
                                                grid_size_profit_calc): # Grid for profit calculation
        """
        Finds the optimal entry location for a new firm, its expected profit,
        and its optimal price at that location (given fixed incumbents).
        Uses a grid search for location optimization.
        Returns: (best_location, max_expected_profit, optimal_price_at_best_location)
        """
        best_location = None
        max_expected_profit = -np.inf 
        optimal_price_at_best_location = self.c * 1.5 # Default

        # Define a grid of potential entry locations
        loc_x_coords = np.linspace(0, self.market_shape[0], grid_size_loc_opt)
        loc_y_coords = np.linspace(0, self.market_shape[1], grid_size_loc_opt)

        for lx_candidate in loc_x_coords:
            for ly_candidate in loc_y_coords:
                candidate_loc = np.array([lx_candidate, ly_candidate])
                
                # Check if candidate_loc is too close to an existing firm (optional, to avoid trivial solutions)
                # For now, we allow any location.
                
                expected_profit, entrant_optimal_price = self._get_expected_profit_for_entrant(
                    candidate_loc, existing_locations_np, existing_prices_np,
                    grid_size_profit_calc
                )
                
                if expected_profit > max_expected_profit:
                    max_expected_profit = expected_profit
                    best_location = candidate_loc
                    optimal_price_at_best_location = entrant_optimal_price
        
        # If no profitable location found (e.g., all profits are negative or zero)
        if best_location is None and grid_size_loc_opt > 0: 
             best_location = np.array([self.market_shape[0]/2, self.market_shape[1]/2]) # Fallback location
             max_expected_profit, optimal_price_at_best_location = self._get_expected_profit_for_entrant(
                    best_location, existing_locations_np, existing_prices_np,
                    grid_size_profit_calc
                )

        return best_location, max_expected_profit, optimal_price_at_best_location

    def run_dynamic_entry_simulation(self, 
                                     entry_cost_F, 
                                     discount_factor_delta, 
                                     max_total_firms,
                                     grid_size_loc_opt, # For entrant's location choice
                                     grid_size_price_eq, # For price equilibrium calculation
                                     price_eq_max_iter, 
                                     price_eq_tolerance,
                                     update_method_price_eq='sequential',
                                     verbose_dynamic=True):
        """
        Runs the dynamic entry simulation.
        Firms enter sequentially if profitable, then all firms compete on prices.
        """
        self.dynamic_simulation_history = []
        
        # Initialize with 0 firms in the market for the purpose of the loop
        # The actual model's n_firms, locations, prices will be updated step-by-step
        current_locations_market = np.empty((0,2))
        current_prices_market = np.empty((0,))
        current_n_firms_market = 0

        for period_num in range(1, max_total_firms * 2 + 1): # Max periods: entry + price for each potential firm
            
            # --- Odd Period: Entry Stage ---
            if period_num % 2 == 1:
                potential_entrant_idx = current_n_firms_market + 1
                if verbose_dynamic:
                    print(f"\n--- Period {period_num} (Entry Stage for Firm {potential_entrant_idx}) ---")

                if current_n_firms_market == 0: # First entrant (Monopolist)
                    # Use solve_monopoly_case for the first entrant for precision
                    # Temporarily set n_firms=1 for solve_monopoly_case to work correctly
                    original_n_firms_temp, original_loc_temp, original_prices_temp = self.n_firms, self.locations.copy(), self.prices.copy()
                    self._initialize_firms(1) # Set up for monopoly calculation
                    
                    mono_results = self.solve_monopoly_case(grid_size=grid_size_price_eq) # Use price_eq grid for consistency
                    optimal_entry_location = mono_results["optimal_location"]
                    expected_profit_for_entrant = mono_results["maximized_profit"]
                    entrant_initial_price_guess = mono_results["optimal_price"] # Price from monopoly solution
                    
                    # Restore original model state before potentially adding the firm
                    self._initialize_firms(original_n_firms_temp, original_loc_temp, original_prices_temp)

                else: # Subsequent entrants
                    optimal_entry_location, expected_profit_for_entrant, entrant_initial_price_guess = \
                        self._find_optimal_entry_location_and_profit(
                            current_locations_market, current_prices_market,
                            grid_size_loc_opt, grid_size_price_eq
                        )

                # Entry condition: Present value of profits > Entry Cost
                present_value_profit = expected_profit_for_entrant / (1 - discount_factor_delta) if (1 - discount_factor_delta) > 1e-9 else float('inf')

                if verbose_dynamic:
                    print(f"Potential Entrant {potential_entrant_idx}: Optimal entry at {optimal_entry_location}, Expected Profit/period = {expected_profit_for_entrant:.3f}")
                    print(f"Present Value of Profit = {present_value_profit:.3f}, Entry Cost F = {entry_cost_F:.3f}")

                if present_value_profit > entry_cost_F and current_n_firms_market < max_total_firms :
                    if verbose_dynamic:
                        print(f"Firm {potential_entrant_idx} ENTERS the market.")
                    
                    current_locations_market = np.vstack([current_locations_market, optimal_entry_location.reshape(1,2)])
                    # The entrant_initial_price_guess is now determined by _find_optimal_entry_location_and_profit
                    # or by the monopoly solution for the first entrant.
                    current_prices_market = np.append(current_prices_market, entrant_initial_price_guess)
                    current_n_firms_market += 1
                    
                    self.dynamic_simulation_history.append({
                        'period': period_num, 'type': 'entry', 
                        'n_firms': current_n_firms_market, 
                        'locations': current_locations_market.copy(), 
                        'prices': current_prices_market.copy(), # Prices before general re-equilibration
                        'message': f"Firm {current_n_firms_market} entered at {optimal_entry_location} with initial price {entrant_initial_price_guess:.2f}. Expected profit/period: {expected_profit_for_entrant:.2f}."
                    })
                else:
                    if verbose_dynamic:
                        print(f"Firm {potential_entrant_idx} DOES NOT ENTER. (PV Profit <= F or max firms reached)")
                    self.dynamic_simulation_history.append({
                        'period': period_num, 'type': 'no_entry', 
                        'n_firms': current_n_firms_market,
                        'locations': current_locations_market.copy(), 
                        'prices': current_prices_market.copy(),
                        'message': "No new firm enters. Entry process ends."
                    })
                    break # End simulation if no new firm enters

            # --- Even Period: Price Competition Stage ---
            elif period_num % 2 == 0 and current_n_firms_market > 0:
                if verbose_dynamic:
                    print(f"\n--- Period {period_num} (Price Competition with {current_n_firms_market} firms) ---")

                # Set up the model for the current market structure
                self._initialize_firms(current_n_firms_market, current_locations_market, current_prices_market)
                
                # Find price equilibrium
                eq_results = self.find_equilibrium(
                    max_iterations=price_eq_max_iter,
                    tolerance=price_eq_tolerance,
                    grid_size=grid_size_price_eq,
                    update_method=update_method_price_eq,
                    verbose=False, # Can be set to True for detailed price convergence logs
                    optimize_locations=False # Locations are fixed in price competition stage
                )
                
                current_prices_market = self.prices.copy() # Update market prices to equilibrium prices
                current_profits_market = self.total_profit(grid_size=grid_size_price_eq)

                if verbose_dynamic:
                    print(f"Price equilibrium reached: Converged={eq_results['converged']}, Iterations={eq_results['iterations']}")
                    for i in range(current_n_firms_market):
                        print(f"  Firm {i+1}: Loc={current_locations_market[i]}, Price={current_prices_market[i]:.3f}, Profit={current_profits_market[i]:.3f}")
                
                self.dynamic_simulation_history.append({
                    'period': period_num, 'type': 'price_equilibrium', 
                    'n_firms': current_n_firms_market, 
                    'locations': current_locations_market.copy(), 
                    'prices': current_prices_market.copy(),
                    'profits': current_profits_market.copy(),
                    'converged': eq_results['converged'],
                    'message': f"Price equilibrium calculated for {current_n_firms_market} firms."
                })
                
                # Check for firm exit (if any firm's profit is negative)
                # As per user clarification, firms do not exit once entered, even if profits are negative.
                # So, this part is commented out unless specified otherwise.
                # if np.any(current_profits_market < 0):
                #     if verbose_dynamic: print("One or more firms have negative profits. (Exit logic not yet implemented)")
                #     pass # Implement exit logic if needed

        if verbose_dynamic:
            print("\nDynamic entry simulation finished.")
        return self.dynamic_simulation_history

    def generate_dynamic_simulation_gif(self, dynamic_history, filename="dynamic_entry.gif", fps=1, 
                                        viz_grid_size_for_density_plot=20, market_plot_grid_size=50):
        """
        Generates a GIF of the dynamic entry simulation.
        Each frame represents a period from the dynamic_history.
        """
        if not dynamic_history:
            print("No dynamic history to generate GIF.")
            return None

        frames = []
        margin_x = 0.1 * self.market_shape[0]
        margin_y = 0.1 * self.market_shape[1]
        market_min_x, market_max_x = 0 - margin_x, self.market_shape[0] + margin_x
        market_min_y, market_max_y = 0 - margin_y, self.market_shape[1] + margin_y

        # Pre-calculate density for background if not uniform (using original model's rho_type)
        density_Z_background = None
        if self.rho_type != 'uniform':
            x_density_bg = np.linspace(0, self.market_shape[0], viz_grid_size_for_density_plot)
            y_density_bg = np.linspace(0, self.market_shape[1], viz_grid_size_for_density_plot)
            X_density_bg, Y_density_bg = np.meshgrid(x_density_bg, y_density_bg)
            density_Z_background = np.zeros_like(X_density_bg)
            for r_idx in range(viz_grid_size_for_density_plot):
                for c_idx in range(viz_grid_size_for_density_plot):
                    density_Z_background[r_idx, c_idx] = self.rho(X_density_bg[r_idx, c_idx], Y_density_bg[c_idx, r_idx]) # Correct indexing for imshow

        num_history_frames = len(dynamic_history)
        for frame_idx, history_item in enumerate(dynamic_history):
            fig, ax = plt.subplots(figsize=(8, 7)) # Consistent figure size
            canvas = FigureCanvas(fig)

            current_locations_frame = history_item['locations']
            current_prices_frame = history_item['prices']
            n_firms_frame = history_item['n_firms']
            period_type = history_item['type']
            period_num_frame = history_item['period']
            message = history_item.get('message', '')

            # Background density
            if density_Z_background is not None:
                ax.imshow(density_Z_background.T, extent=[0, self.market_shape[0], 0, self.market_shape[1]],
                          origin='lower', aspect='auto', cmap='viridis', alpha=0.2, interpolation='bilinear')

            # Market segmentation for this state (if firms exist)
            if n_firms_frame > 0:
                # Temporarily set model to this state to use its choice_prob
                _orig_n, _orig_l, _orig_p = self.n_firms, self.locations.copy(), self.prices.copy()
                self._initialize_firms(n_firms_frame, current_locations_frame, current_prices_frame)

                x_seg = np.linspace(0, self.market_shape[0], market_plot_grid_size)
                y_seg = np.linspace(0, self.market_shape[1], market_plot_grid_size)
                X_seg, Y_seg = np.meshgrid(x_seg, y_seg)
                Z_seg_val = np.zeros_like(X_seg, dtype=int)
                for r in range(market_plot_grid_size):
                    for c_val in range(market_plot_grid_size):
                        consumer_loc_seg = np.array([X_seg[r, c_val], Y_seg[r, c_val]])
                        probs_seg = self.choice_prob(consumer_loc_seg)
                        if probs_seg.size > 0: Z_seg_val[r, c_val] = np.argmax(probs_seg)
                        else: Z_seg_val[r,c_val] = -1 # No firms or error

                seg_colors = plt.cm.get_cmap('tab10', max(1,n_firms_frame)) # Ensure at least 1 color
                seg_cmap = ListedColormap(seg_colors(np.linspace(0, 1, max(1,n_firms_frame))))
                ax.imshow(Z_seg_val.T, extent=[0, self.market_shape[0], 0, self.market_shape[1]],
                          origin='lower', aspect='auto', cmap=seg_cmap, alpha=0.5)
                
                self._initialize_firms(_orig_n, _orig_l, _orig_p) # Restore model

            # Plot firms
            firm_plot_colors = plt.cm.get_cmap('tab10', max(1,n_firms_frame))
            for i in range(n_firms_frame):
                ax.scatter(current_locations_frame[i, 0], current_locations_frame[i, 1], s=120, 
                           color=firm_plot_colors(i), edgecolor='black', zorder=5)
                ax.text(current_locations_frame[i, 0], current_locations_frame[i, 1] + 0.03 * self.market_shape[1],
                        f'F{i+1}\nP:{current_prices_frame[i]:.2f}', ha='center', va='bottom', fontsize=7, 
                        bbox=dict(facecolor='white', alpha=0.6, pad=1, edgecolor='none'), zorder=6)
            
            ax.set_xlim(market_min_x, market_max_x)
            ax.set_ylim(market_min_y, market_max_y)
            title_str = f"Period {period_num_frame}: {period_type.replace('_',' ').title()} ({n_firms_frame} Firms)"
            if period_type == 'no_entry': title_str = f"Period {period_num_frame}: No New Entry ({n_firms_frame} Firms)"
            ax.set_title(title_str, fontsize=10)
            ax.set_xlabel('x-coordinate', fontsize=9); ax.set_ylabel('y-coordinate', fontsize=9)
            ax.set_aspect('equal', adjustable='box')
            
            # Add message as text on plot
            fig.text(0.5, 0.01, message, ha='center', va='bottom', fontsize=8, wrap=True)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for figtext

            canvas.draw()
            image_argb = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
            image_argb = image_argb.reshape(canvas.get_width_height()[::-1] + (4,))
            frames.append(image_argb[:, :, 1:]) # ARGB -> RGB
            plt.close(fig)

        if frames:
            imageio.mimsave(filename, frames, duration=(1000/fps), loop=0)
            print(f"Dynamic entry GIF saved as {filename}")
            return filename
        else:
            print("No frames generated for dynamic entry GIF.")
            return None
