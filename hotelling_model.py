import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from matplotlib.colors import ListedColormap

class HotellingTwoDimensional:
    def __init__(self, 
                 n_firms=2, 
                 market_shape=(1, 1), 
                 beta=2.0, 
                 eta=2.0, 
                 A=1.0, 
                 c=1.0, 
                 max_price=10.0,
                 mu=1.0,
                 d_type='euclidean',
                 rho_type='uniform',
                 density_params=None):
        """
        Initialize the 2D Hotelling model with parametrized functions.
        
        Parameters:
        n_firms (int): Number of firms
        market_shape (tuple): Size of the market in x and y dimensions
        beta (float): Price sensitivity in logit model
        eta (float): Elasticity of demand (should be > 1)
        A (float): Scaling factor for demand
        c (float): Marginal cost
        max_price (float): Upper bound for prices
        mu (float): Strong convexity parameter for transport cost
        d_type (str): Type of distance function ('euclidean', 'manhattan', 'quadratic')
        rho_type (str): Type of density function ('uniform', 'linear', 'gaussian', 'sine', 'multi_gaussian')
        density_params (list): Optional. For 'multi_gaussian', a list of dicts,
                               where each dict defines a Gaussian focus:
                               [{'center': (cx, cy), 'strength': s, 'sigma': sig}, ...]
                               `center` is (x,y) coordinates.
                               `strength` is the amplitude of the Gaussian.
                               `sigma` is the standard deviation (spread).
        """
        self.n_firms = n_firms
        self.market_shape = market_shape
        self.beta = beta
        self.eta = eta
        self.A = A
        self.c = c
        self.max_price = max_price
        self.mu = mu
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
        """Transportation cost function based on specified type."""
        if self.d_type == 'euclidean':
            return np.sqrt(np.sum((consumer_loc - firm_loc)**2))
        elif self.d_type == 'manhattan':
            return np.sum(np.abs(consumer_loc - firm_loc))
        elif self.d_type == 'quadratic':
            # Quadratic distance (strongly convex with μ=2)
            return np.sum((consumer_loc - firm_loc)**2)
        else:
            return np.sqrt(np.sum((consumer_loc - firm_loc)**2))  # Default euclidean
    
    def effective_price(self, consumer_loc, firm_idx):
        """Calculate effective price for a consumer at given location."""
        return self.prices[firm_idx] * (1 + self.d(consumer_loc, self.locations[firm_idx]))
    
    def choice_prob(self, consumer_loc):
        """Calculate the logit choice probabilities for a consumer."""
        effective_prices = np.array([self.effective_price(consumer_loc, i) 
                                    for i in range(self.n_firms)])
        logits = -self.beta * effective_prices
        max_logit = np.max(logits)  # For numerical stability
        exp_logits = np.exp(logits - max_logit)
        return exp_logits / np.sum(exp_logits)
    
    def demand_at_location(self, consumer_loc, firm_idx):
        """Calculate demand from a single location."""
        eff_price = self.effective_price(consumer_loc, firm_idx)
        price_elasticity = self.A * (eff_price ** (-self.eta))
        choice_probability = self.choice_prob(consumer_loc)[firm_idx]
        return price_elasticity * choice_probability * self.rho(*consumer_loc)
    
    def total_demand_grid(self, firm_idx, grid_size=30):
        """Calculate total demand using grid-based numerical integration."""
        x = np.linspace(0, self.market_shape[0], grid_size)
        y = np.linspace(0, self.market_shape[1], grid_size)
        dx = self.market_shape[0] / grid_size
        dy = self.market_shape[1] / grid_size
        
        total = 0
        for i in range(grid_size):
            for j in range(grid_size):
                consumer_loc = np.array([x[i], y[j]])
                total += self.demand_at_location(consumer_loc, firm_idx) * dx * dy
        
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
                        update_method='sequential', verbose=True):
        """
        Find Nash equilibrium by iterative best response.
        
        Parameters:
        max_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        grid_size (int): Grid size for numerical integration
        update_method (str): 'sequential' or 'simultaneous' updates
        verbose (bool): Whether to print progress
        
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
                # Sequential updates (price then location for each firm)
                for i in range(self.n_firms):
                    self.update_price(i, grid_size)
                    self.update_location(i, grid_size)
            elif update_method == 'simultaneous':
                # Simultaneous updates for all firms
                new_prices = np.zeros_like(self.prices)
                new_locations = np.zeros_like(self.locations)
                
                # Calculate new prices and locations
                for i in range(self.n_firms):
                    # Save old price and find best response
                    old_price = self.prices[i]
                    self.update_price(i, grid_size)
                    new_prices[i] = self.prices[i]
                    self.prices[i] = old_price  # Restore
                    
                    # Save old location and find best response
                    old_location = self.locations[i].copy()
                    self.update_location(i, grid_size)
                    new_locations[i] = self.locations[i].copy()
                    self.locations[i] = old_location  # Restore
                
                # Update all at once
                self.prices = new_prices
                self.locations = new_locations
            
            # Store history
            self.price_history.append(self.prices.copy())
            self.location_history.append(self.locations.copy())
            self.profit_history.append(self.total_profit(grid_size))
            
            # Check convergence
            price_change = np.max(np.abs(self.prices - old_prices))
            location_change = np.max(np.abs(self.locations - old_locations))
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"Iteration {iteration+1}/{max_iterations}: "
                     f"Price Δ = {price_change:.6f}, Location Δ = {location_change:.6f}, "
                     f"Time = {elapsed:.2f}s")
            
            if price_change < tolerance and location_change < tolerance:
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
    
    def verify_assumptions(self, grid_size=20):
        """
        Verify if the assumptions in the paper are satisfied.
        
        Returns:
        dict: Results of assumption verification
        """
        results = {
            # Assumption 1: Regularity
            "reg_cost_continuous": True,  # Our cost functions are continuous by design
            "reg_cost_convex": self.d_type in ['euclidean', 'manhattan', 'quadratic'],
            "reg_compact": True,  # Our strategy spaces are compact by design
            "reg_elasticity": self.eta > 1,  # Check elasticity condition
            
            # Assumption 2: Uniqueness
            "uniq_strongly_convex": self.d_type == 'quadratic',  # Only quadratic is strongly convex
            "uniq_beta": self.beta * self.max_transport_cost() < 1,  # Check beta condition
        }
        
        # Calculate ω_ij values for the uniqueness elasticity condition
        omega = np.zeros((self.n_firms, self.n_firms))
        
        x = np.linspace(0, self.market_shape[0], grid_size)
        y = np.linspace(0, self.market_shape[1], grid_size)
        dx = self.market_shape[0] / grid_size
        dy = self.market_shape[1] / grid_size
        
        for i in range(self.n_firms):
            for j in range(self.n_firms):
                for ix in range(grid_size):
                    for iy in range(grid_size):
                        consumer_loc = np.array([x[ix], y[iy]])
                        probs = self.choice_prob(consumer_loc)
                        omega[i, j] += self.rho(x[ix], y[iy]) * probs[i] * probs[j] * dx * dy
        
        # Calculate the ratio term in the elasticity condition
        max_ratio = 0
        for i in range(self.n_firms):
            sum_others = np.sum(omega[i, :]) - omega[i, i]
            if omega[i, i] > 0:
                ratio = sum_others / omega[i, i]
                max_ratio = max(max_ratio, ratio)
        
        # Calculate the minimum required elasticity
        min_eta_required = 1 + self.beta * max_ratio + (self.beta**2 / self.mu)
        results["omega_matrix"] = omega
        results["max_omega_ratio"] = max_ratio
        results["min_eta_required"] = min_eta_required
        results["uniq_elasticity"] = self.eta > min_eta_required
        
        # Check overall if both assumptions are satisfied
        results["assumption1_satisfied"] = (results["reg_cost_continuous"] and 
                                           results["reg_cost_convex"] and 
                                           results["reg_compact"] and 
                                           results["reg_elasticity"])
        
        results["assumption2_satisfied"] = (results["uniq_strongly_convex"] and 
                                          results["uniq_elasticity"] and 
                                          results["uniq_beta"])
        
        return results
    
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
            converged = self.find_equilibrium(max_iterations, tolerance, grid_size, verbose=verbose)
            
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
        n_equilibria = len(equilibria)
        if n_equilibria == 0:
            return {"unique": False, "reason": "No equilibria found"}
        
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
        
        # Compare all equilibria pairwise
        unique = True
        max_price_diff = 0
        max_location_diff = 0
        
        for i in range(n_equilibria):
            for j in range(i+1, n_equilibria):
                # Calculate differences
                price_diff = np.max(np.abs(sorted_equilibria[i]['prices'] - sorted_equilibria[j]['prices']))
                location_diff = np.max(np.abs(sorted_equilibria[i]['locations'] - sorted_equilibria[j]['locations']))
                
                max_price_diff = max(max_price_diff, price_diff)
                max_location_diff = max(max_location_diff, location_diff)
                
                # Check if differences exceed tolerance
                if price_diff > tolerance * 10 or location_diff > tolerance * 10:
                    unique = False
        
        return {
            "unique": unique,
            "n_equilibria_found": n_equilibria,
            "max_price_diff": max_price_diff,
            "max_location_diff": max_location_diff,
            "equilibria": sorted_equilibria
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

# ========= SIMULATION TEST FUNCTIONS =========

def run_baseline_simulation():
    """Run a baseline simulation with parameters that satisfy assumptions"""
    print("\n=== BASELINE SIMULATION ===")
    # Create model with parameters satisfying assumptions
    model = HotellingTwoDimensional(
        n_firms=2,
        market_shape=(1, 1),
        beta=0.5,  # Low beta to ensure beta*d_bar < 1
        eta=3.0,   # High eta to ensure uniqueness
        A=1.0,
        c=1.0,
        max_price=5.0,
        mu=2.0,    # Quadratic distance has mu=2
        d_type='quadratic',
        rho_type='uniform'
    )
    
    # Check if assumptions are satisfied
    assumptions = model.verify_assumptions()
    print("\nAssumption Verification:")
    for key, value in assumptions.items():
        if key not in ['omega_matrix', 'equilibria']:
            print(f"{key}: {value}")
    
    # Find equilibrium
    print("\nFinding equilibrium...")
    model.find_equilibrium(max_iterations=50, tolerance=1e-4, verbose=True)
    
    # Visualize results
    print("\nFinal locations and prices:")
    for i in range(model.n_firms):
        print(f"Firm {i+1}: Location = {model.locations[i]}, Price = {model.prices[i]:.4f}")
    
    model.visualize(show_segmentation=True, show_density=True)
    model.plot_convergence()
    
    return model

def test_equilibrium_uniqueness():
    """Test if the equilibrium is unique when assumptions are satisfied"""
    print("\n=== TESTING EQUILIBRIUM UNIQUENESS ===")
    
    # Create model with parameters satisfying assumptions
    model = HotellingTwoDimensional(
        n_firms=2,
        market_shape=(1, 1),
        beta=0.5,
        eta=3.0,
        d_type='quadratic',
        mu=2.0,
        rho_type='uniform'
    )
    
    # Verify assumptions
    assumptions = model.verify_assumptions()
    print(f"Assumption 1 satisfied: {assumptions['assumption1_satisfied']}")
    print(f"Assumption 2 satisfied: {assumptions['assumption2_satisfied']}")
    
    if not assumptions['assumption1_satisfied'] or not assumptions['assumption2_satisfied']:
        print("Warning: Assumptions are not satisfied. Uniqueness may not hold.")
    
    # Check equilibrium uniqueness
    print("\nChecking equilibrium uniqueness with 3 random starting points...")
    uniqueness_result = model.check_equilibrium_uniqueness(n_attempts=3, max_iterations=30)
    
    print(f"\nUniqueness test results:")
    print(f"Unique equilibrium: {uniqueness_result['unique']}")
    print(f"Number of equilibria found: {uniqueness_result['n_equilibria_found']}")
    print(f"Maximum price difference: {uniqueness_result['max_price_diff']:.6f}")
    print(f"Maximum location difference: {uniqueness_result['max_location_diff']:.6f}")
    
    return uniqueness_result

def test_assumption_violations():
    """Test how violating different assumptions affects equilibrium properties"""
    print("\n=== TESTING ASSUMPTION VIOLATIONS ===")
    
    test_cases = [
        {
            'name': "Baseline (All assumptions satisfied)",
            'params': {'beta': 0.5, 'eta': 3.0, 'd_type': 'quadratic', 'rho_type': 'uniform'}
        },
        {
            'name': "Violation: Low elasticity (η < 1)",
            'params': {'beta': 0.5, 'eta': 0.9, 'd_type': 'quadratic', 'rho_type': 'uniform'}
        },
        {
            'name': "Violation: Non-strongly convex distance",
            'params': {'beta': 0.5, 'eta': 3.0, 'd_type': 'euclidean', 'rho_type': 'uniform'}
        },
        {
            'name': "Violation: High beta (β·d̄ ≥ 1)",
            'params': {'beta': 2.0, 'eta': 3.0, 'd_type': 'quadratic', 'rho_type': 'uniform'}
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        
        # Create model with test parameters
        model = HotellingTwoDimensional(
            n_firms=2,
            market_shape=(1, 1),
            beta=case['params']['beta'],
            eta=case['params']['eta'],
            d_type=case['params']['d_type'],
            rho_type=case['params']['rho_type']
        )
        
        # Check assumptions
        assumptions = model.verify_assumptions()
        print(f"Assumption 1 satisfied: {assumptions['assumption1_satisfied']}")
        print(f"Assumption 2 satisfied: {assumptions['assumption2_satisfied']}")
        
        # Try to find equilibrium
        print("Finding equilibrium...")
        converged = model.find_equilibrium(max_iterations=30, tolerance=1e-4, verbose=False)
        
        # Print results
        print(f"Converged: {converged}")
        if converged:
            print("Final prices:", model.prices)
            print("Final locations:", model.locations)
            
            if case['name'] != "Baseline (All assumptions satisfied)":
                # Check uniqueness for cases with violations
                print("Checking uniqueness...")
                uniqueness = model.check_equilibrium_uniqueness(n_attempts=2, max_iterations=20, verbose=False)
                print(f"Unique equilibrium: {uniqueness['unique']}")
            
            # Visualize
            model.visualize(show_density=False)
        
        # Store results
        results.append({
            'name': case['name'],
            'assumption1': assumptions['assumption1_satisfied'],
            'assumption2': assumptions['assumption2_satisfied'],
            'converged': converged,
            'final_prices': model.prices.copy() if converged else None,
            'final_locations': model.locations.copy() if converged else None
        })
    
    # Print summary
    print("\n=== SUMMARY OF ASSUMPTION VIOLATIONS ===")
    print("Test Case | Assumption 1 | Assumption 2 | Converged")
    print("-" * 60)
    for result in results:
        print(f"{result['name'][:20]}... | {result['assumption1']} | {result['assumption2']} | {result['converged']}")
    
    return results

def test_population_density_effects():
    """Test how different population densities affect equilibrium properties"""
    print("\n=== TESTING DIFFERENT POPULATION DENSITIES ===")
    
    density_types = ['uniform', 'gaussian', 'sine']
    results = []
    
    for density in density_types:
        print(f"\nTesting with {density} population density:")
        
        # Create model
        model = HotellingTwoDimensional(
            n_firms=2,
            market_shape=(1, 1),
            beta=0.5,
            eta=3.0,
            d_type='quadratic',
            mu=2.0,
            rho_type=density
        )
        
        # Find equilibrium
        print("Finding equilibrium...")
        converged = model.find_equilibrium(max_iterations=50, verbose=False)
        
        if converged:
            print(f"Final locations: {model.locations}")
            print(f"Final prices: {model.prices}")
            model.visualize(show_segmentation=True, show_density=True)
            
            results.append({
                'density': density,
                'locations': model.locations.copy(),
                'prices': model.prices.copy(),
                'profits': model.total_profit()
            })
    
    print("\n=== POPULATION DENSITY EFFECTS SUMMARY ===")
    print("With non-uniform density:")
    print("- Firms are drawn toward higher-density regions")
    print("- Price competition is influenced by both density and firm proximity")
    print("- Market segmentation adapts to the population distribution")
    
    return results

def varying_firms_simulation():
    """Test how the number of firms affects the equilibrium"""
    print("\n=== TESTING WITH VARYING NUMBER OF FIRMS ===")
    
    firm_counts = [2, 3, 4]
    results = []
    
    for n_firms in firm_counts:
        print(f"\nTesting with {n_firms} firms:")
        
        # Create model
        model = HotellingTwoDimensional(
            n_firms=n_firms,
            market_shape=(1, 1),
            beta=0.5,
            eta=3.0,
            d_type='quadratic',
            mu=2.0,
            rho_type='uniform'
        )
        
        # Find equilibrium
        print("Finding equilibrium...")
        converged = model.find_equilibrium(max_iterations=50, verbose=False)
        
        if converged:
            # Get results
            print(f"Final prices: {model.prices}")
            print(f"Average price: {np.mean(model.prices):.4f}")
            print(f"Price dispersion: {np.std(model.prices):.4f}")
            
            # Visualize
            model.visualize(show_density=False)
            
            # Store results
            results.append({
                'n_firms': n_firms,
                'converged': converged,
                'avg_price': np.mean(model.prices),
                'price_dispersion': np.std(model.prices),
                'locations': model.locations.copy(),
                'prices': model.prices.copy()
            })
    
    # Plot results
    if results:
        n_firms_list = [r['n_firms'] for r in results]
        avg_prices = [r['avg_price'] for r in results]
        price_dispersions = [r['price_dispersion'] for r in results]
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        axs[0].plot(n_firms_list, avg_prices, 'o-')
        axs[0].set_title('Average Price vs. Number of Firms')
        axs[0].set_xlabel('Number of Firms')
        axs[0].set_ylabel('Average Price')
        
        axs[1].plot(n_firms_list, price_dispersions, 'o-')
        axs[1].set_title('Price Dispersion vs. Number of Firms')
        axs[1].set_xlabel('Number of Firms')
        axs[1].set_ylabel('Price Standard Deviation')
        
        plt.tight_layout()
        plt.show()
    
    return results


# ========= MAIN FUNCTION =========

def main():
    """Main function with menu for different simulations"""
    print("TWO-DIMENSIONAL HOTELLING MODEL SIMULATION")
    print("==========================================")
    print("This program simulates a two-dimensional Hotelling model based on")
    print("the paper 'Beyond the Beach: Pricing Competition in Two Dimensions'")
    print("by Nicolò Badino and Andrea Sbarile.")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    while True:
        print("\nSelect a simulation to run:")
        print("1. Baseline simulation (satisfying all assumptions)")
        print("2. Test equilibrium uniqueness")
        print("3. Test assumption violations")
        print("4. Test population density effects")
        print("5. Test varying number of firms")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            run_baseline_simulation()
        elif choice == '2':
            test_equilibrium_uniqueness()
        elif choice == '3':
            test_assumption_violations()
        elif choice == '4':
            test_population_density_effects()
        elif choice == '5':
            varying_firms_simulation()
        elif choice == '6':
            print("\nExiting program. Thank you!")
            break
        else:
            print("\nInvalid choice, please try again.")


if __name__ == "__main__":
    main()
