import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os # Needed for path joining and file check
# streamlit_drawable_canvas, PIL.Image, base64 non sono più necessari
from hotelling_model import HotellingTwoDimensional  # Assuming your code is in hotelling_model.py

def main():
    st.set_page_config(layout="wide") # Use full width
    st.title("2D Hotelling Model Simulation Dashboard")

    # Initialize session state variables if they don't exist
    if 'simulation_run_once' not in st.session_state:
        st.session_state.simulation_run_once = False
        st.session_state.model = None
        st.session_state.converged = False
        st.session_state.current_profits = None
        st.session_state.iterations = 0
        st.session_state.time_elapsed = 0
        
        # State for specific analysis results to prevent re-computation on unrelated widget changes
        st.session_state.sensitivity_results_data = None
        st.session_state.grid_test_results_data = None
        st.session_state.landscape_plot_fig = None # Store the figure object
        st.session_state.nash_deviation_output_data = None
        st.session_state.convergence_gif_path = None
        # st.session_state.equilibrium_figures = [] # To store figures of multiple equilibria - REMOVED
        # st.session_state.assumption_robustness_results = None # For the new assumption robustness test - REMOVED

    # ===== Sidebar Controls =====
    st.sidebar.header("Simulation Parameters")
    
    # Model parameters
    n_firms = st.sidebar.slider("Number of firms", 2, 5, 2)
    market_shape = (st.sidebar.number_input("Market Width", 1.0, 5.0, 1.0),
                    st.sidebar.number_input("Market Height", 1.0, 5.0, 1.0))
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Demand and Cost Parameters")
    alpha_demand_input = st.sidebar.number_input("Alpha (Demand Intercept)", 0.1, 50.0, 10.0, key="alpha_demand_sb", help="Parameter 'alpha' in linear demand q = alpha - gamma*P")
    gamma_demand_input = st.sidebar.number_input("Gamma (Demand Slope)", 0.1, 10.0, 1.0, key="gamma_demand_sb", help="Parameter 'gamma' in linear demand q = alpha - gamma*P")
    t_transport_cost_input = st.sidebar.number_input("t (Transport Cost Rate)", 0.0, 5.0, 1.0, key="t_transport_sb", help="Transportation cost rate 't'")
    beta_logit_input = st.sidebar.number_input("Beta (Logit Sensitivity)", 0.1, 10.0, 1.0, key="beta_logit_sb", help="Sensitivity parameter 'beta' in the logit choice model")
    # Marginal cost 'c' and 'max_price' are taken from model defaults or could be added here if needed.
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Spatial Configuration")
    # Distance/density controls
    d_type = st.sidebar.selectbox("Distance Type", 
                                 ["euclidean", "manhattan", "quadratic"])
    rho_type = st.sidebar.selectbox("Density Type",
                                   ["uniform", "linear", "gaussian", "sine", "multi_gaussian"])
    
    density_centers_params = None # Initialize
    if rho_type == 'multi_gaussian':
        st.sidebar.subheader("Multi-Gaussian Density Parameters")
        
        num_foci = st.sidebar.number_input("Number of Gaussian Foci", 
                                           min_value=1, max_value=5, value=2, step=1, key="num_foci_input")

        # Initialize session state for randomized foci parameters
        if 'generated_foci_params' not in st.session_state:
            st.session_state.generated_foci_params = {} # Store by index
        if 'use_generated_foci_once' not in st.session_state:
            st.session_state.use_generated_foci_once = False

        def clear_generated_foci_flag_and_param(focus_idx_changed, param_type):
            # When user manually changes a field, we stop overriding for that specific field
            # and also generally stop using the "generated" set for subsequent reruns unless "Randomize" is pressed again.
            st.session_state.use_generated_foci_once = False
            if focus_idx_changed in st.session_state.generated_foci_params:
                if param_type in st.session_state.generated_foci_params[focus_idx_changed]:
                    # Mark this specific param as "manually overridden" by removing it from generated
                    del st.session_state.generated_foci_params[focus_idx_changed][param_type]


        if st.sidebar.button("Randomize Foci", key="randomize_foci_button"):
            st.session_state.generated_foci_params = {} # Clear previous randomizations
            for i_rand in range(int(num_foci)):
                rand_cx = np.random.uniform(0, market_shape[0])
                rand_cy = np.random.uniform(0, market_shape[1])
                rand_strength = np.random.uniform(0.5, 3.0)
                # Ensure sigma is not too large relative to market, but also not too small
                rand_sigma = np.random.uniform(0.1, max(market_shape[0], market_shape[1]) / 3.0) 
                st.session_state.generated_foci_params[i_rand] = {
                    'center_x': rand_cx, 'center_y': rand_cy,
                    'strength': rand_strength, 'sigma': rand_sigma
                }
            st.session_state.use_generated_foci_once = True
            # Streamlit automatically reruns, new values will be picked up by widgets

        density_centers_params = []
        for i in range(int(num_foci)):
            st.sidebar.markdown(f"**Focus {i+1}**")
            
            # Determine initial values for widgets
            # Default values (staggered)
            default_cx = market_shape[0] * ((i + 1) / (int(num_foci) + 1))
            default_cy = market_shape[1] * ((i + 1) / (int(num_foci) + 1))
            default_strength = 1.0
            default_sigma = 0.3

            # Override with generated values if flag is set and params exist for this focus index
            val_cx = st.session_state.generated_foci_params.get(i, {}).get('center_x', default_cx)
            val_cy = st.session_state.generated_foci_params.get(i, {}).get('center_y', default_cy)
            val_strength = st.session_state.generated_foci_params.get(i, {}).get('strength', default_strength)
            val_sigma = st.session_state.generated_foci_params.get(i, {}).get('sigma', default_sigma)

            foci_col1, foci_col2 = st.sidebar.columns(2)
            center_x = foci_col1.number_input(f"Center X ({i+1})", 
                                              min_value=0.0, max_value=float(market_shape[0]), 
                                              value=float(val_cx), 
                                              step=0.1, key=f"mg_cx_{i}",
                                              on_change=clear_generated_foci_flag_and_param, args=(i, "center_x"))
            center_y = foci_col2.number_input(f"Center Y ({i+1})", 
                                              min_value=0.0, max_value=float(market_shape[1]), 
                                              value=float(val_cy), 
                                              step=0.1, key=f"mg_cy_{i}",
                                              on_change=clear_generated_foci_flag_and_param, args=(i, "center_y"))
            
            strength_col, sigma_col = st.sidebar.columns(2)
            strength = strength_col.number_input(f"Strength ({i+1})", 
                                                 min_value=0.1, max_value=10.0, 
                                                 value=float(val_strength), 
                                                 step=0.1, key=f"mg_str_{i}",
                                                 on_change=clear_generated_foci_flag_and_param, args=(i, "strength"))
            sigma = sigma_col.number_input(f"Sigma (spread) ({i+1})", 
                                           min_value=0.01, max_value=float(max(market_shape)/2), 
                                           value=float(val_sigma), 
                                           step=0.01, key=f"mg_sig_{i}",
                                           on_change=clear_generated_foci_flag_and_param, args=(i, "sigma"))
            
            density_centers_params.append({'center': (center_x, center_y), 'strength': strength, 'sigma': sigma})
        
        # After all focus widgets are created for this run, reset the general flag
        # This ensures that if "Randomize" was pressed, values are used for THIS run,
        # but subsequent reruns (e.g. due to other widget changes) won't re-apply them
        # unless "Randomize" is pressed again. Manual changes are preserved by the widget's state.
        if st.session_state.use_generated_foci_once:
            st.session_state.use_generated_foci_once = False


    # Simulation controls
    max_iter = st.sidebar.slider("Max Iterations", 10, 200, 50)
    grid_size = st.sidebar.slider("Grid Size (for simulation)", 10, 100, 30, help="Grid size for demand/profit calculation during optimization.")
    update_method_options = ['sequential', 'simultaneous']
    update_method = st.sidebar.selectbox("Update Method", update_method_options, index=0, help="Method for updating prices and locations during equilibrium search.")
    
    # ===== Main Panel =====
    if st.sidebar.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            # Initialize model
            temp_model_obj = HotellingTwoDimensional( 
                n_firms=n_firms,
                market_shape=market_shape,
                alpha_demand=alpha_demand_input,
                gamma_demand=gamma_demand_input,
                t_transport_cost=t_transport_cost_input,
                beta_logit=beta_logit_input,
                d_type=d_type,
                rho_type=rho_type,
                density_params=density_centers_params
                # c and max_price will use defaults from HotellingTwoDimensional constructor
            )
            
            # Run simulation
            sim_results = temp_model_obj.find_equilibrium( # Use a temporary variable name
                max_iterations=max_iter,
                grid_size=grid_size,
                update_method=update_method, # Pass the selected update method
                verbose=False
            )

            # Store results in session state
            st.session_state.model = temp_model_obj
            st.session_state.converged = sim_results['converged']
            st.session_state.iterations = sim_results['iterations']
            st.session_state.time_elapsed = sim_results['time_elapsed']
            if temp_model_obj and sim_results['converged']: 
                 st.session_state.current_profits = temp_model_obj.total_profit(grid_size=grid_size)
            else:
                 st.session_state.current_profits = None
            st.session_state.simulation_run_once = True
            
            # Clear previous specific analysis results when a new main simulation is run
            st.session_state.sensitivity_results_data = None
            st.session_state.grid_test_results_data = None
            st.session_state.landscape_plot_fig = None
            st.session_state.nash_deviation_output_data = None
            st.session_state.convergence_gif_path = None # Reset GIF path on new simulation
            # st.session_state.equilibrium_figures = [] # Reset equilibrium figures - REMOVED
            # st.session_state.assumption_robustness_results = None # Reset assumption robustness results - REMOVED


    # Display content if simulation has been run at least once
    if st.session_state.simulation_run_once and st.session_state.model:
        # Retrieve model and converged status from session state for use in tabs
        model = st.session_state.model
        converged = st.session_state.converged
        current_profits = st.session_state.current_profits
        iterations = st.session_state.iterations
        time_elapsed = st.session_state.time_elapsed


        # Create tabs for organizing output
        tab_viz_overview, tab_detailed_props, tab_sensitivity_density, \
        tab_dynamics_robustness, tab_advanced_analysis, tab_monopoly = st.tabs([ 
            "📈 Visualizations & Overview", 
            "📊 Detailed Equilibrium Properties", 
            "🔬 Sensitivity & Density Analysis",
            "⚙️ Dynamics & Robustness",
            "🗺️ Advanced Analysis & Metrics",
            "👑 Monopoly Analysis" 
        ])

        with tab_viz_overview:
            st.header("Simulation Overview & Visualizations")
            # Use a potentially higher grid_size for visualization if desired
            viz_grid_size = st.slider("Grid Size for Visualizations", 20, 150, 75, key="viz_grid_slider", help="Finer grid for smoother visuals.")
            # Display results (summary)
            with st.expander("Key Simulation Results", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("##### Final Locations")
                    for i, loc in enumerate(model.locations):
                        st.write(f"Firm {i+1}: ({loc[0]:.2f}, {loc[1]:.2f})")
                        
                with col2:
                    st.write("##### Final Prices")
                    for i, price in enumerate(model.prices):
                        st.write(f"Firm {i+1}: {price:.2f}")
                
                with col3:
                    st.write("##### Final Profits")
                    # current_profits should be available from session_state
                    if current_profits is not None:
                        for i, profit_val in enumerate(current_profits):
                            st.write(f"Firm {i+1}: {profit_val:.2f}")
                    else:
                        st.write("Profits not calculated yet.")

            # Visualizations
            with st.expander("Market Visualization", expanded=True):
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    st.write("##### Market Segmentation")
                    # Use a potentially higher grid_size for visualization if desired
                    # viz_grid_size is now defined at the top of the tab
                    fig_seg = model.visualize(show_segmentation=True, show_density=False, grid_size=viz_grid_size)
                    st.pyplot(fig_seg)
                    plt.clf() 
                
                with viz_col2:
                    st.write("##### Population Density")
                    fig_den = model.visualize(show_segmentation=False, show_density=True, grid_size=viz_grid_size)
                    st.pyplot(fig_den)
                    plt.clf() 
            
            # Convergence plots
            with st.expander("Convergence Metrics", expanded=True):
                fig_conv = model.plot_convergence()
                st.pyplot(fig_conv)
                plt.clf()

                # --- Convergence GIF Section ---
                st.subheader("Convergence GIF")
                gif_fps = st.slider("GIF FPS (Frames Per Second)", 1, 10, 3, key="gif_fps_slider_main") # Changed key for uniqueness
                gif_density_grid = st.slider("GIF Background Density Grid Size", 10, 50, 20, key="gif_density_grid_slider_main", help="Grid size for rendering density in GIF background. Higher is smoother but slower.") # Changed key

                if st.button("Generate Convergence GIF", key="generate_gif_button_main"): # Changed key
                    if model and model.price_history and len(model.price_history) >= 2:
                        # Ensure a unique filename, possibly in a temporary directory if permissions allow
                        # For simplicity, saving in the current directory with a timestamp or unique ID
                        # temp_gif_filename = f"temp_convergence_{n_firms}firms_{int(time.time())}.gif" 
                        # Using a fixed name for simplicity in example, but unique names are better in production
                        gif_filename = f"convergence_viz_tab.gif" 
                        
                        with st.spinner("Generating convergence GIF... This may take a moment."):
                            # Ensure the model instance from session state is used
                            generated_path = st.session_state.model.generate_convergence_gif(
                                filename=gif_filename, 
                                fps=gif_fps,
                                viz_grid_size_for_density_plot=gif_density_grid
                            )
                        if generated_path and os.path.exists(generated_path):
                            st.session_state.convergence_gif_path = generated_path
                            st.success(f"Convergence GIF generated: {generated_path}")
                        else:
                            st.error("Failed to generate convergence GIF or GIF file not found.")
                            st.session_state.convergence_gif_path = None
                    else:
                        st.warning("Not enough simulation history to generate a GIF. Run a longer simulation or ensure it converges over several iterations.")

                if st.session_state.convergence_gif_path and os.path.exists(st.session_state.convergence_gif_path):
                    try:
                        with open(st.session_state.convergence_gif_path, "rb") as f_gif_viz: # Renamed file handle
                            st.image(f_gif_viz.read(), caption="Convergence GIF", use_column_width=True)
                        # Provide a download button for the GIF
                        with open(st.session_state.convergence_gif_path, "rb") as file_bytes_gif_viz: # Renamed file handle
                            st.download_button(
                                label="Download Convergence GIF",
                                data=file_bytes_gif_viz,
                                file_name=os.path.basename(st.session_state.convergence_gif_path),
                                mime="image/gif",
                                key="download_gif_button_main" # Changed key
                            )
                    except FileNotFoundError:
                        st.error("GIF file not found. It might have been deleted or moved.")
                        st.session_state.convergence_gif_path = None # Reset if not found
                elif st.session_state.convergence_gif_path: # Path exists in state but file doesn't
                     st.warning(f"Previously generated GIF at {st.session_state.convergence_gif_path} seems to be missing. Please regenerate.")
                     st.session_state.convergence_gif_path = None

            with tab_detailed_props:
                st.header("Detailed Equilibrium Properties")
                st.write("Test: Contenuto visibile per la scheda Proprietà Dettagliate dell'Equilibrio.") # Riga di diagnostica

                # Firm Locations Analysis
                st.subheader("Firm Locations")
                locations_data = [{"Firm": f"Firm {i+1}", "X": model.locations[i,0], "Y": model.locations[i,1]} for i in range(model.n_firms)]
                st.table(locations_data)
                loc_stats_cols = st.columns(2)
                loc_stats_cols[0].metric("Avg X-coordinate", f"{np.mean(model.locations[:,0]):.2f}", delta=None)
                loc_stats_cols[1].metric("Avg Y-coordinate", f"{np.mean(model.locations[:,1]):.2f}", delta=None)
                loc_stats_cols[0].metric("Std Dev X-coordinate", f"{np.std(model.locations[:,0]):.2f}", delta=None)
                loc_stats_cols[1].metric("Std Dev Y-coordinate", f"{np.std(model.locations[:,1]):.2f}", delta=None)

                # Firm Prices Analysis
                st.subheader("Firm Prices")
                prices_data = [{"Firm": f"Firm {i+1}", "Price": model.prices[i]} for i in range(model.n_firms)]
                st.table(prices_data)
                price_stats_cols = st.columns(2)
                price_stats_cols[0].metric("Average Price", f"{np.mean(model.prices):.2f}")
                price_stats_cols[1].metric("Price Std Dev (Dispersion)", f"{np.std(model.prices):.2f}")

                # Firm Profits Analysis
                st.subheader("Firm Profits")
                # profits are now from current_profits (from session_state)
                if current_profits is not None:
                    profits_data = [{"Firm": f"Firm {i+1}", "Profit": current_profits[i]} for i in range(model.n_firms)]
                    st.table(profits_data)
                    profit_stats_cols = st.columns(3)
                    profit_stats_cols[0].metric("Total Profit (All Firms)", f"{np.sum(current_profits):.2f}")
                    profit_stats_cols[1].metric("Average Profit per Firm", f"{np.mean(current_profits):.2f}")
                    profit_stats_cols[2].metric("Profit Std Dev", f"{np.std(current_profits):.2f}")
                else:
                    st.write("Profits not available.")

                # Market Segmentation Quantification
                st.subheader("Market Segmentation Analysis (Approximate Shares)")
                # Use the simulation grid_size for consistency with profit/demand calculations
                # Or allow a separate grid_size for this analysis if computationally intensive
                segmentation_grid_size = grid_size # Using simulation grid_size
                
                market_x_coords = np.linspace(0, model.market_shape[0], segmentation_grid_size)
                market_y_coords = np.linspace(0, model.market_shape[1], segmentation_grid_size)
                X_viz, Y_viz = np.meshgrid(market_x_coords, market_y_coords)
                Z_segmentation = np.zeros_like(X_viz, dtype=int)

                for r_idx in range(X_viz.shape[0]):
                    for c_idx in range(X_viz.shape[1]):
                        consumer_loc = np.array([X_viz[r_idx, c_idx], Y_viz[r_idx, c_idx]])
                        # Need to temporarily set model's prices and locations if they were changed by uniqueness check
                        # However, model object here should be the one from the main simulation run
                        probs = model.choice_prob(consumer_loc) 
                        Z_segmentation[r_idx, c_idx] = np.argmax(probs)
                
                market_shares = np.zeros(model.n_firms)
                total_cells = Z_segmentation.size
                if total_cells > 0:
                    for firm_i in range(model.n_firms):
                        market_shares[firm_i] = np.sum(Z_segmentation == firm_i) / total_cells * 100 # In percentage

                shares_data = [{"Firm": f"Firm {i+1}", "Market Share (%)": f"{market_shares[i]:.2f}%"} for i in range(model.n_firms)]
                st.table(shares_data)
                
                fig_shares, ax_shares = plt.subplots()
                ax_shares.pie(market_shares, labels=[f"Firm {i+1}" for i in range(model.n_firms)], autopct='%1.1f%%', startangle=90)
                ax_shares.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig_shares)
                plt.clf()

            # THEORETICAL VERIFICATION TAB REMOVED

            with tab_sensitivity_density:
                st.header("Sensitivity Analysis & Population Density Effects")

                # --- Sensitivity Analysis Section ---
                st.subheader("Parametric Sensitivity Analysis (Simplified Comparative Statics)")
                
                param_to_sweep = st.selectbox(
                    "Select parameter to vary:",
                    options=['alpha_demand', 'gamma_demand', 't_transport_cost', 'beta_logit'],
                    key="sensitivity_param_select"
                )

                # Adjust default sweep values based on typical ranges of new parameters
                default_min_val, default_max_val, default_step = 0.5, 2.0, 0.1
                if param_to_sweep == 'alpha_demand':
                    default_min_val, default_max_val, default_step = 5.0, 15.0, 1.0
                elif param_to_sweep == 'gamma_demand':
                    default_min_val, default_max_val, default_step = 0.5, 2.0, 0.1
                elif param_to_sweep == 't_transport_cost':
                    default_min_val, default_max_val, default_step = 0.5, 3.0, 0.25
                elif param_to_sweep == 'beta_logit':
                    default_min_val, default_max_val, default_step = 0.5, 5.0, 0.5
                    
                sweep_col1, sweep_col2, sweep_col3 = st.columns(3)
                min_val = sweep_col1.number_input("Minimum value", value=default_min_val, step=default_step, key="sweep_min")
                max_val = sweep_col2.number_input("Maximum value", value=default_max_val, step=default_step, key="sweep_max")
                num_steps = sweep_col3.number_input("Number of steps", min_value=2, max_value=20, value=5, step=1, key="sweep_steps")

                if st.button("Run Sensitivity Analysis", key="run_sensitivity_analysis_button"):
                    if min_val >= max_val:
                        st.error("Minimum value must be less than maximum value.")
                    else:
                        parameter_values_sweep = np.linspace(min_val, max_val, int(num_steps)) # Renamed to avoid conflict
                        temp_results_sensitivity = { # Use a temporary dict
                            'param_values': [], 'avg_prices': [], 'avg_profits': [],
                            'loc_std_dev_x': [], 'loc_std_dev_y': []
                        }
                        all_simulations_converged_in_sweep = True
                        try:
                            with st.spinner(f"Running sensitivity analysis for {param_to_sweep}... This may take a moment."):
                                for val in parameter_values_sweep:
                                    # Base parameters from the main simulation's setup
                                    # These are the values from the sidebar when "Run Simulation" was last pressed
                                    # or their initial defaults.
                                    current_params = {
                                        "n_firms": n_firms, # From sidebar
                                        "market_shape": market_shape, # From sidebar
                                        "alpha_demand": alpha_demand_input, # From sidebar
                                        "gamma_demand": gamma_demand_input, # From sidebar
                                        "t_transport_cost": t_transport_cost_input, # From sidebar
                                        "beta_logit": beta_logit_input, # From sidebar
                                        "d_type": d_type, # From sidebar
                                        "rho_type": rho_type, # From sidebar
                                        "density_params": density_centers_params, # From sidebar
                                        # c and max_price will use defaults from constructor if not specified
                                        # For sensitivity, it's better to fix them if they are not being swept.
                                        # Assuming they are fixed at the model's defaults for now.
                                        # If c or max_price were sidebar inputs, they'd be used here.
                                    }
                                    # Override the parameter being swept
                                    if param_to_sweep == 'alpha_demand': current_params["alpha_demand"] = val
                                    elif param_to_sweep == 'gamma_demand': current_params["gamma_demand"] = val
                                    elif param_to_sweep == 't_transport_cost': current_params["t_transport_cost"] = val
                                    elif param_to_sweep == 'beta_logit': current_params["beta_logit"] = val
                                    
                                    temp_model_sens = HotellingTwoDimensional(**current_params)
                                    # Use find_equilibrium from the temp_model_sens instance
                                    converged_info_sens = temp_model_sens.find_equilibrium(
                                        max_iterations=max_iter, grid_size=grid_size, verbose=False, update_method=update_method
                                    )

                                    if converged_info_sens['converged']:
                                        temp_results_sensitivity['param_values'].append(val)
                                        temp_results_sensitivity['avg_prices'].append(np.mean(temp_model_sens.prices))
                                        current_total_profits_sens = temp_model_sens.total_profit(grid_size=grid_size)
                                        temp_results_sensitivity['avg_profits'].append(np.mean(current_total_profits_sens))
                                        temp_results_sensitivity['loc_std_dev_x'].append(np.std(temp_model_sens.locations[:, 0]))
                                        temp_results_sensitivity['loc_std_dev_y'].append(np.std(temp_model_sens.locations[:, 1]))
                                    else:
                                        temp_results_sensitivity['param_values'].append(val)
                                        temp_results_sensitivity['avg_prices'].append(np.nan)
                                        temp_results_sensitivity['avg_profits'].append(np.nan)
                                        temp_results_sensitivity['loc_std_dev_x'].append(np.nan)
                                        temp_results_sensitivity['loc_std_dev_y'].append(np.nan)
                                        st.warning(f"Simulation did not converge for {param_to_sweep} = {val:.2f}")
                                        all_simulations_converged_in_sweep = False
                            
                            st.session_state.sensitivity_results_data = temp_results_sensitivity # Store in session state
                            if not temp_results_sensitivity['param_values']:
                                st.error("Sensitivity analysis produced no results.")
                            elif all_simulations_converged_in_sweep:
                                st.success("Sensitivity analysis complete. All simulations converged.")
                            else:
                                st.warning("Sensitivity analysis partially complete. Some simulations did not converge.")
                        except Exception as e:
                            st.error(f"An error occurred during sensitivity analysis: {str(e)}")
                            st.exception(e)
                            st.session_state.sensitivity_results_data = None 
                
                # Display sensitivity analysis results if they exist in session state
                if st.session_state.sensitivity_results_data:
                    results_sensitivity_to_plot = st.session_state.sensitivity_results_data
                    if results_sensitivity_to_plot['param_values'] and \
                       any(not np.isnan(p) for p in results_sensitivity_to_plot.get('avg_prices', [])):
                        
                        param_name_label_for_plot = param_to_sweep # Use the full name from selectbox options
                        fig_sens, axs_sens = plt.subplots(2, 2, figsize=(12, 10))
                        fig_sens.suptitle(f"Sensitivity Analysis: Impact of {param_name_label_for_plot}", fontsize=16)

                        axs_sens[0,0].plot(results_sensitivity_to_plot['param_values'], results_sensitivity_to_plot['avg_prices'], marker='o')
                        axs_sens[0,0].set_xlabel(param_name_label_for_plot); axs_sens[0,0].set_ylabel("Average Price"); axs_sens[0,0].set_title("Average Price")
                        axs_sens[0,1].plot(results_sensitivity_to_plot['param_values'], results_sensitivity_to_plot['avg_profits'], marker='o')
                        axs_sens[0,1].set_xlabel(param_name_label_for_plot); axs_sens[0,1].set_ylabel("Average Profit per Firm"); axs_sens[0,1].set_title("Average Profit")
                        axs_sens[1,0].plot(results_sensitivity_to_plot['param_values'], results_sensitivity_to_plot['loc_std_dev_x'], marker='o')
                        axs_sens[1,0].set_xlabel(param_name_label_for_plot); axs_sens[1,0].set_ylabel("Std Dev of X-Locations"); axs_sens[1,0].set_title("Location Dispersion (X)")
                        axs_sens[1,1].plot(results_sensitivity_to_plot['param_values'], results_sensitivity_to_plot['loc_std_dev_y'], marker='o')
                        axs_sens[1,1].set_xlabel(param_name_label_for_plot); axs_sens[1,1].set_ylabel("Std Dev of Y-Locations"); axs_sens[1,1].set_title("Location Dispersion (Y)")
                        
                        plt.tight_layout(rect=[0, 0, 1, 0.96])
                        st.pyplot(fig_sens)
                        plt.clf()
                    else:
                        st.info("No valid data to plot from the last sensitivity analysis.")
                
                st.markdown("---")
                # --- Population Density Correlation Section ---
                st.subheader("Population Density-Location Correlation")
                # This section uses 'model' and 'converged' which are now retrieved from session_state
                # at the beginning of the 'if st.session_state.simulation_run_once:' block
                if model and converged: # Check if main simulation has run and converged
                    if model.rho_type == 'gaussian':
                        st.markdown("**Gaussian Density:**")
                        center_x, center_y = model.market_shape[0]/2, model.market_shape[1]/2
                        gaussian_center = np.array([center_x, center_y])
                        # Corrected distance calculation: loc - gaussian_center
                        distances_to_center = [np.linalg.norm(loc - gaussian_center) for loc in model.locations]
                        avg_dist_to_center = np.mean(distances_to_center)
                        st.metric("Avg. Firm Distance to Gaussian Center", f"{avg_dist_to_center:.3f}")
                        
                        # Display distances per firm
                        for i, dist in enumerate(distances_to_center):
                             st.write(f"Firm {i+1} distance to center: {dist:.3f}")

                    elif model.rho_type == 'multi_gaussian' and model.density_params: # model.density_params comes from the main model
                        st.markdown("**Multi-Gaussian Density:**")
                        # model.density_params are the ones used in the main simulation
                        if not model.density_params: 
                            st.info("No multi-gaussian foci defined for the main simulation.")
                        else:
                            avg_min_distances = []
                            details_per_firm = []
                            # foci_centers are from the main simulation's model
                            foci_centers = np.array([params['center'] for params in model.density_params]) 

                            for i, firm_loc in enumerate(model.locations): # model.locations from main simulation
                                distances_to_foci = [np.linalg.norm(firm_loc - focus_center) for focus_center in foci_centers]
                                min_dist = np.min(distances_to_foci)
                                closest_focus_idx = np.argmin(distances_to_foci)
                                avg_min_distances.append(min_dist)
                                details_per_firm.append(
                                    f"Firm {i+1}: Closest to Focus {closest_focus_idx+1} "
                                    f"({model.density_params[closest_focus_idx]['center'][0]:.2f}, "
                                    f"{model.density_params[closest_focus_idx]['center'][1]:.2f}) "
                                    f"at distance {min_dist:.3f}"
                                )
                            
                            if avg_min_distances:
                                st.metric("Avg. Firm Distance to Nearest Gaussian Focus", f"{np.mean(avg_min_distances):.3f}")
                                with st.expander("Details per firm (closest focus and distance):"):
                                    for detail in details_per_firm:
                                        st.write(detail)
                            else:
                                st.info("Could not calculate distances to foci.")
                    else:
                        st.info("Density-Location correlation analysis is available for 'gaussian' or 'multi_gaussian' density types.")
                else: # Handles case where model is None or not converged from session_state
                    st.info("Run the main simulation successfully to see Density-Location correlation analysis.")
            
            with tab_dynamics_robustness:
                st.header("Competitive Dynamics & Robustness Checks")

                # --- Nash Equilibrium Verification Section ---
                st.subheader("Nash Equilibrium Verification (Unilateral Deviation)")
                if converged:
                    st.write("Verify if any firm has an incentive to unilaterally deviate from the found equilibrium.")
                    
                    firm_to_check = st.selectbox(
                        "Select firm to check for deviation:", 
                        options=[f"Firm {i+1}" for i in range(model.n_firms)],
                        key="nash_firm_select"
                    )
                    firm_idx_check = int(firm_to_check.split(" ")[1]) - 1
                    
                    current_profit_firm = current_profits[firm_idx_check] if current_profits is not None else model.profit(firm_idx_check, grid_size)
                    st.write(f"**{firm_to_check}'s Current Profit (at equilibrium): {current_profit_firm:.4f}**")

                    deviation_type_choice = st.radio(
                        "Select deviation type:",
                        ("Price", "Location X", "Location Y"), # Reverted to separate X and Y
                        key="nash_deviation_type_choice"
                    )
                    
                    dev_amount = st.number_input("Deviation amount (e.g., +/- 0.01 or 0.1)", value=0.01, step=0.005, format="%.3f", key="nash_dev_amount")

                    if st.button("Check Profit with Deviation", key="nash_check_deviation_button"):
                        deviated_price_nash = None
                        deviated_location_nash = None
                        original_price = model.prices[firm_idx_check]
                        original_location = model.locations[firm_idx_check].copy()
                        value_description = "" # To describe the deviation in the output

                        if deviation_type_choice == "Price":
                            deviated_price_nash = original_price + dev_amount
                            deviated_price_nash = np.clip(deviated_price_nash, model.c, model.max_price)
                            value_description = f"to price {deviated_price_nash:.2f}"
                        elif deviation_type_choice == "Location X":
                            deviated_location_nash = original_location.copy()
                            deviated_location_nash[0] += dev_amount
                            deviated_location_nash[0] = np.clip(deviated_location_nash[0], 0, model.market_shape[0])
                            value_description = f"X-coordinate to {deviated_location_nash[0]:.2f} (Y: {deviated_location_nash[1]:.2f})"
                        elif deviation_type_choice == "Location Y":
                            deviated_location_nash = original_location.copy()
                            deviated_location_nash[1] += dev_amount
                            deviated_location_nash[1] = np.clip(deviated_location_nash[1], 0, model.market_shape[1])
                            value_description = f"Y-coordinate to {deviated_location_nash[1]:.2f} (X: {deviated_location_nash[0]:.2f})"
                        
                        profit_with_deviation = model.calculate_firm_profit_for_deviation(
                            firm_idx_check, 
                            deviation_price=deviated_price_nash,
                            deviation_location=deviated_location_nash,
                            grid_size=grid_size 
                        )
                        st.session_state.nash_deviation_output_data = {
                            "firm": firm_to_check, 
                            "type": deviation_type_choice, 
                            "value_desc": value_description,
                            "profit": profit_with_deviation, 
                            "change": profit_with_deviation - current_profit_firm
                        }
                    
                    # Display Nash deviation results if they exist in session state
                    if st.session_state.nash_deviation_output_data:
                        res = st.session_state.nash_deviation_output_data
                        st.write(f"Profit if {res['firm']} deviates by changing {res['type']} {res['value_desc']}: **{res['profit']:.4f}**")
                        st.write(f"Change in profit: {res['change']:+.4f}")
                        if res['change'] > 1e-5: # Allow for small numerical noise
                            st.warning("Firm could potentially increase profit by deviating.")
                        else:
                            st.success("Firm does not appear to increase profit by this deviation.")
                else:
                    st.info("Run a simulation that converges to an equilibrium to perform Nash Equilibrium verification.")

                st.markdown("---")
                # --- Grid Resolution Robustness Test ---
                st.subheader("Grid Resolution Robustness Test")
                st.write("Test how changing the simulation's grid_size affects key outcomes.")
                
                grid_sizes_to_test_str = st.text_input(
                    "Enter 2-3 grid sizes to test (comma-separated, e.g., 20,30,40):", 
                    value="20,30,40",
                    key="grid_robust_input"
                )
                
                if st.button("Run Grid Resolution Test", key="grid_robust_button"):
                    try:
                        grid_sizes_list_test = [int(s.strip()) for s in grid_sizes_to_test_str.split(',') if s.strip()] # Renamed
                        if not (2 <= len(grid_sizes_list_test) <= 4): 
                            st.error("Please enter 2 to 4 valid, comma-separated grid sizes.")
                        else:
                            temp_results_grid_robustness = [] # Use a temporary list
                            with st.spinner("Running grid resolution tests..."):
                                for test_gs in grid_sizes_list_test:
                                    if test_gs <= 0:
                                        st.warning(f"Skipping invalid grid size: {test_gs}")
                                        continue
                                    
                                    s_model_grid = st.session_state.model # Access model from session state for default c, max_price
                                    temp_params_for_grid_test = {
                                        "n_firms": n_firms, # From sidebar
                                        "market_shape": market_shape, # From sidebar
                                        "alpha_demand": alpha_demand_input, # From sidebar
                                        "gamma_demand": gamma_demand_input, # From sidebar
                                        "t_transport_cost": t_transport_cost_input, # From sidebar
                                        "beta_logit": beta_logit_input, # From sidebar
                                        "d_type": d_type, # From sidebar
                                        "rho_type": rho_type, # From sidebar
                                        "density_params": density_centers_params, # From sidebar
                                        "c": s_model_grid.c, # from model instance
                                        "max_price": s_model_grid.max_price # from model instance
                                    }
                                    temp_model_grid_test = HotellingTwoDimensional(**temp_params_for_grid_test)
                                    
                                    sim_res_gs = temp_model_grid_test.find_equilibrium(
                                        max_iterations=max_iter, grid_size=test_gs,       
                                        update_method=update_method, verbose=False
                                    )
                                    
                                    total_profit_val = None
                                    if sim_res_gs['converged']:
                                        total_profit_val = np.sum(temp_model_grid_test.total_profit(grid_size=test_gs))
                                        
                                    temp_results_grid_robustness.append({
                                        'Grid Size': test_gs, 'Converged': sim_res_gs['converged'],
                                        'Iterations': sim_res_gs['iterations'], 'Time (s)': f"{sim_res_gs['time_elapsed']:.2f}",
                                        'Total Profit': f"{total_profit_val:.2f}" if total_profit_val is not None else "N/A"
                                    })
                            st.session_state.grid_test_results_data = temp_results_grid_robustness # Store in session state
                    except ValueError:
                        st.error("Invalid input for grid sizes. Please use comma-separated numbers (e.g., 20,30,40).")
                    except Exception as e_grid:
                        st.error(f"An error occurred during grid resolution test: {str(e_grid)}")
                        st.exception(e_grid)
                        st.session_state.grid_test_results_data = None
                
                # Display grid resolution test results if they exist in session state
                if st.session_state.grid_test_results_data:
                    st.write("### Grid Resolution Test Results:")
                    st.table(st.session_state.grid_test_results_data)
                elif st.session_state.grid_test_results_data == []: # Explicitly check for empty list if no valid sizes were tested
                     st.info("No valid grid sizes were tested in the last run.")
            
            with tab_advanced_analysis:
                st.header("Advanced Analysis & Metrics")

                # --- Profit Landscape Analysis ---
                st.subheader("Profit Landscape Analysis")
                if converged:
                    st.write("Visualize the profit a selected firm would make by moving to different locations, "
                             "keeping its current price and other firms' strategies fixed.")

                    landscape_firm_options = [f"Firm {i+1}" for i in range(model.n_firms)]
                    landscape_firm_choice = st.selectbox("Select firm for landscape analysis:", 
                                                         options=landscape_firm_options, 
                                                         key="landscape_firm_choice")
                    landscape_firm_idx = int(landscape_firm_choice.split(" ")[1]) - 1

                    landscape_display_grid_size = st.slider(
                        "Landscape Visualization Grid Size:", 
                        min_value=10, max_value=30, value=15, step=1, 
                        key="landscape_viz_grid",
                        help="Resolution of the grid for visualizing the profit landscape. Higher values are more detailed but slower."
                    )
                    
                    # profit_calc_grid_size for the internal profit calculation can be fixed or linked to main sim grid_size
                    # For simplicity, let's use the main simulation's grid_size
                    profit_calc_grid_size_for_landscape = grid_size 

                    if st.button("Generate Profit Landscape", key="generate_landscape_button"):
                        with st.spinner(f"Calculating profit landscape for {landscape_firm_choice}..."):
                            profit_matrix, x_coords_landscape, y_coords_landscape = model.calculate_profit_landscape( # Renamed coords
                                firm_idx=landscape_firm_idx,
                                landscape_grid_size=landscape_display_grid_size,
                                profit_calc_grid_size=profit_calc_grid_size_for_landscape
                            )
                        
                        fig_landscape_plot, ax_landscape_plot = plt.subplots(figsize=(8, 7)) # Renamed fig/ax
                        im = ax_landscape_plot.imshow(profit_matrix.T, 
                                                 extent=[x_coords_landscape.min(), x_coords_landscape.max(), 
                                                         y_coords_landscape.min(), y_coords_landscape.max()], 
                                                 origin='lower', aspect='auto', cmap='viridis', interpolation='bilinear')
                        fig_landscape_plot.colorbar(im, ax=ax_landscape_plot, label=f"{landscape_firm_choice} Profit")
                        
                        for i in range(model.n_firms):
                            if i != landscape_firm_idx:
                                ax_landscape_plot.scatter(model.locations[i,0], model.locations[i,1], 
                                                     s=80, c='white', edgecolor='black', marker='o', 
                                                     label=f"Firm {i+1} (Fixed)")
                        ax_landscape_plot.scatter(model.locations[landscape_firm_idx,0], model.locations[landscape_firm_idx,1], 
                                             s=120, c='red', edgecolor='black', marker='X', 
                                             label=f"{landscape_firm_choice} (Equilibrium)")
                        
                        ax_landscape_plot.set_title(f"Profit Landscape for {landscape_firm_choice}")
                        ax_landscape_plot.set_xlabel("X-coordinate"); ax_landscape_plot.set_ylabel("Y-coordinate")
                        ax_landscape_plot.legend(fontsize='small')
                        st.session_state.landscape_plot_fig = fig_landscape_plot # Store figure in session state
                    
                # Display landscape plot if it exists in session state
                if st.session_state.landscape_plot_fig:
                    st.pyplot(st.session_state.landscape_plot_fig)
                    # plt.clf() # Clearing the global figure might not be needed if we store the fig object
                                # However, if the same fig object is modified elsewhere, this could be an issue.
                                # For safety, if we are done with this specific figure, we can close it.
                                # Or, ensure a new figure is created each time "Generate" is pressed.
                                # The current logic creates a new fig_landscape_plot each time.
                                # To prevent potential memory leaks with many regenerations, explicit close is better.
                                # plt.close(st.session_state.landscape_plot_fig) # This would close it permanently.
                                # For now, let's rely on Streamlit's st.pyplot handling.
                                # If issues arise, we can add plt.clf() or manage figures more explicitly.
                else:
                    st.info("Run a simulation that converges to an equilibrium to perform profit landscape analysis.")

            with tab_monopoly:
                st.header("👑 Monopoly Analysis")
                st.write("""
                This section analyzes the optimal location and price for a single firm (monopolist)
                based on the theoretical model (Propositions \ref{prop:mono_price} and \ref{prop:mono_center} from the paper).
                The parameters used are those currently set in the sidebar.
                Note: The 'Number of firms' parameter in the sidebar is ignored for this specific analysis; it always assumes N=1.
                """)

                mono_grid_size = st.slider(
                    "Grid Size for Monopoly Calculations", 
                    min_value=20, max_value=100, value=30, step=5, 
                    key="mono_grid_size_slider",
                    help="Grid size for numerical integrations (e.g., total mass, weighted distance)."
                )
                
                landscape_grid_mono = st.slider(
                    "Grid Size for d_bar Landscape Visualization",
                    min_value=10, max_value=50, value=20, step=1,
                    key="mono_landscape_grid_slider",
                    help="Resolution for visualizing the weighted average distance landscape."
                )

                if 'monopoly_results' not in st.session_state:
                    st.session_state.monopoly_results = None
                if 'd_bar_landscape_fig' not in st.session_state:
                    st.session_state.d_bar_landscape_fig = None

                if st.button("Solve Monopoly Case", key="solve_monopoly_button"):
                    with st.spinner("Solving for monopolist's optimal strategy..."):
                        # Create a temporary model instance configured for a monopolist
                        # using current sidebar parameters for demand, cost, etc.
                        monopoly_model_instance = HotellingTwoDimensional(
                            n_firms=1, # Crucial for monopoly case
                            market_shape=market_shape,
                            alpha_demand=alpha_demand_input,
                            gamma_demand=gamma_demand_input,
                            t_transport_cost=t_transport_cost_input,
                            beta_logit=beta_logit_input, # Though beta_logit is for choice with >1 firms, it's part of init
                            d_type=d_type,
                            rho_type=rho_type,
                            density_params=density_centers_params
                            # c and max_price will use defaults
                        )
                        
                        # Solve the monopoly case
                        results_mono = monopoly_model_instance.solve_monopoly_case(grid_size=mono_grid_size)
                        st.session_state.monopoly_results = results_mono
                        
                        # Generate landscape for d_bar_rho
                        d_bar_matrix, x_coords_dbar, y_coords_dbar = \
                            monopoly_model_instance.get_weighted_average_distance_landscape(
                                landscape_grid_size=landscape_grid_mono, 
                                calculation_grid_size=mono_grid_size
                            )
                        
                        fig_dbar, ax_dbar = plt.subplots(figsize=(8,7))
                        if d_bar_matrix is not None and not np.all(np.isnan(d_bar_matrix)):
                            im_dbar = ax_dbar.imshow(d_bar_matrix.T, origin='lower', aspect='auto', cmap='coolwarm',
                                                 extent=[x_coords_dbar.min(), x_coords_dbar.max(), 
                                                         y_coords_dbar.min(), y_coords_dbar.max()],
                                                 interpolation='bilinear')
                            fig_dbar.colorbar(im_dbar, ax=ax_dbar, label=r"Weighted Avg. Distance ($\bar{d}_{\rho}$)")
                            
                            # Plot optimal location
                            opt_loc_mono = results_mono.get("optimal_location")
                            if opt_loc_mono is not None:
                                ax_dbar.scatter(opt_loc_mono[0], opt_loc_mono[1], s=150, c='black', marker='*', 
                                                label=f"Optimal Location ({opt_loc_mono[0]:.2f}, {opt_loc_mono[1]:.2f})",
                                                edgecolor='white')
                                ax_dbar.legend()
                        else:
                            ax_dbar.text(0.5, 0.5, "Could not compute d_bar landscape (e.g. M=0).", 
                                         ha='center', va='center', transform=ax_dbar.transAxes)

                        ax_dbar.set_title(r"Landscape of Weighted Average Distance $\bar{d}_{\rho}(x)$")
                        ax_dbar.set_xlabel("X-coordinate")
                        ax_dbar.set_ylabel("Y-coordinate")
                        st.session_state.d_bar_landscape_fig = fig_dbar
                        st.success("Monopoly analysis complete.")

                if st.session_state.monopoly_results:
                    res_m = st.session_state.monopoly_results
                    st.subheader("Monopolist's Optimal Strategy:")
                    
                    mono_col1, mono_col2, mono_col3 = st.columns(3)
                    mono_col1.metric("Optimal Location (x*)", f"({res_m['optimal_location'][0]:.3f}, {res_m['optimal_location'][1]:.3f})")
                    mono_col2.metric("Optimal Price (p*)", f"{res_m['optimal_price']:.3f}")
                    mono_col3.metric("Maximized Profit (Π*)", f"{res_m['maximized_profit']:.3f}")
                    
                    st.markdown(f"- Weighted Average Distance at x* ($\bar{{d}}_{{\rho}}(x^*)$): {res_m['weighted_avg_distance_at_opt_loc']:.4f}")
                    st.markdown(f"- Total Consumer Mass (M): {res_m['total_consumer_mass']:.4f}")
                    
                    profit_cond_term = res_m.get('profit_condition_term', 0)
                    st.markdown(f"- Profit Condition Term ($\propto \sqrt{{\Pi^*}}$): {profit_cond_term:.3f}")
                    if res_m.get('solution_validity_check', True):
                        st.success("Profitability condition (α - γc - γt d̄_ρ(x*) ≥ 0) is met.")
                    else:
                        st.warning("Profitability condition (α - γc - γt d̄_ρ(x*) ≥ 0) is NOT met. "
                                   "This suggests the monopolist might set price at marginal cost (c) or not operate if demand is non-positive at c.")

                if st.session_state.d_bar_landscape_fig:
                    st.subheader(r"Visualization of $\bar{d}_{\rho}(x)$ Landscape")
                    st.pyplot(st.session_state.d_bar_landscape_fig)
                    plt.clf() # Clear the figure after displaying to free memory

    else:
        st.info("Click 'Run Simulation' in the sidebar to start.")

if __name__ == "__main__":
    main()
