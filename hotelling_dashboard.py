import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from hotelling_model import HotellingTwoDimensional  # Assuming your code is in hotelling_model.py

def main():
    st.set_page_config(layout="wide") # Use full width
    st.title("2D Hotelling Model Simulation Dashboard")
    
    # ===== Sidebar Controls =====
    st.sidebar.header("Simulation Parameters")
    
    # Model parameters
    n_firms = st.sidebar.slider("Number of firms", 2, 5, 2)
    market_shape = (st.sidebar.number_input("Market Width", 1.0, 5.0, 1.0),
                    st.sidebar.number_input("Market Height", 1.0, 5.0, 1.0))
    
    col1, col2 = st.sidebar.columns(2)
    beta = col1.number_input("Beta (price sensitivity)", 0.1, 5.0, 0.5)
    eta = col2.number_input("Eta (elasticity)", 1.1, 5.0, 3.0)
    
    # Distance/density controls
    d_type = st.sidebar.selectbox("Distance Type", 
                                 ["euclidean", "manhattan", "quadratic"])
    rho_type = st.sidebar.selectbox("Density Type",
                                   ["uniform", "linear", "gaussian", "sine", "multi_gaussian"])
    
    density_centers_params = None # Initialize
    if rho_type == 'multi_gaussian':
        st.sidebar.subheader("Multi-Gaussian Density Parameters")
        # Ensure market_shape is defined before this widget if its limits depend on it.
        # market_shape is defined above, so it's fine.
        num_foci = st.sidebar.number_input("Number of Gaussian Foci", 
                                           min_value=1, max_value=5, value=2, step=1, key="num_foci")
        
        density_centers_params = []
        for i in range(int(num_foci)):
            st.sidebar.markdown(f"**Focus {i+1}**")
            foci_col1, foci_col2 = st.sidebar.columns(2)
            # Default center positions are staggered for better initial visualization
            default_center_x = market_shape[0] * ( (i+1) / (int(num_foci)+1) )
            default_center_y = market_shape[1] * ( (i+1) / (int(num_foci)+1) )

            center_x = foci_col1.number_input(f"Center X ({i+1})", 
                                              min_value=0.0, max_value=float(market_shape[0]), 
                                              value=float(default_center_x), 
                                              step=0.1, key=f"mg_cx_{i}")
            center_y = foci_col2.number_input(f"Center Y ({i+1})", 
                                              min_value=0.0, max_value=float(market_shape[1]), 
                                              value=float(default_center_y), 
                                              step=0.1, key=f"mg_cy_{i}")
            
            strength_col, sigma_col = st.sidebar.columns(2)
            strength = strength_col.number_input(f"Strength ({i+1})", 
                                                 min_value=0.1, max_value=10.0, 
                                                 value=1.0, step=0.1, key=f"mg_str_{i}")
            # Default sigma related to np.sqrt(0.1) which is approx 0.316
            sigma = sigma_col.number_input(f"Sigma (spread) ({i+1})", 
                                           min_value=0.01, max_value=float(max(market_shape)/2), 
                                           value=0.3, step=0.01, key=f"mg_sig_{i}")
            
            density_centers_params.append({'center': (center_x, center_y), 'strength': strength, 'sigma': sigma})

    # Simulation controls
    max_iter = st.sidebar.slider("Max Iterations", 10, 200, 50)
    grid_size = st.sidebar.slider("Grid Size (for simulation)", 10, 100, 30, help="Grid size for demand/profit calculation during optimization.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Verification Options")
    run_uniqueness_check = st.sidebar.checkbox("Run Equilibrium Uniqueness Check", value=False, help="Empirically check uniqueness by running simulation from multiple random start points.")
    uniqueness_attempts = 1
    if run_uniqueness_check:
        uniqueness_attempts = st.sidebar.number_input("Number of Uniqueness Attempts", min_value=2, max_value=10, value=3, step=1)
    
    # ===== Main Panel =====
    if st.sidebar.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            # Initialize model
            model = HotellingTwoDimensional(
                n_firms=n_firms,
                market_shape=market_shape,
                beta=beta,
                eta=eta,
                d_type=d_type,
                rho_type=rho_type,
                density_params=density_centers_params
            )
            
            # Run simulation
            converged = model.find_equilibrium(
                max_iterations=max_iter,
                grid_size=grid_size,
                verbose=False
            )
            
            # Create tabs for organizing output
            tab_viz_overview, tab_detailed_props, tab_theory = st.tabs([
                "📈 Visualizations & Overview", 
                "📊 Detailed Equilibrium Properties", 
                "📝 Theoretical Verification"
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
                        current_profits = model.total_profit(grid_size=grid_size)
                        for i, profit_val in enumerate(current_profits):
                            st.write(f"Firm {i+1}: {profit_val:.2f}")

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

            with tab_detailed_props:
                st.header("Detailed Equilibrium Properties")

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
                profits = model.total_profit(grid_size=grid_size) # Recalculate or use from history if available
                profits_data = [{"Firm": f"Firm {i+1}", "Profit": profits[i]} for i in range(model.n_firms)]
                st.table(profits_data)
                profit_stats_cols = st.columns(3)
                profit_stats_cols[0].metric("Total Profit (All Firms)", f"{np.sum(profits):.2f}")
                profit_stats_cols[1].metric("Average Profit per Firm", f"{np.mean(profits):.2f}")
                profit_stats_cols[2].metric("Profit Std Dev", f"{np.std(profits):.2f}")

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


            with tab_theory:
                st.header("Theoretical Verification & Assumptions")
                st.subheader("Equilibrium Existence")
                if converged:
                    st.success("Equilibrium found (simulation converged).")
                else:
                    st.error("Equilibrium not found (simulation did not converge).")

                st.subheader("Assumption Verification (based on paper definitions)")
                
                # Run assumption verification
                # Use a potentially different grid_size for verification if needed, or same as simulation
                # For now, using a fixed moderate grid_size for assumption checks to balance speed and accuracy.
                # The paper's omega_ij involves an integral, so grid_size matters.
                assumption_grid_size = st.slider("Grid Size for Assumption Verification", 10, 50, 20, help="Grid size for numerical integration in assumption checks (e.g., omega_ij).")
                assumptions_results = model.verify_assumptions(grid_size=assumption_grid_size)

                # Assumption 1: Regularity
                st.markdown("**Assumption 1: Regularity**")
                ass1_col1, ass1_col2 = st.columns(2)
                with ass1_col1:
                    st.write(f"Overall Satisfied: {assumptions_results['assumption1_satisfied']}")
                
                # 1.1 Cost function and density
                st.markdown("___1.1 Cost function continuous & convex, rho log-concave:___")
                ass1_1_col1, ass1_1_col2 = st.columns(2)
                with ass1_1_col1:
                    st.write(f"- Cost function continuous: {'True (by design)'}")
                    st.write(f"- Cost function convex: {assumptions_results['reg_cost_convex']} (for d_type='{model.d_type}')")
                with ass1_1_col2:
                    st.warning("Log-concavity of rho(s) is stated in paper but not automatically checked by this model's `verify_assumptions` function.")

                # 1.2 Compact strategy spaces
                st.markdown("___1.2 Strategy spaces compact:___")
                st.write(f"- {'True (by design)'}")
                
                # 1.3 Elasticity eta > 1
                st.markdown("___1.3 Elasticity eta > 1:___")
                ass1_3_col1, ass1_3_col2 = st.columns(2)
                with ass1_3_col1:
                    st.write(f"- Condition (eta > 1): {assumptions_results['reg_elasticity']}")
                with ass1_3_col2:
                    st.write(f"Actual eta: {model.eta}")

                st.markdown("---")
                # Assumption 2: Uniqueness
                st.markdown("**Assumption 2: Uniqueness**")
                ass2_col1, ass2_col2 = st.columns(2)
                with ass2_col1:
                    st.write(f"Overall Satisfied: {assumptions_results['assumption2_satisfied']}")

                # 2.1 d(s, x_i) mu-strongly convex
                st.markdown("___2.1 d(s, x_i) is mu-strongly convex:___")
                ass2_1_col1, ass2_1_col2 = st.columns(2)
                with ass2_1_col1:
                    st.write(f"- Condition: {assumptions_results['uniq_strongly_convex']} (for d_type='{model.d_type}')")
                with ass2_1_col2:
                    if model.d_type == 'quadratic':
                        st.write(f"(mu = {model.mu}, typically 2 for quadratic distance)")
                    else:
                        st.write(f"(mu = {model.mu}, ensure this matches d_type properties)")


                # 2.2 Elasticity condition
                st.markdown(r"___2.2 Elasticity condition ($\eta > 1 + \beta \frac{\sum_{j \neq i} \omega_{ij}}{\omega_{ii}} + \frac{\beta^2}{\mu}$):___")
                ass2_2_col1, ass2_2_col2, ass2_2_col3 = st.columns(3)
                with ass2_2_col1:
                    st.write(f"- Condition satisfied: {assumptions_results['uniq_elasticity']}")
                with ass2_2_col2:
                    st.write(f"Actual eta: {model.eta:.3f}")
                with ass2_2_col3:
                    st.write(f"Required eta > {assumptions_results['min_eta_required']:.3f}")
                st.caption(f"(Using max_omega_ratio: {assumptions_results['max_omega_ratio']:.3f}, beta: {model.beta:.2f}, mu: {model.mu:.2f})")


                # 2.3 Beta condition
                st.markdown(r"___2.3 Beta condition ($\beta \bar{d} < 1$):___")
                max_d = model.max_transport_cost()
                beta_d_bar = model.beta * max_d
                ass2_3_col1, ass2_3_col2, ass2_3_col3 = st.columns(3)
                with ass2_3_col1:
                    st.write(f"- Condition satisfied: {assumptions_results['uniq_beta']}")
                with ass2_3_col2:
                    st.write(f"Actual beta * d_bar: {beta_d_bar:.3f}")
                with ass2_3_col3:
                    st.write(f"(beta: {model.beta:.2f}, d_bar: {max_d:.3f})")
                
                # Empirical Uniqueness Check
                if run_uniqueness_check:
                    st.markdown("---")
                    st.subheader("Empirical Equilibrium Uniqueness Test")
                    with st.spinner(f"Running uniqueness check with {uniqueness_attempts} attempts..."):
                        # It's important that check_equilibrium_uniqueness uses appropriate grid_size for its internal find_equilibrium calls.
                        # The model's check_equilibrium_uniqueness has a grid_size parameter.
                        uniqueness_results = model.check_equilibrium_uniqueness(
                            n_attempts=uniqueness_attempts, 
                            max_iterations=max_iter, # Use same max_iter as main sim for consistency
                            tolerance=1e-4, # Standard tolerance
                            grid_size=grid_size, # Use main simulation grid_size
                            verbose=False # Keep dashboard clean
                        )
                    st.write(f"- Empirically Unique: {uniqueness_results['unique']}")
                    st.write(f"- Number of distinct equilibria found: {uniqueness_results['n_equilibria_found']}")
                    if uniqueness_results['n_equilibria_found'] > 0 :
                        st.write(f"- Max price difference between equilibria: {uniqueness_results['max_price_diff']:.4f}")
                        st.write(f"- Max location difference between equilibria: {uniqueness_results['max_location_diff']:.4f}")
                    if uniqueness_results.get("reason"):
                         st.warning(f"Note: {uniqueness_results['reason']}")


if __name__ == "__main__":
    main()
