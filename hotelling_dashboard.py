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
            
            # Display results
            with st.expander("Simulation Results", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Final Locations")
                    for i, loc in enumerate(model.locations):
                        st.write(f"Firm {i+1}: ({loc[0]:.2f}, {loc[1]:.2f})")
                        
                with col2:
                    st.write("### Final Prices")
                    for i, price in enumerate(model.prices):
                        st.write(f"Firm {i+1}: {price:.2f}")
            
            # Visualizations
            with st.expander("Market Visualization", expanded=True):
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    st.write("#### Market Segmentation")
                    fig_seg = model.visualize(show_segmentation=True, show_density=False, grid_size=grid_size) # Pass grid_size
                    st.pyplot(fig_seg)
                    plt.clf() # Clear figure after displaying
                
                with viz_col2:
                    st.write("#### Population Density")
                    fig_den = model.visualize(show_segmentation=False, show_density=True, grid_size=grid_size) # Pass grid_size
                    st.pyplot(fig_den)
                    plt.clf() # Clear figure after displaying
            
            # Convergence plots
            with st.expander("Convergence Metrics", expanded=True):
                fig_conv = model.plot_convergence()
                st.pyplot(fig_conv)
                # plt.clf() might not be necessary if fig_conv is a new figure each time
                # and not relying on plt.gcf(). Let's keep it for now to be safe,
                # or remove if plot_convergence always creates a new fig.
                # Given plot_convergence now returns fig, it's safer to clear the specific figure
                # or rely on Streamlit to handle it. For now, plt.clf() is a broad clear.
                # If model.plot_convergence() always returns a new fig, plt.clf() is fine.
                plt.clf()
            
            # Theoretical Verification & Assumptions
            with st.expander("Theoretical Verification & Assumptions", expanded=True):
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
