import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from hotelling_model import HotellingTwoDimensional  # Assuming your code is in hotelling_model.py

def main():
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
                                   ["uniform", "linear", "gaussian", "sine"])
    
    # Simulation controls
    max_iter = st.sidebar.slider("Max Iterations", 10, 200, 50)
    grid_size = st.sidebar.slider("Grid Size", 10, 100, 30)
    
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
                rho_type=rho_type
            )
            
            # Run simulation
            converged = model.find_equilibrium(
                max_iterations=max_iter,
                grid_size=grid_size,
                verbose=False
            )
            
            # Display results
            st.subheader("Simulation Results")
            
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
            st.write("### Market Visualization")
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.write("Market Segmentation")
                fig = model.visualize(show_segmentation=True, show_density=False)
                st.pyplot(plt.gcf())
                plt.clf()
            
            with viz_col2:
                st.write("Population Density")
                fig = model.visualize(show_segmentation=False, show_density=True)
                st.pyplot(plt.gcf())
                plt.clf()
            
            # Convergence plots
            st.write("### Convergence Metrics")
            model.plot_convergence()
            st.pyplot(plt.gcf())
            plt.clf()

if __name__ == "__main__":
    main()
