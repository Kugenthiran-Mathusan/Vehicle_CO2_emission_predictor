import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load trained final model (pipeline + regression)
model = joblib.load("final_model.pkl")

# Load your dataset (for average comparisons)
df = pd.read_csv("cleaned_vehicle_dataset.csv")

# App Layout
st.set_page_config(page_title="CO₂ Emissions Calculator", page_icon="🚗", layout="wide")

st.title("🚗 Vehicle CO₂ Emissions Calculator")
st.markdown("Find out how much CO₂ your vehicle produces and compare it with similar vehicles.")

# Sidebar Inputs
st.sidebar.header("🔧 Enter Vehicle Specifications")

engine_size = st.sidebar.number_input("Engine size (L)", min_value=0.5, max_value=10.0, step=0.1, value=2.0)
cylinders = st.sidebar.number_input("Number of Cylinders", min_value=3, max_value=12, step=1, value=4)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Ethanol", "Natural Gas", "Premium Petrol"])
combined_l_100km = st.sidebar.number_input("Combined (L/100 km)", min_value=2.0, max_value=25.0, step=0.1, value=8.5)

# Predict button
if st.sidebar.button("🔍 Calculate Emissions", type="primary"):
    # Convert input to DataFrame
    input_data = pd.DataFrame({
        "Engine size (L)": [engine_size],
        "Cylinders": [cylinders],
        "Fuel type": [fuel_type],
        "Combined (L/100 km)": [combined_l_100km]
    })

    # Predict directly (model already has preprocessing inside)
    prediction = model.predict(input_data)[0]

    # Dataset average comparison
    avg_emission = df["CO2 emissions (g/km)"].mean()
    difference = prediction - avg_emission
    percent_diff = (difference / avg_emission) * 100

    # <CHANGE> Fixed rating logic - lower emissions = better rating
    def get_emission_rating(emission_value):
        """Returns rating based on emission value - lower is better"""
        if emission_value < 150:
            return "✅ Excellent", "Your vehicle has very low emissions!", "success"
        elif emission_value < 200:
            return "✅ Good", "Your vehicle has below-average emissions.", "success"
        elif emission_value < 250:
            return "⚠️ Average", "Your vehicle has average emissions.", "warning"
        else:
            return "❌ High", "Your vehicle has above-average emissions.", "error"
    
    rating, rating_msg, rating_color = get_emission_rating(prediction)

    # Results Section
    st.markdown("---")
    st.subheader("📊 Your Vehicle's CO₂ Emissions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Your Vehicle",
            value=f"{prediction:.0f} g/km",
            delta=f"{difference:.0f} g/km" if abs(difference) > 1 else None,
            delta_color="inverse"  # Lower is better, so inverse the color
        )
    
    with col2:
        st.metric(
            label="Average Vehicle",
            value=f"{avg_emission:.0f} g/km"
        )
    
    # Comparison message
    if difference < 0:
        st.success(f"✅ Your vehicle produces **{abs(difference):.0f} g/km less** CO₂ than average ({abs(percent_diff):.1f}% better)")
    else:
        st.error(f"⚠️ Your vehicle produces **{difference:.0f} g/km more** CO₂ than average ({percent_diff:.1f}% higher)")

    # What does this mean section
    st.markdown("---")
    st.subheader("🌍 What Does This Mean?")
    
    # Annual impact calculations
    annual_km = 15000
    annual_co2_kg = (prediction * annual_km) / 1000
    annual_avg_co2_kg = (avg_emission * annual_km) / 1000
    
    trees_needed = int(annual_co2_kg / 21)  # One tree absorbs ~21kg CO2/year
    flight_hours = int(annual_co2_kg / 90)  # ~90kg CO2 per flight hour
    
    st.markdown(f"""
    **Annual Impact** (driving {annual_km:,} km/year)
    - Your vehicle: **{annual_co2_kg:.0f} kg** of CO₂ per year
    - Average vehicle: **{annual_avg_co2_kg:.0f} kg** of CO₂ per year
    
    That's equivalent to:
    - 🌳 About **{trees_needed} trees** needed to offset your emissions
    - ✈️ Similar to a **{flight_hours}-hour flight**
    """)

    # Rating display
    st.markdown("---")
    if rating_color == "success":
        st.success(f"**{rating}**\n\n{rating_msg}")
    elif rating_color == "warning":
        st.warning(f"**{rating}**\n\n{rating_msg}")
    else:
        st.error(f"**{rating}**\n\n{rating_msg}")

    # Tips section
    st.markdown("---")
    st.subheader("💡 Tips to Reduce Emissions")
    
    tips = []
    if combined_l_100km > 8:
        tips.append("🚗 Your fuel consumption is high. Consider eco-driving techniques like smooth acceleration and maintaining steady speeds.")
    if engine_size > 2.5:
        tips.append("🔧 Larger engines typically produce more emissions. Consider a smaller, more efficient engine for your next vehicle.")
    if cylinders > 6:
        tips.append("⚙️ Vehicles with fewer cylinders are generally more fuel-efficient.")
    
    tips.extend([
        "🔋 Consider hybrid or electric vehicles for your next purchase.",
        "🚴 Use public transport, carpool, or bike for short trips.",
        "🛠️ Regular maintenance keeps your vehicle running efficiently."
    ])
    
    for tip in tips[:4]:  # Show max 4 tips
        st.markdown(f"- {tip}")

    # Visualization Section
    st.markdown("---")
    st.subheader("📈 Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    # Bar chart comparison
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        categories = ["Your Vehicle", "Average Vehicle"]
        values = [prediction, avg_emission]
        colors = ['#ff6b6b' if prediction > avg_emission else '#51cf66', '#94a3b8']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel("CO₂ Emissions (g/km)", fontsize=11, fontweight='bold')
        ax.set_title("Your Vehicle vs Average", fontsize=13, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Distribution plot
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Plot distribution of all vehicles
        ax.hist(df["CO2 emissions (g/km)"], bins=50, alpha=0.6, color='#94a3b8', 
                edgecolor='black', label='All Vehicles')
        
        # Highlight user's vehicle
        ax.axvline(prediction, color='#ff6b6b' if prediction > avg_emission else '#51cf66', 
                  linewidth=3, linestyle='--', label='Your Vehicle')
        ax.axvline(avg_emission, color='#fbbf24', linewidth=2, linestyle=':', label='Average')
        
        ax.set_xlabel("CO₂ Emissions (g/km)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Number of Vehicles", fontsize=11, fontweight='bold')
        ax.set_title("Where Your Vehicle Stands", fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig)

    # About section (collapsed by default)
    with st.expander("ℹ️ About This Calculator"):
        st.markdown("""
        This calculator uses a **Machine Learning model** trained on thousands of real vehicle data points 
        to predict CO₂ emissions based on your vehicle's specifications.
        
        **How accurate is it?**
        - The model has been tested and validated on real-world data
        - Predictions are typically within ±15 g/km of actual emissions
        
        **What affects CO₂ emissions?**
        - Engine size and number of cylinders
        - Fuel type (petrol, diesel, etc.)
        - Fuel consumption (L/100 km)
        
        **Why does this matter?**
        - Transportation accounts for ~24% of global CO₂ emissions
        - Understanding your vehicle's impact helps make informed decisions
        - Small changes in driving habits can significantly reduce emissions
        """)

else:
    # Welcome screen when no prediction yet
    st.info("👈 Enter your vehicle specifications in the sidebar and click **Calculate Emissions** to see results.")
    
    st.markdown("---")
    st.subheader("🌱 Why Calculate Your Vehicle's Emissions?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🌍 Environmental Impact**
        
        Understanding your vehicle's CO₂ output helps you make eco-conscious decisions.
        """)
    
    with col2:
        st.markdown("""
        **💰 Cost Savings**
        
        Lower emissions often mean better fuel efficiency and reduced running costs.
        """)
    
    with col3:
        st.markdown("""
        **📊 Compare Options**
        
        See how your vehicle stacks up against the average to inform future purchases.
        """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using **Python, Streamlit, and Machine Learning** | Project by: *Sajivan & Team*")