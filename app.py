import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import gc
import psutil

# Page config
st.set_page_config(
    page_title="🌾 Agriculture Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Cache model loading - loads ONCE and reuses
@st.cache_resource
def load_model():
    """Load the production yield prediction model"""
    model_path = Path('artifacts_yield/yield_pipeline_forward_chain_STACKED_PRODUCTION.pkl')
    
    if not model_path.exists():
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    
    try:
        with st.spinner("Loading model..."):
            model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.exception(e)
        st.stop()

def get_ram_usage():
    """Get current RAM usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

# Load model once at startup
model = load_model()

# Title
st.title("🌾 Agricultural Yield Prediction System")
st.markdown("""
Predict crop yield using our stacked ensemble ML model trained on historical agricultural data.
""")

# Sidebar
with st.sidebar:
    st.header("📊 System Info")
    
    # RAM monitoring
    ram = get_ram_usage()
    ram_percent = (ram / 1024) * 100
    
    st.metric("RAM Usage", f"{ram:.0f} MB")
    st.progress(min(ram_percent / 100, 1.0))
    
    if ram > 900:
        st.error("⚠️ High memory usage!")
    elif ram > 700:
        st.warning("⚠️ Moderate memory usage")
    else:
        st.success("✅ Memory usage normal")
    
    st.divider()
    
    st.markdown("### About")
    st.info("""
    **Model:** Forward Chain Stacked Ensemble
    
    **Features:**
    - XGBoost base models
    - Stacking meta-learner
    - Production-ready pipeline
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Input Features")
    
    with st.form("prediction_form"):
        st.markdown("##### Location Details")
        col_loc1, col_loc2 = st.columns(2)
        with col_loc1:
            state = st.text_input("State *", placeholder="e.g., Karnataka")
        with col_loc2:
            district = st.text_input("District *", placeholder="e.g., Bangalore")
        
        st.markdown("##### Crop Details")
        col_crop1, col_crop2 = st.columns(2)
        with col_crop1:
            crop = st.text_input("Crop *", placeholder="e.g., Rice")
        with col_crop2:
            season = st.selectbox("Season *", ["Kharif", "Rabi", "Whole Year", "Summer", "Autumn", "Winter"])
        
        st.markdown("##### Temporal Information")
        year = st.number_input("Crop Year *", min_value=1997, max_value=2030, value=2024, step=1)
        
        st.markdown("---")
        st.caption("* Required fields")
        
        submitted = st.form_submit_button("🚀 Predict Yield", use_container_width=True, type="primary")

with col2:
    st.header("📈 Prediction Result")
    
    if not submitted:
        st.info("👈 Fill in the form and click **Predict Yield** to see results")
    
    result_placeholder = st.empty()

# Prediction logic
if submitted:
    # Validate inputs
    if not all([state, district, crop]):
        st.error("❌ Please fill in all required fields (*)")
    else:
        try:
            # Prepare input data (adjust column names to match your model's training data)
            input_data = pd.DataFrame({
                'State_Name': [state],
                'District_Name': [district],
                'Crop': [crop],
                'Season': [season],
                'Crop_Year': [year]
            })
            
            # Make prediction
            with st.spinner("Making prediction..."):
                prediction = model.predict(input_data)[0]
            
            # Display result in the right column
            with result_placeholder.container():
                st.success("✅ Prediction Complete!")
                
                # Main prediction metric
                st.metric(
                    label="Predicted Yield",
                    value=f"{prediction:,.2f}",
                    help="Predicted yield value based on input parameters"
                )
                
                st.divider()
                
                # Show input summary
                with st.expander("📋 Input Summary", expanded=True):
                    st.markdown(f"""
                    **Location:**  
                    {state}, {district}
                    
                    **Crop Information:**  
                    {crop} ({season})
                    
                    **Year:**  
                    {year}
                    """)
                
                # Model confidence (if available)
                st.caption("Prediction generated by Stacked Ensemble Model")
            
            # Clean up (though not strictly necessary with single model)
            gc.collect()
            
            # Show updated RAM
            new_ram = get_ram_usage()
            with st.sidebar:
                st.caption(f"RAM after prediction: {new_ram:.0f} MB")
                
        except Exception as e:
            st.error("❌ Prediction failed!")
            with st.expander("🔍 Error Details"):
                st.exception(e)
                st.markdown("""
                **Common issues:**
                - Check if input values match training data format
                - Verify model expects these exact column names
                - Ensure data types are correct
                """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🌾 Agricultural Yield Prediction System</p>
    <p style='font-size: 0.9em;'>Built with Streamlit | Powered by Stacked Ensemble ML</p>
</div>
""", unsafe_allow_html=True)
