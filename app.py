import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import gc
import psutil

# Page config
st.set_page_config(
    page_title="🌾 Agriculture Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# Cache model loading
@st.cache_resource
def load_model(model_type):
    """Load specific model based on prediction type"""
    model_files = {
        'Area': 'artifacts_yield/area_xgb_full_pipeline.joblib',
        'Production': 'artifacts_yield/production_xgb_full_pipeline.joblib',
        'Yield': 'artifacts_yield/yieldcalc_xgb_full_pipeline.joblib'
    }
    
    model_path = Path(model_files[model_type])
    if not model_path.exists():
        st.error(f"❌ Model file not found: {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        st.success(f"✅ {model_type} model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def get_ram_usage():
    """Get current RAM usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

# Title
st.title("🌾 Agricultural Yield Prediction System")
st.markdown("Predict agricultural metrics using ML models trained on historical data")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Selection")
    prediction_type = st.selectbox(
        "Choose Prediction Type",
        ["Area", "Production", "Yield"]
    )
    
    st.divider()
    ram = get_ram_usage()
    st.metric("RAM Usage", f"{ram:.0f} MB")
    
    if ram > 800:
        st.warning("⚠️ High memory usage")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Input Features")
    
    with st.form("prediction_form"):
        state = st.text_input("State *", placeholder="e.g., Karnataka")
        district = st.text_input("District *", placeholder="e.g., Bangalore")
        crop = st.text_input("Crop *", placeholder="e.g., Rice")
        season = st.selectbox("Season *", ["Kharif", "Rabi", "Whole Year", "Summer"])
        year = st.number_input("Year *", min_value=2000, max_value=2030, value=2024)
        
        col_a, col_b = st.columns(2)
        with col_a:
            rainfall = st.number_input("Rainfall (mm)", value=800.0)
            temp = st.number_input("Temperature (°C)", value=25.0)
        with col_b:
            humidity = st.number_input("Humidity (%)", value=60.0)
        
        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

with col2:
    st.header("📈 Prediction Result")
    result_placeholder = st.empty()

if submitted:
    if not all([state, district, crop]):
        st.error("❌ Please fill all required fields (*)")
    else:
        with st.spinner(f"Loading {prediction_type} model..."):
            model = load_model(prediction_type)
            
            if model is not None:
                try:
                    # Create input dataframe (adjust columns based on your model)
                    input_data = pd.DataFrame({
                        'State_Name': [state],
                        'District_Name': [district],
                        'Crop': [crop],
                        'Season': [season],
                        'Crop_Year': [year]
                    })
                    
                    # Make prediction
                    prediction = model.predict(input_data)[0]
                    
                    # Display result
                    with result_placeholder.container():
                        st.metric(
                            label=f"Predicted {prediction_type}",
                            value=f"{prediction:,.2f}"
                        )
                        st.success("✅ Prediction complete!")
                        
                        with st.expander("📋 Input Summary"):
                            st.write(f"**Location:** {state}, {district}")
                            st.write(f"**Crop:** {crop}")
                            st.write(f"**Season:** {season}")
                            st.write(f"**Year:** {year}")
                    
                    # Cleanup
                    del model
                    gc.collect()
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    with st.expander("Error Details"):
                        st.exception(e)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit | Powered by XGBoost ML Models</p>
</div>
""", unsafe_allow_html=True)
