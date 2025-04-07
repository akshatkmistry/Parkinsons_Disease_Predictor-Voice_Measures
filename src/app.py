import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set up the page
st.set_page_config(
    page_title="Parkinson's Disease Detector",
    page_icon="🧠",
    layout="wide"
)
# Load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('./models/random_forest_model.pkl')
    scaler = joblib.load('./models/scaler.pkl')
    return model, scaler

# Function to make prediction
def predict_parkinsons(features, model, scaler):
    # Convert features to dataframe
    df = pd.DataFrame([features])
    
    # Scale the features
    scaled_features = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]
    
    return prediction, probability

# Sample data function
def load_sample_data(condition):
    if condition == "parkinsons":
        return {
            'MDVP:Fo(Hz)': 174.2,
            'MDVP:Jitter(%)': 0.00662,
            'MDVP:Shimmer': 0.04374,
            'NHR': 0.02182,
            'HNR': 21.033,
            'RPDE': 0.5547,
            'DFA': 0.7519,
            'spread1': -6.896,
            'spread2': 0.2211,
            'PPE': 0.3146
        }
    else:  # healthy
        return {
            'MDVP:Fo(Hz)': 120.6,
            'MDVP:Jitter(%)': 0.00339,
            'MDVP:Shimmer': 0.02281,
            'NHR': 0.00913,
            'HNR': 24.889,
            'RPDE': 0.3838,
            'DFA': 0.6016,
            'spread1': -4.429,
            'spread2': 0.1228,
            'PPE': 0.1145
        }


# Main app
def main():
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Introduction section with description
    st.title("Parkinson's Disease Detection")
    
    
    # Model performance metrics
    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Accuracy", value="94.87%")
    with col2:
        st.metric(label="Precision (Class 1)", value="94.00%")
    with col3:
        st.metric(label="Recall (Class 1)", value="100.00%")
    with col4:
        st.metric(label="F1-Score (Class 1)", value="96.97%")
    
    
    
    
    # Sample data buttons
    st.subheader("Input Voice Measurements")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Parkinson's Sample Data"):
            st.session_state['sample_data'] = load_sample_data("parkinsons")
    with col2:
        if st.button("Load Healthy Sample Data"):
            st.session_state['sample_data'] = load_sample_data("healthy")
    
    # Create a form for user input
    with st.form("prediction_form"):
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        # Get sample data if available
        sample_data = st.session_state.get('sample_data', {})
        
        # Add input fields for the 10 features
        with col1:
            fo = st.number_input('MDVP:Fo(Hz) - Average vocal frequency', 
                                min_value=80.0, max_value=260.0, 
                                value=sample_data.get('MDVP:Fo(Hz)', 120.0))
            
            jitter = st.number_input('MDVP:Jitter(%) - Frequency variation', 
                                   min_value=0.0, max_value=1.0, 
                                   value=sample_data.get('MDVP:Jitter(%)', 0.006), 
                                   format="%.6f")
            
            shimmer = st.number_input('MDVP:Shimmer - Amplitude variation', 
                                    min_value=0.0, max_value=0.2, 
                                    value=sample_data.get('MDVP:Shimmer', 0.03), 
                                    format="%.6f")
            
            nhr = st.number_input('NHR - Noise-to-harmonics ratio', 
                                min_value=0.0, max_value=0.5, 
                                value=sample_data.get('NHR', 0.02), 
                                format="%.6f")
            
            hnr = st.number_input('HNR - Harmonics-to-noise ratio', 
                                min_value=8.0, max_value=30.0, 
                                value=sample_data.get('HNR', 20.0))
        
        with col2:
            rpde = st.number_input('RPDE - Nonlinear measure', 
                                 min_value=0.25, max_value=0.85, 
                                 value=sample_data.get('RPDE', 0.5))
            
            dfa = st.number_input('DFA - Signal fractal scaling exponent', 
                                min_value=0.5, max_value=0.9, 
                                value=sample_data.get('DFA', 0.7))
            
            spread1 = st.number_input('Spread1 - Nonlinear measure of frequency variation', 
                                    min_value=-8.0, max_value=0.0, 
                                    value=sample_data.get('spread1', -4.0))
            
            spread2 = st.number_input('Spread2 - Nonlinear measure of frequency variation', 
                                    min_value=0.0, max_value=4.0, 
                                    value=sample_data.get('spread2', 0.25))
            
            ppe = st.number_input('PPE - Nonlinear measure of frequency variation', 
                                min_value=0.0, max_value=0.5, 
                                value=sample_data.get('PPE', 0.2))
        
        # Create a dictionary with the features
        features = {
            'MDVP:Fo(Hz)': fo,
            'MDVP:Jitter(%)': jitter,
            'MDVP:Shimmer': shimmer,
            'NHR': nhr,
            'HNR': hnr,
            'RPDE': rpde,
            'DFA': dfa,
            'spread1': spread1,
            'spread2': spread2,
            'PPE': ppe
        }
        
        # Submit button
        submit_button = st.form_submit_button("Predict")
    
    # Make prediction when the form is submitted
    if submit_button:
        prediction, probability = predict_parkinsons(features, model, scaler)
        
        # Display the prediction
        st.subheader("Prediction Result")
        
        # Create columns for the result display
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 1:
                st.error("**Parkinson's Disease Detected**")
            else:
                st.success("**Healthy - No Parkinson's Disease Detected**")
        
        with col2:
            st.write(f"Probability of Parkinson's Disease: **{probability:.2%}**")
            
            # Create a simple gauge chart for the probability
            progress_html = f"""
            <div style="border-radius: 10px; border: 2px solid {'red' if probability > 0.5 else 'green'}; padding: 5px;">
                <div style="background-color: {'red' if probability > 0.5 else 'green'}; 
                            width: {probability * 100}%; 
                            height: 20px; 
                            border-radius: 5px;">
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
            
        # Feature importance information
        with st.expander("What do these measurements mean?"):
            st.markdown("""
            - **MDVP:Fo(Hz)**: Average vocal fundamental frequency
            - **MDVP:Jitter(%)**: Variation in fundamental frequency (higher in PD patients)
            - **MDVP:Shimmer**: Variation in amplitude (higher in PD patients)
            - **NHR**: Noise-to-harmonics ratio (higher in PD patients)
            - **HNR**: Harmonics-to-noise ratio (lower in PD patients)
            - **RPDE**: Nonlinear dynamical complexity measure (higher in PD patients)
            - **DFA**: Signal fractal scaling exponent (higher in PD patients)
            - **spread1 & spread2**: Nonlinear measures of frequency variation
            - **PPE**: Nonlinear measure of fundamental frequency variation (higher in PD patients)
            """)

    # Future Enhancement Section
    st.subheader("Future Enhancements")
    st.info("""
    **Coming Soon: Direct Audio Analysis**
    
    In future versions, we plan to implement direct audio file processing. Users will be able to:
    - Upload voice recordings directly (.wav, .mp3 files)
    - Have the system automatically extract all required voice features
    - Get instant predictions without manual feature entry
    
    This enhancement will make the tool more accessible and user-friendly for both clinical and home use,
    potentially enabling earlier detection through regular voice monitoring.
    """)
    # Add a disclaimer at the bottom
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This tool is intended for educational purposes only and should not be used for medical diagnosis. 
    Please consult a healthcare professional for proper medical advice.
    """)

if __name__ == "__main__":
    main()