# File: ed_optimizer_app.py
# FINAL POLISHED VERSION: Optimized for white background, high contrast, and visualization clarity.

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import simpy
import os
import plotly.express as px
import plotly.graph_objects as go

# Added imports for ML metrics
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score, mean_squared_error, \
    precision_score, recall_score, f1_score, r2_score

# --- CONFIGURATION (File Paths) ---
DATA_FILE_PATH = 'sy.csv'
REG_MODEL_PATH = 'linear_regression_model.pkl' # Predicts service time (duration)
CLS_MODEL_PATH = 'best_classification_model.pkl' # Predicts admission probability

# --- UNIVERSAL STYLING CONSTANTS ---
PRIMARY_COLOR = '#4361ee'  # Modern blue
SECONDARY_COLOR = '#7209b7'  # Modern purple
ACCENT_COLOR = '#4cc9f0'  # Light blue accent
TEXT_COLOR = '#000000'  # Pure black
BG_COLOR = '#f8f9fa'  # Light gray background
CARD_COLOR = '#ffffff'  # White for cards

# --- Custom Styling for White Background and Interactive Look ---
st.markdown(f"""
    <style>
    /* Sidebar text color overrides */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] div {{
        color: black !important;
        font-weight: 500;
    }}
    
    /* Modern styling for the entire app */
    .stApp {{
        background-color: {BG_COLOR};
    }}

    /* Text styling for dark contrast */
    .stMarkdown, .stText, div[data-testid="stText"] {{
        color: {TEXT_COLOR} !important;
    }}

    /* Header styling */
    h1, h2, h3, h4 {{
        color: {TEXT_COLOR} !important;
        font-weight: 600;
    }}

    /* Metrics and other text elements */
    [data-testid="stMetricLabel"], 
    [data-testid="stMetricValue"],
    .streamlit-expanderHeader {{
        color: {TEXT_COLOR} !important;
    }}

    /* Modernized sidebar */
    [data-testid="stSidebar"] {{
        background-color: {CARD_COLOR};
        box-shadow: 2px 0px 5px rgba(0,0,0,0.05);
    }}

    /* Enhanced metric boxes */
    [data-testid="stMetric"] {{
        background-color: {CARD_COLOR};
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(67, 97, 238, 0.1);
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: transform 0.2s ease;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }}

    /* Modern typography */
    h1, h2, h3, h4 {{
        color: {TEXT_COLOR};
        font-weight: 600;
        letter-spacing: -0.5px;
    }}

    /* Button styling */
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 25px;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    .stButton>button:hover {{
        background-color: {SECONDARY_COLOR};
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}

    /* DataFrames and tables */
    .dataframe {{
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(0,0,0,0.05);
    }}
    </style>
""", unsafe_allow_html=True)


# --- 1. MODEL LOADING AND DATA PREPARATION (CACHED FOR EFFICIENCY) ---

@st.cache_data
def load_and_predict_data():
    """Loads data and models, generates augmented DataFrame, and calculates ML metrics."""
    try:
        with st.spinner("⏳ Loading ML models and generating predictions..."):
            required_files = [DATA_FILE_PATH, REG_MODEL_PATH, CLS_MODEL_PATH]
            if not all(os.path.exists(p) for p in required_files):
                st.error(f"Missing required file(s). Please ensure {', '.join(required_files)} are in the current directory.")
                return None

            # Load Raw Data
            df = pd.read_csv(DATA_FILE_PATH)
            df['arrival_time'] = pd.to_datetime(df['arrival_time'])
            df = df.sort_values(by='arrival_time').reset_index(drop=True)

            # Load Pickled Models
            with open(REG_MODEL_PATH, 'rb') as f:
                reg_pipeline = pickle.load(f)
            with open(CLS_MODEL_PATH, 'rb') as f:
                cls_pipeline = pickle.load(f)

            # Generate Predictions
            df['predicted_service_time'] = reg_pipeline.predict(df)
            df['predicted_service_time'] = np.maximum(5, df['predicted_service_time'])
            df['predicted_admission_prob'] = cls_pipeline.predict_proba(df)[:, 1]

            # --- ML Performance Metric Calculation ---
            y_true_admitted = df['is_admitted']
            y_pred_prob = df['predicted_admission_prob']
            y_pred_class = (y_pred_prob >= 0.5).astype(int)

            # Service time statistics
            service_time_true = df['service_time']
            service_time_pred = df['predicted_service_time']

            st.session_state['ml_metrics'] = {
                # Classification Metrics
                'Classification_Accuracy': accuracy_score(y_true_admitted, y_pred_class),
                'Classification_Precision': precision_score(y_true_admitted, y_pred_class),
                'Classification_Recall': recall_score(y_true_admitted, y_pred_class),
                'Classification_F1': f1_score(y_true_admitted, y_pred_class),
                'Classification_AUC': roc_auc_score(y_true_admitted, y_pred_prob),

                # Regression Metrics
                'Regression_MAE': mean_absolute_error(service_time_true, service_time_pred),
                'Regression_RMSE': np.sqrt(mean_squared_error(service_time_true, service_time_pred)),
                'Regression_R2': r2_score(service_time_true, service_time_pred),
                'Service_Time_Mean': np.mean(service_time_true),
                'Service_Time_Median': np.median(service_time_true),
                'Predicted_Time_Mean': np.mean(service_time_pred),
                'Predicted_Time_Median': np.median(service_time_pred)
            }

            st.success(f"Models loaded successfully. Predictions generated for {len(df)} patients.")
            return df

    except Exception as e:
        st.error(f"Fatal Error during Model/Data Loading: {e}")
        st.warning("HINT: This error often means your **scikit-learn** version is mismatched with the version used to save the model. Try installing scikit-learn version 1.6.1.")
        return None

        # Generate Predictions
        df['predicted_service_time'] = reg_pipeline.predict(df)
        df['predicted_service_time'] = np.maximum(5, df['predicted_service_time']) 
        df['predicted_admission_prob'] = cls_pipeline.predict_proba(df)[:, 1]
        
        # --- ML Performance Metric Calculation ---
        y_true_admitted = df['is_admitted'] 
        y_pred_prob = df['predicted_admission_prob']
        y_pred_class = (y_pred_prob >= 0.5).astype(int)
        
        # Service time statistics
        service_time_true = df['service_time']
        service_time_pred = df['predicted_service_time']
        
        st.session_state['ml_metrics'] = {
            # Classification Metrics
            'Classification_Accuracy': accuracy_score(y_true_admitted, y_pred_class),
            'Classification_Precision': precision_score(y_true_admitted, y_pred_class),
            'Classification_Recall': recall_score(y_true_admitted, y_pred_class),
            'Classification_F1': f1_score(y_true_admitted, y_pred_class),
            'Classification_AUC': roc_auc_score(y_true_admitted, y_pred_prob),
            
            # Regression Metrics
            'Regression_MAE': mean_absolute_error(service_time_true, service_time_pred),
            'Regression_RMSE': np.sqrt(mean_squared_error(service_time_true, service_time_pred)),
            'Regression_R2': r2_score(service_time_true, service_time_pred),
            'Service_Time_Mean': np.mean(service_time_true),
            'Service_Time_Median': np.median(service_time_true),
            'Predicted_Time_Mean': np.mean(service_time_pred),
            'Predicted_Time_Median': np.median(service_time_pred)
        }
        
        st.success(f"Models loaded successfully. Predictions generated for {len(df)} patients.")
        return df

    except Exception as e:
        st.error(f"Fatal Error during Model/Data Loading: {e}")
        st.warning("HINT: This error often means your **scikit-learn** version is mismatched with the version used to save the model. Try installing scikit-learn version 1.6.1.")
        return None

# --- 2. SIMPY CORE LOGIC ---

class Patient:
    """Represents a patient and calculates the ML-Heuristic priority score."""
    def __init__(self, data_row):
        self.triage_level = data_row['triage_level']
        self.service_time_predicted = data_row['predicted_service_time'] 
        self.admission_prob_predicted = data_row['predicted_admission_prob'] 
        self.time_triage_completed = 0
        self.wait_time = 0

        # ML-Heuristic Score Calculation (High Score = Higher Priority)
        triage_weight = 6 - self.triage_level
        epsilon = 0.001 

        self.priority_score = (
            (triage_weight * 100) + 
            (self.admission_prob_predicted * (1 / (self.service_time_predicted + epsilon)))
        )

class EmergencyDepartment:
    def __init__(self, env, capacity, policy='FCFS'):
        self.env = env
        self.capacity = capacity 
        self.beds = simpy.Resource(env, capacity=capacity)
        self.patients_completed = []
        self.policy = policy

    def patient_triage_and_treatment(self, patient):
        patient.time_triage_completed = self.env.now
        with self.beds.request() as request:
            yield request 
            patient.wait_time = self.env.now - patient.time_triage_completed
            yield self.env.timeout(patient.service_time_predicted)
            self.patients_completed.append(patient)

    def patient_arrival_generator(self, patient_data):
        start_time = patient_data.iloc[0]['arrival_time']
        for _, row in patient_data.iterrows():
            patient = Patient(row)
            arrival_offset = (row['arrival_time'] - start_time).total_seconds() / 60.0 
            yield self.env.timeout(arrival_offset - self.env.now)
            self.env.process(self.patient_triage_and_treatment(patient))


class PriorityEmergencyDepartment(EmergencyDepartment):
    def __init__(self, env, capacity):
        super().__init__(env, capacity, policy='Heuristic')
        self.capacity = capacity # Explicitly re-set capacity for robustness (Fixes SimPy AttributeError)
        self.queue = simpy.PriorityStore(env)
        self.env.process(self.scheduler())

    def patient_triage_and_treatment(self, patient):
        patient.time_triage_completed = self.env.now
        priority_tuple = (-patient.priority_score, patient.time_triage_completed)
        yield self.queue.put((priority_tuple, patient)) 

    def scheduler(self):
        while True:
            if self.beds.count < self.capacity and len(self.queue.items) > 0: 
                item = yield self.queue.get()
                patient = item[1]
                self.env.process(self._start_treatment(patient))
            else:
                yield self.env.timeout(1)

    def _start_treatment(self, patient):
        with self.beds.request() as request:
            yield request
            patient.wait_time = self.env.now - patient.time_triage_completed
            yield self.env.timeout(patient.service_time_predicted)
            self.patients_completed.append(patient)


@st.cache_data(show_spinner="Running Discrete Event Simulation (SimPy)...")
def run_simulation(data, capacity, sim_duration_days):
    """Runs FCFS and Heuristic simulations and returns a summary and raw wait times."""
    
    sim_end_time_minutes = sim_duration_days * 24 * 60
    
    # --- FCFS Run ---
    env_fcfs = simpy.Environment()
    ed_fcfs = EmergencyDepartment(env_fcfs, capacity, 'FCFS')
    env_fcfs.process(ed_fcfs.patient_arrival_generator(data))
    env_fcfs.run(until=sim_end_time_minutes)

    # --- Heuristic Run ---
    env_h = simpy.Environment()
    ed_h = PriorityEmergencyDepartment(env_h, capacity)
    env_h.process(ed_h.patient_arrival_generator(data))
    env_h.run(until=sim_end_time_minutes)

    all_results = []
    all_wait_times = []

    for ed in [ed_fcfs, ed_h]:
        wait_times = np.array([p.wait_time for p in ed.patients_completed])
        results = {
            'Policy': ed.policy,
            'Capacity': capacity,
            'Patients Completed': len(ed.patients_completed),
            'Mean Wait Time (min)': np.mean(wait_times) if len(wait_times) > 0 else 0,
            'P90 Wait Time (min)': np.percentile(wait_times, 90) if len(wait_times) > 0 else 0,
        }
        all_results.append(results)
        
        if len(wait_times) > 0:
            all_wait_times.append(pd.DataFrame({'Wait Time (min)': wait_times, 'Policy': ed.policy}))

    summary_df = pd.DataFrame(all_results).set_index('Policy')
    raw_wait_times_df = pd.concat(all_wait_times, ignore_index=True) if all_wait_times else pd.DataFrame()
    
    return summary_df, raw_wait_times_df

# --- 3. STREAMLIT APP LAYOUT & VISUALIZATION ---

def apply_high_contrast_layout(fig):
    """Applies universal dark text and theme styling to a Plotly figure."""
    
    fig.update_layout(
        template='plotly_white',
        font=dict(color=TEXT_COLOR, size=14),
        title_font_color=TEXT_COLOR,
        xaxis=dict(title_font_color=TEXT_COLOR, tickfont_color=TEXT_COLOR, gridcolor='#eeeeee'),
        yaxis=dict(title_font_color=TEXT_COLOR, tickfont_color=TEXT_COLOR, gridcolor='#eeeeee'),
        legend=dict(font=dict(color=TEXT_COLOR)),
        plot_bgcolor=CARD_COLOR,
        paper_bgcolor=CARD_COLOR,
        hoverlabel=dict(bgcolor=CARD_COLOR, font_size=16, font_family="Arial", font_color=TEXT_COLOR)
    )
    return fig

def display_ml_metrics():
    """Displays the performance metrics of the underlying ML models."""
    metrics = st.session_state.get('ml_metrics', {})
    if not metrics: return
        
    st.markdown("## 📊 Model Performance Metrics")
    st.markdown("### Detailed Metrics Analysis")
    # Create three columns for detailed metrics
    col_cls, col_reg, col_time = st.columns(3)
    
    with col_cls:
        st.markdown("#### 📋 Classification Details")
        metrics_cls = [
            ("Precision", 'Classification_Precision', "Positive predictive value"),
            ("Recall", 'Classification_Recall', "True positive rate"),
            ("F1-Score", 'Classification_F1', "Harmonic mean of precision & recall"),
            ("Accuracy", 'Classification_Accuracy', "Overall accuracy"),
            ("AUC-ROC", 'Classification_AUC', "Area under ROC curve")
        ]
        for label, key, help_text in metrics_cls:
            st.metric(label, f"{metrics[key]:.3f}", help=help_text)
    
    with col_reg:
        st.markdown("#### 📊 Regression Analysis")
        metrics_reg = [
            ("R² Score", 'Regression_R2', "Coefficient of determination"),
            ("MAE (min)", 'Regression_MAE', "Mean absolute error"),
            ("RMSE (min)", 'Regression_RMSE', "Root mean squared error")
        ]
        for label, key, help_text in metrics_reg:
            st.metric(label, f"{metrics[key]:.3f}", help=help_text)
    
    with col_time:
        st.markdown("#### ⏱️ Time Statistics")
        metrics_time = [
            ("Mean Service Time", 'Service_Time_Mean', "Average actual time"),
            ("Median Service Time", 'Service_Time_Median', "Median actual time"),
            ("Predicted Mean", 'Predicted_Time_Mean', "Average predicted time"),
            ("Predicted Median", 'Predicted_Time_Median', "Median predicted time")
        ]
        for label, key, help_text in metrics_time:
            st.metric(label, f"{metrics[key]:.2f} min", help=help_text)

    # Add an expander for metric descriptions
    with st.expander("ℹ️ Metrics Explanation"):
        st.markdown("""
        ### Classification Metrics
        - **Accuracy**: Proportion of correct predictions (higher is better)
        - **Precision**: Accuracy of positive predictions (higher is better)
        - **Recall**: Proportion of actual positives correctly identified (higher is better)
        - **F1-Score**: Harmonic mean of precision and recall (higher is better)
        - **AUC-ROC**: Area under ROC curve, measures discrimination (higher is better)
        
        ### Regression Metrics
        - **R² Score**: Proportion of variance explained by model (closer to 1 is better)
        - **MAE**: Average absolute prediction error in minutes (lower is better)
        - **RMSE**: Root mean squared error in minutes (lower is better)
        
        ### Time Statistics
        Comparison between actual and predicted service times in minutes.
        """)

    st.markdown("---")

def main():
    st.set_page_config(layout="wide", page_title="ED Optimization Simulator")
    
    st.title("🏥 ML-Optimized ED Simulation Dashboard")
    st.markdown("Use the sidebar to test different resource strategies. The Heuristic Policy utilizes your Regression (Service Time) and Classification (Admission Probability) models.")
    
    # --- Sidebar for User Input ---
    with st.sidebar:
        st.header("⚙️ Scenario Inputs")
        bed_capacity = st.slider(
            "ED Bed/Room Capacity", 
            min_value=5, max_value=50, value=20, step=1,
            help="The total number of available beds or treatment rooms."
        )
        
        sim_duration_days = st.slider(
            "Simulation Duration (Days)", 
            min_value=1, max_value=30, value=7, step=1,
            help="The number of days to simulate using the patient arrival data."
        )
        st.markdown("---")
        run_button = st.button("▶️ Run Simulation & Update Dashboard", key="run_sim")

    # Load data and run initial checks
    data_with_predictions = load_and_predict_data()

    if data_with_predictions is None:
        return

    # Display ML metrics once
    display_ml_metrics()
    st.markdown("---")

    # --- Run Simulation on Button Click ---
    if run_button:
        
        summary_df, raw_wait_times_df = run_simulation(data_with_predictions, bed_capacity, sim_duration_days)
        
        st.subheader("2. Simulation Results")
        
        # --- Metrics Display ---
        fcfs_mean = summary_df.loc['FCFS', 'Mean Wait Time (min)']
        heuristic_mean = summary_df.loc['Heuristic', 'Mean Wait Time (min)']
        
        mean_diff = fcfs_mean - heuristic_mean
        mean_percent = (mean_diff / fcfs_mean) * 100 if fcfs_mean > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Beds/Capacity", f"{bed_capacity} Beds")
        col2.metric("Patients Simulated", f"{summary_df.loc['FCFS', 'Patients Completed']}")
        col3.metric(
            "Avg. Wait Time Improvement", 
            f"{mean_diff:.1f} min", 
            f"{mean_percent:.1f}% reduction" if mean_diff > 0 else "0.0% change"
        )
        
        st.dataframe(summary_df.drop(columns=['Capacity']), use_container_width=True)
        
        st.markdown("---")

        # --- Visualization Section ---
        st.subheader("3. Detailed Performance Visualization")
        
        col_pie, col_bar = st.columns([1, 2])

        with col_pie:
            # --- Triage Pie Chart ---
            triage_counts = data_with_predictions['triage_level'].value_counts().reset_index()
            triage_counts.columns = ['Triage Level', 'Count']
            
            triage_counts['Triage Level'] = triage_counts['Triage Level'].astype(str)
            triage_counts = triage_counts.sort_values(by='Triage Level')
            
            # Modern color palette for triage levels
            triage_colors = ['#ef476f', '#ffd166', '#06d6a0', '#118ab2', '#073b4c']
            
            fig_pie = px.pie(
                triage_counts,
                values='Count',
                names='Triage Level',
                title='A. Patient Volume Distribution by Triage Level',
                hole=0.4,  # Slightly larger hole for modern look
                color_discrete_sequence=triage_colors,
                template='plotly_white'
            )
            # Enhanced styling without borders
            fig_pie.update_traces(
                textinfo='percent+label',
                textfont_size=14,
                textposition='outside',
                marker=dict(
                    line=dict(color='rgba(0,0,0,0)', width=0),  # Remove borders
                    pattern=dict(shape="")
                ),
                pull=[0.03, 0.03, 0.03, 0.03, 0.03]  # Slight separation between segments
            )
            # Update layout for better appearance
            fig_pie.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                title=dict(
                    font=dict(size=16),
                    y=0.95
                ),
                margin=dict(t=80, l=0, r=0, b=0)
            )
            fig_pie = apply_high_contrast_layout(fig_pie) 
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("Figure 3A: Shows the proportion of patients entering the ED. This distribution heavily influences capacity and wait times.")
            
            st.markdown("---")
            st.markdown("#### Priority Rationale")
            st.code(f"""
SCORE = (6 - Triage Level) * 100
      + P(Admission)
      * (1 / Predicted Service Time)
""")

        with col_bar:
            # 1. Bar Chart of Key Metrics (Mean and P90)
            plot_data = summary_df[['Mean Wait Time (min)', 'P90 Wait Time (min)']].reset_index().melt(
                id_vars='Policy', var_name='Metric', value_name='Time (min)'
            )
            
            fig_bar = px.bar(
                plot_data,
                x='Policy',
                y='Time (min)',
                color='Metric',
                barmode='group',
                title='B. Mean vs. 90th Percentile Wait Time by Policy',
                color_discrete_map={'Mean Wait Time (min)': PRIMARY_COLOR, 'P90 Wait Time (min)': SECONDARY_COLOR},
                height=450,
                template='plotly_white'
            )
            fig_bar.update_layout(xaxis_title="Queueing Policy", yaxis_title="Time (minutes)")
            fig_bar = apply_high_contrast_layout(fig_bar) 
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption("Figure 3B: Compares the overall average (Mean) and the worst-case (P90) wait times. **The goal is to see the Orange (Heuristic) bars lower than the Blue (FCFS) bars.**")


        # 2. CDF Plot of Raw Waiting Times (The most informative plot for queueing)
        st.markdown("#### Cumulative Distribution of Wait Times (CDF)")
        
        # Calculate ECDF for both policies
        def calculate_ecdf(data):
            x = np.sort(data)
            y = np.arange(1, len(data) + 1) / len(data)
            return pd.DataFrame({'Wait Time (min)': x, 'Cumulative Probability': y})

        ecdf_fcfs = calculate_ecdf(raw_wait_times_df[raw_wait_times_df['Policy'] == 'FCFS']['Wait Time (min)'])
        ecdf_heuristic = calculate_ecdf(raw_wait_times_df[raw_wait_times_df['Policy'] == 'Heuristic']['Wait Time (min)'])

        ecdf_fcfs['Policy'] = 'FCFS (Baseline)'
        ecdf_heuristic['Policy'] = 'Heuristic (ML-Optimized)'
        
        ecdf_plot_data = pd.concat([ecdf_fcfs, ecdf_heuristic])

        fig_cdf = px.line(
            ecdf_plot_data,
            x='Wait Time (min)',
            y='Cumulative Probability',
            color='Policy',
            title='C. Patient Wait Time Distribution (Proportion of Patients Waiting Less)',
            color_discrete_map={'FCFS (Baseline)': PRIMARY_COLOR, 'Heuristic (ML-Optimized)': SECONDARY_COLOR},
            height=500,
            template='plotly_white'
        )
        fig_cdf.update_layout(xaxis_title="Wait Time (minutes)", yaxis_title="Proportion of Patients Waiting Less", legend_title="Policy")
        fig_cdf = apply_high_contrast_layout(fig_cdf) 
        st.plotly_chart(fig_cdf, use_container_width=True)
        st.caption("Figure 3C: The **Violet (Heuristic)** line being to the left of the **Blue (FCFS)** line confirms the ML-driven policy shortens waiting times for a greater proportion of patients.")
            
    else:
        st.markdown("Adjust the simulation parameters on the left sidebar and click 'Run Simulation' to begin analysis.")
        st.markdown("### Model and Data Status:")
        st.write(f"- Data File: `{DATA_FILE_PATH}` (Loaded)")
        st.write(f"- Regression Model: `{REG_MODEL_PATH}` (Awaiting execution)")
        st.write(f"- Classification Model: `{CLS_MODEL_PATH}` (Awaiting execution)")

if __name__ == "__main__":
    if 'ml_metrics' not in st.session_state:
        st.session_state['ml_metrics'] = {}
    main()
