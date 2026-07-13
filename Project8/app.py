import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM THEME
# ==========================================
st.set_page_config(
    page_title="Iris KNN Classifier Studio",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom injection for sleek styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background-color: #ffffff;
        border-radius: 4px;
        font-weight: 600;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA PREPARATION (Cached for Performance)
# ==========================================
@st.cache_data
def load_and_prep_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    
    # Map targets to actual flower names for better UI readability
    target_mapping = {i: name for i, name in enumerate(iris.target_names)}
    df['species_name'] = df['species'].map(target_mapping)
    return iris, df

iris, df = load_and_prep_data()

# ==========================================
# 3. SIDEBAR - CONTROLS & HYPERPARAMETERS
# ==========================================
st.sidebar.image("https://images.unsplash.com/photo-1526047932273-341f2a7631f9?w=500&auto=format&fit=crop", use_container_width=True)
st.sidebar.title("🛠️ Model Configuration")
st.sidebar.markdown("Fine-tune your KNN classifier properties live.")

st.sidebar.subheader("Hyperparameters")
n_neighbors = st.sidebar.slider("Number of Neighbors (K)", min_value=1, max_value=21, value=7, step=2)
test_size = st.sidebar.slider("Test Dataset Split Ratio", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
random_state = st.sidebar.number_input("Random Seed State", min_value=0, max_value=100, value=1)

st.sidebar.divider()
st.sidebar.info("💡 **Tip:** Adjusting **K** alters the decision boundaries. Odd numbers prevent voting ties!")
st.sidebar.title("My Portfolio")

st.sidebar.markdown("""
### 👋 About Me

**Name:** Prashant Gupta

🔗 **LinkedIn:** [Click Here](https://www.linkedin.com/in/prashant-gupta-012320389?utm_source=share_via&utm_content=profile&utm_medium=member_android)

💻 **GitHub:** [Click Here](https://github.com/PrashantGupta8281/AI-ML_Summer_Internship)
""")

# ==========================================
# 4. MAIN INTERACTIVE DASHBOARD
# ==========================================
st.title("🌸 Iris Flower Classification Studio")
st.markdown("An interactive platform exploring the **K-Nearest Neighbors (KNN)** algorithm on the classic Iris dataset.")

# Top-level KPI metrics cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Rows", value=df.shape[0])
with col2:
    st.metric(label="Features Count", value=len(iris.feature_names))
with col3:
    st.metric(label="Classes/Species", value=len(iris.target_names))
with col4:
    # Split & Model Logic triggered dynamically on sidebar input change
    X = df[iris.feature_names]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    
    st.metric(label="Model Accuracy", value=f"{acc:.2f}%", delta=f"{acc - 90.0:.1f}% vs baseline")

# Organizing layout using sleek tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "📈 Dimensionality (t-SNE)", "🧠 Model Performance", "🔮 Real-time Predictor"])

# --- TAB 1: DATA EXPLORER ---
with tab1:
    st.subheader("Dataset Preview & Insights")
    
    show_raw = st.checkbox("Show raw dataset dataframe")
    if show_raw:
        st.dataframe(df, use_container_width=True)
        
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Feature Distributions**")
        selected_feature = st.selectbox("Select a feature to check distribution:", iris.feature_names)
        fig_hist = px.histogram(df, x=selected_feature, color="species_name", marginal="box", 
                                color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with c2:
        st.markdown("**Feature Correlations**")
        corr = df[iris.feature_names].corr()
        fig_heat, ax = plt.subplots(figsize=(5, 3.8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar=False)
        plt.tight_layout()
        st.pyplot(fig_heat)

# --- TAB 2: DIMENSIONALITY REDUCTION ---
with tab2:
    st.subheader("t-SNE 2D Manifold Visualization")
    st.write("Because the dataset contains 4 features, we use t-SNE to compress dimensions down to 2D for human interpretation.")
    
    with st.spinner("Calculating t-SNE embedding..."):
        tsne = TSNE(n_components=2, random_state=random_state)
        X_2d = tsne.fit_transform(X)
        tsne_df = pd.DataFrame(X_2d, columns=['Dimension 1', 'Dimension 2'])
        tsne_df['Species'] = df['species_name']
        
    fig_tsne = px.scatter(tsne_df, x='Dimension 1', y='Dimension 2', color='Species',
                          color_discrete_sequence=px.colors.qualitative.Safe,
                          title="2D Projection of Species Boundaries")
    st.plotly_chart(fig_tsne, use_container_width=True)

# --- TAB 3: MODEL PERFORMANCE ---
with tab3:
    st.subheader("KNN Evaluation Matrices")
    
    col_mat1, col_mat2 = st.columns([1, 1])
    
    with col_mat1:
        st.markdown("**Confusion Matrix Heatmap**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax_cm, cbar=False)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        st.pyplot(fig_cm)
        
    with col_mat2:
        st.markdown("**Classification Metrics Report**")
        report_dict = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='YlGnBu').format(precision=2), use_container_width=True)

# --- TAB 4: REAL-TIME PREDICTOR ---
with tab4:
    st.subheader("🔮 Predict Custom Flower Samples")
    st.markdown("Adjust the metric sliders below to see what species the trained model classifies it as:")
    
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        s_length = st.slider("Sepal Length (cm)", float(df.iloc[:,0].min()), float(df.iloc[:,0].max()), float(df.iloc[:,0].mean()))
    with p2:
        s_width = st.slider("Sepal Width (cm)", float(df.iloc[:,1].min()), float(df.iloc[:,1].max()), float(df.iloc[:,1].mean()))
    with p3:
        p_length = st.slider("Petal Length (cm)", float(df.iloc[:,2].min()), float(df.iloc[:,2].max()), float(df.iloc[:,2].mean()))
    with p4:
        p_width = st.slider("Petal Width (cm)", float(df.iloc[:,3].min()), float(df.iloc[:,3].max()), float(df.iloc[:,3].mean()))
        
    # Prediction execution
    user_input = np.array([[s_length, s_width, p_length, p_width]])
    prediction_idx = knn.predict(user_input)[0]
    prediction_proba = knn.predict_proba(user_input)[0]
    predicted_species = iris.target_names[prediction_idx]
    
    st.markdown("---")
    st.markdown(f"### Predicted Class: **`{predicted_species.upper()}`**")
    
    # Confidence Bar Chart Plotting
    prob_df = pd.DataFrame({
        'Species': iris.target_names,
        'Confidence Probability': prediction_proba
    })
    fig_prob = px.bar(prob_df, x='Confidence Probability', y='Species', orientation='h',
                      color='Species', color_discrete_sequence=px.colors.qualitative.Vivid, 
                      range_x=[0, 1])
    st.plotly_chart(fig_prob, use_container_width=True)
