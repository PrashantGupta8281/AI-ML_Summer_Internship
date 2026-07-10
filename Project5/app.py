import streamlit as st
import numpy as np
from PIL import Image
import joblib
import plotly.graph_objects as go

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="AI Gender Classification",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------------------------
# Custom CSS
# ----------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Outfit', sans-serif;
}

.stApp{
background:
radial-gradient(circle at top left,#2563EB 0%,transparent 35%),
radial-gradient(circle at bottom right,#7C3AED 0%,transparent 35%),
linear-gradient(135deg,#020617,#0F172A);
}

.main-title{
text-align:center;
font-size:55px;
font-weight:700;
background:linear-gradient(90deg,#38BDF8,#A855F7);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.subtitle{
text-align:center;
font-size:20px;
color:#CBD5E1;
margin-bottom:30px;
}

.card{
background:rgba(255,255,255,.08);
padding:30px;
border-radius:20px;
border:1px solid rgba(255,255,255,.15);
}

div[data-testid="stImage"] img{
border-radius:20px;
border:3px solid #38BDF8;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.title("🧠 AI Gender Classification")

st.sidebar.markdown("""
### Model Information

- Logistic Regression
- Image Size: **64 × 64**
- RGB Images
- Binary Classification

---

### Features

✅ Upload Image

✅ Instant Prediction

✅ Confidence Score

✅ Interactive Dashboard

---

Made with ❤️ using Streamlit
""")

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("Male_And_Female_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model could not be loaded.\n\n{e}")
    st.stop()

IMG_SIZE = 64

# ----------------------------------------------------
# Header
# ----------------------------------------------------
st.markdown(
'<div class="main-title">AI Gender Classification</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Upload an image and let AI classify it.</div>',
unsafe_allow_html=True
)

# ----------------------------------------------------
# Upload
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "📷 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    resized = image.resize((IMG_SIZE, IMG_SIZE))
    resized = np.array(resized).flatten()

    with st.spinner("Analyzing image..."):

        prediction = model.predict([resized])[0]
        probability = model.predict_proba([resized])[0]

    with col2:

        st.subheader("Prediction")

        if prediction == 0:
            st.success("👨 Male")
        else:
            st.success("👩 Female")

        st.metric(
            "Highest Confidence",
            f"{max(probability)*100:.2f}%"
        )

        st.write("### Confidence")

        st.write(f"👨 Male : {probability[0]*100:.2f}%")
        st.progress(int(probability[0]*100))

        st.write(f"👩 Female : {probability[1]*100:.2f}%")
        st.progress(int(probability[1]*100))

    st.divider()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Male","Female"],
                values=probability,
                hole=0.70,
                marker=dict(
                    colors=["#38BDF8","#A855F7"]
                )
            )
        ]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="white",
            family="Outfit",
            size=16
        ),
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")

st.markdown(
"""
<div style='text-align:center;color:#CBD5E1;'>
AI Gender Classification • Built with Streamlit
</div>
""",
unsafe_allow_html=True
)
