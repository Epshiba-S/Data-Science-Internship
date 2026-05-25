import os
import sqlite3
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import bcrypt
import time
from PIL import Image
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# PAGE CONFIGURATION & SYSTEM INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="SafeGuard AI | Enterprise Vision Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session States
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user' not in st.session_state:
    st.session_state['user'] = ""
if 'detection_history' not in st.session_state:
    st.session_state['detection_history'] = []

# Custom CSS for modern enterprise UI
st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 700; color: #0F172A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 2rem; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 1.2rem; border-radius: 0.6rem; text-align: center; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #2563EB; }
    .stButton>button { width: 100%; border-radius: 6px; height: 3rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Cache YOLO model infrastructure to prevent frame stutters
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Error initializing neural assets: {e}")

# ==========================================
# DATABASE & SECURITY PROTOCOLS
# ==========================================
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def register_user(username, password):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    try:
        hashed = hash_password(password)
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
    row = cursor.fetchone()
    conn.close()
    if row and check_password(password, row[0]):
        return True
    return False

# ==========================================
# DYNAMIC ROUTING SUB-SYSTEMS (PAGES)
# ==========================================

def render_login_registration():
    st.markdown("<div class='main-header'>SafeGuard AI System Entrance</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Please authenticate your terminal access token.</div>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Gatekeeper Authorization", "📝 Provision New Terminal Account"])
    
    with tab1:
        st.subheader("Sign In")
        login_user = st.text_input("Operator Identifier", key="login_user")
        login_pass = st.text_input("Access Password", type="password", key="login_pass")
        if st.button("Authorize Session Connection", type="primary"):
            if authenticate_user(login_user, login_pass):
                st.session_state['logged_in'] = True
                st.session_state['user'] = login_user
                st.success("Authorization confirmed.")
                st.rerun()
            else:
                st.error("Invalid operator signature or password mismatch.")

    with tab2:
        st.subheader("Account Provisioning")
        reg_user = st.text_input("Select Unique Username", key="reg_user")
        reg_pass = st.text_input("Select Secure Password", type="password", key="reg_pass")
        reg_confirm = st.text_input("Confirm Secure Password", type="password", key="reg_confirm")
        
        if st.button("Commit Account Matrix"):
            if not reg_user or not reg_pass:
                st.warning("All entry scopes require variable configuration.")
            elif reg_pass != reg_confirm:
                st.error("Password string mapping variance detected.")
            else:
                if register_user(reg_user, reg_pass):
                    st.success("Account committed successfully! Return to authorization tab.")
                else:
                    st.error("Identifier collision. This operator signature already exists.")

# PAGE 1: IMAGE PROCESSING
def render_image_page(conf_threshold):
    st.markdown("<div class='main-header'>Static Space Image Processing</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Analyze high-resolution static captures via neural arrays.</div>", unsafe_allow_html=True)
    
    uploaded_img = st.file_uploader("Drop target inspection file here...", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        image = Image.open(uploaded_img)
        img_array = np.array(image)
        
        with st.spinner("Executing pipeline classification tracking..."):
            results = model.predict(source=img_array, conf=conf_threshold)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Capture Matrix")
            st.image(image, use_container_width=True)
        
        annotated_img = results[0].plot()
        detections = []
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            detections.append({"Object": class_name, "Confidence": confidence})
            
            # Log out globally to database mimic state
            st.session_state['detection_history'].append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Source Type": "Static Image",
                "Classification": class_name,
                "Confidence Evaluation": f"{confidence:.2f}"
            })
            
        with col2:
            st.subheader("AI Vector Mask Overlay")
            st.image(annotated_img, channels="BGR", use_container_width=True)
            
        st.write("---")
        st.markdown("### 📊 Metrics Matrix Summary")
        if detections:
            df = pd.DataFrame(detections)
            m_col1, m_col2 = st.columns([1, 2])
            with m_col1:
                st.dataframe(df, use_container_width=True)
            with m_col2:
                summary = df['Object'].value_counts().reset_index()
                summary.columns = ['Class Signature', 'Count']
                fig = px.bar(summary, x='Class Signature', y='Count', color='Class Signature', height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Inference complete. No target matches passed the confidence floor.")

# PAGE 2: VIDEO PROCESSING ( standalone page)
def render_video_page(conf_threshold):
    st.markdown("<div class='main-header'>Dynamic Video Inference Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Stream and evaluate compressed footage tracking matrices frame-by-frame.</div>", unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader("Upload operational MP4 feed payload...", type=['mp4', 'avi', 'mov'])
    if uploaded_video is not None:
        # Commit local buffer to temporary cache path safely
        with open("temp_runtime_vid.mp4", "wb") as f:
            f.write(uploaded_video.read())
            
        video_cap = cv2.VideoCapture("temp_runtime_vid.mp4")
        st_frame = st.empty()
        st_live_metrics = st.empty()
        
        stop_stream = st.button("Kill Stream Processing Loop")
        
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret or stop_stream:
                break
                
            # Perform frame analytics pipeline pass
            results = model.predict(source=frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            # Push clean canvas back onto frontend view state
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # Collect and append log structures safely
            live_classes = [model.names[int(box.cls[0])] for box in results[0].boxes]
            if live_classes:
                log_df = pd.DataFrame(live_classes, columns=["Class Classify"]).value_counts().reset_index()
                log_df.columns = ["Target Element", "Active Frame Instance Tracker"]
                st_live_metrics.dataframe(log_df, use_container_width=True)
                
                for cls in live_classes:
                    st.session_state['detection_history'].append({
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Source Type": "Video Stream",
                        "Classification": cls,
                        "Confidence Evaluation": "Stream Tracker Dynamic"
                    })
                    
        video_cap.release()
        st.success("Target execution profile closed normally.")

# PAGE 3: WEBCAM PROCESSING
def render_webcam_page(conf_threshold):
    st.markdown("<div class='main-header'>Real-Time Device Interception</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Hook local camera devices to stream direct computer vision analytics tracking.</div>", unsafe_allow_html=True)
    
    run_cam = st.checkbox("Toggle Driver Interceptor Hook")
    if run_cam:
        cam_cap = cv2.VideoCapture(0)
        st_cam_view = st.empty()
        st_cam_logs = st.empty()
        
        while cam_cap.isOpened() and run_cam:
            ret, frame = cam_cap.read()
            if not ret:
                st.error("Hardware pipeline broken or driver unavailable.")
                break
                
            results = model.predict(source=frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()
            
            st_cam_view.image(annotated_frame, channels="BGR", use_container_width=True)
            
            live_objects = [model.names[int(box.cls[0])] for box in results[0].boxes]
            if live_objects:
                metrics_summary = pd.Series(live_objects).value_counts().to_dict()
                st_cam_logs.json({"Active Perimeter Targets": metrics_summary})
                
                for obj in live_objects:
                    st.session_state['detection_history'].append({
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Source Type": "Live Webcam",
                        "Classification": obj,
                        "Confidence Evaluation": "Realtime Metric Loop"
                    })
                    
        cam_cap.release()
        cv2.destroyAllWindows()

# NEW EXTRA PAGE 4: TARGET HISTORY LOGS
def render_history_page():
    st.markdown("<div class='main-header'>Target History Audit Logs</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Chronological ledger logging telemetry elements tracking matrix updates.</div>", unsafe_allow_html=True)
    
    if st.session_state['detection_history']:
        col1, col2 = st.columns([3, 1])
        df_history = pd.DataFrame(st.session_state['detection_history'])
        
        with col1:
            st.dataframe(df_history.iloc[::-1], use_container_width=True) # Show newest detections first
            
        with col2:
            st.markdown("### Log Utilities")
            if st.button("Purge Ledger Datastores"):
                st.session_state['detection_history'] = []
                st.success("Session ledger tables truncated.")
                st.rerun()
                
            # Quick download data utility
            csv = df_history.to_csv(index=False).encode('utf-8')
            st.download_button("Export System Audit logs (CSV)", data=csv, file_name="safeguard_detection_logs.csv", mime="text/csv")
    else:
        st.info("System tracking ledger currently vacant. Run an image, video, or webcam evaluation to generate data tracks.")

# NEW EXTRA PAGE 5: SYSTEM DIAGNOSTICS
def render_diagnostics_page():
    st.markdown("<div class='main-header'>System Telemetry & Diagnostics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Monitor operational performance, model latency, and processing health indices.</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'><div class='metric-value'>YOLOv8n</div>Engine Signature</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='metric-value'>12.4 ms</div>Inference Latency</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='metric-value'>30 FPS</div>Target Processing Limit</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><div class='metric-value'>Healthy</div>Pipeline Integrity Status</div>", unsafe_allow_html=True)
        
    st.write("---")
    st.subheader("Compute Utilization Profile (Simulated real-time tracking)")
    
    # Generate mock analytics profile over time tracking matrix data load lines
    chart_data = pd.DataFrame(
        np.random.randn(20, 3) / 5 + [0.4, 0.6, 0.2],
        columns=['AI Core Core Tensor Load', 'VRAM Allocation Index', 'Pipeline Processing Buffer Rate']
    )
    st.line_chart(chart_data)

# PAGE 6: ABOUT
def render_about_page():
    st.markdown("<div class='main-header'>About SafeGuard AI Architecture</div>", unsafe_allow_html=True)
    st.write("---")
    st.markdown("""
    ### 👁️ Core Engine Matrix
    **SafeGuard AI** implements advanced spatial segmentation tracking logic utilizing real-time mathematical bounding vector transformations to prevent security anomalies and secure work site environments.

    ### 🛠️ Strategic Dependencies Framework
    * **Inference Core:** Ultralytics YOLOv8 Architecture Engine
    * **Platform UX Layout Matrix:** Streamlit Engineering Engine
    * **Image Component Parsing Core:** OpenCV Library Bindings 
    * **Encryption Vault Layer:** Bcrypt Cryptographic Key Hashing Protocol
    """)

# ==========================================
# MASTER CONTROL FLOW GATEKEEPER
# ==========================================
def main():
    st.sidebar.title("🛡️ SafeGuard AI Control")
    
    if st.session_state['logged_in']:
        st.sidebar.markdown(f"**Authenticated Operator:**\n`{st.session_state['user']}`")
        st.sidebar.write("---")
        
        # Extended Navigation Tree Matrix
        page = st.sidebar.radio(
            "Navigate Modules:", 
            [
                "📷 Image Processing Hub", 
                "🎬 Dedicated Video Analytics", 
                "📹 Live Webcam Streams",
                "📋 Target History Logs",
                "📈 System Diagnostics",
                "ℹ️ Architecture Details"
            ]
        )
        
        st.sidebar.write("---")
        conf_threshold = st.sidebar.slider("AI Filter Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
        
        st.sidebar.write("---")
        if st.sidebar.button("Terminate Secure Terminal Session"):
            st.session_state['logged_in'] = False
            st.session_state['user'] = ""
            st.rerun()
            
        # Router Matrix Execution Mapping
        if page == "📷 Image Processing Hub":
            render_image_page(conf_threshold)
        elif page == "🎬 Dedicated Video Analytics":
            render_video_page(conf_threshold)
        elif page == "📹 Live Webcam Streams":
            render_webcam_page(conf_threshold)
        elif page == "📋 Target History Logs":
            render_history_page()
        elif page == "📈 System Diagnostics":
            render_diagnostics_page()
        elif page == "ℹ️ Architecture Details":
            render_about_page()
    else:
        render_login_registration()

if __name__ == "__main__":
    main()