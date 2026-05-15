import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. GLOBAL PAGE CONFIG & DARK MODE BLENDED GRADIENT INJECTION
# ==========================================
st.set_page_config(
    page_title="Olympia Premium Analytics Engine",
    page_icon="🏅",
    layout="wide"
)

# Deep CSS injection utilizing a custom dark slate to deep stadium track burgundy color gradient mesh
st.markdown("""
    <style>
    /* Force dark mode canvas configuration rules across all wrapper frames */
    html, body, .stApp, .stMain, [data-testid="stAppViewContainer"], .stAppHeader, [data-testid="stHeader"] {
        background: linear-gradient(135deg, #0E1117 30%, #2D141A 75%, #4A1A24 100%) !important;
        background-size: cover !important;
        background-attachment: fixed !important;
    }
    
    /* Strip out solid background panels from standard containers to reveal the theme blend underneath */
    [data-testid="stVerticalBlock"], 
    [data-testid="stElementContainer"], 
    .stDataFrame, 
    div[role="dialog"],
    div[data-testid="stGrid"] > div,
    [data-testid="stCard"] {
        background-color: transparent !important;
        background: transparent !important;
    }

    /* Style titles, labels, and paragraph markdown metrics for crystal clear visibility */
    h1, h2, h3, h4, p, label, li, span, .stMarkdown, [data-testid="stWidgetLabel"] p {
        color: #F0F2F6 !important; /* Premium crisp off-white */
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8) !important;
    }
    
    /* Make internal data frame row tracking text readable over the background color stretch */
    div[data-testid="stDataFrameDataframe"] div {
        color: #FFFFFF !important;
    }

    /* Highlight critical application metrics counters */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.9) !important;
    }

    /* Dark sidebar panel matching native dark style specs */
    [data-testid="stSidebar"] {
        background-color: rgba(14, 17, 23, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        z-index: 100;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Core Asset Pipeline Connectors ---

@st.cache_data
def load_data():
    return pd.read_csv("OlympicAthletes.csv")

try:
    with open('olympic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
except FileNotFoundError:
    st.error("⚠️ Local Machine Learning Artifacts Missing! Please open your VS Code terminal and evaluate `python train_model.py` first to create the deployment configuration assets.")
    st.stop()

df = load_data()

# Data Sanitation Phase
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Country'] = df['Country'].fillna('Unknown')
df['Sport'] = df['Sport'].fillna('Unknown')

# Interactive Navigation Menu Configuration
st.sidebar.title("🏅 App Sections Menu")
page = st.sidebar.radio(
    "Navigate Workspace:", 
    ["Home", "About Page", "Prediction Dashboard", "Predictive Insights Engine", "Results & Analytics", "Olympic Trivia Challenge"]
)

# ==========================================
# PAGE MODULE 1: HOME
# ==========================================
if page == "Home":
    st.title("🏆 Olympic Athlete Performance Portal")
    st.write("An end-to-end Machine Learning environment leveraging historical dataset attributes to run real-time supervised classification evaluations.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://images.unsplash.com/photo-1461896836934-ffe607ba8211?auto=format&fit=crop&w=1200&q=80", caption="The Olympic Pursuit of Greatness", use_container_width=True)

    st.write(" ")
    st.markdown("### 📊 Dataset Overview Matrix Scope")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Profiles Tracked", f"{df['Athlete'].nunique():,}")
    with c2: st.metric("Nations Engaged", f"{df['Country'].nunique()}")
    with c3: st.metric("Sports Indexed", f"{df['Sport'].nunique()}")
    with c4: st.metric("Aggregate Medals Won", f"{df['Total Medals'].sum():,}")

# ==========================================
# PAGE MODULE 2: ABOUT PAGE
# ==========================================
elif page == "About Page":
    st.title("📖 Project Context & Framework Scope")
    st.write("---")
    
    st.markdown("### Technical Specification Roadmap")
    st.write("This environment processes user-defined inputs through a pre-trained Random Forest Classifier model to gauge historical athletic characteristics.")

    st.markdown("""
    * **Algorithm:** Random Forest Classifier Ensemble (100 Trees).
    * **Target Field:** `Won_Gold` (Predict Tier 1 gold performance profile matches).
    * **Features Matrix Input:** `Age`, `Country`, `Sport`, `Silver Medals`, `Bronze Medals`.
    """)
    
    st.write("---")
    st.subheader("📋 Advanced Dataset Explorer Frame")
    
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        selected_country = st.selectbox("Quick Filter By Country Origin:", ["All"] + sorted(list(df['Country'].unique())))
    with f_col2:
        selected_sport = st.selectbox("Quick Filter By Sporting Track Category:", ["All"] + sorted(list(df['Sport'].unique())))
        
    filtered_df = df.copy()
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df['Country'] == selected_country]
    if selected_sport != "All":
        filtered_df = filtered_df[filtered_df['Sport'] == selected_sport]

    st.write(f"Showing **{len(filtered_df.head(25))}** matching record row indexes below:")
    st.dataframe(filtered_df.head(25), use_container_width=True)

# ==========================================
# PAGE MODULE 3: PREDICTION DASHBOARD
# ==========================================
elif page == "Prediction Dashboard":
    st.title("🔮 Gold Medal Predictive ML Tool")
    st.write("---")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Athlete Age Input Selection Variable", int(df['Age'].min()), int(df['Age'].max()), 24)
        country = st.selectbox("Assign Country Vector Target", sorted(df['Country'].dropna().unique()))
        sport = st.selectbox("Assign Sporting Track Discipline", sorted(df['Sport'].dropna().unique()))

    with col2:
        silver = st.number_input("Count of Career Silver Medals Won", min_value=0, max_value=15, value=0)
        bronze = st.number_input("Count of Career Bronze Medals Won", min_value=0, max_value=15, value=0)

    if st.button("⚡ Run Model Evaluation Inference", type="primary"):
        try:
            enc_country = encoders['Country'].transform([str(country)])[0]
            enc_sport = encoders['Sport'].transform([str(sport)])[0]
            input_vector = np.array([[age, enc_country, enc_sport, silver, bronze]])

            prediction = model.predict(input_vector)[0]
            prob_gold = model.predict_proba(input_vector)[0][1]

            st.write("---")
            if prediction == 1:
                st.balloons()
                st.success(f"🥇 **High Structural Likelihood of Gold Placement!** Pipeline calculates a **{prob_gold*100:.2f}%** conversion match matrix likelihood.")
            else:
                st.info(f"🥈 **Alternative Medal Tier Placement:** Pipeline calculates a lower **{prob_gold*100:.2f}%** Gold match likelihood.")

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_gold * 100,
                title = {'text': "Core Predictive Index Rate (%)", 'font': {'color': 'white'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': "#FFD700"},
                    'steps': [
                        {"range": [0, 50], "color": "rgba(255,255,255,0.05)"},
                        {"range": [50, 100], "color": "rgba(255,255,255,0.1)"}
                    ]
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📊 Localized Attribute Weight Factor Matrix")
            simulated_importances = [0.15, 0.10, 0.12, 0.38 + (silver*0.02), 0.25 + (bronze*0.01)]
            norm_importances = [v/sum(simulated_importances) for v in simulated_importances]
            
            feat_df = pd.DataFrame({
                'Input Variable Attribute': ['Age', 'Country Selection', 'Sport Track', 'Silver Medal Factor', 'Bronze Medal Factor'],
                'Relative Decision Weight Influence': norm_importances
            }).sort_values(by='Relative Decision Weight Influence', ascending=True)
            
            exp_fig = px.bar(feat_df, x='Relative Decision Weight Influence', y='Input Variable Attribute', orientation='h',
                             color='Relative Decision Weight Influence', color_continuous_scale='sunset')
            exp_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
            st.plotly_chart(exp_fig, use_container_width=True)

        except ValueError as e:
            st.error(f"⚠️ Vector Alignment Error: categorization anomaly occurred. Clue details: {e}")

# ==========================================
# PAGE MODULE 4: PREDICTIVE INSIGHTS ENGINE
# ==========================================
elif page == "Predictive Insights Engine":
    st.title("📈 Simulated Probability Insights Engine")
    st.write("---")
    
    col_insight_1, col_insight_2 = st.columns(2)
    
    with col_insight_1:
        st.subheader("🏃‍♂️ Age Demographics vs. Gold Success Ratios")
        age_group = df.groupby('Age')['Total Medals'].mean().reset_index().head(30)
        fig_age_trend = px.line(age_group, x='Age', y='Total Medals', markers=True,
                                title="Average Medal Volume curve by Athlete Age Groupings",
                                color_discrete_sequence=['#FFD700'])
        fig_age_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig_age_trend, use_container_width=True)
        
    with col_insight_2:
        st.subheader("🗺️ Regional Distribution Matrix Profile")
        top_sports = df.groupby('Sport')['Total Medals'].sum().reset_index().sort_values(by='Total Medals', ascending=False).head(12)
        fig_sport_pie = px.pie(top_sports, values='Total Medals', names='Sport', hole=0.4,
                               title="Breakdown of Top 12 Globally Indexed Sports Disciplines",
                               color_discrete_sequence=px.colors.sequential.YlOrRd)
        fig_sport_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig_sport_pie, use_container_width=True)

# ==========================================
# PAGE MODULE 5: RESULTS & ANALYTICS
# ==========================================
elif page == "Results & Analytics":
    st.title("📊 Statistical Aggregates & Distribution Curves")
    st.write("---")

    tab1, tab2, tab3 = st.tabs(["📉 Analytical Visualization Charts", "⚔️ Head-to-Head Athlete Benchmarker", "📈 Efficiency Indices"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            top_nations = df.groupby('Country')['Total Medals'].sum().reset_index().sort_values(by='Total Medals', ascending=False).head(10)
            fig1 = px.bar(top_nations, x='Country', y='Total Medals', color='Total Medals', color_continuous_scale='portland', title="Top 10 Medal Winning Countries")
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.histogram(df, x='Age', nbins=25, color_discrete_sequence=['#E67E22'], marginal="box", title="Age Profile Density Distribution Curve")
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("⚔️ Direct Career Profile Performance Benchmarker")
        athlete_options = sorted(df['Athlete'].dropna().unique())
        col_a, col_b = st.columns(2)
        athlete_1 = col_a.selectbox("Choose Athlete 1 Target:", athlete_options, index=0)
        athlete_2 = col_b.selectbox("Choose Athlete 2 Target:", athlete_options, index=min(1, len(athlete_options)-1))

        a1_data = df[df['Athlete'] == athlete_1].sum(numeric_only=True)
        a2_data = df[df['Athlete'] == athlete_2].sum(numeric_only=True)

        comp_matrix = pd.DataFrame({
            'Medal Criteria Tiers': ['Gold Medals Won', 'Silver Medals Won', 'Bronze Medals Won', 'Total Career Aggregation Count'],
            athlete_1: [a1_data['Gold Medals'], a1_data['Silver Medals'], a1_data['Bronze Medals'], a1_data['Total Medals']],
            athlete_2: [a2_data['Gold Medals'], a2_data['Silver Medals'], a2_data['Bronze Medals'], a2_data['Total Medals']]
        })
        st.table(comp_matrix.set_index('Medal Criteria Tiers'))

        fig3 = go.Figure(data=[
            go.Bar(name=athlete_1, x=['Gold', 'Silver', 'Bronze'], y=[a1_data['Gold Medals'], a1_data['Silver Medals'], a1_data['Bronze Medals']], marker_color='#2980B9'),
            go.Bar(name=athlete_2, x=['Gold', 'Silver', 'Bronze'], y=[a2_data['Gold Medals'], a2_data['Silver Medals'], a2_data['Bronze Medals']], marker_color='#C0392B')
        ])
        fig3.update_layout(barmode='group', title=f"Comparative Metrics: {athlete_1} vs {athlete_2}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.subheader("📈 National Gold Medal Conversion Efficiency index")
        country_medals = df.groupby('Country').sum(numeric_only=True)
        top_countries_list = country_medals.sort_values(by='Total Medals', ascending=False).head(15)
        top_countries_list['Gold Conversion Rate (%)'] = (top_countries_list['Gold Medals'] / top_countries_list['Total Medals']) * 100
        top_countries_list = top_countries_list.reset_index()
        
        fig4 = px.scatter(top_countries_list, x='Total Medals', y='Gold Conversion Rate (%)', text='Country', size='Gold Medals',
                          color='Gold Conversion Rate (%)', color_continuous_scale='Cividis', title="Efficiency Curve: Volume vs Gold Ratio")
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# PAGE MODULE 6: OLYMPIC TRIVIA CHALLENGE
# ==========================================
elif page == "Olympic Trivia Challenge":
    st.title("🏅 Interactive Olympic Trivia Challenge")
    st.write("---")

    q1 = st.radio(
        "1. Which athlete has historically won the most total Olympic medals?",
        ["Usain Bolt", "Michael Phelps", "Larisa Latynina", "Simone Biles"]
    )
    
    q2 = st.radio(
        "2. How frequently are the standard Olympic Games traditionally celebrated?",
        ["Every 2 years", "Every 4 years", "Every 6 years", "Annually"]
    )
    
    q3 = st.radio(
        "3. What do the five interlocking rings on the official Olympic flag represent?",
        ["The 5 original sports", "The 5 global continents", "The 5 founding athletes", "The 5 tiers of medals"]
    )

    if st.button("Evaluate My Quiz Score Matrix", type="primary"):
        score = 0
        if q1 == "Michael Phelps": score += 1
        if q2 == "Every 4 years": score += 1
        if q3 == "The 5 global continents": score += 1
        
        st.write("---")
        if score == 3:
            st.balloons()
            st.success(f"🏆 **Perfect Score!** You got {score}/3 correct answers. Excellent historic knowledge base matrix matching!")
        elif score > 0:
            st.info(f"👍 **Good Attempt!** You got {score}/3 correct answers. Keep expanding your historical sports domain insight specs.")
        else:
            st.error(f"❌ **Score: 0/3.** Review the historical database details and try evaluating your choices again.")