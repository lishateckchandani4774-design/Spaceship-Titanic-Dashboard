import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# Keep this class name and structure consistent with the notebook. It must be
# defined before loading the saved DecisionTreeScratch object.
class DecisionTreeScratch:
    def __init__(self, max_depth=6, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def gini(self, y):
        if len(y) == 0:
            return 0
        probabilities = np.bincount(np.asarray(y, dtype=int), minlength=2) / len(y)
        return 1 - np.sum(probabilities ** 2)

    def best_split(self, X, y):
        best_gain, best_feature, best_threshold = 0, None, None
        parent_gini = self.gini(y)
        n_samples, n_features = X.shape
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            if len(thresholds) > 30:
                thresholds = np.unique(np.quantile(thresholds, np.linspace(.05, .95, 20)))
            for threshold in thresholds:
                left = X[:, feature] <= threshold
                right = ~left
                if not left.any() or not right.any():
                    continue
                yl, yr = y[left], y[right]
                weighted = len(yl)/n_samples*self.gini(yl) + len(yr)/n_samples*self.gini(yr)
                gain = parent_gini - weighted
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold
        return best_feature, best_threshold, best_gain

    def build_tree(self, X, y, depth):
        ones = int(np.sum(y))
        zeros = len(y) - ones
        prediction = int(ones > zeros)
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return {"prediction": prediction}
        feature, threshold, gain = self.best_split(X, y)
        if feature is None or gain <= 0:
            return {"prediction": prediction}
        left = X[:, feature] <= threshold
        right = ~left
        return {"feature": feature, "threshold": threshold,
                "left": self.build_tree(X[left], y[left], depth + 1),
                "right": self.build_tree(X[right], y[right], depth + 1),
                "prediction": prediction}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        self.tree = self.build_tree(X, y, 0)
        return self

    def predict_one(self, sample, node):
        if "feature" not in node:
            return node["prediction"]
        if sample[node["feature"]] <= node["threshold"]:
            return self.predict_one(sample, node["left"])
        return self.predict_one(sample, node["right"])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self.predict_one(sample, self.tree) for sample in X])

st.set_page_config(page_title="SpaceVoyage Analytics", page_icon="🚀", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp{background:radial-gradient(circle at 10% 0%,rgba(88,120,255,.16),transparent 28%),radial-gradient(circle at 90% 5%,rgba(171,77,255,.13),transparent 25%),linear-gradient(180deg,#070b18,#0b1020 55%,#080c17);color:#f5f7ff;font-family:Inter,sans-serif}.block-container{max-width:1450px;padding-top:1.5rem}.hero{padding:35px 40px;border:1px solid #ffffff1a;border-radius:28px;margin-bottom:22px;background:radial-gradient(circle at 85% 30%,rgba(101,140,255,.3),transparent 25%),linear-gradient(135deg,#111a36,#10162b 55%,#17112e);box-shadow:0 20px 60px #0004}.hero small{color:#9db6ff;font-weight:800;letter-spacing:2px}.hero h1{font:700 clamp(34px,4vw,52px) 'Space Grotesk';margin:8px 0;color:white}.hero p{color:#bdc7e6;max-width:800px;line-height:1.7}.section{font:700 25px 'Space Grotesk';margin:28px 0 4px;color:white}.sub{color:#8e9abd;font-size:13px;margin-bottom:15px}.kpi{background:linear-gradient(145deg,#171f3af2,#0d1327f2);border:1px solid #91a6ff26;border-radius:18px;padding:18px;min-height:112px;box-shadow:0 12px 35px #0003}.kpi b{display:block;color:#8f9abb;font-size:11px;text-transform:uppercase;letter-spacing:.7px}.kpi strong{display:block;color:white;font:700 28px 'Space Grotesk';margin-top:8px}.kpi span{color:#7f8aaa;font-size:11px}.insight{background:linear-gradient(135deg,#1f2d53d9,#131931e6);border-left:4px solid #7895ff;border-radius:15px;padding:16px;color:#dce3ff;line-height:1.55}.prediction{border-radius:24px;padding:28px;text-align:center;margin-top:18px;background:linear-gradient(135deg,#192955,#24163d);border:1px solid #97abff40}.prediction .good{color:#9ef0c1;font:700 30px 'Space Grotesk'}.prediction .bad{color:#ffb3bd;font:700 30px 'Space Grotesk'}.prediction p{color:#aeb9da;font-size:13px}.stTabs [data-baseweb=tab-list]{gap:8px;background:#0a0f1ea8;border-radius:16px;padding:6px;border:1px solid #ffffff12}.stTabs [data-baseweb=tab]{height:48px;border-radius:11px;color:#8e9abd;font-weight:700;padding:0 22px}.stTabs [aria-selected=true]{background:linear-gradient(135deg,#536fdc,#7450bb);color:white!important}.stButton>button{width:100%;border:0;border-radius:12px;padding:12px;font-weight:700;color:white;background:linear-gradient(135deg,#5976ed,#7953bd)}[data-testid=stSidebar]{background:#080d1c;border-right:1px solid #ffffff12}.flow-card{padding:22px;border-radius:20px;background:linear-gradient(145deg,#121b35,#0d1327);border:1px solid #91a6ff26}.flow-step{padding:13px 14px;border-radius:12px;background:#1b2748;color:#e2e8ff;text-align:center;font-weight:600;margin:8px 0}.flow-step.final{background:linear-gradient(135deg,#334b96,#573b85);color:white}.small{color:#9aa7c8;font-size:11px}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("spaceship_titanic_cleaned.csv")

@st.cache_resource
def load_artifact(path):
    return joblib.load(path)

try:
    df = load_data()
except Exception as e:
    st.error("Could not load spaceship_titanic_cleaned.csv. Keep it in the same folder as app.py.")
    st.stop()

try:
    model = load_artifact("spaceship_titanic_decision_tree.pkl")
    preprocessor = load_artifact("spaceship_titanic_preprocessor.pkl")
    model_error = None
except Exception as e:
    model, preprocessor, model_error = None, None, str(e)

for c in ["CryoSleep", "VIP", "Transported"]:
    if c in df and df[c].dtype == "object":
        df[c] = df[c].map({"True": True, "False": False, "true": True, "false": False}).fillna(df[c])

st.markdown("<div class='hero'><small>MACHINE LEARNING</small><h1>🚀 SpaceVoyage Analytics</h1></div>", unsafe_allow_html=True)
analysis, prediction = st.tabs(["📊 Dataset & Analysis", "🤖 Model & Prediction"])

with analysis:
    st.sidebar.markdown("## 🎛️ Dashboard Filters")
    st.sidebar.caption("Filter the passenger population and explore the charts.")
    filtered = df.copy()
    filter_cols = {"HomePlanet":"🌍 Home Planet", "Destination":"🪐 Destination", "CryoSleep":"❄️ CryoSleep", "VIP":"⭐ VIP", "AgeGroup":"👤 Age Group", "Deck":"🛰️ Deck"}
    for col, label in filter_cols.items():
        if col not in df.columns: continue
        vals = sorted(df[col].dropna().astype(str).unique())
        selected = st.sidebar.multiselect(label, vals, default=vals, key="filter_" + col)
        if selected: filtered = filtered[filtered[col].astype(str).isin(selected)]
    st.sidebar.divider()
    st.sidebar.success(f"{len(filtered):,} passengers selected")
    total = len(filtered)
    transported = int(filtered["Transported"].sum()) if total else 0
    rate = transported / total * 100 if total else 0
    avg_spend = filtered["TotalSpending"].mean() if total and "TotalSpending" in filtered else 0
    st.markdown("<div class='section'>📌 Mission Overview</div><div class='sub'>Current filtered passenger population</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    kpis = [("👥 Passengers", f"{total:,}", "Current records"), ("🚀 Transported", f"{transported:,}", "Observed target = True"), ("📈 Transport Rate", f"{rate:.2f}%", "Selected passengers"), ("💳 Avg. Spending", f"{avg_spend:,.0f}", "TotalSpending")]
    for col, (label, value, note) in zip(cols, kpis):
        with col: st.markdown(f"<div class='kpi'><b>{label}</b><strong>{value}</strong><span>{note}</span></div>", unsafe_allow_html=True)

    st.markdown("<div class='section'>📊 Passenger Behaviour</div><div class='sub'>Charts update when filters are changed.</div>", unsafe_allow_html=True)
    def ratebar(group, title):
        z = filtered.groupby(group, dropna=False)["Transported"].mean().reset_index()
        z["Rate"] = z["Transported"] * 100
        z[group] = z[group].astype(str)
        fig = px.bar(z, x=group, y="Rate", text=z["Rate"].round(1), template="plotly_dark", title=title)
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(yaxis_title="Transport Rate (%)", height=350, margin=dict(l=10,r=10,t=55,b=20), showlegend=False)
        return fig
    a,b = st.columns(2)
    for col, group, title in [(a,"HomePlanet","🌍 Transport Rate by Home Planet"),(b,"CryoSleep","❄️ Transport Rate by CryoSleep")]:
        if group in filtered: col.plotly_chart(ratebar(group,title), use_container_width=True)
    a,b = st.columns(2)
    if "Destination" in filtered: a.plotly_chart(ratebar("Destination","🪐 Transport Rate by Destination"), use_container_width=True)
    if "AgeGroup" in filtered:
        age = filtered.groupby("AgeGroup", dropna=False)["Transported"].mean().reset_index(); age["Rate"] = age["Transported"]*100
        order = [x for x in ["Child","Teen","Young Adult","Adult","Senior"] if x in age["AgeGroup"].astype(str).values]
        age["AgeGroup"] = pd.Categorical(age["AgeGroup"], categories=order, ordered=True); age = age.sort_values("AgeGroup")
        fig = px.line(age, x="AgeGroup", y="Rate", markers=True, template="plotly_dark", title="👤 Transport Rate by Age Group")
        fig.update_layout(yaxis_title="Transport Rate (%)", height=350, margin=dict(l=10,r=10,t=55,b=20)); b.plotly_chart(fig, use_container_width=True)
    a,b = st.columns(2)
    if "Deck" in filtered: a.plotly_chart(ratebar("Deck","🛰️ Transport Rate by Deck"), use_container_width=True)
    if "Transported" in filtered and "TotalSpending" in filtered:
        sp = filtered.groupby("Transported")["TotalSpending"].mean().reset_index(); sp["Status"] = sp["Transported"].map({True:"Transported",False:"Not Transported"})
        fig = px.bar(sp, x="Status", y="TotalSpending", text=sp["TotalSpending"].round(0), template="plotly_dark", title="💳 Average Spending by Transportation Status")
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside"); fig.update_layout(yaxis_title="Average Total Spending", height=350, margin=dict(l=10,r=10,t=55,b=20)); b.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section'>🏆 Model Performance Comparison</div><div class='sub'>Results reported from your notebook experiments; values are percentages.</div>", unsafe_allow_html=True)
    comp = pd.DataFrame({"Algorithm":["Logistic Regression (scratch)","Decision Tree (scratch)","KNN (scratch)","Gaussian Naive Bayes (scratch)"],"Accuracy (%)":[77.29,78.26,77.29,74.35],"Precision (%)":[80.02,73.90,78.00,69.37],"Recall (%)":[73.17,87.90,76.48,87.90],"F1-Score (%)":[76.45,80.29,77.23,77.54]})
    st.dataframe(comp, hide_index=True, use_container_width=True)
    st.markdown("<div class='insight'><b>Selected model: Decision Tree (scratch)</b><br>It had the highest accuracy (78.26%) and F1-score (80.29%) among these four experiments. This is the best result in this experiment, not a guarantee that Decision Trees are always best.</div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>🗂️ Dataset Information</div>", unsafe_allow_html=True)
    a,b = st.columns([1,2])
    a.dataframe(pd.DataFrame({"Property":["Rows","Columns","Missing values","Target","Dashboard model"],"Value":[f"{len(df):,}",df.shape[1],int(df.isna().sum().sum()),"Transported","Custom Decision Tree"]}), hide_index=True, use_container_width=True)
    b.dataframe(df.head(10), hide_index=True, use_container_width=True)

with prediction:
    st.markdown("<div class='hero'><small>INTERACTIVE MACHINE LEARNING</small><h1>🤖 Passenger Prediction Centre</h1><p>Enter passenger details and use your custom Decision Tree to estimate the transportation outcome.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='flow-card'><div class='flow-step'>1️⃣ Passenger details</div><div class='flow-step'>2️⃣ Saved preprocessing pipeline</div><div class='flow-step'>3️⃣ Custom Decision Tree</div><div class='flow-step final'>🚀 Transported / 🌌 Not Transported</div></div>", unsafe_allow_html=True)
    if model is None or preprocessor is None:
        st.error("The custom model or preprocessing file could not be loaded. Put both spaceship_titanic_decision_tree.pkl and spaceship_titanic_preprocessor.pkl beside app.py.")
        st.caption(model_error or "Check that both files were saved from your notebook and that DecisionTreeScratch is defined in this app.")
    else:
        st.markdown("<div class='section'>🧑‍🚀 Passenger Details</div><div class='sub'>The feature names match the 14 input columns used in training.</div>", unsafe_allow_html=True)
        left,right = st.columns(2)
        with left:
            home = st.selectbox("🌍 Home Planet", sorted(df["HomePlanet"].dropna().astype(str).unique()))
            cryo = st.selectbox("❄️ CryoSleep", [False, True], format_func=lambda v: "Yes" if v else "No")
            dest = st.selectbox("🪐 Destination", sorted(df["Destination"].dropna().astype(str).unique()))
            age = st.number_input("👤 Age", min_value=0.0, max_value=100.0, value=28.0, step=1.0)
            vip = st.selectbox("⭐ VIP", [False, True], format_func=lambda v: "Yes" if v else "No")
            deck = st.selectbox("🛰️ Deck", sorted(df["Deck"].dropna().astype(str).unique()))
            side = st.selectbox("🚪 Cabin Side", sorted(df["CabinSide"].dropna().astype(str).unique()))
        with right:
            room = st.number_input("🛏️ Room Service", min_value=0.0, value=0.0, step=10.0)
            food = st.number_input("🍽️ Food Court", min_value=0.0, value=0.0, step=10.0)
            mall = st.number_input("🛍️ Shopping Mall", min_value=0.0, value=0.0, step=10.0)
            spa = st.number_input("💆 Spa", min_value=0.0, value=0.0, step=10.0)
            vr = st.number_input("🥽 VR Deck", min_value=0.0, value=0.0, step=10.0)
            total_spending = room + food + mall + spa + vr
            age_group = "Child" if age < 13 else "Teen" if age < 18 else "Young Adult" if age < 30 else "Adult" if age < 60 else "Senior"
            st.info(f"Total spending: {total_spending:,.0f}\n\nAge group: {age_group}")
        if st.button("🚀 Predict Transportation", use_container_width=True):
            inp = pd.DataFrame([{"HomePlanet":home,"CryoSleep":cryo,"Destination":dest,"Age":age,"VIP":vip,"RoomService":room,"FoodCourt":food,"ShoppingMall":mall,"Spa":spa,"VRDeck":vr,"Deck":deck,"CabinSide":side,"TotalSpending":total_spending,"AgeGroup":age_group}])
            # The notebook's preprocessing pipeline expects boolean features as strings.
            for col in inp.select_dtypes(include=["bool"]).columns:
                inp[col] = inp[col].astype(str)
            try:
                transformed = preprocessor.transform(inp)
                pred = int(model.predict(transformed)[0])
                if pred == 1:
                    st.markdown("<div class='prediction'><div style='font-size:45px'>🚀</div><div class='good'>TRANSPORTED</div><p>The custom Decision Tree predicts that this passenger was transported.</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='prediction'><div style='font-size:45px'>🌌</div><div class='bad'>NOT TRANSPORTED</div><p>The custom Decision Tree predicts that this passenger was not transported.</p></div>", unsafe_allow_html=True)
                st.caption("This is a class prediction only. The custom model does not provide calibrated probability/confidence scores.")
            except Exception as e:
                st.error("Prediction failed. Check that the preprocessor was saved from the same notebook and trained with these 14 features.")
                st.code(str(e))

st.markdown("<hr><div class='small'>SpaceVoyage Analytics • Python • Pandas • Plotly • Joblib • Streamlit</div>", unsafe_allow_html=True)