# ⚡ VoltSense: EV Battery Health (SOH) & Real-World Range Predictor

An industry-grade Machine Learning solution that couples **electrochemical battery degradation mechanics** (Arrhenius reaction rates, cycle aging, DC fast charging SEI layer degradation) with **vehicle longitudinal aerodynamics** to predict:
1. **State of Health (SOH %)** of Electric Vehicle battery packs.
2. **Real-World Driving Range (km)** accounting for dynamic weather, speed, and HVAC climate loads.
3. **Multi-class Health Diagnostic Categorization** (Healthy, Normal Wear, Degradation Warning, Replacement Required).

---

## 🎯 Why This Project Stands Out (Anti-Cliché Highlights)
Unlike generic student projects (e.g., Titanic, House Prices, Iris):
- **Electrochemistry Grounded:** Models battery wear based on real battery pack physical attributes (Internal resistance $m\Omega$, Depth of Discharge, Fast charging ratio, LFP vs NMC cell chemistry).
- **Physics-Informed Vehicle Dynamics:** Captures aerodynamic drag force ($v^2$), rolling friction, and winter PTC cabin heating penalties.
- **Explainable AI (XAI):** Provides global and local feature importance trees explaining *why* battery health degraded.
- **Production-Grade Streamlit Dashboard:** Interactive gauges, What-If route scenario simulator, and live sensitivity curves.

---

## 🏗️ Project Architecture
```
ev_battery_health_predictor/
│
├── generate_dataset.py        # Generates realistic EV telemetry dataset based on battery physics
├── train_model.py             # Preprocessing pipeline, model training (GBM & RF), evaluation metrics
├── app.py                     # Interactive Streamlit Web Application
├── requirements.txt           # Python dependencies
├── ev_battery_telemetry.csv   # 6,000-record generated telemetry dataset
├── saved_models/              # Exported pipelines & evaluation metrics
│   ├── soh_predictor.joblib
│   ├── range_predictor.joblib
│   ├── status_classifier.joblib
│   └── metrics.json
└── README.md                  # Project documentation & Viva Q&A
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Models
```bash
python generate_dataset.py
python train_model.py
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Model Performance Metrics
- **SOH Regressor:** Gradient Boosting Regressor | **R² Score: ~0.98**, MAE: < 0.6%
- **Range Regressor:** Gradient Boosting Regressor | **R² Score: ~0.98**, MAE: < 4.5 km
- **Diagnostic Classifier:** Random Forest Classifier | **Accuracy: > 94%**

---

## 🎓 Viva & Presentation Q&A for Evaluators

**Q1: What is Battery State of Health (SOH)?**
> **Ans:** SOH is the ratio of the battery's current full-charge capacity to its initial rated capacity:
> $$\text{SOH} = \frac{C_{\text{current}}}{C_{\text{rated}}} \times 100\%$$
> When SOH reaches 70–80%, the battery is retired from EV traction and repurposed for secondary ESS (Energy Storage Systems).

**Q2: Why did you choose Gradient Boosting over standard Linear Regression?**
> **Ans:** Linear regression assumes additive independent effects. In EV dynamics, features exhibit heavy non-linear interactions. For instance, aerodynamic drag scales quadratically ($v^2$) with speed, and cold temperatures compound the impact of HVAC heater draws. Gradient Boosted Decision Trees naturally model these non-linear feature splits without severe overfitting.

**Q3: How does temperature affect EV range?**
> **Ans:** Lithium-ion batteries experience reduced electrolyte conductivity below 15°C, increasing internal cell resistance. Furthermore, electric vehicles do not have waste engine heat, so cabin warming relies on high-voltage PTC heaters or heat pumps, which draw 2–5 kW of continuous battery power, reducing cold-weather driving range by up to 30%.
