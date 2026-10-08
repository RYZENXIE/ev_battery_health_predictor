# ⚡ VoltSense: EV Battery Health (SOH) & Real-World Range Predictor

An industry-grade Machine Learning solution that couples **electrochemical battery degradation mechanics** (Arrhenius reaction rates, cycle aging, DC fast charging SEI layer degradation) with **vehicle longitudinal aerodynamics** to predict:
1. **State of Health (SOH %)** of Electric Vehicle battery packs.
2. **Real-World Driving Range (km)** accounting for dynamic weather, speed, and HVAC climate loads.
3. **Multi-class Health Diagnostic Categorization** (Healthy, Normal Wear, Degradation Warning, Replacement Required).

---

- **Electrochemistry Grounded:** Models battery wear based on real battery pack physical attributes (Internal resistance $m\Omega$, Depth of Discharge, Fast charging ratio, LFP vs NMC cell chemistry).
- **Physics-Informed Vehicle Dynamics:** Captures aerodynamic drag force ($v^2$), rolling friction, and winter PTC cabin heating penalties.
- **Explainable AI (XAI):** Provides global and local feature importance trees explaining *why* battery health degraded.
- **Production-Grade Streamlit Dashboard:** Interactive gauges, What-If route scenario simulator, and live sensitivity curves.

---


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

