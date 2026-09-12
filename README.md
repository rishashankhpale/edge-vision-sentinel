# Hi, I'm Risha👋
### Machine Learning Engineer | Tabular ML • Agentic GenAI • MLOps • Edge Vision

Engineering end-to-end, production-grade machine learning pipelines with a strong focus on low-latency inference, model drift observability, agentic self-healing workflows, and hardware-optimized edge deployment.

---

## 🚀 Featured Machine Learning Projects

### 🛡️ [Sentinel: Real-Time Fraud & MLOps Drift Engine](https://github.com/rishashankhpale/fraud-drift-sentinel)
> **Stack:** Python, LightGBM, Isolation Forest, SciPy, Streamlit, Plotly  
* **Dual-Layer Scoring:** Cost-sensitive LightGBM paired with an unsupervised Isolation Forest for zero-day behavioral anomalies on imbalanced data (~3.5% fraud rate) with sub-50ms inference.
* **Continuous MLOps Telemetry:** Live statistical drift detection computing Two-sample Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI) to trigger automated degradation warnings (`STABLE`, `WARNING`, `CRITICAL`).
* **Live Demo:** [Sentinel Web App](https://share.streamlit.io/) *(replace with your Streamlit Cloud URL)* | **Source:** [GitHub](https://github.com/rishashankhpale/fraud-drift-sentinel)

---

### 👁️ [Edge Vision Sentinel: Spatial Safety & Intrusion Engine](https://github.com/rishashankhpale/edge-vision-sentinel)
> **Stack:** Python, YOLOv8, ONNX Runtime, OpenCV, Streamlit  
* **Edge CPU Acceleration:** Exported YOLOv8 weights into a static ONNX graph executed via multi-threaded `onnxruntime` (`intra_op_num_threads=4`) for real-time CPU throughput.
* **Spatial Polygon Analytics:** Uses Ray-Casting (`pointPolygonTest`) anchored to bottom-center ground contact points to eliminate false-positive perimeter alerts from bounding box overlap.
* **Live Demo:** [Vision Web App](https://share.streamlit.io/) *(replace with your Streamlit Cloud URL)* | **Source:** [GitHub](https://github.com/rishashankhpale/edge-vision-sentinel)

---

### 🤖 [Autonomous AI SQL Agent & Business Analyst](https://github.com/rishashankhpale/ai-sql-agent)
> **Stack:** Python, Google Gemini Flash, SQLite, Pandas, Plotly, Streamlit  
* **Self-Healing Agentic Loop:** Catches runtime SQL syntax and constraint exceptions in an execution sandbox, autonomously feeding stack traces back into the model to heal and retry queries.
* **Dynamic Schema Introspection:** Queries catalog metadata (`sqlite_master`) at runtime to dynamically ground prompts with exact schema relationships.
* **Live Demo:** [SQL Agent Web App](https://share.streamlit.io/) *(replace with your Streamlit Cloud URL)* | **Source:** [GitHub](https://github.com/rishashankhpale/ai-sql-agent)

---

### 🌾 [Agricultural Yield & Climate Prediction Platform](https://github.com/rishashankhpale/agri-yield-predictor)
> **Stack:** Python, Scikit-Learn, Random Forest, Pandas, Streamlit Cloud  
* **Leakage-Free Pipeline:** Complete regression pipeline encapsulating `ColumnTransformer` preprocessing and hyperparameter-tuned `RandomForestRegressor` ($R^2 \approx 0.85+$).
* **Live Demo:** [Agri Predictor Web App](https://share.streamlit.io/) *(replace with your Streamlit Cloud URL)* | **Source:** [GitHub](https://github.com/rishashankhpale/agri-yield-predictor)

---

## 🛠️ Technical Stack
* **Languages & Core:** Python, SQL, C++, Bash
* **Machine Learning & MLOps:** Scikit-Learn, LightGBM, ONNX Runtime, Drift Monitoring (KS-Test, PSI), MLflow
* **Deep Learning & Vision:** PyTorch, Ultralytics YOLOv8, OpenCV
* **GenAI & Agentic Systems:** Google Gemini API, Prompt Engineering, Autonomous Feedback Loops
* **Deployment & Tooling:** Streamlit Community Cloud, Git, GitHub, Docker (basics)
