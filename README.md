# ⚡ AI-Powered Energy Consumption Forecasting System

## 🌍 Overview

The **AI-Powered Energy Consumption Forecasting System** is an end-to-end machine learning project designed to predict future electricity usage using historical energy data. This project demonstrates how Artificial Intelligence can be leveraged to optimize energy consumption, reduce operational costs, and support sustainable development.

Built with Python and modern data science tools, the system simulates real-world energy forecasting scenarios used in smart cities, power utilities, and renewable energy sectors. It also features an interactive dashboard that enables users to visualize data and generate forecasts effortlessly.

---

## 🎯 Problem Statement

Efficient energy management remains a global challenge. Organizations often face:

* ⚠️ Unpredictable energy demand leading to blackouts or wastage
* 💸 High electricity bills due to inefficient planning
* 🏭 Excessive energy consumption in industries and buildings
* 🌱 Increased carbon emissions from overproduction of power
* 📉 Mismatch between energy supply and consumption
* 📊 Lack of data-driven decision-making tools

### 💡 Solution

This project utilizes Machine Learning to forecast energy consumption accurately. By analyzing historical usage patterns, it enables smarter decisions that enhance efficiency, reduce costs, and contribute to sustainable energy management.

---

## 🚀 Key Features

* 📊 Exploratory Data Analysis (EDA) for valuable insights
* 🧹 Data preprocessing and feature engineering for time-series forecasting
* 🤖 Machine learning models including Linear Regression and Random Forest
* 🔮 Future energy consumption prediction
* 📈 Interactive visualizations using Plotly
* 🖥️ Upgraded Streamlit dashboard for real-time forecasting
* 📥 Downloadable prediction reports
* 🌐 Deployment-ready architecture
* 📁 GitHub-ready professional project structure

---

## 🛠️ Tech Stack

| Category              | Technologies Used           |
| --------------------- | --------------------------- |
| Programming Language  | Python                      |
| Data Analysis         | Pandas, NumPy               |
| Visualization         | Matplotlib, Seaborn, Plotly |
| Machine Learning      | Scikit-learn                |
| Model Serialization   | Joblib                      |
| Dashboard Development | Streamlit                   |
| Version Control       | Git & GitHub                |
| Development Tools     | Jupyter Notebook, VS Code   |

---

## 📂 Project Structure

```
AI-Powered-Energy-Consumption-Forecasting-System/
│
├── data/                      # Dataset files
├── notebooks/                 # Jupyter notebooks for analysis
├── src/                       # Source code modules
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── forecasting.py
│
├── models/                    # Trained machine learning models
├── outputs/                   # Forecast results and metrics
├── images/                    # Visualizations and screenshots
├── app.py                     # Streamlit dashboard
├── main.py                    # Main execution script
├── requirements.txt           # Project dependencies
├── .gitignore                 # Ignored files
└── README.md                  # Project documentation
```

---
## 📊 Dataset

This project uses the Individual Household Electric Power Consumption Dataset from the UCI Machine Learning Repository.

🔗 Dataset Link:
https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

Due to GitHub's file size limitations, the dataset is not included in this repository.

### Steps to Use the Dataset
1. Download the dataset from the link above.
2. Place it inside the `data/` directory.
3. Run the preprocessing scripts to generate the required files.

### Steps to Use the Dataset

1. Download the dataset from the link above.
2. Place it inside the `data/` directory.
3. Run the preprocessing scripts to generate the required files.

---

## ⚙️ Installation and Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/tanusoul/AI-Powered-Energy-Consumption-Forecasting-System.git
cd AI-Powered-Energy-Consumption-Forecasting-System
```

### 2️⃣ Create and Activate a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### Step 1: Perform Feature Engineering

```bash
python src/feature_engineering.py
```

### Step 2: Train Machine Learning Models

```bash
python src/model_training.py
```

### Step 3: Generate Forecasts

```bash
python src/forecasting.py
```

### Step 4: Launch the Interactive Dashboard

```bash
streamlit run app.py
```

Open the dashboard in your browser:

```
http://localhost:8501
```

---

## 📈 Model Performance

| Model                   | Description                                        |
| ----------------------- | -------------------------------------------------- |
| Linear Regression       | Provides a simple and interpretable baseline model |
| Random Forest Regressor | Captures complex patterns for improved accuracy    |

### Evaluation Metrics

* Root Mean Squared Error (RMSE)
* R² Score (Coefficient of Determination)

---

## 🖥️ Dashboard Features

* 📊 Visualizes historical energy consumption
* 🔮 Predicts future energy demand
* 📈 Compares machine learning models
* 📅 Allows custom forecasting horizons
* 📥 Enables CSV downloads of predictions
* 🎨 Provides an interactive and user-friendly interface

---
## 📸 Project Screenshots

### Dashboard Overview
![Dashboard](images/dashboard_home.png)

### Visualizations
![Visualizations](images/dashboard_visualizations.png)

### Forecasting Output
![Forecast](images/dashboard_forecast.png)

### Model Insights
![Model Insights](images/dashboard_model_insights.png)

---

## 🌐 Deployment

### Local Deployment

```bash
streamlit run app.py
```

### Cloud Deployment

Deploy easily using **Streamlit Community Cloud** by connecting your GitHub repository.

---

## 🎓 Learning Outcomes

* Applied machine learning to a real-world energy forecasting problem
* Gained hands-on experience in time-series analysis and feature engineering
* Built an end-to-end AI system from data preprocessing to deployment
* Developed an interactive dashboard for business intelligence
* Learned GitHub best practices for professional portfolio development
* Explored AI applications in climate tech and smart cities

---

## 💼 Industry Applications

* 🏙️ Smart Cities
* ⚡ Power and Utility Companies
* 🏭 Manufacturing Industries
* 🖥️ Data Centers
* 🌞 Renewable Energy Organizations
* 🏢 Green Buildings and Energy Management Systems

---

## 👩‍💻 Author

**Tanuja**
B.Tech in Computer Science Engineering (AI & ML)
Aspiring Data Scientist | Machine Learning Engineer | AI Enthusiast

🔗 GitHub: https://github.com/tanusoul

---

## 🤝 Contributing

Contributions and suggestions are welcome! Feel free to fork this repository and submit a pull request.

---

## 📜 License

This project is intended for educational and research purposes only.

---

## ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub!
