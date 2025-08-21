# 📊 Sales Prediction Model

This project is a **Sales Prediction Web App** built using **Linear Regression** for machine learning and **Streamlit** for an interactive user interface. The model helps predict sales based on given input features, making it useful for businesses and analysts looking to make **data-driven decisions**.

---

## 🚀 Features

* ✅ **Linear Regression Model** for sales prediction
* ✅ **Interactive Streamlit UI** for easy input and output
* ✅ **Data visualization** for better insights
* ✅ **User-friendly design** – no coding required to use

---

## 🛠️ Tech Stack

* **Python** 🐍
* **Pandas & NumPy** – Data handling & preprocessing
* **Scikit-learn** – Machine learning (Linear Regression)
* **Matplotlib / Seaborn** – Data visualization
* **Streamlit** – Interactive web application

---

## 📂 Project Structure

```
Sales-Prediction-Model/
│── data/                # Dataset used for training/testing
│── model/               # Saved trained model (pickle file)
│── app.py               # Main Streamlit app
│── train_model.py       # Script to train and save the model
│── requirements.txt     # Required dependencies
│── README.md            # Project documentation
```

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/Sales-Prediction-Model.git
cd Sales-Prediction-Model
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

### 4️⃣ Open in browser

The app will run on:
👉 `http://localhost:8501/`

---

## 📸 Screenshots

### 🔹 Home Page

<img src="https://github.com/user-attachments/assets/fa5ebeff-22bb-4797-a3ab-a060dabfdcc9" width="500"/>  

### 🔹 Prediction Page

<img src="https://github.com/user-attachments/assets/f2e89dcf-6276-4ac0-9bb8-824eb93c04ca" width="500"/>  

---

## 📈 How It Works

1. Load dataset & preprocess features
2. Train a **Linear Regression model**
3. Save trained model (Pickle)
4. Deploy using **Streamlit**
5. User inputs data → Model predicts sales

---

## 🔮 Future Improvements

* Add support for more ML algorithms (Random Forest, XGBoost)
* Improve UI with advanced visualizations
* Deploy app on **Streamlit Cloud / Heroku** for public access
