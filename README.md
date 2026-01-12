# SeAS AI: Airline Sentiment Analysis System

A Python-based sentiment analysis system designed to recognize and examine emotional states in airline-related tweets. This project utilizes machine learning techniques to classify text into **Positive**, **Negative**, or **Neutral** polarities.

<img width="1891" height="1020" alt="Image" src="https://github.com/user-attachments/assets/402f15aa-81d5-4614-8413-a604431c91c7" />
<img width="1892" height="1021" alt="Image" src="https://github.com/user-attachments/assets/fd5f90c3-de26-40df-a597-cfdfdd87be05" />

## 🚀 Features
- **Triple Model Training**: Evaluates Naive Bayes, SVM, and Logistic Regression to select the best performer.
- **Advanced NLP**: Uses TF-IDF with Bigrams to understand context (e.g., "don't love").
- **Interactive EDA**: Generates 6 different visual reports including Confusion Matrices and Distribution plots.
- **Premium Web Interface**: 
  - **Emerald Green Theme**: A professional, high-end data dashboard look.
  - **Infographic Results**: Large, bold sentiment display with confidence scoring.
  - **Minimalist B&W Background**: A subtle, scattered emoji pattern for a tech-focused aesthetic.
  - **Dynamic Emoji Blast**: Real-time background animations that change based on the analysis result.

---

## 🛠️ Prerequisites
Ensure you have Python 3.8+ installed. You will need to install the following libraries:

```powershell
pip install pandas numpy matplotlib seaborn nltk scikit-learn joblib flask
```

---

## 📂 Project Structure
- `train.py`: The main pipeline for data cleaning, training, and evaluation.
- `demo.py`: The Flask-based web application for real-time demonstrations.
- `train_test/twitter.csv`: The dataset used for training and testing.
- `train_test/test/test_twitter.csv`: we can also use this dataset for testing.
- `model_evaluation_report.txt`: Automatically generated accuracy reports.

---

## 📖 How to Run

### Step 1: Train the Models
Run the training script to process the data, generate visual reports, and save the best model.
```powershell
python train.py
```
*Note: During execution, 6 graph windows will pop up. Close each window to proceed to the next step of the pipeline.*

### Step 2: Start the Web Demo
Once training is complete, launch the professional web interface for your presentation.
```powershell
python demo.py
```

### Step 3: Access the Application
Open your web browser and go to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🧪 Testing the Demo
Try entering these samples to see the system in action:
- **Positive**: "The service was amazing and the crew was so helpful! 😍"
- **Negative**: "My flight was delayed for 5 hours and they lost my luggage. 😡"
- **Neutral**: "Can someone tell me if flight UA123 has departed yet? 😐"
- https://github.com/user-attachments/assets/36c1dc0e-a809-4bbf-a4dd-01043eb2d63d

---

## 👥 Contributors
- Mohamed Nawran(mhdnawran4@gmail.com), Khalid, Mohamed Afrath (mohamednaseermohamedafrath@gmail.com)


