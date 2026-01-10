from flask import Flask, request, jsonify, render_template_string
import joblib
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# --- NLTK DATA SETUP ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# --- NLP PREPROCESSING UTILITIES ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# --- LOAD MODEL ---
MODEL_PATH = 'sentiment_model.joblib'
VECTORIZER_PATH = 'tfidf_vectorizer.joblib'
REPORT_PATH = 'model_evaluation_report.txt'

def get_accuracy():
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, 'r') as f:
                content = f.read()
                # Find the best accuracy line
                matches = re.findall(r'(\w+ \w+|Naive Bayes) Accuracy: (\d+\.\d+)', content)
                if matches:
                    # Find the highest accuracy from the list
                    return max([float(m[1]) for m in matches])
        except:
            pass
    return 0.82  # Default fallback if report not found

@app.route('/')
def home():
    accuracy = get_accuracy()
    return render_template_string(HTML_TEMPLATE, accuracy=f"{accuracy*100:.2f}%")

@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return jsonify({'error': 'Model files not found. Please run train.py first.'}), 400
    
    data = request.json
    tweet = data.get('tweet', '')
    
    if not tweet:
        return jsonify({'error': 'No tweet provided.'}), 400
    
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    
    cleaned = clean_text(tweet)
    vec = tfidf.transform([cleaned])
    
    # Get Prediction
    prediction = model.predict(vec)[0]
    
    # Calculate Confidence
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec)[0]
            confidence = max(probs)
        elif hasattr(model, "decision_function"):
            # For models like LinearSVC that don't have predict_proba by default
            scores = model.decision_function(vec)[0]
            import numpy as np
            e_x = np.exp(scores - np.max(scores))
            probs = e_x / e_x.sum()
            confidence = max(probs)
        else:
            confidence = 0.0
    except:
        confidence = 0.0
    
    return jsonify({
        'sentiment': prediction.upper(),
        'confidence': f"{confidence*100:.1f}%"
    })

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment AI - Premium Analysis</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #059669; /* Professional Emerald Green */
            --primary-glow: rgba(5, 150, 105, 0.2);
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #064e3b; /* Dark Green Text */
            --subtitle: #475569;
        }

        body {
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #ecfdf5 0%, #f1f5f9 100%);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            position: relative;
        }

        body::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("data:image/svg+xml,%3Csvg width='120' height='120' viewBox='0 0 120 120' xmlns='http://www.w3.org/2000/svg'%3E%3Ctext x='10' y='30' font-size='16' opacity='0.2'%3E😊%3C/text%3E%3Ctext x='70' y='40' font-size='14' opacity='0.15'%3E😐%3C/text%3E%3Ctext x='40' y='80' font-size='18' opacity='0.2'%3E☹️%3C/text%3E%3Ctext x='90' y='95' font-size='15' opacity='0.18'%3E😍%3C/text%3E%3Ctext x='15' y='110' font-size='12' opacity='0.12'%3E😡%3C/text%3E%3Ctext x='60' y='15' font-size='14' opacity='0.15'%3E😴%3C/text%3E%3Ctext x='100' y='50' font-size='16' opacity='0.2'%3E🤔%3C/text%3E%3C/svg%3E");
            filter: grayscale(100%);
            opacity: 0.2; 
            pointer-events: none;
            z-index: 0;
        }

        .emoji-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
        }

        .emoji-particle {
            position: absolute;
            font-size: 2rem;
            animation: floatUp 2s ease-out forwards;
            opacity: 0;
        }

        @keyframes floatUp {
            0% {
                transform: translateY(0) scale(0.5) rotate(0deg);
                opacity: 0;
            }
            20% {
                opacity: 1;
            }
            100% {
                transform: translateY(-200px) scale(1.5) rotate(45deg);
                opacity: 0;
            }
        }

        .container {
            width: 90%;
            max-width: 500px;
            background: var(--card-bg);
            border: 1px solid rgba(255, 255, 255, 0.8);
            border-radius: 32px;
            padding: 48px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
            animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            z-index: 10;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1 {
            font-weight: 700;
            margin-bottom: 8px;
            color: #064e3b;
            text-align: center;
            letter-spacing: -0.02em;
        }

        .subtitle {
            text-align: center;
            color: var(--subtitle);
            font-size: 1.1rem;
            margin-bottom: 32px;
        }

        .stats-badge {
            display: inline-block;
            background: #f0fdf4;
            color: var(--primary);
            padding: 6px 16px;
            border-radius: 99px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 24px;
            border: 1px solid #dcfce7;
        }

        .input-group {
            margin-bottom: 24px;
        }

        textarea {
            width: 100%;
            background: #f1f5f9;
            border: 2px solid transparent;
            border-radius: 16px;
            padding: 20px;
            color: var(--text);
            font-family: inherit;
            font-size: 1rem;
            resize: none;
            box-sizing: border-box;
            transition: all 0.2s ease;
        }

        textarea:focus {
            outline: none;
            background: #ffffff;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px var(--primary-glow);
        }

        button {
            width: 100%;
            background: var(--primary);
            color: white;
            border: none;
            padding: 18px;
            border-radius: 16px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 10px 15px -3px var(--primary-glow);
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 25px -5px var(--primary-glow);
            filter: brightness(1.05);
        }

        button:active { transform: translateY(0); }

        #result {
            margin-top: 32px;
            padding: 40px 20px;
            border-radius: 40px;
            display: none;
            animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
            text-align: center;
            transition: all 0.3s ease;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); scale: 0.95; }
            to { opacity: 1; transform: translateY(0); scale: 1; }
        }

        .sentiment-text {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 8px;
            letter-spacing: -0.03em;
            line-height: 1;
        }

        .label { 
            font-size: 0.85rem; 
            color: #64748b; 
            text-transform: uppercase; 
            letter-spacing: 3px; 
            font-weight: 700; 
            margin-bottom: 24px;
            opacity: 0.8;
        }

        .confidence-text {
            font-size: 1.1rem;
            font-weight: 500;
            margin-top: 16px;
        }

        .confidence-value {
            font-weight: 700;
        }

        .positive { background: #f0fdf4 !important; color: #15803d !important; }
        .negative { background: #fef2f2 !important; color: #b91c1c !important; }
        .neutral { background: #f8fafc !important; color: #475569 !important; }
    </style>
</head>
<body>
    <div class="emoji-overlay" id="emojiOverlay"></div>
    <div class="container">
        <center><div class="stats-badge">Welcome!</div></center>
        <h1>SeAS AI</h1>
        <p class="subtitle">Enter a tweet to analyze its emotional impact</p>
        
        <div class="input-group">
            <textarea id="tweetInput" rows="3" placeholder="I absolutely loved my journey today!"></textarea>
        </div>
        
        <button onclick="analyzeSentiment()" id="btnText">Analyze Sentiment</button>

        <div id="result">
            <div class="label">Analysis Result</div>
            <div id="sentimentValue" class="sentiment-text">---</div>
            <div class="confidence-text">
                Confidence: <span id="confidenceValue" class="confidence-value">--</span>
            </div>
        </div>
    </div>

    <script>
        function createEmojiBlast(sentiment) {
            const overlay = document.getElementById('emojiOverlay');
            const emojis = {
                'POSITIVE': ['😊', '😍', '🎉', '✨', '💖', '🌟'],
                'NEGATIVE': ['😢', '😠', '😒', '😞', '💔', '🌧️'],
                'NEUTRAL': ['😐', '🤔', '😶', '💬', '⚖️', '☁️']
            };
            
            const selectedSet = emojis[sentiment] || emojis['NEUTRAL'];
            
            for (let i = 0; i < 20; i++) {
                const span = document.createElement('span');
                span.className = 'emoji-particle';
                span.innerText = selectedSet[Math.floor(Math.random() * selectedSet.length)];
                
                // Random position
                span.style.left = Math.random() * 100 + 'vw';
                span.style.top = (Math.random() * 50 + 50) + 'vh';
                
                // Random timing
                span.style.animationDelay = (Math.random() * 0.5) + 's';
                
                overlay.appendChild(span);
                
                // Cleanup
                setTimeout(() => span.remove(), 2500);
            }
        }

        async function analyzeSentiment() {
            const tweet = document.getElementById('tweetInput').value;
            const resultDiv = document.getElementById('result');
            const sentimentValue = document.getElementById('sentimentValue');
            const confidenceValue = document.getElementById('confidenceValue');
            const btn = document.getElementById('btnText');

            if (!tweet) return;

            btn.disabled = true;
            btn.innerText = 'Analyzing...';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ tweet })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert(data.error);
                } else {
                    resultDiv.style.display = 'block';
                    sentimentValue.innerText = data.sentiment;
                    confidenceValue.innerText = data.confidence;
                    
                    // Style based on sentiment
                    resultDiv.className = '';
                    const s = data.sentiment.toLowerCase();
                    if (s.includes('pos')) resultDiv.classList.add('positive');
                    else if (s.includes('neg')) resultDiv.classList.add('negative');
                    else resultDiv.classList.add('neutral');

                    // Blast emojis!
                    createEmojiBlast(data.sentiment);
                }
            } catch (e) {
                alert('Connection error. Is the server running?');
            } finally {
                btn.disabled = false;
                btn.innerText = 'Analyze Sentiment';
            }
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True, port=5000)
