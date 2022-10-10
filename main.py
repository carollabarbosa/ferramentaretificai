from flask import Flask, render_template, request, jsonify
import joblib
import sklearn.externals as extjoblib
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
ps = PorterStemmer()

# Load model and vectorizer


model = pickle.load(open('model2.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect2.pkl', 'rb'))

with open('model2.pkl', 'rb') as handle:
    model = pickle.load(handle)
  

# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
  
@app.route('/text')
def text():
   text= pd.read_csv('preprocessed05.csv')
   text_fake = 'text_falso'['label'].sum()
   text_true = 'text_verdadeiro'['label'].sum()
   resposta = {'text_verdadeiro': text_true}
   resposta = {'text_falso': text_fake}
   return jsonify (resposta)

def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('portuguese')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    return prediction
@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)
@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)
if __name__ == "__main__":
    app.run()
