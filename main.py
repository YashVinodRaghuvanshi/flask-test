import numpy as np
from flask import Flask, request, jsonify
import pickle

model = pickle.load(open('regress.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello World'


@app.route('/yash', methods= ['POST'])
def predict():
    experience = request.form.get('exp')
    input_query = np.array([[experience]], dtype=float)

    result  = model.predict(input_query)[0]
    return jsonify({"Salary":str(result)})

if __name__ == '__main__':
    app.run(debug=True)
