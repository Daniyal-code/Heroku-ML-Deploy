import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # to take all inputs from input fields
    int_features = [int(x) for x in request.form.values()]
    # converting them into array
    final_features = [np.array(int_features)]
    # predicting
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Employee Salary should be $ {output}')


if __name__ == '__main__':
    app.run(debug=True)
