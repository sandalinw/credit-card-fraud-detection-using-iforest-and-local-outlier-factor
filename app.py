

from flask import Flask, request,  render_template
import pickle
import numpy as np
import pandas as pd
import random
import sklearn


app = Flask(__name__, template_folder='template')


model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    print('Get data from client:', dict(request.get_json(force=True)))
    data = dict(request.get_json(force=True))

    # loading model

    with open('model.pkl', 'rb') as file:

        pickle_model = pickle.load(file)

    # loading csv file
    xTest = pd.read_csv('creditcard.csv')

    try:
        time_data = float(data['time'].strip())
        amount_data = float(data['amount'].strip())

    except ValueError as e:
        return "invalid input"

    # get the matching row
    pca_credit = xTest[(xTest['Time'] == float(data['time'])) & (xTest['Amount'] == float(data['amount']))]

    if len(pca_credit) == 0:
        return "invalid data"

    final = np.array(pca_credit)
    testData = final(0)[:-1].reshape(1, -1)

    pickle_model.decision_function(testData)
    output = pickle_model.pickle(testData)

    if(final[0][1] == 1.0 and output == [-1]):
        return render_template('index.html', prediction_text='This is fraudulent $ {}'.format(output))
    elif(final[0][1] == 0.0 and output == [1]):
        return render_template('index.html', prediction_text='This is valid $ {}'.format(output))
    else:
        return render_template('index.html', prediction_text='This is valid $ {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)
