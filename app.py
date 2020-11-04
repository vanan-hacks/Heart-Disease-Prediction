import flask as Flask
from flask import Flask,request, url_for, redirect, render_template

import pickle

import numpy as np



app = Flask(__name__, template_folder='template')



model=pickle.load(open('model.pkl','rb'))





@app.route('/')

def hello_world():

    return render_template("index-2.html")





@app.route('/predict',methods=['POST','GET'])

def predict():

    int_features=[int(x) for x in request.form.values()]

    final=[np.array(int_features)]

    print(int_features)

    print(final)

    prediction=model.predict_proba(final)

    output='{0:.{1}f}'.format(prediction[0][1], 2)



    if output>str(0.5):

        return render_template('form.html',pred='You are suffering from heart disease.\nProbability of suffering from heart disease is {}'.format(output))

    else:

        return render_template('form.html',pred='You are not suffering from any heart problem.\n Probability of not suffering from heart disease is {}'.format(output))





if __name__ == '__main__':

    app.run(debug=True)
