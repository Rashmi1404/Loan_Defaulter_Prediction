# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 19:55:41 2020

@author: pravi
"""

from flask import Flask, render_template, request, jsonify
from sklearn import preprocessing
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('randomforest.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1,9)
    
    feature_names = ['Zip','CCSC','ApprovalFY','Term','UrbanRural','LowDoc','DisbursementGross','GrAppv','SBA_Appv']
    df = pd.DataFrame(final_features, columns=feature_names)

    output = model.predict(df)

    if output == 1:
        res_val = "DEFAULTER"
    else:
        res_val = "NOT A DEFAULTER"
        
    return render_template('index.html', prediction_text='Person is  {}'.format(res_val))

if __name__=="__main__":
    app.run(debug=True)