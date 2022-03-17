from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import model_files.model as ML_model
import os
app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def prediction():
    y = 0

    if request.method=='POST':
        file = request.files['file_upload']

        X = pd.read_csv(file)
        size = len(X)

        if size>5 :
            return render_template('error1.html')

        else:
            y = ML_model.predict(X)

            if(not y.any()):
                return render_template('error2.html')
            else:
                return render_template('index.html',
                                       test_time  = 'Testing Time prediction in seconds for selected {} cars: {}'.format(size, list(np.round(y,2))))

    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
