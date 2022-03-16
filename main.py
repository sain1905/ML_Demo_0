from flask import Flask, render_template, request
import pandas as pd
import model_files.model as ML_model

app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def prediction():
    y = 0
    if request.method=='POST':

        file = request.files['file_upload']

        X = pd.read_csv(file)
        # print(X)
        # print(ML_model)
        y = ML_model.predict(X)
        # print(y)

    return render_template('index.html', test_time  = y)

if __name__ == '__main__':
    app.run(debug=True)