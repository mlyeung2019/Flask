from flask import Flask, render_template, request
import pickle
import numpy as np

app=Flask(__name__)

loaded_model = pickle.load(open("bostonregressionmodel.pkl","rb"))
loaded_scaler = pickle.load(open("stdscaler.pkl","rb"))

@app.route('/')
def home():
    return render_template("home.html")

def ValuePredictor(to_predict):
    #to_predict = np.array(to_predict_list).reshape(1,13)
    result = loaded_model.predict(to_predict)
    return result[0][0]

@app.route('/result',methods = ['POST'])
def result():
    prediction=''
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        print(to_predict_list.values())
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        print("Before sending to scaler", to_predict_list)
        to_predict = loaded_scaler.transform(np.array(to_predict_list).reshape(1,13))
        print("Before sending to model", to_predict)
        result = ValuePredictor(to_predict)
        print("result from model", result)
        prediction = str(result)
        print(prediction)
        return render_template("result.html",prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
