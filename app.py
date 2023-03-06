from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)#Initialising the flask App


@app.route("/")#Home page
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def result():

    item_weight= float(request.form['item_weight'])
    item_fat_content=float(request.form['item_fat_content'])
    item_visibility= float(request.form['item_visibility'])
    item_type= float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year= float(request.form['outlet_establishment_year'])
    outlet_size= float(request.form['outlet_size'])
    outlet_location_type= float(request.form['outlet_location_type'])
    outlet_type= float(request.form['outlet_type'])

    x= np.array([[ item_weight,item_fat_content,item_visibility,item_type,item_mrp,outlet_establishment_year,outlet_size,outlet_location_type,outlet_type ]])

    scaler_path= r"D:\End2EndProjects\BigMart Sales Data\Models\BigMart_sc.sav"

    sc=joblib.load(scaler_path)

    x_std= sc.transform(x)

    model_path=r"D:\End2EndProjects\BigMart Sales Data\Models\BigMart_rf.sav"

    model= joblib.load(model_path)

    y_pred=model.predict(x_std)

    return jsonify({'Prediction': float(y_pred)})

if __name__ == "__main__":
    #from waitress import serve
    app.run(debug=True, port=9457)