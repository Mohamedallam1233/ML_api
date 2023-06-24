import joblib
import warnings
from flask import Flask, request, jsonify
warnings.filterwarnings("ignore")
################################################################################################
def load_modelWithScaler(model_path,scaler_path ,data ,returnName=False,dictionary = None):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    dictionary = dictionary
    data = data
    prediction = model.predict(scaler.transform([data]))
    if (returnName == True):
        for x, c in dictionary.items():
            if c == prediction:
                return x
    else:
        return prediction
################################################################################################
mapping_dict = {"Gender": {"Male": 0, "Female": 1}, "Polyuria": {"No": 0, "Yes": 1}, "Polydipsia": {"No": 0, "Yes": 1}, "sudden weight loss": {"No": 0, "Yes": 1}, "weakness": {"Yes": 0, "No": 1}, "Polyphagia": {"No": 0, "Yes": 1}, "Genital thrush": {"No": 0, "Yes": 1}, "visual blurring": {"No": 0, "Yes": 1}, "Itching": {"No": 0, "Yes": 1}, "Irritability": {"No": 0, "Yes": 1}, "delayed healing": {"No": 0, "Yes": 1}, "partial paresis": {"No": 0, "Yes": 1}, "muscle stiffness": {"No": 0, "Yes": 1}, "Alopecia": {"No": 0, "Yes": 1}, "Obesity": {"No": 0, "Yes": 1}, "class": {"Positive": 0, "Negative": 1}}
cat_col_name = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']

################################################################################################
def ret_val_dict(col , val) : return (mapping_dict[col])[val]
################################################################################################
sym_dict = mapping_dict["class"]
################################################################################################
def predict_use_symptoms (sym_data):
    for i in range (1 , len(sym_data)):
        sym_data[i] = ret_val_dict(cat_col_name[i],sym_data[i])
    pred = load_modelWithScaler(model_path ="ml_models/symptoms_model.h5",scaler_path="ml_models/symptoms_scaler.h5" ,data=sym_data ,returnName=True,dictionary =sym_dict)
    return pred
################################################################################################
app = Flask(__name__)
@app.route('/symptoms_model', methods=['POST'])
def symptoms_model():
    if request.method == 'POST':
        sym_data = [request.form.get(i) for i in cat_col_name]
        pred = predict_use_symptoms(sym_data)
        data = {'message': pred}
        return jsonify(data)
    else:
        return jsonify({'message': 'This endpoint only accepts POST requests.'})

if __name__ == '__main__':
    app.run()
