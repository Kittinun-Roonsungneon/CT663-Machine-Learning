from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
port = 9000

sc = StandardScaler()
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# ปรับค่า StandardScaler และแปลงข้อมูลฝึกสอนก่อนโหลดโมเดล
X_train = sc.fit_transform(X_train)

modelde = pickle.load(open("decision.pkl", "rb"))
modelrf = pickle.load(open("randonforest.pkl", "rb"))
modelknn = pickle.load(open("knn.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/', methods=['GET'])
def index():
    return jsonify({"Serveer flask is running on Port" : str(port)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = {
            "age": int(data["age"]),
            "salary": int(data["salary"]),
        }

        # Preprocess the data
        modeldata = sc.transform([[features["age"], features["salary"]]])

        try:
            # Predict with the models
            DE = modelde.predict(modeldata)
            RF = modelrf.predict(modeldata)
            KNN = modelknn.predict(modeldata)
        except Exception as e:
            app.logger.error(f"Error during model prediction: {str(e)}")
            return jsonify({"error": str(e)}), 500

        # Return the result as JSON
        return jsonify({"DE": int(DE[0]), "RF": int(RF[0]), "KNN": int(KNN[0])})
    except ValueError:
        return jsonify({"error": "invalid input"}), 400
    except Exception as e:
        app.logger.error(f"Error in predict function: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/test')
def test():
    # Get the data from the request
    data = request.json
    return jsonify("data")
       
if __name__=='__main__':
    app.run(debug=True,port=port)