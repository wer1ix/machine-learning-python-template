from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("california_price_model.pkl", "rb"))

# Features esperadas: AveRooms, AveOccup, MedInc
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    ave_rooms = float(data["AveRooms"])
    ave_occup = float(data["AveOccup"])
    med_inc   = float(data["MedInc"])
    X = np.array([[ave_rooms, ave_occup, med_inc]], dtype=float)
    pred = float(model.predict(X)[0])
    return jsonify({"predicted_med_house_val": round(pred, 4)})

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "msg": "Usa POST /predict con JSON {'AveRooms','AveOccup','MedInc'} para predecir MedHouseVal"
    })

if __name__ == "__main__":
    app.run(debug=True)
