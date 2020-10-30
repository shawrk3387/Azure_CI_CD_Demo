import json

from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing import sequence

app = Flask(__name__)

MAX_LEN = 150

model = load_model("model/model_v1.h5")
with open("model/tokenizer.json") as f:
    data = json.load(f)
    tok = tokenizer_from_json(data)


@app.route("/infer", methods=["POST"])
def predict():
    msg = request.json["input"]
    if isinstance(msg, str):
        text_sequence = tok.texts_to_sequences([msg])
    elif isinstance(msg, list):
        text_sequence = tok.texts_to_sequences(msg)
    else:
        return "Unsupported data type in request payload. Supported types are string or list of strings."

    padded_sequence = sequence.pad_sequences(text_sequence, maxlen=MAX_LEN)
    predictions = model.predict(padded_sequence)
    predicted_classes = ["spam" if pred > 0.5 else "ham" for pred in predictions]
    return jsonify(predicted_classes)


if __name__ == "__main__":
    app.run(debug=False, threaded=False)
