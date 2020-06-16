import numpy as np
import json, requests
from io import BytesIO
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/image-quality')
def image_quality():
    data = {}

    if not request.files["image"]:
        return jsonify({"status": 400, "message": 'No image passed'})

    # Decoding and pre-processing base64 image
    img = image.img_to_array(image.load_img(BytesIO(request.files["image"].read()),
                                            target_size=(150, 150))) / 255.

    # this line is added because of a bug in tf_serving < 1.11
    img = img.astype('float16')

    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }

    # Making POST request
    r = requests.post('http://image-serving:8501/v1/models/qualitynet:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))

    pred = (np.array(pred['predictions'])[0] > 0.4).astype(np.int)
    if pred == 0:
        prediction = 'Bad'
    else:
        prediction = 'Good'

    data["prediction"] = prediction

    # Returning JSON response
    return jsonify(data)


# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000, debug=True)
