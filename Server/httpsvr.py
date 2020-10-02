from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
import numpy as np
import cv2
import base64
from PIL import Image
import ImageProcessing

my_port = '8000'
scale = 0.00392
conf_threshold = 0.5
nms_threshold = 0.4

app = Flask(__name__)
CORS(app)

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def build_return(class_id, x, y, x_plus_w, y_plus_h):
    return str(class_id) + "," + str(x) + "," + str(y) + "," + str(x_plus_w) + "," + str(y_plus_h)

net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

@app.route('/')
@cross_origin()
def index():
    return "Welcome to flask API!"

@app.route('/detect', methods=['POST'])
@cross_origin()
def detect_form():

    image_b64 = request.form.get('image')
    img = np.fromstring(base64.b64decode(image_b64), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)
    returnString = ImageProcessing.process(img,scale,conf_threshold,nms_threshold,net)
    return returnString

@app.route('/upload', methods=['POST'])
def detect_web():
    img = Image.open(request.files['image'])
    img = np.array(img)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    returnString = ImageProcessing.process(img, scale, conf_threshold, nms_threshold, net)
    print(returnString)
    return returnString

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=my_port)