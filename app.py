from flask import Flask
from demo import Demo
from flask import Flask,request
from PIL import Image
from gevent.pywsgi import WSGIServer
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
D=Demo("checkpoint\\FPAGANS\\2024-05-17_12-56-32\\saved_parameters\\gepoch_5_iter_6500.pth")

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/upload_file',methods=['POST'])
def upload_file():
    img = request.files['img']
    img.save("./test/tmp.jpg")
    img=Image.open("./test/tmp.jpg")
    D.demo(img)
    with open("./test/res.jpg", "rb") as f:
        base64_data = base64.b64encode(f.read())
    return base64_data

@app.route('/get_image/<string:age>',methods=['GET'])
def get_image(age):
    with open("tmp%s.jpg"%age, "rb") as f:
        base64_data = base64.b64encode(f.read())
    return base64_data

if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
