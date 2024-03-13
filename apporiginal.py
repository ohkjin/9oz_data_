import os, io
from flask import Flask, request, jsonify
from PIL import Image
import torch
import numpy as np
import cv2
import base64

# app = Flask(__name__)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# @app.route('/')
def hello():
    return 'Hello'

ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/photo_to_flask', methods=['POST'])
def getPhotoInput():
    # BE에서 json객체 전달받기
    request_obj = request.get_json()
    # print('request_obj',request_obj)
    response_obj = {
        # 'categories':'',
        # 'nukki_image':'',
        'similar':'',
        'style':0,
        'success':'',
        'message':'',
    }
    if request_obj['path']!='':
       
        # cv2
        # npimg = np.fromstring(file, np.uint8)
        # byte_img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        # print(byte_img)
        # img = Image.fromarray(img.astype("uint8"))
        # rawBytes = io.BytesIO()
        # img.save(rawBytes, "JPEG")
        # rawBytes.seek(0)
        # img_base64 = base64.b64encode(rawBytes.read())
        # return jsonify({'status':str(img_base64)})

        # 이미지 열기
        byte_file = request_obj['image'] ## byte file
        # base64
        base64_file = base64.b64decode(byte_file)
        img = Image.open(io.BytesIO(base64_file))
        imgRGB = Image.open(io.BytesIO(base64_file)).convert("RGB")
        img.show()
        
        print("size",img.size)
        print("mode",img.mode)
        print("modeRGB",imgRGB.mode)
    

        # path_img = Image.open(request.get(response_obj['path'],stream=True).raw).convert("RGB")
        # img = Image.open(request.get(response_obj['path'],stream=True).raw).convert("RGB")
        # print('img',img)
        # response_obj['nukki_image']=img
        response_obj['success']='1'
        response_obj['message']='success'
        # 이미지 Base64로 인코딩()
        # with open()
        print('response_obj',jsonify(response_obj))
        return jsonify(response_obj)
    
    # if 'file' not in request.files:
    #     return
    # file = request.files['file']
    # if file.filename == '':
    #     return
    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
    #     file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    #     return redirect(url_for('download_file',name=filename))

