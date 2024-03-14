import os, io
from flask import Flask, request, jsonify
from PIL import Image
import torch
import numpy as np
import cv2
import base64
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import requests
import matplotlib.pyplot as plt
import torch.nn as nn

app = Flask(__name__)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/')
def hello():
    return 'Hello'

ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/test', methods=['POST'])
def segment():
    request_obj = request.files['file']
    print('request_obj',request_obj)
    image_bytes = io.BytesIO(request_obj.read())
    img = Image.open(image_bytes)
    img_base64 = base64.b64encode(image_bytes.read())
    print(type(img))
    files = {'file': open(img, 'rb')}
    # response = requests.post('http://springboot_server:port/receive_image', files=files)
   

    
     # image segmentation
    # processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    # model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    
    # inputs = processor(images=img, return_tensors="pt")

    # outputs = model(**inputs)
    # logits = outputs.logits.cpu()

    # upsampled_logits = nn.functional.interpolate(
    #     logits,
    #     size=img.size[::-1],
    #     mode="bilinear",
    #     align_corners=False,
    # )

    # pred_seg = upsampled_logits.argmax(dim=1)[0]
    # plt.imshow(pred_seg)
    # print(upsampled_logits[0][4].detach().numpy())
    
    # return jsonify({'file':str(img_base64)})
    return send_file(image_file, mimetype='image/jpeg')
    # 이미지 열기

@app.route('/photo_to_flask', methods=['POST'])
def getPhotoInput():
    # BE에서 json객체 전달받기
    request_obj = request.get_json()
    # print('request_obj',request_obj)
    response_obj = {
        # 'categories':'',
        # 'nukki_image':'',
        # 'similar':'',
        # 'style':0,
        # 'season':'봄',
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

        # image segmentation
        processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        
        inputs = processor(images=img, return_tensors="pt")

        outputs = model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=img.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]
        plt.imshow(pred_seg)
        
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
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
