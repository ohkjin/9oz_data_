import io
from flask import Flask, request, jsonify
from PIL import Image
import torch
import base64
from image_segmentation.image_segment import image_segment
from k_fashion.code.classify_by_style import classify_by_style

app = Flask(__name__)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/')
def hello():
    return 'Hello'

ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Test by POSTMAN by file upload
@app.route('/test', methods=['POST'])
def test():   
    #-- image receive --#
    request_obj = request.files['file']
    print('request_obj',request_obj)
    image_bytes = io.BytesIO(request_obj.read())
    img = Image.open(image_bytes)
    #-- data to send --#
    response_obj={
        'upper':{
            'percentage':0.0,
            'style':-1
        },
        'skirt':{
            'percentage':0.0,
            'style':-1
        },
        'pants':{
            'percentage':0.0,
            'style':-1
        },
        'dress':{
            'percentage':0.0,
            'style':-1
        }
    }

    #-- 이미지 전송 --#
    # img_base64 = base64.b64encode(image_bytes.read())
    # print(type(img))
    # files = {'file': open(img, 'rb')}
    # response = requests.post('http://springboot_server:port/receive_image', files=files)
    # return jsonify({'file':str(img_base64)})
    # return send_file(image_file, mimetype='image/jpeg')
    
    #-- image segmentation (dict)--#
    classification, upper_masked, skirt_masked, pants_masked, dress_masked = image_segment(img)
    response_obj['upper']['percentage'] = classification['upper']
    response_obj['skirt']['percentage'] = classification['skirt']
    response_obj['pants']['percentage'] = classification['pants']
    response_obj['dress']['percentage'] = classification['dress']
    dress_masked.show()

    #-- style classification (style dict)--#
    if(upper_masked!=None):
        response_obj['upper']['style'] = int(classify_by_style(upper_masked))
    if(skirt_masked!=None):
        response_obj['skirt']['style'] = int(classify_by_style(skirt_masked))
    if(pants_masked!=None):
        response_obj['pants']['style'] = int(classify_by_style(pants_masked))
    if(dress_masked!=None):
        response_obj['dress']['style'] = int(classify_by_style(dress_masked))

    return jsonify(response_obj)



@app.route('/photo_to_flask', methods=['POST'])
def getPhotoInput():
    # BE에서 json객체 전달받기
    request_obj = request.get_json()
    # print('request_obj',request_obj)
    # 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress"
    #-- data to send --#
    response_obj={
    'data':{
        'upper':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'sub_category':'',
            'similar':[],
        },
        'skirt':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'sub_category':'',
            'similar':[],
        },
        'pants':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'sub_category':'',
            'similar':[],
        },
        'dress':{
            'percentage':0.0,
            'style':-1,
            'season':'봄',
            'sub_category':'',
            'similar':[],
        }
    }
    }
    if request_obj['image']!='':
   
        # 이미지 받기
        byte_file = request_obj['image'] ## byte file
        base64_file = base64.b64decode(byte_file)
        img = Image.open(io.BytesIO(base64_file)).convert("RGB")
        # img.show()
        # print("size",img.size)
        # print("mode",img.mode)

        #-- image segmentation (dict)--#
        classification, upper_masked, skirt_masked, pants_masked, dress_masked = image_segment(img)
        response_obj['upper']['percentage'] = classification['upper']
        response_obj['skirt']['percentage'] = classification['skirt']
        response_obj['pants']['percentage'] = classification['pants']
        response_obj['dress']['percentage'] = classification['dress']
        # dress_masked.show()

        #-- style classification (style dict)--#
        if(upper_masked!=None):
            response_obj['upper']['style'] = int(classify_by_style(upper_masked))
        if(skirt_masked!=None):
            response_obj['skirt']['style'] = int(classify_by_style(skirt_masked))
        if(pants_masked!=None):
            response_obj['pants']['style'] = int(classify_by_style(pants_masked))
        if(dress_masked!=None):
            response_obj['dress']['style'] = int(classify_by_style(dress_masked))

        # print('response_obj',jsonify(response_obj))
        return jsonify(response_obj)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
