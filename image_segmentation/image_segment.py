from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn
from PIL import Image
from image_segmentation.show_segment import show_segment
import matplotlib.pyplot as plt
import io

def image_segment(img):
    clothes_classification={
        'upper':0.00,
        'skirt':0.00,
        'pants':0.00,
        'dress':0.00,
    }
    #-- image segmentation --#
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
    # plt_img = show_segment(model,pred_seg)
    # plt.imshow(pred_seg)
    # plt.show()
    #-- plt to image --#
    # img_buf = io.BytesIO()
    # plt.savefig(img_buf, format='png')
    # plt_img = Image.open(img_buf)

    #-- image percentage --#
    upper_pixels = int((pred_seg==4).sum())
    skirt_pixels = int((pred_seg==5).sum())
    pants_pixels = int((pred_seg==6).sum())
    dress_pixels = int((pred_seg==7).sum())
    total_pixels = upper_pixels + skirt_pixels + pants_pixels + dress_pixels

    clothes_classification['upper']=round(upper_pixels/total_pixels*100,1)
    clothes_classification['skirt']=round(skirt_pixels/total_pixels*100,1)
    clothes_classification['pants']=round(pants_pixels/total_pixels*100,1)
    clothes_classification['dress']=round(dress_pixels/total_pixels*100,1)

    #-- image masking --#
    background = Image.new('RGB', img.size, color=(0, 0, 0))
    upper_masked = None
    skirt_masked = None
    pants_masked = None
    dress_masked = None

    if(clothes_classification['upper']>10.0):
        # tensor to numpy to w/b image for mask
        upper_masked = Image.composite(img, background, Image.fromarray((pred_seg==4).detach().numpy()))
    if(clothes_classification['skirt']>10.0):
        skirt_masked = Image.composite(img, background, Image.fromarray((pred_seg==5).detach().numpy()))
    if(clothes_classification['pants']>10.0):
        pants_masked = Image.composite(img, background, Image.fromarray((pred_seg==6).detach().numpy()))
    if(clothes_classification['dress']>10.0):
        dress_masked = Image.composite(img, background, Image.fromarray((pred_seg==7).detach().numpy()))

    return clothes_classification, upper_masked, skirt_masked, pants_masked, dress_masked