from colorization.colorizers import *
from utils import encode_image, save_image
import pandas as pd


def run_with_path(image_path:pd.DataFrame):
    use_gpu = False

    # load colorizers
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    if(use_gpu):
        colorizer_eccv16.cuda()
        colorizer_siggraph17.cuda()

    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img(image_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    if(use_gpu):
        tens_l_rs = tens_l_rs.cuda()

    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    eccv16_res, siggraph17_res, comparison = encode_image(out_img_eccv16, out_img_siggraph17, img, img_bw)
    
    return [
        {"type": "image", "label": "eccv16", "data":  {"alt": "eccv16 Image Colorization", "src": "data:image/png;base64, " + eccv16_res}},
        {"type": "image", "label": "siggraph17", "data":  {"alt": "siggraph17 Image Colorization", "src": "data:image/png;base64, " + siggraph17_res}},
        {"type": "image", "label": "comparison", "data":  {"alt": "Image Colorization Comparison", "src": "data:image/png;base64, " + comparison}},
    ]

def run(image):
    """
    image: object from PIL.image.open
    """
    use_gpu = False

    # load colorizers
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    if(use_gpu):
        colorizer_eccv16.cuda()
        colorizer_siggraph17.cuda()

    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img_from_object(image)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    if(use_gpu):
        tens_l_rs = tens_l_rs.cuda()

    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    # eccv16_image, siggraph17_image, comparison_image = save_image(out_img_eccv16, out_img_siggraph17, img, img_bw)
    
    return out_img_siggraph17


def run_return_json(image):
    """
    image: object from PIL.image.open
    """
    use_gpu = False

    # load colorizers
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    if(use_gpu):
        colorizer_eccv16.cuda()
        colorizer_siggraph17.cuda()

    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img_from_object(image)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    if(use_gpu):
        tens_l_rs = tens_l_rs.cuda()

    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    eccv16_res, siggraph17_res, comparison = encode_image(out_img_eccv16, out_img_siggraph17, img, img_bw)
    
    return [
        {"type": "image", "label": "eccv16", "data":  {"alt": "eccv16 Image Colorization", "src": "data:image/png;base64, " + eccv16_res}},
        {"type": "image", "label": "siggraph17", "data":  {"alt": "siggraph17 Image Colorization", "src": "data:image/png;base64, " + siggraph17_res}},
        {"type": "image", "label": "comparison", "data":  {"alt": "Image Colorization Comparison", "src": "data:image/png;base64, " + comparison}},
    ]
