import io
import base64
import matplotlib.pyplot as plt


def encode_image(out_img_eccv16, out_img_siggraph17, img, img_bw):
    s1 = io.BytesIO()
    plt.imsave(s1, out_img_eccv16)
    s1 = base64.b64encode(s1.getvalue()).decode("utf-8").replace("\n", "")

    s2 = io.BytesIO()
    plt.imsave(s2, out_img_siggraph17)
    plt.savefig(s2, format='png', bbox_inches="tight", transparent = True)
    s2 = base64.b64encode(s2.getvalue()).decode("utf-8").replace("\n", "")

    s3 = io.BytesIO()
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(img_bw)
    plt.title('Input')
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(out_img_eccv16)
    plt.title('Output (ECCV 16)')
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(out_img_siggraph17)
    plt.title('Output (SIGGRAPH 17)')
    plt.axis('off')
    plt.show()
    plt.savefig(s3, format='png', bbox_inches="tight", transparent = True)
    s3 = base64.b64encode(s3.getvalue()).decode("utf-8").replace("\n", "")

    return s1, s2, s3

def save_image(out_img_eccv16, out_img_siggraph17, img, img_bw):
    s1 = io.BytesIO()
    plt.imsave(s1, out_img_eccv16)
    s1 = s1.getvalue()

    s2 = io.BytesIO()
    plt.imsave(s2, out_img_siggraph17)
    plt.savefig(s2, format='png', bbox_inches="tight", transparent = True)
    s2 = s2.getvalue()

    s3 = io.BytesIO()
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(img_bw)
    plt.title('Input')
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(out_img_eccv16)
    plt.title('Output (ECCV 16)')
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(out_img_siggraph17)
    plt.title('Output (SIGGRAPH 17)')
    plt.axis('off')
    plt.show()
    plt.savefig(s3, format='png', bbox_inches="tight", transparent = True)
    s3 = s3.getvalue()
    
    return s1, s2, s3
