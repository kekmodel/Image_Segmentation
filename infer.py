import urllib
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from IPython.display import clear_output, display
import segmentation_models_pytorch as smp

device = torch.device('cuda')
model = smp.Unet(encoder_name="timm-efficientnet-b8",
                 encoder_weights="advprop",
                 decoder_use_batchnorm=True,  
                 in_channels=3,                  
                 classes=1)
model.to(device)
model.load_state_dict(torch.load('./tb8_ap.pkl'))
trsm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_concat(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def infer(inputs):
    path, _ = urllib.request.urlretrieve(inputs)
    image = TF.center_crop(TF.resize(Image.open(path).convert("RGB"), 224), 224)
    img = trsm(image)
    model.eval()
    with torch.no_grad():
        img = img.unsqueeze(0).to(device)
        output = model(img)
        mask = output.ge(0.5).long().reshape(224, 224, 1).cpu().numpy()
        masked_image = Image.fromarray((mask * np.array(image)).astype("uint8"))
        image_cat = get_concat(image, masked_image)
        return image_cat


def main():
    while True:
        url = input('URL:')
        if url is "":
            print("closed")
            return
        try:
            clear_output(True)
            image_cat = infer(url)
            display(TF.resize(image_cat, 360))
        except:
            continue
        

if __name__ == '__main__':
    infer('https://lh3.googleusercontent.com/proxy/8JRSkiQeJ6yGZ9guZO_3tzEXfHrhVw3-hLfTNfadgloVVb7mGIASXrpiqIoaqlx7IeL_IsK7BtI__XAfyNsh-guHcu6C_DVkzC-B4cQjm4KVXRkPcH-ZN15lDY-9wxKBcGSZoiiBf49bXJfMI0mEHWQvvoPA6Q')