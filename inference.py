import torch
import pickle
import cv2
import albumentations as A
import albumentations.pytorch as A_torch
from config import PATH

net1 = torch.load(PATH + 'segm_butts.pth')
net2 = torch.load(PATH + 'segm_butts_dotted.pth')
net1.eval()
net2.eval()

port2camname = {
    8006: 'C05_03',
    8007: 'C05_04',
    8010: 'C05_02',
    8011: 'C05_01'
}

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    A_torch.ToTensorV2()                      
])

def undistort(img, cam):
    h,w = img.shape[:2]
    with open(PATH + cam + '_undistort_maps.pkl', 'rb') as f:
        map1, map2 = pickle.load(f)
    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return img


def preprocess(img, cam):
    img = undistort(img, cam)
    img = transform(image=img)['image']
    return img

def get_butts(img, port):
    '''
    img: RGB image in np.array
    '''
    cam = port2camname[port]
    img = preprocess(img, cam)
    with torch.no_grad():
        mask_butts = net1(img[None].cuda())['out'].cpu().numpy()[0,0]
    return mask_butts

def get_butts_dotted(img_butts):
    '''
    img_butts: custom size image with only butts
    '''
    img_butts = transform(image=img_butts)['image']
    with torch.no_grad():
        mask_butts_dotted = net2(img_butts[None].cuda())['out'].cpu().numpy()[0,0]
    return mask_butts_dotted