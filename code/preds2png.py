from fastai.vision.all import *
from glob2 import glob
from tqdm import tqdm
from skimage import measure
import pydicom as dcm
from pydicom.pixel_data_handlers.util import apply_voi_lut
from collections import defaultdict
import argparse
import PIL
import os

parser = argparse.ArgumentParser(description='Make predictions and store them in folder as png')
parser.add_argument('--id', help='meassurement id, e.g. A20', required=True)
parser.add_argument('--model', help='path to (exported) model file', required=True)
parser.add_argument('--state', help='model (saved) state to load (e.g. 90-frozen)', required=False)
parser.add_argument('--indir', help='input directory with dicom images (in subfolders)', required=True)
parser.add_argument('--outdir', help='output directory', required=True)
args = parser.parse_args()
print(args.id)
measurement_id = args.id

defaults.device = torch.device('cpu')

def label_func(x): pass
def acc_seg(x,y): pass
def diceComb(x,y): pass
def diceLV(x,y): pass
def diceMY(x,y): pass

def get_img(file):
    d = dcm.read_file(file)
    img = d.pixel_array
    img_data = torch.Tensor(apply_voi_lut(img, d))
    img_data = img_data - img_data.min()
    img_data = img_data / img_data.max()
    img_data = torch.stack((img_data,img_data,img_data))
    return PILImage(to_image(img_data))

def get_pred(img):
    with trainedModel.no_bar():
        pred = trainedModel.predict(img)[0]
    pred = PILMask.create(pred)
    mindim = np.min(img.shape)
    pred = pred.resize((mindim,mindim), Image.NEAREST)
    pred = pred.crop_pad(img.shape[::-1])
    #pred = image2tensor(pred)
    return pred

def get_info(file):
    d = dcm.read_file(file)
    return file, d.SliceLocation, d.TriggerTime, d.SeriesNumber

trainedModel = load_learner(args.model)
if args.state:
    # the path from training is "../model" but when called from the repo root it should be just "model"
    trainedModel.path = Path("model")
    trainedModel.load(args.state)

files = glob("{}/**/*.dcm".format(args.indir))
files_with_info = [get_info(file) for file in files]
files_by_slice = defaultdict(list)

for file in files_with_info:
    files_by_slice[float(file[1])].append(file)

os.makedirs(args.outdir, exist_ok=True)    

for i,l in tqdm(enumerate(sorted(files_by_slice.keys())), total=len(files_by_slice.keys())):
    slice_files = sorted(files_by_slice[l],key=lambda x: x[2])
    for j,f in tqdm(enumerate(slice_files),leave=False,total=len(slice_files)):
        # print(i, j, f[0])
        img = get_img(f[0])
        pred = get_pred(img)
        #print(pred.shape)
        PIL.Image.fromarray(np.array(pred, dtype=np.int32),'I').save("{}/{}_slice{:03d}_frame{:03d}-mask.png".format(args.outdir,measurement_id,i,j))
