from fastai.vision.all import *
from glob2 import glob
from tqdm import tqdm
from skimage import measure
import pydicom as dcm
from pydicom.pixel_data_handlers.util import apply_voi_lut
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='Make predictions and store them in con file for medis')
parser.add_argument('--id', help='meassurement id, e.g. A20', required=True)
parser.add_argument('--model', help='path to (exported) model file', required=True)
parser.add_argument('--state', help='model (saved) state to load (e.g. 90-frozen)', required=False)
parser.add_argument('--indir', help='input directory with dicom images (in subfolders)', required=True)
parser.add_argument('--con', help='path to input con file', required=True)
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

def contour_to_con(contour, s, f):
    con = "[XYCONTOUR]\n"
    con = con + f"{s} {f} 1  1.0\n"
    con = con + str(len(contour[0])) + "\n"
    for c in contour[0]:
        con = con + str(c[1]) + " " + str(c[0]) + "\n"
    con = con + "103\n0 0\n"
    
    con = con + "[XYCONTOUR]\n"
    con = con + f"{s} {f} 0  1.0\n"
    con = con + str(len(contour[1])) + "\n"
    for c in contour[1]:
        con = con + str(c[1]) + " " + str(c[0]) + "\n"
    con = con + "103\n0 0\n"
    
    return con

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
    pred = image2tensor(pred)
    return pred

def get_info(file):
    d = dcm.read_file(file)
    return file, d.SliceLocation, d.TriggerTime, d.SeriesNumber

def smooth(contours):
    smooth_contours = list()
    for contour in contours:
        cont = scipy.ndimage.zoom(np.concatenate((contour[::16,:],contour[-1:,:]),axis=0), (16,1), order=2)
        smooth_contours.append(cont)
    return smooth_contours

trainedModel = load_learner(args.model)
if args.state:
    trainedModel.load(args.state)

files = glob("{}/**/*.dcm".format(args.indir))
files_with_info = [get_info(file) for file in files]
files_by_slice = defaultdict(list)

con_file = args.con

for file in files_with_info:
    files_by_slice[float(file[1])].append(file)

all_contours = ""
all_smooth_contours = ""
for i,l in tqdm(enumerate(sorted(files_by_slice.keys())), total=len(files_by_slice.keys())):
    slice_files = sorted(files_by_slice[l],key=lambda x: x[2])
    for j,f in tqdm(enumerate(slice_files),leave=False,total=len(slice_files)):
        # print(i, j, f[0])
        img = get_img(f[0])
        pred = get_pred(img)
        contours = measure.find_contours(pred[0].numpy(), 1)
        smooth_contours = smooth(contours)
        if(len(contours) == 2):
            all_contours += contour_to_con(contours, i, j)
            all_smooth_contours += contour_to_con(smooth_contours, i, j)

os.makedirs(args.outdir, exist_ok=True)    

in_old_contour_part = False
with open(con_file, 'r') as infile:
    with open("{}/autopred_{}.con".format(args.outdir,args.id), "w") as outfile:
        with open("{}/smooth_autopred_{}.con".format(args.outdir,args.id), "w") as smoothfile:
            for line in infile.readlines():
                if("[XYCONTOUR]" in line):
                    in_old_contour_part = True
                if("[DISTANCE LABELS]" in line):
                    outfile.write(all_contours)
                    smoothfile.write(all_smooth_contours)
                    in_old_contour_part = False
                if not in_old_contour_part:
                    outfile.write(line)
                    smoothfile.write(line)
