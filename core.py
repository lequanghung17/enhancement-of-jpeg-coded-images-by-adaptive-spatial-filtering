# core.py — Core algorithms for 'Enhancement of JPEG Coded Images by Adaptive Spatial Filtering'
from __future__ import annotations
import os
from typing import Optional, Dict
import numpy as np
from PIL import Image
import cv2
from PIL import Image



try:
    from brisque import BRISQUE
    _HAS_BRISQUE = True
except Exception:
    _HAS_BRISQUE = False

try:
    from pypiqe import piqe as _piqe
    _HAS_PIQE = True
except Exception:
    _HAS_PIQE = False

def add_gaussian_noise(image: Image.Image, std: float = 25) -> Image.Image:
    arr = np.array(image.convert('L'), dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def add_salt_pepper_noise(image: Image.Image, prob: float = 0.05) -> Image.Image:
    arr = np.array(image.convert('L'))
    out = arr.copy()
    num_salt = int(np.ceil(prob * arr.size * 0.5))
    coords = [np.random.randint(0, i - 1, num_salt) for i in arr.shape]
    out[coords[0], coords[1]] = 255
    num_pepper = int(np.ceil(prob * arr.size * 0.5))
    coords = [np.random.randint(0, i - 1, num_pepper) for i in arr.shape]
    out[coords[0], coords[1]] = 0
    return Image.fromarray(out)

def apply_noise(image: Image.Image, noise_type: str, noise_level: float) -> Image.Image:
    if noise_type == 'Gaussian':
        return add_gaussian_noise(image, std=float(noise_level))
    elif noise_type == 'Salt & Pepper':
        return add_salt_pepper_noise(image, prob=float(noise_level)/1000.0)
    return image.convert('L')

def compress_image(input_path: str, output_path: str, quality: int) -> None:
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(input_path)
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    with open(output_path, 'wb') as f:
        f.write(buf)

def calculate_bitrate(image_path: str, compressed_image_path: str) -> float:
    with Image.open(image_path) as im:
        w, h = im.size
        total = w * h
    size = os.path.getsize(compressed_image_path)
    return float(size * 8.0 / max(1, total))

def padding_for_filter(image: np.ndarray, filter_size: int) -> np.ndarray:
    pad = filter_size // 2
    return np.pad(image, pad_width=pad, mode='edge')

class EdgeDetector:
    def __init__(self, window: np.ndarray, tau1: float=20.0, tau2: float=16.0, tau3: float=2.3):
        self.window = window.copy()
        self.tau1 = float(tau1); self.tau2 = float(tau2); self.tau3 = float(tau3)
    def step1(self) -> bool:
        w = self.window.flatten()
        r = float(w.max() - w.min())
        return r > self.tau1

    def step2(self) -> int:
        order = self.window.flatten()
        order.sort()
        S_h = order[21]                      # phần tử ~88% của 25 phần tử 
        Tavg = float(np.average(order))
        cnt = 0
        for i in range(5):
            for j in range(5):
                if (Tavg - float(self.window[i,j])) > S_h / self.tau2:
                    cnt += 1
        return cnt


    def step3(self, count: int) -> bool:
        # H: with 25 samples (5×5), E=12.5, sigma=2.5
        z = (count - 12.5) / 2.5
        return abs(z) <= self.tau3
    def classify(self) -> str:
        if self.step1():
            c = self.step2()
            if self.step3(c):
                return 'Dominant Edge'
        return 'Constant Region'

class TexelDetector:
    def __init__(self, window: np.ndarray, beta: float=12):
        self.window = window.copy().astype(np.float64)
        self.n, _ = self.window.shape
        self.eta = int((self.n*self.n) * 0.5)
        self.beta = float(beta)
    def subtract_mean(self):
        self.window -= float(np.mean(self.window))
    def operate_thresholding(self):
        self.window[np.abs(self.window) < self.beta] = 0.0
    def compute_rnzc(self) -> int:
        dirs = [(1,-1),(1,0),(1,1),(0,1)]
        rnzc = 0
        for x in range(self.n):
            for y in range(self.n):
                for dx,dy in dirs:
                    nx, ny = x+dx, y+dy
                    if 0<=nx<self.n and 0<=ny<self.n and self.window[nx,ny]*self.window[x,y] < 0:
                        rnzc += 1
        return int(rnzc)
    def classify(self) -> str:
        self.subtract_mean()
        self.operate_thresholding()
        return 'Texel' if self.compute_rnzc() > self.eta else 'QC Region'

class ThreeWayLabel:
    def __init__(self, window: np.ndarray, tau1: float, tau2: float, tau3: float, beta: float):
        self.window = window.copy()
        self.edge = EdgeDetector(self.window, tau1, tau2, tau3)
        self.beta = float(beta)
    def classify(self) -> str:
        if self.edge.classify() == 'Dominant Edge':
            return 'Dominant Edge'
        return TexelDetector(self.window, self.beta).classify()

def get_edges(image_array: np.ndarray, tau1: float, tau2: float, tau3: float, beta: float, window: int=5) -> np.ndarray:
    assert window % 2 == 1
    half = window // 2
    h, w = image_array.shape
    out = np.zeros_like(image_array, dtype=np.uint8)
    for x in range(half, h-half):
        for y in range(half, w-half):
            win = image_array[x-half:x+half+1, y-half:y+half+1].astype(np.float64)
            lab = ThreeWayLabel(win, tau1, tau2, tau3, beta).classify()
            out[x,y] = 128 if lab=='Texel' else (255 if lab=='QC Region' else 0)
    kernel = np.ones((1,3), dtype=np.uint8) # Tạo kernel giãn nở (dilation) 1×3 – ngang. Đây là mẹo “bảo vệ biên” trong bài báo (kéo dày biên theo hướng ngang, giảm nguy cơ lọc làm mờ biên).
    edge_mask = (out==0).astype(np.uint8)
    dilated = cv2.dilate(edge_mask, kernel, iterations=1)
    out[dilated==1] = 0
    return out

def merge_y_into_rgb(enhanced_y: Image.Image,
                     original_rgb: Image.Image,
                     smooth_chroma: bool = True) -> Image.Image:
    
    if original_rgb.mode != "RGB":
        original_rgb = original_rgb.convert("RGB")

    ycbcr = original_rgb.convert("YCbCr")
    Y0, Cb, Cr = ycbcr.split()

    if smooth_chroma:
        Cb = Image.fromarray(cv2.medianBlur(np.array(Cb), 3))
        Cr = Image.fromarray(cv2.medianBlur(np.array(Cr), 3))

    return Image.merge("YCbCr", (enhanced_y.convert("L"), Cb, Cr)).convert("RGB")

def Median_filter(image_array: np.ndarray, x: int, y: int, window: int) -> float:
    half = window // 2
    return float(np.median(image_array[x-half:x+half+1, y-half:y+half+1]))

def D_filter(image_array: np.ndarray, x: int, y: int, window: int) -> float:
    half = window // 2
    arr = image_array[x-half:x+half+1, y-half:y+half+1].flatten()
    arr.sort(); n = arr.shape[0]; m = (n+1)//2
    a = np.zeros(m, dtype=np.float64)
    for i in range(m):
        a[i] = (arr[i] + arr[n-i-1]) / 2.0
    return float(np.median(a))

def Multistage_median_filter(image_array: np.ndarray, x: int, y: int, window_size: int) -> float:
    half = window_size // 2
    H, W = image_array.shape
    sub = []
    for di,dj in [(0,1),(1,0),(1,1),(-1,1)]:
        vals = []
        for o in range(-half, half+1):
            xx, yy = x+di*o, y+dj*o
            if 0<=xx<H and 0<=yy<W:
                vals.append(image_array[xx,yy])
        sub.append(vals if vals else [image_array[x,y]])
    meds = [np.median(v) for v in sub]
    return float(np.median([max(meds), min(meds), float(image_array[x,y])]))

class AdaptiveFilterScheme1:
    def __init__(self, image: np.ndarray, edge_image: np.ndarray, d_fil_times: int=2, window: int=5):
        assert 1 <= d_fil_times <= 2 and window%2==1
        self.half = window//2
        self.H, self.W = image.shape
        self.img = padding_for_filter(image.copy(), window).astype(np.float64)
        self.lab = padding_for_filter(edge_image.copy(), window)
        self.times = int(d_fil_times)
    def run(self) -> np.ndarray:
        for x in range(self.half, self.H-self.half):
            for y in range(self.half, self.W-self.half):
                if self.lab[x,y] == 0:
                    self.img[x,y] = Median_filter(self.img, x, y, 5)
        for _ in range(self.times):
            for x in range(self.half, self.H-self.half):
                for y in range(self.half, self.W-self.half):
                    if self.lab[x,y] == 255:
                        self.img[x,y] = D_filter(self.img, x, y, 3)
        return self.img[self.half:self.half+self.H, self.half:self.half+self.W]

class AdaptiveFilterScheme2:
    def __init__(self, image: np.ndarray, edge_image: np.ndarray, window: int=5):
        assert window%2==1
        self.half = window//2
        self.H, self.W = image.shape
        self.img = padding_for_filter(image.copy(), window).astype(np.float64)
        self.lab = padding_for_filter(edge_image.copy(), window)
    def run(self) -> np.ndarray:
        for x in range(self.half, self.H-self.half):
            for y in range(self.half, self.W-self.half):
                if self.lab[x,y] == 255:
                    self.img[x,y] = D_filter(self.img, x, y, 3)
        for x in range(self.half, self.H-self.half):
            for y in range(self.half, self.W-self.half):
                if self.lab[x,y] == 0:
                    self.img[x,y] = Multistage_median_filter(self.img, x, y, 5)
        for x in range(self.half, self.H-self.half):
            for y in range(self.half, self.W-self.half):
                self.img[x,y] = D_filter(self.img, x, y, 3)
        return self.img[self.half:self.half+self.H, self.half:self.half+self.W]

def calculate_M1_qc(original: np.ndarray, processed: np.ndarray, qc_mask: np.ndarray) -> float:
    idx = np.where(qc_mask == 255)
    diff = original[idx] - processed[idx]
    if diff.size == 0: return 0.0
    mu = float(np.mean(diff))
    return float(np.sum(np.abs(mu - diff)) / len(diff))

def calculate_brisque(image: Image.Image) -> float:
    if not _HAS_BRISQUE: return float('nan')
    try:
        im = image.convert('RGB')
        br = BRISQUE(url=False)
        return float(br.score(im))
    except Exception:
        return float('nan')

def calculate_piqe(gray_image: Image.Image) -> float:
    if not _HAS_PIQE: return float('nan')
    try:
        arr = np.array(gray_image.convert('L'))
        score, _, _, _ = _piqe(arr)
        return float(score)
    except Exception:
        return float('nan')

class FullPipeline:
    def __init__(self, image_path: str, tau1: float, tau2: float, tau3: float, beta: float):
        self.image_path = image_path
        self.tau1=tau1; self.tau2=tau2; self.tau3=tau3; self.beta=beta
        self.original_image=None; self.original_array=None
        self.compressed_image=None; self.compressed_array=None
        self.scheme1_image=None; self.scheme1_array=None
        self.scheme2_image=None; self.scheme2_array=None
        self.three_way_image=None; self.three_way_array=None
    def encode_image(self, quality: int=10, noise_type: str='None', noise_level: float=25) -> float:
        self.original_image = Image.open(self.image_path)
        self.original_image = apply_noise(self.original_image, noise_type, noise_level)
        self.original_image.convert('L').save('example_gray.jpg')
        compress_image('example_gray.jpg', 'compressed_image.jpg', int(quality))
        self.compressed_image = Image.open('compressed_image.jpg').convert('L')
        bpp = calculate_bitrate('example_gray.jpg', 'compressed_image.jpg')
        self.original_array = np.array(self.original_image.convert('L'))
        self.compressed_array = np.array(self.compressed_image)
        return bpp
    def label_edges(self, window: int=5) -> None:
        img = np.array(self.compressed_image)
        H, W = img.shape
        half = window//2
        img_pad = padding_for_filter(img, window)
        lab = get_edges(img_pad, self.tau1, self.tau2, self.tau3, self.beta, window)
        self.three_way_image = Image.fromarray(lab[half:half+H, half:half+W])
        self.three_way_array = np.array(self.three_way_image)
    def apply_filters(self) -> None:
        arr = np.array(self.compressed_image).astype(np.float64)
        s1 = AdaptiveFilterScheme1(arr.copy(), self.three_way_array.copy()).run().astype(np.uint8)
        self.scheme1_image = Image.fromarray(s1); self.scheme1_array = s1
        s2 = AdaptiveFilterScheme2(arr.copy(), self.three_way_array.copy()).run().astype(np.uint8)
        self.scheme2_image = Image.fromarray(s2); self.scheme2_array = s2
    def assess_quality(self) -> Dict[str, float]:
        out = {k: float('nan') for k in [
            'M1_original_qc','M1_compressed_qc','M1_scheme1_qc','M1_scheme2_qc',
            'brisque_compressed','brisque_scheme1','brisque_scheme2',
            'piqe_compressed','piqe_scheme1','piqe_scheme2'
        ]}
        if self.original_array is None or self.compressed_array is None or self.three_way_array is None:
            return out
        out['M1_original_qc']  = calculate_M1_qc(self.original_array, self.original_array, self.three_way_array)
        out['M1_compressed_qc']= calculate_M1_qc(self.original_array, self.compressed_array, self.three_way_array)
        if self.scheme1_array is not None:
            out['M1_scheme1_qc'] = calculate_M1_qc(self.original_array, self.scheme1_array, self.three_way_array)
        if self.scheme2_array is not None:
            out['M1_scheme2_qc'] = calculate_M1_qc(self.original_array, self.scheme2_array, self.three_way_array)
        if self.compressed_image is not None:
            out['brisque_compressed'] = calculate_brisque(self.compressed_image)
            out['piqe_compressed'] = calculate_piqe(self.compressed_image)
        if self.scheme1_image is not None:
            out['brisque_scheme1'] = calculate_brisque(self.scheme1_image)
            out['piqe_scheme1'] = calculate_piqe(self.scheme1_image)
        if self.scheme2_image is not None:
            out['brisque_scheme2'] = calculate_brisque(self.scheme2_image)
            out['piqe_scheme2'] = calculate_piqe(self.scheme2_image)
        return out
    def run(self, quality: int=10, noise_type: str='None', noise_level: float=25):
        bpp = self.encode_image(quality, noise_type, noise_level)
        self.label_edges(); self.apply_filters()
        metrics = self.assess_quality()
        return (self.original_image, self.compressed_image, self.scheme1_image,
                self.scheme2_image, self.three_way_image, bpp, metrics)
    
    
