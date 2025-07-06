import io
import json
from fastapi import FastAPI, UploadFile, Request, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from skimage.feature import hog
from skimage import filters
from scipy.signal import find_peaks

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://kevin-lejava.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DIPMethod:
    name: str

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

class Greyscale(DIPMethod):
    name = "greyscale"
    def process(self, image, **kwargs):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
class HSV(DIPMethod):
    name = "HSV"
    def process(self, image, **kwargs):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

class HLS(DIPMethod):
    name = "HLS"
    def process(self, image, **kwargs):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
class Log(DIPMethod):
    name = "logarithmic"
    def process(self, image, **kwargs):
        img_f = image.astype(np.float32)
        loged = cv2.log(img_f + 1)
        scaled = np.uint8(255 * (loged / np.max(loged)))
        return scaled

class Exp(DIPMethod):
    name = "exponential"
    def process(self, image, **kwargs):
        img_f = image.astype(np.float32) / 255.0
        exped = np.exp(img_f) - 1
        scaled = np.uint8(255 * (exped / np.max(exped)))
        return scaled

class Sqrt(DIPMethod):
    name = "sqrt"
    def process(self, image, **kwargs):
        img_f = image.astype(np.float32)
        sq = cv2.sqrt(img_f)
        scaled = np.uint8(255 * (sq / np.max(sq)))
        return scaled

class Sin(DIPMethod):
    name = "sine"
    def process(self, image, **kwargs):
        radians = np.radians(image.astype(np.float32))
        s = np.sin(radians)
        scaled = np.uint8(255 * ((s + 1) / 2))
        return scaled

class Cos(DIPMethod):
    name = "cosine"
    def process(self, image, **kwargs):
        radians = np.radians(image.astype(np.float32))
        c = np.cos(radians)
        scaled = np.uint8(255 * ((c + 1) / 2))
        return scaled

class Tan(DIPMethod):
    name = "tangent"
    def process(self, image, **kwargs):
        radians = np.radians(image.astype(np.float32))
        t = np.tan(radians)
        t = np.clip(t, -10, 10)
        scaled = np.uint8(255 * ((t + 10) / 20))
        return scaled

class ContrastStretched(DIPMethod):
    name = "contrast_stretch"
    def process(self, image, **kwargs):
        r_min, r_max = image.min(), image.max()
        stretched = ((image.astype(np.float32) - r_min) / (r_max - r_min)) * 255
        return np.uint8(np.clip(stretched, 0, 255))

class HistEq(DIPMethod):
    name = "hist_eq"
    def process(self, image, **kwargs):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.equalizeHist(gray)

class Gamma(DIPMethod):
    name = "gamma"
    def process(self, image, gamma: float = 1.0, **kwargs):
        img_norm = image.astype(np.float32) / 255.0
        corrected = np.power(img_norm, gamma) * 255
        return np.uint8(np.clip(corrected, 0, 255))

class GaussianBlur(DIPMethod):
    name = "gaussian_blur"
    def process(self, image, kernel_size: int = 5, **kwargs):
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.GaussianBlur(image, (k, k), 0)

class CLAHE(DIPMethod):
    name = "clahe"
    def process(self, image, **kwargs):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

class Sharpen(DIPMethod):
    name = "sharpen"
    def process(self, image, **kwargs):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        sharp = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        return np.uint8(np.clip(sharp, 0, 255))

class Segment(DIPMethod):
    name = "segment"
    def process(self, image, **kwargs):
        if image.ndim != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        peaks, _ = find_peaks(hist, height=100)
        if len(peaks) >= 2:
            valley = int((peaks[0] + peaks[1]) / 2)
            _, image = cv2.threshold(image, valley, 255, cv2.THRESH_BINARY)
        return image
    
class MultiSegment(DIPMethod):
    name ="multisegment"
    def process(self, image):
        if image.ndim != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pixels = image.reshape((-1, 1)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 3
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        segmented = centers[labels.flatten()].reshape(image.shape)
        return np.uint8(segmented)

class AdaptiveThreshold(DIPMethod):
    name = "adaptive_thresh"
    def process(self, image, **kwargs):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

class BinaryThreshold(DIPMethod):
    name = "binary"
    def process(self, image, thresh: int = 127, **kwargs):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, img2 = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        return img2

class Open(DIPMethod):
    name = "open"
    def process(self, image, **kwargs):
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return opened
    
class Dilate(DIPMethod):
    name = "dilate"
    def process(self, image, **kwargs):
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(image, kernel, iterations=1)
        return dilated
    
class Erode(DIPMethod):
    name = "erode"
    def process(self, image, **kwargs):
        kernel = np.ones((3, 3), np.uint8)
        erroded = cv2.erode(image, kernel, iterations=1)
        return erroded
    
class MorphGradient(DIPMethod):
    name = "morphgradient"
    def process(self, image, **kwargs):
        kernel = np.ones((3, 3), np.uint8)
        morph_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        return morph_gradient

class CannyEdge(DIPMethod):
    name = "canny"
    def process(self, image, threshold1: int = 100, threshold2: int = 200, **kwargs):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return cv2.Canny(gray, threshold1, threshold2)

class SobelEdge(DIPMethod):
    name = "sobel"
    def process(self, image, thresh: int = 100, **kwargs):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        edges = np.zeros_like(mag, dtype=np.uint8)
        edges[mag > thresh] = 255
        return edges

class Niblack(DIPMethod):
    name = "niblack"
    def process(self, image, k: float = 0.2, window_size: int = 15, **kwargs):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray_f = gray.astype(np.float32)
        thresh = filters.threshold_niblack(gray_f, window_size=window_size, k=k)
        return np.uint8((gray_f > thresh) * 255)

class Sauvola(DIPMethod):
    name = "sauvola"
    def process(self, image, k: float = 0.2, window_size: int = 15, **kwargs):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray_f = gray.astype(np.float32)
        thresh = filters.threshold_sauvola(gray_f, window_size=window_size, k=k)
        return np.uint8((gray_f > thresh) * 255)
    
class HOG(DIPMethod):
    name = "HOG"
    def process(self, image, **kwargs):
        if image.ndim != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image_vis = hog(
            image, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys'
        )
        return image_vis
class LIPmultiply(DIPMethod):
    name = "LIPmultiply"
    def process(self, image, c: float = 1.2, **kwargs):
        return 255 * (1 - np.power(1 - image / 255, c))

DIP_CLASSES = {
    Greyscale.name: Greyscale,
    HSV.name: HSV,
    HLS.name: HLS,
    Log.name: Log,
    Exp.name: Exp,
    Sqrt.name: Sqrt,
    Sin.name: Sin,
    Cos.name: Cos,
    Tan.name: Tan,
    ContrastStretched.name: ContrastStretched,
    HistEq.name: HistEq,
    Gamma.name: Gamma,
    GaussianBlur.name: GaussianBlur,
    CLAHE.name: CLAHE,
    Sharpen.name: Sharpen,
    Segment.name: Segment,
    MultiSegment.name: MultiSegment,
    AdaptiveThreshold.name: AdaptiveThreshold,
    BinaryThreshold.name: BinaryThreshold,
    Open.name: Open,
    Dilate.name: Dilate,
    Erode.name: Erode,
    MorphGradient.name: MorphGradient,
    CannyEdge.name: CannyEdge,
    SobelEdge.name: SobelEdge,
    Niblack.name: Niblack,
    Sauvola.name: Sauvola,
    HOG.name: HOG,
    LIPmultiply.name: LIPmultiply
}

@app.post("/process")
async def process_image(request: Request, upload: UploadFile = File(...), upload2: UploadFile = File(None)):
    form = await request.form()
    methods_field = form.get('methods')
    try:
        methods = json.loads(methods_field) if methods_field else []
    except json.JSONDecodeError:
        return {"error": "Invalid methods JSON"}

    contents = await upload.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"error": "cannot decode image"}
    
    img2 = None
    
    if ('add' in methods or
        'subtract' in methods or
        'divide' in methods or
        'multiply' in methods or
        'LIPadd' in methods or
        'LIPsubtract' in methods or
        'LIPdivide' in methods) and upload2 is not None:
        contents2 = await upload2.read()
        np_arr2 = np.frombuffer(contents2, np.uint8)
        img2 = cv2.imdecode(np_arr2, cv2.IMREAD_UNCHANGED)
        if img2 is None:
            return {"error": "cannot decode second image"}
        if img2.shape != img.shape:
            img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))

    output = img
    for method in methods:
        if method in ('add', 'subtract', 'multiply', 'divide', 'LIPadd', 'LIPsubtract', 'LIPdivide'):
            if img2 is None:
                return {"error": f"Second image required for {method}"}
            if output.ndim == 2 and img2.ndim == 3:
                img2_proc = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            elif output.ndim == 3 and img2.ndim == 2:
                img2_proc = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            else:
                img2_proc = img2

            if method == 'add':
                output = cv2.add(output, img2_proc)
            elif method == 'subtract':
                output = cv2.subtract(output, img2_proc)
            elif method == 'multiply':    
                output = cv2.multiply(output, img2_proc)
            elif method == 'divide':
                output = cv2.divide(output, img2_proc, scale=16.0)
                output = np.clip(output, 0, 255)
                output = np.uint8(output)
            elif method == 'LIPadd':
                output = img + img2 - ((img * img2) / 255)
            elif method == 'LIPsubtract':
                output = 255 * ((img.astype(np.float32) - img2_proc.astype(np.float32)) / (255 - img2_proc.astype(np.float32)))
                output = np.nan_to_num(output, nan=0.0, posinf=255, neginf=0)
                output = np.clip(output, 0, 255).astype(np.uint8)
            elif method == 'LIPdivide':
                img_f = img.astype(np.float32) / 255
                img2_f = img2_proc.astype(np.float32) / 255
                la = np.log1p(-img_f)
                lb = np.log1p(-img2_f)
                denom = la + lb
                j = np.where(denom != 0, la / denom, 0.5)
                result = 255 * (1 - np.power(1 - img_f, j) * np.power(1 - img2_f, 1 - j))
                output = np.nan_to_num(result, nan=0.0, posinf=255, neginf=0)
                output = np.clip(output, 0, 255).astype(np.uint8)
            continue

        if method not in DIP_CLASSES:
            return {"error": f"Unknown method '{method}'"}
        cls = DIP_CLASSES[method]()
        kwargs = {}
        for suffix in ["gamma", "thresh", "threshold1", "threshold2", "k", "window_size", "kernel_size", "c"]:
            field = f"{method}_{suffix}"
            if field in form:
                val = form.get(field)
                try:
                    if suffix in ['gamma', 'k', 'c']:
                        kwargs[suffix] = float(val)
                    else:
                        kwargs[suffix] = int(val)
                except:
                    pass
        try:
            output = cls.process(output, **kwargs)
        except Exception as e:
            return {"error": f"processing '{method}' failed: {e}"}

    if output.ndim == 2:
        success, buf = cv2.imencode('.png', output)
    else:
        success, buf = cv2.imencode('.png', output)
    if not success:
        return {"error": "could not encode result"}

    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")

