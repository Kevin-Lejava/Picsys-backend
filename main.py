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
    allow_origins=["http://localhost:3000"],
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
        _, b = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        return b

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
    def process(self, image, threshold: int = 100, **kwargs):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx**2 + sobely**2)
        edges = np.zeros_like(mag, dtype=np.uint8)
        edges[mag > threshold] = 255
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
}

@app.post("/process")
async def process_image(request: Request, upload: UploadFile = File(...)):
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

    output = img
    for method in methods:
        if method not in DIP_CLASSES:
            return {"error": f"Unknown method '{method}'"}
        cls = DIP_CLASSES[method]()
        kwargs = {}
        for suffix in ["gamma", "thresh", "threshold1", "threshold2", "k", "window_size", "kernel_size"]:
            field = f"{method}_{suffix}"
            if field in form:
                val = form.get(field)
                try:
                    if suffix in ['gamma', 'k']:
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

