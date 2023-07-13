# The project at university: Object detection

## Demo

![Demo](demo/demo.gif)

## Installation

### Requirements

- Windows 10 or higher
- Python 3.10

### Installation on Windows

Clone this repository:

```powershell
$ git clone https://github.com/vinhgiga/vnid-extractor.git
```

Change the working directory to `vnid-extractor`:

```powershell
$ cd vnid-extractor 
```

Download [VietOCR weight](https://vocr.vn/data/vietocr/vgg_transformer.pth) and put into `saved_models\vietocr` directory:

```powershell
$ c:\Windows\System32\curl.exe -o saved_models\vietocr\vgg_transformer.pth https://vocr.vn/data/vietocr/vgg_transformer.pth
```

(Optional) Create and activate a Python virtual environment:

```powershell
$ python -m venv .venv
$ .venv\Scripts\activate 
```

Install Python modules:

```powershell
$ pip install -r requirements.txt
```

Run app:

```powershell
$ python main.py
```

## Sources

1. [A technical view of FVI: End-to-end Vietnamese ID card OCR](https://fpt.ai/technical-view-fvi-end-end-vietnamese-id-card-ocr)
2. [Trích xuất thông tin từ chứng minh thư](https://viblo.asia/p/trich-xuat-thong-tin-tu-chung-minh-thu-bJzKmaRwK9N)
3. [TensorFlow 2 Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/)
4. [VietOCR repository](https://github.com/pbcquoc/vietocr)
