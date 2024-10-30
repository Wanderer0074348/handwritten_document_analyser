# import cv2
# import numpy as np
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
# from PIL import Image
# import easyocr
# from pix2tex.cli import LatexOCR
# import os
from src.PaperProcessor import MathPaperProcessor

processor = MathPaperProcessor()

if __name__ == '__main__':
    try:
        results = processor.process_paper('data/first.jpg')
        
        print("\nProcessed Results:")
        print("\nEquations found:")
        for eq in results['equations']:
            print(f"Original text: {eq['text']}")
            print(f"LaTeX: {eq['latex']}\n")
    except Exception as e:
        print(f"Error occurred: {str(e)}")