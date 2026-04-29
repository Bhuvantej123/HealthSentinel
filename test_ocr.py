import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageDraw, ImageFont
from modules.ocr_extract import extract_text

# Create a sample image
img = Image.new('RGB', (400, 300), color = (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((10,10), "Patient Report", fill=(0,0,0))
d.text((10,50), "Glucose: 115 mg/dL", fill=(0,0,0))
d.text((10,90), "BP: 120/80", fill=(0,0,0))
d.text((10,130), "Hemoglobin: 13.5", fill=(0,0,0))
img.save('test_report.png')

print("Image created. Running extraction...")
text = extract_text('test_report.png')
print("--- Extracted Text ---")
print(text)
print("----------------------")

from modules.report_parser import parse_health_parameters
print(parse_health_parameters(text, "North"))
