import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from modules.report_parser import parse_health_parameters

# Test 1: Ideal text
text1 = """
Patient Report
Glucose: 120
Blood Pressure: 130/85
Hemoglobin: 14.2
Temperature: 98.6 F
Cholesterol: 190
Symptoms: fever, cough
"""
print("Test 1 (Ideal):")
print(parse_health_parameters(text1, "North"))

# Test 2: Messy OCR text
text2 = """
G1ucose 120 mg/dl
BP 130 / 85
Hb 14.2
Temp 98.6
chol 190
"""
print("\nTest 2 (Messy):")
print(parse_health_parameters(text2, "North"))
