import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.detection_services.pedestrian_detector import PedestrianDetector

if __name__ == "__main__":  
    print("Testing PedestrianDetector initialization...")
    try:
        detector = PedestrianDetector()
        if detector.initialize():
            print("✅ PedestrianDetector initialized successfully")
        else:
            print("❌ PedestrianDetector initialization failed")
    except Exception as e:
        print(f"❌ PedestrianDetector initialization raised an exception: {e}")