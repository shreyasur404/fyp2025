import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO

class DatasetCollector:
    """Helper class to collect and organize traffic images"""
    
    def __init__(self, save_dir: str = "traffic_dataset"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def download_image(self, url: str, category: str) -> bool:
        """Download and save image from URL"""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create category directory
                category_dir = self.save_dir / category
                category_dir.mkdir(exist_ok=True)
                
                # Save image with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"traffic_{category}_{timestamp}.jpg"
                save_path = category_dir / filename
                img.save(save_path)
                return True
            return False
        except Exception as e:
            print(f"Error downloading image: {e}")
            return False

class TrafficAnalyzer:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """Initialize traffic analyzer with YOLO model"""
        self.model = YOLO(model_path)
        self.vehicle_classes = {
            2: 'car', 3: 'motorcycle', 
            5: 'bus', 7: 'truck'
        }
        
    def analyze_image(self, image_path: str) -> Dict:
        """Analyze single traffic image and return vehicle counts"""
        try:
            # Read and process image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
                
            # Run detection
            results = self.model(img)[0]
            
            # Count vehicles by type
            vehicle_counts = {v_type: 0 for v_type in self.vehicle_classes.values()}
            
            # Calculate congestion score based on vehicle density
            total_vehicles = 0
            image_area = img.shape[0] * img.shape[1]
            
            for box in results.boxes:
                if int(box.cls) in self.vehicle_classes:
                    vehicle_type = self.vehicle_classes[int(box.cls)]
                    vehicle_counts[vehicle_type] += 1
                    total_vehicles += 1
            
            # Calculate basic congestion metrics
            congestion_score = min(1.0, total_vehicles / 20)  # Normalize to 0-1
            
            return {
                'filename': Path(image_path).name,
                'vehicle_counts': vehicle_counts,
                'total_vehicles': total_vehicles,
                'congestion_score': congestion_score,
                'suggested_green_time': self._calculate_green_time(congestion_score)
            }
            
        except Exception as e:
            return {
                'filename': Path(image_path).name,
                'error': str(e)
            }
    
    def _calculate_green_time(self, congestion_score: float) -> int:
        """Calculate suggested green light duration based on congestion"""
        min_time = 30  # Minimum green light duration
        max_time = 90  # Maximum green light duration
        return int(min_time + (max_time - min_time) * congestion_score)

class TrafficManagementSystem:
    def __init__(self, dataset_dir: str = "traffic_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.analyzer = TrafficAnalyzer()
        
    def process_dataset(self) -> Dict:
        """Process all images in dataset and generate report"""
        results = []
        
        # Process each category directory
        for category_dir in self.dataset_dir.iterdir():
            if category_dir.is_dir():
                category_results = []
                for img_path in category_dir.glob("*.jpg"):
                    analysis = self.analyzer.analyze_image(str(img_path))
                    category_results.append(analysis)
                
                results.append({
                    'category': category_dir.name,
                    'analyses': category_results
                })
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.dataset_dir / f"analysis_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        return results

def main():
    # Example URLs for different traffic conditions
    sample_images = {
        'heavy_traffic': [
            'URL1_for_heavy_traffic_image',
            'URL2_for_heavy_traffic_image'
        ],
        'normal_traffic': [
            'URL1_for_normal_traffic_image',
            'URL2_for_normal_traffic_image'
        ],
        'light_traffic': [
            'URL1_for_light_traffic_image',
            'URL2_for_light_traffic_image'
        ]
    }
    
    # Initialize system
    collector = DatasetCollector()
    
    # Download sample images (you'll need to replace with actual URLs)
    print("Collecting dataset...")
    for category, urls in sample_images.items():
        for url in urls:
            print(f"Downloading {category} image...")
            # Note: Replace placeholder URLs with actual ones
            # collector.download_image(url, category)
    
    # Process dataset
    system = TrafficManagementSystem()
    print("\nAnalyzing traffic images...")
    results = system.process_dataset()
    
    # Print summary
    print("\nTraffic Analysis Summary:")
    for category_result in results:
        print(f"\nCategory: {category_result['category']}")
        for analysis in category_result['analyses']:
            if 'error' not in analysis:
                print(f"\nImage: {analysis['filename']}")
                print(f"Total Vehicles: {analysis['total_vehicles']}")
                print(f"Congestion Score: {analysis['congestion_score']:.2f}")
                print(f"Suggested Green Time: {analysis['suggested_green_time']} seconds")
                print("Vehicle Counts:")
                for vehicle_type, count in analysis['vehicle_counts'].items():
                    print(f"  {vehicle_type}: {count}")

if __name__ == "__main__":
    main()