import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict


class TrafficVisualizer:
    """Class for creating traffic analysis visualizations"""

    def create_heatmap(self, analyses: list) -> np.ndarray:
        """Create heatmap visualization of traffic patterns."""
        data = []

        for analysis in analyses:
            if 'error' not in analysis and 'hour' in analysis and 'congestion_score' in analysis:
                data.append({'hour': analysis['hour'], 'congestion': analysis['congestion_score']})

        if not data:
            print("No valid data available for heatmap.")
            plt.figure(figsize=(12, 6))
            plt.text(
                0.5, 0.5, 'No valid data available', 
                horizontalalignment='center', verticalalignment='center'
            )
            plt.title('Traffic Congestion Heatmap')
            plt.tight_layout()
            fig = plt.gcf()
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close()
            return img_array

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Group by hour and compute average congestion
        pivot_data = df.groupby('hour')['congestion'].mean().reset_index()

        # Pivot data to structure it for a heatmap (hour vs. congestion score)
        pivot_table = pd.pivot_table(
            pivot_data, 
            values='congestion', 
            index='hour', 
            aggfunc='mean'
        ).fillna(0)

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            pivot_table, 
            annot=True, 
            cmap='YlOrRd', 
            cbar_kws={'label': 'Average Congestion Score'},
            fmt=".1f"
        )
        plt.xlabel('Hour of Day')
        plt.ylabel('Congestion')
        plt.title('Traffic Congestion Heatmap')
        plt.tight_layout()

        fig = plt.gcf()
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close()
        return img_array




    def create_vehicle_distribution_plot(self, analyses: list) -> np.ndarray:
        """Create pie chart of vehicle distribution"""
        vehicle_counts = defaultdict(int)
        valid_analyses = False

        for analysis in analyses:
            if 'error' not in analysis and 'vehicle_counts' in analysis:
                valid_analyses = True
                for v_type, count in analysis['vehicle_counts'].items():
                    vehicle_counts[v_type] += count

        if not valid_analyses:
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5, 'No valid vehicle data available', horizontalalignment='center', verticalalignment='center')
            plt.title('Vehicle Type Distribution')
        else:
            plt.figure(figsize=(10, 8))
            plt.pie(
                vehicle_counts.values(),
                labels=vehicle_counts.keys(),
                autopct='%1.1f%%',
                startangle=90
            )
            plt.title('Vehicle Type Distribution')

        plt.tight_layout()
        fig = plt.gcf()
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close()
        return img_array


class TrafficAnalyzer:
    """Improved TrafficAnalyzer."""
    def analyze_image(self, img_path: str) -> dict:
        # Dummy analysis simulating congestion score based on filename or random factors
        congestion_score = np.random.randint(10, 100)  # Randomized for now
        vehicle_counts = {
            "car": np.random.randint(20, 100),
            "truck": np.random.randint(5, 30),
            "motorcycle": np.random.randint(10, 50),
            "bus": np.random.randint(2, 15)
        }

        try:
            # Extract hour from filename if possible (e.g., 'traffic_20240416_0900.jpg')
            filename = Path(img_path).stem
            hour = int(filename.split('_')[-1][:2])  # Extract hour assuming filename format
        except Exception:
            hour = np.random.randint(0, 24)  # Fallback to random hour

        return {
            "filename": img_path,
            "hour": hour,
            "congestion_score": congestion_score,
            "vehicle_counts": vehicle_counts
        }



class TrafficPatternAnalyzer:
    """Improved TrafficPatternAnalyzer."""
    def analyze_patterns(self, analyses: list) -> dict:
        if not analyses:
            return {"error": "No valid analyses to analyze patterns."}

        congestion_by_hour = defaultdict(list)
        vehicle_type_totals = defaultdict(int)

        for analysis in analyses:
            congestion_by_hour[analysis["hour"]].append(analysis["congestion_score"])
            for v_type, count in analysis["vehicle_counts"].items():
                vehicle_type_totals[v_type] += count

        avg_congestion_by_hour = {
            hour: np.mean(scores) for hour, scores in congestion_by_hour.items()
        }
        peak_hour = max(avg_congestion_by_hour, key=avg_congestion_by_hour.get)
        busiest_vehicle_type = max(vehicle_type_totals, key=vehicle_type_totals.get)

        return {
            "average_congestion_by_hour": avg_congestion_by_hour,
            "peak_hour": peak_hour,
            "busiest_vehicle_type": busiest_vehicle_type,
        }


class TrafficReportGenerator:
    """Enhanced TrafficReportGenerator."""
    def generate_report(self, analyses: list, patterns: dict) -> dict:
        if not analyses:
            return {"error": "No valid analyses available for generating a report."}

        total_congestion = sum(a["congestion_score"] for a in analyses)
        avg_congestion = total_congestion / len(analyses)

        recommendations = []
        if patterns.get("peak_hour") in [7, 8, 9, 17, 18]:
            recommendations.append("Increase traffic police presence during peak hours.")
        if patterns.get("busiest_vehicle_type") == "car":
            recommendations.append("Introduce carpooling incentives.")

        return {
            "summary_stats": {
                "average_congestion": avg_congestion,
                "total_vehicles_analyzed": sum(
                    sum(a["vehicle_counts"].values()) for a in analyses
                ),
            },
            "peak_hour": patterns.get("peak_hour"),
            "busiest_vehicle_type": patterns.get("busiest_vehicle_type"),
            "recommendations": recommendations,
        }


class EnhancedTrafficManagementSystem:
    def __init__(self, dataset_dir: str = "traffic_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.analyzer = TrafficAnalyzer()
        self.visualizer = TrafficVisualizer()
        self.pattern_analyzer = TrafficPatternAnalyzer()
        self.report_generator = TrafficReportGenerator()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def run_analysis(self) -> dict:
        image_files = (
            list(self.dataset_dir.rglob("*.jpg")) +
            list(self.dataset_dir.rglob("*.jpeg")) +
            list(self.dataset_dir.rglob("*.png"))
        )

        if not image_files:
            print(f"Error: No image files (JPG, JPEG, PNG) found in '{self.dataset_dir}'")
            return {}

        analyses = []
        for img_path in image_files:
            try:
                analysis = self.analyzer.analyze_image(str(img_path))
                analyses.append(analysis)
            except Exception as e:
                print(f"Warning: Failed to process {img_path.name}: {str(e)}")
                continue

        valid_analyses = [a for a in analyses if 'error' not in a]
        patterns = self.pattern_analyzer.analyze_patterns(valid_analyses)
        report = self.report_generator.generate_report(valid_analyses, patterns)

        visualizations = {
            'heatmap': self.visualizer.create_heatmap(valid_analyses),
            'vehicle_distribution': self.visualizer.create_vehicle_distribution_plot(valid_analyses),
        }

        # Save visualizations and report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.dataset_dir / 'reports'
        report_dir.mkdir(exist_ok=True)

        with open(report_dir / f'analysis_report_{timestamp}.json', 'w') as f:
            json.dump({'analyses': valid_analyses, 'patterns': patterns, 'report': report}, f, indent=4)

        for name, img in visualizations.items():
            cv2.imwrite(str(report_dir / f'{name}_{timestamp}.png'), cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR))

        return {'analyses': valid_analyses, 'patterns': patterns, 'report': report}


def main():
    system = EnhancedTrafficManagementSystem()
    results = system.run_analysis()
    if results:
        print("\nAnalysis complete. Check the 'reports' directory for results and visualizations.")
    else:
        print("\nNo results were generated. Please check the error messages above.")


if __name__ == "__main__":
    main()
