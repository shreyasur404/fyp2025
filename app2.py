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
        """Create heatmap visualization of traffic patterns with suggested green time."""
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

        # Add suggested green time based on congestion score
        pivot_data['suggested_green_time'] = pivot_data['congestion'].apply(
            lambda x: round(30 + (x / 2))  # Example logic: base time of 30 seconds + dynamic adjustment
        )

        # Pivot data to structure it for a heatmap (hour vs. congestion score)
        pivot_table = pd.pivot_table(
            pivot_data, 
            values='congestion', 
            index='hour', 
            aggfunc='mean'
        ).fillna(0)

        # Use Rocket palette for heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            pivot_table, 
            annot=True, 
            cmap='rocket',  # Rocket colormap
            cbar_kws={'label': 'Average Congestion Score'},
            fmt=".1f"
        )
        plt.xlabel('Hour of Day')
        plt.ylabel('Congestion')
        plt.title('Traffic Congestion Heatmap with Suggested Green Times')
        plt.tight_layout()

        fig = plt.gcf()
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close()

        # Save green time suggestions to a CSV for further action
        green_time_file = "green_time_suggestions.csv"
        pivot_data.to_csv(green_time_file, index=False)
        print(f"Suggested green times saved to {green_time_file}")

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


    def create_hourly_traffic_distribution(self, analyses: list) -> np.ndarray:
        """Create bar plot of traffic distribution throughout the day"""
        hourly_data = defaultdict(int)

        for analysis in analyses:
            if 'hour' in analysis and 'congestion_score' in analysis:
                hourly_data[analysis['hour']] += analysis['congestion_score']

        # Convert to DataFrame for easy plotting
        hourly_df = pd.DataFrame(list(hourly_data.items()), columns=['Hour', 'Total Congestion'])

        plt.figure(figsize=(12, 6))
        sns.barplot(x='Hour', y='Total Congestion', data=hourly_df, palette='rocket')
        plt.title('Hourly Traffic Distribution')
        plt.xlabel('Hour of Day')
        plt.ylabel('Total Congestion')
        plt.tight_layout()

        fig = plt.gcf()
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close()

        return img_array


    def create_congestion_over_time_plot(self, analyses: list) -> np.ndarray:
        """Create time series plot of traffic congestion over time"""
        time_series_data = []

        for analysis in analyses:
            if 'timestamp' in analysis and 'congestion_score' in analysis:
                time_series_data.append({'time': analysis['timestamp'], 'congestion': analysis['congestion_score']})

        if not time_series_data:
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, 'No valid data for time series plot', horizontalalignment='center', verticalalignment='center')
            plt.title('Traffic Congestion Over Time')
            plt.tight_layout()
            fig = plt.gcf()
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close()
            return img_array

        df = pd.DataFrame(time_series_data)
        df['time'] = pd.to_datetime(df['time'])

        plt.figure(figsize=(12, 6))
        sns.lineplot(x='time', y='congestion', data=df, palette='rocket')
        plt.title('Traffic Congestion Over Time')
        plt.xlabel('Time')
        plt.ylabel('Congestion Score')
        plt.tight_layout()

        fig = plt.gcf()
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close()

        return img_array


class TrafficAnalyzer:
    """Dummy TrafficAnalyzer for example"""
    def analyze_image(self, img_path: str) -> dict:
        # Dummy analysis logic
        return {
            "filename": img_path,
            "hour": np.random.randint(0, 24),  # Random hour for demonstration
            "congestion_score": np.random.randint(0, 100),
            "vehicle_counts": {
                "car": np.random.randint(0, 50),
                "truck": np.random.randint(0, 20),
                "motorcycle": np.random.randint(0, 30),
                "bus": np.random.randint(0, 10)
            },
            "timestamp": datetime.now().isoformat()  # Add timestamp for time series
        }


class EnhancedTrafficManagementSystem:
    def __init__(self, dataset_dir: str = "traffic_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.analyzer = TrafficAnalyzer()
        self.visualizer = TrafficVisualizer()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def run_analysis(self) -> dict:
        if not self.dataset_dir.exists():
            print(f"Error: Directory '{self.dataset_dir}' does not exist.")
            return {}

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
                print(f"Error analyzing image {img_path}: {e}")

        return analyses


# Main function to execute the system
def main():
    system = EnhancedTrafficManagementSystem("traffic_dataset")
    analyses = system.run_analysis()

    if analyses:
        visualizer = TrafficVisualizer()

        # Get the heatmap image
        heatmap = visualizer.create_heatmap(analyses)
        plt.imshow(heatmap)
        plt.show()

        # Get the vehicle distribution plot
        vehicle_dist_plot = visualizer.create_vehicle_distribution_plot(analyses)
        plt.imshow(vehicle_dist_plot)
        plt.show()




if __name__ == "__main__":
    main()
