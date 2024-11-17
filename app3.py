import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from queue import PriorityQueue

class LightState(Enum):
    RED = "RED"
    GREEN = "GREEN"

@dataclass
class Lane:
    id: str
    direction: str  # N, S, E, W
    congestion_score: float
    vehicle_count: int
    waiting_time: float = 0.0
    current_state: LightState = LightState.RED

class TrafficLightScheduler:
    def __init__(self, min_green_time: int = 30, max_green_time: int = 120):
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.lanes: Dict[str, Lane] = {}
        self.current_green_lane = None
        self.last_switch_time = time.time()
        
    def update_lane_status(self, lane_id: str, analysis: dict) -> None:
        """Update lane status with new traffic analysis data"""
        if lane_id not in self.lanes:
            self.lanes[lane_id] = Lane(
                id=lane_id,
                direction=lane_id[0],  # First character of lane_id should be direction
                congestion_score=analysis['congestion_score'],
                vehicle_count=sum(analysis['vehicle_counts'].values())
            )
        else:
            self.lanes[lane_id].congestion_score = analysis['congestion_score']
            self.lanes[lane_id].vehicle_count = sum(analysis['vehicle_counts'].values())

    def calculate_priority_score(self, lane: Lane) -> float:
        """Calculate priority score for a lane based on multiple factors"""
        congestion_weight = 0.4
        vehicle_weight = 0.3
        waiting_weight = 0.3
        
        # Normalize each factor
        max_congestion = max(lane.congestion_score for lane in self.lanes.values())
        max_vehicles = max(lane.vehicle_count for lane in self.lanes.values())
        max_waiting = max(lane.waiting_time for lane in self.lanes.values())
        
        normalized_congestion = lane.congestion_score / max_congestion if max_congestion > 0 else 0
        normalized_vehicles = lane.vehicle_count / max_vehicles if max_vehicles > 0 else 0
        normalized_waiting = lane.waiting_time / max_waiting if max_waiting > 0 else 0
        
        return (
            congestion_weight * normalized_congestion +
            vehicle_weight * normalized_vehicles +
            waiting_weight * normalized_waiting
        )

    def calculate_green_time(self, lane: Lane) -> int:
        """Calculate optimal green time based on congestion and vehicle count"""
        base_time = self.min_green_time
        
        # Add time based on congestion score (0-100)
        congestion_factor = lane.congestion_score / 100
        congestion_time = congestion_factor * 30  # Up to 30 additional seconds
        
        # Add time based on vehicle count
        vehicle_factor = min(lane.vehicle_count / 50, 1)  # Cap at 50 vehicles
        vehicle_time = vehicle_factor * 30  # Up to 30 additional seconds
        
        total_time = int(base_time + congestion_time + vehicle_time)
        return min(total_time, self.max_green_time)

    def update_waiting_times(self, elapsed_time: float) -> None:
        """Update waiting times for all red lanes"""
        for lane in self.lanes.values():
            if lane.current_state == LightState.RED:
                lane.waiting_time += elapsed_time

    def get_next_green_lane(self) -> Tuple[Lane, int]:
        """Determine which lane should get green light next and for how long"""
        priority_queue = PriorityQueue()
        
        # Calculate priority scores for all lanes
        for lane in self.lanes.values():
            if lane.current_state == LightState.RED:
                priority_score = self.calculate_priority_score(lane)
                # Negative priority score because PriorityQueue is min-heap
                priority_queue.put((-priority_score, lane))
        
        if not priority_queue.empty():
            _, next_lane = priority_queue.get()
            green_time = self.calculate_green_time(next_lane)
            return next_lane, green_time
        
        return None, 0

    def get_current_states(self) -> Dict[str, Dict]:
        """Get current states of all traffic lights with relevant metrics"""
        states = {}
        current_time = time.time()
        elapsed_time = current_time - self.last_switch_time
        
        self.update_waiting_times(elapsed_time)
        
        for lane_id, lane in self.lanes.items():
            states[lane_id] = {
                'current_state': lane.current_state.value,
                'congestion_score': lane.congestion_score,
                'vehicle_count': lane.vehicle_count,
                'waiting_time': round(lane.waiting_time, 2),
                'priority_score': round(self.calculate_priority_score(lane), 2)
            }
        
        return states

    def update_schedule(self, analyses: List[dict]) -> Dict[str, Dict]:
        """Update schedule based on new analyses and return current states"""
        # Update lane statuses
        for i, analysis in enumerate(analyses):
            lane_id = f"{['N', 'S', 'E', 'W'][i % 4]}{i//4 + 1}"  # Assign directions cyclically
            self.update_lane_status(lane_id, analysis)
        
        current_time = time.time()
        elapsed_time = current_time - self.last_switch_time
        
        # Check if we need to switch lights
        if self.current_green_lane is None or elapsed_time >= self.calculate_green_time(self.current_green_lane):
            # Reset current green lane to red if it exists
            if self.current_green_lane:
                self.current_green_lane.current_state = LightState.RED
                self.current_green_lane.waiting_time = 0
            
            # Get next lane to turn green
            next_lane, green_time = self.get_next_green_lane()
            if next_lane:
                next_lane.current_state = LightState.GREEN
                next_lane.waiting_time = 0
                self.current_green_lane = next_lane
                self.last_switch_time = current_time
        
        return self.get_current_states()

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

        # Pivot data for heatmap
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
            cmap='rocket',
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

        # Save green time suggestions
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

    def create_traffic_light_status_plot(self, traffic_states: Dict[str, Dict]) -> np.ndarray:
        """Create visualization of current traffic light states"""
        plt.figure(figsize=(12, 8))
        
        lanes = list(traffic_states.keys())
        states = [state['current_state'] == 'GREEN' for state in traffic_states.values()]
        congestion = [state['congestion_score'] for state in traffic_states.values()]
        
        # Create bar chart
        bars = plt.bar(lanes, congestion)
        
        # Color bars based on traffic light state
        for bar, state in zip(bars, states):
            bar.set_color('green' if state else 'red')
            
        plt.title('Current Traffic Light States and Congestion Scores')
        plt.xlabel('Lanes')
        plt.ylabel('Congestion Score')
        
        # Add waiting time annotations
        for i, lane in enumerate(lanes):
            plt.text(i, congestion[i] + 2, 
                    f"Wait: {traffic_states[lane]['waiting_time']}s\n"
                    f"Priority: {traffic_states[lane]['priority_score']}", 
                    ha='center')
        
        plt.tight_layout()
        
        fig = plt.gcf()
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close()
        
        return img_array

class TrafficAnalyzer:
    """Traffic Analyzer for processing traffic images"""
    def analyze_image(self, img_path: str) -> dict:
        # Dummy analysis logic - replace with actual computer vision analysis
        return {
            "filename": img_path,
            "hour": np.random.randint(0, 24),
            "congestion_score": np.random.randint(0, 100),
            "vehicle_counts": {
                "car": np.random.randint(0, 50),
                "truck": np.random.randint(0, 20),
                "motorcycle": np.random.randint(0, 30),
                "bus": np.random.randint(0, 10)
            },
            "timestamp": datetime.now().isoformat()
        }

class EnhancedTrafficManagementSystem:
    def __init__(self, dataset_dir: str = "traffic_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.analyzer = TrafficAnalyzer()
        self.visualizer = TrafficVisualizer()
        self.scheduler = TrafficLightScheduler()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def run_analysis(self) -> Tuple[List[dict], Dict[str, Dict]]:
        """Run analysis on traffic images and return both analyses and traffic states"""
        if not self.dataset_dir.exists():
            print(f"Error: Directory '{self.dataset_dir}' does not exist.")
            return [], {}

        image_files = (
            list(self.dataset_dir.rglob("*.jpg")) +
            list(self.dataset_dir.rglob("*.jpeg")) +
            list(self.dataset_dir.rglob("*.png"))
        )

        if not image_files:
            print(f"Error: No image files found in '{self.dataset_dir}'")
            return [], {}

        analyses = []
        for img_path in image_files:
            try:
                analysis = self.analyzer.analyze_image(str(img_path))
                analyses.append(analysis)
            except Exception as e:
                print(f"Error analyzing image {img_path}: {e}")

        # Get traffic light states
        traffic_states = self.scheduler.update_schedule(analyses)
        
        return analyses, traffic_states

# [Previous code remains exactly the same until the main function]

def main():
    # Initialize the system
    print("Initializing Traffic Management System...")
    system = EnhancedTrafficManagementSystem("traffic_dataset")
    
    # Run analysis
    print("\nAnalyzing traffic data...")
    analyses, traffic_states = system.run_analysis()
    
    if analyses:
        visualizer = TrafficVisualizer()
        
        # Create and display all visualizations
        print("\nGenerating visualizations...")
        
        # 1. Traffic Light Status Plot
        print("\n1. Creating traffic light status visualization...")
        status_plot = visualizer.create_traffic_light_status_plot(traffic_states)
        plt.figure(figsize=(12, 8))
        plt.imshow(status_plot)
        plt.axis('off')
        plt.title("Current Traffic Light States")
        plt.show()
        
        # 2. Congestion Heatmap
        print("\n2. Creating congestion heatmap...")
        heatmap = visualizer.create_heatmap(analyses)
        plt.figure(figsize=(12, 6))
        plt.imshow(heatmap)
        plt.axis('off')
        plt.title("Traffic Congestion Heatmap")
        plt.show()
        
        # 3. Vehicle Distribution
        sns.set_palette("flare")
        print("\n3. Creating vehicle distribution plot...")
        vehicle_dist = visualizer.create_vehicle_distribution_plot(analyses)
        plt.figure(figsize=(10, 8))
        plt.imshow(vehicle_dist)
        plt.axis('off')
        plt.title("Vehicle Type Distribution")
        plt.show()
        
        # Print detailed traffic states report
        print("\nCurrent Traffic States Report:")
        print("-" * 80)
        for lane_id, state in traffic_states.items():
            print(f"\nLane {lane_id}:")
            print(f"  Current State: {state['current_state']}")
            print(f"  Congestion Score: {state['congestion_score']}")
            print(f"  Vehicle Count: {state['vehicle_count']}")
            print(f"  Waiting Time: {state['waiting_time']}s")
            print(f"  Priority Score: {state['priority_score']}")
        
        # Save analysis results to JSON
        print("\nSaving analysis results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'timestamp': timestamp,
            'traffic_states': traffic_states,
            'analyses': analyses
        }
        
        with open(f'traffic_analysis_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to traffic_analysis_{timestamp}.json")
        
        # Print recommendations
        print("\nTraffic Management Recommendations:")
        print("-" * 80)
        high_congestion_lanes = [
            lane_id for lane_id, state in traffic_states.items()
            if state['congestion_score'] > 70
        ]
        
        if high_congestion_lanes:
            print("\nHigh Congestion Alerts:")
            for lane_id in high_congestion_lanes:
                print(f"- Lane {lane_id} requires immediate attention "
                      f"(Congestion Score: {traffic_states[lane_id]['congestion_score']})")
        
        long_wait_lanes = [
            lane_id for lane_id, state in traffic_states.items()
            if state['waiting_time'] > 120
        ]
        
        if long_wait_lanes:
            print("\nLong Wait Time Alerts:")
            for lane_id in long_wait_lanes:
                print(f"- Lane {lane_id} has excessive wait time "
                      f"({traffic_states[lane_id]['waiting_time']}s)")
        
        # Calculate and display system performance metrics
        print("\nSystem Performance Metrics:")
        print("-" * 80)
        avg_congestion = np.mean([state['congestion_score'] for state in traffic_states.values()])
        avg_wait_time = np.mean([state['waiting_time'] for state in traffic_states.values()])
        max_wait_time = max([state['waiting_time'] for state in traffic_states.values()])
        
        print(f"Average Congestion Score: {avg_congestion:.2f}")
        print(f"Average Wait Time: {avg_wait_time:.2f}s")
        print(f"Maximum Wait Time: {max_wait_time:.2f}s")
        
    else:
        print("No analysis data available. Please check your dataset directory.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        print("\nTraffic Management System shutting down...")