import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import time
from queue import PriorityQueue
from typing import List, Dict, Tuple
import os

# Reuse the existing classes with minor modifications
class LightState(Enum):
    RED = "RED"
    GREEN = "GREEN"

@dataclass
class Lane:
    id: str
    direction: str
    congestion_score: float
    vehicle_count: int
    waiting_time: float = 0.0
    current_state: LightState = LightState.RED

# Reuse TrafficLightScheduler class as is
class TrafficLightScheduler:
    def __init__(self, min_green_time: int = 30, max_green_time: int = 120):
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.lanes: Dict[str, Lane] = {}
        self.current_green_lane = None
        self.last_switch_time = time.time()
        
    def update_lane_status(self, lane_id: str, analysis: dict) -> None:
        if lane_id not in self.lanes:
            self.lanes[lane_id] = Lane(
                id=lane_id,
                direction=lane_id[0],
                congestion_score=analysis['congestion_score'],
                vehicle_count=sum(analysis['vehicle_counts'].values())
            )
        else:
            self.lanes[lane_id].congestion_score = analysis['congestion_score']
            self.lanes[lane_id].vehicle_count = sum(analysis['vehicle_counts'].values())

    def calculate_priority_score(self, lane: Lane) -> float:
        congestion_weight = 0.4
        vehicle_weight = 0.3
        waiting_weight = 0.3
        
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
        base_time = self.min_green_time
        congestion_factor = lane.congestion_score / 100
        congestion_time = congestion_factor * 30
        vehicle_factor = min(lane.vehicle_count / 50, 1)
        vehicle_time = vehicle_factor * 30
        total_time = int(base_time + congestion_time + vehicle_time)
        return min(total_time, self.max_green_time)

    def update_waiting_times(self, elapsed_time: float) -> None:
        for lane in self.lanes.values():
            if lane.current_state == LightState.RED:
                lane.waiting_time += elapsed_time

    def get_next_green_lane(self) -> Tuple[Lane, int]:
        priority_queue = PriorityQueue()
        
        for lane in self.lanes.values():
            if lane.current_state == LightState.RED:
                priority_score = self.calculate_priority_score(lane)
                priority_queue.put((-priority_score, lane))
        
        if not priority_queue.empty():
            _, next_lane = priority_queue.get()
            green_time = self.calculate_green_time(next_lane)
            return next_lane, green_time
        
        return None, 0

    def get_current_states(self) -> Dict[str, Dict]:
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
        for i, analysis in enumerate(analyses):
            lane_id = f"{['N', 'S', 'E', 'W'][i % 4]}{i//4 + 1}"
            self.update_lane_status(lane_id, analysis)
        
        current_time = time.time()
        elapsed_time = current_time - self.last_switch_time
        
        if self.current_green_lane is None or elapsed_time >= self.calculate_green_time(self.current_green_lane):
            if self.current_green_lane:
                self.current_green_lane.current_state = LightState.RED
                self.current_green_lane.waiting_time = 0
            
            next_lane, green_time = self.get_next_green_lane()
            if next_lane:
                next_lane.current_state = LightState.GREEN
                next_lane.waiting_time = 0
                self.current_green_lane = next_lane
                self.last_switch_time = current_time
        
        return self.get_current_states()

class TrafficSimulator:
    """Simulates traffic data for demonstration purposes"""
    @staticmethod
    def generate_sample_data(num_lanes: int = 4) -> List[dict]:
        analyses = []
        current_hour = datetime.now().hour
        
        for _ in range(num_lanes):
            analysis = {
                "hour": current_hour,
                "congestion_score": np.random.randint(0, 100),
                "vehicle_counts": {
                    "car": np.random.randint(0, 50),
                    "truck": np.random.randint(0, 20),
                    "motorcycle": np.random.randint(0, 30),
                    "bus": np.random.randint(0, 10)
                },
                "timestamp": datetime.now().isoformat()
            }
            analyses.append(analysis)
        
        return analyses

def main():
    st.set_page_config(page_title="Traffic Management System", layout="wide")
    
    st.title("Real-Time Traffic Management System")
    st.markdown("---")

    # Initialize session state for the scheduler if it doesn't exist
    if 'scheduler' not in st.session_state:
        st.session_state.scheduler = TrafficLightScheduler()
        st.session_state.last_update = time.time()

    # Sidebar controls
    st.sidebar.title("Control Panel")
    num_lanes = st.sidebar.slider("Number of Lanes", 2, 8, 4)
    update_interval = st.sidebar.slider("Update Interval (seconds)", 1, 10, 3)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    # Auto-refresh mechanism
    current_time = time.time()
    if current_time - st.session_state.last_update > update_interval:
        st.session_state.last_update = current_time
        
        # Generate new traffic data
        analyses = TrafficSimulator.generate_sample_data(num_lanes)
        traffic_states = st.session_state.scheduler.update_schedule(analyses)
        
        # Display traffic light status
        with col1:
            st.subheader("Traffic Light Status")
            status_df = pd.DataFrame.from_dict(traffic_states, orient='index')
            
            # Style the dataframe
            def color_state(val):
                color = 'green' if val == 'GREEN' else 'red'
                return f'background-color: {color}; color: white'
            
            styled_df = status_df.style.applymap(
                color_state, 
                subset=['current_state']
            )
            
            st.dataframe(styled_df)
        
        # Display metrics
        with col2:
            st.subheader("System Metrics")
            metrics_col1, metrics_col2 = st.columns(2)
            
            avg_congestion = np.mean([state['congestion_score'] for state in traffic_states.values()])
            avg_wait_time = np.mean([state['waiting_time'] for state in traffic_states.values()])
            max_wait_time = max([state['waiting_time'] for state in traffic_states.values()])
            
            metrics_col1.metric("Average Congestion", f"{avg_congestion:.1f}%")
            metrics_col1.metric("Average Wait Time", f"{avg_wait_time:.1f}s")
            metrics_col2.metric("Max Wait Time", f"{max_wait_time:.1f}s")
            
            # Alerts
            high_congestion_lanes = [
                lane_id for lane_id, state in traffic_states.items()
                if state['congestion_score'] > 70
            ]
            
            if high_congestion_lanes:
                st.warning("⚠️ High Congestion Alert")
                for lane in high_congestion_lanes:
                    st.write(f"Lane {lane}: {traffic_states[lane]['congestion_score']:.1f}% congestion")
        
        # Vehicle distribution
        st.subheader("Vehicle Distribution")
        vehicle_counts = defaultdict(int)
        for analysis in analyses:
            for v_type, count in analysis['vehicle_counts'].items():
                vehicle_counts[v_type] += count
        
        fig, ax = plt.subplots()
        plt.pie(
            vehicle_counts.values(),
            labels=vehicle_counts.keys(),
            autopct='%1.1f%%',
            startangle=90
        )
        st.pyplot(fig)
        
    # Force refresh button
    if st.button("Force Refresh"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()