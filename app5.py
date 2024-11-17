import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # Add this import
from datetime import datetime
import time
from collections import defaultdict
from queue import PriorityQueue
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


# Set the page configuration
st.set_page_config(page_title="Traffic Management System", layout="wide")


# Custom CSS for styling
st.markdown("""
    <style>
    .green-light { background-color: #4CAF50; color: white; }
    .red-light { background-color: #F44336; color: white; }
    .dataframe td, .dataframe th { text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Enums and Dataclasses
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
        weights = {'congestion': 0.4, 'vehicles': 0.3, 'waiting': 0.3}
        max_values = {
            'congestion': max((lane.congestion_score for lane in self.lanes.values()), default=1),
            'vehicles': max((lane.vehicle_count for lane in self.lanes.values()), default=1),
            'waiting': max((lane.waiting_time for lane in self.lanes.values()), default=1)
        }
        normalized = {
            'congestion': lane.congestion_score / (max_values['congestion'] or 1e-9),
            'vehicles': lane.vehicle_count / (max_values['vehicles'] or 1e-9),
            'waiting': lane.waiting_time / (max_values['waiting'] or 1e-9)
        }
        return sum(weights[k] * normalized[k] for k in weights)


    def calculate_green_time(self, lane: Lane) -> int:
        base_time = self.min_green_time
        congestion_time = (lane.congestion_score / 100) * 30
        vehicle_time = min((lane.vehicle_count / 50), 1) * 30
        return min(int(base_time + congestion_time + vehicle_time), self.max_green_time)

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
            return next_lane, self.calculate_green_time(next_lane)
        return None, 0

    def get_current_states(self) -> Dict[str, Dict]:
        current_time = time.time()
        elapsed_time = current_time - self.last_switch_time
        self.update_waiting_times(elapsed_time)
        return {
            lane_id: {
                'current_state': lane.current_state.value,
                'congestion_score': lane.congestion_score,
                'vehicle_count': lane.vehicle_count,
                'waiting_time': round(lane.waiting_time, 2),
                'priority_score': round(self.calculate_priority_score(lane), 2)
            }
            for lane_id, lane in self.lanes.items()
        }

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
            next_lane, _ = self.get_next_green_lane()
            if next_lane:
                next_lane.current_state = LightState.GREEN
                next_lane.waiting_time = 0
                self.current_green_lane = next_lane
                self.last_switch_time = current_time
        return self.get_current_states()

class TrafficSimulator:
    @staticmethod
    def generate_sample_data(num_lanes: int = 4) -> List[dict]:
        analyses = []
        current_hour = datetime.now().hour
        for _ in range(num_lanes):
            analyses.append({
                "hour": current_hour,
                "congestion_score": np.random.randint(0, 100),
                "vehicle_counts": {
                    "car": np.random.randint(0, 50),
                    "truck": np.random.randint(0, 20),
                    "motorcycle": np.random.randint(0, 30),
                    "bus": np.random.randint(0, 10)
                },
                "timestamp": datetime.now().isoformat()
            })
        return analyses

def main():
    st.title("Real-Time Traffic Management System")
    st.markdown("---")
    if 'scheduler' not in st.session_state:
        st.session_state.scheduler = TrafficLightScheduler()
        st.session_state.history = {"timestamps": [], "avg_congestion": [], "avg_wait_time": []}
    lanes_config = st.sidebar.expander("Configure Lanes")
    num_lanes = lanes_config.slider("Number of Lanes", 2, 8, 4)
    update_interval = st.sidebar.slider("Update Interval (seconds)", 1, 10, 3)
    current_time = time.time()
    if 'last_update' not in st.session_state or current_time - st.session_state.last_update > update_interval:
        st.session_state.last_update = current_time
        analyses = TrafficSimulator.generate_sample_data(num_lanes)
        traffic_states = st.session_state.scheduler.update_schedule(analyses)
        avg_congestion = np.mean([state['congestion_score'] for state in traffic_states.values()])
        avg_wait_time = np.mean([state['waiting_time'] for state in traffic_states.values()])
        st.session_state.history['timestamps'].append(time.time())
        st.session_state.history['avg_congestion'].append(avg_congestion)
        st.session_state.history['avg_wait_time'].append(avg_wait_time)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Traffic Light Status")
            status_df = pd.DataFrame.from_dict(traffic_states, orient='index')
            
            # Updated styling code
            def color_state(val):
                if val == 'GREEN':
                    return 'background-color: green; color: white'
                elif val == 'RED':
                    return 'background-color: red; color: white'
                return ''

            styled_df = status_df.style.map(
                lambda val: color_state(val) if isinstance(val, str) else '',
                subset=['current_state']
            )
            
            st.dataframe(styled_df)

        with col2:
            st.subheader("System Metrics")
            st.metric("Average Congestion", f"{avg_congestion:.1f}%")
            st.metric("Average Wait Time", f"{avg_wait_time:.1f}s")
            fig = px.line(pd.DataFrame(st.session_state.history), x="timestamps", y=["avg_congestion", "avg_wait_time"])
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
