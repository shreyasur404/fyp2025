This code implements a **real-time traffic management system** using Streamlit. Below is a function-by-function explanation:

### **1. `Lane` Class**
- **Purpose**: Represents a single traffic lane's state and statistics.
- **Attributes**:
  - `id`: Lane identifier.
  - `direction`: Direction (e.g., North, South).
  - `congestion_score`: How congested the lane is (0-100).
  - `vehicle_count`: Number of vehicles in the lane.
  - `waiting_time`: Time vehicles have been waiting.
  - `current_state`: Current light state (RED/GREEN).

---

### **2. `TrafficLightScheduler` Class**
- **Purpose**: Manages the traffic lights and prioritizes lanes based on congestion, vehicle count, and waiting time.

#### **Methods**:
- **`__init__`**:
  - Initializes the scheduler with default minimum and maximum green times.
  - Tracks lanes, the current green lane, and last switch time.

- **`update_lane_status`**:
  - Updates or initializes the status of a lane based on traffic analysis.

- **`calculate_priority_score`**:
  - Calculates a priority score for each lane based on:
    - Congestion, vehicle count, and waiting time.
  - Normalizes each metric and combines them with weights.

- **`calculate_green_time`**:
  - Determines the green light duration for a lane based on congestion and vehicle count.

- **`update_waiting_times`**:
  - Updates waiting times for all lanes, increasing them if the lane's light is red.

- **`get_next_green_lane`**:
  - Selects the next lane to get a green light using priority scores and determines its green time.

- **`get_current_states`**:
  - Returns the current state of all lanes (light state, congestion score, vehicle count, waiting time, priority score).

- **`update_schedule`**:
  - Updates lane statuses and switches the green light based on priority.

---

### **3. `TrafficSimulator` Class**
- **Purpose**: Simulates traffic data for demonstration.
- **Method**:
  - **`generate_sample_data`**:
    - Generates random data for a specified number of lanes, including:
      - Congestion scores, vehicle counts by type, and timestamps.

---

### **4. `main` Function**
- **Purpose**: Implements the Streamlit app interface.
- **Key Elements**:
  - **Initialize Scheduler**:
    - Creates a `TrafficLightScheduler` instance in Streamlit session state.
  
  - **Sidebar Controls**:
    - Allows users to adjust the number of lanes and update interval.

  - **Main Content**:
    - Periodically updates traffic data and schedules green lights.
    - Displays:
      - Traffic light status (as a styled DataFrame).
      - System metrics (e.g., average congestion, wait times).
      - Alerts for high congestion.
      - Vehicle distribution (pie chart).

  - **Force Refresh Button**:
    - Allows users to manually trigger data refresh.

---

### **5. Streamlit Features**
- **Real-Time Updates**: Periodically refreshes traffic data and light states.
- **Visualizations**:
  - Styled table for lane statuses.
  - Metrics for congestion and wait times.
  - Pie chart for vehicle type distribution.
- **User Controls**: Customizable lane count and update frequency.

This code creates a dynamic, interactive traffic management system simulation.
