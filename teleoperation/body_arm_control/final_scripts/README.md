# Synchronized Data Collection for ACT++ Training

This guide explains how to run the synchronized data collection system that captures Spot robot teleoperation data from multiple sources (joint data, ZED camera, Kiwi iPhone camera) and saves it in ACT++ format.

## Architecture Overview

The system has **two components**:

1. **GPU Server (Nova machine)** - `synchronized_data_collection.py`
   - Receives joint data from Mac on port 9999
   - Receives Kiwi frames from iPhone on port 8888
   - Streams ZED camera frames locally (~12 Hz)
   - Synchronizes all data to a master clock at 20 Hz
   - Writes final dataset to HDF5 in ACT++ format

2. **Mac Client** - `formatted_collect_body_arm_data_remote.py`
   - Connects to Spot robot
   - Collects joint/gripper/body state at ~50 Hz
   - Sends data packets to GPU server on port 9999

## Quick Start

### Prerequisites

**GPU Machine (Nova):**
```bash
pip install h5py numpy opencv-python
# Also need ZED Python SDK (if not already installed)
```

**Mac Client:**
```bash
pip install bosdyn  # Boston Dynamics Spot SDK
```

---

## Step-by-Step Instructions

### 1. Start the GPU Server

On the **Nova GPU machine**, run:

```bash
cd /path/to/teleoperation/body_arm_control/final_scripts

python synchronized_data_collection.py \
    --output-dir teleoperation_data \
    --policy-hz 20 \
    --joint-port 9999 \
    --kiwi-port 8888 \
    --duration 120
```

**Parameters:**
- `--output-dir`: Directory to save HDF5 files (default: `teleoperation_data`)
- `--policy-hz`: Master clock frequency in Hz (default: 20 Hz)
  - Controls final data rate in HDF5
  - Joint data downsampled to this rate
  - Images upsampled to this rate
- `--joint-port`: Port for receiving joint data from Mac (default: 9999)
- `--kiwi-port`: Port for receiving Kiwi frames from iPhone (default: 8888)
- `--duration`: Approximate collection duration in seconds (default: 60)

**Expected Output:**
```
======================================================================
Synchronized Data Collection Server (ACT++ Format)
======================================================================
Master clock: 20 Hz (dt = 0.0500s)
Joint data: ~50 Hz → 20 Hz (interpolate + downsample)
ZED images: ~12 Hz → 20 Hz (nearest neighbor upsample)
Kiwi images: ~5 Hz → 20 Hz (nearest neighbor upsample)
Collection duration: ~120s

Started ZED camera streaming (~12 Hz)...

Joint server listening on 0.0.0.0:9999
Waiting for Mac client connection...

Kiwi server listening on 0.0.0.0:8888
Waiting for iPhone client connection...

All streams started. Collecting data...
Press Ctrl+C to stop.

[0.1s] Buffers: joint=1, zed=1, kiwi=0
[5.0s] Buffers: joint=251, zed=61, kiwi=0
[10.1s] Buffers: joint=501, zed=122, kiwi=2
...
```

### 2. Connect Mac Client (in another terminal)

On the **Mac**, run:

```bash
cd /path/to/teleoperation/body_arm_control/final_scripts

python formatted_collect_body_arm_data_remote.py \
    --hostname <SPOT_IP> \
    --nova-host <NOVA_IP> \
    --nova-port 9999 \
    --rate-hz 50
```

**Parameters:**
- `--hostname`: Spot robot IP address (required)
- `--nova-host`: Nova GPU machine IP address (required)
- `--nova-port`: Port on Nova for receiving data (default: 9999, must match server)
- `--rate-hz`: Collection rate on Mac in Hz (default: 100)
  - Recommend 50 Hz for stable data
  - Server will downsample to 20 Hz anyway

**Expected Output:**
```
Connected to Nova at <NOVA_IP>:9999

Collecting at 50 Hz
Streaming to Nova...

[Timestep 0]
  arm0.sh0: 0.000000 rad
  arm0.sh1: 0.000000 rad
  arm0.el0: 0.000000 rad
  arm0.el1: 0.000000 rad
  arm0.wr0: 0.000000 rad
  arm0.wr1: 0.000000 rad
  gripper: 0.000000 [CLOSED]
  body: x=0.000000 m, y=0.000000 m, z=0.000000 m
  velocity: 0.000000 m/s

[Timestep 1]
...
```

### 3. Connect iPhone with Kiwi App (Optional)

If you want to include iPhone camera data:

1. Start the Kiwi app on iPhone
2. Configure it to send frames to `<NOVA_IP>:8888`
3. The Kiwi stream will automatically start sending frames when connected

If Kiwi is not available, the system will still work with just ZED frames.

### 4. Perform Teleoperation

While the server and client are running:
- Teleoperate the robot (use whatever control interface you have)
- Data will automatically stream to the server
- All streams will be synchronized to the master 20 Hz clock

### 5. Stop Collection

- Press **Ctrl+C** on the GPU server terminal
- The server will:
  - Stop all data collection threads
  - Resample all data to the 20 Hz master grid
  - Write HDF5 file to disk
  - Print summary statistics

---

## Output Format

When collection finishes, you'll see:

```
======================================================================
Dataset Summary
======================================================================
File: episode_20250125_143022.hdf5
Timesteps: 2400
Duration: 120.00s
Policy frequency: 20 Hz

Data Shapes:
  observations/qpos: (2400, 11)
  observations/qvel: (2400, 11)
  observations/images/zed_camera: (2400, 720, 1280, 3)
  observations/images/arm_camera: (2400, 720, 1280, 3)
  action: (2400, 11)

File size: 15.32 GB
======================================================================
```

### HDF5 File Structure

```
episode_20250125_143022.hdf5
├── attributes:
│   ├── sim: False
│   └── compress: False
├── observations/
│   ├── qpos (2400, 11) float64
│   │   [0-5]: arm joint positions (radians)
│   │   [6]: gripper position (0-1, 0=closed, 1=open)
│   │   [7-9]: body position (x, y, z in meters)
│   │   [10-12]: body euler angles (yaw, pitch, roll in radians)
│   │
│   ├── qvel (2400, 11) float64
│   │   Same structure as qpos, but velocities
│   │   Computed via finite difference: (qpos[t+1] - qpos[t]) / dt
│   │
│   └── images/
│       ├── zed_camera (2400, 720, 1280, 3) uint8
│       │   RGB images from ZED camera
│       │   Upsampled from ~12 Hz to 20 Hz
│       │   GZIP compressed
│       │
│       └── arm_camera (2400, 720, 1280, 3) uint8
│           RGB images from Kiwi iPhone
│           Upsampled from ~5 Hz to 20 Hz
│           GZIP compressed
│
└── action (2400, 11) float64
    Teleoperation actions (= qpos in this case)
```

---

## Data Rates and Synchronization

| Source | Native Rate | Final Rate | Resampling Method |
|--------|------------|-----------|------------------|
| Joint Data (Mac) | ~50 Hz | 20 Hz | Linear interpolation (downsample) |
| ZED Camera | ~12 Hz | 20 Hz | Nearest neighbor (upsample) |
| Kiwi Camera | ~5 Hz | 20 Hz | Nearest neighbor (upsample) |

**Master Clock:** All data aligned to a fixed 20 Hz grid on the GPU machine using system time (`time.time()`)

---

## Troubleshooting

### Joint Client Won't Connect

**Problem:** Mac shows "Connection refused" or "Cannot connect to Nova"

**Solutions:**
1. Check Nova IP address is correct: `ifconfig | grep "inet "`
2. Check port 9999 is open on Nova firewall
3. Ensure GPU server is running and waiting for connection
4. Try connecting with `nc -zv <NOVA_IP> 9999` to test connectivity

### Kiwi Frames Not Appearing

**Problem:** Server shows `[0s] Buffers: joint=N, zed=N, kiwi=0`

**Solutions:**
1. iPhone might not be connected or sending
2. Check Kiwi app is configured to send to correct IP:port
3. Make sure Kiwi app is started and streaming
4. It's OK if Kiwi is missing - server will still collect ZED + joint data

### ZED Camera Not Streaming

**Problem:** Server shows `[5.0s] Buffers: joint=N, zed=0, kiwi=N`

**Solutions:**
1. Check ZED camera is plugged in and recognized
2. Verify ZED SDK is properly installed
3. Check USB 3.0 cable is being used
4. Try running ZED test: `python -c "from zed import stream_zed_frames; print(next(stream_zed_frames()))"`

### Low Joint Data Rate

**Problem:** Server shows very low joint buffer count

**Solutions:**
1. Check Mac is connected to Nova via network
2. Check `--rate-hz` on Mac isn't too low
3. Verify Spot robot connection is stable
4. Check system load on Mac

### HDF5 File is Very Large

**Problem:** Episode file is >20 GB

**Context:** This is expected!
- 2400 frames × 2 cameras × 720×1280 × 3 channels = ~6.6 GB per camera
- With GZIP compression, typically reduces to 60-70% of original size
- GZIP is applied per frame for efficient single-frame access during training

---

## Usage with ACT++ Training

Once you have collected episodes, you can train with ACT++:

```python
# In your ACT++ training script
SIM_TASK_CONFIGS = {
    'spot_teleop': {
        'dataset_dir': '/path/to/teleoperation_data',
        'num_episodes': 5,  # Number of episode_*.hdf5 files
        'episode_len': 2400,  # Timesteps per episode
        'camera_names': ['arm_camera', 'zed_camera']
    }
}
```

---

## Advanced Configuration

### Change Master Clock Frequency

To collect at 50 Hz instead of 20 Hz:

```bash
python synchronized_data_collection.py --policy-hz 50
```

This will:
- Store 5× more timesteps (more data)
- Increase file size ~5×
- Joint data will still interpolate from ~50 Hz (now minimal downsampling)
- Image data will be upsampled more aggressively

### Longer Collection Sessions

For sessions >5 minutes, increase buffer sizes by editing `synchronized_data_collection.py`:

```python
# Line ~130
joint_buffer = CircularTimestampBuffer(max_size=5000)  # ~40s at 50 Hz → now ~100s
zed_buffer = CircularTimestampBuffer(max_size=1000)    # ~80s at 12 Hz
kiwi_buffer = CircularTimestampBuffer(max_size=1000)   # ~200s at 5 Hz
```

### Monitor Network Bandwidth

During collection, on the Mac:

```bash
# Monitor network traffic
nettop -P -n -L 1 | grep python
```

Expected bandwidth: ~50 KB/s for joint data (very low)

---

## Performance Notes

- **CPU Usage:** GPU server uses <10% CPU during collection (mostly waiting for I/O)
- **Memory Usage:** Circular buffers use ~500 MB RAM
- **Disk I/O:** ~100-200 MB/s during HDF5 write (fast on modern drives)
- **Network:** Joint data is only ~3.2 KB per second (negligible)

---

## Next Steps

1. **Collect multiple episodes** using the same process
2. **Verify data quality** by loading an episode:
   ```python
   import h5py
   with h5py.File('episode_20250125_143022.hdf5', 'r') as f:
       print(f['observations/qpos'].shape)  # Should be (2400, 11)
       print(f['observations/images/zed_camera'].shape)  # Should be (2400, 720, 1280, 3)
   ```
3. **Train ACT++ model** with collected episodes
