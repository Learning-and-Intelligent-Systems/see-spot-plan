  I've created a complete pipeline for remote data collection that avoids network camera streaming issues. Here's how it works:

  Two New Scripts:

  1. zed_collection_server.py (Nova GPU machine)
    - Listens on TCP port 9999 for connections
    - Receives joint data from Mac in real-time
    - Captures ZED frames locally (no network overhead)
    - Saves both synchronized to HDF5 file
  2. formatted_collect_body_arm_data_remote.py (Mac)
    - Connects to Spot robot
    - Collects joint/body data at 100 Hz
    - Streams data to Nova over the network (small packets, no images)
    - Completes when you press Ctrl+C

  Workflow:

  Mac (Spot Connection)              Nova GPU (ZED Camera)
          │                                  │
          ├──────── joint data ────────────→ │ Captures ZED frames
          │      (small packets)            │ locally in background
          │                                  │
          └─ Ctrl+C ──────────────────────→ │ Saves combined HDF5

  Key Benefits:

  - ✅ No network image streaming (avoids bandwidth/latency issues)
  - ✅ ZED frames captured locally on Nova (faster, more reliable)
  - ✅ Joint data synchronized with timestamps
  - ✅ Single HDF5 file with both robot data and camera images
  - ✅ Works exactly like the original formatted_replay_body_arm_data.py

  Quick Start:

  Terminal 1 (Nova):
  python zed_collection_server.py --output-dir teleoperation_data --port 9999

  Terminal 2 (Mac):
  python formatted_collect_body_arm_data_remote.py \
      --hostname 192.168.80.3 \
      --nova-host <nova-ip> \
      --nova-port 9999