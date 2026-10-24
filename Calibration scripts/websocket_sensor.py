import websocket
import ujson as json
import threading
import time
import csv
import numpy as np
import os

# Filename for logging CSV output
CSV_FILENAME = "imu_calibrated_data.csv"

# Thread-safe dictionary holding the absolute latest raw strings
latest_data = {
    'accel': None,
    'gyro': None,
    'mag': None
}

# --- CALIBRATION PARAMETERS ---
# 1. Gyroscope Zero-Rate Offsets (rad/s or deg/s)
gyro_bias = np.array([0.0, 0.0, 0.0])

# 2. Accelerometer Scale Factor & Offsets (m/s^2)
accel_offset = np.array([0.0, 0.0, 0.0])
accel_scale = np.array([1.0, 1.0, 1.0])

# 3. Magnetometer Hard Iron (Offsets) & Soft Iron (Scaling matrix/vector)
mag_bias = np.array([0.0, 0.0, 0.0])      # Hard-iron offset
mag_scale = np.array([1.0, 1.0, 1.0])     # Soft-iron axis scale correction


def create_sensor_stream(sensor_type, dict_key):
    """Creates an independent WebSocket connection for a specific sensor."""
    def on_message(ws, message):
        latest_data[dict_key] = message

    def on_error(ws, error):
        print(f"Error [{dict_key.upper()}]: {error}")

    def on_close(ws, close_status_code, close_msg):
        print(f"--- {dict_key.upper()} Stream Disconnected ---")

    url = f"ws://localhost:8080/sensor/connect?type={sensor_type}"
    ws = websocket.WebSocketApp(url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    
    ws.run_forever(ping_interval=10, skip_utf8_validation=True)


def start_all_streams():
    """Spawns 3 daemon threads to run the WebSockets concurrently."""
    sensors = {
        'accel': 'android.sensor.accelerometer_uncalibrated',
        'gyro': 'android.sensor.gyroscope_uncalibrated',
        'mag': 'android.sensor.magnetic_field_uncalibrated'
    }

    for key, s_type in sensors.items():
        t = threading.Thread(target=create_sensor_stream, args=(s_type, key), daemon=True)
        t.start()
        print(f"Started dedicated thread for {key.upper()}...")


def calibrate_gyroscope(duration=5):
    """
    Computes zero-rate offsets by averaging stationary gyro data for 'duration' seconds.
    Keep the phone completely STILL during this process.
    """
    global gyro_bias
    print(f"\n[CALIBRATION] Keep your phone completely STATIONARY for {duration} seconds...")
    for i in range(3, 0, -1):
        print(f"Starting in {i}...", end='\r')
        time.sleep(1)
    
    samples = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        if latest_data['gyro']:
            gyro_parsed = json.loads(latest_data['gyro'])
            gx, gy, gz = gyro_parsed['values'][0:3]
            samples.append([gx, gy, gz])
        time.sleep(0.01)
        
    if samples:
        gyro_bias = np.mean(samples, axis=0)
        print(f"\n[SUCCESS] Gyro Bias Calculated: X={gyro_bias[0]:.5f}, Y={gyro_bias[1]:.5f}, Z={gyro_bias[2]:.5f}")
    else:
        print("\n[ERROR] No gyro data received during calibration!")


def calibrate_magnetometer(duration=15):
    """
    Computes Hard-Iron offsets and Soft-Iron scaling by collecting 3D sphere data.
    Rotate phone in a Figure-8 pattern across all axes for 'duration' seconds.
    """
    global mag_bias, mag_scale
    print(f"\n[CALIBRATION] Rotate your phone in a FIGURE-8 / 3D SPHERE pattern for {duration} seconds...")
    for i in range(3, 0, -1):
        print(f"Starting in {i}...", end='\r')
        time.sleep(1)

    samples = []
    start_time = time.time()

    while time.time() - start_time < duration:
        if latest_data['mag']:
            mag_parsed = json.loads(latest_data['mag'])
            mx, my, mz = mag_parsed['values'][0:3]
            samples.append([mx, my, mz])
        time.sleep(0.01)

    if len(samples) > 50:
        pts = np.array(samples)
        # Min/Max bounding box technique for quick Hard & Soft iron correction
        min_xyz = np.min(pts, axis=0)
        max_xyz = np.max(pts, axis=0)

        # Hard-iron offset (center of the bounding box)
        mag_bias = (max_xyz + min_xyz) / 2.0

        # Soft-iron scale (normalize radius along all axes)
        radii = (max_xyz - min_xyz) / 2.0
        avg_radius = np.mean(radii)
        mag_scale = avg_radius / radii

        print(f"\n[SUCCESS] Mag Bias (Hard Iron): {mag_bias}")
        print(f"[SUCCESS] Mag Scale (Soft Iron): {mag_scale}")
    else:
        print("\n[ERROR] Not enough magnetometer samples collected!")


def apply_calibration(raw_acc, raw_gyr, raw_mag):
    """Applies calibration constants to raw readings."""
    raw_acc = np.array(raw_acc)
    raw_gyr = np.array(raw_gyr)
    raw_mag = np.array(raw_mag)

    # 1. Accelerometer Correction
    cal_acc = (raw_acc - accel_offset) * accel_scale

    # 2. Gyroscope Offset Subtraction
    cal_gyr = raw_gyr - gyro_bias

    # 3. Magnetometer Hard Iron & Soft Iron Correction
    cal_mag = (raw_mag - mag_bias) * mag_scale

    return cal_acc, cal_gyr, cal_mag


if __name__ == "__main__":
    print("Initializing Multi-Sensor Stream Engine...")
    start_all_streams()
    
    # Wait for sensor connection
    time.sleep(1.5)
    
    # --- OPTIONAL: RUN LIVE CALIBRATION ROUTINES ---
    user_cal = input("\nDo you want to run quick live calibration? (y/n): ").strip().lower()
    if user_cal == 'y':
        calibrate_gyroscope(duration=5)
        calibrate_magnetometer(duration=12)

    print("\nStarting stream & CSV logging... Press Ctrl+C to stop.")

    # Initialize CSV File Header
    headers = [
        "Timestamp",
        "Raw_Ax", "Raw_Ay", "Raw_Az",
        "Raw_Gx", "Raw_Gy", "Raw_Gz",
        "Raw_Mx", "Raw_My", "Raw_Mz",
        "Cal_Ax", "Cal_Ay", "Cal_Az",
        "Cal_Gx", "Cal_Gy", "Cal_Gz",
        "Cal_Mx", "Cal_My", "Cal_Mz"
    ]

    file_exists = os.path.isfile(CSV_FILENAME)
    
    with open(CSV_FILENAME, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(headers)

        try:
            while True:
                raw_accel = latest_data['accel']
                raw_gyro = latest_data['gyro']
                raw_mag = latest_data['mag']
                
                if raw_accel and raw_gyro and raw_mag:
                    # Time stamp
                    current_time = time.time()

                    # Parse JSON
                    accel_parsed = json.loads(raw_accel)
                    gyro_parsed = json.loads(raw_gyro)
                    mag_parsed = json.loads(raw_mag)
                    
                    # Extract 3D values
                    raw_a = accel_parsed['values'][0:3]
                    raw_g = gyro_parsed['values'][0:3]
                    raw_m = mag_parsed['values'][0:3]

                    # Apply General Calibration
                    cal_a, cal_g, cal_m = apply_calibration(raw_a, raw_g, raw_m)

                    # Save row to CSV
                    writer.writerow([
                        current_time,
                        *raw_a, *raw_g, *raw_m,
                        *cal_a, *cal_g, *cal_m
                    ])

                    # Live terminal display
                    print(f"ACC: [{cal_a[0]:6.2f}, {cal_a[1]:6.2f}, {cal_a[2]:6.2f}] | "
                          f"GYR: [{cal_g[0]:6.2f}, {cal_g[1]:6.2f}, {cal_g[2]:6.2f}] | "
                          f"MAG: [{cal_m[0]:6.2f}, {cal_m[1]:6.2f}, {cal_m[2]:6.2f}]", end='\r')
                
                time.sleep(0.016)  # ~60 FPS rate

        except KeyboardInterrupt:
            print(f"\n\nData collection stopped. CSV saved to: {os.path.abspath(CSV_FILENAME)}")