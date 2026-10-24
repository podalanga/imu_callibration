import websocket
import ujson as json
import threading
import time

# Thread-safe dictionary holding the absolute latest raw strings
latest_data = {
    'accel': None,
    'gyro': None,
    'mag': None
}

def create_sensor_stream(sensor_type, dict_key):
    """Creates an independent WebSocket connection for a specific sensor."""
    def on_message(ws, message):
        # Instantly overwrite the state. DO NOT parse JSON here.
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
    
    # skip_utf8_validation=True eliminates string checks for max speed
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

if __name__ == "__main__":
    print("Initializing Multi-Sensor Stream Engine...")
    start_all_streams()
    
    # Give the threads 1 second to handshake with the phone
    time.sleep(1)
    print("Streaming live! Press Ctrl+C to stop.")
    
    try:
        while True:
            # 1. Grab the latest strings from RAM instantly
            raw_accel = latest_data['accel']
            raw_gyro = latest_data['gyro']
            raw_mag = latest_data['mag']
            
            # 2. Proceed only if all three sensors have fired at least once
            if raw_accel and raw_gyro and raw_mag:
                # ujson parses the strings significantly faster than standard json
                accel_parsed = json.loads(raw_accel)
                gyro_parsed = json.loads(raw_gyro)
                mag_parsed = json.loads(raw_mag)
                
                # 3. Extract the uncalibrated X, Y, Z hardware values
                ax, ay, az = accel_parsed['values'][0:3]
                gx, gy, gz = gyro_parsed['values'][0:3]
                mx, my, mz = mag_parsed['values'][0:3]
                
                # --- EXECUTE YOUR SENSOR FUSION MATH / KALMAN FILTER HERE ---
                
                # 4. Print to terminal without scrolling spam
                # The end='\r' carriage return rewrites the same line over and over
                print(f"ACC [X:{ax:6.2f} Y:{ay:6.2f} Z:{az:6.2f}] | "
                      f"GYR [X:{gx:6.2f} Y:{gy:6.2f} Z:{gz:6.2f}] | "
                      f"MAG [X:{mx:6.2f} Y:{my:6.2f} Z:{mz:6.2f}]", end='\r')
            
            # 5. Sleep briefly to dictate your main processing loop speed (~60 FPS)
            # The background threads will still catch every 200Hz frame silently!
            time.sleep(0.016)
            
    except KeyboardInterrupt:
        print("\n\nSafely shutting down all threads...")