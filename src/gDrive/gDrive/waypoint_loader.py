import csv
import os
import rclpy

class WaypointLoader:
    def __init__(self):
        pass

    def load_from_csv(self, file_path):
        """
        Reads a CSV file and returns a list of waypoints.
        Expected format: x, y (and optionally z or yaw, but we usually just need x,y)
        
        Args:
            file_path (str): Absolute or relative path to the .csv file
            
        Returns:
            list: [[x1, y1], [x2, y2], ...]
        """
        waypoints = []
        
        if not os.path.exists(file_path):
            print(f"[WaypointLoader] Error: File not found at {file_path}")
            return []

        try:
            with open(file_path, mode='r') as csv_file:
                csv_reader = csv.reader(csv_file)
                
                for line_num, row in enumerate(csv_reader):
                    # Skip empty lines
                    if not row:
                        continue
                        
                    # Skip header lines (if first char is a letter or #)
                    if row[0].strip().startswith('#') or row[0].strip()[0].isalpha():
                        continue
                        
                    try:
                        # Parse X and Y
                        x = float(row[0])
                        y = float(row[1])
                        waypoints.append([x, y])
                    except ValueError:
                        print(f"[WaypointLoader] Warning: Could not parse line {line_num + 1}: {row}")
                        continue
                        
            print(f"[WaypointLoader] Successfully loaded {len(waypoints)} waypoints.")
            return waypoints

        except Exception as e:
            print(f"[WaypointLoader] Failed to read file: {e}")
            return []