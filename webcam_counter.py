"""
Real-time Footfall Counter using Webcam/RTSP Stream
===================================================
Processes live video feed from webcam or RTSP stream in real-time.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

class RealtimeFootfallCounter:
    """Real-time footfall counter for webcam or RTSP streams."""
    
    def __init__(self, source=0, line_position=0.5):
        """
        Initialize real-time counter.
        
        Args:
            source: 0 for webcam, or RTSP URL string
            line_position: Position of counting line (0.0-1.0)
        """
        self.source = source
        self.line_position = line_position
        
        # Open video stream
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.line_y = int(self.frame_height * line_position)
        
        print(f"Stream opened successfully!")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"Counting line at Y={self.line_y}")
        
        # Load YOLO model
        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')
        print("Model loaded!")
        
        # Tracking variables
        self.track_history = defaultdict(lambda: [])
        self.counted_ids = set()
        self.entry_count = 0
        self.exit_count = 0
        
        # Performance tracking
        self.fps_list = []
        self.frame_count = 0
        self.start_time = time.time()
    
    def get_centroid(self, box):
        """Calculate centroid of bounding box."""
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def check_line_crossing(self, track_id, current_centroid):
        """Check if person crossed the counting line."""
        history = self.track_history[track_id]
        
        if len(history) < 2:
            return None
        
        prev_y = history[-2][1]
        curr_y = current_centroid[1]
        
        # Check downward crossing (entry)
        if prev_y < self.line_y <= curr_y:
            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                return 'entry'
        # Check upward crossing (exit)
        elif prev_y > self.line_y >= curr_y:
            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                return 'exit'
        
        return None
    
    def draw_ui(self, frame, fps):
        """Draw UI elements on frame."""
        # Draw counting line
        cv2.line(frame, (0, self.line_y), (self.frame_width, self.line_y), 
                (0, 255, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (10, self.line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw semi-transparent info panel
        panel_height = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (450, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw statistics
        cv2.putText(frame, f"ENTRIES: {self.entry_count}", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"EXITS: {self.exit_count}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"NET: {self.entry_count - self.exit_count}", (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"TRACKING: {len(self.track_history)}", (10, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'Q' to quit | 'R' to reset", 
                   (10, self.frame_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def reset_counts(self):
        """Reset all counters."""
        self.entry_count = 0
        self.exit_count = 0
        self.counted_ids.clear()
        self.track_history.clear()
        print("\n🔄 Counters reset!")
    
    def run(self):
        """Run real-time processing."""
        print("\n" + "="*60)
        print("REAL-TIME FOOTFALL COUNTER STARTED")
        print("="*60)
        print("Press 'Q' to quit")
        print("Press 'R' to reset counters")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                self.frame_count += 1
                frame_start = time.time()
                
                # Run YOLO detection with tracking
                results = self.model.track(frame, persist=True, classes=[0], 
                                          verbose=False, conf=0.3, iou=0.5)
                
                # Process detections
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    # Process each detection
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        x1, y1, x2, y2 = box
                        centroid = self.get_centroid(box)
                        
                        # Update tracking history
                        self.track_history[track_id].append(centroid)
                        if len(self.track_history[track_id]) > 30:
                            self.track_history[track_id].pop(0)
                        
                        # Check for line crossing
                        crossing = self.check_line_crossing(track_id, centroid)
                        if crossing == 'entry':
                            self.entry_count += 1
                            print(f"🟢 ENTRY! ID: {track_id} | Total: {self.entry_count}")
                        elif crossing == 'exit':
                            self.exit_count += 1
                            print(f"🔴 EXIT! ID: {track_id} | Total: {self.exit_count}")
                        
                        # Draw bounding box
                        color = (0, 255, 0) if centroid[1] < self.line_y else (0, 165, 255)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw label
                        label = f"ID:{track_id}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Draw trajectory
                        points = self.track_history[track_id]
                        for i in range(1, len(points)):
                            cv2.line(frame, points[i-1], points[i], (230, 230, 230), 2)
                        
                        # Draw centroid
                        cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_list.append(fps)
                if len(self.fps_list) > 30:
                    self.fps_list.pop(0)
                avg_fps = sum(self.fps_list) / len(self.fps_list)
                
                # Draw UI
                frame = self.draw_ui(frame, avg_fps)
                
                # Display frame
                cv2.imshow('Real-time Footfall Counter', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nStopping...")
                    break
                elif key == ord('r') or key == ord('R'):
                    self.reset_counts()
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            total_time = time.time() - self.start_time
            print("\n" + "="*60)
            print("SESSION COMPLETE")
            print("="*60)
            print(f"Total Frames: {self.frame_count}")
            print(f"Runtime: {total_time:.1f} seconds")
            print(f"Average FPS: {self.frame_count/total_time:.1f}")
            print(f"\nFinal Count:")
            print(f"  Entries: {self.entry_count}")
            print(f"  Exits: {self.exit_count}")
            print(f"  Net Flow: {self.entry_count - self.exit_count}")
            print("="*60)


def main():
    """Main function to start real-time counter."""
    
    # Configuration
    SOURCE = 0  # 0 for webcam, or use RTSP URL: "rtsp://username:password@ip:port/stream"
    LINE_POSITION = 0.5  # 0.0 = top, 0.5 = middle, 1.0 = bottom
    
    print("="*60)
    print("REAL-TIME FOOTFALL COUNTER")
    print("="*60)
    print(f"Source: {'Webcam' if SOURCE == 0 else SOURCE}")
    print(f"Line Position: {LINE_POSITION*100}%")
    print("="*60 + "\n")
    
    try:
        counter = RealtimeFootfallCounter(source=SOURCE, line_position=LINE_POSITION)
        counter.run()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()