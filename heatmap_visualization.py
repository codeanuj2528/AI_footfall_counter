"""
Footfall Counter with Heatmap Visualization
===========================================
Generates activity heatmaps showing where people spend most time.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import os

class FootfallCounterWithHeatmap:
    """Footfall counter with heatmap visualization."""
    
    def __init__(self, video_path, line_position=0.5):
        self.video_path = video_path
        self.line_position = line_position
        
        # Validate video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.line_y = int(self.frame_height * line_position)
        
        if self.fps == 0:
            self.fps = 30
        
        print(f"Video: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")
        
        # Load YOLO model
        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')
        print("Model loaded!")
        
        # Tracking variables
        self.track_history = defaultdict(lambda: [])
        self.counted_ids = set()
        self.entry_count = 0
        self.exit_count = 0
        
        # Heatmap initialization
        self.heatmap = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        self.heatmap_scale = 4  # Downscale for performance
        self.heatmap_small = np.zeros(
            (self.frame_height // self.heatmap_scale, 
             self.frame_width // self.heatmap_scale), 
            dtype=np.float32
        )
    
    def get_centroid(self, box):
        """Calculate centroid of bounding box."""
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def update_heatmap(self, centroids):
        """Update heatmap with current detections."""
        for cx, cy in centroids:
            # Scale coordinates
            sx = cx // self.heatmap_scale
            sy = cy // self.heatmap_scale
            
            # Add Gaussian blob around detection
            if 0 <= sx < self.heatmap_small.shape[1] and 0 <= sy < self.heatmap_small.shape[0]:
                # Create Gaussian kernel (5x5)
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nx, ny = sx + dx, sy + dy
                        if 0 <= nx < self.heatmap_small.shape[1] and 0 <= ny < self.heatmap_small.shape[0]:
                            # Gaussian weight
                            weight = np.exp(-(dx*dx + dy*dy) / 2.0)
                            self.heatmap_small[ny, nx] += weight
    
    def get_heatmap_overlay(self, frame):
        """Generate heatmap overlay for current frame."""
        # Resize heatmap to full resolution
        heatmap_full = cv2.resize(self.heatmap_small, 
                                 (self.frame_width, self.frame_height),
                                 interpolation=cv2.INTER_LINEAR)
        
        # Normalize to 0-255
        if heatmap_full.max() > 0:
            heatmap_norm = (heatmap_full / heatmap_full.max() * 255).astype(np.uint8)
        else:
            heatmap_norm = np.zeros_like(heatmap_full, dtype=np.uint8)
        
        # Apply colormap (COLORMAP_JET: blue=cold, red=hot)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Blend with original frame
        overlay = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)
        
        return overlay
    
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
    
    def draw_ui(self, frame):
        """Draw UI elements on frame."""
        # Draw counting line
        cv2.line(frame, (0, self.line_y), (self.frame_width, self.line_y), 
                (0, 255, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (10, self.line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw info panel
        panel_height = 150
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw statistics
        cv2.putText(frame, f"ENTRIES: {self.entry_count}", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"EXITS: {self.exit_count}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"TRACKING: {len(self.track_history)}", (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        return frame
    
    def process_video(self, output_path='output_heatmap.mp4', heatmap_path='heatmap_final.jpg'):
        """Process video and generate heatmap."""
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                             (self.frame_width, self.frame_height))
        
        frame_count = 0
        start_time = time.time()
        
        print("\n" + "="*60)
        print("PROCESSING WITH HEATMAP")
        print("="*60)
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run YOLO detection with tracking
                results = self.model.track(frame, persist=True, classes=[0], 
                                          verbose=False, conf=0.3, iou=0.5)
                
                centroids = []
                
                # Process detections
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        x1, y1, x2, y2 = box
                        centroid = self.get_centroid(box)
                        centroids.append(centroid)
                        
                        # Update tracking history
                        self.track_history[track_id].append(centroid)
                        if len(self.track_history[track_id]) > 30:
                            self.track_history[track_id].pop(0)
                        
                        # Check for line crossing
                        crossing = self.check_line_crossing(track_id, centroid)
                        if crossing == 'entry':
                            self.entry_count += 1
                        elif crossing == 'exit':
                            self.exit_count += 1
                        
                        # Draw bounding box
                        color = (0, 255, 0) if centroid[1] < self.line_y else (0, 165, 255)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw label
                        label = f"ID:{track_id}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update heatmap
                if centroids:
                    self.update_heatmap(centroids)
                
                # Create heatmap overlay
                frame_with_heatmap = self.get_heatmap_overlay(frame)
                
                # Draw UI on top
                frame_final = self.draw_ui(frame_with_heatmap)
                
                # Write frame
                out.write(frame_final)
                
                # Progress
                if frame_count % 50 == 0:
                    progress = (frame_count / self.total_frames * 100) if self.total_frames > 0 else 0
                    print(f"Progress: {progress:.1f}% - Entries: {self.entry_count} | Exits: {self.exit_count}")
        
        finally:
            self.cap.release()
            out.release()
        
        # Save final heatmap
        heatmap_full = cv2.resize(self.heatmap_small, 
                                 (self.frame_width, self.frame_height),
                                 interpolation=cv2.INTER_LINEAR)
        if heatmap_full.max() > 0:
            heatmap_norm = (heatmap_full / heatmap_full.max() * 255).astype(np.uint8)
        else:
            heatmap_norm = np.zeros_like(heatmap_full, dtype=np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_path, heatmap_color)
        
        # Final report
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Entries: {self.entry_count} | Exits: {self.exit_count}")
        print(f"Output: {output_path}")
        print(f"Heatmap: {heatmap_path}")
        print("="*60)


def main():
    VIDEO_PATH = r"C:\Users\shiva\Downloads\footfall_count (2)\footfall_count\trial_vedio.mp4"
    OUTPUT_PATH = 'output_heatmap.mp4'
    HEATMAP_PATH = 'heatmap_final.jpg'
    LINE_POSITION = 0.5
    
    print("FOOTFALL COUNTER WITH HEATMAP")
    print("="*60)
    
    try:
        counter = FootfallCounterWithHeatmap(VIDEO_PATH, line_position=LINE_POSITION)
        counter.process_video(output_path=OUTPUT_PATH, heatmap_path=HEATMAP_PATH)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()