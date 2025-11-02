"""
FIXED VERSION - Footfall Counter with Video Validation
======================================================
This version includes proper error handling and video validation.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import os

class FootfallCounterDebug:
    """Debug version with enhanced logging and error handling."""
    
    def __init__(self, video_path, line_position=0.5):
        self.video_path = video_path
        self.line_position = line_position
        
        # Validate video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"Found video file: {video_path}")
        print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
        
        # Try to open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate video properties
        if self.frame_width == 0 or self.frame_height == 0:
            raise ValueError(f"Invalid video dimensions: {self.frame_width}x{self.frame_height}")
        
        if self.fps == 0:
            print("Warning: FPS is 0, defaulting to 30")
            self.fps = 30
        
        self.line_y = int(self.frame_height * line_position)
        
        print(f"Video properties:")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Duration: {self.total_frames/self.fps:.1f} seconds")
        print(f"  Counting line at Y={self.line_y} ({line_position*100}%)")
        
        # Load YOLO model
        print("\nLoading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')
        print("Model loaded successfully!")
        
        # Initialize tracking variables
        self.track_history = defaultdict(lambda: [])
        self.counted_ids = set()
        self.entry_count = 0
        self.exit_count = 0
        
        # Debug counters
        self.total_detections = 0
        self.frames_with_detection = 0
    
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
        
        # Debug: Show when close to line
        if abs(curr_y - self.line_y) < 50:
            print(f"  [NEAR LINE] ID {track_id}: prev_y={prev_y}, curr_y={curr_y}, line_y={self.line_y}")
        
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
    
    def process_video(self, output_path='output_debug.mp4', max_frames=None, display=False):
        """Process video and count footfall."""
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                             (self.frame_width, self.frame_height))
        
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")
        
        frame_count = 0
        start_time = time.time()
        
        print("\n" + "="*60)
        print("STARTING PROCESSING")
        print("="*60)
        print("Watching for people detections and line crossings...")
        print("="*60 + "\n")
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("\nEnd of video reached")
                    break
                
                frame_count += 1
                
                # Stop early for debugging
                if max_frames and frame_count > max_frames:
                    print(f"\n[DEBUG] Stopped at frame {frame_count} (max_frames={max_frames})")
                    break
                
                # Run YOLO detection with tracking
                results = self.model.track(frame, persist=True, classes=[0], 
                                          verbose=False, conf=0.3, iou=0.5)
                
                # Process detections
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    detections_this_frame = len(track_ids)
                    self.total_detections += detections_this_frame
                    self.frames_with_detection += 1
                    
                    # Log detections periodically
                    if frame_count % 30 == 0:
                        print(f"\n[Frame {frame_count}] Detected {detections_this_frame} people")
                    
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
                            print(f"\n🟢 ENTRY! ID: {track_id} at frame {frame_count} | Total entries: {self.entry_count}")
                        elif crossing == 'exit':
                            self.exit_count += 1
                            print(f"\n🔴 EXIT! ID: {track_id} at frame {frame_count} | Total exits: {self.exit_count}")
                        
                        # Draw bounding box
                        color = (0, 255, 0) if centroid[1] < self.line_y else (0, 165, 255)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw label
                        label = f"ID:{track_id} {conf:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Draw trajectory
                        points = self.track_history[track_id]
                        for i in range(1, len(points)):
                            cv2.line(frame, points[i-1], points[i], (230, 230, 230), 2)
                        
                        # Draw centroid
                        cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
                else:
                    if frame_count % 30 == 0 and frame_count > 0:
                        print(f"\n[Frame {frame_count}] No people detected")
                
                # Draw UI
                frame = self.draw_ui(frame)
                
                # Write frame
                out.write(frame)
                
                # Display frame (optional)
                if display:
                    cv2.imshow('Footfall Counter', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user")
                        break
                
                # Progress update
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    progress = (frame_count / self.total_frames * 100) if self.total_frames > 0 else 0
                    print(f"\n--- Progress: {progress:.1f}% ({frame_count}/{self.total_frames}) | FPS: {fps_current:.1f} ---")
                    print(f"    Entries: {self.entry_count} | Exits: {self.exit_count}")
                    print(f"    Active tracks: {len(self.track_history)}")
        
        finally:
            # Cleanup
            self.cap.release()
            out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Final report
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Total Frames Processed: {frame_count}")
        print(f"Processing Time: {total_time:.1f} seconds")
        print(f"Average FPS: {frame_count/total_time:.1f}")
        print(f"\nDetection Statistics:")
        print(f"  Frames with People: {self.frames_with_detection} ({self.frames_with_detection/max(frame_count,1)*100:.1f}%)")
        print(f"  Total Detections: {self.total_detections}")
        print(f"  Unique Tracks: {len(self.track_history)}")
        print(f"  Counted IDs: {len(self.counted_ids)}")
        print(f"\nFootfall Count:")
        print(f"  Entries: {self.entry_count}")
        print(f"  Exits: {self.exit_count}")
        print(f"  Net Flow: {self.entry_count - self.exit_count}")
        print(f"\nOutput saved to: {output_path}")
        print("="*60)
        
        # Diagnosis
        self.print_diagnosis(frame_count)
    
    def print_diagnosis(self, frame_count):
        """Print diagnostic information."""
        print("\n📊 DIAGNOSIS:")
        
        if frame_count == 0:
            print("❌ NO FRAMES PROCESSED!")
            print("   The video file could not be read.")
            print("   Possible reasons:")
            print("   1. Video file is corrupted")
            print("   2. Video codec not supported by OpenCV")
            print("   3. File path is incorrect")
            print("\n   Solutions:")
            print("   - Try converting video to MP4 with H.264 codec")
            print("   - Use a tool like ffmpeg: ffmpeg -i input.mp4 -vcodec libx264 output.mp4")
            print("   - Verify the file plays in a media player")
        
        elif self.frames_with_detection == 0:
            print("❌ NO PEOPLE DETECTED!")
            print("   Possible reasons:")
            print("   1. Video has no people in frame")
            print("   2. Camera angle makes people too small")
            print("   3. Video quality/lighting is poor")
            print("   4. People are occluded or partially visible")
            print("\n   Solutions:")
            print("   - Try lowering confidence threshold (currently 0.3)")
            print("   - Use a video with clear, visible people")
            print("   - Ensure good lighting and camera angle")
        
        elif self.entry_count + self.exit_count == 0:
            print("⚠️  PEOPLE DETECTED BUT NO LINE CROSSINGS!")
            print(f"   Line is at Y={self.line_y} ({self.line_position*100}% of height)")
            print("   Possible reasons:")
            print("   1. People never cross the counting line")
            print("   2. Line position doesn't match movement pattern")
            print("   3. People move parallel to the line")
            print("\n   Solutions:")
            print("   - Adjust LINE_POSITION in main() (try 0.3, 0.5, or 0.7)")
            print("   - Ensure people walk perpendicular to line")
            print("   - Check output video to see where line is positioned")
        
        else:
            print("✅ SYSTEM WORKING CORRECTLY!")
            print(f"   Successfully counted {self.entry_count + self.exit_count} line crossings")
            avg_detections_per_frame = self.total_detections / max(self.frames_with_detection, 1)
            print(f"   Average {avg_detections_per_frame:.1f} people per frame with detections")


def main():
    # ============================================================
    # CONFIGURATION - UPDATED WITH YOUR PATHS
    # ============================================================
    
    VIDEO_PATH = r"C:\Users\shiva\Downloads\footfall_count (2)\footfall_count\trial_vedio.mp4"
    OUTPUT_PATH = 'output_vedio.mp4'
    LINE_POSITION = 0.5  # 0.0 = top, 0.5 = middle, 1.0 = bottom
    MAX_FRAMES = None  # Set to number (e.g., 300) to process only first N frames
    DISPLAY = False  # Set to True to show video while processing (slower)
    
    # ============================================================
    
    print("="*60)
    print("FOOTFALL COUNTER - DEBUG VERSION")
    print("="*60)
    print(f"Video Path: {VIDEO_PATH}")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"Line Position: {LINE_POSITION*100}%")
    print(f"Max Frames: {MAX_FRAMES if MAX_FRAMES else 'All'}")
    print(f"Display: {DISPLAY}")
    print("="*60 + "\n")
    
    try:
        counter = FootfallCounterDebug(VIDEO_PATH, line_position=LINE_POSITION)
        counter.process_video(output_path=OUTPUT_PATH, max_frames=MAX_FRAMES, display=DISPLAY)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease check:")
        print("1. Video file path is correct")
        print("2. File exists at the specified location")
        print("3. File name includes the extension (.mp4, .avi, etc.)")
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease check:")
        print("1. Video file is not corrupted")
        print("2. Video codec is supported")
        print("3. Try converting to MP4 with H.264 codec")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()