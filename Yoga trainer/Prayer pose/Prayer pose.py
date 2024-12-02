from flask import Flask, render_template, Response#  r cp
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3


app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

text_speech = pyttsx3.init()
voices = text_speech.getProperty('voices')
text_speech.setProperty('voice',voices[1].id)
answer1_2 = "Prayer pose"
answer2_2 = "completed"

text_speech.say(answer1_2)
text_speech.runAndWait()



# Function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Function to check if arms are >145 angle
def are_arms_parallel_2(left_shoulder, left_elbow, left_wrist):
    # Calculate angles
    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    
    return abs(angle_left) < 80 and abs(angle_left) >20


def body_posture_upper_2(left_hip, left_shoulder, left_elbow):
    # Calculate angles
    angle_left = calculate_angle(left_hip, left_shoulder, left_elbow)
    
    return abs(angle_left) < 30




@app.route('/')
def index():
    return render_template('index.html')

def gen():
    cap = cv2.VideoCapture(0)

    #  detection variables
    arms_parallel_2 = False
    angles_90_2 = False
   


    # Timer variables
    start_time = None
    timer_running = False

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                # Not using
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                
                arms_parallel_2 = are_arms_parallel_2(left_shoulder, left_elbow, left_wrist)
                angles_90_2 = body_posture_upper_2(left_hip, left_shoulder, left_elbow)
                

                # Start timer if pose is correct and timer is not already running
                if arms_parallel_2 and angles_90_2 and not timer_running:
                    start_time = time.time()
                    timer_running = True
                # Restart timer if pose becomes incorrect while timer is running
                elif not (arms_parallel_2 and angles_90_2 ) and timer_running:
                    start_time = time.time()
                # Check if timer has reached 4 seconds
                elif timer_running and time.time() - start_time >= 4:
                    text_speech.say(answer2_2)
                    text_speech.runAndWait()
                    print("Successfully Completed")
                    timer_running = False
                        
            except:
                pass
            
            # text on image
            cv2.putText(image, 'Arms straight: {}'.format(arms_parallel_2), (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if arms_parallel_2 else (0, 0, 255), 2)
            
            
            cv2.putText(image, 'Body posture:  {}'.format(angles_90_2), (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if angles_90_2 else (0, 0, 255), 2)
            

            
            cv2.putText(image, 'Prayer pose', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if arms_parallel_2 and angles_90_2  else (0, 0, 255), 2)


            # Render timer status
            if timer_running:
                elapsed_time = 4 - (time.time() - start_time)
                cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(image, 'Timer Stopped', (10, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_2 and angles_90_2  else (255, 0, 0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_2 and angles_90_2 else (255, 0, 0), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)


             # Convert image to jpeg
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()


            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    

@app.route('/video_feed')
def video_feed():
   return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
    