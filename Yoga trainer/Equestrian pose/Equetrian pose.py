from flask import Flask, render_template, Response
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
answer1_4 = "Equetrian pose"
answer2_4 = "completed"
text_speech.say(answer1_4)
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


def back_leg_4(right_ankle, right_knee, right_hip):
    # Calculate angles
    angle_right = calculate_angle(right_ankle, right_knee, right_hip)
    
    
    return abs(angle_right) > 20 and abs(angle_right) <100


def front_leg_4(left_ankle, left_knee, left_hip):
    # Calculate angles
    angle_left = calculate_angle(left_ankle, left_knee, left_hip)
    

    return abs(angle_left) > 100

def leg_split_4(left_knee, left_hip, right_knee):
    # Calculate angles
    angle_left = calculate_angle(left_knee, left_hip, right_knee)
    

    return abs(angle_left) > 120

@app.route('/')
def index():
    return render_template('index.html')

# Video capture
def gen():
    cap = cv2.VideoCapture(0)

    #  detection variables
    check_front_leg_4 = False
    check_back_leg_4 = False
    check_leg_split_4 = False

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
        
            # Recolor     back to BGR
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
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                
                check_front_leg_4 = front_leg_4(left_ankle, left_knee, left_hip)
                
                
                check_back_leg_4 = back_leg_4(right_ankle, right_knee, right_hip)
                check_leg_split_4 = leg_split_4(left_knee, left_hip, right_knee)

                # Start timer if pose is correct and timer is not already running
                if check_front_leg_4 and check_back_leg_4 and check_leg_split_4 and not timer_running:
                    start_time = time.time()
                    timer_running = True
                # Restart timer if pose becomes incorrect while timer is running
                elif not (check_front_leg_4 and check_back_leg_4 and check_leg_split_4 and timer_running):
                    start_time = time.time()
                # Check if timer has reached 4 seconds
                elif timer_running and time.time() - start_time >= 4:
                    text_speech.say(answer2_4)
                    text_speech.runAndWait()
                    print("Successfully Completed")
                    timer_running = False
                        
            except:
                pass
            
            # Render arms parallel status
            cv2.putText(image, 'Front leg bent: {}'.format(check_front_leg_4), (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_front_leg_4 else (0, 0, 255), 2)
            
            # Render check_leg_straight status
            cv2.putText(image, 'Straightened back leg: {}'.format(check_back_leg_4), (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_back_leg_4 else (0, 0, 255), 2)
            
            cv2.putText(image, 'Leg split: {}'.format(check_leg_split_4), (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_leg_split_4 else (0, 0, 255), 2)
            
            cv2.putText(image, 'Equestrian pose', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if check_leg_split_4 and check_front_leg_4 and check_back_leg_4 else (0, 0, 255), 2)
            # Render timer status
            if timer_running:
                elapsed_time = 4 - (time.time() - start_time)
                cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(image, 'Timer Stopped', (10, 160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0, 255, 0) if check_back_leg_4 and check_front_leg_4 and check_leg_split_4 else (255, 0, 0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0, 255, 0) if check_back_leg_4 and check_front_leg_4 and check_leg_split_4 else (255, 0, 0), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(4) & 0xFF == ord('q'):
                break

    cap.release()
@app.route('/video_feed')
def video_feed():
   return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)