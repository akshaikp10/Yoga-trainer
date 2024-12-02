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
answer1_2 = "Mountain Pose"
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

def arm_angle_8(left_shoulder, left_elbow, left_wrist):
    # Calculate angles
    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    return abs(angle_left) > 160


def hip_angle_8(left_knee, left_hip, left_shoulder):
    # Calculate angles
    angle_left = calculate_angle(left_knee, left_hip, left_shoulder)
   
    return abs(angle_left) > 45 and abs(angle_left) < 100

def leg_straight_8(left_ankle, left_knee, left_hip):
    # Calculate angles
    angle_left = calculate_angle(left_ankle, left_knee, left_hip)
    
    return abs(angle_left) > 160




@app.route('/')
def index():
    return render_template('index.html')

def gen():
    cap = cv2.VideoCapture(0)

    #  detection variables
    check_arm_angle_8 = False
    check_hip_angle_8 = False
    check_leg_straight_8 = False
   


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
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Not using
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                    
                    
                check_arm_angle_8 = arm_angle_8(left_shoulder, left_elbow, left_wrist)
                
                
                check_hip_angle_8 = hip_angle_8(left_knee, left_hip, left_shoulder)
                check_leg_straight_8 = leg_straight_8(left_ankle, left_knee, left_hip)

                # Start timer if pose is correct and timer is not already running
                if check_arm_angle_8 and check_hip_angle_8 and check_leg_straight_8 and not timer_running:
                    start_time = time.time()
                    timer_running = True
                # Restart timer if pose becomes incorrect while timer is running
                elif not (check_arm_angle_8 and check_hip_angle_8 and 8) and timer_running:
                    start_time = time.time()
                elif timer_running and time.time() - start_time >= 4:
                    text_speech.say(answer2_2)
                    text_speech.runAndWait()
                    print("Successfully Completed")
                    timer_running = False
                        
            except:
                pass
            
                # text on image
            cv2.putText(image, 'Arm posture: {}'.format(check_arm_angle_8), (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_arm_angle_8 else (0, 0, 255), 2)
        
        
            cv2.putText(image, 'Hip bent: {}'.format(check_hip_angle_8), (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_hip_angle_8 else (0, 0, 255), 2)
        
            cv2.putText(image, 'Leg straight: {}'.format(check_leg_straight_8), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_leg_straight_8 else (0, 0, 255), 2)
            cv2.putText(image, 'Mountain Pose', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_arm_angle_8 and check_hip_angle_8 and check_leg_straight_8 else (0, 0, 255), 2)
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
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_arm_angle_8 and check_hip_angle_8 and check_leg_straight_8 else (255, 0, 0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_arm_angle_8 and check_hip_angle_8 and check_leg_straight_8 else (255, 0, 0), thickness=2, circle_radius=2) 
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
    