from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# Initialize Text to Speech engine
text_speech = pyttsx3.init()
voices = text_speech.getProperty('voices')
text_speech.setProperty('voice', voices[1].id)

answer1_3 = "Hand-To-Foot Pose"
answer2_3 = "completed"

text_speech.say(answer1_3)
text_speech.runAndWait()

# Function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def are_arms_straight_3(left_shoulder, left_elbow, left_wrist):
    # Calculate angles
    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)

    return abs(angle_left) > 160


def leg_straight_3(left_ankle, left_knee, left_hip):
    # Calculate angles
    angle_left = calculate_angle(left_ankle, left_knee, left_hip)

    
    return abs(angle_left) > 160

def body_posture_lower_3(left_knee, left_hip, left_shoulder):
    # Calculate angles
    angle_left = calculate_angle(left_knee, left_hip, left_shoulder)

   
    return abs(angle_left) < 90

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    cap = cv2.VideoCapture(0)

    # Arms parallel detection variables
    arms_parallel_3 = False
    check_leg_straight_3 = False
    check_body_lower_3 = False

    # Timer variables
    start_time = None
    timer_running = False

    # Setup mediapipe instance
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
                left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y]

               
                arms_parallel_3 = are_arms_straight_3(left_shoulder, left_elbow, left_wrist)

                
                check_leg_straight_3 = leg_straight_3(left_ankle, left_knee, left_hip)
                check_body_lower_3 = body_posture_lower_3(left_knee, left_hip, left_shoulder)

                # Start timer if pose is correct and timer is not already running
                if arms_parallel_3 and check_leg_straight_3 and check_body_lower_3 and not timer_running:
                    start_time = time.time()
                    timer_running = True
                # Restart timer if pose becomes incorrect while timer is running
                elif not (arms_parallel_3 and check_leg_straight_3 and check_body_lower_3) and timer_running:
                    start_time = time.time()
                # Check if timer has reached 4 seconds
                elif timer_running and time.time() - start_time >= 4:
                    text_speech.say(answer2_3)
                    text_speech.runAndWait()
                    print("Successfully Completed")
                    timer_running = False

            except:
                pass

            # Render arms parallel status
            cv2.putText(image, 'Arms straight: {}'.format(arms_parallel_3), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if arms_parallel_3 else (0, 0, 255), 2)

            # Render check_leg_straight status
            cv2.putText(image, 'Straightened knee: {}'.format(check_leg_straight_3), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if check_leg_straight_3 else (0, 0, 255), 2)

            cv2.putText(image, 'Hip bent: {}'.format(check_body_lower_3), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if check_body_lower_3 else (0, 0, 255), 2)

            cv2.putText(image, 'Hand to Foot Pose', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if check_body_lower_3 and arms_parallel_3 and check_leg_straight_3  else (0, 0, 255), 2)

            # Check timer status
            if timer_running:
                elapsed_time = 4 - (time.time() - start_time)
                cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(image, 'Timer Stopped', (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_3 and check_leg_straight_3 and check_body_lower_3 else (255, 0, 0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_3 and check_leg_straight_3 and check_body_lower_3 else (255, 0, 0), thickness=2, circle_radius=2) 
                                    )               
        
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            # Convert image to jpeg
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

