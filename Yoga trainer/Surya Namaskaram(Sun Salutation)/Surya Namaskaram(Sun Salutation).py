import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

text_speech = pyttsx3.init()
voices = text_speech.getProperty('voices')
text_speech.setProperty('voice',voices[1].id)

answer1_1 = "Step 1, Prayer pose"
answer2_1 = "Step 1 completed"

text_speech.say(answer1_1)
text_speech.runAndWait()

answer1_2 = "Step 2, Raised Arms Pose"
answer2_2 = "Step 2 completed"

answer1_3 = "Step 3, Hand-To-Foot Pose"
answer2_3 = "Step 3 completed"

answer1_4 = "Step 4, Equestrian pose"
answer2_4 = "Step 4 completed"

answer1_5 = "Step 5, Points Pose"
answer2_5 = "Step 5 completed"

answer1_6 = "Step 6, Eight-Limbed Salute"
answer2_6 = "Step 6 completed"

answer1_7 = "Step 7, Cobra Pose"
answer2_7 = "Step 7 completed"

answer1_8 = "Step 8, Mountain Pose"
answer2_8 = "Step 8 completed"

answer1_9 = "Step 9, Equestrian pose"
answer2_9 = "Step 9 completed"

answer1_10 = "Step 10, Hand-To-Foot Pose"
answer2_10 = "Step 10 completed"

answer1_11 = "Step 11, Raised Arms Pose"
answer2_11 = "Step 11 completed"

answer1_12 = "Step 12, Prayer pose"
answer2_12 = "Step 12 completed"

step = 1

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

# Function to check if arms are parallel to camera
def are_arms_parallel_1(left_shoulder, left_elbow, left_wrist):
    # Calculate angles
    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    # Check if both angles are close to 180 degrees
    return abs(angle_left) < 80 and abs(angle_left) > 20

# Function to check if hips, shoulder, and elbow form 90 degrees angle
def hip_shoulder_elbow_1(left_hip, left_shoulder, left_elbow):
    # Calculate angles
    angle_left = calculate_angle(left_hip, left_shoulder, left_elbow)
    
    # Check if both angles are close to 5 degrees
    return abs(angle_left) < 30

def are_arms_parallel_2(left_shoulder, left_elbow, left_wrist):
    # Calculate angles
    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    # Check if both angles are close to 180 degrees
    return abs(angle_left) > 145

# Function to check if hips, shoulder, and elbow form 90 degrees angle
def body_posture_upper_2(left_hip, left_shoulder, left_elbow):
    # Calculate angles
    angle_left = calculate_angle(left_hip, left_shoulder, left_elbow)
    
    # Check if both angles are close to 90 degrees
    return abs(angle_left) < 180 and abs(angle_left) > 140 

def body_posture_lower_2(left_knee, left_hip, left_shoulder):
    # Calculate angles
    angle_left = calculate_angle(left_knee, left_hip, left_shoulder)
    
    # Check if both angles are close to 90 degrees
    return abs(angle_left) < 175 and abs(angle_left) > 140

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


# Function to check if arms are parallel to camera
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

# Function to check if arms are parallel to camera
def arm_90_5(left_hip, left_shoulder, left_elbow):
    # Calculate angles
    angle_left = calculate_angle(left_hip, left_shoulder, left_elbow)
    
    # Check if both angles are close to 180 degrees
    return abs(angle_left) > 55 and abs(angle_left) < 95

# Function to check if hips, shoulder, and elbow form 90 degrees angle
def leg_straight_5(left_ankle, left_knee, left_hip):
    # Calculate angles
    angle_left = calculate_angle(left_ankle, left_knee, left_hip)
    
    # Check if both angles are close to 90 degrees
    return abs(angle_left) > 165

def body_straight_5(left_knee, left_hip, left_shoulder):
    # Calculate angles
    angle_left = calculate_angle(left_knee, left_hip, left_shoulder)
    
    # Check if both angles are close to 90 degrees
    return abs(angle_left) > 165

# Function to check if arms are parallel to camera
def arm_angle_6(left_shoulder, left_elbow, left_wrist):
    # Calculate angles
    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    # Check if both angles are close to 180 degrees
    return abs(angle_left) > 15 and abs(angle_left) < 85

# Function to check if hips, shoulder, and elbow form 90 degrees angle
def hip_angle_6(left_knee, left_hip, left_shoulder):
    # Calculate angles
    angle_left = calculate_angle(left_knee, left_hip, left_shoulder)
    
    # Check if both angles are close to 90 degrees
    return abs(angle_left) < 175 and abs(angle_left) > 90

def body_straight_6(left_ankle, left_knee, left_shoulder):
    # Calculate angles
    angle_left = calculate_angle(left_ankle, left_knee, left_shoulder)
    
    # Check if both angles are close to 90 degrees
    return abs(angle_left) > 155

# Function to check if arms are parallel to camera
def arm_angle_7(left_shoulder, left_elbow, left_wrist):
    # Calculate angles
    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    # Check if both angles are close to 180 degrees
    return abs(angle_left) > 100

# Function to check if hips, shoulder, and elbow form 90 degrees angle
def hip_angle_7(left_knee, left_hip, left_shoulder):
    # Calculate angles
    angle_left = calculate_angle(left_knee, left_hip, left_shoulder)
    
    # Check if both angles are close to 90 degrees
    return abs(angle_left) < 170 and abs(angle_left) > 90

def leg_straight_7(left_ankle, left_knee, left_hip):
    # Calculate angles
    angle_left = calculate_angle(left_ankle, left_knee, left_hip)
    
    # Check if both angles are close to 90 degrees
    return abs(angle_left) > 160
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

# Video capture
cap = cv2.VideoCapture(0)

# Arms parallel detection variables
arms_parallel_1 = False
angles_90_1 = False
# Arms parallel detection variables
arms_parallel_2 = False
angles_90_2 = False
check_body_lower_2 = False
# Arms parallel detection variables
arms_parallel_3 = False
check_leg_straight_3 = False
check_body_lower_3 = False

check_front_leg_4 = False
check_back_leg_4 = False
check_leg_split_4 = False

# Arms parallel detection variables
check_arm_90_5 = False
check_leg_straight_5 = False
check_body_straight_5 = False

# Arms parallel detection variables
check_arm_angle_6 = False
check_hip_angle_6 = False
check_body_straight_6 = False

# Arms parallel detection variables
check_arm_angle_7 = False
check_hip_angle_7 = False
check_leg_straight_7 = False

check_arm_angle_8 = False
check_hip_angle_8 = False
check_leg_straight_8 = False


# Timer variables
start_time = None
timer_running = False

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while step == 1 and cap.isOpened():
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
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Check if arms are parallel to camera
            arms_parallel_1 = are_arms_parallel_1(left_shoulder, left_elbow, left_wrist)
            
            # Check if hips, shoulder, and elbow form 90 degrees angle
            angles_90_1 = hip_shoulder_elbow_1(left_hip, left_shoulder, left_elbow)
            
            # Start timer if pose is correct and timer is not already running
            if arms_parallel_1 and angles_90_1 and not timer_running:
                start_time = time.time()
                timer_running = True
            # Restart timer if pose becomes incorrect while timer is running
            elif not (arms_parallel_1 and angles_90_1) and timer_running:
                start_time = time.time()
            # Check if timer has reached 4 seconds
            elif timer_running and time.time() - start_time >= 4:
                text_speech.say(answer2_1)
                text_speech.runAndWait()
                text_speech.say(answer1_2)
                text_speech.runAndWait()
                print("Successfully Completed")
                timer_running = False
                step = 2
                       
        except:
            pass
        
        # Render arms parallel status
        cv2.putText(image, 'Hand pose: {}'.format(arms_parallel_1), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if arms_parallel_1 else (0, 0, 255), 2)
        
        # Render angles_90 status
        cv2.putText(image, 'Body posture: {}'.format(angles_90_1), (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if angles_90_1 else (0, 0, 255), 2)
        
        # Render timer status
        if timer_running:
            elapsed_time = 4 - (time.time() - start_time)
            cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(image, 'Timer Stopped', (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_1 and angles_90_1 else (255, 0, 0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_1 and angles_90_1 else (255, 0, 0), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    while step == 2 and cap.isOpened():
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

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Check if arms are parallel to camera
            arms_parallel_2 = are_arms_parallel_2(left_shoulder, left_elbow, left_wrist)
            
            # Check if hips, shoulder, and elbow form 90 degrees angle
            angles_90_2 = body_posture_upper_2(left_hip, left_shoulder, left_elbow)
            check_body_lower_2 = body_posture_lower_2(left_knee, left_hip, left_shoulder)

            # Start timer if pose is correct and timer is not already running
            if arms_parallel_2 and angles_90_2 and check_body_lower_2 and not timer_running:
                start_time = time.time()
                timer_running = True
            # Restart timer if pose becomes incorrect while timer is running
            elif not (arms_parallel_2 and angles_90_2 and check_body_lower_2) and timer_running:
                start_time = time.time()
            # Check if timer has reached 4 seconds
            elif timer_running and time.time() - start_time >= 4:
                text_speech.say(answer2_2)
                text_speech.runAndWait()
                text_speech.say(answer1_3)
                text_speech.runAndWait()
                print("Successfully Completed")
                timer_running = False
                step = 3
                       
        except:
            pass
        
        # Render arms parallel status
        cv2.putText(image, 'Arms straight: {}'.format(arms_parallel_2), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if arms_parallel_2 else (0, 0, 255), 2)
        
        # Render angles_90 status
        cv2.putText(image, 'Angles 90: {}'.format(angles_90_2), (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if angles_90_2 else (0, 0, 255), 2)
        
        cv2.putText(image, 'Lower body posture: {}'.format(check_body_lower_2), (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_body_lower_2 else (0, 0, 255), 2)
        

        # Render timer status
        if timer_running:
            elapsed_time = 4 - (time.time() - start_time)
            cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(image, 'Timer Stopped', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_2 and angles_90_2 and check_body_lower_2 else (255, 0, 0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_2 and angles_90_2 and check_body_lower_2 else (255, 0, 0), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    while step == 3 and cap.isOpened():
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

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Check if arms are parallel to camera
            arms_parallel_3 = are_arms_straight_3(left_shoulder, left_elbow, left_wrist)
            
            # Check if hips, shoulder, and elbow form 90 degrees angle
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
                text_speech.say(answer1_4)
                text_speech.runAndWait()
                print("Successfully Completed")
                timer_running = False
                step = 4
                       
        except:
            pass
        
        # Render arms parallel status
        cv2.putText(image, 'Arms straight: {}'.format(arms_parallel_3), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if arms_parallel_3 else (0, 0, 255), 2)
    
    # Render check_leg_straight status
        cv2.putText(image, 'Straightened knee: {}'.format(check_leg_straight_3), (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_leg_straight_3 else (0, 0, 255), 2)
    
        cv2.putText(image, 'Hip bent: {}'.format(check_body_lower_3), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_body_lower_3 else (0, 0, 255), 2)
        
        #Check timer status
        if timer_running:
            elapsed_time = 4 - (time.time() - start_time)
            cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            cv2.putText(image, 'Timer Stopped', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_3 and check_leg_straight_3 and check_body_lower_3 else (255, 0, 0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_3 and check_leg_straight_3 and check_body_lower_3 else (255, 0, 0), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    while step == 4 and cap.isOpened():
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

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Check if arms are parallel to camera
            check_front_leg_4 = front_leg_4(left_ankle, left_knee, left_hip)
            
            # Check if hips, shoulder, and elbow form 90 degrees angle
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
                text_speech.say(answer1_5)
                text_speech.runAndWait()
                print("Successfully Completed")
                timer_running = False
                step = 5
                    
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
        

        # Render timer status
        if timer_running:
            elapsed_time = 4 - (time.time() - start_time)
            cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(image, 'Timer Stopped', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_back_leg_4 and check_front_leg_4 and check_leg_split_4 else (255, 0, 0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_back_leg_4 and check_front_leg_4 and check_leg_split_4 else (255, 0, 0), thickness=2, circle_radius=2) 
                                )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(4) & 0xFF == ord('q'):
            break

    while step ==5 and cap.isOpened():
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

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Check if arms are parallel to camera
            check_arm_90_5 = arm_90_5(left_hip, left_shoulder, left_elbow)
            
            # Check if hips, shoulder, and elbow form 90 degrees angle
            check_leg_straight_5 = leg_straight_5(left_ankle, left_knee, left_hip)
            check_body_straight_5 = body_straight_5(left_knee, left_hip, left_shoulder)

            # Start timer if pose is correct and timer is not already running
            if check_arm_90_5 and check_leg_straight_5 and check_body_straight_5 and not timer_running:
                start_time = time.time()
                timer_running = True
            # Restart timer if pose becomes incorrect while timer is running
            elif not (check_arm_90_5 and check_leg_straight_5 and check_body_straight_5) and timer_running:
                start_time = time.time()
            # Check if timer has reached 4 seconds
            elif timer_running and time.time() - start_time >= 4:
                text_speech.say(answer2_5)
                text_speech.runAndWait()
                text_speech.say(answer1_6)
                text_speech.runAndWait()
                print("Successfully Completed")
                timer_running = False
                step = 6
                       
        except:
            pass
        
        # Render arms parallel status
        cv2.putText(image, 'Arm posture: {}'.format(check_arm_90_5), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_arm_90_5 else (0, 0, 255), 2)
    
    # Render check_leg_straight status
        cv2.putText(image, 'Straightened knee: {}'.format(check_leg_straight_5), (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_leg_straight_5 else (0, 0, 255), 2)
    
        cv2.putText(image, 'Hip bent: {}'.format(check_body_straight_5), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_body_straight_5 else (0, 0, 255), 2)
        
        #Check timer status
        if timer_running:
            elapsed_time = 4 - (time.time() - start_time)
            cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            cv2.putText(image, 'Timer Stopped', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_arm_90_5 and check_leg_straight_5 and check_body_straight_5 else (255, 0, 0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_arm_90_5 and check_leg_straight_5 and check_body_straight_5 else (255, 0, 0), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    while step == 6 and cap.isOpened():
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

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Check if arms are parallel to camera
            check_arm_angle_6 = arm_angle_6(left_shoulder, left_elbow, left_wrist)
            
            # Check if hips, shoulder, and elbow form 90 degrees angle
            check_hip_angle_6 = hip_angle_6(left_knee, left_hip, left_shoulder)
            check_body_straight_6 = body_straight_6(left_ankle, left_knee, left_shoulder)

            # Start timer if pose is correct and timer is not already running
            if check_arm_angle_6 and check_hip_angle_6 and check_body_straight_6 and not timer_running:
                start_time = time.time()
                timer_running = True
            # Restart timer if pose becomes incorrect while timer is running
            elif not (check_arm_angle_6 and check_hip_angle_6 and check_body_straight_6) and timer_running:
                start_time = time.time()
            # Check if timer has reached 4 seconds
            elif timer_running and time.time() - start_time >= 4:
                text_speech.say(answer2_6)
                text_speech.runAndWait()
                text_speech.say(answer1_7)
                text_speech.runAndWait()
                print("Successfully Completed")
                timer_running = False
                step=7
                       
        except:
            pass
        
        # Render arms parallel status
        cv2.putText(image, 'Arm posture: {}'.format(check_arm_angle_6), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_arm_angle_6 else (0, 0, 255), 2)
    
    # Render check_leg_straight status
        cv2.putText(image, 'Hip bent: {}'.format(check_hip_angle_6), (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_hip_angle_6 else (0, 0, 255), 2)
    
        cv2.putText(image, 'Ground: {}'.format(check_body_straight_6), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_body_straight_6 else (0, 0, 255), 2)
        
        #Check timer status
        if timer_running:
            elapsed_time = 4 - (time.time() - start_time)
            cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            cv2.putText(image, 'Timer Stopped', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_arm_angle_6 and check_hip_angle_6 and check_body_straight_6 else (255, 0, 0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_arm_angle_6 and check_hip_angle_6 and check_body_straight_6 else (255, 0, 0), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    while step == 7 and cap.isOpened():
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

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Check if arms are parallel to camera
            check_arm_angle_7 = arm_angle_7(left_shoulder, left_elbow, left_wrist)
            
            # Check if hips, shoulder, and elbow form 90 degrees angle
            check_hip_angle_7 = hip_angle_7(left_knee, left_hip, left_shoulder)
            check_leg_straight_7 = leg_straight_7(left_ankle, left_knee, left_hip)

            # Start timer if pose is correct and timer is not already running
            if check_arm_angle_7 and check_hip_angle_7 and check_leg_straight_7 and not timer_running:
                start_time = time.time()
                timer_running = True
            # Restart timer if pose becomes incorrect while timer is running
            elif not (check_arm_angle_7 and check_hip_angle_7 and check_leg_straight_7) and timer_running:
                start_time = time.time()
            # Check if timer has reached 4 seconds
            elif timer_running and time.time() - start_time >= 4:
                text_speech.say(answer2_7)
                text_speech.runAndWait()
                text_speech.say(answer1_8)
                text_speech.runAndWait()
                print("Successfully Completed")
                timer_running = False
                step = 8
                       
        except:
            pass
        
        # Render arms parallel status
        cv2.putText(image, 'Arm posture: {}'.format(check_arm_angle_7), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_arm_angle_7 else (0, 0, 255), 2)
    
    # Render check_leg_straight status
        cv2.putText(image, 'Hip bent: {}'.format(check_hip_angle_7), (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_hip_angle_7 else (0, 0, 255), 2)
    
        cv2.putText(image, 'Leg straight: {}'.format(check_leg_straight_7), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_leg_straight_7 else (0, 0, 255), 2)
        
        #Check timer status
        if timer_running:
            elapsed_time = 4 - (time.time() - start_time)
            cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            cv2.putText(image, 'Timer Stopped', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_arm_angle_7 and check_hip_angle_7 and check_leg_straight_7 else (255, 0, 0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_arm_angle_7 and check_hip_angle_7 and check_leg_straight_7 else (255, 0, 0), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    while step == 8 and cap.isOpened():
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

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Check if arms are parallel to camera
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
            # Check if timer has reached 4 seconds
            elif timer_running and time.time() - start_time >= 4:
                text_speech.say(answer2_8)
                text_speech.runAndWait()
                text_speech.say(answer1_9)
                text_speech.runAndWait()
                print("Successfully Completed")
                timer_running = False
                step=9
                       
        except:
            pass
        
        # Render arms parallel status
        cv2.putText(image, 'Arm posture: {}'.format(check_arm_angle_8), (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_arm_angle_8 else (0, 0, 255), 2)
        
        
        cv2.putText(image, 'Hip bent: {}'.format(check_hip_angle_8), (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_hip_angle_8 else (0, 0, 255), 2)
        
        cv2.putText(image, 'Leg straight: {}'.format(check_leg_straight_8), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_leg_straight_8 else (0, 0, 255), 2)
        
        #Check timer status
        if timer_running:
            elapsed_time = 4 - (time.time() - start_time)
            cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            cv2.putText(image, 'Timer Stopped', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_arm_angle_6 and check_hip_angle_6 and check_body_straight_6 else (255, 0, 0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_arm_angle_6 and check_hip_angle_6 and check_body_straight_6 else (255, 0, 0), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    while step == 9 and cap.isOpened():
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

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Check if arms are parallel to camera
            check_front_leg_4 = front_leg_4(left_ankle, left_knee, left_hip)
            
            # Check if hips, shoulder, and elbow form 90 degrees angle
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
                text_speech.say(answer2_9)
                text_speech.runAndWait()
                text_speech.say(answer1_10)
                text_speech.runAndWait()
                print("Successfully Completed")
                timer_running = False
                step = 10
                    
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
        

        # Render timer status
        if timer_running:
            elapsed_time = 4 - (time.time() - start_time)
            cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(image, 'Timer Stopped', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_back_leg_4 and check_front_leg_4 and check_leg_split_4 else (255, 0, 0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if check_back_leg_4 and check_front_leg_4 and check_leg_split_4 else (255, 0, 0), thickness=2, circle_radius=2) 
                                )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(4) & 0xFF == ord('q'):
            break   
    while step == 10 and cap.isOpened():
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

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Check if arms are parallel to camera
            arms_parallel_3 = are_arms_straight_3(left_shoulder, left_elbow, left_wrist)
            
            # Check if hips, shoulder, and elbow form 90 degrees angle
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
                text_speech.say(answer2_10)
                text_speech.runAndWait()
                text_speech.say(answer1_11)
                text_speech.runAndWait()
                print("Successfully Completed")
                timer_running = False
                step = 11
                       
        except:
            pass
        
        # Render arms parallel status
        cv2.putText(image, 'Arms straight: {}'.format(arms_parallel_3), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if arms_parallel_3 else (0, 0, 255), 2)
    
    # Render check_leg_straight status
        cv2.putText(image, 'Straightened knee: {}'.format(check_leg_straight_3), (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_leg_straight_3 else (0, 0, 255), 2)
    
        cv2.putText(image, 'Hip bent: {}'.format(check_body_lower_3), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_body_lower_3 else (0, 0, 255), 2)
        
        #Check timer status
        if timer_running:
            elapsed_time = 4 - (time.time() - start_time)
            cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            cv2.putText(image, 'Timer Stopped', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_3 and check_leg_straight_3 and check_body_lower_3 else (255, 0, 0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_3 and check_leg_straight_3 and check_body_lower_3 else (255, 0, 0), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break    
        
    
    while step == 11 and cap.isOpened():
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

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Check if arms are parallel to camera
            arms_parallel_2 = are_arms_parallel_2(left_shoulder, left_elbow, left_wrist)
            
            # Check if hips, shoulder, and elbow form 90 degrees angle
            angles_90_2 = body_posture_upper_2(left_hip, left_shoulder, left_elbow)
            check_body_lower_2 = body_posture_lower_2(left_knee, left_hip, left_shoulder)

            # Start timer if pose is correct and timer is not already running
            if arms_parallel_2 and angles_90_2 and check_body_lower_2 and not timer_running:
                start_time = time.time()
                timer_running = True
            # Restart timer if pose becomes incorrect while timer is running
            elif not (arms_parallel_2 and angles_90_2 and check_body_lower_2) and timer_running:
                start_time = time.time()
            # Check if timer has reached 4 seconds
            elif timer_running and time.time() - start_time >= 4:
                text_speech.say(answer2_11)
                text_speech.runAndWait()
                text_speech.say(answer1_12)
                text_speech.runAndWait()
                print("Successfully Completed")
                timer_running = False
                step = 12
                       
        except:
            pass
        
        # Render arms parallel status
        cv2.putText(image, 'Arms straight: {}'.format(arms_parallel_2), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if arms_parallel_2 else (0, 0, 255), 2)
        
        # Render angles_90 status
        cv2.putText(image, 'Angles 90: {}'.format(angles_90_2), (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if angles_90_2 else (0, 0, 255), 2)
        
        cv2.putText(image, 'Lower body posture: {}'.format(check_body_lower_2), (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if check_body_lower_2 else (0, 0, 255), 2)
        

        # Render timer status
        if timer_running:
            elapsed_time = 4 - (time.time() - start_time)
            cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(image, 'Timer Stopped', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_2 and angles_90_2 and check_body_lower_2 else (255, 0, 0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_2 and angles_90_2 and check_body_lower_2 else (255, 0, 0), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break  

    while step == 12 and cap.isOpened():
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
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            
            # Check if arms are parallel to camera
            arms_parallel_1 = are_arms_parallel_1(left_shoulder, left_elbow, left_wrist)
            
            # Check if hips, shoulder, and elbow form 90 degrees angle
            angles_90_1 = hip_shoulder_elbow_1(left_hip, left_shoulder, left_elbow)
            
            # Start timer if pose is correct and timer is not already running
            if arms_parallel_1 and angles_90_1 and not timer_running:
                start_time = time.time()
                timer_running = True
            # Restart timer if pose becomes incorrect while timer is running
            elif not (arms_parallel_1 and angles_90_1) and timer_running:
                start_time = time.time()
            # Check if timer has reached 4 seconds
            elif timer_running and time.time() - start_time >= 4:
                text_speech.say(answer2_12)
                text_speech.runAndWait()
                print("Successfully Completed")
                timer_running = False
               
                       
        except:
            pass
        
        # Render arms parallel status
        cv2.putText(image, 'Hand pose: {}'.format(arms_parallel_1), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if arms_parallel_1 else (0, 0, 255), 2)
        
        # Render angles_90 status
        cv2.putText(image, 'Body posture: {}'.format(angles_90_1), (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if angles_90_1 else (0, 0, 255), 2)
        
        # Render timer status
        if timer_running:
            elapsed_time = 4 - (time.time() - start_time)
            cv2.putText(image, f'Timer: {elapsed_time:.1f}s', (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(image, 'Timer Stopped', (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_1 and angles_90_1 else (255, 0, 0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(0, 255, 0) if arms_parallel_1 and angles_90_1 else (255, 0, 0), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break    

cap.release()
cv2.destroyAllWindows()        