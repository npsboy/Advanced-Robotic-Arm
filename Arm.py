import threading
import time
import sys
import itertools

done = False

def spinner():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rLoading ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rStartup complete!\n')

Loading_animation = threading.Thread(target=spinner, daemon=True)
Loading_animation.start()

import cv2
import numpy as np
import math
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import os
import keyboard
import json
import serial
import soundfile as sf
import sounddevice as sd

import webrtcvad
from openai import OpenAI
from dotenv import load_dotenv
import pvporcupine
from pvrecorder import PvRecorder

import io


try:
    #capture = cv2.VideoCapture("http://192.168.68.103:8080/video")
    capture = cv2.VideoCapture(1)
except Exception as e:
    print(f"Error: Webcam unavailable.")
    sys.exit(1)


processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")


serial_port = 'COM6'
try:
    pico_serial = serial.Serial(serial_port, 115200, timeout=1)
except serial.SerialException:
    print(f"Error: Could not open serial port {serial_port}. Please check the connection.")
    sys.exit(1)
time.sleep(2)

done = True
Loading_animation.join()
print("\n")


base_servo_angle = 50
shoulder_servo_angle = 90
elbow_servo_angle = 90
claw_servo_angle = 100

real_width = 45.0  # cm

pixel_per_cm = 640 / real_width  # pixels per cm

real_height = 480 / pixel_per_cm  # cm

forearm_length = 12.0  # cm or could be 15.5
upperarm_length = 12.0  # cm

base_location_x = 295
base_location_y = 385

object_texts = [["sharpener", "eraser", "pen", "pencil"]]
drop_target_texts = []

abort_event = threading.Event()
annotated_frame = None
pickup_thread = None  # Track the current pickup thread
wake_word_enabled = True  # Toggle for wake word listening

def check_abort():
    """Raise an exception if abort was requested"""
    if abort_event.is_set():
        raise InterruptedError("Pickup aborted by user!")

def interruptible_sleep(duration):
    """Sleep for duration seconds, but check for abort every 0.1s"""
    steps = int(duration / 0.1)
    for _ in range(steps):
        check_abort()
        time.sleep(0.1)
    # Sleep remaining time
    remaining = duration - (steps * 0.1)
    if remaining > 0:
        check_abort()
        time.sleep(remaining)

def draw_base_circle(frame):
    """Draw a circle representing the arm's reach around the base location"""
    radius_cm = 20
    radius_pixels = int(radius_cm * pixel_per_cm)
    cv2.circle(frame, (base_location_x - 30, base_location_y), radius_pixels, (255, 255, 0), 2)  # Light blue (cyan)

# Ignore Region variables
ignore_x1, ignore_y1, ignore_x2, ignore_y2 = 100, 100, 540, 380  # Default ignore region
dragging_corner = None  # Which corner is being dragged
corner_size = 15  # Size of the draggable corner areas

def mouse_callback(event, x, y, flags, param):
    global ignore_x1, ignore_y1, ignore_x2, ignore_y2, dragging_corner
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is near any corner
        if abs(x - ignore_x1) < corner_size and abs(y - ignore_y1) < corner_size:
            dragging_corner = 'top_left'
        elif abs(x - ignore_x2) < corner_size and abs(y - ignore_y1) < corner_size:
            dragging_corner = 'top_right'
        elif abs(x - ignore_x1) < corner_size and abs(y - ignore_y2) < corner_size:
            dragging_corner = 'bottom_left'
        elif abs(x - ignore_x2) < corner_size and abs(y - ignore_y2) < corner_size:
            dragging_corner = 'bottom_right'
    
    elif event == cv2.EVENT_MOUSEMOVE and dragging_corner:
        # Update corner position based on which corner is being dragged
        if dragging_corner == 'top_left':
            ignore_x1, ignore_y1 = max(0, min(x, ignore_x2 - 50)), max(0, min(y, ignore_y2 - 50))
        elif dragging_corner == 'top_right':
            ignore_x2, ignore_y1 = min(640, max(x, ignore_x1 + 50)), max(0, min(y, ignore_y2 - 50))
        elif dragging_corner == 'bottom_left':
            ignore_x1, ignore_y2 = max(0, min(x, ignore_x2 - 50)), min(480, max(y, ignore_y1 + 50))
        elif dragging_corner == 'bottom_right':
            ignore_x2, ignore_y2 = min(640, max(x, ignore_x1 + 50)), min(480, max(y, ignore_y1 + 50))
    
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_corner = None

keyboard.on_press_key('a', lambda e: on_a_press(e))

keyboard.on_press_key('g', lambda e: on_g_press(e))
keyboard.on_press_key('0', lambda e: on_0_press(e))
keyboard.on_press_key('semicolon', lambda e: on_semicolon_press(e))
keyboard.on_press_key('o', lambda e: on_o_press(e))
keyboard.on_press_key('k', lambda e: on_k_press(e))
keyboard.on_press_key('l', lambda e: on_l_press(e))

def object_identification(frame, texts):
    # Create a mask for the ignore region
    global ignore_x1, ignore_y1, ignore_x2, ignore_y2
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    mask[ignore_y1:ignore_y2, ignore_x1:ignore_x2] = 0  # Set ignore region to 0
    
    # Apply mask to frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    img = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    inputs = processor(text=texts, images=img, return_tensors="pt")
    outputs = model(**inputs)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=[img.size[::-1]], threshold=0.005)[0]

    # Check if any objects were detected
    if len(results["scores"]) == 0:
        print("No objects detected.")
        return
    
    highest_score_idx = results["scores"].argmax().item()
    object_label = results["labels"][highest_score_idx].item()
    box = results["boxes"][highest_score_idx].tolist()
    
    print(f"Object: {texts[0][object_label]}, Score: {results['scores'][highest_score_idx].item():.3f}")
    print(f"Box Coordinates: {box}")

    object_center_x = (box[0] + box[2]) / 2
    object_center_y = (box[1] + box[3]) / 2
    return {
        "object_label": object_label,
        "box": box,
        "object_center_x": object_center_x,
        "object_center_y": object_center_y,
    }

def calculate_distance_and_angle(frame, object_center_x, object_center_y):
    height, width = frame.shape[:2]
    global base_location_x, base_location_y
    distance_x = object_center_x - base_location_x
    distance_y = object_center_y - base_location_y

    angle_rad = math.atan2(distance_y, distance_x)
    angle_deg = math.degrees(angle_rad)

    distance = math.hypot(distance_x, distance_y)
    real_distance = distance / pixel_per_cm  + 4 # Adding 4 cm as an offset

    return distance, real_distance, angle_deg


def move_upper_servos_to_default():
    shoulder_servo_angle = 90
    move_slowly(2, shoulder_servo_angle)
    elbow_servo_angle = 90
    move_slowly(3, elbow_servo_angle)

def pick_up_object(real_distance):
    elbow_servo_angle = math.acos(np.clip((upperarm_length**2 + forearm_length**2 - real_distance**2) / (2 * upperarm_length * forearm_length), -1.0, 1.0))
    elbow_servo_angle = (math.degrees(elbow_servo_angle) + 0)
    move_slowly(3, elbow_servo_angle)
    print(f"Elbow Servo Angle: {elbow_servo_angle:.2f} degrees")

    shoulder_servo_angle = math.acos(np.clip((upperarm_length**2 + real_distance**2 - forearm_length**2) / (2 * upperarm_length * real_distance), -1.0, 1.0))
    shoulder_servo_angle = 180 - (math.degrees(shoulder_servo_angle) - 20)
    move_slowly(2, shoulder_servo_angle)
    print(f"Shoulder Servo Angle: {shoulder_servo_angle:.2f} degrees")
    time.sleep(0.5)
    global claw_servo_angle
    claw_servo_angle = 20  # Close claw

def drop_object(drop_real_distance):
    drop_real_distance -= 4  # Adjust for drop offset
    print("Dropping object...")
    drop_height = 25  # cm
    drop_point_distance = math.sqrt(drop_real_distance**2 + drop_height**2)

    elbow_servo_angle = math.acos(np.clip((upperarm_length**2 + forearm_length**2 - drop_point_distance**2) / (2 * upperarm_length * forearm_length), -1.0, 1.0))
    elbow_servo_angle = (math.degrees(elbow_servo_angle) + 0) + 10
    print(f"Elbow Servo Angle for Drop: {elbow_servo_angle:.2f} degrees")
    move_slowly(3, elbow_servo_angle)

    shoulder_servo_angle = math.acos(np.clip((upperarm_length**2 + drop_point_distance**2 - forearm_length**2) / (2 * upperarm_length * drop_point_distance), -1.0, 1.0))
    shoulder_servo_angle = math.degrees(shoulder_servo_angle) + math.degrees(math.atan(drop_height / drop_real_distance))  # Adjust for drop height

    shoulder_servo_angle = 180 - (shoulder_servo_angle - 20)
    print(f"Shoulder Servo Angle for Drop: {shoulder_servo_angle:.2f} degrees")
    move_slowly(2, shoulder_servo_angle)
    time.sleep(0.5)
    global claw_servo_angle
    claw_servo_angle = 100  # Open claw

    move_upper_servos_to_default()


def annotate_frame(frame, box, object_center_x, object_center_y, drop_target_box, drop_target_center_x, drop_target_center_y):
    global annotated_frame, ignore_x1, ignore_y1, ignore_x2, ignore_y2
    
    # Draw ignore region rectangle
    cv2.rectangle(frame, (ignore_x1, ignore_y1), (ignore_x2, ignore_y2), (0, 0, 255), 2)  # Red
    
    # Draw diagonal lines across ignore region
    num_lines = 10
    step_x = (ignore_x2 - ignore_x1) // num_lines
    step_y = (ignore_y2 - ignore_y1) // num_lines
    
    # Diagonal lines from top-left to bottom-right
    for i in range(num_lines + 1):
        cv2.line(frame, 
                (ignore_x1 + i * step_x, ignore_y1), 
                (ignore_x1, ignore_y1 + i * step_y), 
                (0, 0, 255), 1)
        cv2.line(frame, 
                (ignore_x1 + i * step_x, ignore_y2), 
                (ignore_x2, ignore_y1 + i * step_y), 
                (0, 0, 255), 1)
    
    
    draw_base_circle(frame)
    
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    cv2.circle(frame, (int(object_center_x), int(object_center_y)), 5, (255, 0, 0), -1)
    cv2.line(frame, (base_location_x, base_location_y), (int(object_center_x), int(object_center_y)), (0, 0, 255), 2)
    cv2.putText(frame, f"Angle: {base_servo_angle:.2f} degrees", (int(object_center_x), int(object_center_y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)


    if drop_target_box is not None:
        cv2.rectangle(frame, (int(drop_target_box[0]), int(drop_target_box[1])), (int(drop_target_box[2]), int(drop_target_box[3])), (0, 255, 255), 2)
        cv2.circle(frame, (int(drop_target_center_x), int(drop_target_center_y)), 5, (0, 255, 255), -1)
        cv2.line(frame, (base_location_x, base_location_y), (int(drop_target_center_x), int(drop_target_center_y)), (0, 255, 255), 2)
        cv2.putText(frame, f"Drop Target", (int(drop_target_center_x), int(drop_target_center_y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)

    annotated_frame = frame.copy()

def on_a_press(event):
    global pickup_thread
    # If a pickup is already running, don't start another one
    if pickup_thread is not None and pickup_thread.is_alive():
        print("Pickup already in progress!")
        return
    
    # Start pickup in a new thread
    pickup_thread = threading.Thread(target=pickup_sequence, daemon=True)
    pickup_thread.start()

def pickup_sequence():
    """The actual pickup logic that runs in a separate thread"""
    abort_event.clear()  # Reset abort flag

    print("Starting pickup sequence...")
    try:
        ret, frame = capture.read()
        if not ret:
            return

        frame = cv2.resize(frame, (640, 480))

        # Retry mechanism for distance validation
        max_retries = 3
        retry_count = 0
        real_distance = 0
        
        while retry_count < max_retries:
            check_abort()
            
            results = object_identification(frame, [object_texts[0]])
            box = results["box"]
            object_label = results["object_label"]
            object_center_x = results["object_center_x"]
            object_center_y = results["object_center_y"]

            distance, real_distance, angle_deg = calculate_distance_and_angle(frame, object_center_x, object_center_y)
            
            if real_distance >= 4:
                break
            
            retry_count += 1
            print(f"Distance too small ({real_distance:.2f} cm). Retrying detection... ({retry_count}/{max_retries})")
            interruptible_sleep(0.5)
            ret, frame = capture.read()
            if not ret:
                return
            frame = cv2.resize(frame, (640, 480))
        
        if real_distance < 4:
            print(f"Warning: Distance still too small after {max_retries} retries ({real_distance:.2f} cm). Aborting pickup.")
            return

        check_abort()

        if drop_target_texts and drop_target_texts[0]:
            print("drop target texts =", drop_target_texts)
            results = object_identification(frame, drop_target_texts)
            drop_target_box = results["box"]
            drop_target_label = results["object_label"]
            drop_target_center_x = results["object_center_x"]
            drop_target_center_y = results["object_center_y"]
        else:
            drop_target_box = None
            drop_target_center_x = None
            drop_target_center_y = None

        annotate_frame(frame, box, object_center_x, object_center_y, drop_target_box, drop_target_center_x, drop_target_center_y)
        interruptible_sleep(0.5)
        
        distance, real_distance, angle_deg = calculate_distance_and_angle(frame, object_center_x, object_center_y)
        if drop_target_box is not None:
            drop_distance, drop_real_distance, drop_angle_deg = calculate_distance_and_angle(frame, drop_target_center_x, drop_target_center_y)

        check_abort()

        move_slowly(1, angle_deg * -1 - 30)
        check_abort()
        move_upper_servos_to_default()
        check_abort()
        interruptible_sleep(0.5)

        check_abort()

        pick_up_object(real_distance)
        check_abort()
        interruptible_sleep(1.0)

        check_abort()

        move_upper_servos_to_default()
        check_abort()

        interruptible_sleep(1.0)

        check_abort()

        if drop_target_box is not None:
            move_slowly(1, drop_angle_deg * -1 - 30)
            check_abort()
            interruptible_sleep(1.0)

            check_abort()

            drop_object(drop_real_distance)
            check_abort()
            interruptible_sleep(1.0)
        else:
            move_slowly(1, 180)
            check_abort()
        
            interruptible_sleep(1.0)
            global claw_servo_angle
            claw_servo_angle = 100  # Open claw
        
        print("Pickup sequence completed successfully!")
    
    except InterruptedError as e:
        print(str(e))
        return


def move_slowly(servo, end_angle):
    global base_servo_angle, shoulder_servo_angle, elbow_servo_angle, claw_servo_angle
    if servo == 1:
        start_angle = base_servo_angle
    elif servo == 2:
        start_angle = shoulder_servo_angle
    elif servo == 3:
        start_angle = elbow_servo_angle
    elif servo == 4:
        start_angle = claw_servo_angle
    else:
        return

    step = 1 if end_angle > start_angle else -1
    for angle in range(int(start_angle), int(end_angle + step), int(step)):
        check_abort()  # Check once per step
        if servo == 1:
            base_servo_angle = angle
        elif servo == 2:
            shoulder_servo_angle = angle
        elif servo == 3:
            elbow_servo_angle = angle
        time.sleep(0.02)

def on_o_press(event):
    global base_location_y
    base_location_y -= 5

def on_k_press(event):
    global base_location_x
    base_location_x -= 5

def on_l_press(event):
    global base_location_y
    base_location_y += 5

def on_semicolon_press(event):
    global base_location_x
    base_location_x += 5

def on_g_press(event):
    abort_event.set()
    print("Abort requested! Stopping pickup operation.")

def on_0_press(event):
    global wake_word_enabled
    wake_word_enabled = not wake_word_enabled
    status = "enabled" if wake_word_enabled else "disabled"
    print(f"Wake word listening {status}")

# Speech Recognition and AI starts here
load_dotenv()  # Load environment variables from .env file

# Settings
sample_rate = 16000  # 16 kHz recommended for Whisper
channels = 1  # mono audio

chunk_duration = 30  # ms
chunk_size = int(sample_rate * chunk_duration / 1000)  # samples per chunk  

vad = webrtcvad.Vad(2)   # 0-3

silent_chunks_threshold = 40

audio_chunks = []
silent_chunks = 0

final_text = ""

def callback(indata, frames, time, status):
    global silent_chunks
    global audio_chunks
    audio_data = indata[:, 0]
    # Convert float32 audio to int16 PCM format for webrtcvad
    audio_int16 = (audio_data * 32767).astype(np.int16)
    is_speech = vad.is_speech(audio_int16.tobytes(), sample_rate)

    if is_speech:
        audio_chunks.append(audio_data.copy())
        silent_chunks = 0
    else:
        silent_chunks += 1

buffer = io.BytesIO()
buffer.name = 'recording.wav'


def record_audio():
    global silent_chunks
    global audio_chunks
    global buffer
    audio_chunks.clear()
    silent_chunks = 0
    print("Recording...")
    with sd.InputStream(samplerate=16000, channels=1, blocksize=chunk_size, callback=callback):
        while silent_chunks < silent_chunks_threshold:
            pass
    print("Recording complete.")
    if not audio_chunks:
        return
    audio_data = np.concatenate(audio_chunks)
    if audio_data.size == 0:
        return
    buffer = io.BytesIO()
    buffer.name = "audio.wav"
    sf.write(buffer, audio_data, sample_rate, format='WAV')
    buffer.seek(0)
    transcribe_audio_openai()
    

def transcribe_audio_openai():
    global final_text
    final_text = ""
    audio_file = buffer
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file,
        response_format="text",
        language="en",
    )
    print(transcript)

    if not transcript.strip():
        return

    final_text = transcript

    
    chat(final_text)

    return transcript


client = OpenAI()


def chat(final_text):
    if not final_text:
        print("Error: No prompt provided")
        return None
    try:
        print("time before api call:", time.strftime("%H:%M:%S", time.localtime()), f"{int((time.time() % 1) * 1000):03d}ms")
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are an AI named HELIX that controls a robotic arm with vision capabilities and outputs JSON. \
                    Identify the user's intent and return a set of objects you need to look for as 'object-texts' field. eg: Pen, Pencil.\
                    Identify if any specific drop location is requested be the user and return a set of drop locations to look for as\
                    'drop-target-texts' field. eg: Hand, Basket etc.\
                    Leave both fields blank it blank if not applicable. \
                    Also return a 'text-reply' field with a short response to the user."
                },
                {"role": "user", "content": final_text}
            ],
            response_format={ "type": "json_object" }
        )
        reply = json.loads(response.choices[0].message.content)["text-reply"]
        print("time after api call:", time.strftime("%H:%M:%S", time.localtime()), f"{int((time.time() % 1) * 1000):03d}ms")
        print(reply)
        print("object texts =", json.loads(response.choices[0].message.content)["object-texts"])
        print("drop target texts =", json.loads(response.choices[0].message.content)["drop-target-texts"])
        global object_texts
        object_texts = [json.loads(response.choices[0].message.content)["object-texts"]]

        
        global drop_target_texts
        drop_target_texts = [json.loads(response.choices[0].message.content)["drop-target-texts"]]

        
        TTS(reply)
        if object_texts != [[]]:
            on_a_press(None)  # This will now start a thread
        else:
            print("no request found. cancelling pickup ")

        return reply
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def TTS(text):
    audio_response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="echo",
        input=text
    )
    
    audio_buffer = io.BytesIO(audio_response.read())
    audio_buffer.seek(0)
    data, samplerate = sf.read(audio_buffer, dtype='float32')
    
    sd.play(data, samplerate)
    sd.wait()  # Wait until playback is finished


PV_API_KEY = os.getenv("PV_API_KEY")
keyword_path = "./Helix_en_windows_v3_0_0.ppn"
porcupine = pvporcupine.create(keyword_paths=[keyword_path], access_key=PV_API_KEY)  # keyword_paths expects a list
recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)


def ListenForWake():
    while True:
        if wake_word_enabled:
            pcm = recorder.read()
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Wake word detected!")
                record_audio()
        else:
            time.sleep(0.1)  # Sleep briefly when disabled to reduce CPU usage

def voice_thread():
    recorder.start()
    try:
        while True:
            print("Listening for wakeword...")
            ListenForWake()
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        porcupine.delete()
        recorder.delete()

# Start voice recognition in a separate thread
voice_thread_instance = threading.Thread(target=voice_thread, daemon=True)
voice_thread_instance.start()

# Main loop to keep the program running
cv2.namedWindow("Live Feed")
cv2.setMouseCallback("Live Feed", mouse_callback)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]

    # Draw ignore region rectangle
    cv2.rectangle(frame, (ignore_x1, ignore_y1), (ignore_x2, ignore_y2), (0, 0, 255), 2)  # Red
    
    # Draw diagonal lines across ignore region
    num_lines = 10
    step_x = (ignore_x2 - ignore_x1) // num_lines
    step_y = (ignore_y2 - ignore_y1) // num_lines
    
    # Diagonal lines from top-left to bottom-right
    for i in range(num_lines + 1):
        cv2.line(frame, 
                (ignore_x1 + i * step_x, ignore_y1), 
                (ignore_x1, ignore_y1 + i * step_y), 
                (0, 0, 255), 1)
        cv2.line(frame, 
                (ignore_x1 + i * step_x, ignore_y2), 
                (ignore_x2, ignore_y1 + i * step_y), 
                (0, 0, 255), 1)
    
    # Draw corner handles
    cv2.circle(frame, (ignore_x1, ignore_y1), 5, (0, 0, 255), -1)  # Top-left
    cv2.circle(frame, (ignore_x2, ignore_y1), 5, (0, 0, 255), -1)  # Top-right
    cv2.circle(frame, (ignore_x1, ignore_y2), 5, (0, 0, 255), -1)  # Bottom-left
    cv2.circle(frame, (ignore_x2, ignore_y2), 5, (0, 0, 255), -1)  # Bottom-right

    draw_base_circle(frame)
    
    cv2.circle(frame, (base_location_x, base_location_y), 5, (0, 0, 255), -1)

    # Display live feed
    cv2.imshow("Live Feed", frame)
    
    # Display annotated frame in a separate window if available
    if annotated_frame is not None:
        cv2.imshow("Detected Object", annotated_frame)

    instructions = {"angle": base_servo_angle, "servo": 1}
    pico_serial.write(f"{json.dumps(instructions)}\n".encode())
    instructions = {"angle": shoulder_servo_angle, "servo": 2}
    pico_serial.write(f"{json.dumps(instructions)}\n".encode())
    instructions = {"angle": elbow_servo_angle, "servo": 3}
    pico_serial.write(f"{json.dumps(instructions)}\n".encode())
    instructions = {"angle": claw_servo_angle, "servo": 4}
    pico_serial.write(f"{json.dumps(instructions)}\n".encode())
    pico_serial.flush()

    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False  # Signal threads to stop
        keyboard.unhook_all()  # Remove all keyboard listeners
        pico_serial.close()
        
        # Wait for voice thread to finish
        voice_thread_instance.join(timeout=2)
        break

capture.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()
sys.exit()