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

texts = [["sharpener", "eraser", "pen", "pencil"]]


def nothing(x):
    return

print("Press 'a' to capture an image and detect objects.")
print("Press 'q' to quit.")

annotated_frame = None

keyboard.on_press_key('a', lambda e: on_a_press(e))

keyboard.on_press_key('semicolon', lambda e: on_semicolon_press(e))
keyboard.on_press_key('o', lambda e: on_o_press(e))
keyboard.on_press_key('k', lambda e: on_k_press(e))
keyboard.on_press_key('l', lambda e: on_l_press(e))

def object_identification(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    global texts
    inputs = processor(text=texts, images=img, return_tensors="pt")
    outputs = model(**inputs)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=[img.size[::-1]], threshold=0.01)[0]

    highest_score_idx = results["scores"].argmax().item()
    object_label = results["labels"][highest_score_idx].item() #.item() to get value from tensor
    box = results["boxes"][highest_score_idx].tolist()

    # Check if any objects were detected
    if len(results["scores"]) == 0:
        print("No objects detected.")
        return
    
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
    real_distance = distance / pixel_per_cm + 2  # Adding 4a cm as an offset

    return distance, real_distance, angle_deg

def move_base_servo(angle_deg):
    base_servo_angle = angle_deg * -1 - 30
    move_slowly(1, base_servo_angle)
    print("angle_deg:", angle_deg)
    print(f"Base Servo Angle: {base_servo_angle:.2f} degrees")
    if base_servo_angle < 0:
        base_servo_angle = 0
    elif base_servo_angle > 180:
        base_servo_angle = 180

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
    time.sleep(1)
    global claw_servo_angle
    claw_servo_angle = 20  # Close claw


def annotate_frame(frame, box, object_center_x, object_center_y):
    global annotated_frame
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    cv2.circle(frame, (int(object_center_x), int(object_center_y)), 5, (255, 0, 0), -1)
    cv2.line(frame, (base_location_x, base_location_y), (int(object_center_x), int(object_center_y)), (0, 0, 255), 2)
    cv2.putText(frame, f"Angle: {base_servo_angle:.2f} degrees", (int(object_center_x), int(object_center_y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    annotated_frame = frame.copy()  # Assign to global variable



def on_a_press(event):

    print("a pressed")
    ret, frame = capture.read()
    if not ret:
        return

    frame = cv2.resize(frame, (640, 480))


    results = object_identification(frame)
    box = results["box"]
    object_label = results["object_label"]
    object_center_x = results["object_center_x"]
    object_center_y = results["object_center_y"]

    annotate_frame(frame, box, object_center_x, object_center_y)


    distance, real_distance, angle_deg = calculate_distance_and_angle(frame, object_center_x, object_center_y)

    move_base_servo(angle_deg)
    move_upper_servos_to_default()
    time.sleep(1)

    pick_up_object(real_distance)
    time.sleep(1)


    move_upper_servos_to_default()

    time.sleep(1)

    
    move_slowly(1, 180)

    time.sleep(1)
    global claw_servo_angle
    claw_servo_angle = 100  # Open claw


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
                {"role": "system", "content": "You are an AI that controls a robotic arm with vision capabilities and outputs JSON. \
                    Identify the user's intent and return a set of objects you need to look for as 'texts' field. eg: Pen, Pencil.\
                 Leave it blank if not applicable. \
                 Also return a 'text-reply' field with a short response to the user."},
                {"role": "user", "content": final_text}
            ],
            response_format={ "type": "json_object" }
        )
        reply = json.loads(response.choices[0].message.content)["text-reply"]
        print("time after api call:", time.strftime("%H:%M:%S", time.localtime()), f"{int((time.time() % 1) * 1000):03d}ms")
        print(reply)
        global texts
        texts = [json.loads(response.choices[0].message.content)["texts"]]
        
        TTS(reply)
        on_a_press(None)
        
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
        pcm = recorder.read()
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            print("Wake word detected!")
            record_audio()

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
while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]


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