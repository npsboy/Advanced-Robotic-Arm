<img width="4400" height="1017" alt="Advanced Robotic Arm" src="https://github.com/user-attachments/assets/fc1bc382-4a15-4a1b-b5b5-1dc9242e6f6c" />

# What is it?

- A standard robotic arm with 3 DOF + 1 gripper. IE: Rotating Base, Shoulder, Elbow Joint, Gripper.
- Computer vision-assisted. Should be able to detect common items and pick them up as long as they are in sight.
- Controlled by voice commands in natural language.

# How does it work?
- Controlled by a Raspberry Pi Pico.
- The main program runs on the computer.
- The computer relays the instructions to the Pico.
- A webcam is positioned with a top-down view of the surroundings to help the arm detect objects and their positions.
<img width="4405" height="1414" alt="diagram" src="https://github.com/user-attachments/assets/806618e4-5085-4f0b-a2ec-c85382208395" />

## Native dependencies

Install the PortAudio runtime before installing Python requirements so microphone input works:
- Ubuntu / Debian / WSL: `sudo apt-get install libportaudio2 portaudio19-dev`
- macOS (Homebrew): `brew install portaudio`
- Windows: `pip install pipwin && pipwin install portaudio`
