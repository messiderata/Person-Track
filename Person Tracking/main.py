import customtkinter as ctk
import serial
import threading
import time
from ultralytics import YOLO
import cv2

# Load your weight
model = YOLO('Model.pt')

# Set up the serial connection to the Arduino board
arduino = serial.Serial('COM17', 115200, timeout=1)  # Adjust COM port, baudrate, and timeout as needed

# Function to write to the servo motor
def write_servo_angle(angle):
    command = f'{angle}\n'.encode()
    arduino.write(command)
    arduino.flush()

# Define the center position of the camera
camera_center = 320  # Assuming the camera resolution width is 640 pixels
center_threshold = 10  # Define the threshold for considering the object to be centered

# Counter to track consecutive frames without detecting any object
no_object_count = 0

# Initialize the servo angle
servo_angle = 90  # Start at the center position

# Function to run inference and servo control
def run_inference_and_servo_control(source, conf):
    global no_object_count, servo_angle

    cap = cv2.VideoCapture(int(source))  # Open the default camera

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return
    
    print(f"Successfully opened camera with source {source}.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the camera.")
            break

        results = model.predict(source=frame, save=False, imgsz=640, conf=conf, show=True, stream=True)

        if not results:
            print("No results from the model.")
            continue

        object_detected = False  # Flag to indicate if any object is detected in the current frame

        for r in results:
            boxes = r.boxes
            bounding_boxes = boxes.xyxy.tolist()

            if bounding_boxes:
                no_object_count = 0

                bbox = bounding_boxes[0]
                x_center = (bbox[0] + bbox[2]) / 2

                # Determine the direction to move the servo
                if x_center < camera_center - center_threshold:  # Object is to the left of the center
                    servo_angle += 2
                elif x_center > camera_center + center_threshold:  # Object is to the right of the center
                    servo_angle -= 2

                # Clamp the servo angle to [0, 180]
                servo_angle = max(0, min(180, servo_angle))

                write_servo_angle(servo_angle)
                print(f"Servo angle: {servo_angle}")

                object_detected = True

                # Draw the angle on the frame
                cv2.putText(frame, f'Angle: {servo_angle}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            else:
                print("No bounding boxes detected.")

        if not object_detected:
            no_object_count += 1

            if no_object_count >= 20:
                write_servo_angle(90)
                print("No object detected for 20 consecutive frames. Stopping servo at 90 degrees.")
                no_object_count = 0
        
        time.sleep(0.0001)  # Adjust this value to control the speed of servo adjustments

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- GUI Section ---
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        my_font = ctk.CTkFont(family="Satoshi-Bold", size=14)

        self.title("Object Detection and Servo Control")
        self.geometry("400x300")
        self.configure(fg_color="#20242a") 
        self.center_window()

        # --- Frames for GUI Elements ---
        button_frame = ctk.CTkFrame(master=self, fg_color="transparent")
        button_frame.pack(pady=10, padx=10)

        button_frame2 = ctk.CTkFrame(master=self, fg_color="transparent")
        button_frame2.pack(pady=5, padx=10)

        # --- Labels and ComboBoxes ---
        self.result_label = ctk.CTkLabel(
            master=button_frame,
            text="Source:",
            fg_color="transparent",
            text_color="white",
            font=my_font
        )
        self.result_label.pack(pady=5, padx=10, expand=True)

        self.result_label1 = ctk.CTkLabel(
            master=button_frame2,
            text="Confidence:",
            fg_color="transparent",
            text_color="white",
            font=my_font
        )
        self.result_label1.pack(pady=5, padx=10, expand=True)

        self.optionmenu_1 = ctk.CTkComboBox(
            button_frame,
            fg_color="#fed32c",
            text_color="#20242a",
            button_color="#fed32c",
            border_color="#fed32c",
            font=my_font,
            values=["0", "1", "2"]
        )
        self.optionmenu_1.pack(pady=10)

        self.optionmenu_2 = ctk.CTkComboBox(
            button_frame2,
            fg_color="#fed32c",
            text_color="#20242a",
            button_color="#fed32c",
            border_color="#fed32c",
            font=my_font,
            values=["0.25", "0.50", "0.75", "1.00"]
        )
        self.optionmenu_2.pack(pady=10)

        self.button1 = ctk.CTkButton(
            master=button_frame2,
            text="START",
            command=self.print_combo_box,
            fg_color="white",
            hover_color="#fede5f",
            text_color="#20242a",
            font=my_font
        )
        self.button1.pack(pady=5, padx=10, side="bottom")

    def print_combo_box(self):
        source = self.optionmenu_1.get()
        conf = float(self.optionmenu_2.get())
        Run(source, conf)

    def center_window(self):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - self.winfo_reqwidth()) // 2
        y = (screen_height - self.winfo_reqheight()) // 2
        self.geometry(f"+{x}+{y}")


def Run(source, conf):
    # Create and start the thread
    inference_thread = threading.Thread(target=run_inference_and_servo_control, args=(source, conf))
    inference_thread.daemon = True  # Daemonize the thread so it automatically closes when the main program exits
    inference_thread.start()

    # Optionally return the thread object if needed
    return inference_thread


if __name__ == "__main__":
    app = App()
    app.mainloop()