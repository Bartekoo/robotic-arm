import pygame
import sys
import math
import cv2
import mediapipe as mp
import numpy as np
import serial
from pygame import Vector2


class Game(object):
    def __init__(self):
        pygame.init()
        self.setup_screen()
        self.initialize_game_variables()
        self.setup_hand_tracking()
        self.main_game_loop()

    def setup_screen(self):
        # Setup display parameters
        self.BACKGROUND = (255, 255, 255)
        self.WINDOW_WIDTH = 800
        self.WINDOW_HEIGHT = 800
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Rotation and Geometry Example")
        self.font = pygame.font.Font(None, 30)

    def initialize_game_variables(self):
        # Initialize game state variables
        self.line_color = (230, 40, 40)
        self.line_length = 100
        self.line3_length = 20

        self.startPos = Vector2(400, 400)
        self.line1_pos = Vector2(490, 400)
        self.line2_pos = Vector2(580, 490)
        self.line3_pos = Vector2(670, 580)

        self.line1_rotation = 30
        self.line2_rotation = 60
        self.line3_rotation = 90

        self.looping = True
        self.fpsClock = pygame.time.Clock()
        self.FPS = 60
        self.isSet = False

        self.pivot_rotation = 0
        self.pivot_point = Vector2(0, 0)

        self.smoothing_factor = 0.2
        self.smoothed_pos = Vector2(400, 400)

        try:
            self.ser = serial.Serial('COM4', 9600)
        except serial.SerialException as e:
            print(f"Error initializing serial connection: {e}")
            self.ser = None

    def setup_hand_tracking(self):
        # Initialize Mediapipe Hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.cap = cv2.VideoCapture(0)

    def set_servo_angle(self, angle1, angle2):
        if self.ser and 0 <= angle1 <= 180 and 0 <= angle2 <= 180:
            command = f"{angle1} {angle2}\n"
            self.ser.write(command.encode())
            print("set")
            # time.sleep(0.1)

    def main_game_loop(self):
        # Main game loop
        while self.looping:
            self.handle_events()
            self.update_game_state()
            self.render()
            pygame.display.flip()
            self.set_servo_angle(int(round(self.line1_rotation)), int(round(self.line2_rotation)))
            self.fpsClock.tick(self.FPS)

        self.cleanup()

    def cleanup(self):
        # Proper cleanup of resources
        self.cap.release()
        if self.ser:
            self.ser.close()
        pygame.quit()
        sys.exit()

    def handle_events(self):
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.looping = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_d]:
            self.pivot_rotation += 4
        if keys[pygame.K_a]:
            self.pivot_rotation -= 4

    def update_game_state(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find hand landmarks
        result = self.hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract the coordinates of the tip of the right index finger (landmark 8)
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)

                # Map the coordinates to the Pygame window size
                mapped_x = int(cx * self.WINDOW_WIDTH / w)
                mapped_y = int(cy * self.WINDOW_HEIGHT / h)

                # Smooth the position using exponential moving average
                self.smoothed_pos.x = self.smoothing_factor * mapped_x + (1 - self.smoothing_factor) * self.smoothed_pos.x
                self.smoothed_pos.y = self.smoothing_factor * mapped_y + (1 - self.smoothing_factor) * self.smoothed_pos.y

                # Use the smoothed position in pivot_around function
                self.pivot_around(800 - int(self.smoothed_pos.x), int(self.smoothed_pos.y), 15, self.pivot_rotation)

    def render(self):
        # Render all game elements
        self.screen.fill(self.BACKGROUND)
        self.draw_lines()

    def update_rotation(self):
        self.line1_rotation = math.degrees(math.atan2(self.line1_pos.y - self.startPos.y, self.line1_pos.x - self.startPos.x)) % 360
        self.line2_rotation = (math.degrees(math.atan2(self.line2_pos.y - self.line1_pos.y, self.line2_pos.x - self.line1_pos.x)) - self.line1_rotation) % 360
        self.line3_rotation = (math.degrees(math.atan2(self.line3_pos.y - self.line2_pos.y, self.line3_pos.x - self.line2_pos.x)) - self.line1_rotation - self.line2_rotation) % 360

        print(self.line1_rotation, self.line2_rotation)

    def draw_lines(self):
        # Draw lines on the screen
        pygame.draw.aaline(self.screen, self.line_color, self.startPos, self.line1_pos, 5)
        pygame.draw.aaline(self.screen, self.line_color, self.line1_pos, self.line2_pos, 5)
        pygame.draw.aaline(self.screen, self.line_color, self.line2_pos, self.line3_pos, 5)

    def find_intersection(self, x, y):
        # Calculate intersection point that is within line length from start
        d = math.sqrt((x - self.startPos.x) ** 2 + (y - self.startPos.y) ** 2)
        if d > self.line_length * 2 or d < abs(self.line_length - self.line_length):
            return self.line1_pos

        a = (self.line_length ** 2 - self.line_length ** 2 + d ** 2) / (2 * d)

        h = math.sqrt(self.line_length ** 2 - a ** 2)

        x3 = self.startPos.x + a * (x - self.startPos.x) / d
        y3 = self.startPos.y + a * (y - self.startPos.y) / d
        return Vector2((x3 + h * (y - self.startPos.y) / d, y3 - h * (x - self.startPos.x) / d))

    def set_target_point(self, x, y, offset):
        self.line1_pos = self.find_intersection(x, y)
        self.line2_pos = Vector2(x, y)
        self.line3_pos = Vector2((x - self.line3_length * math.cos(offset * math.pi / 180), y - self.line3_length * math.sin(offset * math.pi / 180)))
        self.update_rotation()

    def pivot_around(self, x, y, distance, offset):
        # Pivot a point around a given center
        self.set_target_point(x + (distance + self.line3_length) * math.cos(offset * math.pi / 180), y + (distance + self.line3_length) * math.sin(offset * math.pi / 180), offset)
        self.update_rotation()


if __name__ == "__main__":
    Game()
