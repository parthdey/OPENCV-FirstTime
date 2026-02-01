import cv2
import mediapipe as mp
import random  # Add this at the top with other imports
import time


# === FRUIT CLASS (add this near the top, after imports) ===
class Fruit:
    def __init__(self, x, y):
        # Your fruit properties here
        self.x = x
        self.y = y
        self.radius = 30
        self.velocity_y = 0
        self.gravity = 2
        self.is_sliced = False
        self.color = (0, 255, 0)  # Green color for the fruit
    
    def update(self):
        # Apply gravity (increase velocity)
        self.velocity_y += self.gravity
    
        # Update position based on velocity
        self.y += self.velocity_y

        # === ADD THIS: Respawn if it goes off screen ===
        if self.y > 480:  # Assuming camera height is 480
            self.y = 0
            self.velocity_y = 0
        
    def draw(self, frame):
        if not self.is_sliced:
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)


def check_collision(trail, fruit):
    """Check if the finger trail intersects with a fruit"""
    if fruit.is_sliced:
        return False
    
    # Check each point in the trail
    for point in trail:
        x, y = point
        # Calculate distance between trail point and fruit center
        distance = ((x - fruit.x) ** 2 + (y - fruit.y) ** 2) ** 0.5
        
        # If distance is less than radius, we hit it!
        if distance < fruit.radius:
            return True
    
    return False


trail = []


cap = cv2.VideoCapture(0)

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

landmarks = None

def callback(result, output_image, timestamp_ms):
    global landmarks
    if result and result.hand_landmarks:
        landmarks = result.hand_landmarks
    else:
        landmarks = None

options = HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=callback
)

detector = HandLandmarker.create_from_options(options)


fruits = []  # Empty list to hold multiple fruits
spawn_timer = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detector.detect_async(mp_image, int(time.time() * 1000))

    h, w, _ = frame.shape  # Get screen size
    
    # Spawn new fruits randomly
    spawn_timer += 1
    if spawn_timer > 30:  # Every 30 frames (~1 second)
        spawn_timer = 0
        random_x = random.randint(50, w - 50)
        fruits.append(Fruit(random_x, 0))

    # Update and draw all fruits
    for fruit in fruits[:]:  # Use [:] to avoid modifying list while iterating
        fruit.update()
        fruit.draw(frame)
        
        # Check if hand trail slices the fruit
        if trail and check_collision(trail, fruit):
            fruit.is_sliced = True
            print("SLICED!")  # You'll see this in console when you slice
        
        # Remove fruits that fall off screen OR are sliced
        if fruit.y > h or fruit.is_sliced:
            fruits.remove(fruit)

    if landmarks:
        hand = landmarks[0]   # only 1 hand
        h, w, _ = frame.shape

        # Index finger tip
        lm = hand[8]
        x, y = int(lm.x * w), int(lm.y * h)

        trail.append((x, y))
        if len(trail) > 15:
            trail.pop(0)

        # draw blade trail
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i-1], trail[i], (0, 0, 255), 5)


    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# # Now my_fruit has all those properties:
# print(my_fruit.x)           # 100
# print(my_fruit.radius)      # 30
# print(my_fruit.is_sliced)   # False

cap.release()
cv2.destroyAllWindows()
