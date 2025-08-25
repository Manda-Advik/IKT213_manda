import cv2

cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with open("camera_outputs.txt", "w") as f:
    f.write(f"fps: {fps}\n")
    f.write(f"height: {height}\n")
    f.write(f"width: {width}\n")

cap.release()
