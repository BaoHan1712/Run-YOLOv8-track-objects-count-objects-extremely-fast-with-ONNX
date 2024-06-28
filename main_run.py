import cv2
from ultralytics import YOLO, solutions


#   resize frame to 256x256 
model_init = YOLO("yolov8s.onnx")

model = model_init(imgsz=(256, 256), conf = 0.2, iou = 0.5)

cap = cv2.VideoCapture(1)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

#Line counting 
line_points = [(10, 300), (1080, 300)]

counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    classes_names=model_init.names,
    draw_tracks=True,
    line_thickness=2,
    cls_txtdisplay_gap = 15
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        continue

    tracks = model_init.track(im0, persist=True, show=False)

    im0 = counter.start_counting(im0, tracks)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
