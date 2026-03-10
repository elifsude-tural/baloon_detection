from ultralytics import YOLO
import cv2
import numpy as np


model = YOLO("models/baloon_model.pt")
print(model.names)

def get_balloon_color(frame, x1, y1, x2, y2):
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "Unknown", (128, 128, 128)
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    
    red_mask1 = cv2.inRange(hsv, (0, 100, 100),   (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    red_mask  = cv2.bitwise_or(red_mask1, red_mask2)
    
    
    #blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
    blue_mask = cv2.inRange(hsv, (85, 60, 80), (135, 255, 255))
    
    red_pixels  = cv2.countNonZero(red_mask)
    blue_pixels = cv2.countNonZero(blue_mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    
    
    threshold = total_pixels * 0.15
    
    if red_pixels > blue_pixels and red_pixels > threshold:
        return "Enemy",   (0, 0, 255)      
    elif blue_pixels > red_pixels and blue_pixels > threshold:
        return "Friendly", (255, 0, 0)     
    else:
        return "Balloon",  (0, 255, 0)     

def draw_detection(frame, x1, y1, x2, y2, label, color, conf):
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # etiket arka planı
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
    
    # etiket metni
    cv2.putText(frame, text,
                (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)


crosshair_img = cv2.imread("crosshairfinal.png", cv2.IMREAD_UNCHANGED)  # UNCHANGED şeffaflık için


def draw_crosshair(frame, size=200, cx=None, cy=None):
    h, w = frame.shape[:2]
    if cx is None: cx = w // 2
    if cy is None: cy = h // 2

    # --- Nişangah görseli ---
    ch = cv2.resize(crosshair_img, (size, size))
    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = x1 + size
    y2 = y1 + size
    alpha = ch[:, :, 3] / 255.0
    for c in range(3):
        frame[y1:y2, x1:x2, c] = (
            alpha * ch[:, :, c] +
            (1 - alpha) * frame[y1:y2, x1:x2, c]
        )

    # --- Uzun çizgiler ---
    color = (132, 224, 0)
    thick = 1
    gap   = size // 2 + 10

    cv2.line(frame, (cx - gap, cy), (0, cy), color, thick)
    cv2.line(frame, (cx + gap, cy), (w, cy), color, thick)
    cv2.line(frame, (cx, cy - gap), (cx, 0), color, thick)
    cv2.line(frame, (cx, cy + gap), (cx, h), color, thick)

    # --- Tick işaretleri ---
    tick_len   = 8
    tick_space = 40
    tick_color = color
    tick_thick = 1

    for tx in range(cx - gap, 0, -tick_space):
        cv2.line(frame, (tx, cy - tick_len), (tx, cy + tick_len), tick_color, tick_thick)
    for tx in range(cx + gap, w, tick_space):
        cv2.line(frame, (tx, cy - tick_len), (tx, cy + tick_len), tick_color, tick_thick)

    for ty in range(cy - gap, 0, -tick_space):
        cv2.line(frame, (cx - tick_len, ty), (cx + tick_len, ty), tick_color, tick_thick)
    for ty in range(cy + gap, h, tick_space):
        cv2.line(frame, (cx - tick_len, ty), (cx + tick_len, ty), tick_color, tick_thick)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows için
    if not cap.isOpened():
        print("Camera didn't Open.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        results = model(frame)[0]

        enemy_count = 0
        friendly_count = 0
        unknown_count = 0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            label,color = get_balloon_color(frame, x1,y1,x2,y2)
            draw_detection(frame,x1,y1,x2,y2,label,color,conf)

            if label == "Enemy": enemy_count +=1
            elif label =="Friendly": friendly_count += 1
            else: unknown_count +=1

        draw_crosshair(frame)
        cv2.putText(frame, f"Enemy:    {enemy_count}",    (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),   2)
        cv2.putText(frame, f"Friendly: {friendly_count}", (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),   2)
        cv2.putText(frame, f"Unknown:  {unknown_count}",  (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),   2)
        
        cv2.imshow("Baloon detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if __name__ == "__main__":
    main()
