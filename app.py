from ultralytics import YOLO
import cv2



model = YOLO("models/baloon_model.pt")

print(model.names)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera didn't Open.")
        return
    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        results = model(frame)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2), (0,0,255), 2)
            cv2.putText(frame, f"Baloon {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,0,255),2)
        cv2.imshow("Baloon detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
