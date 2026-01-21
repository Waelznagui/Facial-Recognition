import cv2
import os 


def add_from_camera():
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return  
    while True:
        ok, frame=cap.read()
        if not ok:
            print("Error: Could not read frame.")
            break
        cv2.imshow("Add to Database - Press 'a' to add, 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            name = input("Enter name for the new face: ")
            img_path = os.path.join("faces_db", f"{name}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved {img_path} to database.")
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def main():
    add_from_camera()
if __name__ == "__main__":
    main()
