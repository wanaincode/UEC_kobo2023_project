import numpy as np
import cv2


# create window
cv2.namedWindow("window", cv2.WINDOW_NORMAL)

# capture video
cap = cv2.VideoCapture(0)

# define process flag
process_flag = ['Processing', 'Processed']

# define action flag
show_text = ['First', 'Second', 'Third']

input_message = ["Object "]

user_input = ""
exit_program = False
abstract_flag = False
recognize_flag = False
input_key_in = False


def show_xy(event, x, y, flags, userdata):
    global exit_program
    # reply(event, x, y)
    if event == cv2.EVENT_LBUTTONDOWN and x >= x1_right_side1 and y <= y2_right_side1:
        abstract_flag = True
    elif event == cv2.EVENT_LBUTTONDOWN and x >= x1_right_side2 and y <= y2_right_side2:
        recognize_flag = True
    elif event == cv2.EVENT_LBUTTONDOWN and x >= x1_right_side3 and y <= y2_right_side3:
        exit_program = True
    elif event == cv2.EVENT_LBUTTONDOWN and x >= x1_enter_bottom and y <= y2_enter_bottom:
        input_key_in = True

# capture frame and show in window
while True:
    ret, frame = cap.read()
    if ret:
        frame_copy = frame.copy()
    
    # left-side
    cv2.rectangle(frame_copy, (0, 0), (170, 750), (0,0,0), -1)
    

    img = cv2.imread("./saved_obj_img/a/x_0.jpg")
    
    img_h, img_w, _ = img.shape
    x, y = 10, 30
    width = 150
    scale = img_w / width
    img_h, img_w = int(img_h / scale), int(img_w / scale)
    img = cv2.resize(img, (img_w, img_h))
    x_offset = 50
    y_offset = 50
    frame_copy[y_offset:y_offset + img_h, x_offset:x_offset + img_w] = img
    
    # right-side
    x1_right_side1, y1_right_side1 = 1100, 240
    x2_right_side1, y2_right_side1 = x1_right_side1 + 145, y1_right_side1 + 50
    cv2.rectangle(frame_copy, (x1_right_side1, y1_right_side1), (x2_right_side1, y2_right_side1), (255, 255, 255), 2)
    cv2.putText(frame_copy, "Abstract", (x1_right_side1 + 20, y1_right_side1 + 35), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 255, 255), 2)
    
    x1_right_side2, y1_right_side2 = 1100, 300
    x2_right_side2, y2_right_side2 = x1_right_side2 + 145, y1_right_side2 + 50
    cv2.rectangle(frame_copy, (x1_right_side2, y1_right_side2), (x2_right_side2, y2_right_side2), (255, 255, 255), 2)
    cv2.putText(frame_copy, "Recognize", (x1_right_side2 + 10, y1_right_side2 + 35), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 255, 255), 2)
    
    x1_right_side3, y1_right_side3 = 1100, 360
    x2_right_side3, y2_right_side3 = x1_right_side3 + 145, y1_right_side3 + 50
    cv2.rectangle(frame_copy, (x1_right_side3, y1_right_side3), (x2_right_side3, y2_right_side3), (255, 255, 255), 2)
    cv2.putText(frame_copy, "Exit", (x1_right_side3 + 45, y1_right_side3 + 35), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 255, 255), 2)

    # key-in frame 
    x1_key_in, y1_key_in = 500, 600
    x2_key_in, y2_key_in = 800, 650
    cv2.rectangle(frame_copy, (x1_key_in, y1_key_in), (x2_key_in, y2_key_in), (255, 255, 255), 2)
    # enter bottom
    x1_enter_bottom, y1_enter_bottom = 810, 600
    x2_enter_bottom, y2_enter_bottom = 910, 650
    cv2.rectangle(frame_copy, (x1_enter_bottom, y1_enter_bottom), (x2_enter_bottom, y2_enter_bottom), (255, 255, 255), 2)
    cv2.putText(frame_copy, "Enter", (x1_enter_bottom + 10, y1_enter_bottom + 35), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 255, 255), 2)

    # input msg & user-input
    cv2.putText(frame_copy, input_message[0], (500, 580), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_copy, user_input, (550, 635), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 255, 255), 2)

     
    # show the frame
    cv2.imshow("window", frame_copy)
    cv2.setMouseCallback('window', show_xy)  # 設定偵測事件的函式與視窗

    
    # get user input
    key = cv2.waitKey(1) & 0xFF

    if key == 27 or exit_program:
        break
    elif key == 8 or key == 127:  # Backspace
        user_input = user_input[:-1]
    elif key == 13:  # Enter
        print(user_input)
    elif 32 <= key <= 126: 
        user_input += chr(key)



cv2.destroyAllWindows()
cap.release()