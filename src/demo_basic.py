import cv2
import threading
import clip, os
import torch
from PIL import Image

import clip, os
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity

print("Loading model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# key value
# q: quit
# s: save image
# a: start a new process to extract the feature of saved images
# r: recognize one object

'''
press a: obj_idx will add 1
'''

process_flag = ["waiting"]
SAVE_DURATION = 100
SHOW_TIME_BY_FRAME = SAVE_DURATION + 10

show_text = ["press [s-key] to capture images.", 0]
intro_text = ["", "", ""]
save_meta = {
    "save_duration": 0,
    "save_frame_gap": 20
}
base_features = [None]

def extract_feature(obj_idx):
    process_flag[0] = "running"
    img_list = os.listdir("saved_img")
    img_list = [os.path.join("saved_img", img) for img in img_list if "obj_"+str(obj_idx) in img]
    if len(img_list) == 0:
        process_flag[0] = "waiting"
    else:
        regist_img_list = [Image.open(img_i) for img_i in img_list]
        base_feature_all = []
        for img_i in regist_img_list:
            base_feature_all.append(model.encode_image(preprocess(img_i).unsqueeze(0)))
        base_feature_all = torch.cat(base_feature_all, dim=0)
        torch.save(base_feature_all, f"./saved_obj_feature/obj_{obj_idx}.pt")
        process_flag[0] = "waiting"

def test_img(img_path, base_feature_all):
    print(f"test image: {img_path}")
    img = Image.open(img_path)
    img_feature = model.encode_image(preprocess(img).unsqueeze(0))
    feature_i_repeat = img_feature.repeat(base_feature_all.shape[0], 1)
    similarity = cosine_similarity(feature_i_repeat, base_feature_all)
    max = 0
    max_idx = 0
    for i in range(obj_idx):
        obj_mean = _sum(similarity[i+0:i+5])/5
        if obj_mean > max:
            max = obj_mean
            max_obj_idx = i
        print(f"obj_{i}-test similarity: {obj_mean}")
    _set_show_text(f"This is {max*100:.2f}% similar to obj_{max_obj_idx}")
    return similarity

def _sum(arr):
    sum = 0
    for x in arr:
        sum = sum + x
    return(sum)

def generate_base_feature(obj_idx):
    process_flag[0] = "running"
    img_list = os.listdir("saved_img")
    img_list.sort()
    saved_img_list = [Image.open(os.path.join("saved_img", img_i)) for img_i in img_list]

    base_feature_all = []
    for img_i in saved_img_list:
        base_feature_all.append(model.encode_image(preprocess(img_i).unsqueeze(0)))
    base_feature_all = torch.cat(base_feature_all, dim=0)
    process_flag[0] = "waiting"
    base_features[0] = base_feature_all    

def recognize_img(obj_idx):
    if os.path.exists(f"./test_img/test.jpg"):
        test_img(f"test_img/test.jpg", base_features[0])
    else:
        print("no test image")

def read_saved_feature(obj_idx):
    feature = torch.load(f"./saved_obj_feature/obj_{obj_idx}.pt")
    return feature

def _set_show_text(text):
    show_text[0] = text
    show_text[1] = SHOW_TIME_BY_FRAME


# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print("Open camera...")

# create window
cv2.namedWindow("window", cv2.WINDOW_NORMAL)

# use webcam
cap = cv2.VideoCapture(1)

image_num = 0
obj_idx = 0

# capture frame and show in window
while True:
    ret, frame = cap.read()
    frame_copy = frame.copy()
    # show process flag on right top of the window
    cv2.putText(frame_copy, process_flag[0], (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # show action flag below process flag
    cv2.putText(frame_copy, show_text[0], (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if show_text[1] > 0:
        show_text[1] -= 1
        cv2.putText(frame_copy, show_text[0], (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if save_meta["save_duration"] > 0:
        save_meta["save_duration"] -= 1
        if save_meta["save_duration"] % save_meta["save_frame_gap"] == 0:
            cv2.imwrite(f"./saved_img/obj_{obj_idx}_{image_num}.jpg", frame)
            cv2.waitKey(150)
            image_num += 1
    else:
        if image_num == SAVE_DURATION // save_meta["save_frame_gap"]:
            print("save images finished.")
            # start a new thread    
            process_flag[0] = "extracting"
            thread = threading.Thread(target=extract_feature, args=(obj_idx,))
            thread.start()
            _set_show_text("press [s-key] again or [a-key] to extract the features.")
            obj_idx += 1
            image_num = 0

    cv2.imshow("window", frame_copy)

    # key
    key = cv2.waitKey(5)

    # press q to quit
    if key == ord("q") and process_flag[0] == "waiting":
        break


    # if key is 's' pressed, save image
    if key == ord("s") and save_meta["save_duration"] <= 0:
        _set_show_text(f"capturing {SAVE_DURATION//save_meta['save_frame_gap']} images in {SAVE_DURATION} frames.")
        process_flag[0] = "running"
        save_meta["save_duration"] = SAVE_DURATION

    # if key is 'a' pressed, start a new process to extract the feature of saved images
    if key == ord("a") and process_flag[0] == "waiting":
        _set_show_text("press [r-key] to recognize test-images.")
        thread = threading.Thread(target=generate_base_feature, args=(obj_idx,))
        thread.start()

    # if ket is 'r' pressed, excute clip_test.
    if key == ord("r") and process_flag[0] == "waiting": 
        cv2.imwrite(f"./test_img/test.jpg", frame)
        cv2.waitKey(150)
        thread = threading.Thread(target=recognize_img, args=(obj_idx,))
        thread.start()

cv2.destroyAllWindows()
cap.release()