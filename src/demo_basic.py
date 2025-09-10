import os
import cv2
import threading
import clip
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity

print("Loading model...")

# ---- paths (use data/ hierarchy) ----
DATA_DIR = "data"
SAVED_IMG_DIR = os.path.join(DATA_DIR, "saved_img")            # where captured registration images go
SAVED_FEAT_DIR = os.path.join(DATA_DIR, "saved_obj_feature")   # where per-object feature .pt files go
TEST_IMG_DIR  = os.path.join(DATA_DIR, "test_img")             # where test.jpg is saved

# create directories if missing
os.makedirs(SAVED_IMG_DIR, exist_ok=True)
os.makedirs(SAVED_FEAT_DIR, exist_ok=True)
os.makedirs(TEST_IMG_DIR,  exist_ok=True)

# load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# key value
# q: quit
# s: save image
# a: start a new process to extract the feature of saved images
# r: recognize one object

"""
press a: obj_idx will add 1
"""

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


def extract_feature(obj_idx: int):
    """Extract features from saved images for a specific object index and save as .pt."""
    process_flag[0] = "running"
    img_list = os.listdir(SAVED_IMG_DIR)
    img_list = [os.path.join(SAVED_IMG_DIR, img) for img in img_list if f"obj_{obj_idx}" in img]
    if len(img_list) == 0:
        process_flag[0] = "waiting"
    else:
        regist_img_list = [Image.open(img_i) for img_i in img_list]
        base_feature_all = []
        for img_i in regist_img_list:
            base_feature_all.append(model.encode_image(preprocess(img_i).unsqueeze(0)))
        base_feature_all = torch.cat(base_feature_all, dim=0)
        torch.save(base_feature_all, os.path.join(SAVED_FEAT_DIR, f"obj_{obj_idx}.pt"))
        process_flag[0] = "waiting"


def test_img(img_path: str, base_feature_all: torch.Tensor):
    """Compute similarity between test image and base features; show best match percentage."""
    print(f"test image: {img_path}")
    img = Image.open(img_path)
    img_feature = model.encode_image(preprocess(img).unsqueeze(0))
    feature_i_repeat = img_feature.repeat(base_feature_all.shape[0], 1)
    similarity = cosine_similarity(feature_i_repeat, base_feature_all)
    max_val = 0
    max_obj_idx = 0
    for i in range(obj_idx):
        obj_mean = _sum(similarity[i + 0:i + 5]) / 5
        if obj_mean > max_val:
            max_val = obj_mean
            max_obj_idx = i
        print(f"obj_{i}-test similarity: {obj_mean}")
    _set_show_text(f"This is {max_val * 100:.2f}% similar to obj_{max_obj_idx}")
    return similarity


def _sum(arr):
    s = 0
    for x in arr:
        s = s + x
    return s


def generate_base_feature(obj_idx: int):
    """Aggregate all saved images' features into a single base_features[0] tensor in memory."""
    process_flag[0] = "running"
    img_list = os.listdir(SAVED_IMG_DIR)
    img_list.sort()
    saved_img_list = [Image.open(os.path.join(SAVED_IMG_DIR, img_i)) for img_i in img_list]

    base_feature_all = []
    for img_i in saved_img_list:
        base_feature_all.append(model.encode_image(preprocess(img_i).unsqueeze(0)))
    base_feature_all = torch.cat(base_feature_all, dim=0)
    process_flag[0] = "waiting"
    base_features[0] = base_feature_all


def recognize_img(obj_idx: int):
    """Run recognition for the latest captured test image."""
    test_path = os.path.join(TEST_IMG_DIR, "test.jpg")
    if os.path.exists(test_path):
        test_img(test_path, base_features[0])
    else:
        print("no test image")


def read_saved_feature(obj_idx: int):
    """Read a previously saved per-object feature file."""
    feature = torch.load(os.path.join(SAVED_FEAT_DIR, f"obj_{obj_idx}.pt"))
    return feature


def _set_show_text(text: str):
    show_text[0] = text
    show_text[1] = SHOW_TIME_BY_FRAME


print("Open camera...")

# create window
cv2.namedWindow("window", cv2.WINDOW_NORMAL)

# use webcam
# Use camera index 0 by default; change to 1 if you have an external camera
cap = cv2.VideoCapture(0)

image_num = 0
obj_idx = 0

# capture frame and show in window
while True:
    ret, frame = cap.read()
    if not ret:
        # If frame not captured, continue trying (or break if you prefer)
        continue

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
            cv2.imwrite(os.path.join(SAVED_IMG_DIR, f"obj_{obj_idx}_{image_num}.jpg"), frame)
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
        _set_show_text(f"capturing {SAVE_DURATION // save_meta['save_frame_gap']} images in {SAVE_DURATION} frames.")
        process_flag[0] = "running"
        save_meta["save_duration"] = SAVE_DURATION

    # if key is 'a' pressed, start a new process to extract the feature of saved images
    if key == ord("a") and process_flag[0] == "waiting":
        _set_show_text("press [r-key] to recognize test-images.")
        thread = threading.Thread(target=generate_base_feature, args=(obj_idx,))
        thread.start()

    # if key is 'r' pressed, execute recognition
    if key == ord("r") and process_flag[0] == "waiting":
        cv2.imwrite(os.path.join(TEST_IMG_DIR, "test.jpg"), frame)
        cv2.waitKey(150)
        thread = threading.Thread(target=recognize_img, args=(obj_idx,))
        thread.start()

cv2.destroyAllWindows()
cap.release()