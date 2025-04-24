import cv2
import threading
import clip, os, sys
import torch
from PIL import Image
import shutil
from tqdm import tqdm
import os.path as osp
import numpy as np
from torch.nn.functional import cosine_similarity
from torchvision.ops import box_convert, box_iou
import matplotlib.pyplot as plt

sys.path.append("./GroundingDINO/")
sys.path.append("segment-anything")
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import sam_model_registry, SamPredictor


TEXT_PROMPT_CAPTURE = "object on the hand"
TEXT_PROMPT_RECOGNIZE = "object"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25


# key value
# q: quit
# s: save image
# a: start a new process to extract the feature of saved images
# r: recognize one object

'''
press a: obj_idx will add 1
'''

process_flag = ["Waiting"]
IMG_NUM_PER_D = 10  
SAVE_DURATION = 20 * IMG_NUM_PER_D * 3 + 5
SHOW_TIME_BY_FRAME = SAVE_DURATION + 10

show_text = ["Press [s] to capture images of objects.", 0]
show_text_2 = ["Press [q] to quit."]
show_text_3 = [" "]

intro_text = ["", "", ""]
save_meta = {
    "save_duration": 0,
    "save_frame_gap": 20
}
base_features = [None]
a = [np.zeros((720, 1280, 3)).astype(np.uint8)]


def nms(bboxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = torch.argsort(-scores)
    indices = torch.arange(bboxes.shape[0])
    keep = torch.ones_like(indices, dtype=torch.bool)
    for i in indices:
        if keep[i]:
            bbox = bboxes[order[i]]
            iou = box_iou(bbox[None,...],(bboxes[order[i + 1:]]) * keep[i + 1:][...,None])
            overlapped = torch.nonzero(iou > iou_threshold)
            keep[overlapped + i + 1] = 0
    return order[keep]

def getJetColorRGB(v, vmin, vmax):
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)): 
        c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5                      
    return c

def ground_dino_predict(model, img_path, text_prompt, box_threshold=0.35, text_threshold=0.25, topK=10):
    image_source, image = load_image(img_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    # nms
    keep = nms(boxes, logits, 0.4)
    boxes = boxes[keep]
    logits = logits[keep]

    print("Predicted boxes:", boxes.shape[0])

    return boxes, logits

def sam_predict(predictor, img_path, boxes):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_rgb = image.copy()
    predictor.set_image(image)

    # get from object detector
    h, w, _ = image.shape
    boxes = boxes * torch.Tensor([w, h, w, h], device=boxes.device)
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

    input_boxes = xyxy.to(predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    return masks, xyxy, img_rgb

def ground_dino_sam_predict(model, predictor, img_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
    boxes, logits = ground_dino_predict(model, img_path, text_prompt, box_threshold, text_threshold)
    masks, boxes, img_rgb = sam_predict(predictor, img_path, boxes)
    return masks, boxes, img_rgb

def load_model_and_predict():
    #Use GroundingDino to detect items.
    print("Loading GroundingDINO model...")
    model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth")

    # Use SAM to generate the mask.
    print("Loading Segment Anything model...")
    sam_checkpoint = "segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device) 
    predictor = SamPredictor(sam)

    print("Loading clip ViT-B/32 model...")
    extractor, _ = clip.load("ViT-B/32", device, jit=False)

    return model, predictor, extractor

def extract_saved_obj_features(model, predictor, extractor):
    output_path = "./extract_saved_obj_feature"
    if not osp.exists(output_path):
        os.mkdir(output_path)
    else:
        # remove all files
        shutil.rmtree(output_path)
        os.mkdir(output_path)

    obj_list = [oi for oi in os.listdir(input_path) if not "DS_Store" in oi]
    obj_list.sort()
    
    obj_features = []

    for obj_dir in tqdm(obj_list):
        obj_path = osp.join(input_path, obj_dir)
        features_all = []
        if not osp.exists(osp.join(output_path, obj_dir)):
            os.mkdir(osp.join(output_path, obj_dir))
        for obj in [oi for oi in os.listdir(obj_path) if not "DS_Store" in oi]:
            img_path = osp.join(obj_path, obj)

            # ground dino and sam inference
            masks, boxes, img_rgb = ground_dino_sam_predict(model, predictor, img_path, TEXT_PROMPT_CAPTURE)
            x1, y1, x2, y2 = boxes[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # mask
            masked_img = img_rgb * masks[0].cpu().numpy().transpose(1, 2, 0)
            cv2.imwrite(osp.join(output_path, obj_dir, "mask_"+obj), cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

            # crop and pad to center
            masked_img = masked_img[y1:y2, x1:x2, :]
            h, w, _ = masked_img.shape
            size = max(h, w)
            img_pad = np.zeros((size, size, 3)).astype(np.uint8)
            img_pad[size//2-h//2:size//2+(h-h//2), size//2-w//2:size//2+(w-w//2)] = masked_img
            img_pad = cv2.resize(img_pad, (224, 224))
            cv2.imwrite(osp.join(output_path, obj_dir, "mask_crop_"+obj), cv2.cvtColor(img_pad, cv2.COLOR_RGB2BGR))

            # normalize
            image = img_pad.astype(np.float32) / 255.
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cpu()
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(image.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(image.device)
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)

            image = (image - mean) / std

            features = extractor.encode_image(image)
            features_all.append(features)

        features_all = torch.cat(features_all, dim=0)
        obj_features.append(features_all.unsqueeze(0))

    obj_features = torch.cat(obj_features, dim=0)
    obj_features = {
        "features": obj_features,
        "obj_list": obj_list
    }
    torch.save(obj_features, osp.join(output_path, "obj_features.pt"))
    print("saved obj_features.pt!!")
    _set_show_text("Press [r] to recognize test-images.")
    _set_show_text_2("Press [q] to quit.")
    _set_show_text_3("")
    process_flag[0] = "Waiting" 


def recognize_pipeline(model, predictor, extractor, obj_features, recognize_img_path, box_threshold=0.35, text_threshold=0.25, idx=0):
    obj_list = obj_features["obj_list"]
    obj_features = obj_features["features"]
    
    if os.path.exists(recognize_img_path):
        print(f"test image: {recognize_img_path}")
        
        if not osp.exists(f"recognized_results"):
            os.mkdir(f"recognized_results")
        else:
            # remove all files
            shutil.rmtree(f"recognized_results")
            os.mkdir(f"recognized_results")

        # ground dino and sam inference
        masks, boxes, img_rgb = ground_dino_sam_predict(model, predictor, recognize_img_path, TEXT_PROMPT_RECOGNIZE, box_threshold, text_threshold)
        img_rgb_copy = img_rgb.copy()

        for obj_idx, (boxi, maski) in enumerate(zip(boxes, masks)):
            x1, y1, x2, y2 = boxi.cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            area = (x2-x1) * (y2-y1)
            if area > 1000*1000:
                continue

            # mask
            masked_img = img_rgb * maski.cpu().numpy().transpose(1, 2, 0)
            cv2.imwrite(f"./recognized_results/mask_{obj_idx}.jpg", cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

            # crop and pad to center
            masked_img = masked_img[y1:y2, x1:x2, :]
            h, w, _ = masked_img.shape
            size = max(h, w)
            img_pad = np.zeros((size, size, 3)).astype(np.uint8)
            img_pad[size//2-h//2:size//2+(h-h//2), size//2-w//2:size//2+(w-w//2)] = masked_img
            img_pad = cv2.resize(img_pad, (224, 224))
            cv2.imwrite(f"./recognized_results/mask_crop_{obj_idx}.jpg", cv2.cvtColor(img_pad, cv2.COLOR_RGB2BGR))

            # normalize
            image = img_pad.astype(np.float32) / 255.
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cpu()
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(image.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(image.device)
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)

            image = (image - mean) / std

            # extract features
            features = extractor.encode_image(image)  # (1, 512)

            similarity = cosine_similarity(features, obj_features, dim=-1)
            sim_order = torch.argmax(similarity, dim=-1)

            res_text = []
            max_score_idx = 0
            max_score = 0
            for i in range(len(sim_order)):
                sim_t = similarity[i][sim_order[i]]
                if sim_t > max_score:
                    max_score = sim_t
                    max_score_idx = i
                res_text.append(f"{obj_list[i]}: {sim_t:.3f}")

            # write to image
            y1_t = y1
            for rti_idx, rti in enumerate(res_text):
                if not rti_idx == max_score_idx:
                    cv2.putText(img_rgb_copy, rti, (x1+6, y1_t+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                else:
                    cv2.putText(img_rgb_copy, rti, (x1+6, y1_t+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                y1_t += 40
            cv2.rectangle(img_rgb_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.imwrite(f"./recognized_results/res_{box_threshold}_{text_threshold}.jpg", cv2.cvtColor(img_rgb_copy, cv2.COLOR_RGB2BGR))
        img_bgr_copy = cv2.cvtColor(img_rgb_copy, cv2.COLOR_RGB2BGR)
        a[0] = img_bgr_copy.copy()
        print("Recognizition finished.")
    else:
        print("No test image")
    
    process_flag[0] = "Waiting"
    _set_show_text("Press [s] to capture more objects' images.")
    _set_show_text_2("Press [r] to recognize again.")
    _set_show_text_3("Press [q] to quit.")

def _set_show_text(text):
    show_text[0] = text
    show_text[1] = SHOW_TIME_BY_FRAME

def _set_show_text_2(text):
    show_text_2[0] = text

def _set_show_text_3(text):
    show_text_3[0] = text

# load mode
print("Loading model...")
model, predictor, extractor = load_model_and_predict()

# create window
cv2.namedWindow("results", cv2.WINDOW_NORMAL)
cv2.moveWindow("results", 200, 50)
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.moveWindow("window", 100, 250)

# use webcam
print("Open camera...")
cap = cv2.VideoCapture(1)

image_num = 0
obj_idx = 0
obj_name = []
direction = ["x", "y", "z"]
i = 0

# capture frame and show in window
while True:
    ret, frame = cap.read()
    if ret:
        frame_copy = frame.copy()
    # show process flag on right top of the window
    cv2.putText(frame_copy, process_flag[0], (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # show action flag below process flag
    cv2.putText(frame_copy, show_text[0], (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_copy, show_text_2[0], (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_copy, show_text_3[0], (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if show_text[1] > 0:
        show_text[1] -= 1
        cv2.putText(frame_copy, show_text[0], (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    #object image path
    input_path = "./saved_obj_img/"
    if not osp.exists(input_path):
        os.mkdir(input_path)

    if save_meta["save_duration"] > 0 and i <= 2:
        save_meta["save_duration"] -= 1

        obj_path = osp.join(input_path, f"{obj_name[obj_idx]}")
        if not osp.exists(obj_path):
            os.mkdir(obj_path)        
        
        if save_meta["save_duration"] % (20 * IMG_NUM_PER_D) == 0:
            print(f"Starting caputuring {IMG_NUM_PER_D} pictures of {direction[i]} direction")
        if save_meta["save_duration"] % save_meta["save_frame_gap"] == 0:
            cv2.waitKey(300)
            print(f"{obj_name[obj_idx]}: {direction[i]}_{image_num % IMG_NUM_PER_D}.jpg saved.")
            cv2.imwrite(f"{obj_path}/{direction[i]}_{image_num % IMG_NUM_PER_D}.jpg", frame)
            image_num += 1
            if image_num % IMG_NUM_PER_D == 0:
                i += 1
    else:
        if image_num == IMG_NUM_PER_D * 3:
            print(f"Images of {obj_name[obj_idx]} are saved.")
            _set_show_text("Press [s] again to capture more objects.")
            _set_show_text_2("Press [a] to extract the features.")
            process_flag[0] = "Waiting"
            obj_idx += 1
            image_num = 0
            save_meta["save_duration"] = 0
            i = 0

    cv2.imshow("window", frame_copy)
    cv2.imshow("results", a[0])

    # key
    key = cv2.waitKey(5)

    # press q to quit
    if key == ord("q") and process_flag[0] == "Waiting":
        break

    # if key is 's' pressed, save image
    if key == ord("s") and save_meta["save_duration"] <= 0:
        user_input = input(f"Enter the name of object {obj_idx+1}: ")
        obj_name.append(user_input)
        _set_show_text(f"Capturing {IMG_NUM_PER_D * 3} images of {obj_name[obj_idx]} in {SAVE_DURATION-5} frames.")
        _set_show_text_2(" ")
        _set_show_text_3(" ")
        process_flag[0] = "Running"
        save_meta["save_duration"] = SAVE_DURATION
   
    # if key is 'a' pressed, start a new process to extract the feature of saved images
    if key == ord("a") and process_flag[0] == "Waiting":
        _set_show_text(" ")
        _set_show_text_2(" ")
        _set_show_text_3(" ")
        process_flag[0] = "Extracting"
        thread = threading.Thread(target=extract_saved_obj_features, args=(model, predictor, extractor))
        thread.start()

    # if ket is 'r' pressed, excute clip_test.
    if key == ord("r") and process_flag[0] == "Waiting": 
        _set_show_text("Test-image was captured. Please wait for recognized result.")
        _set_show_text_2(" ")
        _set_show_text_3(" ")
        process_flag[0] = "Recognizing"
        obj_features = torch.load("./extract_saved_obj_feature/obj_features.pt")
        cv2.imwrite("./test_img/test.jpg", frame)
        thread = threading.Thread(target=recognize_pipeline, args=(model, predictor, extractor, obj_features, f"./test_img/test.jpg", 0.15, 0.15))
        thread.start()

cv2.destroyAllWindows()
cap.release()