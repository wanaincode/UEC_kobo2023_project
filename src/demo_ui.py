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
process_flag_color = [(0, 255, 0)]

IMG_NUM_PER_D = 1  
SAVE_DURATION = 20 * IMG_NUM_PER_D * 3 + 5
SHOW_TIME_BY_FRAME = SAVE_DURATION + 10


intro_text = ["", "", ""]
save_meta = {
    "save_duration": 0,
    "save_frame_gap": 20
}
base_features = [None]
a = [np.zeros((720, 1280, 3)).astype(np.uint8)]

def show_xy(event, x, y, flags, userdata):
    global abstract_flag
    global recognize_flag
    global exit_program
    global input_key_in
    # reply(event, x, y)
    if event == cv2.EVENT_LBUTTONDOWN and x >= x1_right_side1 and y <= y2_right_side1:
        abstract_flag = True
    elif event == cv2.EVENT_LBUTTONDOWN and x >= x1_right_side2 and y <= y2_right_side2:
        recognize_flag = True
    elif event == cv2.EVENT_LBUTTONDOWN and x >= x1_right_side3 and y <= y2_right_side3:
        exit_program = True
    elif event == cv2.EVENT_LBUTTONDOWN and x >= x1_enter_bottom and y <= y2_enter_bottom:
        input_key_in = True

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
    print("Entering extract_saved_obj_features")
    input_path = "./saved_obj_img"
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
            # mask 
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
            
            img_pad_rtop = img_pad.copy()
            img_pad_ltop = img_pad.copy()
            img_pad_rdown = img_pad.copy()
            img_pad_ldown = img_pad.copy()

            # righttop covered mask
            img_pad_rtop[0:112, 112:224, : ] = (0, 0, 0)
            img_pad_ltop[0:112, 0:112, : ] = (0, 0, 0)
            img_pad_rdown[112:224, 112:224, : ] = (0, 0, 0)
            img_pad_ldown[112:224, 0:112, : ] = (0, 0, 0)
                
            if not osp.exists(osp.join(output_path, obj_dir, "covered")):
                os.mkdir(osp.join(output_path, obj_dir, "covered"))
            output_covered_path = osp.join(output_path, obj_dir, "covered")

            cv2.imwrite(osp.join(output_path, obj_dir, "mask_crop_"+obj), cv2.cvtColor(img_pad, cv2.COLOR_RGB2BGR))
            cv2.imwrite(osp.join(output_covered_path, "mask_crop_rtop_"+obj), cv2.cvtColor(img_pad_rtop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(osp.join(output_covered_path, "mask_crop_ltop_"+obj), cv2.cvtColor(img_pad_ltop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(osp.join(output_covered_path, "mask_crop_rdown_"+obj), cv2.cvtColor(img_pad_rdown, cv2.COLOR_RGB2BGR))
            cv2.imwrite(osp.join(output_covered_path, "mask_crop_ldown_"+obj), cv2.cvtColor(img_pad_ldown, cv2.COLOR_RGB2BGR))


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

    process_flag[0] = "Waiting"
    process_flag_color[0] = (0, 255, 0)

    return obj_features

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
                # print all
                if not rti_idx == max_score_idx:
                    cv2.putText(img_rgb_copy, rti, (x1+6, y1_t+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                else:
                    cv2.putText(img_rgb_copy, rti, (x1+6, y1_t+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                y1_t += 40

                # # print max score only
                # if rti_idx == max_score_idx:
                #     cv2.putText(img_rgb_copy, rti, (x1+6, y1_t+40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                # cv2.rectangle(img_rgb_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.imwrite(f"./recognized_results/res_{box_threshold}_{text_threshold}.jpg", cv2.cvtColor(img_rgb_copy, cv2.COLOR_RGB2BGR))
        img_bgr_copy = cv2.cvtColor(img_rgb_copy, cv2.COLOR_RGB2BGR)
        a[0] = img_bgr_copy.copy()
        print("Recognizition finished.")

    else:
        print("No test image")
    
    process_flag[0] = "Waiting"
    process_flag_color[0] = (0, 255, 0)

# load mode
print("Loading model...")
model, predictor, extractor = load_model_and_predict()

# create window
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.namedWindow("results", cv2.WINDOW_NORMAL)

# use webcam
print("Open camera...")
cap = cv2.VideoCapture(1)

image_num = 0
obj_idx = 0
obj_name = []
direction = ["x", "y", "z"]
dir_i = 0

input_message = [f" Object {obj_idx + 1}: "]

user_input = ""
exit_program = False
abstract_flag = False
recognize_flag = False
input_key_in = False
update_preview = False


img_paths = ["img0_path", "img1_path", "img2_path", "img3_path", "img4_path", "img5_path", "img6_path"]



# capture frame and show in window
while True:
    ret, frame = cap.read()
    if ret:
        frame_copy = frame.copy()
    
    # show process flag on right top of the window
    cv2.putText(frame_copy, process_flag[0], (630, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, process_flag_color[0], 2)

    # left-side
    cv2.rectangle(frame_copy, (0, 0), (180, 750), (0,0,0), -1)
    
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
    cv2.putText(frame_copy, "Enter", (x1_enter_bottom + 15, y1_enter_bottom + 35), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255, 255, 255), 2)

    # input msg & user-input
    cv2.putText(frame_copy, input_message[0], (500, 580), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_copy, user_input, (550, 635), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 255, 255), 2)

    cv2.setMouseCallback('window', show_xy)  # mouse callback function


    # object image path
    input_path = "./saved_obj_img/"
    if not osp.exists(input_path):
        os.mkdir(input_path)

    if save_meta["save_duration"] > 0 and dir_i <= 2:
        save_meta["save_duration"] -= 1

        obj_path = osp.join(input_path, f"{obj_name[obj_idx]}")
        if not osp.exists(obj_path):
            os.mkdir(obj_path)        
    
        if save_meta["save_duration"] % (20 * IMG_NUM_PER_D) == 0:
            print(f"Starting caputuring {IMG_NUM_PER_D} pictures of {direction[dir_i]} direction")
        if save_meta["save_duration"] % save_meta["save_frame_gap"] == 0:
            cv2.waitKey(300)
            print(f"{obj_name[obj_idx]}: {direction[dir_i]}_{image_num % IMG_NUM_PER_D}.jpg saved.")
            cv2.imwrite(f"{obj_path}/{direction[dir_i]}_{image_num % IMG_NUM_PER_D}.jpg", frame)
            image_num += 1
            if image_num % IMG_NUM_PER_D == 0:
                dir_i += 1
        
    else:
        if image_num == IMG_NUM_PER_D * 3:
            print(f"Images of {obj_name[obj_idx]} are saved.")

            #update counters
            obj_idx += 1
            image_num = 0
            save_meta["save_duration"] = 0
            dir_i = 0

            # show next input msg
            input_message[0] = f"Object {obj_idx + 1}: "

            # update process flag
            process_flag[0] = "Waiting"
            process_flag_color[0] = (0, 255, 0)

            # update preview images
            image_name = f"{direction[-1]}_{image_num % IMG_NUM_PER_D}.jpg"

            if len(obj_name) > 6:
                for i in range(6):
                    img_paths[5-i] = os.path.join(input_path, obj_name[-1-i], image_name)
            else:
                for i in range(len(obj_name)): 
                    img_paths[i] = os.path.join(input_path, obj_name[i], image_name)


    # show preview image
    width = 150
    x_offset, y_offset = 10, 30
    for i, path in enumerate(img_paths):
        if os.path.exists(path):
            image = cv2.imread(path)
            image_h, image_w, _ = image.shape
            scale = int(image_w / width)
            image_h, image_w = int(image_h / scale), int(image_w / scale)
            image = cv2.resize(image, (image_w, image_h))
            frame_copy[y_offset + i * (image_h + 10): y_offset + (i+1) * image_h + i * 10, x_offset: x_offset + image_w] = image
    
    cv2.imshow("window", frame_copy)
    cv2.imshow("results", a[0])

    # key
    key = cv2.waitKey(5) & 0xFF

    # if press Esc, quit the program
    if (key == 27 or exit_program) and process_flag[0] == "Waiting":
        break

    # show user input
    if key == 8 or key == 127:  # Backspace
        user_input = user_input[:-1]

    # if press Enter, capture the object
    if (key == 13 or input_key_in) and save_meta["save_duration"] <= 0:
        if(user_input != ""):
            input_message[0] = ""
            obj_name.append(user_input)
            user_input = ""
            process_flag[0] = "Running"
            process_flag_color[0] = (0, 0, 255)
            save_meta["save_duration"] = SAVE_DURATION
        else:
            input_message[0] = "Please enter the object name:"

    # typing
    if 32 <= key <= 126 and process_flag[0] == "Waiting": 
        user_input += chr(key)    

    # if press Extract, start a new process to extract the feature of saved images
    if abstract_flag and process_flag[0] == "Waiting":
        print("Starting extract_saved_obj_features thread")
        process_flag[0] = "Extracting"
        process_flag_color[0] = (0, 0, 255)
        thread = threading.Thread(target=extract_saved_obj_features, args=(model, predictor, extractor))
        thread.start()
        abstract_flag = False
        print("Thread started")

    # if press Recognize, excute clip_test.
    if recognize_flag and process_flag[0] == "Waiting": 
        process_flag[0] = "Recognizing"
        process_flag_color[0] = (0, 0, 255)
        obj_features = torch.load("./extract_saved_obj_feature/obj_features.pt")
        cv2.imwrite("./test_img/test.jpg", frame)
        thread = threading.Thread(target=recognize_pipeline, args=(model, predictor, extractor, obj_features, f"./test_img/test.jpg", 0.15, 0.15))
        thread.start()
        recognize_flag = False

cv2.destroyAllWindows()
cap.release()