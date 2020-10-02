import numpy as np
import cv2

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
def build_return(class_id, x, y, x_plus_w, y_plus_h):
    return str(class_id) + "," + str(x) + "," + str(y) + "," + str(x_plus_w) + "," + str(y_plus_h)
def process(img, scale, conf_threshold, nms_threshold, net):
    Width = img.shape[1]
    Height = img.shape[0]

    # Nhan dien bang YOLO

    blob = cv2.dnn.blobFromImage(img, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    retString = ""

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        # Xay dung chuoi tra ve client
        retString += build_return(class_ids[i], round(x), round(y), round(w), round(h)) + "|"

    return retString