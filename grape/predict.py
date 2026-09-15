import cv2
import numpy as np
import onnxruntime as ort

# Список классов датасета
CLASS_NAMES = [
    'Grape__BlackRot', 
    'Grape__Esca', 
    'Grape__Healthy', 
    'Grape__LeafBlight'
]

# Цвета для отрисовки рамок 
COLORS = [
    (255, 0, 0),     # Синий - BlackRot
    (255, 255, 0),   # Голубой - Esca
    (0, 255, 0),     # Зеленый - Healthy
    (0, 165, 255)    # Оранжевый - LeafBlight
]

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Пропорциональное изменение размера с добавлением полей (padding),
    как это делает YOLOv8.
    """
    shape = img.shape[:2] # Текущая высота и ширина
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2 # Разделяем отступы на обе стороны
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)


def predict_onnx(model_path, image_path, conf_threshold=0.55, iou_threshold=0.45):
    # 1. Загрузка ONNX модели
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 2. Загрузка и предобработка изображения
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

    img, ratio, (pad_w, pad_h) = letterbox(orig_img, new_shape=(640, 640))
    
    # BGR в RGB, HWC в CHW, нормализация [0.0, 1.0]
    blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blob = blob.transpose((2, 0, 1)).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0) # [1, 3, 640, 640]

    # 3. Выполнение инференса через ONNX Runtime
    outputs = session.run([output_name], {input_name: blob})[0] # Форма: [1, 8, 8400]

    # 4. Постобработка (YOLOv8 output format: [batch, 4 + num_classes, anchors])
    predictions = np.squeeze(outputs).T # Транспонируем к размеру [8400, 8]

    # Координаты боксов (cx, cy, w, h) и вероятности классов
    boxes_cxcywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    # Находим класс с максимальной вероятностью для каждого бокса
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    # Маска для фильтрации по порогу уверенности (conf >= 0.55)
    mask = confidences >= conf_threshold
    boxes_cxcywh = boxes_cxcywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        print("Объекты с заданным порогом уверенности не найдены.")
        return orig_img, []

    # Преобразуем координаты из (cx, cy, w, h) в (x1, y1, x2, y2)
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    # Возвращаем координаты на исходное разрешение изображения (учитываем padding и масштаб)
    x1 = (x1 - pad_w) / ratio
    y1 = (y1 - pad_h) / ratio
    x2 = (x2 - pad_w) / ratio
    y2 = (y2 - pad_h) / ratio

    # Ограничиваем рамки размерами оригинала
    h_orig, w_orig = orig_img.shape[:2]
    x1 = np.clip(x1, 0, w_orig)
    y1 = np.clip(y1, 0, h_orig)
    x2 = np.clip(x2, 0, w_orig)
    y2 = np.clip(y2, 0, h_orig)

    max_wh = 4096.0
    c = class_ids * max_wh
    x1_offset = x1 + c
    y1_offset = y1 + c

    # 5. Применяем NMS (Non-Maximum Suppression)
    boxes_for_nms = np.stack([x1_offset, y1_offset, x2 - x1, y2 - y1], axis=1).tolist() # Формат OpenCV NMS: [x, y, w, h]
    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms, 
        confidences.tolist(), 
        conf_threshold, 
        iou_threshold
    )

    results = []
    # 6. Отрисовка финальных рамок
    if len(indices) > 0:
        for i in indices.flatten():
            box = [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])]
            score = float(confidences[i])
            cls_id = int(class_ids[i])
            cls_name = CLASS_NAMES[cls_id]
            color = COLORS[cls_id]

            # Рисуем прямоугольник
            cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Подпись с классом и уверенностью
            label = f"{cls_name}: {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(orig_img, (box[0], box[1] - text_h - 10), (box[0] + text_w, box[1]), color, -1)
            cv2.putText(orig_img, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            results.append({
                "class_name": cls_name,
                "confidence": score,
                "bbox": box
            })

    return orig_img, results


if __name__ == "__main__":
    MODEL_PATH = "best.onnx"            # Путь к вашей ONNX модели
    IMAGE_PATH = "test_leaf.jpg"        # Путь к тестовому изображению
    OUTPUT_PATH = "result_onnx.jpg"     # Куда сохранить результат

    # Запуск предсказания с фильтром confidence >= 0.55
    annotated_img, detections = predict_onnx(
        model_path=MODEL_PATH,
        image_path=IMAGE_PATH,
        conf_threshold=0.55
    )

    # Сохраняем результат
    cv2.imwrite(OUTPUT_PATH, annotated_img)
    print(f"Результат сохранен в {OUTPUT_PATH}")
    print("Найденные объекты:", detections)