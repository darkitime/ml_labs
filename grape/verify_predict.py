import numpy as np
from ultralytics import YOLO
from predict import predict_onnx  # Импортируем функцию из твоего predict.py

IMAGE_PATH = "test.jpg"  # Возьми любую валидационную картинку с листьями
PT_MODEL_PATH = "best.pt"
ONNX_MODEL_PATH = "best.onnx"
CONF_THRESH = 0.55
IOU_THRESH = 0.45

def calculate_iou(box1, box2):
    """Вычисляет IoU (Intersection over Union) между двумя рамками [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


print("=== 1. Прогон через PyTorch (Ultralytics) ===")
model_pt = YOLO(PT_MODEL_PATH)
results_pt = model_pt.predict(IMAGE_PATH, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]

pt_detections = []
for box in results_pt.boxes:
    cls_id = int(box.cls[0].item())
    conf = float(box.conf[0].item())
    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
    pt_detections.append({
        "class_name": model_pt.names[cls_id],
        "confidence": round(conf, 4),
        "bbox": xyxy
    })

print(f"PyTorch нашел объектов: {len(pt_detections)}")
for d in pt_detections:
    print("  ", d)


print("\n=== 2. Прогон через ONNX Runtime (predict.py) ===")
_, onnx_detections = predict_onnx(
    model_path=ONNX_MODEL_PATH,
    image_path=IMAGE_PATH,
    conf_threshold=CONF_THRESH,
    iou_threshold=IOU_THRESH
)

print(f"ONNX Runtime нашел объектов: {len(onnx_detections)}")
for d in onnx_detections:
    print("  ", d)


print("\n=== 3. Сравнение результатов ===")
if len(pt_detections) != len(onnx_detections):
    print("⚠️  ВНИМАНИЕ: Количество найденных объектов различается!")
else:
    print("✅ Количество найденных объектов совпадает!")

# Сравниваем координатную точность и уверенность рамок
for i, (pt_det, onnx_det) in enumerate(zip(pt_detections, onnx_detections)):
    iou = calculate_iou(pt_det['bbox'], onnx_det['bbox'])
    conf_diff = abs(pt_det['confidence'] - onnx_det['confidence'])
    
    print(f"\nОбъект #{i+1}:")
    print(f"  Класс PyTorch: {pt_det['class_name']} | Класс ONNX: {onnx_det['class_name']}")
    print(f"  Перекрытие рамок (IoU): {iou * 100:.2f}%")
    print(f"  Разница в Confidence: {conf_diff:.4f}")

    if pt_det['class_name'] == onnx_det['class_name'] and iou > 0.90 and conf_diff < 0.02:
        print("  Status: ✅ ИДЕАЛЬНОЕ СОВПАДЕНИЕ")
    else:
        print("  Status: ⚠️ Есть нестыковки")