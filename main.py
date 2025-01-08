from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
from autodistill.utils import plot
import cv2

base_model = GroundedSAM(ontology=CaptionOntology({"teal ball": "algae", "white plastic pipe": "coral"}))

results = base_model.predict("/kaggle/input/frc-8816-2025-v1/imgs/img_001.jpg")

plot(
    image=cv2.imread("/kaggle/input/frc-8816-2025-v1/imgs/img_001.jpg"),
    classes=base_model.ontology.classes(),
    detections=results
)

base_model.label(
    input_folder="/kaggle/input/frc-8816-2025-v1/imgs",
    extension=".jpg",
    output_folder="/kaggle/working/dataset-v1-labeled"
)

target_model = YOLOv8("yolov8n.pt")
target_model.train("/kaggle/working/dataset-v1-labeled/data.yaml", epochs=200)