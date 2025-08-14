from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
from label_studio_tools.core.utils.io import get_local_path
import os

class NewModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(NewModel, self).__init__(**kwargs)

        # Load YOLO model (ultralytics v8 API)
        self.model = YOLO("/content/drive/MyDrive/yolo-training/best.pt")
        self.labels = self.model.names  # class names
        self.model_dir = "./model-data"

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            image_url = task['data']['image']
            print(task)
            print(image_url)
            local_path = get_local_path(image_url, task_id=task['id'])
            print("Exists: ", os.path.exists(local_path))
            print("SIZE: ", os.path.getsize(local_path))

            # Run inference
            results = self.model(local_path)  # returns list of Results objects

            for r in results:  # iterate over each Results object
                img_h, img_w = r.orig_img.shape[:2]

                output = []
                for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    x_min, y_min, x_max, y_max = box.tolist()
                    output.append({
                        "from_name": "label",  # name from labeling config
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": (x_min / img_w) * 100,
                            "y": (y_min / img_h) * 100,
                            "width": ((x_max - x_min) / img_w) * 100,
                            "height": ((y_max - y_min) / img_h) * 100,
                            "rectanglelabels": [self.labels[int(cls)]]
                        },
                        "score": float(conf)
                    })

                predictions.append({"result": output})

        return predictions

