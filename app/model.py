from ultralytics import YOLO
import torch

class SpeechClassifier:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.model.fuse()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self, image):
        """
        Returns:
            dict: {class_name: probability}
        """
        results = self.model(
            image,
            device=self.device,
            verbose=False
        )

        probs = results[0].probs  # ultralytics.engine.results.Probs
        class_names = results[0].names

        return {
            class_names[i]: float(probs.data[i])
            for i in range(len(class_names))
        }
