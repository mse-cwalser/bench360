import os
import json
import base64
from benchmark.tasks.base_task import BaseTask


class SimpleVisionTask(BaseTask):
    api_type = "chat_completion"

    def __init__(self, dataset_dir="benchmark/data/vision"):
        self.dataset_dir = dataset_dir
        self.dataset_path = os.path.join(dataset_dir, "vision_dataset.json")
        self.dataset = []

        # Ensure the data directory exists
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Load the dataset if it exists, otherwise create a demanding template
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
        else:
            self._create_demanding_template()

    def _create_demanding_template(self):
        print(f"[WARN] No vision dataset found. Generating template at {self.dataset_path}")
        print(">>> For a demanding benchmark, copy your high-res images to the folder above. <<<")

        # Create the dummy fallback image so the script doesn't crash on the first run
        dummy_path = os.path.join(self.dataset_dir, "dummy_red.png")
        img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        if not os.path.exists(dummy_path):
            with open(dummy_path, "wb") as f:
                f.write(base64.b64decode(img_b64))

        # Template for demanding VLMs tasks
        self.dataset = [
            {
                "image": "chart.png",
                "prompt": "Analyze this chart and tell me the exact value of the highest bar. Respond with only the number.",
                "reference": "42"
            },
            {
                "image": "receipt.jpg",
                "prompt": "Extract the total amount including tax from this receipt. Return only the final price.",
                "reference": "124.50"
            },
            {
                "image": "math.png",
                "prompt": "Solve the integral shown in this image. State the final simplified answer.",
                "reference": "pi/2"
            },
            {
                "image": "crowd.jpg",
                "prompt": "How many red cars are visible in this street scene? Return only the number.",
                "reference": "3"
            },
            {
                "image": "dummy_red.png",
                "prompt": "What color is the main subject in this image? Answer in one word.",
                "reference": "red"
            }
        ]

        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=4)

    def generate_prompts(self, num_examples: int) -> tuple[list, list[str]]:
        prompts = []
        refs = []

        count = max(1, num_examples if num_examples else len(self.dataset))

        # Filter dataset to ONLY include images that actually exist on your disk
        # This prevents crashes if you haven't copied all the template images yet.
        valid_items = []
        for item in self.dataset:
            img_path = os.path.join(self.dataset_dir, item["image"])
            if os.path.exists(img_path):
                valid_items.append((item, img_path))

        if not valid_items:
            raise FileNotFoundError(
                f"No valid images found in {self.dataset_dir}. "
                f"Please add at least one image listed in {self.dataset_path}."
            )

        for i in range(count):
            item, img_path = valid_items[i % len(valid_items)]

            prompts.append({
                "messages": item["prompt"],
                "images": img_path
            })
            refs.append(item["reference"])

        return prompts, refs

    def quality_metrics(self, generated: str, reference: str) -> dict[str, float]:
        # Exact match is often too strict for LLMs, so we check if the reference is *in* the generated text.
        is_correct = 1.0 if reference.lower() in generated.lower() else 0.0
        return {"accuracy": is_correct}