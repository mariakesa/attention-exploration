import torch
from collections import OrderedDict
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

class ImageLoader:
    def __init__(self, image_folder_path, image_processor, num_images):
        self.image_folder_path = image_folder_path
        self.dataset = load_dataset("imagefolder", data_dir=image_folder_path)
        self.image_processor = image_processor
        self.num_images = num_images
        self.processed_images = []
        self.original_images = []

    def load_images(self):
        for i in tqdm(range(self.num_images)):
            image = self.dataset["train"]["image"][i]
            self.original_images.append(image)
            inputs = self.image_processor(image, return_tensors="pt")
            self.processed_images.append(inputs)
        return self.processed_images


class ModuleHook:
    def __init__(self, module):
        self.module = module
        self.features = []
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features.append(output.detach())

    def close(self):
        self.hook.remove()

class FeatureExtractor:
    def __init__(self, model, adjust_bias_term=False):
        self.model = model
        self.adjust_bias_term = adjust_bias_term
        self.all_attentions = {}
        self.all_output = {}

    def extract_features(self, processed_images):
        for idx, inputs in enumerate(tqdm(processed_images)):
            features = OrderedDict()

            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear) and ("key" in name or "query" in name):
                    features[name] = ModuleHook(module)

            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

            for feature in features.values():
                feature.close()

            self.all_attentions[idx] = outputs.attentions

            # Process features per layer and head
            self.process_layer_features(features, idx)
        return self.all_output

    def process_layer_features(self, features, idx):
        for layer in range(12):
            if layer not in self.all_output:
                self.all_output[layer] = {}
            layer_query_name = f"encoder.layer.{layer}.attention.attention.query"
            layer_key_name = f"encoder.layer.{layer}.attention.attention.key"

            if self.adjust_bias_term:
                raw_query_feature = (
                    features[layer_query_name].features[0].cpu().numpy()[0, :]
                    - self.model.encoder.layer[layer].attention.attention.query.bias.cpu().numpy()
                )
                raw_key_feature = (
                    features[layer_key_name].features[0].cpu().numpy()[0, :]
                    - self.model.encoder.layer[layer].attention.attention.key.bias.cpu().numpy()
                )
            else:
                raw_query_feature = features[layer_query_name].features[0].cpu().numpy()[0, :]
                raw_key_feature = features[layer_key_name].features[0].cpu().numpy()[0, :]

            layer_query_feature = np.array(np.split(raw_query_feature, 12, axis=-1))
            layer_key_feature = np.array(np.split(raw_key_feature, 12, axis=-1))

            for head in range(12):
                combined_features = np.concatenate([layer_query_feature[head], layer_key_feature[head]])
                if head not in self.all_output[layer]:
                    self.all_output[layer][head] = combined_features
                else:
                    self.all_output[layer][head] = np.vstack([self.all_output[layer][head], combined_features])


class TransformerPipeline:
    def __init__(self, paths, image_processor, model, num_images, adjust_bias_term=False):
        self.image_folder_path = paths["image_folder_path"]
        self.dataset = load_dataset("imagefolder", data_dir=self.image_folder_path)
        self.image_processor = image_processor
        self.model = model
        self.num_images = num_images
        self.adjust_bias_term = adjust_bias_term

    def run(self):
        # Step 1: Load and preprocess images
        image_loader = ImageLoader(self.dataset, self.image_processor, self.num_images)
        processed_images = image_loader.load_images()

        # Step 2: Extract features
        feature_extractor = FeatureExtractor(self.model, self.adjust_bias_term)
        all_output = feature_extractor.extract_features(processed_images)

        # Step 3: Reduce dimensions
        dimension_reducer = DimensionReducer(all_output)
        vit_embeddeds = dimension_reducer.reduce_dimensions()

        # You can add visualization or saving steps here

        return vit_embeddeds
