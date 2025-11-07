import os
import pickle
import numpy as np
import torch
import faiss
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModelForCausalLM
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16, DenseNet121, MobileNetV2
from tensorflow.keras.models import Model, load_model as keras_load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class AquaGPT:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

        # Paths
        self.saved_documents_file = 'K:/ME Project/ME Project/1 New Fish Disease and AquaGPT/AquaGPT/AquaGPT Model/documents.pkl'
        self.saved_embeddings_file = 'K:/ME Project/ME Project/1 New Fish Disease and AquaGPT/AquaGPT/AquaGPT Model/document_embeddings.pkl'
        self.saved_faiss_index_file = 'K:/ME Project/ME Project/1 New Fish Disease and AquaGPT/AquaGPT/AquaGPT Model/faiss_index.pkl'

        # Classes
        self.classes = [
            "Bacterial diseases - Aeromoniasis",
            "Bacterial gill disease",
            "Bacterial Red disease",
            "Fungal diseases Saprolegniasis",
            "Healthy Fish",
            "Parasitic diseases",
            "Viral diseases White tail disease",
        ]

        # Placeholders
        self.documents = []
        self.faiss_index = None
        self.embeddings = None
        self.image_models = {}

        self.load_saved_data()
        self.initialize_models()

    # ---------- Image model helpers ----------
    def load_model(self, path):
        try:
            model = keras_load_model(path, compile=False)
            return model
        except Exception:
            print(f"Warning: Could not load full model from {path}, building architecture instead.")
            basename = os.path.basename(path).lower()
            if "vgg" in basename:
                base = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
            elif "densenet" in basename:
                base = DenseNet121(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
            elif "mobilenet" in basename:
                base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
            else:
                raise ValueError(f"Unknown model architecture for path: {path}")

            x = GlobalAveragePooling2D()(base.output)
            x = Dense(1024, activation="relu")(x)
            x = Dropout(0.5)(x)
            output = Dense(len(self.classes), activation="softmax")(x)
            model = Model(inputs=base.input, outputs=output)
            model.load_weights(path)
            return model

    def majority_vote(self, preds):
        return np.argmax(np.bincount(preds))

    def predict_disease(self, img_path):
        if not self.image_models:
            image_paths = [
                'K:/ME Project/ME Project/1 New Fish Disease and AquaGPT/Fish Disease/MobileNetV2/model/MobileNetV2_fish_disease_model.h5',
                'K:/ME Project/ME Project/1 New Fish Disease and AquaGPT/Fish Disease/DenseNet121/model/densenet121_fish_disease_model.h5',
                'K:/ME Project/ME Project/1 New Fish Disease and AquaGPT/Fish Disease/VGG16/model/vgg16_fish_disease_model.h5'
            ]
            for p in image_paths:
                try:
                    self.image_models[p] = self.load_model(p)
                except Exception as e:
                    print(f"Warning: Failed to load image model {p}: {e}")

        img = load_img(img_path, target_size=(128, 128))
        arr = img_to_array(img)[np.newaxis, ...]
        arr = preprocess_input(arr)

        preds = []
        for m in self.image_models.values():
            try:
                p = m.predict(arr, verbose=0)
                preds.append(np.argmax(p))
            except Exception as e:
                print(f"Warning: Image model prediction failed: {e}")

        if not preds:
            return "Predicted Disease: No image models available to predict."

        final_idx = self.majority_vote(preds)
        return f"Predicted Disease: {self.classes[final_idx]}"

    # ---------- Text QA ----------
    def encode_query(self, q):
        inputs = self.t5_tokenizer(text=q, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.device.type == "cuda":
                try:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = self.t5_model(**inputs)
                except Exception:
                    outputs = self.t5_model(**inputs)
            else:
                outputs = self.t5_model(**inputs)

        emb = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        return emb

    def generate_answer(self, query, image_path=None):
        if image_path:
            return self.predict_disease(image_path)

        try:
            q_emb = self.encode_query(query).astype("float32")
            if self.faiss_index is None:
                retrieved = []
            else:
                dists, inds = self.faiss_index.search(q_emb, 10)
                retrieved = [
                    self.documents[idx]
                    for dist, idx in zip(dists[0], inds[0])
                    if idx >= 0 and idx < len(self.documents)
                ]
        except Exception as e:
            print(f"Warning: FAISS retrieval failed: {e}")
            retrieved = []

        if not retrieved:
            retrieved = ["No relevant content found in dataset. Use your own knowledge."]
        context = " ".join(retrieved[:5])

        prompt = f"""
You are AquaGPT, an expert in aquaculture management and fish health.
Answer clearly and naturally using only factual and relevant information.

Question: {query}
Context: {context}
Answer:
"""

        inputs = self.mistral_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=500,  # allow longer answers
            temperature=0.5,
            top_p=0.8,
            do_sample=False
        )

        with torch.no_grad():
            if self.device.type == "cuda":
                try:
                    with torch.amp.autocast(device_type="cuda"):
                        out = self.mistral_model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask", None),
                            **gen_kwargs
                        )
                except Exception:
                    out = self.mistral_model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        **gen_kwargs
                    )
            else:
                out = self.mistral_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    **gen_kwargs
                )

        decoded = self.mistral_tokenizer.decode(out[0], skip_special_tokens=True)
        if "Answer:" in decoded:
            decoded = decoded.split("Answer:")[-1].strip()

        return decoded

    # ---------- FAISS / Documents loading ----------
    def load_saved_data(self):
        try:
            with open(self.saved_documents_file, "rb") as f:
                self.documents = pickle.load(f)
        except Exception as e:
            self.documents = []
            print(f"Warning: could not load documents: {e}")

        try:
            if os.path.exists(self.saved_faiss_index_file):
                with open(self.saved_faiss_index_file, "rb") as f:
                    loaded_index = pickle.load(f)
                try:
                    if torch.cuda.is_available():
                        res = faiss.StandardGpuResources()
                        self.faiss_index = faiss.index_cpu_to_gpu(res, 0, loaded_index)
                    else:
                        self.faiss_index = loaded_index
                except Exception:
                    self.faiss_index = loaded_index
            elif os.path.exists(self.saved_embeddings_file):
                with open(self.saved_embeddings_file, "rb") as f:
                    emb = pickle.load(f)
                emb = emb.astype("float32")
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                emb /= norms
                self.embeddings = emb
                dim = emb.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(emb)
                with open(self.saved_faiss_index_file, "wb") as f:
                    pickle.dump(self.faiss_index, f)
            else:
                self.embeddings = None
                self.faiss_index = None
        except Exception as e:
            print(f"Warning: error setting up FAISS: {e}")
            self.faiss_index = None

    # ---------- Model initialization ----------
    def initialize_models(self):
        try:
            self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            t5_kwargs = {}
            if self.device.type == "cuda":
                t5_kwargs["torch_dtype"] = torch.float16
            self.t5_model = T5EncoderModel.from_pretrained("t5-small", **t5_kwargs).to(self.device)
            self.t5_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize T5 encoder: {e}")

        try:
            self.mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
            if self.mistral_tokenizer.pad_token is None:
                self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token

            mistral_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
            if self.device.type == "cuda":
                mistral_kwargs["device_map"] = {"": "cuda:0"}
                mistral_kwargs["torch_dtype"] = torch.float16

            self.mistral_model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1",
                **mistral_kwargs,
            )
            self.mistral_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Mistral model: {e}")

        image_paths = [
            'K:/ME Project/ME Project/1 New Fish Disease and AquaGPT/Fish Disease/MobileNetV2/model/MobileNetV2_fish_disease_model.h5',
            'K:/ME Project/ME Project/1 New Fish Disease and AquaGPT/Fish Disease/DenseNet121/model/densenet121_fish_disease_model.h5',
            'K:/ME Project/ME Project/1 New Fish Disease and AquaGPT/Fish Disease/VGG16/model/vgg16_fish_disease_model.h5'
        ]
        for p in image_paths:
            try:
                self.image_models[p] = self.load_model(p)
            except Exception as e:
                print(f"Warning: failed to load image model {p}: {e}")

    def device_info(self):
        info = {"device": str(self.device)}
        try:
            info.update({
                "cuda_available": torch.cuda.is_available(),
                "cuda_mem_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "cuda_mem_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                "faiss_on_gpu": hasattr(self, "faiss_index") and self.faiss_index is not None
                               and "Gpu" in type(self.faiss_index).__name__,
                "t5_loaded": hasattr(self, "t5_model") and self.t5_model is not None,
                "mistral_loaded": hasattr(self, "mistral_model") and self.mistral_model is not None,
            })
        except Exception as e:
            info["device_info_error"] = str(e)
        return info
