import numpy as np
from transformers import BatchFeature

from auto_captioning.auto_captioning_model import AutoCaptioningModel
from auto_captioning import captioning_thread
from torch import nn
import torch
import transformers

from utils.image import Image

CLIP = "google/siglip-so400m-patch14-384"

class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)

    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class JoyCaption(AutoCaptioningModel):
    transformers_model_class = transformers.AutoModelForCausalLM

    def __init__(self, captioning_thread_: 'captioning_thread.CaptioningThread', caption_settings: dict):
        super().__init__(captioning_thread_, caption_settings)
        # JoyCaption uses LLaMA 3.1 8b with SigLIP CLIP, so the base model should be LLaMA
        self.model_id = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
        self.clip_model = None
        self.clip_processor = None
        self.image_adapter = None

        self.temp_1 = 0

    @staticmethod
    def get_default_prompt() -> str:
        return "Write a detailed description for the image."

    # def get_processor(self):
    #     return transformers.AutoTokenizer.from_pretrained(self.model_id, use_fast=False)

    def get_tokenizer(self):
        # The processor IS the tokenizer for JoyCaption
        return self.processor
    def monkey_patch_after_loading(self):
        self.clip_processor = transformers.AutoProcessor.from_pretrained(CLIP)
        self.clip_model = transformers.AutoModel.from_pretrained(CLIP).vision_model
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.clip_model.to(self.device)

        self.image_adapter = ImageAdapter(self.clip_model.config.hidden_size, self.model.config.hidden_size)
        self.image_adapter.load_state_dict(torch.load("joycaption/image_adapter.pt", map_location="cpu"))
        self.image_adapter.eval()
        self.image_adapter.to(self.device)

    def get_model_inputs(self, image_prompt: str,
                         image: Image) -> BatchFeature | dict | np.ndarray:
        # Embed image
        pil_image = self.load_image(image)
        pixels = self.clip_processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        with torch.amp.autocast_mode.autocast(self.device.type, enabled=True):
            vision_outputs = self.clip_model(pixel_values=pixels, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = self.image_adapter(image_features)
            embedded_images = embedded_images.to(self.device)

        # Tokenize the prompt
        text = self.get_input_text(image_prompt)
        prompt = self.processor.encode(
            text,
            return_tensors='pt',
            padding=False,
            truncation=False,
            add_special_tokens=False
        )

        # Embed prompt
        prompt_embeds = self.model.model.embed_tokens(prompt.to(self.device))
        embedded_bos = self.model.model.embed_tokens(torch.tensor(
            [[self.processor.bos_token_id]],
            device=self.model.device,
            dtype=torch.int64
        ))

        # Construct prompts
        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        ], dim=1)

        input_ids = torch.cat([
            torch.tensor([[self.processor.bos_token_id]], dtype=torch.long),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            prompt,
        ], dim=1).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        self.temp_1 = input_ids.shape[1]
        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "suppress_tokens": None,
        }

    def get_caption_from_generated_tokens(
            self, generated_token_ids: torch.Tensor, image_prompt: str) -> str:
        # Due to how JoyCaption embeds the image into the prompt,I have to trim the prompt differently
        # to the other models
        generated_token_ids = generated_token_ids[:, self.temp_1:]
        if generated_token_ids[0][-1] == self.tokenizer.eos_token_id:
            generated_token_ids = generated_token_ids[:, :-1]
        generated_text = self.processor.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
        # The start of the caption will be cut off due to the trimming, so I manually add it back in here
        generated_text = f"{self.caption_start}{generated_text}"
        image_prompt = self.postprocess_image_prompt(image_prompt)
        generated_text = self.postprocess_generated_text(generated_text)
        if image_prompt.strip() and generated_text.startswith(image_prompt):
            caption = generated_text[len(image_prompt):]
        elif (self.caption_start.strip()
              and generated_text.startswith(self.caption_start)):
            caption = generated_text
        else:
            caption = f'{self.caption_start.strip()} {generated_text.strip()}'
        caption = caption.strip()
        if self.remove_tag_separators:
            caption = caption.replace(self.thread.tag_separator, ' ')
        return caption
