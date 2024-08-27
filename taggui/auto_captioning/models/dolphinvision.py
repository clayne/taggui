import numpy as np
from transformers import BatchFeature

from auto_captioning import auto_captioning_model
import transformers
import torch

from utils.image import Image


class DolphinVision7b(auto_captioning_model.AutoCaptioningModel):
    transformers_model_class = transformers.AutoModelForCausalLM

    def __init__(self, captioning_thread_: 'captioning_thread.CaptioningThread', caption_settings: dict):
        super().__init__(captioning_thread_, caption_settings)
        self.temp_1 = -1

    @staticmethod
    def get_default_prompt() -> str:
        return "Describe this image in detail."

    def get_tokenizer(self):
        return self.processor

    def get_model_inputs(self, image_prompt: str,
                         image: Image) -> BatchFeature | dict | np.ndarray:
        text = self.get_input_text(image_prompt)
        messages = [{"role": "user", "content": f"<image>\n{text}"}]
        tokenizer = self.get_tokenizer()
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        text_chunks = [tokenizer(chunk).input_ids for chunk in text.split("<image>")]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0)
        self.temp_1 = input_ids.shape[1]

        model = self.model
        pil_image = self.load_image(image)
        image_tensor = model.process_images([pil_image], model.config).to(self.device, **self.dtype_argument)

        return {
            "input_ids": input_ids,
            "images": image_tensor,
            "use_cache": True
        }

    def get_caption_from_generated_tokens(
            self, generated_token_ids: torch.Tensor, image_prompt: str) -> str:
        # generated_text = self.processor.batch_decode(
        #     generated_token_ids, skip_special_tokens=True)[0]

        generated_text = self.processor.decode(
            generated_token_ids[0][self.temp_1:],
            skip_special_tokens=True
        )
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
