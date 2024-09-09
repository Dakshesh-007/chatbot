import torch
import asyncio
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import warnings
import gc

warnings.simplefilter(action="ignore", category=FutureWarning)

import concurrent.futures


class Model:
    def __init__(self) -> None:
        self.device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
        self.img_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        self.img_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(self.device)
        self.llm = Ollama(model="llama3.1")
        self.prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Here is an image description: '{image_description}'. {question}"
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.timeout = 15

    def input_img(self, img_data):
        inputs = self.img_processor(
            img_data, return_tensors="pt", clean_up_tokenization_spaces=True
        ).to(self.device)
        caption_ids = self.img_model.generate(**inputs, max_new_tokens=50)
        image_description = self.img_processor.decode(
            caption_ids[0], skip_special_tokens=True
        )
        return image_description

    async def get_response(self, user_input, image_description):
        prompt = self.prompt_template.format_prompt(
            image_description=image_description, question=user_input
        )
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            self.executor, self.llm.invoke, prompt.to_string()
        )
        return response

    def clear_memory(self):
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    from PIL import Image
    import requests
    from io import BytesIO

    def get_Image():
        url = input("Enter the image URL: ")
        response = requests.get(url)
        img_data = Image.open(BytesIO(response.content))
        return img_data

    model = Model()
    img = get_Image()
    description = model.input_img(img)

    while True:
        user_inp = input("Type \\exit to close\nEnter: ")
        if user_inp == "\\exit":
            print("Bye!")
            break
        response = asyncio.run(
            model.get_response(user_input=user_inp, image_description=description)
        )
        print(response)
        model.clear_memory()
