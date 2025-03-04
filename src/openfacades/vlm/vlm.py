import os
import torch
import os
import pandas as pd
import torch
from .base import update_json_file, load_image
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time


class InternVLProcessor:
    def __init__(self, model):
        self.model = AutoModel.from_pretrained(
            model, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            use_flash_attn=True, 
            trust_remote_code=True,
            device_map='auto'
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=False)

    def process_image(self, image_path, question):
        pixel_values = load_image(image_path, max_num=12)
        if pixel_values is None:
            print(f"Skipping {image_path}, image could not be loaded.")
            return None

        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        generation_config = {"max_new_tokens": 1024, "do_sample": False}
        response, _ = self.model.chat(
            self.tokenizer, 
            pixel_values, 
            question, 
            generation_config,
            history=None,
            return_history=True
        )
        return response

    def question_answering(self, question, single_image=None, single_directory=None, output_directory=None, csv_input=None, image_directory=None):
        if sum(arg is not None for arg in [single_image, single_directory, csv_input]) > 1:
            raise ValueError("Multiple inputs provided. Please provide only one of single_image, single_directory, or csv_input.")
        if csv_input:
            if not image_directory:
                raise ValueError("For CSV input, image_directory must be provided.")
            if not output_directory:
                raise ValueError("For CSV input, output_directory must be provided.")
            os.makedirs(output_directory, exist_ok=True)
            output_file = os.path.join(output_directory, "responses.json")
            reader = pd.read_csv(csv_input)
            if 'image_name' not in reader.columns:
                raise ValueError("CSV does not contain 'image_name' column.")
            for _, row in tqdm(reader.iterrows(), total=len(reader), desc="Processing CSV images"):
                image_name = row['image_name']
                full_image_path = os.path.join(image_directory, image_name)
                response = self.process_image(full_image_path, question)
                result = {image_name: response}
                update_json_file(result, output_file)
            print(f"Responses saved to: {output_file}")
        elif single_image:
            response = self.process_image(single_image, question)
            if response is not None:
                # print(f"Response for {single_image}:\n{response}")
                for char in response:
                    print(char, end='', flush=True)
                    time.sleep(0.02)  # Adjust the delay as needed  
        elif single_directory:
            if not output_directory:
                raise ValueError("For single_directory, output_directory must be provided.")
            os.makedirs(output_directory, exist_ok=True)
            output_file = os.path.join(output_directory, "responses.json")
            image_files = [f for f in os.listdir(single_directory) if os.path.isfile(os.path.join(single_directory, f))]
            for image_name in tqdm(image_files, desc="Processing images"):
                full_image_path = os.path.join(single_directory, image_name)
                response = self.process_image(full_image_path, question)
                result = {image_name: response}
                update_json_file(result, output_file)
            print(f"Responses saved to: {output_file}")
        else:
            raise ValueError("Invalid input. Provide a csv_input, single_image, or single_directory.")

def building_vlm(model, question, single_image=None, single_directory=None, output_directory=None, csv_input=None, image_directory=None):
    """
    Process images using the given model and question.

    :param model: Name or path of the model to load.
    :param question: Question to pass to the model.
    :param csv_input: Path to a CSV file containing an 'image_name' column.
    :param single_image: Path to a single image file.
    :param single_directory: Path to a directory of images.
    :param output_directory: Directory to save JSON output if processing multiple images.
    :param image_directory: (Required if csv_input is provided) Directory where images are stored.
    """
    processor = InternVLProcessor(model)
    processor.question_answering(
        question, 
        single_image=single_image, 
        single_directory=single_directory, 
        output_directory=output_directory, 
        csv_input=csv_input, 
        image_directory=image_directory
    )
