import os
import cv2
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# 模型加载
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_length = 64
num_beams = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_caption(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
    preds = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return preds

def sanitize_filename(text, max_length=50):
    import re
    text = text.strip().replace(" ", "_")
    text = re.sub(r'[\\/:*?"<>|]', '', text)
    return text[:max_length]

def extract_caption_from_gif(gif_path):
    cap = cv2.VideoCapture(gif_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    caption = generate_caption(img)
    return caption

def rename_gifs(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.gif'):
            old_path = os.path.join(directory, filename)
            caption = extract_caption_from_gif(old_path)
            if not caption:
                continue
            new_name = sanitize_filename(caption) + ".gif"
            new_path = os.path.join(directory, new_name)
            counter = 1
            while os.path.exists(new_path):
                new_path = os.path.join(directory, f"{sanitize_filename(caption)}_{counter}.gif")
                counter += 1
            os.rename(old_path, new_path)

if __name__ == "__main__":
    rename_gifs("loop")
