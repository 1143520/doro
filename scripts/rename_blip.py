import os
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 初始化 BLIP 模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def sanitize_filename(text, max_length=50):
    import re
    text = text.lower().strip().replace(" ", "_")
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)
    return text[:max_length]

def extract_caption_from_gif(gif_path):
    cap = cv2.VideoCapture(gif_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
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

            # 避免重名
            counter = 1
            while os.path.exists(new_path):
                new_path = os.path.join(directory, f"{sanitize_filename(caption)}_{counter}.gif")
                counter += 1
            os.rename(old_path, new_path)

if __name__ == "__main__":
    rename_gifs("loop")
