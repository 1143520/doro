import os
import cv2
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    MarianMTModel, MarianTokenizer
)

# 初始化 BLIP 英文描述模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 初始化 Marian 翻译模型（英 → 中）
translator_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

def translate_to_chinese(text):
    inputs = translator_tokenizer(text, return_tensors="pt", padding=True)
    translated = translator_model.generate(**inputs)
    chinese = translator_tokenizer.decode(translated[0], skip_special_tokens=True)
    return chinese.strip()

def sanitize_filename(text, max_length=50):
    import re
    text = text.strip().replace(" ", "_")
    text = re.sub(r'[\\/:*?"<>|]', '', text)  # 去掉非法字符
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
    caption_en = processor.decode(output[0], skip_special_tokens=True)
    caption_zh = translate_to_chinese(caption_en)
    return caption_zh

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
