import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

img_path = '/home/moriki/PoseEstimation/mmpose/data/pose/CrowdPose/images/100000.jpg'
img = Image.open(img_path)                       # (H, W, 3)  H, W >= 32
image = preprocess(img).unsqueeze(0).to(device)    # (1, 3, 224, 224)
text = clip.tokenize(["A image of a human", "A image of an object"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)      # (1, 512)
    text_features = model.encode_text(text)         # (2, 512)
    
    logits_per_image, logits_per_text = model(image, text)  # (1, 2),  (2, 1)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # (1, 2)

# breakpoint()
print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]