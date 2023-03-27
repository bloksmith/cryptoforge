import random
import re
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch.nn.functional as F
import os
import time
from PIL import ImageOps
import numpy as np
import openai
import os
import json
import csv
import os
from nft_metadata import create_metadata, save_metadata_to_json, save_metadata_to_csv
import openai  # Add this import at the beginning of your code
import tensorflow as tf
import sys
sys.path.append('/home/viktor/NFT/DALLE-pytorch')


# Add these missing imports
from torch.autograd import Variable
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.utils import save_image
from pathlib import Path

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import random
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from nft_metadata import create_metadata, save_metadata_to_csv, save_metadata_to_json

#from attribute_extraction import extract_attribute_values

import random
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from nft_metadata import create_metadata, save_metadata_to_csv, save_metadata_to_json
# Set seed value
seed_value = 42
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Generate filename dynamically based on seed value
input_folder = "/home/viktor/NFT/output"
nft_image_filename = os.path.join(input_folder, f"nft_seed42.jpg")


# Load image
img = image.load_img(nft_image_filename, target_size=(224, 224))


# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Load your NFT image and resize it to the input size expected by the VGG16 model


# Preprocess the image by converting it to a numpy array and applying the VGG16-specific preprocessing function
img_arr = image.img_to_array(img)
img_arr = np.expand_dims(img_arr, axis=0)
img_arr = preprocess_input(img_arr)

# Extract features from the image by passing it through the VGG16 model and getting the output of an intermediate layer
features = model.predict(img_arr)




filename = "your_filename_here"



phrases = [
    "Unique Artwork",
    "Exclusive Design",
    "One of a Kind",
    "Limited Edition",
    "Rare NFT",
    "Creative Masterpiece",
    "Digital Treasure",
    "Artistic Expression",
    "Innovative Art",
    "Ethereal Beauty",
]
from PIL import Image
import os
import random

def draw_spiral(draw, cx, cy, radius, steps, start_angle, angle_step, color, width_range=(1, 5)):
    angle = start_angle
    for _ in range(steps):
        width = random.randint(width_range[0], width_range[1])  # Randomize brush thickness
        x1 = cx + math.cos(math.radians(angle)) * radius
        y1 = cy + math.sin(math.radians(angle)) * radius
        angle += angle_step
        x2 = cx + math.cos(math.radians(angle)) * radius
        y2 = cy + math.sin(math.radians(angle)) * radius

        draw.line((x1, y1, x2, y2), fill=color, width=width)
        radius += 0.3

def draw_line(draw, x1, y1, x2, y2, color, width_range=(1, 5)):
    width = random.randint(width_range[0], width_range[1])  # Randomize brush thickness
    draw.line((x1, y1, x2, y2), fill=color, width=width)

def save_metadata_to_csv(metadata, csv_filename):
    metadata_list = [metadata]  # Convert metadata to a list containing a single dictionary
    keys = metadata_list[0].keys()

    with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
        dict_writer = csv.DictWriter(csv_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(metadata_list)

def save_metadata_to_json(metadata, json_filename):
    with open(json_filename, "w") as json_file:
        json.dump(metadata, json_file, indent=4)

    return json_filename

def extract_attribute_values(image_path):
    # Implement your attribute extraction logic here.
    # For now, I will return an example dictionary.
    return {"attribute1": "value1", "attribute2": "value2"}
def extract_image_features(image_path):
    # Load the InceptionV3 model
    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', pooling='avg')

    # Load the image and resize it to the required dimensions for the model
    image = Image.open(image_path)
    image = image.resize((299, 299))

    # Convert the image to a numpy array and normalize the pixel values
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Extract features from the image using the model
    features = model.predict(image_array)
    return features
image_features = extract_image_features(image_path)
print("Image features:", image_features)

def extract_attributes(image_path):
    # Open the image using PIL
    img = Image.open(image_path)

    # Extract image size (width and height)
    width, height = img.size

    # Extract file format
    file_format = img.format

    # Extract file size
    file_size = os.path.getsize(image_path)

    # Create a dictionary with the extracted attribute values
    attribute_values = {
        "width": width,
        "height": height,
        "format": file_format,
        "file_size": file_size
    }
    image_path = f"{output_folder}/{filename}"
    attribute_values = extract_attributes(image_path)
    return attribute_values

def generate_text(prompt, model="text-davinci-003", max_tokens=50):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()
def add_equation_curves(draw, content_image):
    width, height = content_image.size
    center_x, center_y = width // 2, height // 2

    # Golden Spiral
    a, b = 1 * random.uniform(0.5, 1.5), 0.3 * random.uniform(0.5, 1.5)
    theta = np.linspace(0, 8 * np.pi, 1000)
    x = center_x + a * np.exp(b * theta) * np.cos(theta)
    y = center_y + a * np.exp(b * theta) * np.sin(theta)
    for i in range(1, len(x)):
        draw.line((x[i - 1], y[i - 1], x[i], y[i]), fill=(255, 0, 0, 255), width=2)

    # Rose Curve
    a, b = 100 * random.uniform(0.5, 1.5), 4 * random.uniform(0.5, 1.5)
    theta = np.linspace(0, 2 * np.pi, 1000)
    x = center_x + a * np.cos(b * theta) * np.cos(theta)
    y = center_y + a * np.cos(b * theta) * np.sin(theta)
    for i in range(1, len(x)):
        draw.line((x[i - 1], y[i - 1], x[i], y[i]), fill=(0, 255, 0, 255), width=2)

    # Archimedean Spiral
    a, b = 1 * random.uniform(0.5, 1.5), 50 * random.uniform(0.5, 1.5)
    theta = np.linspace(0, 6 * np.pi, 1000)
    x = center_x + (a + b * theta) * np.cos(theta)
    y = center_y + (a + b * theta) * np.sin(theta)
    for i in range(1, len(x)):
        draw.line((x[i - 1], y[i - 1], x[i], y[i]), fill=(0, 0, 255, 255), width=2)

    # Cardioid
    a = 100 * random.uniform(0.5, 1.5)
    theta = np.linspace(0, 2 * np.pi, 1000)
    x = center_x + a * (1 + np.cos(theta)) * np.cos(theta)
    y = center_y + a * (1 + np.cos(theta)) * np.sin(theta)
    for i in range(1, len(x)):
        draw.line((x[i - 1], y[i - 1], x[i], y[i]), fill=(255, 255, 0, 255), width=2)

def draw_golden_spiral(image, a=1, b=0.306348, theta_start=0, theta_end=2 * np.pi, num_points=1000):
    draw = ImageDraw.Draw(image)
    angle_range = np.linspace(theta_start, theta_end, num_points)
    r_values = a * np.exp(b * angle_range)
    x_center, y_center = image.size[0] // 2, image.size[1] // 2
    points = [(x_center + r * np.cos(theta), y_center + r * np.sin(theta)) for theta, r in zip(angle_range, r_values)]
    draw.line(points, fill=(0, 0, 0), width=5)
def create_gradient(width, height):
    gradient = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(gradient)

    # Create a linear gradient from top to bottom
    for y in range(height):
        color = (y * 255 // height, y * 255 // height, y * 255 // height, 255)
        draw.line([(0, y), (width, y)], fill=color)

    return gradient

def add_random_lines(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    num_lines = random.randint(10, 20)

    for _ in range(num_lines):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
        thickness = random.randint(1, 5)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=thickness)


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]
    state_dict = map_keys(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def create_timestamped_folder():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"nft_images_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def apply_style(content_image, model, style_weight):
    transform = transforms.Compose([
        transforms.Resize(1080, Image.LANCZOS),  # Resize the height to 1080 pixels, preserving the aspect ratio
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    content_image = content_image.convert("RGB")  # Convert the image to RGB format
    content_image = transform(content_image).unsqueeze(0)
    with torch.no_grad():
        output = model(content_image)

    output = output.squeeze(0).cpu().detach()
    output = (output * 0.5 + 0.5).clamp(0, 1)

    return output
    
def draw_golden_spiral(draw, width, height, color=(255, 255, 255, 255), thickness=5):
    num_points = 1000
    num_quarters = 8
    phi = (1 + np.sqrt(5)) / 2
    t = np.linspace(0, num_quarters * np.pi, num_points)
    r = np.sqrt(t) / np.sqrt(phi ** 2 - 1)
    x = r * np.cos(t)
    y = r * np.sin(t)

    max_dimension = max(width, height)
    x = max_dimension * (x + 1) / 2
    y = max_dimension * (y + 1) / 2

    for i in range(1, num_points):
        draw.line(
            [(x[i - 1], y[i - 1]), (x[i], y[i])],
            fill=color,
            width=thickness,
        )



def create_image(seed, selected_model, style_weight, feature, output_folder, my_text, text):
    random.seed(seed)

    content_image = Image.new('RGBA', (7680, 4320), (255, 255, 255, 255))
    gradient = create_gradient(7680, 4320)
    content_image.paste(gradient, (0, 0))

    if feature == "golden_spiral":
        draw_golden_spiral(content_image)

    add_random_lines(content_image)

    draw = ImageDraw.Draw(content_image)
    output_folder = Path(output_folder)
    output_path = output_folder / f"{seed}-{feature}.png"

    filename = output_path.stem

    add_equation_curves(draw, content_image)

    frame_color = (0, 0, 0, 255)
    frame_width = 50
    draw.rectangle([(frame_width, frame_width), (content_image.width - frame_width, content_image.height - frame_width)], outline=frame_color, width=frame_width)

    if output_path:
        attribute_values = extract_attribute_values(output_path)
    else:
        attribute_values = {}

    metadata = create_metadata(seed=seed, feature=feature, filename=filename, text=text, my_text=my_text, attribute_values=attribute_values)

    content_image.save(output_path)
    print(f"Image saved to: {output_path}")

    json_filename = output_folder / f"{seed}-{feature}.json"
    save_metadata_to_json(metadata, json_filename)
    print(f"Metadata saved to: {json_filename}")

    csv_filename = os.path.join(output_folder, f"metadata_{seed}_{feature}.csv")
    save_metadata_to_csv(metadata, csv_filename)

    art_phrases = [
        "Aesthetic Adventure",
    ]

    text = random.choice(art_phrases)
    font = ImageFont.truetype("arial.ttf", 600)
    text_size = font.getbbox(text)[2:4]
    draw.text(((content_image.width - text_size[0]) // 2, (content_image.height - text_size[1]) // 2), text, fill=(255, 255, 255, 255), font=font)

    styled_image = apply_style(content_image, selected_model, style_weight)
    styled_filename = os.path.join(output_folder, f"nft_{seed}_styled_{feature}.png")
    save_image(styled_image, styled_filename)
    save_path = os.path.join(output_folder, f"nft_seed{seed}.png")
    final_image.save(save_path)
    print(f"Image saved as {save_path}")
nft_image_filename = create_image(seed)
img = image.load_img(nft_image_filename, target_size=(224, 224))
# Generate the image
seed = 42
nft_image_filename = create_image(seed)

# Load the generated image
img = image.load_img(nft_image_filename, target_size=(224, 224))
return save_path

    # Add text
    art_phrases = [
        "Aesthetic Adventure",
        "Imaginative Journey",
        "Visionary Voyage",
        "Dreamlike Discovery",
        "Creative Quest",
        "Colorful Composition",
        "Timeless Treasure",
        "Bold Brilliance",
        "Artful Allure",
        "Surreal Splendor",
        "Elegant Essence",
        "Masterful Melange",
        "Stunning Spectrum",
        "Vivid Vision",
        "Artistic Ascent",
        "Inspired Illusion",
        "Abstract Attraction",
        "Dynamic Dimension",
        "Fanciful Fusion",
        "Dreamy Design",
        "Enchanted Elegance",
        "Expressive Energy",
        "Glorious Gallery",
        "Whimsical Wonderland",
        "Captivating Creation",
        "Mystical Masterpiece",
        "Seductive Symphony",
        "Intricate Imagination",
        "Enlightened Expression",
        "Majestic Muse",
        "Luminous Landscape",
        "Miraculous Mirage",
        "Harmonious Haven",
        "Radiant Revelation",
        "Ethereal Experience",
        "Dazzling Dreamscape",
        "Wondrous Work",
        "Fantastic Fantasy",
        "Mesmerizing Magic",
        "Passionate Palette",
        "Divine Delight",
        "Infinite Inspiration",
        "Charming Charm",
        "Poetic Perfection",
        "Marvelous Movement",
        "Lyrical Lines",
        "Stylish Story",
        "Captivating Canvas",
        "Dreamy Depths",
        "Sensational Scene",
    ]

    text = random.choice(art_phrases)
    font = ImageFont.truetype("arial.ttf", 600)
    text_size = font.getbbox(text)[2:4]
    draw.text(((content_image.width - text_size[0]) // 2, (content_image.height - text_size[1]) // 2), text, fill=(255, 255, 255, 255), font=font)

    # Draw the golden spiral
    draw_golden_spiral(draw, content_image.width, content_image.height)

    # Apply style transfer
    styled_image = apply_style(content_image, selected_model, style_weight)


    # Save image
    filename = os.path.join(output_folder, "nft_{}_styled_{}.png".format(seed, feature))
    save_image(styled_image, filename)






# ... (rest of the code remains unchanged)



import torch.nn as nn

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.conv3(x)
        x = self.in3(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.deconv1(x)
        x = self.in4(x)
        x = self.deconv2(x)
        x = self.in5(x)
        x = self.deconv3(x)
        return x

def map_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace(".weight.weight", ".weight").replace(".weight.bias", ".bias")

        new_key = re.sub(r"^conv(\d+)\.weight$", r"encoder.\1.weight", new_key)
        new_key = re.sub(r"^conv(\d+)\.bias$", r"encoder.\1.bias", new_key)

        new_key = re.sub(r"^residual_blocks\.conv(\d+)\.weight$", r"residual_blocks.\1.conv1.weight", new_key)
        new_key = re.sub(r"^residual_blocks\.conv(\d+)\.bias$", r"residual_blocks.\1.conv1.bias", new_key)

        new_key = re.sub(r"^residual_blocks\.in(\d+)\.weight$", r"residual_blocks.\1.conv2.weight", new_key)
        new_key = re.sub(r"^residual_blocks\.in(\d+)\.bias$", r"residual_blocks.\1.conv2.bias", new_key)

        new_key = re.sub(r"^decoder\.weight\.weight$", r"decoder.0.weight", new_key)
        new_key = re.sub(r"^decoder\.weight\.bias$", r"decoder.0.bias", new_key)

        new_state_dict[new_key] = v
    return new_state_dict



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.in1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.in2(x)
        x += residual
        return x

        
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.in1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.in2(x)
        x += residual
        return x


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = F.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

def create_gradient(width, height):
    gradient = Image.new('RGBA', (width, height))

    for i in range(height):
        for j in range(width):
            color = (int(j / width * 255), int(i / height * 255), 125, 255)  # Adjusted the formula for color
            gradient.putpixel((j, i), color)

    return gradient

def add_random_lines(image, num_lines=10):
    width, height = image.size
    draw = ImageDraw.Draw(image)

    for _ in range(num_lines):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=5)

    return image
       



# Load pre-trained models
# Model 1 is trained on a "Mosaic" style, inspired by mosaic artwork
model_path1 = "/home/viktor/NFT/saved_models/mosaic.pth"

# Model 2 is trained on a "Candy" style, which results in abstract and colorful artwork
model_path2 = "/home/viktor/NFT/saved_models/candy.pth"

model1 = load_model(model_path1)
model2 = load_model(model_path2)



# Number of images to generate
num_images = 100

output_folder = os.path.join("output")
os.makedirs(output_folder, exist_ok=True)

for i in range(num_images):
    seed = random.randint(0, 100000)
    selected_model = model1 if i % 2 == 0 else model2  # Alternate between models
    style_name = "Mosaic" if i % 2 == 0 else "Candy"
    style_weight = random.uniform(0.1, 1.0)
    feature = random.choice(['gradient', 'text', 'frame', 'random_lines'])  # Add new features to the list
# Define the text you want to use
    my_text = "This is my custom text for the NFT artwork"
    text = "Some description"  # Replace this with the desired description or a variable containing the description.
    # Pass the output_folder to the create_image function
create_image(seed, selected_model, style_weight, feature, output_folder, my_text, text)


image_path = output_folder / filename
attributes = attribute_extractor.extract_attributes(image_path)
print(attributes)


print("Image {} generated with {} style, feature: {}, and style weight: {}".format(i+1, style_name, feature, style_weight))
json_file_path = save_metadata_to_json(metadata, output_folder, seed, feature)
print(f"JSON file saved at: {json_file_path}")

print(f"JSON file saved at: {json_filename}")
print(f"CSV file saved at: {os.path.join(output_folder, 'metadata.csv')}")

    
metadata_list = []
# Generate metadata
metadata = create_metadata(seed, feature, "Viktor Sandstrøm Kristensen", filename,title )

# Save metadata to JSON file
json_filename = save_metadata_to_json(metadata, output_folder)


# Add metadata to the list
metadata_list.append(metadata)
save_metadata_to_csv(metadata_list, output_folder)
# Save image
filename = os.path.join(output_folder, f"nft_{seed}_styled_{feature}.png")
filename = filename.replace("nft_images_", "https://babyrottweiler.com/NFT/CryptoForge/collection/1/nft_images_")
save_image(styled_image, filename)
