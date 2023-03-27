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

# Add these missing imports
from torch.autograd import Variable
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.utils import save_image
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
    # Define the text you want to use
    my_text = "This is my custom text for the NFT artwork"
    text = "Some description"  # Replace this with the desired description or a variable containing the description.
    
    content_image = Image.new('RGBA', (7680, 4320), (255, 255, 255, 255))
    gradient = create_gradient(7680, 4320)
    content_image.paste(gradient, (0, 0))

    # Draw the Golden Spiral
    if feature == "golden_spiral":
        draw_golden_spiral(content_image)

    add_random_lines(content_image)

    # Define the draw variable here
    draw = ImageDraw.Draw(content_image)

    add_equation_curves(draw, content_image)  # Call add_equation_curves

    frame_color = (0, 0, 0, 255)  # Changed the frame color
    frame_width = 50  # Increased the frame width
    draw.rectangle([(frame_width, frame_width), (content_image.width - frame_width, content_image.height - frame_width)], outline=frame_color, width=frame_width)

    metadata = create_metadata(seed=seed, feature=feature, filename=filename, text=text,
                           title="NFT Artwork", artist="Viktor S. Kristensen", description=my_text)


    # Save metadata to JSON file
    json_filename = os.path.join(output_folder, f"metadata_{seed}_{feature}.json")
    save_metadata_to_json(metadata, json_filename)

    # Save metadata to CSV file
    csv_filename = os.path.join(output_folder, f"metadata_{seed}_{feature}.csv")
    save_metadata_to_csv(metadata, csv_filename)

    # Add text
    art_phrases = [
        "Aesthetic Adventure",
    ]

    text = random.choice(art_phrases)
    font = ImageFont.truetype("arial.ttf", 600)
    text_size = font.getbbox(text)[2:4]
    draw.text(((content_image.width - text_size[0]) // 2, (content_image.height - text_size[1]) // 2), text, fill=(255, 255, 255, 255), font=font)

    # Draw the golden spiral
    draw_golden_spiral(draw, content_image.width, content_image.height)

    # Apply style transfer
    styled_image = apply_style(content_image, model, style_weight)

    # Save image
    filename = os.path.join(output_folder, "nft_{}_styled_{}.png".format(seed, feature))
    save_image(styled_image, filename)

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
    styled_image = apply_style(content_image, model, style_weight)

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


    print("Image {} generated with {} style, feature: {}, and style weight: {}".format(i+1, style_name, feature, style_weight))
    json_file_path = save_metadata_to_json(metadata, output_folder, seed, feature)
    print(f"JSON file saved at: {json_file_path}")

    print(f"JSON file saved at: {json_filename}")
    print(f"CSV file saved at: {os.path.join(output_folder, 'metadata.csv')}")

    
    metadata_list = []

# Save metadata to JSON file
json_filename = save_metadata_to_json(metadata, output_folder)


# Add metadata to the list
metadata_list.append(metadata)
save_metadata_to_csv(metadata_list, output_folder)
# Save image
filename = os.path.join(output_folder, f"nft_{seed}_styled_{feature}.png")
filename = filename.replace("nft_images_", "https://babyrottweiler.com/NFT/CryptoForge/collection/1/nft_images_")
save_image(styled_image, filename)
# Generate metadata
metadata = create_metadata(seed, feature, "Viktor Sandstrøm Kristensen", filename,title )
