import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

path_to_model = Path('resnet18-5c106cde.pth')
path_to_labels = Path('imagenet_classes2.txt')
path_to_img = list(Path('Pictures').glob('*'))

model = torchvision.models.resnet18(weights=None)
model.load_state_dict(torch.load(str(path_to_model)))
model = model.to(device)

with open('imagenet_classes2.txt', 'r') as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]
print(lines)

lines_str = '\n'.join(lines)
lines_list = lines_str.split('\n')
print(lines_list)

code_to_label = {}
for i, line in enumerate(lines_list):
    code_to_label[i + 1] = line

print(code_to_label)

code_to_label = {}
for i, line in enumerate(lines_list):
    code_to_label[i + 1] = line

print(code_to_label)

def select_most_likely(preds, count=3):
  indices_sorted = np.argsort(preds)[::-1]
  return [(i, preds[i]) for i in indices_sorted[:count]]

def build_summary(preds, code_to_label, count= 3, include_other=True):
  most_likely = select_most_likely(preds, count)

  lines= ["Predict classes:\n"]

  for i, p in most_likely:
    lbl= code_to_label[i]
    lbl_parts = [lbl[i:i+40] for i in range(0, len(lbl), 40)]
    lines.append("- {:>5.1%}: {}\n".format(p, '\n       '.join(lbl_parts)))

  if include_other:
    s= sum([p for _, p in most_likely])
    lines.append("- {:>5.1%}: other\n".format(1-s))
    return ''.join(lines)

  import torchvision.transforms as transforms
  def display_img_and_labels(img, preds, code_to_label, count=5, include_other=True):
      summary = build_summary(preds, code_to_label, count, include_other)

      gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2])
      plt.figure(figsize=(12, 6))
      plt.subplot(gs[0]).imshow(img_pil)
      plt.axis('off')
      plt.subplot(gs[1]).text(0.0, 0.50, summary, fontsize=16, fontname='monospace')
      plt.axis('off')
      plt.show()


  def get_predictions(model: torch.nn.Module, img: Image, device: torch.device) -> np.ndarray:
      img_tensor = for_pytorch(img)
      img_tensor = img_tensor.to(device)
      img_tensor = img_tensor.unsqueeze(0)
      output = model(img_tensor)
      preds = torch.nn.Softmax(dim=1)(output)
      return preds.cpu().numpy().squeeze()

  for_pytorch = torchvision.transforms.Compose([
      torchvision.transforms.Resize((224, 224)),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.225, 0.225]
      )
  ])

  model.eval()
  with torch.no_grad():
      for path in path_to_img:
          print(path.name)
          img_pil = Image.open(path)
          print(img_pil.size)
          preds = get_predictions(model, img_pil, device)
          display_img_and_labels(img_pil, preds, code_to_label)
