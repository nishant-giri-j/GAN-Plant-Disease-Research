import os
import torch
from torchvision.utils import save_image

def save_classwise_images(generator, z_dim, classes, num_per_class, device, out_dir="../synthetic"):
    """
    generator     : trained generator model
    z_dim         : noise dimension
    classes       : list of class names
    num_per_class : number of images to generate per class
    device        : cpu/cuda
    """

    generator.eval()

    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for label, cls_name in enumerate(classes):

            class_folder = os.path.join(out_dir, cls_name)
            os.makedirs(class_folder, exist_ok=True)

            for i in range(num_per_class):
                noise = torch.randn(1, z_dim, 1, 1).to(device)

                # class conditioning handled later (for conditional GAN)
                fake_img = generator(noise, torch.tensor([label]).to(device))

                save_path = os.path.join(class_folder, f"{cls_name}_{i}.png")

                save_image(fake_img, save_path, normalize=True)

    print("Synthetic images saved class-wise.")
