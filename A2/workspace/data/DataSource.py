import os
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2

class SubtractMeanImage(object):
    def __init__(self, mean_image):
        # Ensure mean_image is a numpy array
        if isinstance(mean_image, Image.Image):
            self.mean_image = np.array(mean_image)
        else:
            self.mean_image = mean_image

    def __call__(self, img):
        # Convert the PIL image to a NumPy array
        img_array = np.array(img)

        # Resize the mean image to match the img_array dimensions
        mean_image_resized = np.array(Image.fromarray(self.mean_image).resize(img_array.shape[1::-1], Image.BILINEAR))

        # Subtract the resized mean image
        result = img_array - mean_image_resized

        # Convert back to PIL image and return
        return Image.fromarray(np.clip(result, 0, 255).astype('uint8'))


class DataSource(data.Dataset):
    def __init__(self, root, resize=256, crop_size=224, train=True):
        self.root = os.path.expanduser(root)
        self.resize = resize
        self.crop_size = crop_size
        self.train = train

        self.image_poses = []
        self.images_path = []

        self._get_data()

        # Load mean image
        self.mean_image_path = os.path.join(self.root, 'mean_image.npy')
        if os.path.exists(self.mean_image_path):
            mean_image_np = np.load(self.mean_image_path)
            mean_image_pil = Image.fromarray(mean_image_np.astype('uint8'))
        else:
            mean_image_np = self.generate_mean_image()
            mean_image_pil = Image.fromarray(mean_image_np.astype('uint8'))

        # TODO: Define preprocessing
        self.transforms = T.Compose([
            T.Resize(self.resize),  # Resize the image to have the smaller dimension equal to 256
            SubtractMeanImage(mean_image_pil),  # Subtract the mean image using the custom transform
            T.RandomCrop(self.crop_size) if self.train else T.CenterCrop(self.crop_size),  # Crop the image to 224x224
            T.ToTensor(),  # Convert the image to a tensor
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image tensor
        ])

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + fname)

    def generate_mean_image(self):
        print("Computing mean image:")

        # TODO: Compute mean image

        # Initialize mean_image
        sum_image = None
        num_images = 0

        # Iterate over all training images
        # Resize, Compute mean, etc...
        for img_path in self.images_path:
            with Image.open(img_path) as img:
                # Resize the image
                img = img.resize((self.resize, self.resize), Image.BILINEAR)

                # Convert to numpy array and accumulate
                img_np = np.array(img)
                if sum_image is None:
                    sum_image = np.zeros_like(img_np, dtype=np.float64)
                sum_image += img_np
                num_images += 1

        # Compute the mean image
        mean_image = sum_image / num_images

        # Store mean image
        np.save(self.mean_image_path, mean_image)

        print("Mean image computed!")

        return mean_image

    def __getitem__(self, index):
        """
        return the data of one image
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]

        data = Image.open(img_path)

        # TODO: Perform preprocessing
        data = self.transforms(data)

        return data, img_pose

    def __len__(self):
        return len(self.images_path)




# for debugging 
    
# import matplotlib.pyplot as plt

# dataset = DataSource(root='data/datasets/KingsCollege/', train=True)

# # Test a few images
# for i in range(5):
#     image, pose = dataset[i]
#     image = image.numpy().transpose(1, 2, 0)  # Convert from tensor (C, H, W) to numpy array (H, W, C)

#     # Rescale the image data to [0, 1] range
#     image = np.clip(image, 0, 1)

#     # Visualize the image
#     plt.imshow(image)
#     plt.title(f"Pose: {pose}")
#     plt.show()