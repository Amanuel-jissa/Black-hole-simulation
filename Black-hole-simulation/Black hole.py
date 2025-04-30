import numpy as np
import matplotlib.pyplot as plt
import cv2
# Load and read the image
image_path = "/Users/perfectoid/Downloads/milkyway-3.jpg"
galaxy = cv2.imread(image_path)
galaxy = cv2.cvtColor(galaxy, cv2.COLOR_BGR2RGB)
# Gets the image size
H, W, _ = galaxy.shape
# M is the mass of the Black hole (affects the bending strength)
M = W // 10.0
# Location of the Black hole
center_x, center_y = W // 2, H // 2
# By creating X and Y we get a grid of coordinates
X, Y = np.meshgrid(np.arange(W), np.arange(H))
#Defining dx and dy, the horizontal and vertical distance of each image point from the black hole's center.
dx = X - center_x
dy = Y - center_y
#The radial distance from each pixel to the center of the black hole
r = np.sqrt(dx**2 + dy**2) + 1e-6  # Avoid division by zero
# Einstein's bending angle approximation
alpha = (2 * M) / r
# Computes the new horizontal and vertical position of a pixel after gravitational lensing effect.
new_X = np.clip((center_x + dx * (1 - alpha)).astype(int), 0, W - 1)
new_Y = np.clip((center_y + dy * (1 - alpha)).astype(int), 0, H - 1)
# Apply lensing effect on the image, galaxy.
lensed_image = galaxy[new_Y, new_X]
# This sets the radius of the event horizon in pixels using the mass of the black hole
black_hole_radius = int(M * 0.375)
#identifies all the pixels that fall within the event horizon.
mask = r < black_hole_radius
# Makes the mask black
lensed_image[mask] = [0, 0, 0]
# Display the result
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(galaxy)
ax[0].set_title("Original Galaxy Image")
ax[0].axis("off")
ax[1].imshow(lensed_image)
ax[1].set_title("Gravitationally Lensed Image with Black Hole")
ax[1].axis("off")
plt.show()