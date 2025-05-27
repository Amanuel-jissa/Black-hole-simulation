import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def apply_gravitational_lens(image_path, mass_factor=0.1, event_horizon_factor=0.375, debug_mode=False):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'. Please check the path.")
        return None, None


    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error: Could not load image from '{image_path}'. It might be corrupt or an invalid format.")
        return None, None

    # Convert it from BGR to RGB to display on matplotlib
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Gets the image dimensions
    H, W, _ = original_image_rgb.shape
    if debug_mode:
        print(f"Image dimensions: Width={W}, Height={H}")

    # Define the 'mass' (M) which defines the strength of the lensing.
    # We relate it to the image width to make it somewhat scale-independent.
    M = W * mass_factor
    if debug_mode:
        print(f"Calculated 'mass' (M) for lensing: {M:.2f}")

    # Location of the Black Hole - set to the center of the image
    center_x, center_y = W // 2, H // 2
    if debug_mode:
        print(f"Black hole center: ({center_x}, {center_y})")

    # Create coordinate grids X, Y for every pixel
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    # Defining dx and dy, the horizontal and vertical distance of each image point
    # from the black hole's center.
    dx = X - center_x
    dy = Y - center_y

    # Calculate the radial distance from each pixel to the center of the black hole.
    # Adding a small epsilon to prevent division by zero for pixels exactly at the center.
    r = np.sqrt(dx**2 + dy**2) + 1e-6

    # Einstein's simplified bending angle approximation for light.
    # This formula is a common approximation used for visual effects, not a full
    # general relativistic calculation.
    alpha = (2 * M) / r

    # Compute the new horizontal and vertical position of a pixel after the
    # gravitational lensing effect. Pixels closer to the black hole are displaced more.
    # np.clip ensures that the new coordinates stay within the image boundaries.
    new_X = np.clip((center_x + dx * (1 - alpha)).astype(int), 0, W - 1)
    new_Y = np.clip((center_y + dy * (1 - alpha)).astype(int), 0, H - 1)

    # Apply the lensing effect by remapping pixels.
    # This creates the distortion based on the new coordinates.
    lensed_image = original_image_rgb[new_Y, new_X]

    # Calculate the radius of the event horizon in pixels.
    # Pixels within this radius will be rendered as black.
    black_hole_radius = int(M * event_horizon_factor)
    if debug_mode:
        print(f"Black hole event horizon radius: {black_hole_radius} pixels")

    # Create a mask for pixels that fall within the event horizon.
    mask = r < black_hole_radius

    # Make the masked area black to represent the black hole.
    lensed_image[mask] = [0, 0, 0] # Set RGB to black

    return original_image_rgb, lensed_image

def display_lensing_results(original_img, lensed_img, title_original="Image without lensing effect", title_lensed="After lensing effect is applied"):

    fig, ax = plt.subplots(1, 2, figsize=(14, 7)) # Slightly larger figure for better viewing

    ax[0].imshow(original_img)
    ax[0].set_title(title_original)
    ax[0].axis("off") # Hide axes for cleaner image display

    ax[1].imshow(lensed_img)
    ax[1].set_title(title_lensed)
    ax[1].axis("off") # Hide axes

    plt.tight_layout() # Adjust subplot params for a tight layout
    plt.show()

if __name__ == "__main__":
    # --- Main Execution Block ---
    # This block runs only when the script is executed directly, not when imported as a module.

    # Define your image path here. Make sure this path is correct for your system.
    # For a more robust solution, you could use a relative path, but an absolute
    # path is fine for a personal script.
    my_image_path = "/Users/perfectoid/PycharmProjects/Black-hole-simulation/Black-hole-simulation/milkyway-2.jpg"

    print(f"Attempting to apply lensing effect to: {my_image_path}")

    # Apply the lensing effect
    original_galaxy, lensed_galaxy = apply_gravitational_lens(
        my_image_path,
        mass_factor=0.08,  # Experiment with this value! Lower for subtle, higher for strong bending.
        event_horizon_factor=0.4, # Adjust the black hole's visual size
        debug_mode=True # Set to True to see more output during calculation
    )

    # Display the results if the process was successful
    if original_galaxy is not None and lensed_galaxy is not None:
        display_lensing_results(original_galaxy, lensed_galaxy)
        print("Lensing simulation complete. Check the displayed images!")
    else:
        print("Lensing simulation failed due to image loading or processing issues.")
    print("Script finished.")