from PIL import Image
import os

def images_to_gif(image_folder, output_path, duration=50):
    """
    Convert a series of images in a folder into a GIF file.

    Parameters:
        image_folder (str): Path to the folder containing images.
        output_path (str): Path to save the resulting GIF.
        duration (int): Duration of each frame in milliseconds.
    """
    # Get all image file paths from the folder
    images = []
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            images.append(os.path.join(image_folder, file_name))
    # sorted_images = sorted(images, key= lambda x: int(x.split('/')[-1].split('_')[0]))
    sorted_images = images
    if not images:
        raise ValueError("No images found in the folder.")

    # Open images
    frames = [Image.open(img) for img in sorted_images]

    # Convert to GIF
    first_frame = frames[0]
    first_frame.save(
        output_path,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0
    )

    print(f"GIF created successfully and saved to {output_path}")

# Example usage
# image_folder = "../../taco/interactive/5_i-7_1_c_l_l_1_0z/variant_scenario/11/rgb/front"  # Replace with your folder path
image_folder = "/media/hcis-s15/ssd2/nuScenes"  # Replace with your folder path
output_gif = "../gif/nuscene.gif"  # Replace with your desired output file name
images_to_gif(image_folder, output_gif)