import os
import cv2
from progress.bar import Bar

deleted_count = 0

def try_compute_histogram(image_path):
    global deleted_count
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        os.remove(image_path)
        deleted_count += 1


# get image list
real_images = [f"REAL/{file}" for file in os.listdir("REAL") if file.lower().endswith(('.jpg', '.png'))]
ai_images = [f"AI/{file}" for file in os.listdir("AI") if file.lower().endswith(('.jpg', '.png'))]
images = real_images + ai_images

# cleaning images 
with Bar("Cleaning images...", max=len(images)) as bar:
    for image in images:
        try_compute_histogram(image)
        bar.next()
    
print(f"\nDeleted a total of {deleted_count} images.")
print("\nDone!")
