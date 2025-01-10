import os
import cv2
from face_detection import compute_yolo, init_yolo
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
from datetime import datetime

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--start-maximized')
    return webdriver.Chrome(options=options)

def login_to_twitter(driver, username, password):
    driver.get("https://twitter.com/login")
    try:
        username_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "text"))
        )
        username_field.send_keys(username)
        
        next_button = driver.find_element(By.XPATH, "//span[text()='Next']")
        next_button.click()
        
        password_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "password"))
        )
        password_field.send_keys(password)
        
        login_button = driver.find_element(By.XPATH, "//span[text()='Log in']")
        login_button.click()
        
        time.sleep(5)
    except Exception as e:
        print(f"Login failed: {str(e)}")
        driver.quit()
        return False
    return True

def generate_and_save_images(driver, prompt):
    try:
        driver.get("https://twitter.com/i/grok")
        time.sleep(5)
        
        prompt_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "textarea"))
        )
        prompt_input.clear()
        prompt_input.send_keys(prompt)
        
        generate_button = driver.find_element(By.XPATH, "//span[contains(text(), 'Generate')]")
        generate_button.click()
        
        time.sleep(20)
        
        save_dir = "generated_faces"
        os.makedirs(save_dir, exist_ok=True)
        
        images = driver.find_elements(By.TAG_NAME, "img")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for idx, img in enumerate(images):
            try:
                img_url = img.get_attribute('src')
                if img_url and 'data:image' not in img_url:
                    response = requests.get(img_url)
                    if response.status_code == 200:
                        filename = f"{save_dir}/face_{timestamp}_{idx+1}.png"
                        with open(filename, 'wb') as f:
                            f.write(response.content)
                        print(f"Saved image {idx+1} to {filename}")
            except Exception as e:
                print(f"Failed to save image {idx+1}: {str(e)}")
                
    except Exception as e:
        print(f"Generation failed: {str(e)}")

def rename_dataset_files(directory="generated_faces"):
    """
    Renames all files in the specified directory to squint_1, squint_2, etc.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found!")
        return
    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()
    
    for index, filename in enumerate(files, start=1):
        _, ext = os.path.splitext(filename)
        new_filename = f"squint_{index}{ext}"
        
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
        except Exception as e:
            print(f"Error renaming {filename}: {str(e)}")

def crop_and_save_faces(source_dir: str, target_dir: str, model) -> None:
    """
    Crops faces from images in source directory and saves them to target directory.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    for filename in files:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        image = cv2.imread(source_path)
        if image is None:
            print(f"Could not read image: {filename}")
            continue
            
        bboxes = compute_yolo(image, model=model, conf=0.25)
        
        if not bboxes[0]:
            print(f"No face detected in {filename}")
            shutil.copy2(source_path, target_path)
            continue
            
        x1, y1, x2, y2 = bboxes[0]
        face_crop = image[y1:y2, x1:x2]
        cv2.imwrite(target_path, face_crop)
        print(f"Processed: {filename}")

def process_images_to_grayscale(directories):
    """
    Converts images in specified directories to grayscale and resizes them to 170x170.
    
    Args:
        directories (list): List of directory paths to process
    """
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory '{directory}' not found!")
            continue
        
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        
        for filename in files:
            file_path = os.path.join(directory, filename)
            try:
                # Read image
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Could not read image: {filename}")
                    continue
                
                # Convert to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Resize to 170x170
                resized_image = cv2.resize(gray_image, (170, 170))
                
                # Save the processed image, overwriting the original
                cv2.imwrite(file_path, resized_image)
                print(f"Processed: {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def main():
    # Process both directories
    directories = [
        "data/non_squinting",
        "data/squinting"
    ]
    
    process_images_to_grayscale(directories)

if __name__ == "__main__":
    main()
