import requests

# Specify the API endpoint URL
url = "https://baseline-sky-detection.onrender.com/baseline_mean_shift_app"

# Define the path to the image file
image_file = "C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\\data\\dataset\\OriginalImages\\8733\\20130102_150639.jpg"

# Create a dictionary with the UploadFile object as "image" key
files = {'image': open(image_file, "rb")}

# Send the POST request to the API endpoint
response = requests.post(url, files=files)

# Check the response status code
if response.status_code == 200:
    # Save the segmented image locally
    with open('image.jpg', "wb") as f:
        f.write(response.content)
    print(f"Segmented image saved successfully at: {'image.jpg'}")
else:
    print(f"Error: {response.status_code} - {response.text}")