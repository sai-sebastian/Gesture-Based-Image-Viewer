import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name='',  # Replace with your Cloud name
    api_key='',        # Replace with your API key
    api_secret=''   # Replace with your API secret
)
        
def sendingImageTocloudinary( image_path):
    image_url=""
    result = cloudinary.uploader.upload(
            image_path,
             upload_preset='schoolwork'
        )
    print("File URL:", result.get("url"))
    print("Secure URL:", result.get("secure_url"))
        
