import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv
import os

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def upload_face_image(file_path, enrollment, sample_num):
    public_id = f"faces/{enrollment}/{enrollment}_{sample_num}"
    result = cloudinary.uploader.upload(
        file_path,
        public_id=public_id,
        folder="attendance_faces",
        overwrite=True,
        resource_type="image"
    )
    return result.get("secure_url", "")

def delete_student_images(enrollment):
    try:
        cloudinary.api.delete_resources_by_prefix(f"attendance_faces/faces/{enrollment}/")
    except Exception as e:
        print(f"Cloudinary delete error: {e}")
