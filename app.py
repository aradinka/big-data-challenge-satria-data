import os
from pathlib import Path
import shutil
import streamlit as st
import face_recognition

from predict_utils import *


def run(save_path):
    # extract face
    save_extracted_face_path = "extracted_face"
    if os.path.exists(save_extracted_face_path):
        shutil.rmtree(save_extracted_face_path)
        os.makedirs(save_extracted_face_path)
    else:
        os.makedirs(save_extracted_face_path)

    image = face_recognition.load_image_file(save_path)
    face_locations = face_recognition.face_locations(image, model="hog")
    
    if len(face_locations) == 0:
        print("No face detected :(")
        cropped_faces = {"message": "No face detected :("}
    elif len(face_locations) == 1:
        temp_cropped_face = crop_face(save_path, face_locations)
        img_path = save_extracted_face_path + "/Face 1.jpg"
        cropped_faces = {"Face 1": {
            "face": temp_cropped_face,
            "gender": "unknown",
            "img_path": img_path
            }
        }
        temp_cropped_face.save(img_path)
    else:
        cropped_faces = {}
        for x in range(len(face_locations)):
            temp_cropped_face = crop_face(save_path, [list(face_locations)[x]])
            face_number = "Face " + str(x+1)
            img_path = save_extracted_face_path + "/" + face_number + ".jpg"
            temp_cropped_face.save(img_path)
            cropped_faces[face_number] = {
                "face": temp_cropped_face,
                "gender": "unknown",
                "img_path": img_path
            }

    # predict gender
    if len(face_locations) != 0:
        model_path = "model/model.h5"
        custom_object = {"f1_m":f1_m, "precision_m":precision_m, "recall_m":recall_m}

        model = tf.keras.models.load_model(model_path, custom_objects=custom_object)

        for k, v in cropped_faces.items():
            img_path = cropped_faces[k]["img_path"]
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = model.predict(images)

            if classes[0]>0.5:
                cropped_faces[k]["gender"] = "Woman"
            else:
                cropped_faces[k]["gender"] = "Man"

        for f in os.listdir(save_extracted_face_path):
            face = f.split(".")[0]
            st.image(Image.open(save_extracted_face_path + "/" + f), caption=cropped_faces[face]["gender"])
    else:
        st.write("No face detected :(")

class application():
    
    def __init__(self):
        
        st.write("## Sample photo")
            
        with st.form("Analyze sample photo 2"):
            sample_path = "sample_image/sample-2.jpg"
            st.image(Image.open(sample_path), caption="My Fams")
            sample_photo_2 = st.form_submit_button(label="Analyze")
        
        if sample_photo_2:
            run(sample_path)
            
        st.write("## Try and upload your photos")
        with st.form("Upload photos"):
            uploaded_file = st.file_uploader(label = "Upload file", type=["jpg"])
            submit_photo = st.form_submit_button(label='Submit')
        
        if submit_photo:
            # save photo
            save_folder = "uploaded photo"
            if os.path.exists(save_folder):
                shutil.rmtree(save_folder)
                os.makedirs(save_folder)
            else:
                os.makedirs(save_folder)   
                        
            uploaded_path = Path(save_folder, uploaded_file.name)
            with open(uploaded_path, mode='wb') as w:
                w.write(uploaded_file.getvalue())
                
            run(uploaded_path)
            




app = application()