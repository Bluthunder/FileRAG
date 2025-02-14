import os
import shutil
import gradio as gr

UPLOAD_FOLDER = "files/uploads/"


def upload_and_save(file):
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    # saved_path = os.path.join(UPLOAD_FOLDER, file.name)
    shutil.copy(file, UPLOAD_FOLDER)
    gr.Info("File Uploaded !!")
    # return saved_path
