import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(file):
    filename = secure_filename(file.filename)
    if not os.path.exists('uploads'):
        os.mkdir('uploads')
    filepath = os.path.join('uploads', filename)
    file.save(filepath)
    return filename
