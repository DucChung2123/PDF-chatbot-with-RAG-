import os
from flask import Blueprint, request, jsonify
from .utils import allowed_file, save_file
from .query_processing import query_pdf

main = Blueprint('main', __name__)

@main.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = save_file(file)
        return jsonify({"message": "File uploaded successfully", "filename": filename}), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400

@main.route('/query', methods=['POST'])
def query_pdf_api():
    data = request.json
    if 'filename' not in data or 'question' not in data:
        return jsonify({"error": "Filename and question are required"}), 400

    filename = data['filename']
    question = data['question']

    try:
        response = query_pdf(filename, question)
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
