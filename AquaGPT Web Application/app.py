from flask import Flask, request, jsonify, render_template
import imghdr
import os
from aquagpt import AquaGPT
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge, HTTPException

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

aquagpt = AquaGPT()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

        if query.lower() == 'disease':
            return jsonify({'response': 'Please upload a fish image for disease prediction.', 'request_image': True})

        file = request.files.get('file') or request.files.get('image')
        if file:
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            if ext not in {'png','jpg','jpeg'}:
                return jsonify({'error': 'Invalid file extension. Only png/jpg/jpeg allowed.'}), 400

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            if imghdr.what(filepath) not in {'png','jpeg','jpg'}:
                os.remove(filepath)
                return jsonify({'error': 'Uploaded file is not a valid image.'}), 400

            try:
                response = aquagpt.generate_answer(query, filepath)
            finally:
                os.remove(filepath)
            return jsonify({'response': response})

        # Text-only query
        response = aquagpt.generate_answer(query)
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext not in {'png','jpg','jpeg'}:
            return jsonify({'error': 'Invalid file extension'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        if imghdr.what(filepath) not in {'png','jpeg','jpg'}:
            os.remove(filepath)
            return jsonify({'error': 'Invalid image content'}), 400

        try:
            response = aquagpt.predict_disease(filepath)
        finally:
            os.remove(filepath)

        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/device_info')
def device_info():
    try:
        info = aquagpt.device_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({'error':'File too large. Max 16MB.'}), 413

@app.errorhandler(HTTPException)
def handle_http(e):
    return jsonify({'error': e.description}), e.code

@app.errorhandler(Exception)
def handle_all(e):
    return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)  # disable debug/reloader

