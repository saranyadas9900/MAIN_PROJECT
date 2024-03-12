from flask import Flask, render_template, request, redirect, url_for
from training1 import AppraisalVisionTrainer
import os
from prediction import predict_floor_plan



app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/New folder'
app.config['FLOORPLAN_FOLDER'] = 'floorplan_images'
app.config['NONFLOORPLAN_FOLDER'] = 'nonfloorplan_images'

# Load the trained model
trainer = AppraisalVisionTrainer()
trainer.load_model("rf_model_max.pkl")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction_upload', methods=['GET', 'POST'])
def predict_by_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('prediction_upload.html', error="No file selected.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('prediction_upload.html', error="No file selected.")
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        result = predict_floor_plan(file_path)
        
        
        return render_template('result.html', result=result, filename=file_path)
    
    return render_template('prediction_upload.html')


if __name__ == '__main__':
    app.run(debug=True)
