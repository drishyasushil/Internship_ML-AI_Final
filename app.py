from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import cv2
import numpy as np
from modules.object_detection_module import detect_objects
from modules.emotion_analysis_module import analyze_emotions
from modules.event_detection_module import detect_events
from modules.temporal_analysis_module import analyze_temporal_relationships
from modules.scenario_analysis_module import predict_scenarios
from modules.output_module import generate_segmented_videos

app = Flask(__name__)

# Set the upload folder and allow certain file extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    # Retrieve user inputs
    selected_scenario = request.form['scenario']
    segment_duration = int(request.form['duration'])
    emotion_analysis_enabled = bool(request.form.get('emotion', False))

    # Get the video file path from the form submission
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected video file'}), 400
    if not allowed_file(video_file.filename):
        return jsonify({'error': 'Invalid file extension'}), 400
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    # Perform video analysis
    frames = get_video_frames(video_path)
    bounding_boxes, object_labels = detect_objects(frames)
    emotions = analyze_emotions(frames)
    events = detect_events()
    temporal_context = analyze_temporal_relationships()
    scenarios = predict_scenarios()
    segmented_videos = generate_segmented_videos()

    # Write segmented videos to new files
    for scenario, segment in segmented_videos.items():
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{scenario}_segment.mp4')
        write_segment(segment, output_path)

    # Return the output
    return jsonify({
        'scenario': selected_scenario,
        'segment_duration': segment_duration,
        'emotion_analysis_enabled': emotion_analysis_enabled,
        'bounding_boxes': bounding_boxes,
        'object_labels': object_labels,
        'emotions': emotions,
        'events': events,
        'temporal_context': temporal_context,
        'scenarios': scenarios,
        'segmented_videos': segmented_videos,
        'segmented_video_paths': {
            scenario: f'/uploads/{scenario}_segment.mp4' for scenario in segmented_videos.keys()
        }
    })

def get_video_frames(video_path):
    # Function to extract frames from video file
    # Read video file
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def write_segment(segment, output_path):
    # Write segmented video to a new file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25, (segment[0].shape[1], segment[0].shape[0]))

    for frame in segment:
        out.write(frame)

    out.release()

@app.route('/uploads/goal_segment.mp4')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)