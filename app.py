from flask import Flask, request, jsonify, send_from_directory, abort
import subprocess
import tempfile
from openai import OpenAI
import os
import base64

app = Flask(__name__)
client = OpenAI()
DIRECTORY = os.getcwd()  # Current working directory
last_enriched = "Mandarin / English conversation."

@app.route('/transcribe/', methods=['POST'])
def transcribe_audio():
    if request.content_type.startswith('multipart/form-data'):
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        audio_file = request.files['audio'].read()
    elif request.content_type.startswith('application/json'):
        audio_data = request.json
        if not audio_data or 'audio' not in audio_data:
            return jsonify({'error': 'No audio data provided'}), 400
        audio_base64 = audio_data['audio']
        audio_file = base64.b64decode(audio_base64.split(",")[-1])
    else:
        return jsonify({'error': 'Unsupported Content-Type'}), 415

    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_input_file, tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_output_file:
        temp_input_file.write(audio_file)
        temp_input_file.flush()  # Ensure all data is written to disk

        # Use ffmpeg to ensure audio compatibility, writing the output to a new temporary file
        command = f"ffmpeg -i {temp_input_file.name} -ac 1 -ar 16000 -y {temp_output_file.name}"
        subprocess.run(command, shell=True, check=True)

        # Transcribe the audio file using the OpenAI Whisper API
        with open(temp_output_file.name, "rb") as ready_audio_file:
            print(f'sending {temp_output_file.name} to OpenAI API')
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=ready_audio_file,
                prompt=last_enriched,
                response_format="text"
            )
            print('got',transcription)
        # Process the transcription as needed with generate_corrected_transcript

    return jsonify({"transcript": transcription, "enriched":generate_corrected_transcript(transcription)})  # Adjust based on your needs


def generate_corrected_transcript(transcribed_text):
    global last_enriched
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": "For every input text, we need Emojis, English, Chinese and Pinyin. Output all three. Fix likely speech recognition errors too. Ignore text about subtitles, or youtube common terms such as 'Please feel free to like, subscribe, share' (the speech recognition system says common terms when it can't detect obvious utterances)."
            },
            {
                "role": "user",
                "content": f"This is the input text that we need Emojis, English, Chinese and Pinyin for: `{transcribed_text}`. But ignore text like this: ` let me introduce my channel`, `thank you for watching`. For text that seems like YouTube filler or is about subtitles, or other examples given, please return an empty string only."
            }
        ]
    )
    content= response.choices[0].message.content
    print(content)
    last_enriched = last_enriched + '\n'+transcribed_text
    return content

@app.route('/<path:filename>')
def serve_file(filename):
    try:
        if not os.path.exists(os.path.join(DIRECTORY, filename)):
            abort(404)
        return send_from_directory(DIRECTORY, filename)
    except FileNotFoundError:
        abort(404)

if __name__ == '__main__':
    app.run(debug=True)
