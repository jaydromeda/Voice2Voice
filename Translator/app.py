import gradio as gr
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import os
import requests
import soundfile as sf
import librosa
from pathlib import Path
from openai import OpenAI

# Load Whisper models
model = whisper.load_model("base", device="cuda")

# get current folder
current_dir = os.path.dirname(os.path.abspath(__file__))

# get parent folder
parent_dir = os.path.dirname(current_dir)

model_file_path = os.path.join(parent_dir, "/VoiceModels/HomerSimpsonModel.pth")
index_file_path = os.path.join(parent_dir, "/VoiceModels/HomerSimpson.index")

openai_google_langs = {
    "Auto Detect": "auto",
    "Afrikaans": "af",
    "Arabic": "ar",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Belarusian": "be",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Chinese": "zh-CN",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "German": "de",
    "Greek": "el",
    "Hebrew": "iw",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Malay": "ms",
    "Marathi": "mr",
    "Maori": "mi",
    "Nepali": "ne",
    "Norwegian": "no",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian": "sr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tamil": "ta",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
    "Welsh": "cy"
}

def configure_rvc_model(pth_path, index_path):
    """Configure the RVC API with model paths"""
    try:
        # Get available devices
        input_devices = requests.get("http://127.0.0.1:6242/inputDevices", timeout=5).json()
        output_devices = requests.get("http://127.0.0.1:6242/outputDevices", timeout=5).json()
        
        config = {
            "pth_path": pth_path,
            "index_path": index_path,
            "sg_input_device": input_devices[0],
            "sg_output_device": output_devices[0],
            "pitch": 0,
            "f0method": "fcpe",
            "index_rate": 0.3,
            "rms_mix_rate": 0.0,
            "block_time": 0.25,
            "crossfade_length": 0.05,
            "extra_time": 2.5,
            "n_cpu": 4,
            "threhold": -60,
            "formant": 0.0,
            "I_noise_reduce": False,
            "O_noise_reduce": False,
            "use_pv": False
        }
        
        # Stop any existing conversion
        try:
            requests.post("http://127.0.0.1:6242/stop", timeout=5)
        except:
            pass
        
        # Configure
        resp = requests.post("http://127.0.0.1:6242/config", json=config, timeout=10)
        
        if resp.status_code == 200:
            return f"RVC configured successfully!\nInput: {input_devices[0]}\nOutput: {output_devices[0]}"
        else:
            return f"Configuration failed: {resp.text}"
            
    except requests.exceptions.ConnectionError:
        return "Cannot connect to RVC API. Make sure api_240604.py is running on port 6242"
    except Exception as e:
        return f"Configuration failed: {str(e)}"

def transcribe(audio):
    audio_path = audio
    result = model.transcribe(audio_path)
    return result["text"]

def translate_text(text, src_lang, target_lang):
    if not text:
        return ""
    return GoogleTranslator(
        source=openai_google_langs.get(src_lang), 
        target=openai_google_langs.get(target_lang)
    ).translate(text)

def generate_speech(translated_text, target_lang, voice_model):
    if not translated_text:
        return None
    
    lang_code = openai_google_langs.get(target_lang, "en")
    """"
    if voice_model != "Google":
        client = OpenAI()
        speech_file_path = Path(__file__).parent / "speech.wav"

        with client.audio.speech.with_streaming_response.create(
            model = "gpt-40-mini-tts",
            voice = voice_model,
            input = translated_text,
            instructions = "Calm"
        ) as response:
            response.stream_to_file(speech_file_path)
            return speech_file_path
    """
    tts = gTTS(translated_text, lang=lang_code)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tts.save(tmpfile.name)
        return tmpfile.name

def convert_voice_with_rvc(input_audio, model_pitch=0, speed_adjust=1.0):
    try:
        if isinstance(input_audio, tuple):
            input_audio = input_audio[0]
        
        if not input_audio or not os.path.exists(input_audio):
            print(f"Invalid input_audio path: {input_audio}")
            return None

        # only need to load/save if speed_adjust is NOT 1.0
        if speed_adjust != 1.0:
            print(f"Applying time stretch: {speed_adjust}x")
            
            # Load the audio file into a numpy array
            y, sr = librosa.load(input_audio, sr=None) 
            
            # Apply time stretch
            y_stretched = librosa.effects.time_stretch(y, rate=speed_adjust)
            
            # Save to a temporary file to send to the API
            temp_processed = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            sf.write(temp_processed, y_stretched, sr)
            
            # Update input_audio to point to this new processed file
            audio_to_send = temp_processed
        else:
            # If no speed change, just use the original file
            audio_to_send = input_audio

        # send to RVC API
        infer_url = "http://127.0.0.1:6242/infer"
        print(f"Using pitch shift: {model_pitch} semitones")
        print(f"📡 Sending request to {infer_url}...")

        with open(audio_to_send, "rb") as f:
            files = {"file": (os.path.basename(audio_to_send), f, "audio/wav")}
            data = {
                "f0up_key": str(model_pitch),  
                "f0method": "fcpe",
                "index_rate": "0.3",
                "rms_mix_rate": "0.0"
            }
            response = requests.post(infer_url, files=files, data=data, timeout=60)

        # clean up temp file if we created one for speed adjustment
        if speed_adjust != 1.0 and os.path.exists(audio_to_send):
            os.remove(audio_to_send)

        # handle response
        if response.status_code == 200:
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with open(output_path, "wb") as out_f:
                out_f.write(response.content)
            print(f"Voice conversion complete: {output_path}")
            return output_path
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        import traceback
        print(f"Voice conversion failed: {e}")
        traceback.print_exc()
        return None

def full_pipeline(audio, src_lang, target_lang, pitch=0):
    """Complete pipeline: transcribe -> translate -> TTS -> RVC"""
    try:
        print("Starting full pipeline...")
        
        # 1. Transcribe
        print("Transcribing")
        text = transcribe(audio)
        if not text:
            return "Transcription failed", "", None, None
        
        # 2. Translate
        print("translating")
        translated = translate_text(text, src_lang, target_lang)
        if not translated:
            return text, "Translation failed", None, None
        
        # 3. Generate TTS
        print("Generating speech")
        tts_audio = generate_speech(translated, target_lang)
        if not tts_audio:
            return text, translated, None, None
        
        # 4. Apply RVC
        print("Covnerting voice")
        rvc_audio = convert_voice_with_rvc(tts_audio, pitch)
        
        return text, translated, tts_audio, rvc_audio
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", "", None, None


with gr.Blocks(gr.themes.Ocean()) as demo:
    gr.Markdown("# Voice2Voice Translator")
    
    with gr.Accordion("RVC Configuration", open=True):
        with gr.Row():
            pth_input = gr.Textbox(
                label="Model Path", 
                value=model_file_path,
            )
            index_input = gr.Textbox(
                label="Index Path", 
                value=index_file_path,
            )
        
        config_button = gr.Button("Configure RVC Model", variant="primary")
        config_status = gr.Textbox(label="Configuration Status", interactive=False)
        
        config_button.click(
            fn=configure_rvc_model,
            inputs=[pth_input, index_input],
            outputs=config_status
        )
        with gr.Column():
            gr.Markdown("### *Run RVC api_240604.py Concurrently")
    
    gr.Markdown("---")
    
    # main interface
    with gr.Row():
        with gr.Column():
            src_lang = gr.Dropdown(
                list(openai_google_langs.keys()), 
                value="Auto Detect", 
                label="Original Language"
            )
            target_lang = gr.Dropdown(
                list(openai_google_langs.keys()), 
                value="English", 
                label="Target Language"
            )
            voice_model = gr.Dropdown(
                ["Google", "Coral", "Alloy", "Ballad", "Echo", 
                 "Fable", "Nova", "Onyx", "Sage", "Shimmer"],
                value="Google",
                label="Text-to-Speech"
            )
            pitch_shift = gr.Slider(
                minimum=-12,
                maximum=12,
                value=0,
                step=1,
                label="Pitch Shift"
            )
            speed_adjust = gr.components.Slider(
                minimum=.1,
                maximum=3,
                value=1,
                step=.05,
                interactive=True,
                label="Speed Adjustment"
            )

        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or Upload Audio"
            )

    # buttons
    with gr.Row():
        transcribe_button = gr.Button("Transcribe", size="sm", variant="primary")
        translate_button = gr.Button("Translate", size="sm", variant="primary")
        tts_button = gr.Button("Generate Speech", size="sm", variant="primary")
        rvc_button = gr.Button("Apply Voice Conversion", size="sm", variant="primary")
        full_pipeline_button = gr.Button("Run Full Pipeline", variant="primary")

    # outputs
    with gr.Row():
        with gr.Column():
            transcription_output = gr.Textbox(label="Transcription", lines=3)
            translation_output = gr.Textbox(label="Translation", lines=3)
        
        with gr.Column():
            tts_output = gr.Audio(label="Generated Speech (TTS)", type="filepath")
            rvc_output = gr.Audio(label="RVC Converted Voice", type="filepath")

    # button functions
    transcribe_button.click(
        fn=transcribe, 
        inputs=audio_input, 
        outputs=transcription_output
    )
    
    translate_button.click(
        fn=translate_text, 
        inputs=[transcription_output, src_lang, target_lang], 
        outputs=translation_output
    )
    
    tts_button.click(
        fn=generate_speech, 
        inputs=[translation_output, target_lang, voice_model], 
        outputs=tts_output
    )
    
    rvc_button.click(
        fn=convert_voice_with_rvc,
        inputs=[tts_output, pitch_shift, speed_adjust],
        outputs=rvc_output
    )
    
    # pipeline function
    full_pipeline_button.click(
        fn=full_pipeline,
        inputs=[audio_input, src_lang, target_lang, pitch_shift],
        outputs=[transcription_output, translation_output, tts_output, rvc_output]
    )



demo.launch()