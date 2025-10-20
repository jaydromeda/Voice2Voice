import gradio as gr
import whisper
from deep_translator import GoogleTranslator
from io import BytesIO
from gtts import gTTS
import tempfile
# Load Whisper model (choose "tiny", "base", "small", "medium", "large")

model = whisper.load_model("base")

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



def transcribe(audio):
    audio_path = audio
    result = model.transcribe(audio_path)
    return result["text"]

def translate_text(text, src_lang, target_lang):
    return GoogleTranslator(source=openai_google_langs.get(src_lang), target=openai_google_langs.get(target_lang)).translate(text)

def generate_speech(translated_text, target_lang):
    if not translated_text:
        return None

    lang_code = openai_google_langs.get(target_lang, "en")  # Default to English
    tts = gTTS(translated_text, lang=lang_code)

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts.save(tmpfile.name)
        return tmpfile.name

with gr.Blocks() as demo:
    gr.Markdown("# Speech-to-Speech Translator")
    
    with gr.Row():
        with gr.Column():
            with gr.Column():
                src_lang = gr.Dropdown(list(openai_google_langs.keys()), value = "Auto Detect", label="Original Language")
            with gr.Column():
                target_lang = gr.Dropdown(list(openai_google_langs.keys()), value = "Auto Detect", label="Target Language")
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="Record or upload audio"
            )
            btn = gr.Button("Transcribe")
            btn2 = gr.Button("Translate")
        with gr.Column():
            output_text = gr.Textbox(label="Transcription")
        with gr.Column():
            translation_text = gr.Textbox(label="Translation")
        with gr.Column():
            speech = gr.Audio(label="Generated Speech")

    btn.click(fn=transcribe, inputs=audio_input, outputs=output_text)
    btn2.click(fn=translate_text, inputs=[output_text, src_lang, target_lang], outputs= translation_text)

    btn3 = gr.Button("Generate Speech")
    btn3.click(fn=generate_speech, inputs=[translation_text, target_lang], outputs=speech)

demo.launch()
