import subprocess
import gradio as gr
import numpy as np
import faster
import os
import subprocess
import importlib.util
import json

from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


#pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.float16, device="cuda")

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.float16, device=0)
#pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", device="cuda")
#pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    #del variables

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")

def import_script(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def read_file_content(filename):
    with open(filename, 'r') as file:
        return file.read()

def summarize(transcribed_text):
    print("before prompt")
    prompt_template = """Based on the provided conversation, your task is to summarize the key findings and derive insights. The diarization may not be 100 percent accurate, so take into consideration the conversation. Please create a thorough summary note under the heading 'SUMMARY KEY NOTES' and include bullet points about the key items discussed.
    Ensure that your summary is clear and informative, conveying all necessary information (include how the caller was feeling, meaning sentiment analysis). Focus on the main points mentioned in the conversation, such as Claims, Benefits, Providers, and other relevant topics. Additionally, create an action items/to-do list based on the insights and findings from the conversation.
    The main points to look for in a conversation are: Claims, Correspondence and Documents, Eligibility & Benefits, Financials, Grievance & Appeal, Letters, Manage Language, Accumulators, CGHP & Spending Account Buy Up, Group Search, Member Enrollment & Billing, Manage ID Cards, Member Limited Liability, Member Maintenance, Other Health insurance (COB), Provider Lookup, Search/ Update UM Authorization, Prefix and Inter Plan Search, Promised Action Search Inventory.
    Please note that while you can look for other points, it is important to prioritize the main points mentioned above.

        """

    print("before message")
    messages = [
        {
            "role": "system",
            "content": prompt_template,
        },
        {"role": "user", "content": transcribed_text},
    ]

    print("before tokenizer")
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("before pipe output")
    outputs = pipe(prompt, max_new_tokens=2000, do_sample=True, temperature=0.5, top_k=50, top_p=0.95)
    print("before generated text (all)")
    my_string = outputs[0]["generated_text"]
    print("before answer (split)")
    summarizing = my_string.split("<|assistant|>",1)[1]
    print("before return")
    clear_gpu_memory()
    return summarizing

text = ""
total = ""
def transcribe(audio, state=""):
    #global state
    global text
    global total
    time.sleep(3)
    text = transcriber(audio)["text"]
    state += text + " "
    total = state
    torch.cuda.empty_cache()
    gc.collect()
    return state, state

def aud_summarize2():
    global total
    audio_summarized = summarize(total)
    torch.cuda.empty_cache()
    gc.collect()
    return audio_summarized

def transcribe2(audio_file):
    if audio_file:
        head, tail = os.path.split(audio_file)
        path = head

        if tail[-3:] != 'wav':
            subprocess.call(['ffmpeg', '-i', audio_file, "audio.wav", '-y'])
            tail = "audio.wav"

        subprocess.call(['ffmpeg', '-i', audio_file, "audio.wav", '-y'])
        tail = "audio.wav"
        print("before diarize")
        faster.run_cli()
        #import convertJson

        # Read JSON data from file
        with open('output.json', 'r') as file:
            json_data = json.load(file)

        # Now you can call functions from your script as:
        script_module = import_script('convertJson.py')
        # Assume `process_transcript` is a function in convertJson.py
        srt_content = script_module.process_transcript(json_data)

        # Write to SRT file
        with open('output.srt', 'w') as srt_file:
            srt_file.write(srt_content)

        with open('output.srt', 'r') as file:
            input_data = file.read()

        # Now you can call the function directly
        script_module = import_script('convert_text.py')
        # Assume `convert_srt_to_dialogue` is a function in convertJson.py
        text_content = script_module.convert_srt_to_dialogue(input_data)

        # Write to a text file
        with open('output.txt', 'w') as text_file:
            text_file.write(text_content)
        #filename = 'convertJson.py'
        #with open(filename, 'r') as file:
        #    exec(file.read())

        srt = read_file_content('output.srt')
        text = read_file_content('output.txt')
        clear_gpu_memory()
        torch.cuda.empty_cache()
        gc.collect()
        summarized = summarize(text)
        return(srt, summarized)

with gr.Blocks() as demo:

    gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(sources="microphone", type="filepath", streaming=True),
            'state'
        ],
        outputs=[
            "textbox",
            "state"
        ],
        title="Real-Time Transcription and Summarization",
        live=True
    )

    output_text = gr.Textbox(label="Summary 2:")
    greet_btn = gr.Button("Get Summary 2")
    greet_btn.click(fn=aud_summarize2, inputs=[], outputs=output_text)

    gr.Interface(transcribe2,
        inputs=[
            gr.Audio(sources ='upload', type='filepath', label='Audio File'),

            ],
        outputs=["text", "text"],
        title="Transcribe and Summarize Files"
    )

    gr.Interface(summarize,
            inputs="text",
            outputs="text",
            title="Summarize Transcription")

demo.queue().launch(debug=True, share=True, inline=False)
