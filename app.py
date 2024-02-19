import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import gradio as gr
import re

tokenizer = AutoTokenizer.from_pretrained("Wintersmith/LLM_generated_text_detector")
model = TFAutoModelForSequenceClassification.from_pretrained("Wintersmith/LLM_generated_text_detector")

def clean_text(text):
  text = re.sub(r"[^A-Za-z0-9\s]", "", text)
  text = re.sub(r"\s+", " ", text)
  text = text.lower()
  return text

def get_probabilities(input_text):
    cleaned_text = clean_text(input_text)
    model_input = tokenizer(cleaned_text, max_length=512, padding=True, truncation=True, return_tensors='tf')
    model_input = dict(model_input)
    logits_pred = model.predict(model_input)['logits']
    probs = tf.nn.sigmoid(logits_pred)
    probs_list = list(probs[0].numpy())
    class_names = ["Written by student", "AI generated"]
    return {class_names[i]: probs_list[i] for i in range(2)}

student_written_eassay = "Most schools allow students to have cell phones for safety, which seems unlikely to change as long as school shootings remain a common occurrence. But phones aren't just tools for emergencies; they can also be valuable tools in the classroom. If there's a word or concept a student doesn't understand, the student can find information instantly. Phones have calculators as well as spelling and grammar checks. Most importantly, phones allow students to communicate with one another and with experts in fields of interest. The question remains, however, whether the use of cell phones in school outweighs the distraction they cause."

ai_generated_essay = "In today’s digital age, the presence of smartphones in schools has sparked heated debates among educators, parents, and policymakers. While some argue that banning phones is essential for maintaining classroom focus, others emphasize the need to consider students’ sense of connection and adapt to the changing educational landscape."

description = """This is a Distilbert model fine-tuned on thousands of essays of various themes like 'Phones in school', 'Car-free cities' etc. <img src="https://image.khaleejtimes.com/?uuid=99d0d917-5420-4344-84d8-4fabd2578882&function=cropresize&type=preview&source=false&q=75&crop_w=0.99999&crop_h=0.75&x=0&y=0&width=1500&height=844" width=300px>"""

article = "This app is based on a Kaggle competition, learn more about it [here](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)."

iface = gr.Interface(
    fn=get_probabilities,
    inputs="text",
    outputs=gr.Label(),
    title="Detect AI Generated Text - Identify which essay was written by a LLM",
    examples = [student_written_eassay, ai_generated_essay],
    description=description,
    article=article,
)

iface.launch(share=True)
