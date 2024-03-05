# LLM detect AI generated text
[Try web application here:](https://huggingface.co/spaces/Wintersmith/Detect_AI_generated_text)

![gradio app](https://github.com/koldamartin/LLM-detect-AI-generated-text/assets/68967537/09b576f5-ca01-4b5f-ac07-630e72c4bffc)

### Overview:
Originaly a [Kaggle competition](https://www.kaggle.com/competitions/llm-detect-ai-generated-text).
Model trained on thousands of AI generated / student written essays. Try to input a text and the model should tell you the probability of who wrote the essay. Either student or AI.

### Features:

Because of lack of training data, I downloaded an external labeled dataset with AI generated / Human written texts.

I use a Tensorflow/Keras to fine tune a basic DistilBERT model from HuggingFace. Model is then saved on Model hub.

User interface made in Gradio.

The model is deployed and hosted on HuggingFace Spaces





