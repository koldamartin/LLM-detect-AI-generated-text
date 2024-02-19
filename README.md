# LLM-detect-AI-generated-text
Originaly a Kaggle competition https://www.kaggle.com/competitions/llm-detect-ai-generated-text

Because of lack of training data, I downloaded an external labeled dataset with AI generated / Human written texts.

I use a Tensorflow/Keras to fine tune a basic DistilBERT model from HuggingFace. Model is then saved on Model hub.

The model is then deployed and hosted on HuggingFace Spaces via a Gradio interface.

Try web application here:  https://huggingface.co/spaces/Wintersmith/Detect_AI_generated_text
