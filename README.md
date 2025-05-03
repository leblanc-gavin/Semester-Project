Mini-Storyteller: GPT-2 Author Style Generator

Project Summary

- Mini-Storyteller is a transformer-based text generation system that fine-tunes GPT-2 to create fictional short stories (up to 100 words) in the style of famous authors. This project focuses on two distinctive literary voices: Jack London and Lewis Carroll. It takes a prompt and generates a full micro-fiction narrative reflecting each author's stylistic tendencies.

Goals

- Train a language model to distinguish and reproduce the stylistic traits of Jack London and Lewis Carroll.

- Generate short stories (up to 100 words) from simple prompts.

- Demonstrate how fine-tuned transformers can capture narrative tone, pacing, and language specific to different authors.

Methods

- Preprocessed text data from Project Gutenberg for both authors.

- Added author tokens (e.g., <|JackLondon|>) to guide style conditioning.

- Used HuggingFace Transformers and fine-tuned GPT-2 / GPT-2 Medium / GPT-2 Large.

- Trained over multiple epochs with different batch sizes and learning rates to find optimal performance without GPU support.

- Evaluated using perplexity and human evaluation for coherence, length, and stylistic fidelity.

Results & Conclusion

- Training on the base GPT-2 model yielded short but coherent outputs.

- Switching to GPT-2 Medium and GPT-2 Large improved fluency and stylistic accuracy.

- Despite hardware limitations (CPU-only), careful tuning of max_length, prompt engineering, and training steps led to outputs averaging 30–50 words.

- Future improvements would include gradient accumulation and longer training cycles on GPU hardware.

![image](https://github.com/user-attachments/assets/416e9865-53ec-42ac-8420-0cf799d6b97e)
![image](https://github.com/user-attachments/assets/2411a523-a59b-41db-9c23-313f81edb797)

