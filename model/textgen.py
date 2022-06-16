import numpy

def generate_text(model, prompt, n_texts, text_len):
    out = model.generate(
    input_ids=prompt,
    max_length=text_len,
    num_beams=5,
    do_sample=True,
    temperature=10.,
    top_k=50,
    top_p=0.6,
    no_repeat_ngram_size=3,
    num_return_sequences=n_texts,
    ).numpy()
    return out