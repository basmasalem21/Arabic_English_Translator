from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Basma21/ar-en_translator")
model = AutoModelForSeq2SeqLM.from_pretrained("Basma21/ar-en_translator")



def translate(text):

    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=50,
        num_beams=4
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)




    