from transformers import T5ForConditionalGeneration, T5Tokenizer
project_dir = os.path.abspath(os.path.dirname(__file__))

tokenizer = T5Tokenizer.from_pretrained("/t5-small-tapaco")
model = T5ForConditionalGeneration.from_pretrained("hetpandya/t5-small-tapaco")

def get_paraphrases(sentence, prefix="paraphrase: ", n_predictions=5, top_k=40, max_length=256,device="cpu"):
    text = prefix + sentence + " </s>"
    encoding = tokenizer.encode_plus(
        text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding[
    "attention_mask"
    ].to(device)

    model_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=True,
        max_length=max_length,
        top_k=top_k,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=n_predictions,
        )

    outputs = []
    for output in model_output:
        generated_sent = tokenizer.decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if (
            generated_sent.lower() != sentence.lower() and generated_sent not in outputs):
            outputs.append(generated_sent)
    return outputs


paraphrases = get_paraphrases("He was a man of his words and the greatest spokesman, he always stood like a rock in front of opponents and never lay down. Gandhi called him “an impossible man ” due to his determinacy over his principles. Jinnah said: “Think a hundred times before you make a decision, but once that decision is taken, stand by it as one man”.In 1930 he became an undisputed leader of all the Muslims of sub-continent and started to lead Muslim League in 1933. In 1940 Pakistan resolution was drafted by Muslim league at Minar e Pakistan -Lahore which has been proved as a backbone in the war of freedom.")

for sent in paraphrases:
  print(sent)