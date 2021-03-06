from flask import Flask, request, jsonify, render_template, send_file
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

app = Flask(__name__)


@app.route('/')
def home():
    return 'I am running'

@app.route('/rmodel', methods=['POST', 'GET'])
def raw_model():
    if request.method='POST':
        txt = request.form['text']

        model = AutoModelForSeq2SeqLM.from_pretrained("mlwork/phrmodel/t5_large/")
        tokenizer = AutoTokenizer.from_pretrained("mlwork/phrmodel/t5_large/")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print ("device ",device)
        model = model.to(device)
        dic = dict()
        # Beam Search

        # context = "Once, a group of frogs were roaming around the forest in search of water."

        context = txt
        text = "paraphrase: "+context + " </s>"

        encoding = tokenizer.encode_plus(text,max_length =128, padding=True, return_tensors="pt")
        input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        model.eval()
        beam_outputs = model.generate(
            input_ids=input_ids,attention_mask=attention_mask,
            max_length=128,
            early_stopping=True,
            num_beams=15,
            num_return_sequences=3

        )

        lst1 = []
        # print ("\n\n")
        # print ("Original: ",context)
        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            lst1.append(sent)

        dic['paraphrase_output'] = lst1    

        return jsonify(output=dic)



# Diverse Beam search

# context = "In the year of 1928, the reputation of Sir Allama Iqbal was solidly established and he delivered lectures at Hyderabad, Madras, and Aligarh. The cherry on the top was, this lecture was published as a book named” the reconstruction of Religious Thought in Islam”. In 1932 Iqbal came to England as a Muslim delegate to the Third Round Table Conference. When Quaid e Azam Muhammad Ali Jinnah was in England, Mr. Iqbal Persuaded him to come and asked for his personal views on problems and the Indian state of affairs. His letter was powerful with irreplaceable words and power of thoughts."
# text = "paraphrase: "+context + " </s>"

# encoding = tokenizer.encode_plus(text, max_length=102, padding=True, return_tensors="pt")
# input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

# model.eval()
# diverse_beam_outputs = model.generate(
#     input_ids=input_ids,attention_mask=attention_mask,
#     max_length=150,
#     early_stopping=True,
#     num_beams=5,
#     temperature=0.96,
#     top_p=0.95,
#     no_repeat_ngram_size=4,
#     num_beam_groups = 5,
#     num_return_sequences=2,
#     diversity_penalty = 0.70

# )

# print ("\n\n")
# print ("Original: ",context)
# for beam_output in diverse_beam_outputs:
#     sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
#     print(sent)


@app.route('/pegasus', methods=['GET', 'POST'])
def pegasus_model():
    if request.method== 'POST':
        txt = request.form['text']

        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model_name = 'tuner007/pegasus_paraphrase'

        tokenizer = PegasusTokenizer.from_pretrained('/var/www/html/mlwork/phrmodel/pegasus_paraphrase/')
        model = PegasusForConditionalGeneration.from_pretrained('/var/www/html/mlwork/phrmodel/pegasus_paraphrase/').to(torch_device)

        def get_response(input_text=txt, num_return_sequences=4, num_beams=10):
            batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
            translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
            dic = dict()
            dic['paraphrase_output'] = tgt_text 

            return jsonify(output=dic)


        # num_beams = 10
        # num_return_sequences = 4
        # # context = "Once, a group of frogs were roaming around the forest in search of water."
        # context = txt
        # resp = get_response(context,num_return_sequences,num_beams)
        # return resp

if __name__== '__main__':
    app.run(debug=True)





# !pip install transformers==4.10.2
# !pip install sentencepiece==0.1.96    