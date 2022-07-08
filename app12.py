import neuspell
from neuspell import BertChecker, CnnlstmChecker
from happytransformer import HappyTextToText, TTSettings
from deepmultilingualpunctuation import PunctuationModel
from flask import Flask,render_template, request, redirect, url_for,send_file,json,jsonify
import language_tool_python
import nltk
import requests
import difflib
from nltk.tokenize import sent_tokenize
import os
import requests
import torch
import re

app = Flask(__name__)

dirn = os.path.abspath(os.path.dirname(__file__))

def spellmd(text1):

	checker_bert = BertChecker()
	checker_bert.from_pretrained("/usr/local/lib/python3.8/dist-packages/data/checkpoints/subwordbert-probwordnoise/")

	inp_txt1 = text1
	sentcor = checker_bert.correct_strings([f"{inp_txt1}", ])
	sp_cr2 = str(sentcor)[2:-2]

	# sp_cr2 = checker.correct(inp_txt1)
	return sp_cr2

def punctmodel(text2):

	model = PunctuationModel(model = f"{dirn}/fullstop-punctuation/")
	inp_txt2 = text2

	result = model.restore_punctuation(inp_txt2)
	return result

# clean_text = model.preprocess(text)
# labled_words = model.predict(clean_text)
# print(labled_words)


def grammarmd(txt3):
	happy_tt = HappyTextToText("T5", f"{dirn}/t5-base-grammar-correction/")

	args = TTSettings(num_beams=5, min_length=1, max_length=100)

	inp_txt3 = txt3

	sents = sent_tokenize(inp_txt3)


	# result = happy_tt.generate_text(f"grammar: {inp_txt3}", args=args)
	# return result.text


	lst = []
	for sent in sents:
		# Add the prefix "grammar: " before each input 
		result = happy_tt.generate_text(f"grammar: {sent}", args=args)
		res = result.text
		# res_jn = ' '.join(res)
		lst.append(res)

	sen_jn = ' '.join(lst)
	# set_rep = sen_jn.replace('  ', ' ')	

	return sen_jn


# def langpre(text4):

# 	inp_txt4 = text4

# 	url = "https://languagetool.org/api/v2/check"
	
# 	payload={'text': inp_txt4, 'language': 'en-us'}
# 	files=[]
#     headers = {}

#     response = requests.request("POST", url, headers=headers, data=payload, files=files)
#     x = response.text
# 	return x

def diffchecker(d1, d2):
    split_data_1= d1.split()
    split_data_2 = d2.split()
    d = difflib.SequenceMatcher(None, split_data_1, split_data_2, autojunk=None)
    changes = [op for op in d.get_opcodes()]

    raw_data_list =[]

    for change in changes:
        if change[0]=="equal":
            same=" ".join(split_data_1[change[1]:change[2]])
            raw_data_list.append(same)
        else:
            changed = f'<b>{" ".join(split_data_2[change[3]:change[4]])}</b>'
            raw_data_list.append(changed)
    diff_data = " ".join(raw_data_list)
    remove_empty = diff_data.replace('<b></b>','')
    cleaned_data=re.sub(' +', ' ',remove_empty)
    return remove_empty

@app.route('/', methods=['GET', 'POST'])
def home():
	return 'I am running'


@app.route('/grammar_check', methods=['GET', 'POST'])
def grammarfn():
	if request.method == 'POST':
		txt = request.form['text']

		sp1 = spellmd(txt)
		punct = punctmodel(sp1)
		t5_r = grammarmd(punct)
		# langt = language(t5_r)
		newlineid = id('new_line_pattern')
		content = txt.replace('\n', f" {newlineid} ")
		results = t5_r.replace('\n', f' {newlineid} ')  
		difference = diffchecker(content, results)
		final_result = difference.replace(str(newlineid), '\n')

		dic = dict()
		dic['original_content'] = txt
		dic['grammar_correction'] = final_result

		return dic

if __name__== '__main__':
    app.run(debug=True)




""" see available checkers """
# print(f"available checkers: {neuspell.available_checkers()}")
# → available checkers: ['BertsclstmChecker', 'CnnlstmChecker', 'NestedlstmChecker', 'SclstmChecker', 'SclstmbertChecker', 'BertChecker', 'SclstmelmoChecker', 'ElmosclstmChecker']

""" select spell checkers & load """

""" spell correction """
# sp_cr1 = checker.correct()
# print(sp_cr1)

# → ["I look forward to receiving your reply"]
# checker.correct_from_file(src="noisy_texts.txt")
# # → "Found 450 mistakes in 322 lines, total_lines=350"

# """ evaluation of models """
# checker.evaluate(clean_file="bea60k.txt", corrupt_file="bea60k.noise.txt")
# → data size: 63044
# → total inference time for this data is: 998.13 secs
# → total token count: 1032061
# → confusion table: corr2corr:940937, corr2incorr:21060,
#                    incorr2corr:55889, incorr2incorr:14175
# → accuracy is 96.58%
# → word correction rate is 79.76%