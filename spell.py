import neuspell
from neuspell import available_checkers, BertChecker, CnnlstmChecker

""" see available checkers """
# print(f"available checkers: {neuspell.available_checkers()}")
# → available checkers: ['BertsclstmChecker', 'CnnlstmChecker', 'NestedlstmChecker', 'SclstmChecker', 'SclstmbertChecker', 'BertChecker', 'SclstmelmoChecker', 'ElmosclstmChecker']

""" select spell checkers & load """
path = ''
checker = BertChecker()
checker.from_pretrained()

""" spell correction """
sp_cr1 = checker.correct("I luk foward to receving your reply")
print(sp_cr1)
# → "I look forward to receiving your reply"
sp_cr2 = checker.correct_strings(["I luk foward to receving your reply", ])
print(sp_cr2)
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