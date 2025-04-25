import re

text = ("<s> The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced `` no evidence '' that any irregularities took place . </s> "
        "<s> The couple was married Aug. 2 , 1913 . </s> ")
pattern = r"<s>(.*?)</s>"
token_pattern = r'<s>|</s>|\b\w+\b'

# Use re.match to find a match
match = re.findall(pattern, text)[0]
print(match)

match = re.findall(token_pattern, text)
print(match)

text = "axxxbxxxab"
match_greedy = re.search(r"^a.*b$" , text)
print("贪婪模式匹配结果:", match_greedy.group())
match_non_greedy = re.search(r"^a.*?b$" , text)
print("非贪婪模式匹配结果:", match_non_greedy.group())

# if match:
#     print("Matched content:", match.group(1).strip())
# else:
#     print("No match found.")

#
# with open('./brown_100.txt', 'r') as file_:
#     corpus = file_.read()
#
# text =corpus.splitlines()
# print(text[:1])
# pattern = r"<s>(.*?)</s>"
# for line in text[:3]:
#     print(line)
#     line = re.findall(line, pattern)
#     print(line)



import string

words = ["Hello", ",", "world", "!", "This $", "</s>","is", "a", "test", ".", "Sentence", "?"]

def remove_punctuation(word_list):
    """Removes punctuation from a list of words."""
    punctuation = string.punctuation

    return [word for word in word_list if word not in punctuation]

cleaned_words = remove_punctuation(words)
print(cleaned_words)  # Output: ['Hello', 'world', 'This', 'is', 'a', 'test', 'Sentence']