from nltk.corpus import brown as b
import re
from collections import Counter
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from nltk import word_tokenize, pos_tag,FreqDist
import matplotlib.pyplot as plt
import spacy


print(b.categories())# 打印分类名
genres_token_dict = defaultdict(list)
genres_sents_dict = defaultdict(list)
whole_token = []
whole_sents = []
for file in b.fileids():
    file_text = b.words(file)
    file_sents = b.sents(file)
    # print("yes",file_sents[0:2][0:50])

    category = b.categories(file)[0]
    genres_token_dict[category] = genres_token_dict[category] + list(file_text)  # 将相同类别的文本添加到对应的列表中
    genres_sents_dict[category] = genres_sents_dict[category] + list(file_sents)

    whole_token += list(file_text)
    whole_sents += list(file_sents)

genres_token = [(category, texts) for category, texts in genres_token_dict.items()]  # 将字典转换为列表形式
genres_sents = [(category, texts) for category, texts in genres_sents_dict.items()]  # 将字典转换为列表形式
# print("2",genres_sents[0][0:50],"big lsit")





#step1
class NGramModel_step1:
    def __init__(self, text, n, alpha=0.0):
        """
        Initialize the NGramModel with text and the value of n.
        """
        self.text = text
        self.n = n
        self.alpha = alpha  # Alpha value for additive smoothing
        self.ngrams = {}
        self.probabilities = {}
        self.vocab = set()

    def tokenize(self) -> list:
        """
        Tokenize the text into words.
        Fill in the code to split the text into a list of words.
        """

        token_pattern = r'<s>|</s>|\b\w+\b'

        tokens = re.findall(token_pattern, self.text)

        return tokens

    def generate_ngrams(self, tokens: list) -> dict:
        """
        Generate n-grams from the list of tokens.
        Fill in the code to create n-grams.
        Make sure to treat each sentence independently, include the <s> and </s> tokens.
        """

        if self.n == 1:
            ngram = [i for i in tokens]
        elif self.n == 2:
            ngram = [" ".join(tokens[i:i + 2]) for i in range(len(tokens))]
        elif self.n == 3:
            ngram = [" ".join(tokens[i:i + 3]) for i in range(len(tokens))]
        self.ngrams = {"n-gram": ngram}

        return self.ngrams

    def count_frequencies(self):
        """
        Count the frequencies of each n-gram.
        Fill in the code to count n-gram occurrences.
        """

        for i in self.ngrams.values():
            result = Counter(i)
            result = sorted(result.items(), key=lambda item: item[1], reverse=True)
            print(result)
        return result

    def calculate_probabilities(self) :
        """
        Calculate probabilities of each n-gram based on its frequency. Add alpha smoothing separately.
        """
        for i in self.ngrams.values():
            counts = Counter(i)
            total_count = sum(counts.values())
            probabilities = {item: count / total_count for item, count in counts.items()}
        self.probabilities = probabilities
        print(probabilities)

        return self.probabilities

    def most_frequent_ngrams(self, top_n: int = 10) -> list:
        """
        Return the most frequent n-grams and their probabilities.
        """

        top_n_grams = sorted(self.probabilities.items(), key=lambda item: item[1], reverse=True)[:top_n]
        print(top_n_grams)

        return top_n_grams

def step1():
    genres = genres_token
    for n in range(3):
        n += 1
        text = genres[0][1]  #  genres[k][1] or whole
        name = genres[0][0]  # genres[k][0] or "corpus"
        corpus = NGramModel_step1(text, n)
        # ngrams = corpus.generate_ngrams(text)
        # print(ngrams)
        fre = corpus.count_frequencies()
        # p = corpus.calculate_probabilities()




        if n == 1:
            with open(f"{name}_unigarm", "w")as f:
                for i in fre:
                    f.write(f"{i} \n")

        elif n == 2:
            with open(f"{name}_bigarm", "w")as f:
                for i in fre:
                    f.write(f"{i} \n")

        elif n == 3:
            with open(f"{name}_trigarm", "w")as f:
                for i in fre:
                    f.write(f"{i} \n")




#step2,3,4
class NGramModel_step2:
    def __init__(self, text,sents, n, alpha=0.0):
        """
        Initialize the NGramModel with text and the value of n.
        """
        self.text = text
        self.sents = sents
        self.n = n
        self.alpha = alpha  # Alpha value for additive smoothing
        self.ngrams = {}
        self.fre = 0
        self.probabilities = {}
        self.vocab = set()

    def tokenize(self) -> list:
        """
        Tokenize the text into words.
        Fill in the code to split the text into a list of words.
        """

        token_pattern = r'<s>|</s>|\b\w+\b'

        tokens = re.findall(token_pattern, self.text)

        return tokens

    def generate_ngrams(self, tokens: list) -> dict:
        """
        Generate n-grams from the list of tokens.
        Fill in the code to create n-grams.
        Make sure to treat each sentence independently, include the <s> and </s> tokens.
        """

        if self.n == 1:
            ngram = [i for i in tokens]
        elif self.n == 2:
            ngram = [" ".join(tokens[i:i + 2]) for i in range(len(tokens))]
        elif self.n == 3:
            ngram = [" ".join(tokens[i:i + 3]) for i in range(len(tokens))]
        self.ngrams = {"n-gram": ngram}

        return self.ngrams

    def count_frequencies(self):
        """
        Count the frequencies of each n-gram.
        Fill in the code to count n-gram occurrences.
        """

        # for i in self.text:
        #     result = Counter(i)
        #     print("number of token is :",len(result))
        #     result = sorted(result.items(), key=lambda item: item[1], reverse=True)

        return len(self.text)

    def calculate_probabilities(self) :
        """
        Calculate probabilities of each n-gram based on its frequency. Add alpha smoothing separately.
        """
        for i in self.ngrams.values():
            counts = Counter(i)
            total_count = sum(counts.values())
            probabilities = {item: count / total_count for item, count in counts.items()}
        self.probabilities = probabilities
        print(probabilities)

        return self.probabilities

    def most_frequent_ngrams(self, top_n: int = 10) -> list:
        """
        Return the most frequent n-grams and their probabilities.
        """

        top_n_grams = sorted(self.probabilities.items(), key=lambda item: item[1], reverse=True)[:top_n]
        print(top_n_grams)

        return top_n_grams

    def is_word(self,s) -> bool:
        """
        Determine if the given string is a word.
        """
        word_pattern = r'^\b\w+\b$'  # 匹配单词
        return bool(re.match(word_pattern, s))

    def count_words(self):
        """
        Count the number of unique words in the text using regex.
        """
        tt_word_len = 0
        word_list = []
        for i in self.text:
            if self.is_word(i):
                word_list.append(i)
                tt_word_len += len(i)
        word_con =  len(word_list)
        avg_word_len = tt_word_len/word_con

        return word_con,avg_word_len
    def count_type(self):
        """
        Count the number of types of corpus. it's the words without repetition and punctuation.
        """
        for i in self.text:
            if self.is_word(i):
                self.vocab.add(i)
        return len(self.vocab)

    def avg_sent_words(self):
        sents_words = []
        for sent in self.sents:
            word_in_sent = 0
            for token in sent:
                if self.is_word(token):
                    word_in_sent += 1
            sents_words.append(word_in_sent)
        if len(sents_words) == 0:
            print(self.sents[0:20])
        return sum(sents_words)/len(sents_words)

    def count_lemma(self):
        nlp = spacy.load("en_core_web_sm")
        lemma_set = set()
        for sent in self.sents:
            words = nlp(" ".join(sent))
            for token in words:
                if not token.is_punct:
                    lemma = token.lemma_
                    lemma_set.add(lemma)

        return len(lemma_set)

    def pos_tag_count(self):
        total_pos= []
        for tokens in self.sents:
            tagged = pos_tag(tokens)
            # print(tagged)
            pos_tags = [tag for word, tag in tagged]
            total_pos += pos_tags
        pos_tag_counts = Counter(total_pos)
        freq_dist = FreqDist(pos_tag_counts)
        pos_frq_10 = freq_dist.most_common(10)
        pos_frq_10 = [f"{tag}:{count}" for tag, count in pos_frq_10]
        # print(pos_tag_counts)
        return pos_frq_10





def save_to_file(task, name, result):
    with open(f"{task}", "a") as f:
        f.write(f"{name} : {result}\n")



def plot_and_save(data, filename):
    """
    绘制并保存两张图，一张是线性坐标图，另一张是log坐标图。

    参数:
    data (list): 数值列表。
    filename (str): 保存的文件名前缀。
    """
    # 创建第一个图形，线性坐标图
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o')
    plt.title('Linear Scale Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(f'{filename}_linear.png')
    plt.close()

    # 创建第二个图形，log坐标图
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o')
    plt.yscale('log')
    plt.title('Log Scale Plot')
    plt.xlabel('Index')
    plt.ylabel('Value (log scale)')
    plt.grid(True)
    plt.savefig(f'{filename}_log.png')
    plt.close()

# 示例调用
# plot_and_save([1, 10, 100, 1000, 10000], 'example_plot')



# the following are step 2&3

# files = genres_token.copy()
# files.append(("corpus",whole_token))
files = genres_sents.copy()
files.append(("corpus",whole_sents))

# task = "step1"
# task = "word_count"
# task = "token_count"
# task = "type_count"
# task = "sentence_word_avg"
# task = "lemma_count"
# task = "pos"
task = "curve"

for i in files:
    for k in range(1):
        n = 1
        sents = i[1]
        # print(sents[0:10],"sents")
        text = [item for sents in sents for item in sents]
        # print("text", text[0:10])
        # text = i[1]  #  genres[k][1] or whole
        name = i[0]  # genres[k][0] or "corpus"
        corpus = NGramModel_step2(text, sents, n)
        ngrams = corpus.generate_ngrams(text)
        # # print(ngrams)
        # corpus.count_frequencies()
        if task == "step1":
            step1()
        if task[0] == "w":
            word_count,avg_word_len = corpus.count_words()
            save_to_file(task, name, word_count)
            save_to_file("word_avg_length", name, avg_word_len)

        elif task == "type_count":
            type_count = corpus.count_type()
            save_to_file(task, name, type_count)

        elif task == "token_count":
            token_count = corpus.count_frequencies()
            save_to_file(task, name, token_count)

        elif task[0] == "s":
            sentence_word_avg = corpus.avg_sent_words()
            save_to_file(task, name, sentence_word_avg)

        elif task[0] == "l":
            lemma_count = corpus.count_lemma()
            save_to_file(task, name, lemma_count)


        elif task[0] == "p":
            pos_count = corpus.pos_tag_count()
            save_to_file(task, name, pos_count)

        elif task[0] == "c":
            my_choice = ["news", "corpus","editorial"]
            if name in my_choice:
                # fre,_ = corpus.count_words()
                fre = Counter(corpus.text)
                # print(fre)
                fre_list = []
                for i in fre.values():
                    fre_list.append(i)
                print(sorted(fre_list[0:10], reverse=True))
                plot_and_save(sorted(fre_list, reverse=True), f"{name}")

        # p = corpus.calculate_probabilities()






