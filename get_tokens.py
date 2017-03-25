from nltk.data import load
from nltk.tokenize.texttiling import TextTilingTokenizer
from nltk.tokenize.punkt import PunktWordTokenizer

class sentence_fetcher:
	def load_trained_data(self,datafile):
		self.sentence_detector = load(datafile)

	def get_sentences_from_text(self,text):
		return self.sentence_detector.tokenize(text)

def get_sentences_from_text(text):
	sfetcher = sentence_fetcher()
	sfetcher.load_trained_data('punkt/english.pickle')
	sentences = sfetcher.get_sentences_from_text(text)
	return sentences

def get_words_from_sentence(text):
	return PunktWordTokenizer().tokenize(text)

def get_paragraphs_from_text(text):
	tiling_tokenizer = TextTilingTokenizer()
	paragraphs = tiling_tokenizer.tokenize(text)
	return paragraphs


