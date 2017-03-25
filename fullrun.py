from lexrank import lexranker
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

documents = ["simi.txt"]

lex = lexranker(documents)
scores = lex.lex_rank()

for s in scores[:6]:
	print s
