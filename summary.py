from lexrank import lexranker
import sys  
import subprocess

reload(sys)  
sys.setdefaultencoding('utf8')

documents = ["modi.txt"]

lex = lexranker(documents)
scores = []

step = 1

print "START\n"

lex.calc_cosine_matrix()
lex.drop_edges()
lex.normalize_matrix()
lex.init_sent_vector()
scores = lex.calc_sent_scores()
for s in scores[:6]:
	print s

