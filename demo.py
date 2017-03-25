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

while True:
	nextstep = raw_input()

	if nextstep.isdigit():
		if step == 1:
			lex.calc_cosine_matrix()
			step+=1
			print "Built sentence graph. Drop insignificant edges\n"
			subprocess.call(["libreoffice","cosine_similarity_matrix.csv"])
		elif step == 2:
			lex.drop_edges()
			step+=1
			print "Edges dropped. Normalize the graph/matrix\n"
			subprocess.call(["libreoffice","matrix_before_normalization.csv"])
		elif step == 3:
			lex.normalize_matrix()
			step+=1
			print "Normalized. Initialize the Markov chain with the first state scores\n"
			subprocess.call(["libreoffice","stochastic_matrix.csv"])
		elif step == 4:
			lex.init_sent_vector()
			step+=1
			print "First state set up. Start the Markov process to calculate sentence scores\n"
			subprocess.call(["libreoffice","markov_process.csv"])
		elif step == 5:
			scores = lex.calc_sent_scores()
			step+=1
			print "Scores calculated. Print important sentences?\n"
			subprocess.call(["libreoffice","markov_process.csv"])
		elif step == 6:
			print "\n\n"
			for s in scores[:6]:
				print s
			print "\n\n"
			break
