MODEL_FILE = model.zip
FILES = Makefile generate.py run.py $(MODEL_FILE)

eval:	$(MODEL_FILE)
	./run.py -e -i $(MODEL_FILE)

train:
	./run.py -t -o $(MODEL_FILE) -n 25

ssr:	$(MODEL_FILE)
	graphgen -t li_maini 100 >orig.dot
	graphfilt -f min <orig.dot >coarse.dot
	./run.py -i $(MODEL_FILE) -r coarse.dot -n 100
	graphview orig.dot
	graphview out.dot


generate-data:
	[ -d data-original ] || mkdir -p data-original
	[ -d data-coarse ] || mkdir -p data-coarse
	./generate.py -n 20

clean:
	rm -rf data-* *.dot

dist:
	tar czvf ../gnn-`date +%y%m%d`.tar.gz $(FILES)
