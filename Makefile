TRAIN = flow-100.in random-100.dot bw-orig.out qcoarse-50.dot bwcoarse-50.dot
EVAL = flow-100.in rm-50.dot bw-qcoarse.out bw-bwcoarse.out bw-rm.out

ALPHA = 0.7
# for i in (seq 1 100); make run; end
# →100回実行

run-rm:
	make clean
	make flow-100.in
	make random-100.dot
	make bw-orig.out
	make rm-50.dot
	make bw-rm.out
	./merge-bw.py bw-orig.out bw-rm.out >orig-rm.out
	./cal_mse.py orig-rm.out >> updatelog/mse-$(ALPHA)-rm.log

run:
	make clean
	make all
	make merge
	./cal_mse.py orig-bwcoarse.out
	./cal_mse.py orig-qcoarse.out
	make mse
#
all:
	make train
	make eval

train:	$(TRAIN)

eval:	$(EVAL)

flow-100.in:
	./gen-flows.py 100 0.01 >$@

rm-50.dot:
	graphfilt -f rm random-100.dot >$@

bw-rm.out:
	./q-coarse.py -l rm-50.dot flow-100.in >$@

random-100.dot:
	graphgen -t barandom 100 200 3 >$@

qcoarse-50.dot:
	./q-coarse.py -a $(ALPHA) random-100.dot flow-100.in >$@

bw-orig.out:
	./q-coarse.py -l random-100.dot flow-100.in >$@

bwcoarse-50.dot:
	./bw_coarse.py -i random-100.dot -f bw-orig.out -o bwcoarse-50.dot -t qcoarse-50.dot

bw-qcoarse.out:
	./q-coarse.py -l qcoarse-50.dot flow-100.in >$@


bw-bwcoarse.out:
	./q-coarse.py -l bwcoarse-50.dot flow-100.in >$@

mse:
	./cal_mse.py orig-bwcoarse.out >> updatelog/mse-$(ALPHA)-bwcoarse.log
	./cal_mse.py orig-qcoarse.out >> updatelog/mse-$(ALPHA)-qcoarse.log
	./cal_mse.py orig-rm.out >> updatelog/mse-$(ALPHA)-rm.log

clean:
	rm -f $(TRAIN)
	rm -f $(EVAL)
	rm -f orig-qcoarse.out
	rm -f orig-bwcoarse.out

merge:
	./merge-bw.py bw-orig.out bw-qcoarse.out >orig-qcoarse.out
	./merge-bw.py bw-orig.out bw-bwcoarse.out >orig-bwcoarse.out
	./merge-bw.py bw-orig.out bw-rm.out >orig-rm.out
