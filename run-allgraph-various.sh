#!/usr/bin/sh

#N=1000
#学習モデルの作成
#for coarse in 'COARSENET' 'MGC'
# for coarse in 'LVN' 'LVE' 'kron' 'HEM' 'COARSENET' 'MGC'
# # # for coarse in 'RM'
# do
#     #学習-----------------------------
#     #学習データ作成
#     make clean
#     [ -d data-original ] || mkdir -p data-original
#     [ -d data-coarse ] || mkdir -p data-coarse
#     ./generate_random.py -N 1000 -m $coarse
#     ./run.py -t -o various-1000-10-$coarse-model.zip
#     #./run_coarse.py -t -o various-1000-10-$coarse-coarse-model.zip
#     #学習データ保存
#     cp -r data-coarse/ data/$coarse-0.5-coarse
#     cp -r data-original/ data/$coarse-0.5-original
# done

#評価------------------------------
#for graph in 'er' 'random' 'ba' 'barandom' 'db' 'tree' 'li_maini' '4-regular' 'voronoi'
for graph in 'voronoi'
do
    make clean
    [ -d data-original ] || mkdir -p data-original
    ./generate.py -g $graph -N 500
    #for coarse in 'COARSENET' 'MGC' 'LVN' 'LVE' 'kron' 'HEM'
    for coarse in 'COARSENET'
    do    
    #評価データ作成
    rm -rf data-coarse/*.dot
    [ -d data-coarse ] || mkdir -p data-coarse
    [ -d plot/$coarse] || mkdir -p plot/$coarse
    #
    ./generate_coarse_graph.py -m $coarse -N 500 -g $graph

    #機械学習の評価実行(復元方向)
    ./run.py -i various-1000-10-$coarse-model.zip -e > degree-distributions/$coarse/various-1000-10-$graph-degree.res
    #./run.py -i various-1000-10-$coarse-model.zip -e > tmp.res
    #./r2score.py -i various-1000-10-$coarse-model.zip -e > plot/$coarse/various-1000-10-$graph-r2score.res
    #粗視化グラフの次数分布をプロット

    #./degree_distrib.py data-coarse/* > coarse.res
    #[ -d degree-distributions/$coarse] || mkdir -p degree-distributions/$coarse
    #./merge_files.py degree-distributions/$coarse/various-1000-10-$graph-degree.res tmp.res coarse.res
    # #機械学習の評価実行(粗視化方向)
    # ./run_coarse.py -i various-1000-10-$coarse-coarse-model.zip -e > tmp.res
    # ./r2score_coarse.py -i various-1000-10-$coarse-coarse-model.zip -e > plot/$coarse/various-1000-10-coarse-$graph-r2score.res
    # #粗視化グラフの次数分布をプロット
    # ./degree_distrib_coarse.py data-original/* > original.res
    # ./merge_files.py plot/$coarse/various-1000-10-coarse-$graph-degree.res tmp.res original.res
    done
done
