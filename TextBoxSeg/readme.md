(1) 生成word 分割的synth gt
synthtext crop，然后生成word mask gt.
在demo/st800k_crop2.py(mask gt是按照char连接的).
在demo/st800k_crop.py(mask gt 是synthtext默认的gt).


(2) 训练分割网络
sh tools/dist_train.sh configs/textseg2.yaml
textseg.yaml 和textseg2.yaml 对应(1)中的crop和corp2

(3) 生成pseudo label
生成ic15 pseudo label: python3 tools/gen_ic15_pslabel.py --config-file configs/textseg2.yaml
可视化ic15 pseudo label: python3 tools/demo_ic15.py --config-file configs/textseg2.yaml
图片生成在demo/trash/IC15

生成totaltext pseudo label: python3 tools/gen_tt_pslabel.py --config-file configs/textseg2.yaml
可视化totaltext pseudo label: python3 tools/demo_tt.py --config-file configs/textseg2.yaml
图片生成在demo/trash/TT

# cv2.imread和Image.open效果不知道谁好。。。。。。

