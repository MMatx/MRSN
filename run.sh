run() {
    index=$1
    stage=$2
    command=("CUDA_VISIBLE_DEVICES=${index} python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/voc_simclr_only_novel/resume_base1_1shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2" \
    "CUDA_VISIBLE_DEVICES=${index} python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/voc_simclr_only_novel/resume_base1_5shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2" \
    "CUDA_VISIBLE_DEVICES=${index} python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/voc_simclr_only_novel/resume_base1_10shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2" \
    "CUDA_VISIBLE_DEVICES=${index} python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/voc_consis_simclr_add_novel/resume_base1_10shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2" \
    )
    # command=("echo 0" "echo 1" "echo 2" "echo 3")
    eval ${command[$stage]}
}


(( stage = 0 ))
for((;;));
do 

    declare -a num=("0" "0")
    while read lineA
    do
        (( num[$lineA]+=1 ))
    done < <(nvidia-smi | grep MiB | grep python | awk -F "|" '{ print $2 }' | awk -F " " '{ print $1 }' | sed 's/ //g')

    if [[ $stage -eq 4 ]] && [[ ${num[0]} -lt 2 ]] && [[ ${num[1]} -lt 2 ]];then
        echo "[Submit] 提交 python3 occupy.py 的任务" > run.log
        cd /apdcephfs/private_v_magtxma/v100
        CUDA_VISIBLE_DEVICES=1,0 python3 occupy.py
        break
    fi

    for (( i=0;i<=7;i++ ));
    do
        echo "[`date "+%F %k:%M"`] 当前 ${i} 卡运行任务数为 ${num[$i]} , stage = ${stage}" > run.log
        if [[ $stage -lt 4 ]] && [[ ${num[${i}]} -lt 1 ]];then
            echo "[Submit] 在 ${i} 卡, 提交 stage = ${stage} 的任务" > run.log
            run $i $stage &
            (( stage=stage+1 ))
        fi
    done

    sleep 360s
done

CUDA_VISIBLE_DEVICES=0 python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/test.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2

CUDA_VISIBLE_DEVICES=0 python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/cut_mix_base1_1shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=1 python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/cut_mix_base1_2shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=2 python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/cut_mix_base1_3shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=3 python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/cut_mix_base1_5shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=6 python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/cut_mix_base1_10shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2


CUDA_VISIBLE_DEVICES=0 python3  train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_1shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=1 python3  train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_2shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=2 python3  train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_3shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=3 python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_5shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=6 python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_10shot.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2


#CUDA_VISIBLE_DEVICES=4 python3  train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_1shot2.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=1 python3  train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_2shot2.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=7 python3  train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_3shot2.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=3 python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_5shot2.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=6 python3 train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_10shot2.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2

/hdd/master2/code/utt/configs/voc/rebuttal/1_3shot_099.yaml

#CUDA_VISIBLE_DEVICES=7 python3  train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_3shot_099.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2

#CUDA_VISIBLE_DEVICES=2 python train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_3shot_lam_1.yaml  SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2

CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_3shot_lam01.yaml  SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=3 python train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_3shot_lam_2.yaml  SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
CUDA_VISIBLE_DEVICES=2 python train_net.py --num-gpus 1 --config-file /hdd/master2/code/utt/configs/voc/rebuttal/1_3shot_lam_05.yaml  SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2
