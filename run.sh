model_list='ResNet18 ResNet34 ResNet50 ResNet101 ResNet152'
num_classes='200' #$2
opt_list='SGD Adam'
lr_list='0.1 0.01 0.001'
aug_list='default weak strong'
bs_list='16 64 256'

for model in $model_list
do
    for bs in $bs_list
    do
        for opt in $opt_list
        do
            for lr in $lr_list
            do
                for aug in $aug_list
                do
                    # use scheduler
                    echo "model: $model, bs: $bs, opt: $opt, lr: $lr, aug: $aug, use_sched: True"
                    EXP_NAME="model_$model-bs_$bs-opt_$opt-lr_$lr-aug_$aug-use_sched"

                    if [ -d "$EXP_NAME" ]
                    then
                        echo "$EXP_NAME is exist"
                    else
                        python main.py \
                            --exp-name $EXP_NAME \
                            --model-name $model \
                            --num-classes $num_classes \
                            --opt-name $opt \
                            --aug-name $aug \
                            --batch-size $bs \
                            --lr $lr \
                            --use_scheduler \
                            --epochs 50
                    fi

                    # not use scheduler
                    echo "model: $model, bs: $bs, opt: $opt, lr: $lr, aug: $aug, use_sched: False"
                    EXP_NAME="model_$model-bs_$bs-opt_$opt-lr_$lr-aug_$aug"

                    if [ -d "$EXP_NAME" ]
                    then
                        echo "$EXP_NAME is exist"
                    else
                        python main.py \
                            --exp-name $EXP_NAME \
                            --model-name $model \
                            --num-classes $num_classes \
                            --opt-name $opt \
                            --aug-name $aug \
                            --batch-size $bs \
                            --lr $lr \
                            --epochs 50
                    fi
                done
            done
        done
    done
done