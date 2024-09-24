

# CIFAR-10 (IF=50)
# GLMC
python main.py --dataset cifar10 -a resnet34 --num_classes 10 --imbanlance_rate 0.02 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4 --rho -1
python main.py --dataset cifar10 -a resnet34 --num_classes 10 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4 --rho -1
python main.py --dataset cifar100 -a resnet34 --num_classes 100 --imbanlance_rate 0.02 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4 --rho -1
python main.py --dataset cifar100 -a resnet34 --num_classes 100 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4 --rho -1

# GLMC+SAM (IF=50)
python main.py --dataset cifar10 -a resnet34 --num_classes 10 --imbanlance_rate 0.02 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1 --rho 0.05

# GLMC+SAM+FCC
python main.py --dataset cifar10 -a resnet34 --num_classes 10 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4 --rho 0.05 --fcc 0.1
python main.py --dataset cifar10 -a resnet34 --num_classes 10 --imbanlance_rate 0.01 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4 --rho 0.05 --fcc_same 0.1

# cifar-100 runs (IF=50)
#GLMC

python main.py --dataset cifar100 -a resnet34 --num_classes 100 --imbanlance_rate 0.02 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2  --contrast_weight 4 --rho -1

#GLMC+SAM
python main.py --dataset cifar100 -a resnet34 --num_classes 100 --imbanlance_rate 0.01 --beta 0.5 --lr 0.02 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2  --contrast_weight 4 --rho 0.05

#FSGL
python main.py --dataset cifar10 -a resnet34 --num_classes 10 --imbanlance_rate 0.02 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2  --contrast_weight 4 --rho 0.05 --fcc 0.5
python main.py --dataset cifar100 -a resnet34 --num_classes 100 --imbanlance_rate 0.02 --beta 0.5 --lr 0.01 --epochs 200 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2  --contrast_weight 4 --rho 0.05 --fcc 1
python main.py --dataset ImageNet-LT -a resnext50_32x4d --num_classes 1000 --beta 0.5 --lr 0.1 --epochs 200 -b 64 --momentum 0.9 --weight_decay 2e-4 --resample_weighting 0.0 --label_weighting 1.0  --contrast_weight 4 --rho 0.05 --fcc 0.1 --root '../../../../../datasets/ImageNet/' --dir_train_txt '../../../../../datasets/ImageNet/data_txt/ImageNet_LT_train.txt' --dir_test_txt '../../../../../datasets/ImageNet/data_txt/ImageNet_LT_test.txt'