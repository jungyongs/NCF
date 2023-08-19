# Neural Collaborative Filtering

## Reference Paper
Neural Collaborative Filtering
WWW 2017  ·  Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua

## Dataset
MovieLens

## Example to Run : GMF
```
python GMF.py --epochs 20 --batch_size 256 --num_factors 32 --lr 0.001
```

## Example to Run : MLP
```
python MLP.py --epochs 20 --batch_size 256 --lr 0.001
```

## Example to Run : NeuMF
```
python NeuMF.py --epochs 20 --batch_size 256 --num_factors 32 --lr 0.001 --mf_pretrain 'ml-1m_GMF_32.h5' --mlp_pretrain 'ml-1m_MLP_[64,32,16,8].h5'
```
