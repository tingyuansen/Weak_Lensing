CUDA_VISIBLE_DEVICES=1 python make_scattering.py train 100
CUDA_VISIBLE_DEVICES=0 python make_scattering.py test 100
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 100
