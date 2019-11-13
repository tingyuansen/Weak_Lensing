#CUDA_VISIBLE_DEVICES=1 python make_scattering.py train 10
#CUDA_VISIBLE_DEVICES=1 python make_scattering.py test 10

#CUDA_VISIBLE_DEVICES=1 python make_scattering.py train 30
#CUDA_VISIBLE_DEVICES=1 python make_scattering.py test 30

#CUDA_VISIBLE_DEVICES=1 python make_scattering.py train 50
#CUDA_VISIBLE_DEVICES=1 python make_scattering.py test 50

#CUDA_VISIBLE_DEVICES=1 python make_scattering.py train 70
#CUDA_VISIBLE_DEVICES=1 python make_scattering.py test 70

#CUDA_VISIBLE_DEVICES=1 python make_scattering.py train 90
#CUDA_VISIBLE_DEVICES=1 python make_scattering.py test 90

CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 10 30
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 30 30
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 50 30
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 70 30
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 90 30

CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 10 100
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 30 100
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 50 100
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 70 100
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 90 100

CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 10 1000
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 30 1000
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 50 1000
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 70 1000
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 90 1000

CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 10 28000
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 30 28000
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 50 28000
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 70 28000
CUDA_VISIBLE_DEVICES=1 python wavelet_scattering.py 90 28000
