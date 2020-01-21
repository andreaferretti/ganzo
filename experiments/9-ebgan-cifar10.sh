set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

python src/ganzo.py \
  --experiment ebgan-cifar10 \
  --dataset cifar10 \
  --data-dir images/cifar10 \
  --data-format single-image \
  --image-size 32 \
  --image-colors 3 \
  --loss ebgan \
  --ebgan-threshold 50 \
  --generator upsampling \
  --generator-layers 10 \
  --generator-upsamples 2 \
  --generator-lr 0.001 \
  --noise uniform \
  --discriminator autoencoder \
  --discriminator-layers 10 \
  --discriminator-upsamples 2 \
  --discriminator-lr 0.001 \
  --beta1 0.5 \
  --beta2 0.999 \
  --soft-labels \
  --noisy-labels \
  --epochs 100 \
  --sample-every 1 \
  --sample-from-fixed-noise