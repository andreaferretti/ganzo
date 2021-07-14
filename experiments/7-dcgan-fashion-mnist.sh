set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

python src/ganzo/ganzo.py \
  --experiment dcgan-fashion-mnist \
  --dataset fashion-mnist \
  --data-dir images/fashion-mnist \
  --data-format single-image \
  --image-size 64 \
  --image-colors 1 \
  --loss gan \
  --generator conv \
  --generator-layers 5 \
  --generator-lr 0.0002 \
  --discriminator conv \
  --discriminator-layers 5 \
  --discriminator-lr 0.0002 \
  --beta1 0.5 \
  --beta2 0.999 \
  --soft-labels \
  --noisy-labels \
  --epochs 100 \
  --sample-every 1 \
  --sample-from-fixed-noise