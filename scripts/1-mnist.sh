set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

python src/ganzo.py \
  --experiment mnist-gan \
  --dataset mnist \
  --data-dir images/mnist \
  --data-format single-image \
  --image-size 28 \
  --image-colors 1 \
  --loss gan \
  --generator fc \
  --generator-layers 4 \
  --generator-lr 0.0002 \
  --discriminator fc \
  --discriminator-layers 4 \
  --discriminator-dropout 0.3 \
  --discriminator-lr 0.0002 \
  --soft-labels \
  --noisy-labels \
  --epochs 10000 \
  --sample-every 100 \
  --sample-from-fixed-noise \
  --delete