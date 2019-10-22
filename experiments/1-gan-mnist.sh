set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

python src/ganzo.py \
  --experiment gan-mnist \
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
  --epochs 100 \
  --sample-every 1 \
  --sample-from-fixed-noise