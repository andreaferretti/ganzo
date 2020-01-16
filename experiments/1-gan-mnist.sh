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
  --noise uniform \
  --generator constant-fc \
  --generator-layers 3 \
  --generator-layer-size 1200 \
  --generator-lr 0.1 \
  --discriminator fc-maxout \
  --discriminator-layers 3 \
  --discriminator-layer-size 240 \
  --discriminator-maxout-size 5 \
  --discriminator-iterations 1 \
  --discriminator-dropout 0.3 \
  --discriminator-lr 0.1 \
  --batch-size 100 \
  --soft-labels \
  --noisy-labels \
  --epochs 100 \
  --sample-every 1 \
  --sample-from-fixed-noise