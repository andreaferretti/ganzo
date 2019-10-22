set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

python src/ganzo.py \
  --experiment dcgan-bedrooms \
  --dataset lsun \
  --data-dir $LSUN_HOME \
  --data-format single-image \
  --image-class bedroom \
  --image-size 64 \
  --image-colors 3 \
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
  --epochs 200 \
  --max-batches-per-epoch 1000 \
  --sample-every 1 \
  --sample-from-fixed-noise