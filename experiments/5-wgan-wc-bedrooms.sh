set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

python src/ganzo.py \
  --experiment wgan-wc-bedrooms \
  --dataset lsun \
  --data-dir $LSUN_HOME \
  --data-format single-image \
  --image-class bedroom \
  --image-size 64 \
  --image-colors 3 \
  --loss wgan-gp \
  --generator good \
  --generator-lr 1e-4 \
  --generator-iterations 1 \
  --discriminator good \
  --discriminator-lr 1e-4 \
  --discriminator-iterations 5 \
  --discriminator-hook weight-clipper \
  --clip-to 0.01 \
  --beta1 0 \
  --beta2 0.9 \
  --epochs 200 \
  --max-batches-per-epoch 1000 \
  --sample-every 1 \
  --sample-from-fixed-noise