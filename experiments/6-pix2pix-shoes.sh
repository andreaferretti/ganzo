set -e
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."

python src/ganzo.py \
  --experiment pix2pix-shoes \
  --game translate \
  --dataset folder \
  --data-dir $EDGES2SHOES \
  --data-format pair-of-images \
  --split horizontal \
  --loss pix2pix \
  --generator u-gen \
  --generator-layers 4 \
  --generator-channels 8 \
  --generator-lr 0.0002 \
  --discriminator patch-gan \
  --discriminator-layers 4 \
  --discriminator-channels 4 \
  --discriminator-lr 0.0002 \
  --soft-labels \
  --noisy-labels \
  --epochs 100 \
  --max-batches-per-epoch 100 \
  --sample-every 1 \
  --snapshot-translate