#!/bin/bash
#SBATCH --job-name=siren_fetch_hic
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Downloads the K562 intact Hi-C matrix used by ENCODE-rE2G (ENCFF621AIY, GRCh38, 33,783,625,697 bytes)
# and checks the md5 published by the ENCODE portal. Resumable. Usage: sbatch hpc/fetch_hic.sbatch
set -eu
export SIREN_ROOT="${SIREN_ROOT:-$HOME/siren_repro}"
cd "$SIREN_ROOT"
DEST="data/raw/HIC_K562"
URL="https://encode-public.s3.amazonaws.com/2022/05/15/0571c671-3645-4f92-beae-51dfd3f42c36/ENCFF621AIY.hic"
MD5="71e5dc072beddbdcda11c9016696e142"
mkdir -p "$DEST" logs
curl -sS -L --retry 5 --retry-delay 30 -C - -o "$DEST/ENCFF621AIY.hic" "$URL"
echo "$MD5  $DEST/ENCFF621AIY.hic" | md5sum -c -
echo "$MD5  ENCFF621AIY.hic  $URL  $(date -u +%FT%TZ)" > "$DEST/manifest.md5"
echo FETCH_HIC_OK
