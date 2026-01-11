
index=/home/peterjin/mnt/index/wiki-18/e5_Flat.index

split -b 40G $index part_

python upload.py
