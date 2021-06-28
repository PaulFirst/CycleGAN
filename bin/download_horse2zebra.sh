FILE='horse2zebra'

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/${FILE}.zip
ZIP_FILE=${FILE}.zip
TARGET_DIR=${FILE}
wget ${URL}
unzip ${ZIP_FILE}
rm ${ZIP_FILE}

