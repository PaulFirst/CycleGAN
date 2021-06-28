FILE=$1


URL=https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1lgujk5rMPZRXAVNvnaXfp9cUbZrKYkeP

ZIP_FILE=${FILE}.zip
TARGET_DIR=${FILE}

FILE_ID=1lgujk5rMPZRXAVNvnaXfp9cUbZrKYkeP
FILE_NAME=${ZIP_FILE}
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate '${URL}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/1n/p')&id=$FILE_ID" -O $FILE_NAME && rm -rf /tmp/cookies.txt

unzip ${ZIP_FILE}
rm ${ZIP_FILE}
