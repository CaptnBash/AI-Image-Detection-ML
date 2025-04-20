#!/usr/bin/env bash

rename_files() {
        cd $1
        for file in $(ls)
        do
                new_name=$(md5sum $file | cut -d ' ' -f1).jpg
                echo "renaming file $file to $new_name"
                mv $file $new_name
        done
        cd -
}

curl -L -o ./archive.zip\
  https://www.kaggle.com/api/v1/datasets/download/tristanzhang32/ai-generated-images-vs-real-images

unzip ./archive.zip

mkdir -p REAL FAKE
rename_files train/fake
rename_files train/real
rename_files test/fake
rename_files test/real

mv train/fake/* test/fake/* FAKE
mv train/real/* test/real/* REAL

rm -r train test
rm archive.zip

