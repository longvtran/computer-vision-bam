#!/bin/bash

# Purpose: download all images with positive labels for each combination of
# media/emotion and store them in folders by their attributes.

# number of each combination of lables to store (results in total of 28 * num_images)
num_images=5000

# wait time in seconds between each image downloaded (set to 1 if getting errors)
wait_time=1

Media=(
    media_3d_graphics
    media_comic
    media_graphite
    media_oilpaint
    media_pen_ink
    media_vectorart
    media_watercolor
    )

Emotion=(
    emotion_gloomy
    emotion_happy
    emotion_peaceful
    emotion_scary
    )

for media in ${Media[@]}; do
    mkdir -p $media
    for emotion in ${Emotion[@]}; do
        mkdir -p $media/$emotion
        echo ==== Downloading: $media / $emotion ====
        sqlite3 bam.sqlite <<EOF | parallel -C'\|' "wget --wait=$wait_time \
                {1} -O '$media/$emotion/{4}.jpg'"
            SELECT src, $media, $emotion, modules.mid
            FROM modules, automatic_labels WHERE modules.mid = automatic_labels.mid
            AND $media="positive" AND $emotion="positive"
            LIMIT $num_images;
EOF
    done
done
