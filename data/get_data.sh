#!/bin/bash

Attributes=(
    #emotion_gloomy
    #emotion_happy
    #emotion_peaceful
    #emotion_scary
    media_3d_graphics
    media_comic
    media_graphite
    media_oilpaint
    media_pen_ink
    media_vectorart
    media_watercolor
    )

# Download all images with positive labels for each media attribute
# and store them in folders by that attribute.

for attribute in ${Attributes[@]}; do
    echo Downloading: $attribute
    sqlite3 bam.sqlite <<EOF | parallel -C'\|' 'mkdir -p {2}; wget --wait=1 {1} -O {2}/{3}.jpg'
        SELECT src, attribute, modules.mid
        FROM modules, crowd_labels WHERE modules.mid = crowd_labels.mid
        AND label="positive" AND attribute="$attribute"
        LIMIT 2000;
EOF
done
