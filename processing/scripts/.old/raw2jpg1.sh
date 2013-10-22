#!/bin/bash

if [ ! -d ../processed_images ]; then mkdir ../processed_images; fi;

# processes raw files
for f in ../*.NEF;
do
	echo "Processing $f"
	ufraw-batch \
		--wb=camera \
		--exposure=auto \
		--saturation=1.69 \
		--gamma=0.38 \
		--crop-left=1600 \
		--crop-right=3400 \
		--crop-top=600 \
		--crop-bottom=2600 \
		--out-type=png \
		--out-path=../processed_images \
		--out-depth=16 \
		$f
done
