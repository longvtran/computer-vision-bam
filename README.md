# computer-vision-bam

## How to run

### Get the dataset
To obtain the BAM dataset, follow the instructions [here](https://bam-dataset.org/). Once finished, you should have be able to download a `.sqlite` file. Save this file in the /data folder, then run from the /data folder:
```
./get_data.sh
``` 
To process the data into the appropriate format, run:
```
python main.py --mode=gen_data --update=True --remove_broken=True
```

### Run training
To train your model, run (TO BE IMPLEMENTED):

```
python main.py --mode=train
```

